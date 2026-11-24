"""Candidate event proposal from proprioceptive gripper and TCP signals."""

from __future__ import annotations

import csv
import io
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from lerobot_check.dataset import ArmState, DatasetView

GripperEventKind = Literal["close", "open"]

PROPOSAL_COLUMNS: tuple[str, ...] = (
    "episode_index",
    "arm",
    "event_type",
    "frame",
    "end_frame",
    "time_s",
    "value",
)


@dataclass(frozen=True)
class ProposeConfig:
    """Thresholds for gripper transition and low-speed segment detection.

    Attributes:
        close_threshold: Gripper position at or below which an open gripper
            transitions to closed (``1.0`` = open, toward ``0.0`` = closed).
        open_threshold: Gripper position at or above which a closed gripper
            transitions back to open. Must be strictly greater than
            ``close_threshold`` to provide hysteresis against signal jitter.
        low_speed_threshold_mps: TCP linear speed (m/s) below which a frame is
            considered slow.
        min_low_speed_s: Minimum contiguous duration (seconds) required to emit
            a low-speed segment.
    """

    close_threshold: float = 0.5
    open_threshold: float = 0.7
    low_speed_threshold_mps: float = 0.02
    min_low_speed_s: float = 0.3

    def __post_init__(self) -> None:
        if self.close_threshold >= self.open_threshold:
            raise ValueError(
                f"close_threshold ({self.close_threshold}) must be < "
                f"open_threshold ({self.open_threshold}) for hysteresis."
            )
        if self.low_speed_threshold_mps <= 0.0:
            raise ValueError(
                f"low_speed_threshold_mps must be positive, got {self.low_speed_threshold_mps}"
            )
        if self.min_low_speed_s <= 0.0:
            raise ValueError(f"min_low_speed_s must be positive, got {self.min_low_speed_s}")


@dataclass(frozen=True)
class GripperEvent:
    """Detected gripper close or open transition.

    Attributes:
        episode_index: Zero-based episode index.
        arm: Arm identifier (e.g. ``"left"`` or ``"right"``).
        kind: ``"close"`` or ``"open"``.
        frame: Frame index where the threshold crossing occurred.
        time_s: Timestamp of ``frame`` in seconds.
        gripper_pos: Observed gripper position at ``frame``.
    """

    episode_index: int
    arm: str
    kind: GripperEventKind
    frame: int
    time_s: float
    gripper_pos: float


@dataclass(frozen=True)
class LowSpeedSegment:
    """Contiguous low-speed TCP motion interval ``[start, end)``.

    Attributes:
        episode_index: Zero-based episode index.
        arm: Arm identifier.
        start: Inclusive start frame of the low-speed segment.
        end: Exclusive end frame of the low-speed segment.
        start_time_s: Start timestamp in seconds.
        end_time_s: End timestamp in seconds.
        min_speed_mps: Minimum TCP linear speed inside the segment (m/s).
        median_speed_mps: Median TCP linear speed inside the segment (m/s).
    """

    episode_index: int
    arm: str
    start: int
    end: int
    start_time_s: float
    end_time_s: float
    min_speed_mps: float
    median_speed_mps: float


@dataclass(frozen=True)
class EpisodeProposal:
    """All candidate events extracted for a single episode.

    Attributes:
        episode_index: Zero-based episode index.
        length: Total number of frames in the episode.
        fps: Dataset frame rate in Hz.
        gripper_events: Sorted per-arm gripper close/open events.
        low_speed_segments: Sorted per-arm low-speed segments.
    """

    episode_index: int
    length: int
    fps: float
    gripper_events: tuple[GripperEvent, ...]
    low_speed_segments: tuple[LowSpeedSegment, ...]

    def first_close_frame(
        self,
        start: int,
        end: int,
        *,
        arm: str | None = None,
    ) -> int | None:
        """Return the earliest gripper-close frame inside ``[start, end)``.

        Args:
            start: Inclusive start frame of the search window.
            end: Exclusive end frame of the search window.
            arm: Optional arm filter; when ``None``, considers all arms.

        Returns:
            Earliest matching close frame index, or ``None`` if none occurred.
        """
        candidates = [
            ev.frame
            for ev in self.gripper_events
            if ev.kind == "close" and start <= ev.frame < end and (arm is None or ev.arm == arm)
        ]
        return min(candidates) if candidates else None


def detect_gripper_events(
    gripper: np.ndarray,
    *,
    episode_index: int,
    arm: str,
    fps: float,
    close_threshold: float = 0.5,
    open_threshold: float = 0.7,
) -> list[GripperEvent]:
    """Detect gripper close/open transitions using Schmitt-trigger hysteresis.

    On Xense/Elite rigs, ``1.0`` represents an open gripper and smaller values
    toward ``0.0`` represent a closed gripper. Using separate ``close_threshold``
    and ``open_threshold`` prevents signal jitter around a single threshold from
    emitting spurious alternating events.

    Args:
        gripper: 1D array of gripper positions across an episode.
        episode_index: Episode index recorded in emitted events.
        arm: Arm label recorded in emitted events.
        fps: Frame rate in Hz.
        close_threshold: Upper bound that triggers a ``"close"`` event.
        open_threshold: Lower bound that triggers an ``"open"`` event.

    Returns:
        Chronologically ordered list of ``GripperEvent`` records.

    Raises:
        ValueError: If ``close_threshold >= open_threshold`` or ``fps <= 0``.
    """
    if close_threshold >= open_threshold:
        raise ValueError(
            f"close_threshold ({close_threshold}) must be < open_threshold ({open_threshold})"
        )
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    values = np.asarray(gripper, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected 1D gripper array, got shape {values.shape}")
    if len(values) < 2:
        return []

    midpoint = 0.5 * (close_threshold + open_threshold)
    is_closed = bool(values[0] <= close_threshold or values[0] < midpoint)

    events: list[GripperEvent] = []
    for frame in range(1, len(values)):
        pos = float(values[frame])
        if not is_closed and pos <= close_threshold:
            is_closed = True
            events.append(
                GripperEvent(
                    episode_index=episode_index,
                    arm=arm,
                    kind="close",
                    frame=frame,
                    time_s=frame / fps,
                    gripper_pos=pos,
                )
            )
        elif is_closed and pos >= open_threshold:
            is_closed = False
            events.append(
                GripperEvent(
                    episode_index=episode_index,
                    arm=arm,
                    kind="open",
                    frame=frame,
                    time_s=frame / fps,
                    gripper_pos=pos,
                )
            )
    return events


def compute_tcp_linear_speed(tcp_xyz: np.ndarray, fps: float) -> np.ndarray:
    """Compute per-frame TCP linear speed (m/s) from finite XYZ differences.

    ``speed[0]`` is set to ``0.0`` and ``speed[t] = ||xyz[t] - xyz[t-1]|| * fps``
    for ``t >= 1``.

    Args:
        tcp_xyz: TCP positions of shape ``(N, 3)`` in meters.
        fps: Frame rate in Hz.

    Returns:
        1D float64 array of length ``N`` containing speed in meters per second.

    Raises:
        ValueError: If ``tcp_xyz`` is not shape ``(N, 3)`` or ``fps <= 0``.
    """
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    xyz = np.asarray(tcp_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected tcp_xyz of shape (N, 3), got {xyz.shape}")

    speeds = np.zeros(len(xyz), dtype=np.float64)
    if len(xyz) >= 2:
        deltas = np.diff(xyz, axis=0)
        speeds[1:] = np.linalg.norm(deltas, axis=1) * fps
    return speeds


def detect_low_speed_segments(
    speed_mps: np.ndarray,
    *,
    episode_index: int,
    arm: str,
    fps: float,
    threshold_mps: float = 0.02,
    min_duration_s: float = 0.3,
) -> list[LowSpeedSegment]:
    """Identify contiguous low-speed intervals ``[start, end)`` from TCP speed.

    Args:
        speed_mps: 1D array of TCP speeds in m/s.
        episode_index: Episode index recorded in emitted segments.
        arm: Arm identifier recorded in emitted segments.
        fps: Frame rate in Hz.
        threshold_mps: Maximum speed (m/s) classified as low speed.
        min_duration_s: Minimum segment length in seconds.

    Returns:
        Chronologically ordered list of ``LowSpeedSegment`` records.
    """
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if threshold_mps <= 0.0:
        raise ValueError(f"threshold_mps must be positive, got {threshold_mps}")
    if min_duration_s <= 0.0:
        raise ValueError(f"min_duration_s must be positive, got {min_duration_s}")

    speeds = np.asarray(speed_mps, dtype=np.float64)
    if speeds.ndim != 1:
        raise ValueError(f"Expected 1D speed array, got shape {speeds.shape}")
    if len(speeds) == 0:
        return []

    min_frames = max(1, round(min_duration_s * fps))
    is_slow = speeds <= threshold_mps
    # Pad with False on both ends so every contiguous True run has an exact start/end pair.
    padded = np.concatenate(([False], is_slow, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    segments: list[LowSpeedSegment] = []
    for start_idx, end_idx in zip(starts, ends, strict=True):
        start = int(start_idx)
        end = int(end_idx)
        if end - start < min_frames:
            continue
        seg_speeds = speeds[start:end]
        segments.append(
            LowSpeedSegment(
                episode_index=episode_index,
                arm=arm,
                start=start,
                end=end,
                start_time_s=start / fps,
                end_time_s=end / fps,
                min_speed_mps=float(np.min(seg_speeds)),
                median_speed_mps=float(np.median(seg_speeds)),
            )
        )
    return segments


def propose_episode_from_arms(
    episode_index: int,
    length: int,
    fps: float,
    arms: Sequence[ArmState],
    config: ProposeConfig | None = None,
) -> EpisodeProposal:
    """Compute candidate gripper events and low-speed segments from arm trajectories.

    Args:
        episode_index: Zero-based episode index.
        length: Total frame count of the episode.
        fps: Frame rate in Hz.
        arms: Per-arm state trajectories.
        config: Optional detection thresholds; uses ``ProposeConfig()`` by default.

    Returns:
        ``EpisodeProposal`` containing sorted gripper events and low-speed segments.
    """
    cfg = config or ProposeConfig()
    gripper_events: list[GripperEvent] = []
    low_speed_segments: list[LowSpeedSegment] = []

    for arm_state in arms:
        gripper_events.extend(
            detect_gripper_events(
                arm_state.gripper,
                episode_index=episode_index,
                arm=arm_state.arm,
                fps=fps,
                close_threshold=cfg.close_threshold,
                open_threshold=cfg.open_threshold,
            )
        )
        speeds = compute_tcp_linear_speed(arm_state.tcp_xyz, fps)
        low_speed_segments.extend(
            detect_low_speed_segments(
                speeds,
                episode_index=episode_index,
                arm=arm_state.arm,
                fps=fps,
                threshold_mps=cfg.low_speed_threshold_mps,
                min_duration_s=cfg.min_low_speed_s,
            )
        )

    gripper_events.sort(key=lambda ev: (ev.frame, ev.arm, ev.kind))
    low_speed_segments.sort(key=lambda seg: (seg.start, seg.arm, seg.end))

    return EpisodeProposal(
        episode_index=episode_index,
        length=length,
        fps=fps,
        gripper_events=tuple(gripper_events),
        low_speed_segments=tuple(low_speed_segments),
    )


def propose_episode(
    dataset: DatasetView,
    episode_index: int,
    config: ProposeConfig | None = None,
) -> EpisodeProposal:
    """Load an episode from ``dataset`` and propose candidate annotation events.

    Args:
        dataset: Open ``DatasetView``.
        episode_index: Zero-based episode index.
        config: Optional detection thresholds.

    Returns:
        ``EpisodeProposal`` for the requested episode.
    """
    record = dataset.get_episode(episode_index)
    arms = dataset.load_episode_arm_states(episode_index)
    return propose_episode_from_arms(
        episode_index=episode_index,
        length=record.length,
        fps=dataset.fps,
        arms=arms,
        config=config,
    )


def proposal_to_rows(proposal: EpisodeProposal) -> list[dict[str, str]]:
    """Serialize one ``EpisodeProposal`` into flat CSV row dictionaries."""
    rows: list[tuple[int, str, str, dict[str, str]]] = []
    for ev in proposal.gripper_events:
        event_type = f"gripper_{ev.kind}"
        rows.append(
            (
                ev.frame,
                ev.arm,
                event_type,
                {
                    "episode_index": str(ev.episode_index),
                    "arm": ev.arm,
                    "event_type": event_type,
                    "frame": str(ev.frame),
                    "end_frame": "",
                    "time_s": f"{ev.time_s:.3f}",
                    "value": f"{ev.gripper_pos:.4f}",
                },
            )
        )
    for seg in proposal.low_speed_segments:
        rows.append(
            (
                seg.start,
                seg.arm,
                "low_speed",
                {
                    "episode_index": str(seg.episode_index),
                    "arm": seg.arm,
                    "event_type": "low_speed",
                    "frame": str(seg.start),
                    "end_frame": str(seg.end),
                    "time_s": f"{seg.start_time_s:.3f}",
                    "value": f"{seg.median_speed_mps:.4f}",
                },
            )
        )
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in rows]


def format_proposals_csv(proposals: Sequence[EpisodeProposal]) -> str:
    """Render proposals for one or more episodes as a CSV string."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=PROPOSAL_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for proposal in proposals:
        for row in proposal_to_rows(proposal):
            writer.writerow(row)
    return buf.getvalue()


def write_proposals_csv(proposals: Sequence[EpisodeProposal], path: Path) -> None:
    """Write proposals for one or more episodes to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_proposals_csv(proposals), encoding="utf-8")


def load_proposal_event_frames(
    path: Path,
    episode_index: int,
    *,
    event_types: Collection[str] = ("gripper_close", "gripper_open"),
) -> list[int]:
    """Extract sorted unique event frames for ``episode_index`` from a proposals CSV.

    Args:
        path: Path to a CSV written by ``lerobot-check propose``.
        episode_index: Target episode index.
        event_types: Set of ``event_type`` values to include.

    Returns:
        Sorted list of unique frame indices matching ``episode_index`` and
        ``event_types``.
    """
    frames: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["episode_index"]) != episode_index:
                continue
            if row["event_type"] in event_types:
                frames.add(int(row["frame"]))
    return sorted(frames)
