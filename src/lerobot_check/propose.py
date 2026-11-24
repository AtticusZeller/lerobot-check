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

DEFAULT_RENDER_EVENT_TYPES: tuple[str, ...] = (
    "gripper_close",
    "gripper_open",
    "squeeze",
)


@dataclass(frozen=True)
class ProposeConfig:
    """Parameters for relative gripper motion, squeeze, and low-speed detection.

    Attributes:
        gripper_delta: Minimum relative amplitude change in observed gripper
            position from the running local extremum to confirm a close or open
            motion (default ``0.06``, calibrated above +-0.01-0.02 jitter and below
            tool-held egg pick strokes of ~0.08-0.15).
        squeeze_gap: Minimum positive magnitude ``gap`` such that
            ``action_gripper - state_gripper <= -gap`` indicates the gripper is
            actively commanding further closure against a grasped object or tool
            (default ``0.05``).
        min_squeeze_s: Minimum contiguous duration (seconds) required to emit a
            ``squeeze`` interval (default ``0.2``).
        low_speed_threshold_mps: TCP linear speed (m/s) below which a frame is
            considered slow (default ``0.02``).
        min_low_speed_s: Minimum contiguous duration (seconds) required to emit
            a low-speed segment (default ``0.3``).
    """

    gripper_delta: float = 0.06
    squeeze_gap: float = 0.05
    min_squeeze_s: float = 0.2
    low_speed_threshold_mps: float = 0.02
    min_low_speed_s: float = 0.3

    def __post_init__(self) -> None:
        if self.gripper_delta <= 0.0:
            raise ValueError(f"gripper_delta must be positive, got {self.gripper_delta}")
        if self.squeeze_gap <= 0.0:
            raise ValueError(f"squeeze_gap must be positive, got {self.squeeze_gap}")
        if self.min_squeeze_s <= 0.0:
            raise ValueError(f"min_squeeze_s must be positive, got {self.min_squeeze_s}")
        if self.low_speed_threshold_mps <= 0.0:
            raise ValueError(
                f"low_speed_threshold_mps must be positive, got {self.low_speed_threshold_mps}"
            )
        if self.min_low_speed_s <= 0.0:
            raise ValueError(f"min_low_speed_s must be positive, got {self.min_low_speed_s}")


@dataclass(frozen=True)
class GripperEvent:
    """Detected relative-amplitude gripper close or open motion.

    Attributes:
        episode_index: Zero-based episode index.
        arm: Arm identifier (e.g. ``"left"`` or ``"right"``).
        kind: ``"close"`` (decreasing gripper position) or ``"open"`` (increasing).
        frame: Frame index where the motion begins leaving its prior plateau.
        end_frame: Frame index where the motion settles near its new extremum.
        time_s: Timestamp of ``frame`` in seconds.
        end_time_s: Timestamp of ``end_frame`` in seconds.
        start_pos: Gripper state at the pre-motion extremum.
        end_pos: Gripper state at the settled extremum.
    """

    episode_index: int
    arm: str
    kind: GripperEventKind
    frame: int
    end_frame: int
    time_s: float
    end_time_s: float
    start_pos: float
    end_pos: float

    @property
    def amplitude(self) -> float:
        """Signed position change ``end_pos - start_pos``."""
        return self.end_pos - self.start_pos


@dataclass(frozen=True)
class SqueezeSegment:
    """Contiguous active-squeeze interval ``[start, end)``.

    Emitted when ``action_gripper - state_gripper <= -gap`` persists for at
    least ``min_duration_s`` (indicating the gripper commands closure while an
    object or tool physically resists full closure).

    Attributes:
        episode_index: Zero-based episode index.
        arm: Arm identifier.
        start: Inclusive start frame of the squeeze interval.
        end: Exclusive end frame of the squeeze interval.
        start_time_s: Start timestamp in seconds.
        end_time_s: End timestamp in seconds.
        min_gap: Most negative ``action - state`` value inside ``[start, end)``.
        median_gap: Median ``action - state`` value inside ``[start, end)``.
    """

    episode_index: int
    arm: str
    start: int
    end: int
    start_time_s: float
    end_time_s: float
    min_gap: float
    median_gap: float


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
        gripper_events: Sorted per-arm relative close/open motions.
        squeeze_segments: Sorted per-arm active-squeeze intervals.
        low_speed_segments: Sorted per-arm low-speed TCP segments.
    """

    episode_index: int
    length: int
    fps: float
    gripper_events: tuple[GripperEvent, ...]
    squeeze_segments: tuple[SqueezeSegment, ...] = ()
    low_speed_segments: tuple[LowSpeedSegment, ...] = ()

    def first_close_frame(
        self,
        start: int,
        end: int,
        *,
        arm: str | None = None,
    ) -> int | None:
        """Return the earliest gripper-close motion start frame inside ``[start, end)``.

        Args:
            start: Inclusive start frame of the search window.
            end: Exclusive end frame of the search window.
            arm: Optional arm filter; when ``None``, considers all arms.

        Returns:
            Earliest matching close-motion start frame, or ``None`` if none occurred.
        """
        candidates = [
            ev.frame
            for ev in self.gripper_events
            if ev.kind == "close" and start <= ev.frame < end and (arm is None or ev.arm == arm)
        ]
        return min(candidates) if candidates else None


def _refine_motion_bounds(
    values: np.ndarray,
    *,
    kind: GripperEventKind,
    ext_start: int,
    ext_end: int,
    trigger_idx: int,
    delta: float,
) -> tuple[int, int]:
    """Tighten a confirmed hysteresis swing to its actual motion start and settle frames.

    Because plateau jitter can place the raw running extremum many frames before
    motion begins (or after it settles), walk backward from ``trigger_idx`` to
    the last frame still within a small tolerance of the start extremum and
    forward from that start to the first frame reaching the end extremum band.
    """
    val_start = float(values[ext_start])
    val_end = float(values[ext_end])
    amp = abs(val_start - val_end)
    eps = min(0.15 * amp, 0.25 * delta)

    start_frame = ext_start
    end_frame = ext_end
    if kind == "close":
        for k in range(trigger_idx, ext_start - 1, -1):
            if float(values[k]) >= val_start - eps:
                start_frame = min(k + 1, trigger_idx)
                break
        for k in range(start_frame, ext_end + 1):
            if float(values[k]) <= val_end + eps:
                end_frame = k
                break
    else:
        for k in range(trigger_idx, ext_start - 1, -1):
            if float(values[k]) <= val_start + eps:
                start_frame = min(k + 1, trigger_idx)
                break
        for k in range(start_frame, ext_end + 1):
            if float(values[k]) >= val_end - eps:
                end_frame = k
                break

    return start_frame, end_frame


def detect_gripper_events(
    gripper: np.ndarray,
    *,
    episode_index: int,
    arm: str,
    fps: float,
    delta: float = 0.06,
) -> list[GripperEvent]:
    """Detect gripper close/open motions using relative-amplitude hysteresis.

    Tracks the running local extremum of the observed gripper position without
    any absolute thresholds: a ``"close"`` motion fires once position drops by
    at least ``delta`` from the running local maximum, and an ``"open"`` motion
    fires once position rises by at least ``delta`` from the running local
    minimum. Small oscillations below ``delta`` are ignored.

    Each emitted ``GripperEvent`` records the frame where the motion leaves its
    pre-motion plateau in ``frame`` and where it settles near the new extremum
    in ``end_frame``.

    Args:
        gripper: 1D array of observed gripper positions across an episode.
        episode_index: Episode index recorded in emitted events.
        arm: Arm label recorded in emitted events.
        fps: Frame rate in Hz.
        delta: Minimum relative amplitude change required to confirm a motion.

    Returns:
        Chronologically ordered list of ``GripperEvent`` records.

    Raises:
        ValueError: If ``delta <= 0``, ``fps <= 0``, or ``gripper`` is not 1D.
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    values = np.asarray(gripper, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected 1D gripper array, got shape {values.shape}")
    if len(values) < 2:
        return []

    # Direction state: 0 = undecided, -1 = closing (tracking min), +1 = opening (tracking max).
    direction = 0
    run_max = float(values[0])
    max_idx = 0
    run_min = float(values[0])
    min_idx = 0
    peak_idx = 0
    trough_idx = 0
    trigger_idx = 0

    raw_swings: list[tuple[GripperEventKind, int, int, int]] = []

    for i in range(1, len(values)):
        v = float(values[i])
        if direction == 0:
            if v >= run_max:
                run_max = v
                max_idx = i
            if v <= run_min:
                run_min = v
                min_idx = i
            if run_max - v >= delta:
                direction = -1
                peak_idx = max_idx
                trigger_idx = i
                run_min = v
                min_idx = i
            elif v - run_min >= delta:
                direction = 1
                trough_idx = min_idx
                trigger_idx = i
                run_max = v
                max_idx = i
        elif direction == -1:
            if v < run_min:
                run_min = v
                min_idx = i
            elif v - run_min >= delta:
                raw_swings.append(("close", peak_idx, min_idx, trigger_idx))
                direction = 1
                trough_idx = min_idx
                trigger_idx = i
                run_max = v
                max_idx = i
        else:
            if v > run_max:
                run_max = v
                max_idx = i
            elif run_max - v >= delta:
                raw_swings.append(("open", trough_idx, max_idx, trigger_idx))
                direction = -1
                peak_idx = max_idx
                trigger_idx = i
                run_min = v
                min_idx = i

    if direction == -1:
        raw_swings.append(("close", peak_idx, min_idx, trigger_idx))
    elif direction == 1:
        raw_swings.append(("open", trough_idx, max_idx, trigger_idx))

    events: list[GripperEvent] = []
    for kind, ext_start, ext_end, trig in raw_swings:
        start_frame, end_frame = _refine_motion_bounds(
            values,
            kind=kind,
            ext_start=ext_start,
            ext_end=ext_end,
            trigger_idx=trig,
            delta=delta,
        )
        events.append(
            GripperEvent(
                episode_index=episode_index,
                arm=arm,
                kind=kind,
                frame=start_frame,
                end_frame=end_frame,
                time_s=start_frame / fps,
                end_time_s=end_frame / fps,
                start_pos=float(values[ext_start]),
                end_pos=float(values[ext_end]),
            )
        )
    return events


def detect_squeeze_segments(
    state_gripper: np.ndarray,
    action_gripper: np.ndarray,
    *,
    episode_index: int,
    arm: str,
    fps: float,
    gap: float = 0.05,
    min_duration_s: float = 0.2,
) -> list[SqueezeSegment]:
    """Identify contiguous active-squeeze intervals ``[start, end)``.

    A frame belongs to a squeeze when ``action_gripper - state_gripper <= -gap``
    (the commanded gripper position is significantly more closed than the
    observed state, e.g. pinching an egg with tongs or gripping a rigid object).

    Args:
        state_gripper: 1D array of observed gripper positions.
        action_gripper: 1D array of commanded gripper positions.
        episode_index: Episode index recorded in emitted segments.
        arm: Arm identifier recorded in emitted segments.
        fps: Frame rate in Hz.
        gap: Positive threshold for ``state_gripper - action_gripper >= gap``.
        min_duration_s: Minimum segment duration in seconds.

    Returns:
        Chronologically ordered list of ``SqueezeSegment`` records.

    Raises:
        ValueError: If array shapes mismatch or parameters are non-positive.
    """
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if gap <= 0.0:
        raise ValueError(f"gap must be positive, got {gap}")
    if min_duration_s <= 0.0:
        raise ValueError(f"min_duration_s must be positive, got {min_duration_s}")

    state = np.asarray(state_gripper, dtype=np.float64)
    action = np.asarray(action_gripper, dtype=np.float64)
    if state.ndim != 1 or action.ndim != 1 or len(state) != len(action):
        raise ValueError(
            f"Expected matching 1D arrays for state and action gripper, "
            f"got {state.shape} and {action.shape}"
        )
    if len(state) == 0:
        return []

    diff = action - state
    is_squeezing = diff <= -gap
    min_frames = max(1, round(min_duration_s * fps))

    padded = np.concatenate(([False], is_squeezing, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)

    segments: list[SqueezeSegment] = []
    for start_idx, end_idx in zip(starts, ends, strict=True):
        start = int(start_idx)
        end = int(end_idx)
        if end - start < min_frames:
            continue
        seg_diff = diff[start:end]
        segments.append(
            SqueezeSegment(
                episode_index=episode_index,
                arm=arm,
                start=start,
                end=end,
                start_time_s=start / fps,
                end_time_s=end / fps,
                min_gap=float(np.min(seg_diff)),
                median_gap=float(np.median(seg_diff)),
            )
        )
    return segments


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
    """Compute candidate gripper motions, squeezes, and low-speed segments.

    Args:
        episode_index: Zero-based episode index.
        length: Total frame count of the episode.
        fps: Frame rate in Hz.
        arms: Per-arm state (and optional action gripper) trajectories.
        config: Optional detection parameters; uses ``ProposeConfig()`` by default.

    Returns:
        ``EpisodeProposal`` containing sorted gripper events, squeeze intervals,
        and low-speed segments.
    """
    cfg = config or ProposeConfig()
    gripper_events: list[GripperEvent] = []
    squeeze_segments: list[SqueezeSegment] = []
    low_speed_segments: list[LowSpeedSegment] = []

    for arm_state in arms:
        gripper_events.extend(
            detect_gripper_events(
                arm_state.gripper,
                episode_index=episode_index,
                arm=arm_state.arm,
                fps=fps,
                delta=cfg.gripper_delta,
            )
        )
        if arm_state.action_gripper is not None:
            squeeze_segments.extend(
                detect_squeeze_segments(
                    arm_state.gripper,
                    arm_state.action_gripper,
                    episode_index=episode_index,
                    arm=arm_state.arm,
                    fps=fps,
                    gap=cfg.squeeze_gap,
                    min_duration_s=cfg.min_squeeze_s,
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
    squeeze_segments.sort(key=lambda seg: (seg.start, seg.arm, seg.end))
    low_speed_segments.sort(key=lambda seg: (seg.start, seg.arm, seg.end))

    return EpisodeProposal(
        episode_index=episode_index,
        length=length,
        fps=fps,
        gripper_events=tuple(gripper_events),
        squeeze_segments=tuple(squeeze_segments),
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
        config: Optional detection parameters.

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
                    "end_frame": str(ev.end_frame),
                    "time_s": f"{ev.time_s:.3f}",
                    "value": f"{ev.amplitude:+.4f}",
                },
            )
        )
    for squeeze in proposal.squeeze_segments:
        rows.append(
            (
                squeeze.start,
                squeeze.arm,
                "squeeze",
                {
                    "episode_index": str(squeeze.episode_index),
                    "arm": squeeze.arm,
                    "event_type": "squeeze",
                    "frame": str(squeeze.start),
                    "end_frame": str(squeeze.end),
                    "time_s": f"{squeeze.start_time_s:.3f}",
                    "value": f"{squeeze.median_gap:+.4f}",
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
    event_types: Collection[str] = DEFAULT_RENDER_EVENT_TYPES,
) -> list[int]:
    """Extract sorted unique event start frames for ``episode_index`` from a proposals CSV.

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
