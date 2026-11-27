"""Distribution analysis for subtask frame shares, poses, trajectories, and coverage."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from lerobot_check.calib import HeadCalibration
from lerobot_check.dataset import ArmState, DatasetView, rotation_6d_to_matrix
from lerobot_check.object_pose import (
    OBJECT_LAYER_SOURCE,
    OBJECT_POSE_FRAME,
    object_yaw_period_deg,
    validate_object_poses_sidecar,
)
from lerobot_check.schema import AnnotationRow, CoverageSpec, ObjectSpec, TaskSpec

DISTRIBUTION_SCHEMA_VERSION = 1
DEFAULT_TRAJECTORY_SAMPLES = 50


@dataclass(frozen=True)
class ProvenanceCounts:
    """Annotation row counts broken down by ``annotator`` provenance.

    Attributes:
        agent: Rows written by an agent and not yet reviewed by a human.
        human: Rows written or reviewed by a human.
        converted: Rows converted or exported automatically from another format.
    """

    agent: int = 0
    human: int = 0
    converted: int = 0

    @property
    def total(self) -> int:
        """Total number of annotation rows across all provenance categories."""
        return self.agent + self.human + self.converted

    @property
    def has_unreviewed_agent(self) -> bool:
        """Return ``True`` when at least one unreviewed ``agent`` row is present."""
        return self.agent > 0


@dataclass(frozen=True)
class SubtaskFrameShare:
    """Frame count and share statistics for a single subtask.

    Attributes:
        subtask: Subtask name from the task specification.
        episodes: Number of annotated episodes containing this subtask.
        frames: Total annotated frames belonging to this subtask.
        share: Fraction of total annotated frames in ``[0, 1]``.
    """

    subtask: str
    episodes: int
    frames: int
    share: float


@dataclass(frozen=True)
class FrameShareSummary:
    """Dataset-level frame-share breakdown across subtasks.

    Attributes:
        annotated_episodes: Sorted episode indices present in the annotation table.
        total_episodes: Total episode count considered (full dataset or filtered selection).
        skipped_episodes: Sorted episode indices omitted from the annotation table.
        total_frames: Total annotated frames across all subtasks.
        subtasks: Per-subtask breakdown ordered according to ``task_spec.subtasks``.
    """

    annotated_episodes: tuple[int, ...]
    total_episodes: int
    skipped_episodes: tuple[int, ...]
    total_frames: int
    subtasks: tuple[SubtaskFrameShare, ...]


@dataclass(frozen=True)
class ObjectPoseSample:
    """Extracted event-frame pose for one object in one episode.

    Attributes:
        episode_index: Zero-based episode index.
        object_name: Object name from ``task_spec.objects``.
        subtask: Subtask name where the pose was sampled.
        event: Event type (``"grasp"`` or ``"release"``).
        arm: Robot arm whose base frame the pose is expressed in.
        frame: Frame index used for sampling.
        x_m: Position X in meters (calibrated tool point when available, else TCP).
        y_m: Position Y in meters (calibrated tool point when available, else TCP).
        z_m: Position Z in meters (calibrated tool point when available, else TCP).
        raw_yaw_deg: Raw TCP yaw angle in ``(-180, 180]`` degrees.
        folded_yaw_deg: Symmetry-folded yaw angle in ``[0, yaw_symmetry_deg)`` degrees.
        rot6d: 6D TCP orientation ``(r1, r2, r3, r4, r5, r6)`` at the event frame.
    """

    episode_index: int
    object_name: str
    subtask: str
    event: str
    arm: str
    frame: int
    x_m: float
    y_m: float
    z_m: float
    raw_yaw_deg: float
    folded_yaw_deg: float
    rot6d: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


@dataclass(frozen=True)
class SkippedObjectEvent:
    """Record of an object whose event pose could not be extracted for an episode.

    Attributes:
        episode_index: Zero-based episode index.
        object_name: Object name from ``task_spec.objects``.
        subtask: Subtask referenced by the object spec.
        event: Event selector (``"grasp"`` or ``"release"``).
        arm: Arm referenced by the object spec.
        reason: Human-readable reason why the sample was skipped.
    """

    episode_index: int
    object_name: str
    subtask: str
    event: str
    arm: str
    reason: str


@dataclass(frozen=True)
class CoverageBin:
    """Single ``(ix, iy, yaw_bin)`` cell in an object's workspace coverage grid.

    Attributes:
        ix: Integer X grid index (``floor(x_m / xy_bin_m)``).
        iy: Integer Y grid index (``floor(y_m / xy_bin_m)``).
        yaw_bin: Zero-based yaw bin index inside ``[0, num_yaw_bins)``.
        x_min_m: Inclusive lower X bound in meters.
        x_max_m: Exclusive upper X bound in meters.
        y_min_m: Inclusive lower Y bound in meters.
        y_max_m: Exclusive upper Y bound in meters.
        yaw_min_deg: Inclusive lower folded-yaw bound in degrees.
        yaw_max_deg: Exclusive upper folded-yaw bound in degrees.
        count: Number of observed object pose samples falling in this cell.
    """

    ix: int
    iy: int
    yaw_bin: int
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    yaw_min_deg: float
    yaw_max_deg: float
    count: int


@dataclass(frozen=True)
class ObjectCoverage:
    """Workspace XY x yaw coverage grid for one object.

    Attributes:
        spec: ``ObjectSpec`` from the task spec.
        coverage_spec: ``CoverageSpec`` binning parameters.
        sample_count: Number of extracted samples contributing to this grid.
        ix_range: Inclusive ``(ix_min, ix_max)`` bounds, or ``None`` if empty.
        iy_range: Inclusive ``(iy_min, iy_max)`` bounds, or ``None`` if empty.
        num_yaw_bins: Number of yaw bins covering ``[0, yaw_symmetry_deg)``.
        bins: All grid cells across ``ix_range x iy_range x [0..num_yaw_bins-1]``.
    """

    spec: ObjectSpec
    coverage_spec: CoverageSpec
    sample_count: int
    ix_range: tuple[int, int] | None
    iy_range: tuple[int, int] | None
    num_yaw_bins: int
    bins: tuple[CoverageBin, ...]


@dataclass(frozen=True)
class PoseXYZYaw:
    """3D position and ZYX yaw angle in an arm's base frame.

    Attributes:
        x_m: X coordinate in meters.
        y_m: Y coordinate in meters.
        z_m: Z coordinate in meters.
        yaw_deg: Yaw angle in ``(-180, 180]`` degrees.
    """

    x_m: float
    y_m: float
    z_m: float
    yaw_deg: float


@dataclass(frozen=True)
class ResampledTrajectoryPoint:
    """Single progress-sampled pose along a subtask trajectory.

    Attributes:
        progress: Normalized progress in ``[0.0, 1.0]``.
        x_m: X coordinate in meters.
        y_m: Y coordinate in meters.
        z_m: Z coordinate in meters.
        yaw_deg: Yaw angle in ``(-180, 180]`` degrees.
    """

    progress: float
    x_m: float
    y_m: float
    z_m: float
    yaw_deg: float


@dataclass(frozen=True)
class EndpointCoverage:
    """Spatial and angular spread summary for a set of subtask start or end poses.

    Attributes:
        occupied_xy_cells: Number of distinct ``(floor(x/xy_bin_m), floor(y/xy_bin_m))`` cells.
        yaw_span_deg: Shortest circular arc in degrees covering all observed yaws (``[0, 360)``).
    """

    occupied_xy_cells: int
    yaw_span_deg: float


def count_annotators(rows: Sequence[AnnotationRow]) -> ProvenanceCounts:
    """Count annotation rows per ``annotator`` category."""
    agent = 0
    human = 0
    converted = 0
    for row in rows:
        if row.annotator == "agent":
            agent += 1
        elif row.annotator == "human":
            human += 1
        elif row.annotator == "converted":
            converted += 1
    return ProvenanceCounts(agent=agent, human=human, converted=converted)


def parse_episode_filter(values: Sequence[str], total_episodes: int) -> list[int]:
    """Parse episode ranges/lists into sorted unique zero-based episode indices.

    Supports single integers (``"5"``), inclusive hyphen ranges (``"0-10"``),
    and half-open colon ranges (``"0:11"``), separated by commas or whitespace.

    Args:
        values: Raw CLI episode filter tokens.
        total_episodes: Total episode count in dataset metadata for bounds checks.

    Returns:
        Sorted list of unique episode indices within ``[0, total_episodes)``.

    Raises:
        ValueError: If any token is malformed, empty, or out of range.
    """
    episodes: set[int] = set()
    for raw in values:
        for token in (part.strip() for part in raw.split(",")):
            if not token:
                continue
            if ":" in token:
                parts = token.split(":")
                if len(parts) != 2:
                    raise ValueError(f"Invalid half-open episode range: {token!r}")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if parts[1] else total_episodes
                if start < 0 or end > total_episodes or start >= end:
                    raise ValueError(
                        f"Episode range {token!r} is empty or outside [0, {total_episodes})"
                    )
                episodes.update(range(start, end))
            elif "-" in token:
                parts = token.split("-")
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    raise ValueError(f"Invalid inclusive episode range: {token!r}")
                start, end_incl = int(parts[0]), int(parts[1])
                if start < 0 or end_incl >= total_episodes or start > end_incl:
                    raise ValueError(
                        f"Episode range {token!r} is empty or outside [0, {total_episodes - 1}]"
                    )
                episodes.update(range(start, end_incl + 1))
            else:
                ep = int(token)
                if not 0 <= ep < total_episodes:
                    raise ValueError(f"Episode {ep} is outside [0, {total_episodes - 1}]")
                episodes.add(ep)

    if not episodes:
        raise ValueError("Episode filter matched no episodes")
    return sorted(episodes)


def compute_frame_shares(
    rows: Sequence[AnnotationRow],
    task_spec: TaskSpec,
    *,
    episode_lengths: Mapping[int, int],
) -> FrameShareSummary:
    """Compute per-subtask frame counts and shares.

    Args:
        rows: Validated annotation rows (already filtered if an episode filter is active).
        task_spec: Task specification providing subtask order.
        episode_lengths: Mapping from considered ``episode_index`` to frame count.

    Returns:
        ``FrameShareSummary`` with per-subtask frame counts and shares.
    """
    subtask_frames: dict[str, int] = {s.name: 0 for s in task_spec.subtasks}
    subtask_episodes: dict[str, set[int]] = {s.name: set() for s in task_spec.subtasks}
    annotated_episodes: set[int] = set()

    for row in rows:
        annotated_episodes.add(row.episode_index)
        length = row.end - row.start
        subtask_frames[row.subtask] += length
        subtask_episodes[row.subtask].add(row.episode_index)

    total_frames = sum(subtask_frames.values())
    skipped = tuple(sorted(set(episode_lengths) - annotated_episodes))

    subtask_shares: list[SubtaskFrameShare] = []
    for subtask_spec in task_spec.subtasks:
        name = subtask_spec.name
        frames = subtask_frames[name]
        subtask_shares.append(
            SubtaskFrameShare(
                subtask=name,
                episodes=len(subtask_episodes[name]),
                frames=frames,
                share=(frames / total_frames) if total_frames > 0 else 0.0,
            )
        )

    return FrameShareSummary(
        annotated_episodes=tuple(sorted(annotated_episodes)),
        total_episodes=len(episode_lengths),
        skipped_episodes=skipped,
        total_frames=total_frames,
        subtasks=tuple(subtask_shares),
    )


def rotation_matrix_to_yaw_deg(rot: np.ndarray) -> float:
    """Extract ZYX yaw angle in degrees from a 3x3 rotation matrix.

    Uses ``atan2(R[1, 0], R[0, 0])`` (direction of the TCP x-axis projected onto
    the arm base XY plane), falling back to ``atan2(-R[0, 1], R[1, 1])`` near
    gimbal lock when the TCP x-axis is vertical.

    Args:
        rot: 3x3 rotation matrix reconstructed via ``rotation_6d_to_matrix``.

    Returns:
        Yaw angle in degrees in ``(-180.0, 180.0]``.
    """
    matrix = np.asarray(rot, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got shape {matrix.shape}")

    horiz = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    if horiz > 1e-6:
        yaw_rad = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        yaw_rad = math.atan2(-float(matrix[0, 1]), float(matrix[1, 1]))
    deg = math.degrees(yaw_rad)
    # Normalize -180.0 to +180.0 so the half-open range is strictly (-180, 180].
    if deg <= -180.0:
        deg += 360.0
    return deg


def _normalize_signed_yaw_deg(yaw_deg: float) -> float:
    """Normalize an angle in degrees into ``(-180.0, 180.0]``."""
    wrapped = (float(yaw_deg) + 180.0) % 360.0 - 180.0
    if wrapped <= -180.0:
        wrapped += 360.0
    return wrapped


def fold_yaw_deg(yaw_deg: float, yaw_symmetry_deg: float) -> float:
    """Fold a yaw angle into ``[0, yaw_symmetry_deg)`` respecting rotational symmetry.

    For example, with ``yaw_symmetry_deg=180.0``, both ``170.0`` and ``-10.0``
    fold to ``170.0`` degrees.

    Args:
        yaw_deg: Raw yaw angle in degrees.
        yaw_symmetry_deg: Positive rotational symmetry period in degrees (<= 360).

    Returns:
        Folded yaw angle strictly in ``[0.0, yaw_symmetry_deg)``.
    """
    if yaw_symmetry_deg <= 0.0 or yaw_symmetry_deg > 360.0:
        raise ValueError(f"yaw_symmetry_deg must be in (0, 360], got {yaw_symmetry_deg}")

    folded = float(yaw_deg) % float(yaw_symmetry_deg)
    # Guard against negative near-zero floats wrapping to yaw_symmetry_deg under modulo.
    if folded >= yaw_symmetry_deg or abs(folded - yaw_symmetry_deg) < 1e-9:
        folded = 0.0
    return folded


def yaw_bin_index(
    folded_yaw_deg: float,
    *,
    yaw_symmetry_deg: float,
    yaw_bin_deg: float,
) -> tuple[int, int]:
    """Map a folded yaw angle to ``(bin_index, num_yaw_bins)``.

    Args:
        folded_yaw_deg: Yaw already folded into ``[0, yaw_symmetry_deg)``.
        yaw_symmetry_deg: Symmetry period in degrees.
        yaw_bin_deg: Width of each yaw bin in degrees.

    Returns:
        Tuple ``(bin_index, num_yaw_bins)`` where ``0 <= bin_index < num_yaw_bins``.
    """
    if yaw_bin_deg <= 0.0:
        raise ValueError(f"yaw_bin_deg must be positive, got {yaw_bin_deg}")
    num_bins = max(1, math.ceil(yaw_symmetry_deg / yaw_bin_deg - 1e-9))
    idx = min(math.floor(folded_yaw_deg / yaw_bin_deg), num_bins - 1)
    return max(0, idx), num_bins


def circular_yaw_span_deg(yaws_deg: Sequence[float]) -> float:
    """Compute the minimum circular arc in degrees covering all ``yaws_deg``.

    Returns ``0.0`` for 0 or 1 angles. For ``M >= 2`` angles wrapped to ``[0, 360)``,
    sorts the angles, finds the largest consecutive gap on the circle (including
    the wrap-around gap), and returns ``360.0 - max_gap``.

    Args:
        yaws_deg: Sequence of yaw angles in degrees.

    Returns:
        Circular span in ``[0.0, 360.0)`` degrees.
    """
    if len(yaws_deg) <= 1:
        return 0.0
    wrapped = sorted(float(y) % 360.0 for y in yaws_deg)
    max_gap = (wrapped[0] + 360.0) - wrapped[-1]
    for idx in range(len(wrapped) - 1):
        gap = wrapped[idx + 1] - wrapped[idx]
        if gap > max_gap:
            max_gap = gap
    return max(0.0, 360.0 - max_gap)


def compute_endpoint_coverage(
    poses: Sequence[PoseXYZYaw],
    xy_bin_m: float,
) -> EndpointCoverage:
    """Compute occupied XY cell count and circular yaw span for a set of poses.

    Args:
        poses: Start or end poses across episodes.
        xy_bin_m: Side length of each XY grid cell in meters.

    Returns:
        ``EndpointCoverage`` containing occupied cell count and circular yaw span.
    """
    if xy_bin_m <= 0.0:
        raise ValueError(f"xy_bin_m must be positive, got {xy_bin_m}")
    if not poses:
        return EndpointCoverage(occupied_xy_cells=0, yaw_span_deg=0.0)

    cells = {(math.floor(p.x_m / xy_bin_m), math.floor(p.y_m / xy_bin_m)) for p in poses}
    span = circular_yaw_span_deg([p.yaw_deg for p in poses])
    return EndpointCoverage(occupied_xy_cells=len(cells), yaw_span_deg=span)


def compute_trajectory_dispersion_cm(trajectories_xyz: Sequence[np.ndarray]) -> float | None:
    """Compute trajectory dispersion across episodes in centimeters.

    Given ``M`` trajectories each resampled to ``K`` progress points of shape ``(K, 3)``
    in meters, computes at each progress index ``k`` the 3D Euclidean RMS spread
    around the cross-episode mean position (population std, ``ddof=0``):

    ``sigma_k = sqrt(mean_e ||P[e, k] - mean_e(P[:, k])||_2^2)``

    and returns the mean of ``sigma_k`` over ``k = 0 .. K-1`` converted to centimeters.

    Args:
        trajectories_xyz: Sequence of arrays each of shape ``(K, 3)`` in meters.

    Returns:
        Mean positional spread in centimeters, ``0.0`` for a single trajectory, or
        ``None`` if ``trajectories_xyz`` is empty.

    Raises:
        ValueError: If trajectories do not share the same ``(K, 3)`` shape.
    """
    if not trajectories_xyz:
        return None
    stacked = np.stack([np.asarray(t, dtype=np.float64) for t in trajectories_xyz], axis=0)
    if stacked.ndim != 3 or stacked.shape[2] != 3 or stacked.shape[1] == 0:
        raise ValueError(f"Expected trajectories of shape (M, K, 3), got {stacked.shape}")
    if stacked.shape[0] == 1:
        return 0.0

    mean_pos = np.mean(stacked, axis=0, keepdims=True)
    sq_dist = np.sum((stacked - mean_pos) ** 2, axis=2)  # (M, K)
    std_per_progress_m = np.sqrt(np.mean(sq_dist, axis=0))  # (K,)
    return float(np.mean(std_per_progress_m) * 100.0)


def resample_segment_pose(
    xyz: np.ndarray,
    rot6d: np.ndarray,
    num_samples: int = DEFAULT_TRAJECTORY_SAMPLES,
) -> list[ResampledTrajectoryPoint]:
    """Resample a subtask TCP trajectory uniformly over normalized progress ``0 -> 1``.

    Uses linear interpolation for ``(x, y, z)`` and shortest-arc linear interpolation
    for yaw angles between adjacent frames, guaranteeing exact agreement with frame
    ``0`` at ``progress=0.0`` and frame ``L-1`` at ``progress=1.0``.

    Args:
        xyz: Float64 array of shape ``(L, 3)`` in meters for frames ``[start, end)``.
        rot6d: Float64 array of shape ``(L, 6)`` for frames ``[start, end)``.
        num_samples: Number of evenly spaced progress samples ``K >= 2``.

    Returns:
        List of ``num_samples`` ``ResampledTrajectoryPoint`` objects.

    Raises:
        ValueError: If ``num_samples < 2`` or input arrays are empty / mismatched.
    """
    if num_samples < 2:
        raise ValueError(f"num_samples must be >= 2, got {num_samples}")
    xyz_arr = np.asarray(xyz, dtype=np.float64)
    rot_arr = np.asarray(rot6d, dtype=np.float64)
    n = len(xyz_arr)
    if n == 0 or len(rot_arr) != n:
        raise ValueError(
            f"Expected non-empty matching xyz and rot6d arrays, got {xyz_arr.shape}, "
            f"{rot_arr.shape}"
        )

    rots = rotation_6d_to_matrix(rot_arr)
    yaws = np.array([rotation_matrix_to_yaw_deg(rots[i]) for i in range(n)], dtype=np.float64)

    progress_grid = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    if n == 1:
        return [
            ResampledTrajectoryPoint(
                progress=float(p),
                x_m=float(xyz_arr[0, 0]),
                y_m=float(xyz_arr[0, 1]),
                z_m=float(xyz_arr[0, 2]),
                yaw_deg=float(yaws[0]),
            )
            for p in progress_grid
        ]

    points: list[ResampledTrajectoryPoint] = []
    max_idx = n - 1
    for p in progress_grid:
        t = float(p) * max_idx
        f0 = min(math.floor(t), max_idx)
        f1 = min(f0 + 1, max_idx)
        alpha = t - f0

        pos = (1.0 - alpha) * xyz_arr[f0] + alpha * xyz_arr[f1]
        delta_yaw = _normalize_signed_yaw_deg(float(yaws[f1]) - float(yaws[f0]))
        yaw = _normalize_signed_yaw_deg(float(yaws[f0]) + alpha * delta_yaw)
        points.append(
            ResampledTrajectoryPoint(
                progress=float(p),
                x_m=float(pos[0]),
                y_m=float(pos[1]),
                z_m=float(pos[2]),
                yaw_deg=yaw,
            )
        )
    return points


def apply_tool_offset(
    tcp_xyz: np.ndarray,
    tcp_rot6d: np.ndarray,
    tool_offset: np.ndarray,
) -> np.ndarray:
    """Transform TCP positions to the calibrated tool point ``p_tcp + R_tcp @ tool_offset``.

    Args:
        tcp_xyz: Float64 array of shape ``(N, 3)`` or ``(3,)`` in base frame (meters).
        tcp_rot6d: Float64 array of shape ``(N, 6)`` or ``(6,)`` storing 6D rotation.
        tool_offset: Float64 3-vector ``(ox, oy, oz)`` in the TCP frame (meters).

    Returns:
        Tool-point positions in base frame with the same shape as ``tcp_xyz``.
    """
    xyz = np.asarray(tcp_xyz, dtype=np.float64)
    offset = np.asarray(tool_offset, dtype=np.float64).reshape(3)
    rot = rotation_6d_to_matrix(tcp_rot6d)
    if xyz.ndim == 1:
        return xyz + rot @ offset
    return xyz + np.einsum("nij,j->ni", rot, offset)


def extract_episode_object_poses(
    *,
    episode_index: int,
    ep_rows: Sequence[AnnotationRow],
    arms_by_name: Mapping[str, ArmState],
    objects: Sequence[ObjectSpec],
    tool_offsets: Mapping[str, np.ndarray] | None = None,
) -> tuple[list[ObjectPoseSample], list[SkippedObjectEvent]]:
    """Extract object initial poses at event frames for one episode.

    Objects whose subtask is absent from the episode or whose ``grasp`` frame is
    missing are returned in ``skipped`` rather than silently dropped.

    Args:
        episode_index: Zero-based episode index.
        ep_rows: Annotation rows for this episode.
        arms_by_name: Mapping from arm name to ``ArmState`` for this episode.
        objects: Object specifications from ``task_spec.objects``.
        tool_offsets: Optional mapping ``arm -> tool_offset (3,)`` from head-camera
            calibration; when present for ``obj.arm``, positions are shifted to
            ``p_tcp + R_tcp @ tool_offset``.

    Returns:
        Tuple ``(samples, skipped_events)``.
    """
    row_by_subtask = {row.subtask: row for row in ep_rows}
    samples: list[ObjectPoseSample] = []
    skipped: list[SkippedObjectEvent] = []

    for obj in objects:
        row = row_by_subtask.get(obj.subtask)
        if row is None:
            skipped.append(
                SkippedObjectEvent(
                    episode_index=episode_index,
                    object_name=obj.name,
                    subtask=obj.subtask,
                    event=obj.event,
                    arm=obj.arm,
                    reason=f"subtask {obj.subtask!r} absent in episode",
                )
            )
            continue

        if obj.arm not in arms_by_name:
            skipped.append(
                SkippedObjectEvent(
                    episode_index=episode_index,
                    object_name=obj.name,
                    subtask=obj.subtask,
                    event=obj.event,
                    arm=obj.arm,
                    reason=f"arm {obj.arm!r} not present in dataset",
                )
            )
            continue

        frame = row.grasp if obj.event == "grasp" else row.end - 1
        if frame is None:
            skipped.append(
                SkippedObjectEvent(
                    episode_index=episode_index,
                    object_name=obj.name,
                    subtask=obj.subtask,
                    event=obj.event,
                    arm=obj.arm,
                    reason="missing grasp frame",
                )
            )
            continue

        arm_state = arms_by_name[obj.arm]
        rot6d_arr = arm_state.tcp_rot6d[frame]
        rot = rotation_6d_to_matrix(rot6d_arr)
        xyz = arm_state.tcp_xyz[frame]
        if tool_offsets is not None and obj.arm in tool_offsets:
            xyz = xyz + rot @ np.asarray(tool_offsets[obj.arm], dtype=np.float64).reshape(3)
        raw_yaw = rotation_matrix_to_yaw_deg(rot)
        folded_yaw = fold_yaw_deg(raw_yaw, obj.yaw_symmetry_deg)
        rot6d = (
            float(rot6d_arr[0]),
            float(rot6d_arr[1]),
            float(rot6d_arr[2]),
            float(rot6d_arr[3]),
            float(rot6d_arr[4]),
            float(rot6d_arr[5]),
        )

        samples.append(
            ObjectPoseSample(
                episode_index=episode_index,
                object_name=obj.name,
                subtask=obj.subtask,
                event=obj.event,
                arm=obj.arm,
                frame=frame,
                x_m=float(xyz[0]),
                y_m=float(xyz[1]),
                z_m=float(xyz[2]),
                raw_yaw_deg=raw_yaw,
                folded_yaw_deg=folded_yaw,
                rot6d=rot6d,
            )
        )

    return samples, skipped


def compute_object_coverage(
    obj: ObjectSpec,
    samples: Sequence[ObjectPoseSample],
    coverage_spec: CoverageSpec,
) -> ObjectCoverage:
    """Build an XY x yaw workspace coverage grid for one object.

    XY cell indices use canonical ``floor(coord / xy_bin_m)`` coordinates so grid
    alignment stays stable across episode subsets and datasets.

    Args:
        obj: Target object specification.
        samples: Pose samples (filtered internally to ``obj.name``).
        coverage_spec: Grid cell size and sparsity threshold.

    Returns:
        ``ObjectCoverage`` with all bounding-box cells sorted deterministically.
    """
    obj_samples = [s for s in samples if s.object_name == obj.name]
    _, num_yaw_bins = yaw_bin_index(
        0.0,
        yaw_symmetry_deg=obj.yaw_symmetry_deg,
        yaw_bin_deg=coverage_spec.yaw_bin_deg,
    )
    if not obj_samples:
        return ObjectCoverage(
            spec=obj,
            coverage_spec=coverage_spec,
            sample_count=0,
            ix_range=None,
            iy_range=None,
            num_yaw_bins=num_yaw_bins,
            bins=(),
        )

    xy_step = coverage_spec.xy_bin_m
    yaw_step = coverage_spec.yaw_bin_deg
    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    ix_vals: list[int] = []
    iy_vals: list[int] = []

    for sample in obj_samples:
        ix = math.floor(sample.x_m / xy_step)
        iy = math.floor(sample.y_m / xy_step)
        y_bin, _ = yaw_bin_index(
            sample.folded_yaw_deg,
            yaw_symmetry_deg=obj.yaw_symmetry_deg,
            yaw_bin_deg=yaw_step,
        )
        counts[(ix, iy, y_bin)] += 1
        ix_vals.append(ix)
        iy_vals.append(iy)

    ix_min, ix_max = min(ix_vals), max(ix_vals)
    iy_min, iy_max = min(iy_vals), max(iy_vals)

    all_bins: list[CoverageBin] = []
    for ix in range(ix_min, ix_max + 1):
        for iy in range(iy_min, iy_max + 1):
            for y_bin in range(num_yaw_bins):
                yaw_lo = y_bin * yaw_step
                yaw_hi = min(obj.yaw_symmetry_deg, (y_bin + 1) * yaw_step)
                all_bins.append(
                    CoverageBin(
                        ix=ix,
                        iy=iy,
                        yaw_bin=y_bin,
                        x_min_m=ix * xy_step,
                        x_max_m=(ix + 1) * xy_step,
                        y_min_m=iy * xy_step,
                        y_max_m=(iy + 1) * xy_step,
                        yaw_min_deg=yaw_lo,
                        yaw_max_deg=yaw_hi,
                        count=counts.get((ix, iy, y_bin), 0),
                    )
                )

    return ObjectCoverage(
        spec=obj,
        coverage_spec=coverage_spec,
        sample_count=len(obj_samples),
        ix_range=(ix_min, ix_max),
        iy_range=(iy_min, iy_max),
        num_yaw_bins=num_yaw_bins,
        bins=tuple(all_bins),
    )


def _serialize_object_coverage(cov: ObjectCoverage) -> dict[str, Any]:
    return {
        "sample_count": cov.sample_count,
        "xy_bin_m": cov.coverage_spec.xy_bin_m,
        "yaw_bin_deg": cov.coverage_spec.yaw_bin_deg,
        "sparse_threshold": cov.coverage_spec.sparse_threshold,
        "num_yaw_bins": cov.num_yaw_bins,
        "ix_range": list(cov.ix_range) if cov.ix_range is not None else None,
        "iy_range": list(cov.iy_range) if cov.iy_range is not None else None,
        "bins": [
            {
                "ix": b.ix,
                "iy": b.iy,
                "yaw_bin": b.yaw_bin,
                "x_min_m": b.x_min_m,
                "x_max_m": b.x_max_m,
                "y_min_m": b.y_min_m,
                "y_max_m": b.y_max_m,
                "yaw_min_deg": b.yaw_min_deg,
                "yaw_max_deg": b.yaw_max_deg,
                "count": b.count,
            }
            for b in cov.bins
        ],
    }


def _build_object_layer(
    obj: ObjectSpec,
    sidecar_entry: Mapping[str, Any],
    *,
    coverage_spec: CoverageSpec,
    considered_episodes: Collection[int],
) -> dict[str, Any]:
    sidecar_arm = str(sidecar_entry["arm"])
    if sidecar_arm != obj.arm:
        raise ValueError(
            f"Object-poses sidecar arm mismatch for {obj.name!r}: "
            f"sidecar={sidecar_arm!r}, task_spec={obj.arm!r}"
        )
    yaw_period = object_yaw_period_deg(obj.yaw_symmetry_deg)
    plane_z_raw = sidecar_entry.get("plane_z_m")
    plane_z_m = float(plane_z_raw) if plane_z_raw is not None else None

    poses_out: list[dict[str, Any]] = []
    skipped_out: list[dict[str, Any]] = []
    cov_samples: list[ObjectPoseSample] = []

    raw_episodes = sorted(
        sidecar_entry.get("episodes", []),
        key=lambda item: int(item["episode_index"]),
    )
    for ep_item in raw_episodes:
        ep_idx = int(ep_item["episode_index"])
        if ep_idx not in considered_episodes:
            continue
        status = str(ep_item["status"])
        if status != "ok":
            skipped_out.append({"episode_index": ep_idx, "reason": status})
            continue
        frame = int(ep_item.get("frame", OBJECT_POSE_FRAME))
        x_m = float(ep_item["x_m"])
        y_m = float(ep_item["y_m"])
        z_m = float(ep_item["z_m"])
        raw_yaw_deg = float(ep_item["raw_yaw_deg"])
        folded_yaw_deg = fold_yaw_deg(raw_yaw_deg, yaw_period)
        length_m = float(ep_item["length_m"])
        width_m = float(ep_item["width_m"])
        poses_out.append(
            {
                "episode_index": ep_idx,
                "frame": frame,
                "arm": obj.arm,
                "x_m": x_m,
                "y_m": y_m,
                "z_m": z_m,
                "raw_yaw_deg": raw_yaw_deg,
                "folded_yaw_deg": folded_yaw_deg,
                "length_m": length_m,
                "width_m": width_m,
            }
        )
        cov_samples.append(
            ObjectPoseSample(
                episode_index=ep_idx,
                object_name=obj.name,
                subtask=obj.subtask,
                event=obj.event,
                arm=obj.arm,
                frame=frame,
                x_m=x_m,
                y_m=y_m,
                z_m=z_m,
                raw_yaw_deg=raw_yaw_deg,
                folded_yaw_deg=folded_yaw_deg,
            )
        )

    cov_obj = replace(obj, yaw_symmetry_deg=yaw_period)
    cov = compute_object_coverage(cov_obj, cov_samples, coverage_spec)
    return {
        "source": OBJECT_LAYER_SOURCE,
        "frame": OBJECT_POSE_FRAME,
        "arm": obj.arm,
        "plane_z_m": plane_z_m,
        "yaw_period_deg": yaw_period,
        "poses": poses_out,
        "skipped": skipped_out,
        "coverage": _serialize_object_coverage(cov),
    }


def compute_distribution_from_states(
    *,
    repo_id: str,
    revision: str,
    fps: float,
    episode_lengths: Mapping[int, int],
    arm_states_by_episode: Mapping[int, Mapping[str, ArmState]],
    rows: Sequence[AnnotationRow],
    task_spec: TaskSpec,
    annotations_path: Path | str | None = None,
    task_spec_path: Path | str | None = None,
    episode_filter: Collection[int] | None = None,
    trajectory_samples: int = DEFAULT_TRAJECTORY_SAMPLES,
    tool_offsets: Mapping[str, np.ndarray] | None = None,
    calib_path: Path | str | None = None,
    calib_mtime: str | None = None,
    calib_sha256: str | None = None,
    object_poses: Mapping[str, Any] | None = None,
    object_poses_sha256: str | None = None,
) -> dict[str, Any]:
    """Compute the complete ``distribution.json`` payload from in-memory arm states.

    When ``tool_offsets`` is provided, every position for calibrated arms is computed
    at the calibrated tool point ``p_tcp + R_tcp @ tool_offset`` rather than the raw
    TCP; yaw angles and ``rot6d`` orientations are unchanged.

    Args:
        repo_id: Dataset repository ID.
        revision: Dataset revision string.
        fps: Dataset frame rate in Hz.
        episode_lengths: Full dataset ``{episode_index: length}`` mapping.
        arm_states_by_episode: Mapping ``episode_index -> {arm_name: ArmState}``
            for every annotated episode included in the computation.
        rows: Validated annotation rows for the dataset.
        task_spec: Task specification.
        annotations_path: Path to the source annotations CSV recorded in provenance.
        task_spec_path: Path to the source task spec YAML recorded in provenance.
        episode_filter: Optional subset of episode indices to restrict all metrics to.
        trajectory_samples: Number of evenly spaced progress points per trajectory (default 50).
        tool_offsets: Optional mapping ``arm -> tool_offset (3,)`` from head calibration.
        calib_path: Optional path to ``head.json`` recorded in provenance.
        calib_mtime: Optional UTC ISO-8601 modification timestamp of ``head.json``.

    Returns:
        JSON-serializable dictionary matching the ``distribution.json`` v1 schema.
    """
    if trajectory_samples < 2:
        raise ValueError(f"trajectory_samples must be >= 2, got {trajectory_samples}")

    normalized_offsets: dict[str, np.ndarray] = (
        {
            str(arm): np.asarray(vec, dtype=np.float64).reshape(3)
            for arm, vec in tool_offsets.items()
        }
        if tool_offsets
        else {}
    )
    position_point = "tool_point" if normalized_offsets else "tcp"

    dataset_total_episodes = len(episode_lengths)
    if episode_filter is not None:
        selected_eps = sorted(set(episode_filter))
        selected_set = set(selected_eps)
        considered_lengths = {
            ep: length for ep, length in episode_lengths.items() if ep in selected_set
        }
        filtered_rows = [row for row in rows if row.episode_index in selected_set]
    else:
        selected_eps = None
        considered_lengths = dict(episode_lengths)
        filtered_rows = list(rows)

    provenance = count_annotators(filtered_rows)
    frame_shares = compute_frame_shares(
        filtered_rows,
        task_spec,
        episode_lengths=considered_lengths,
    )

    by_episode: dict[int, list[AnnotationRow]] = defaultdict(list)
    for row in filtered_rows:
        by_episode[row.episode_index].append(row)

    all_object_poses: list[ObjectPoseSample] = []
    all_skipped_objects: list[SkippedObjectEvent] = []
    episode_records: list[dict[str, Any]] = []

    subtask_arm_episodes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    subtask_arm_start_poses: dict[tuple[str, str], list[PoseXYZYaw]] = defaultdict(list)
    subtask_arm_end_poses: dict[tuple[str, str], list[PoseXYZYaw]] = defaultdict(list)
    subtask_arm_traj_xyz: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)

    participating_arms_by_subtask: dict[str, tuple[str, ...]] = {
        st.name: task_spec.arms_for(st.name) for st in task_spec.subtasks
    }
    last_spec_subtask = task_spec.subtasks[-1].name

    for ep_idx in sorted(by_episode):
        ep_rows = sorted(by_episode[ep_idx], key=lambda r: (r.start, r.end))
        arms_by_name = arm_states_by_episode[ep_idx]

        episode_records.append(
            {
                "episode_index": ep_idx,
                "length": considered_lengths[ep_idx],
                "subtasks": [
                    {
                        "subtask": r.subtask,
                        "start": r.start,
                        "end": r.end,
                        "frames": r.end - r.start,
                        "grasp": r.grasp,
                        "success": r.success,
                        "annotator": r.annotator,
                    }
                    for r in ep_rows
                ],
            }
        )

        ep_poses, ep_skipped = extract_episode_object_poses(
            episode_index=ep_idx,
            ep_rows=ep_rows,
            arms_by_name=arms_by_name,
            objects=task_spec.objects,
            tool_offsets=normalized_offsets or None,
        )
        all_object_poses.extend(ep_poses)
        all_skipped_objects.extend(ep_skipped)

        for row in ep_rows:
            is_last_subtask = row.subtask == last_spec_subtask
            if row.grasp is not None:
                start_frame = row.grasp
                start_source = "grasp"
            else:
                start_frame = row.start
                start_source = "start"
            start_offset = start_frame - row.start

            for arm_name in participating_arms_by_subtask[row.subtask]:
                if arm_name not in arms_by_name:
                    continue
                arm_state = arms_by_name[arm_name]
                seg_xyz = arm_state.tcp_xyz[row.start : row.end]
                seg_rot6d = arm_state.tcp_rot6d[row.start : row.end]
                if arm_name in normalized_offsets:
                    seg_xyz = apply_tool_offset(seg_xyz, seg_rot6d, normalized_offsets[arm_name])
                resampled = resample_segment_pose(
                    seg_xyz, seg_rot6d, num_samples=trajectory_samples
                )

                start_xyz = seg_xyz[start_offset]
                start_rot6d_arr = seg_rot6d[start_offset]
                start_yaw = rotation_matrix_to_yaw_deg(rotation_6d_to_matrix(start_rot6d_arr))
                start_pose = PoseXYZYaw(
                    x_m=float(start_xyz[0]),
                    y_m=float(start_xyz[1]),
                    z_m=float(start_xyz[2]),
                    yaw_deg=start_yaw,
                )

                key = (row.subtask, arm_name)
                subtask_arm_start_poses[key].append(start_pose)
                subtask_arm_traj_xyz[key].append(
                    np.array([[pt.x_m, pt.y_m, pt.z_m] for pt in resampled], dtype=np.float64)
                )

                end_frame: int | None
                end_pose_payload: dict[str, Any] | None
                if is_last_subtask:
                    end_frame = None
                    end_pose_payload = None
                else:
                    end_frame = row.end - 1
                    end_xyz = seg_xyz[-1]
                    end_rot6d_arr = seg_rot6d[-1]
                    end_yaw = rotation_matrix_to_yaw_deg(rotation_6d_to_matrix(end_rot6d_arr))
                    end_pose = PoseXYZYaw(
                        x_m=float(end_xyz[0]),
                        y_m=float(end_xyz[1]),
                        z_m=float(end_xyz[2]),
                        yaw_deg=end_yaw,
                    )
                    subtask_arm_end_poses[key].append(end_pose)
                    end_pose_payload = {
                        "x_m": end_pose.x_m,
                        "y_m": end_pose.y_m,
                        "z_m": end_pose.z_m,
                        "yaw_deg": end_pose.yaw_deg,
                        "rot6d": [float(v) for v in end_rot6d_arr],
                    }

                subtask_arm_episodes[key].append(
                    {
                        "episode_index": ep_idx,
                        "start_frame": start_frame,
                        "start_source": start_source,
                        "end_frame": end_frame,
                        "start_pose": {
                            "x_m": start_pose.x_m,
                            "y_m": start_pose.y_m,
                            "z_m": start_pose.z_m,
                            "yaw_deg": start_pose.yaw_deg,
                            "rot6d": [float(v) for v in start_rot6d_arr],
                        },
                        "end_pose": end_pose_payload,
                        "trajectory": [
                            {
                                "progress": pt.progress,
                                "x_m": pt.x_m,
                                "y_m": pt.y_m,
                                "z_m": pt.z_m,
                                "yaw_deg": pt.yaw_deg,
                            }
                            for pt in resampled
                        ],
                    }
                )

    sidecar_objects: Mapping[str, Any] = {}
    if object_poses is not None:
        validate_object_poses_sidecar(
            object_poses,
            repo_id=repo_id,
            revision=revision,
            calibration_sha256=calib_sha256,
        )
        sidecar_objects = object_poses["objects"]

    objects_payload: list[dict[str, Any]] = []
    has_object_layer = False
    for obj in task_spec.objects:
        obj_poses = [p for p in all_object_poses if p.object_name == obj.name]
        obj_skipped = [s for s in all_skipped_objects if s.object_name == obj.name]
        cov = compute_object_coverage(obj, obj_poses, task_spec.coverage)
        median_z_m = float(np.median([p.z_m for p in obj_poses])) if obj_poses else None
        obj_entry: dict[str, Any] = {
            "name": obj.name,
            "subtask": obj.subtask,
            "event": obj.event,
            "arm": obj.arm,
            "yaw_symmetry_deg": obj.yaw_symmetry_deg,
            "median_z_m": median_z_m,
            "poses": [
                {
                    "episode_index": p.episode_index,
                    "frame": p.frame,
                    "arm": p.arm,
                    "x_m": p.x_m,
                    "y_m": p.y_m,
                    "z_m": p.z_m,
                    "raw_yaw_deg": p.raw_yaw_deg,
                    "folded_yaw_deg": p.folded_yaw_deg,
                    "rot6d": list(p.rot6d),
                }
                for p in obj_poses
            ],
            "skipped": [
                {"episode_index": s.episode_index, "reason": s.reason} for s in obj_skipped
            ],
            "coverage": _serialize_object_coverage(cov),
        }
        if obj.name in sidecar_objects:
            obj_entry["object_layer"] = _build_object_layer(
                obj,
                sidecar_objects[obj.name],
                coverage_spec=task_spec.coverage,
                considered_episodes=considered_lengths,
            )
            has_object_layer = True
        objects_payload.append(obj_entry)

    subtasks_payload: list[dict[str, Any]] = []
    for st in task_spec.subtasks:
        arms_list = list(participating_arms_by_subtask[st.name])
        arms_dict: dict[str, Any] = {}
        for arm_name in arms_list:
            key = (st.name, arm_name)
            starts = subtask_arm_start_poses[key]
            ends = subtask_arm_end_poses[key]
            trajs = subtask_arm_traj_xyz[key]
            start_cov = compute_endpoint_coverage(starts, task_spec.coverage.xy_bin_m)
            end_cov = compute_endpoint_coverage(ends, task_spec.coverage.xy_bin_m)
            dispersion_cm = compute_trajectory_dispersion_cm(trajs)
            arms_dict[arm_name] = {
                "metrics": {
                    "episode_count": len(starts),
                    "start_coverage": {
                        "occupied_xy_cells": start_cov.occupied_xy_cells,
                        "yaw_span_deg": start_cov.yaw_span_deg,
                    },
                    "end_coverage": {
                        "occupied_xy_cells": end_cov.occupied_xy_cells,
                        "yaw_span_deg": end_cov.yaw_span_deg,
                    },
                    "trajectory_dispersion_cm": dispersion_cm,
                },
                "episodes": subtask_arm_episodes[key],
            }
        subtasks_payload.append(
            {
                "name": st.name,
                "prompt": st.prompt,
                "participating_arms": arms_list,
                "arms": arms_dict,
            }
        )

    provenance_payload: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "task_name": task_spec.name,
        "annotations_path": str(annotations_path) if annotations_path is not None else None,
        "task_spec_path": str(task_spec_path) if task_spec_path is not None else None,
        "position_point": position_point,
        "calib_path": str(calib_path) if calib_path is not None else None,
        "calib_mtime": calib_mtime,
        "calibrated_arms": sorted(normalized_offsets.keys()),
        "fps": fps,
        "annotator_counts": {
            "agent": provenance.agent,
            "human": provenance.human,
            "converted": provenance.converted,
            "total": provenance.total,
        },
        "dataset_total_episodes": dataset_total_episodes,
        "episode_filter": selected_eps,
        "annotated_episodes": list(frame_shares.annotated_episodes),
        "skipped_episodes": list(frame_shares.skipped_episodes),
    }
    if has_object_layer:
        if not object_poses_sha256:
            raise ValueError("object_poses_sha256 is required when object_layer is attached")
        provenance_payload["object_poses_sha256"] = object_poses_sha256

    return {
        "version": DISTRIBUTION_SCHEMA_VERSION,
        "provenance": provenance_payload,
        "coverage_config": {
            "xy_bin_m": task_spec.coverage.xy_bin_m,
            "yaw_bin_deg": task_spec.coverage.yaw_bin_deg,
            "sparse_threshold": task_spec.coverage.sparse_threshold,
        },
        "trajectory_samples": trajectory_samples,
        "frame_shares": {
            "total_frames": frame_shares.total_frames,
            "subtasks": [
                {
                    "subtask": item.subtask,
                    "episodes": item.episodes,
                    "frames": item.frames,
                    "share": item.share,
                }
                for item in frame_shares.subtasks
            ],
        },
        "episodes": episode_records,
        "objects": objects_payload,
        "subtasks": subtasks_payload,
    }


def compute_distribution(
    dataset: DatasetView,
    rows: Sequence[AnnotationRow],
    task_spec: TaskSpec,
    *,
    annotations_path: Path | str | None = None,
    task_spec_path: Path | str | None = None,
    episode_filter: Collection[int] | None = None,
    trajectory_samples: int = DEFAULT_TRAJECTORY_SAMPLES,
    calibration: HeadCalibration | None = None,
    calib_path: Path | str | None = None,
    calib_mtime: str | None = None,
    calib_sha256: str | None = None,
    object_poses: Mapping[str, Any] | None = None,
    object_poses_sha256: str | None = None,
) -> dict[str, Any]:
    """Compute the ``distribution.json`` payload directly from a ``DatasetView``.

    Loads each annotated (and selected) episode's parquet state once.

    Args:
        dataset: Open ``DatasetView``.
        rows: Validated annotation rows for ``dataset``.
        task_spec: Task specification.
        annotations_path: Source annotation CSV path for provenance.
        task_spec_path: Source task spec YAML path for provenance.
        episode_filter: Optional collection of episode indices to restrict to.
        trajectory_samples: Number of evenly spaced progress points per trajectory.
        calibration: Optional ``HeadCalibration`` providing per-arm ``tool_offset``
            vectors so positions use ``p_tcp + R_tcp @ tool_offset``.
        calib_path: Source ``head.json`` path recorded in provenance.
        calib_mtime: UTC ISO-8601 modification timestamp of ``head.json``.

    Returns:
        JSON-serializable ``distribution.json`` v1 dictionary.
    """
    filter_set = set(episode_filter) if episode_filter is not None else None
    target_episodes = sorted(
        {row.episode_index for row in rows if filter_set is None or row.episode_index in filter_set}
    )
    arm_states_by_episode: dict[int, dict[str, ArmState]] = {}
    for ep_idx in target_episodes:
        arm_states_by_episode[ep_idx] = {
            arm.arm: arm for arm in dataset.load_episode_arm_states(ep_idx)
        }

    tool_offsets = (
        {arm: cal.tool_offset for arm, cal in calibration.arms.items()}
        if calibration is not None
        else None
    )

    return compute_distribution_from_states(
        repo_id=dataset.repo_id,
        revision=dataset.revision,
        fps=dataset.fps,
        episode_lengths=dataset.episode_lengths,
        arm_states_by_episode=arm_states_by_episode,
        rows=rows,
        task_spec=task_spec,
        annotations_path=annotations_path,
        task_spec_path=task_spec_path,
        episode_filter=episode_filter,
        trajectory_samples=trajectory_samples,
        tool_offsets=tool_offsets,
        calib_path=calib_path,
        calib_mtime=calib_mtime,
        calib_sha256=calib_sha256,
        object_poses=object_poses,
        object_poses_sha256=object_poses_sha256,
    )


def write_distribution_json(payload: Mapping[str, Any], path: Path) -> Path:
    """Write a ``distribution.json`` payload cleanly with UTF-8 indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_distribution_json(path: Path) -> dict[str, Any]:
    """Load and lightly validate a ``distribution.json`` file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != DISTRIBUTION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported distribution JSON at {path} "
            f"(expected version={DISTRIBUTION_SCHEMA_VERSION})"
        )
    return data


def format_distribution_summary(payload: Mapping[str, Any]) -> str:
    """Format a concise CLI summary table of frame shares and subtask-arm metrics."""
    prov = payload["provenance"]
    counts = prov["annotator_counts"]
    annotated = prov["annotated_episodes"]
    skipped = prov["skipped_episodes"]
    lines = [
        f"Dataset: {prov['repo_id']} @ {prov['revision']} (fps={prov['fps']:g})",
        f"Position point: {prov.get('position_point', 'tcp')}",
        (
            f"Episodes: {len(annotated)} annotated, {len(skipped)} skipped "
            f"(dataset total: {prov['dataset_total_episodes']})"
        ),
        (
            f"Annotator rows: agent={counts['agent']}, human={counts['human']}, "
            f"converted={counts['converted']} (total={counts['total']})"
        ),
        "",
        "Subtask Frame Shares:",
        "  subtask\tepisodes\tframes\tshare",
    ]
    for item in payload["frame_shares"]["subtasks"]:
        lines.append(
            f"  {item['subtask']}\t{item['episodes']}\t{item['frames']}\t{item['share'] * 100:.2f}%"
        )
    lines.append(f"  TOTAL\t{len(annotated)}\t{payload['frame_shares']['total_frames']}\t100.00%")
    lines.extend(
        [
            "",
            "Subtask x Participating Arm Metrics:",
            (
                "  subtask\tarm\teps\tstart_cells\tstart_yaw_span\t"
                "end_cells\tend_yaw_span\tdispersion_cm"
            ),
        ]
    )
    for st in payload["subtasks"]:
        for arm_name, arm_data in st["arms"].items():
            m = arm_data["metrics"]
            disp = m["trajectory_dispersion_cm"]
            disp_str = "n/a" if disp is None else f"{disp:.2f}"
            lines.append(
                f"  {st['name']}\t{arm_name}\t{m['episode_count']}\t"
                f"{m['start_coverage']['occupied_xy_cells']}\t"
                f"{m['start_coverage']['yaw_span_deg']:.1f} deg\t"
                f"{m['end_coverage']['occupied_xy_cells']}\t"
                f"{m['end_coverage']['yaw_span_deg']:.1f} deg\t"
                f"{disp_str}"
            )
    return "\n".join(lines)
