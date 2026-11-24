"""Core analysis computations for subtask frame shares, speed, poses, trajectories, and coverage."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from lerobot_check.dataset import ArmState, DatasetView, rotation_6d_to_matrix
from lerobot_check.propose import compute_tcp_linear_speed
from lerobot_check.schema import AnnotationRow, CoverageSpec, ObjectSpec, TaskSpec

DEFAULT_SPEED_RATIO_THRESHOLD = 0.8
DEFAULT_MIN_REST_SPEED_MPS = 1e-3
DEFAULT_MAX_TRAJ_POINTS = 60


@dataclass(frozen=True)
class SubtaskFrameShare:
    """Frame and fine-window statistics for a single subtask.

    Attributes:
        subtask: Subtask name from the task specification.
        episodes: Number of annotated episodes containing this subtask.
        frames: Total annotated frames belonging to this subtask.
        frame_share: Fraction of total annotated frames in ``[0, 1]``.
        fine_frames: Total fine-window frames within this subtask.
        fine_share_of_subtask: Fraction ``fine_frames / frames`` in ``[0, 1]``.
        fine_share_of_total: Fraction ``fine_frames / total_frames`` in ``[0, 1]``.
    """

    subtask: str
    episodes: int
    frames: int
    frame_share: float
    fine_frames: int
    fine_share_of_subtask: float
    fine_share_of_total: float


@dataclass(frozen=True)
class FrameShareSummary:
    """Dataset-level or combined frame-share breakdown across subtasks.

    Attributes:
        label: Human-readable dataset or mix label.
        annotated_episodes: Number of annotated episodes included.
        total_episodes: Total episodes in dataset metadata (sum across inputs for mixes).
        skipped_episodes: Sorted episode indices omitted from the annotation table.
        total_frames: Total annotated frames across all subtasks.
        total_fine_frames: Total fine-window frames across all subtasks.
        total_fine_share: Fraction ``total_fine_frames / total_frames``.
        subtasks: Per-subtask breakdown ordered according to ``task_spec.subtasks``.
    """

    label: str
    annotated_episodes: int
    total_episodes: int
    skipped_episodes: tuple[int, ...]
    total_frames: int
    total_fine_frames: int
    total_fine_share: float
    subtasks: tuple[SubtaskFrameShare, ...]


@dataclass(frozen=True)
class MixEntry:
    """One weighted input dataset for ``compute_mix``.

    Attributes:
        label: Human-readable dataset label.
        rows: Validated annotation rows for this dataset.
        fps: Dataset frame rate in Hz.
        total_episodes: Total episode count in dataset metadata.
        episode_lengths: Mapping from episode index to frame count.
        repeat: Positive integer repetition multiplier.
    """

    label: str
    rows: Sequence[AnnotationRow]
    fps: float
    total_episodes: int
    episode_lengths: Mapping[int, int]
    repeat: int = 1


@dataclass(frozen=True)
class EpisodeSpeedMetric:
    """Per-episode, per-arm TCP speed comparison between fine and non-fine windows.

    Attributes:
        dataset_label: Label of the source dataset.
        episode_index: Zero-based episode index.
        arm: Robot arm name (e.g. ``"left"`` or ``"right"``).
        fine_frames: Number of fine-window frames in the episode.
        rest_frames: Number of non-fine frames in the episode.
        fine_median_mps: Median TCP speed inside fine-window frames (m/s), or ``None`` if empty.
        rest_median_mps: Median TCP speed outside fine-window frames (m/s), or ``None`` if empty.
        ratio: ``fine_median_mps / rest_median_mps``, or ``None`` when the arm is idle
            (``rest_median_mps < min_rest_speed_mps``) or either mask is empty.
        not_slowed_down: ``True`` when ``ratio is not None`` and ``ratio > threshold``.
        idle: ``True`` when ``rest_median_mps is not None`` and
            ``rest_median_mps < min_rest_speed_mps``.
    """

    dataset_label: str
    episode_index: int
    arm: str
    fine_frames: int
    rest_frames: int
    fine_median_mps: float | None
    rest_median_mps: float | None
    ratio: float | None
    not_slowed_down: bool
    idle: bool


@dataclass(frozen=True)
class ObjectPoseSample:
    """Extracted event-frame TCP pose for one object in one episode.

    Attributes:
        dataset_label: Source dataset label.
        episode_index: Zero-based episode index.
        object_name: Object name from ``task_spec.objects``.
        subtask: Subtask name where the pose was sampled.
        event: Event type (``"grasp"`` or ``"release"``).
        arm: Robot arm whose base frame the pose is expressed in.
        frame: Frame index used for sampling.
        x: TCP x position in meters (arm base frame).
        y: TCP y position in meters (arm base frame).
        z: TCP z position in meters (arm base frame).
        raw_yaw_deg: Raw TCP yaw angle in ``(-180, 180]`` degrees.
        folded_yaw_deg: Symmetry-folded yaw angle in ``[0, yaw_symmetry_deg)`` degrees.
    """

    dataset_label: str
    episode_index: int
    object_name: str
    subtask: str
    event: str
    arm: str
    frame: int
    x: float
    y: float
    z: float
    raw_yaw_deg: float
    folded_yaw_deg: float


@dataclass(frozen=True)
class SkippedObjectEvent:
    """Record of an object whose event pose could not be extracted for an episode.

    Attributes:
        dataset_label: Source dataset label.
        episode_index: Zero-based episode index.
        object_name: Object name from ``task_spec.objects``.
        subtask: Subtask referenced by the object spec.
        event: Event selector (``"grasp"`` or ``"release"``).
        arm: Arm referenced by the object spec.
        reason: Human-readable reason why the sample was skipped.
    """

    dataset_label: str
    episode_index: int
    object_name: str
    subtask: str
    event: str
    arm: str
    reason: str


@dataclass(frozen=True)
class PairwiseRelativeOffset:
    """Same-arm relative displacement ``pos(object_b) - pos(object_a)`` within one episode.

    Attributes:
        dataset_label: Source dataset label.
        episode_index: Zero-based episode index.
        object_a: Reference object name.
        object_b: Target object name.
        arm: Shared arm base frame.
        dx: ``x_b - x_a`` in meters.
        dy: ``y_b - y_a`` in meters.
        dz: ``z_b - z_a`` in meters.
        dist_xy: Planar Euclidean distance ``hypot(dx, dy)`` in meters.
    """

    dataset_label: str
    episode_index: int
    object_a: str
    object_b: str
    arm: str
    dx: float
    dy: float
    dz: float
    dist_xy: float


@dataclass(frozen=True)
class TrajectorySegment:
    """Downsampled 3D TCP trajectory for one episode's subtask on one arm.

    Attributes:
        dataset_label: Source dataset label.
        episode_index: Zero-based episode index.
        subtask: Subtask name.
        arm: Robot arm name.
        xyz: Float64 array of shape ``(K, 3)`` in the arm's base frame.
        t_norm: Float64 array of shape ``(K,)`` with normalized time in ``[0, 1]``.
        grasp_relative_xyz: Optional float64 array of shape ``(M, 3)`` covering
            ``[start, grasp]`` shifted so the grasp-frame TCP sits at ``(0, 0, 0)``.
        grasp_relative_t_norm: Optional float64 array of shape ``(M,)`` in ``[0, 1]``.
    """

    dataset_label: str
    episode_index: int
    subtask: str
    arm: str
    xyz: np.ndarray
    t_norm: np.ndarray
    grasp_relative_xyz: np.ndarray | None
    grasp_relative_t_norm: np.ndarray | None


@dataclass(frozen=True)
class CoverageBin:
    """Single ``(ix, iy, yaw_bin)`` cell in an object's workspace coverage grid.

    Attributes:
        ix: Integer X grid index (``floor(x / xy_bin_m)``).
        iy: Integer Y grid index (``floor(y / xy_bin_m)``).
        yaw_bin: Zero-based yaw bin index inside ``[0, num_yaw_bins)``.
        x_min: Inclusive lower X bound in meters.
        x_max: Exclusive upper X bound in meters.
        y_min: Inclusive lower Y bound in meters.
        y_max: Exclusive upper Y bound in meters.
        yaw_min_deg: Inclusive lower folded-yaw bound in degrees.
        yaw_max_deg: Exclusive upper folded-yaw bound in degrees.
        count: Number of observed object pose samples falling in this cell.
    """

    ix: int
    iy: int
    yaw_bin: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    yaw_min_deg: float
    yaw_max_deg: float
    count: int


@dataclass(frozen=True)
class ObjectCoverage:
    """Workspace XY x yaw coverage grid for one object across all input datasets.

    Attributes:
        spec: ``ObjectSpec`` from the task spec.
        coverage_spec: ``CoverageSpec`` binning parameters.
        sample_count: Number of extracted samples contributing to this grid.
        ix_range: Inclusive ``(ix_min, ix_max)`` bounds, or ``None`` if empty.
        iy_range: Inclusive ``(iy_min, iy_max)`` bounds, or ``None`` if empty.
        num_yaw_bins: Number of yaw bins covering ``[0, yaw_symmetry_deg)``.
        bins: All grid cells across ``ix_range x iy_range x [0..num_yaw_bins-1]``.
        sparse_bins: Subset of ``bins`` with ``count < coverage_spec.sparse_threshold``.
    """

    spec: ObjectSpec
    coverage_spec: CoverageSpec
    sample_count: int
    ix_range: tuple[int, int] | None
    iy_range: tuple[int, int] | None
    num_yaw_bins: int
    bins: tuple[CoverageBin, ...]
    sparse_bins: tuple[CoverageBin, ...]


@dataclass(frozen=True)
class PlacementSuggestion:
    """One suggested placement pose sampled from a sparse coverage cell.

    Attributes:
        batch_index: 1-based episode slot index in the next collection batch.
        object_name: Object name from ``task_spec.objects``.
        arm: Arm base frame in which ``(x_m, y_m, yaw_deg)`` are expressed.
        x_m: Suggested X coordinate in meters.
        y_m: Suggested Y coordinate in meters.
        yaw_deg: Suggested folded yaw angle in degrees.
        ix: Source grid cell X index.
        iy: Source grid cell Y index.
        yaw_bin: Source grid cell yaw bin index.
        current_count: Current sample count in the selected grid cell.
        from_sparse_bin: ``True`` if drawn from a sparse bin; ``False`` if all bins
            already met ``sparse_threshold`` and lowest-count bins were used.
    """

    batch_index: int
    object_name: str
    arm: str
    x_m: float
    y_m: float
    yaw_deg: float
    ix: int
    iy: int
    yaw_bin: int
    current_count: int
    from_sparse_bin: bool


@dataclass(frozen=True)
class DatasetAnalysis:
    """Complete analysis output for a single ``(dataset, annotations)`` input.

    Attributes:
        label: Human-readable dataset label used in legends and tables.
        repo_id: Dataset repository ID.
        revision: Dataset revision string.
        fps: Dataset frame rate in Hz.
        arms: Ordered tuple of arm identifiers present in the dataset.
        frame_shares: Per-subtask and fine-window frame share summary.
        speed_metrics: Per-episode, per-arm speed check results.
        object_poses: Extracted object initial pose samples.
        skipped_objects: Object events skipped due to missing frame or absent subtask.
        pairwise_offsets: Same-arm pairwise relative object displacements.
        trajectories: Downsampled per-subtask, per-arm 3D trajectory segments.
    """

    label: str
    repo_id: str
    revision: str
    fps: float
    arms: tuple[str, ...]
    frame_shares: FrameShareSummary
    speed_metrics: tuple[EpisodeSpeedMetric, ...]
    object_poses: tuple[ObjectPoseSample, ...]
    skipped_objects: tuple[SkippedObjectEvent, ...]
    pairwise_offsets: tuple[PairwiseRelativeOffset, ...]
    trajectories: tuple[TrajectorySegment, ...]


def fine_window_intervals(row: AnnotationRow, delta_frames: int) -> tuple[tuple[int, int], ...]:
    """Compute disjoint half-open fine-window intervals ``[a, b)`` for one annotation row.

    Implements the contract formula:
    ``[max(start, contact - delta), grasp] U [max(start, end - delta), end)``,
    using only the release segment ``[max(start, end - delta), end)`` when either
    ``contact`` or ``grasp`` is missing.

    Args:
        row: Subtask annotation row with half-open bounds ``[start, end)``.
        delta_frames: Fine-window duration ``round(fine_window_s * fps)`` in frames.

    Returns:
        Sorted tuple of merged, non-empty half-open intervals ``(start, end)``
        contained inside ``[row.start, row.end)``.

    Raises:
        ValueError: If ``delta_frames < 0``.
    """
    if delta_frames < 0:
        raise ValueError(f"delta_frames must be non-negative, got {delta_frames}")

    raw_intervals: list[tuple[int, int]] = []
    if row.contact is not None and row.grasp is not None:
        seg1_start = max(row.start, row.contact - delta_frames)
        # Contract defines [max(start, contact - delta), grasp] as right-closed on grasp.
        seg1_end = min(row.end, row.grasp + 1)
        if seg1_start < seg1_end:
            raw_intervals.append((seg1_start, seg1_end))

    seg2_start = max(row.start, row.end - delta_frames)
    seg2_end = row.end
    if seg2_start < seg2_end:
        raw_intervals.append((seg2_start, seg2_end))

    if not raw_intervals:
        return ()

    raw_intervals.sort()
    merged: list[tuple[int, int]] = [raw_intervals[0]]
    for start, end in raw_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def fine_window_frame_count(row: AnnotationRow, delta_frames: int) -> int:
    """Return total number of unique frames inside ``row``'s fine window."""
    return sum(end - start for start, end in fine_window_intervals(row, delta_frames))


def episode_fine_mask(
    ep_rows: Sequence[AnnotationRow],
    episode_length: int,
    delta_frames: int,
) -> np.ndarray:
    """Build a boolean mask of shape ``(episode_length,)`` marking fine-window frames.

    Args:
        ep_rows: Annotation rows for a single episode.
        episode_length: Total frame count of the episode.
        delta_frames: Fine-window duration in frames.

    Returns:
        Boolean NumPy array where ``True`` indicates a fine-window frame.
    """
    mask = np.zeros(episode_length, dtype=bool)
    for row in ep_rows:
        for start, end in fine_window_intervals(row, delta_frames):
            mask[start:end] = True
    return mask


def compute_frame_shares(
    rows: Sequence[AnnotationRow],
    task_spec: TaskSpec,
    *,
    fps: float,
    episode_lengths: Mapping[int, int],
    label: str = "dataset",
) -> FrameShareSummary:
    """Compute per-subtask and fine-window frame counts and shares.

    Args:
        rows: Validated annotation rows for one dataset.
        task_spec: Task specification providing subtask order and ``fine_window_s``.
        fps: Dataset frame rate in Hz.
        episode_lengths: Mapping from dataset ``episode_index`` to frame count.
        label: Label attached to the returned summary.

    Returns:
        ``FrameShareSummary`` with per-subtask and total frame/fine-window counts.
    """
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    delta_frames = round(task_spec.fine_window_s * fps)
    subtask_frames: dict[str, int] = {s.name: 0 for s in task_spec.subtasks}
    subtask_fine_frames: dict[str, int] = {s.name: 0 for s in task_spec.subtasks}
    subtask_episodes: dict[str, set[int]] = {s.name: set() for s in task_spec.subtasks}
    annotated_episodes: set[int] = set()

    for row in rows:
        annotated_episodes.add(row.episode_index)
        length = row.end - row.start
        fine_count = fine_window_frame_count(row, delta_frames)
        subtask_frames[row.subtask] += length
        subtask_fine_frames[row.subtask] += fine_count
        subtask_episodes[row.subtask].add(row.episode_index)

    total_frames = sum(subtask_frames.values())
    total_fine_frames = sum(subtask_fine_frames.values())
    skipped = tuple(sorted(set(episode_lengths) - annotated_episodes))

    subtask_shares: list[SubtaskFrameShare] = []
    for subtask_spec in task_spec.subtasks:
        name = subtask_spec.name
        frames = subtask_frames[name]
        fine_frames = subtask_fine_frames[name]
        subtask_shares.append(
            SubtaskFrameShare(
                subtask=name,
                episodes=len(subtask_episodes[name]),
                frames=frames,
                frame_share=(frames / total_frames) if total_frames > 0 else 0.0,
                fine_frames=fine_frames,
                fine_share_of_subtask=(fine_frames / frames) if frames > 0 else 0.0,
                fine_share_of_total=(fine_frames / total_frames) if total_frames > 0 else 0.0,
            )
        )

    return FrameShareSummary(
        label=label,
        annotated_episodes=len(annotated_episodes),
        total_episodes=len(episode_lengths),
        skipped_episodes=skipped,
        total_frames=total_frames,
        total_fine_frames=total_fine_frames,
        total_fine_share=(total_fine_frames / total_frames) if total_frames > 0 else 0.0,
        subtasks=tuple(subtask_shares),
    )


def compute_mix(
    entries: Sequence[MixEntry],
    task_spec: TaskSpec,
    *,
    label: str = "mixed",
) -> tuple[list[tuple[FrameShareSummary, int]], FrameShareSummary]:
    """Compute weighted combined frame and fine-window shares across repeated datasets.

    Args:
        entries: Sequence of ``MixEntry`` inputs with positive integer repeat counts.
        task_spec: Task specification shared by all inputs.
        label: Label assigned to the combined ``FrameShareSummary``.

    Returns:
        Tuple ``(per_input_summaries, combined_summary)`` where each element of
        ``per_input_summaries`` is ``(summary, repeat)``.

    Raises:
        ValueError: If ``entries`` is empty or any entry has ``repeat < 1``.
    """
    if not entries:
        raise ValueError("compute_mix requires at least one MixEntry")

    per_input: list[tuple[FrameShareSummary, int]] = []
    combined_frames: dict[str, int] = {s.name: 0 for s in task_spec.subtasks}
    combined_fine_frames: dict[str, int] = {s.name: 0 for s in task_spec.subtasks}
    combined_subtask_eps: dict[str, int] = {s.name: 0 for s in task_spec.subtasks}
    total_annotated_eps = 0
    total_metadata_eps = 0

    for entry in entries:
        if entry.repeat < 1:
            raise ValueError(
                f"Mix repeat count for {entry.label!r} must be >= 1, got {entry.repeat}"
            )
        summary = compute_frame_shares(
            entry.rows,
            task_spec,
            fps=entry.fps,
            episode_lengths=entry.episode_lengths,
            label=entry.label,
        )
        per_input.append((summary, entry.repeat))
        total_annotated_eps += summary.annotated_episodes * entry.repeat
        total_metadata_eps += summary.total_episodes * entry.repeat
        for st in summary.subtasks:
            combined_frames[st.subtask] += st.frames * entry.repeat
            combined_fine_frames[st.subtask] += st.fine_frames * entry.repeat
            combined_subtask_eps[st.subtask] += st.episodes * entry.repeat

    total_frames = sum(combined_frames.values())
    total_fine_frames = sum(combined_fine_frames.values())
    combined_subtasks = tuple(
        SubtaskFrameShare(
            subtask=s.name,
            episodes=combined_subtask_eps[s.name],
            frames=combined_frames[s.name],
            frame_share=(combined_frames[s.name] / total_frames) if total_frames > 0 else 0.0,
            fine_frames=combined_fine_frames[s.name],
            fine_share_of_subtask=(
                (combined_fine_frames[s.name] / combined_frames[s.name])
                if combined_frames[s.name] > 0
                else 0.0
            ),
            fine_share_of_total=(
                (combined_fine_frames[s.name] / total_frames) if total_frames > 0 else 0.0
            ),
        )
        for s in task_spec.subtasks
    )
    combined = FrameShareSummary(
        label=label,
        annotated_episodes=total_annotated_eps,
        total_episodes=total_metadata_eps,
        skipped_episodes=(),
        total_frames=total_frames,
        total_fine_frames=total_fine_frames,
        total_fine_share=(total_fine_frames / total_frames) if total_frames > 0 else 0.0,
        subtasks=combined_subtasks,
    )
    return per_input, combined


def compute_episode_speed_metrics(
    *,
    dataset_label: str,
    episode_index: int,
    ep_rows: Sequence[AnnotationRow],
    arms: Sequence[ArmState],
    fps: float,
    fine_window_s: float,
    speed_ratio_threshold: float = DEFAULT_SPEED_RATIO_THRESHOLD,
    min_rest_speed_mps: float = DEFAULT_MIN_REST_SPEED_MPS,
) -> list[EpisodeSpeedMetric]:
    """Compare median TCP linear speed in fine vs non-fine frames for each arm.

    Arms whose non-fine median speed falls below ``min_rest_speed_mps`` are marked
    ``idle=True`` with ``ratio=None`` so stationary support arms are not falsely
    flagged as failing to slow down.

    Args:
        dataset_label: Source dataset label.
        episode_index: Zero-based episode index.
        ep_rows: Annotation rows covering this episode.
        arms: Per-arm state trajectories for this episode.
        fps: Dataset frame rate in Hz.
        fine_window_s: Fine window half-width in seconds.
        speed_ratio_threshold: Maximum acceptable ``fine_median / rest_median`` ratio.
        min_rest_speed_mps: Minimum non-fine median speed (m/s) for an arm to be
            considered active rather than idle.

    Returns:
        List of ``EpisodeSpeedMetric`` records, one per arm.
    """
    if speed_ratio_threshold <= 0.0:
        raise ValueError(f"speed_ratio_threshold must be positive, got {speed_ratio_threshold}")
    if min_rest_speed_mps < 0.0:
        raise ValueError(f"min_rest_speed_mps must be non-negative, got {min_rest_speed_mps}")

    delta_frames = round(fine_window_s * fps)
    metrics: list[EpisodeSpeedMetric] = []
    for arm_state in arms:
        length = len(arm_state.tcp_xyz)
        fine_mask = episode_fine_mask(ep_rows, length, delta_frames)
        rest_mask = ~fine_mask
        speeds = compute_tcp_linear_speed(arm_state.tcp_xyz, fps)

        fine_count = int(np.count_nonzero(fine_mask))
        rest_count = int(np.count_nonzero(rest_mask))
        fine_med = float(np.median(speeds[fine_mask])) if fine_count > 0 else None
        rest_med = float(np.median(speeds[rest_mask])) if rest_count > 0 else None

        idle = rest_med is not None and rest_med < min_rest_speed_mps
        ratio: float | None = None
        if fine_med is not None and rest_med is not None and not idle and rest_med > 0.0:
            ratio = fine_med / rest_med

        not_slowed_down = ratio is not None and ratio > speed_ratio_threshold
        metrics.append(
            EpisodeSpeedMetric(
                dataset_label=dataset_label,
                episode_index=episode_index,
                arm=arm_state.arm,
                fine_frames=fine_count,
                rest_frames=rest_count,
                fine_median_mps=fine_med,
                rest_median_mps=rest_med,
                ratio=ratio,
                not_slowed_down=not_slowed_down,
                idle=idle,
            )
        )
    return metrics


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
    return math.degrees(yaw_rad)


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


def extract_episode_object_poses(
    *,
    dataset_label: str,
    episode_index: int,
    ep_rows: Sequence[AnnotationRow],
    arms_by_name: Mapping[str, ArmState],
    objects: Sequence[ObjectSpec],
) -> tuple[list[ObjectPoseSample], list[SkippedObjectEvent]]:
    """Extract object initial poses at event frames for one episode.

    Objects whose subtask is absent from the episode or whose ``grasp`` frame is
    missing are returned in ``skipped`` rather than silently dropped.

    Args:
        dataset_label: Source dataset label.
        episode_index: Zero-based episode index.
        ep_rows: Annotation rows for this episode.
        arms_by_name: Mapping from arm name to ``ArmState`` for this episode.
        objects: Object specifications from ``task_spec.objects``.

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
                    dataset_label=dataset_label,
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
                    dataset_label=dataset_label,
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
                    dataset_label=dataset_label,
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
        xyz = arm_state.tcp_xyz[frame]
        rot = rotation_6d_to_matrix(arm_state.tcp_rot6d[frame])
        raw_yaw = rotation_matrix_to_yaw_deg(rot)
        folded_yaw = fold_yaw_deg(raw_yaw, obj.yaw_symmetry_deg)

        samples.append(
            ObjectPoseSample(
                dataset_label=dataset_label,
                episode_index=episode_index,
                object_name=obj.name,
                subtask=obj.subtask,
                event=obj.event,
                arm=obj.arm,
                frame=frame,
                x=float(xyz[0]),
                y=float(xyz[1]),
                z=float(xyz[2]),
                raw_yaw_deg=raw_yaw,
                folded_yaw_deg=folded_yaw,
            )
        )

    return samples, skipped


def compute_pairwise_offsets(
    samples: Sequence[ObjectPoseSample],
    objects: Sequence[ObjectSpec],
) -> list[PairwiseRelativeOffset]:
    """Compute within-episode relative displacements for same-arm object pairs.

    Pairs on different arms are skipped because each arm uses its own base frame
    with no inter-arm extrinsics.

    Args:
        samples: Extracted object pose samples.
        objects: Ordered object specs from ``task_spec.objects``.

    Returns:
        List of ``PairwiseRelativeOffset`` records ordered by pair then episode.
    """
    same_arm_pairs = [
        (obj_a, obj_b) for obj_a, obj_b in combinations(objects, 2) if obj_a.arm == obj_b.arm
    ]
    if not same_arm_pairs:
        return []

    by_key: dict[tuple[str, int, str], ObjectPoseSample] = {
        (s.dataset_label, s.episode_index, s.object_name): s for s in samples
    }
    episodes = sorted({(s.dataset_label, s.episode_index) for s in samples})

    offsets: list[PairwiseRelativeOffset] = []
    for obj_a, obj_b in same_arm_pairs:
        for ds_label, ep_idx in episodes:
            sample_a = by_key.get((ds_label, ep_idx, obj_a.name))
            sample_b = by_key.get((ds_label, ep_idx, obj_b.name))
            if sample_a is None or sample_b is None:
                continue
            dx = sample_b.x - sample_a.x
            dy = sample_b.y - sample_a.y
            dz = sample_b.z - sample_a.z
            offsets.append(
                PairwiseRelativeOffset(
                    dataset_label=ds_label,
                    episode_index=ep_idx,
                    object_a=obj_a.name,
                    object_b=obj_b.name,
                    arm=obj_a.arm,
                    dx=dx,
                    dy=dy,
                    dz=dz,
                    dist_xy=math.hypot(dx, dy),
                )
            )
    return offsets


def _downsample_segment(
    xyz: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly subsample a trajectory segment to at most ``max_points`` points."""
    n = len(xyz)
    if n == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)
    if n == 1:
        return xyz.copy(), np.array([0.0], dtype=np.float64)

    if n <= max_points:
        indices = np.arange(n, dtype=np.int64)
    else:
        indices = np.unique(np.round(np.linspace(0, n - 1, max_points)).astype(np.int64))
    t_norm = indices.astype(np.float64) / float(n - 1)
    return xyz[indices].copy(), t_norm


def extract_episode_trajectories(
    *,
    dataset_label: str,
    episode_index: int,
    ep_rows: Sequence[AnnotationRow],
    arms: Sequence[ArmState],
    max_points: int = DEFAULT_MAX_TRAJ_POINTS,
) -> list[TrajectorySegment]:
    """Extract downsampled base-frame and grasp-relative 3D TCP trajectories.

    Args:
        dataset_label: Source dataset label.
        episode_index: Zero-based episode index.
        ep_rows: Annotation rows for this episode.
        arms: Per-arm state trajectories for this episode.
        max_points: Maximum sampled points per trajectory segment.

    Returns:
        List of ``TrajectorySegment`` records for each ``(row, arm)`` pair.
    """
    if max_points < 2:
        raise ValueError(f"max_points must be >= 2, got {max_points}")

    segments: list[TrajectorySegment] = []
    for row in ep_rows:
        for arm_state in arms:
            raw_xyz = arm_state.tcp_xyz[row.start : row.end]
            xyz, t_norm = _downsample_segment(raw_xyz, max_points)

            grasp_rel_xyz: np.ndarray | None = None
            grasp_rel_t: np.ndarray | None = None
            if row.grasp is not None:
                origin = arm_state.tcp_xyz[row.grasp]
                approach_raw = arm_state.tcp_xyz[row.start : row.grasp + 1] - origin
                grasp_rel_xyz, grasp_rel_t = _downsample_segment(approach_raw, max_points)

            segments.append(
                TrajectorySegment(
                    dataset_label=dataset_label,
                    episode_index=episode_index,
                    subtask=row.subtask,
                    arm=arm_state.arm,
                    xyz=xyz,
                    t_norm=t_norm,
                    grasp_relative_xyz=grasp_rel_xyz,
                    grasp_relative_t_norm=grasp_rel_t,
                )
            )
    return segments


def compute_object_coverage(
    obj: ObjectSpec,
    samples: Sequence[ObjectPoseSample],
    coverage_spec: CoverageSpec,
) -> ObjectCoverage:
    """Build an XY x yaw workspace coverage grid and identify sparse cells.

    XY cell indices use canonical ``floor(coord / xy_bin_m)`` coordinates so grid
    alignment stays stable across datasets. Every cell inside the observed
    ``[ix_min, ix_max] x [iy_min, iy_max] x [0, num_yaw_bins)`` bounding volume
    is enumerated, and cells with ``count < coverage_spec.sparse_threshold`` are
    reported in ``sparse_bins``.

    Args:
        obj: Target object specification.
        samples: Pose samples (filtered internally to ``obj.name``).
        coverage_spec: Grid cell size and sparsity threshold.

    Returns:
        ``ObjectCoverage`` with all cells and sparse cells sorted deterministically.
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
            sparse_bins=(),
        )

    xy_step = coverage_spec.xy_bin_m
    yaw_step = coverage_spec.yaw_bin_deg
    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    ix_vals: list[int] = []
    iy_vals: list[int] = []

    for sample in obj_samples:
        ix = math.floor(sample.x / xy_step)
        iy = math.floor(sample.y / xy_step)
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
                        x_min=ix * xy_step,
                        x_max=(ix + 1) * xy_step,
                        y_min=iy * xy_step,
                        y_max=(iy + 1) * xy_step,
                        yaw_min_deg=yaw_lo,
                        yaw_max_deg=yaw_hi,
                        count=counts.get((ix, iy, y_bin), 0),
                    )
                )

    sparse = [b for b in all_bins if b.count < coverage_spec.sparse_threshold]
    sparse.sort(key=lambda b: (b.count, b.ix, b.iy, b.yaw_bin))

    return ObjectCoverage(
        spec=obj,
        coverage_spec=coverage_spec,
        sample_count=len(obj_samples),
        ix_range=(ix_min, ix_max),
        iy_range=(iy_min, iy_max),
        num_yaw_bins=num_yaw_bins,
        bins=tuple(all_bins),
        sparse_bins=tuple(sparse),
    )


def compute_all_coverages(
    samples: Sequence[ObjectPoseSample],
    task_spec: TaskSpec,
) -> list[ObjectCoverage]:
    """Compute ``ObjectCoverage`` for every object defined in ``task_spec``."""
    return [compute_object_coverage(obj, samples, task_spec.coverage) for obj in task_spec.objects]


def suggest_placements(
    coverages: Sequence[ObjectCoverage],
    count: int,
    *,
    seed: int = 0,
) -> tuple[list[PlacementSuggestion], list[str]]:
    """Sample ``count`` next-batch placement poses independently per object from sparse bins.

    Draws without replacement across sparse bins before repeating so suggestions
    spread evenly across under-covered cells. Inter-object collisions are not
    checked.

    Args:
        coverages: Per-object coverage grids.
        count: Number of batch episodes ``N`` to suggest.
        seed: Random seed for reproducible sampling.

    Returns:
        Tuple ``(suggestions, omitted_objects)`` where ``omitted_objects`` lists
        object names that had zero observed samples and could not be gridded.

    Raises:
        ValueError: If ``count < 1``.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")

    rng = np.random.default_rng(seed)
    suggestions: list[PlacementSuggestion] = []
    omitted_objects: list[str] = []

    for cov in coverages:
        if not cov.bins:
            omitted_objects.append(cov.spec.name)
            continue

        if cov.sparse_bins:
            pool = list(cov.sparse_bins)
            from_sparse = True
        else:
            min_count = min(b.count for b in cov.bins)
            pool = [b for b in cov.bins if b.count == min_count]
            from_sparse = False

        chosen_bins: list[CoverageBin] = []
        while len(chosen_bins) < count:
            order = rng.permutation(len(pool))
            for idx in order:
                chosen_bins.append(pool[int(idx)])
                if len(chosen_bins) == count:
                    break

        for batch_idx, cell in enumerate(chosen_bins, start=1):
            x_m = float(rng.uniform(cell.x_min, cell.x_max))
            y_m = float(rng.uniform(cell.y_min, cell.y_max))
            yaw_deg = float(rng.uniform(cell.yaw_min_deg, cell.yaw_max_deg))
            suggestions.append(
                PlacementSuggestion(
                    batch_index=batch_idx,
                    object_name=cov.spec.name,
                    arm=cov.spec.arm,
                    x_m=x_m,
                    y_m=y_m,
                    yaw_deg=yaw_deg,
                    ix=cell.ix,
                    iy=cell.iy,
                    yaw_bin=cell.yaw_bin,
                    current_count=cell.count,
                    from_sparse_bin=from_sparse,
                )
            )

    suggestions.sort(key=lambda item: (item.batch_index, item.object_name))
    return suggestions, omitted_objects


def analyze_dataset(
    dataset: DatasetView,
    rows: Sequence[AnnotationRow],
    task_spec: TaskSpec,
    *,
    label: str | None = None,
    speed_ratio_threshold: float = DEFAULT_SPEED_RATIO_THRESHOLD,
    min_rest_speed_mps: float = DEFAULT_MIN_REST_SPEED_MPS,
    max_traj_points: int = DEFAULT_MAX_TRAJ_POINTS,
    include_trajectories: bool = True,
) -> DatasetAnalysis:
    """Run all T3 analyses over a single validated ``(dataset, rows)`` pair.

    Loads each annotated episode's parquet state once and computes speed checks,
    object initial poses, pairwise offsets, and 3D trajectories in a single pass.

    Args:
        dataset: Open ``DatasetView``.
        rows: Validated annotation rows belonging to ``dataset``.
        task_spec: Task specification.
        label: Optional human-readable label (defaults to ``dataset.repo_id``).
        speed_ratio_threshold: Ratio threshold above which an episode is flagged
            as "not slowed down".
        min_rest_speed_mps: Non-fine median speed below which an arm is treated
            as idle.
        max_traj_points: Maximum points retained per trajectory segment.
        include_trajectories: Whether to extract 3D trajectories (skip when only
            computing coverage for ``suggest``).

    Returns:
        Populated ``DatasetAnalysis`` record.
    """
    ds_label = label or dataset.repo_id
    frame_shares = compute_frame_shares(
        rows,
        task_spec,
        fps=dataset.fps,
        episode_lengths=dataset.episode_lengths,
        label=ds_label,
    )

    by_episode: dict[int, list[AnnotationRow]] = defaultdict(list)
    for row in rows:
        by_episode[row.episode_index].append(row)

    speed_metrics: list[EpisodeSpeedMetric] = []
    object_poses: list[ObjectPoseSample] = []
    skipped_objects: list[SkippedObjectEvent] = []
    trajectories: list[TrajectorySegment] = []

    for episode_index in sorted(by_episode):
        ep_rows = sorted(by_episode[episode_index], key=lambda r: (r.start, r.end))
        arms = dataset.load_episode_arm_states(episode_index)
        arms_by_name = {arm.arm: arm for arm in arms}

        speed_metrics.extend(
            compute_episode_speed_metrics(
                dataset_label=ds_label,
                episode_index=episode_index,
                ep_rows=ep_rows,
                arms=arms,
                fps=dataset.fps,
                fine_window_s=task_spec.fine_window_s,
                speed_ratio_threshold=speed_ratio_threshold,
                min_rest_speed_mps=min_rest_speed_mps,
            )
        )

        ep_poses, ep_skipped = extract_episode_object_poses(
            dataset_label=ds_label,
            episode_index=episode_index,
            ep_rows=ep_rows,
            arms_by_name=arms_by_name,
            objects=task_spec.objects,
        )
        object_poses.extend(ep_poses)
        skipped_objects.extend(ep_skipped)

        if include_trajectories:
            trajectories.extend(
                extract_episode_trajectories(
                    dataset_label=ds_label,
                    episode_index=episode_index,
                    ep_rows=ep_rows,
                    arms=arms,
                    max_points=max_traj_points,
                )
            )

    pairwise_offsets = compute_pairwise_offsets(object_poses, task_spec.objects)

    return DatasetAnalysis(
        label=ds_label,
        repo_id=dataset.repo_id,
        revision=dataset.revision,
        fps=dataset.fps,
        arms=tuple(dataset.arm_indices.keys()),
        frame_shares=frame_shares,
        speed_metrics=tuple(speed_metrics),
        object_poses=tuple(object_poses),
        skipped_objects=tuple(skipped_objects),
        pairwise_offsets=tuple(pairwise_offsets),
        trajectories=tuple(trajectories),
    )
