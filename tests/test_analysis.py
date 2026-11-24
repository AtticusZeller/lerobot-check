"""Unit tests for lerobot_check.analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lerobot_check.analysis import (
    MixEntry,
    compute_all_coverages,
    compute_episode_speed_metrics,
    compute_frame_shares,
    compute_mix,
    compute_object_coverage,
    compute_pairwise_offsets,
    episode_fine_mask,
    extract_episode_object_poses,
    extract_episode_trajectories,
    fine_window_frame_count,
    fine_window_intervals,
    fold_yaw_deg,
    rotation_matrix_to_yaw_deg,
    suggest_placements,
    yaw_bin_index,
)
from lerobot_check.dataset import ArmState, rotation_6d_to_matrix
from lerobot_check.schema import (
    AnnotationRow,
    CoverageSpec,
    ObjectSpec,
    SubtaskSpec,
    TaskSpec,
)


def _make_task_spec() -> TaskSpec:
    return TaskSpec(
        name="demo_task",
        subtasks=(
            SubtaskSpec(name="stage_a", prompt="Do stage A."),
            SubtaskSpec(name="stage_b", prompt="Do stage B."),
        ),
        objects=(
            ObjectSpec(
                name="obj_a",
                subtask="stage_a",
                event="release",
                arm="left",
                yaw_symmetry_deg=180.0,
            ),
            ObjectSpec(
                name="obj_b",
                subtask="stage_b",
                event="grasp",
                arm="left",
                yaw_symmetry_deg=360.0,
            ),
            ObjectSpec(
                name="obj_r",
                subtask="stage_b",
                event="release",
                arm="right",
                yaw_symmetry_deg=90.0,
            ),
        ),
        fine_window_s=1.0,
        coverage=CoverageSpec(xy_bin_m=0.05, yaw_bin_deg=30.0, sparse_threshold=2),
    )


def _row(
    episode_index: int,
    subtask: str,
    start: int,
    end: int,
    *,
    contact: int | None = None,
    grasp: int | None = None,
) -> AnnotationRow:
    return AnnotationRow(
        repo_id="org/demo",
        revision="abc123",
        episode_index=episode_index,
        subtask=subtask,
        start=start,
        end=end,
        contact=contact,
        grasp=grasp,
        success=True,
        annotator="human",
        note="",
    )


def _yaw_to_rot6d(yaw_deg: float) -> np.ndarray:
    """Construct a 6D rotation vector (first two columns of Rz(yaw))."""
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    # Column 0 = [c, s, 0], Column 1 = [-s, c, 0].
    return np.array([c, s, 0.0, -s, c, 0.0], dtype=np.float64)


def test_fine_window_intervals_hand_computed_cases() -> None:
    # Case 1: contact and grasp present, disjoint from release tail.
    # [max(0, 40-10), 45] = [30, 46) (16 frames) U [90, 100) (10 frames) -> 26 frames.
    r1 = _row(0, "stage_a", 0, 100, contact=40, grasp=45)
    assert fine_window_intervals(r1, delta_frames=10) == ((30, 46), (90, 100))
    assert fine_window_frame_count(r1, delta_frames=10) == 26

    # Case 2: clipping contact - delta at row.start.
    # [max(20, 25-15), 30] = [20, 31) (11 frames) U [85, 100) (15 frames) -> 26 frames.
    r2 = _row(0, "stage_b", 20, 100, contact=25, grasp=30)
    assert fine_window_intervals(r2, delta_frames=15) == ((20, 31), (85, 100))
    assert fine_window_frame_count(r2, delta_frames=15) == 26

    # Case 3: overlapping contact-grasp and release tail merge without double counting.
    # [max(0, 20-30), 35] = [0, 36) U [20, 50) = [0, 50) -> 50 frames.
    r3 = _row(0, "stage_a", 0, 50, contact=20, grasp=35)
    assert fine_window_intervals(r3, delta_frames=30) == ((0, 50),)
    assert fine_window_frame_count(r3, delta_frames=30) == 50

    # Case 4: no contact (even when grasp is present) -> only release tail [70, 100).
    r4 = _row(0, "stage_a", 0, 100, contact=None, grasp=45)
    assert fine_window_intervals(r4, delta_frames=30) == ((70, 100),)
    assert fine_window_frame_count(r4, delta_frames=30) == 30

    # Case 5: no grasp (even when contact is present) -> only release tail.
    r5 = _row(0, "stage_a", 0, 100, contact=40, grasp=None)
    assert fine_window_intervals(r5, delta_frames=30) == ((70, 100),)

    # Case 6: short subtask clipping release tail at row.start.
    r6 = _row(0, "stage_b", 50, 62, contact=None, grasp=None)
    assert fine_window_intervals(r6, delta_frames=30) == ((50, 62),)
    assert fine_window_frame_count(r6, delta_frames=30) == 12


def test_frame_shares_hand_computed_and_skipped_episodes() -> None:
    spec = _make_task_spec()
    # fps=10, fine_window_s=1.0 -> delta_frames = 10.
    # Episode 0 (len=100):
    #   stage_a [0, 40), contact=15, grasp=19 -> fine: [5, 20) (15) U [30, 40) (10) = 25 frames.
    #   stage_b [40, 100), contact=None, grasp=55 -> fine: [90, 100) = 10 frames.
    # Episode 1: skipped (absent from annotation table).
    # Episode 2 (len=50, single-stage supplement):
    #   stage_b [0, 50), contact=4, grasp=10 -> fine: [0, 11) (clipped at 0) U [40, 50) = 21 frames.
    rows = [
        _row(0, "stage_a", 0, 40, contact=15, grasp=19),
        _row(0, "stage_b", 40, 100, contact=None, grasp=55),
        _row(2, "stage_b", 0, 50, contact=4, grasp=10),
    ]
    lengths = {0: 100, 1: 80, 2: 50}
    summary = compute_frame_shares(rows, spec, fps=10.0, episode_lengths=lengths, label="toy")

    assert summary.annotated_episodes == 2
    assert summary.total_episodes == 3
    assert summary.skipped_episodes == (1,)
    assert summary.total_frames == 150
    assert summary.total_fine_frames == 25 + 10 + 21
    assert summary.total_fine_share == pytest.approx(56 / 150)

    by_name = {s.subtask: s for s in summary.subtasks}
    assert by_name["stage_a"].episodes == 1
    assert by_name["stage_a"].frames == 40
    assert by_name["stage_a"].frame_share == pytest.approx(40 / 150)
    assert by_name["stage_a"].fine_frames == 25
    assert by_name["stage_a"].fine_share_of_subtask == pytest.approx(25 / 40)
    assert by_name["stage_a"].fine_share_of_total == pytest.approx(25 / 150)

    assert by_name["stage_b"].episodes == 2
    assert by_name["stage_b"].frames == 110
    assert by_name["stage_b"].frame_share == pytest.approx(110 / 150)
    assert by_name["stage_b"].fine_frames == 31
    assert by_name["stage_b"].fine_share_of_subtask == pytest.approx(31 / 110)
    assert by_name["stage_b"].fine_share_of_total == pytest.approx(31 / 150)


def test_mix_with_repeat_counts_and_doubling_invariant() -> None:
    spec = _make_task_spec()
    rows_a = [
        _row(0, "stage_a", 0, 40, contact=15, grasp=19),
        _row(0, "stage_b", 40, 100, contact=None, grasp=None),
    ]
    rows_b = [_row(0, "stage_a", 0, 50, contact=None, grasp=None)]

    single = compute_frame_shares(rows_a, spec, fps=10.0, episode_lengths={0: 100}, label="A")
    _, doubled = compute_mix(
        [
            MixEntry("A1", rows_a, fps=10.0, total_episodes=1, episode_lengths={0: 100}, repeat=1),
            MixEntry("A2", rows_a, fps=10.0, total_episodes=1, episode_lengths={0: 100}, repeat=1),
        ],
        spec,
    )
    assert doubled.total_frames == single.total_frames * 2
    assert doubled.total_fine_frames == single.total_fine_frames * 2
    assert doubled.total_fine_share == pytest.approx(single.total_fine_share)
    for st_single, st_double in zip(single.subtasks, doubled.subtasks, strict=True):
        assert st_double.frames == st_single.frames * 2
        assert st_double.fine_frames == st_single.fine_frames * 2
        assert st_double.frame_share == pytest.approx(st_single.frame_share)
        assert st_double.fine_share_of_total == pytest.approx(st_single.fine_share_of_total)

    # Weighted mix: A x2 (stage_a: 80f/50fine, stage_b: 120f/20fine) + B x3 (stage_a: 150f/30fine)
    # Combined: total=350, stage_a=230 (80 fine), stage_b=120 (20 fine), total fine=100.
    _, weighted = compute_mix(
        [
            MixEntry("A", rows_a, fps=10.0, total_episodes=1, episode_lengths={0: 100}, repeat=2),
            MixEntry("B", rows_b, fps=10.0, total_episodes=1, episode_lengths={0: 50}, repeat=3),
        ],
        spec,
    )
    assert weighted.total_frames == 350
    assert weighted.total_fine_frames == 100
    by_name = {s.subtask: s for s in weighted.subtasks}
    assert by_name["stage_a"].frames == 230
    assert by_name["stage_a"].fine_frames == 80
    assert by_name["stage_a"].frame_share == pytest.approx(230 / 350)
    assert by_name["stage_b"].frames == 120
    assert by_name["stage_b"].fine_frames == 20
    assert by_name["stage_b"].frame_share == pytest.approx(120 / 350)


def test_yaw_extraction_and_folding_symmetry() -> None:
    for deg in (-170.0, -90.0, -10.0, 0.0, 35.0, 90.0, 170.0):
        rot = rotation_6d_to_matrix(_yaw_to_rot6d(deg))
        assert rotation_matrix_to_yaw_deg(rot) == pytest.approx(deg, abs=1e-6)

    # Contract verification: with 180 deg symmetry, 170 deg and -10 deg land in the same bin.
    f_170 = fold_yaw_deg(170.0, 180.0)
    f_m10 = fold_yaw_deg(-10.0, 180.0)
    assert f_170 == pytest.approx(170.0)
    assert f_m10 == pytest.approx(170.0)
    bin_170, n_bins = yaw_bin_index(f_170, yaw_symmetry_deg=180.0, yaw_bin_deg=30.0)
    bin_m10, _ = yaw_bin_index(f_m10, yaw_symmetry_deg=180.0, yaw_bin_deg=30.0)
    assert n_bins == 6
    assert bin_170 == bin_m10 == 5

    # Near-zero negative wraparound lands at 0.0 (bin 0), not 180.0.
    assert fold_yaw_deg(-1e-12, 180.0) == 0.0
    assert yaw_bin_index(fold_yaw_deg(-1e-12, 180.0), yaw_symmetry_deg=180.0, yaw_bin_deg=30.0) == (
        0,
        6,
    )

    # 90 deg symmetry: 85 deg and -5 deg fold to 85 deg (bin 2 of 3).
    assert fold_yaw_deg(85.0, 90.0) == pytest.approx(fold_yaw_deg(-5.0, 90.0))
    assert yaw_bin_index(fold_yaw_deg(-5.0, 90.0), yaw_symmetry_deg=90.0, yaw_bin_deg=30.0) == (
        2,
        3,
    )


def test_speed_check_slowed_not_slowed_and_idle_arm() -> None:
    fps = 10.0
    n = 100
    ep_rows = [_row(0, "stage_a", 0, n, contact=None, grasp=None)]
    # Fine window is [90, 100) for fine_window_s=1.0 at 10 fps.
    mask = episode_fine_mask(ep_rows, n, delta_frames=10)
    assert int(np.count_nonzero(mask)) == 10

    # Left arm slows down in fine window: rest=0.10 m/s, fine=0.02 m/s -> ratio=0.20 (OK).
    left_xyz = np.zeros((n, 3), dtype=np.float64)
    for t in range(1, n):
        step = 0.002 if mask[t] else 0.010
        left_xyz[t, 0] = left_xyz[t - 1, 0] + step

    # Right arm stays completely stationary (idle): rest=0.0 m/s -> not flagged.
    right_xyz = np.zeros((n, 3), dtype=np.float64)
    rot6d = np.tile(_yaw_to_rot6d(0.0), (n, 1))
    grip = np.ones(n, dtype=np.float64)

    metrics = compute_episode_speed_metrics(
        dataset_label="ds",
        episode_index=0,
        ep_rows=ep_rows,
        arms=[
            ArmState("left", left_xyz, rot6d, grip),
            ArmState("right", right_xyz, rot6d, grip),
        ],
        fps=fps,
        fine_window_s=1.0,
        speed_ratio_threshold=0.8,
    )
    by_arm = {m.arm: m for m in metrics}
    assert by_arm["left"].ratio == pytest.approx(0.2, abs=1e-6)
    assert not by_arm["left"].not_slowed_down
    assert not by_arm["left"].idle

    assert by_arm["right"].idle
    assert by_arm["right"].ratio is None
    assert not by_arm["right"].not_slowed_down

    # Now make left arm speed up in fine window: rest=0.05 m/s, fine=0.06 m/s -> ratio=1.2 > 0.8.
    fast_xyz = np.zeros((n, 3), dtype=np.float64)
    for t in range(1, n):
        step = 0.006 if mask[t] else 0.005
        fast_xyz[t, 0] = fast_xyz[t - 1, 0] + step

    fast_metrics = compute_episode_speed_metrics(
        dataset_label="ds",
        episode_index=1,
        ep_rows=ep_rows,
        arms=[ArmState("left", fast_xyz, rot6d, grip)],
        fps=fps,
        fine_window_s=1.0,
        speed_ratio_threshold=0.8,
    )
    assert fast_metrics[0].ratio == pytest.approx(1.2, abs=1e-6)
    assert fast_metrics[0].not_slowed_down


def test_object_poses_skipped_events_pairwise_and_trajectories() -> None:
    spec = _make_task_spec()
    n = 40
    left_xyz = np.column_stack(
        [np.linspace(0.1, 0.5, n), np.linspace(0.0, 0.4, n), np.full(n, 0.8)]
    )
    right_xyz = np.column_stack(
        [np.linspace(0.2, 0.3, n), np.linspace(-0.2, -0.1, n), np.full(n, 0.7)]
    )
    left_rot = np.tile(_yaw_to_rot6d(-10.0), (n, 1))
    right_rot = np.tile(_yaw_to_rot6d(95.0), (n, 1))
    grip = np.ones(n, dtype=np.float64)
    arms = {
        "left": ArmState("left", left_xyz, left_rot, grip),
        "right": ArmState("right", right_xyz, right_rot, grip),
    }

    # Episode 0: stage_b has grasp=25 -> all 3 objects extracted.
    ep0_rows = [
        _row(0, "stage_a", 0, 20, contact=None, grasp=None),
        _row(0, "stage_b", 20, 40, contact=22, grasp=25),
    ]
    poses0, skipped0 = extract_episode_object_poses(
        dataset_label="ds",
        episode_index=0,
        ep_rows=ep0_rows,
        arms_by_name=arms,
        objects=spec.objects,
    )
    assert skipped0 == []
    by_obj0 = {p.object_name: p for p in poses0}
    assert by_obj0["obj_a"].frame == 19  # release -> end - 1
    assert by_obj0["obj_a"].folded_yaw_deg == pytest.approx(170.0)
    assert by_obj0["obj_b"].frame == 25  # grasp -> grasp
    assert by_obj0["obj_b"].folded_yaw_deg == pytest.approx(350.0)
    assert by_obj0["obj_r"].frame == 39  # release -> end - 1
    assert by_obj0["obj_r"].folded_yaw_deg == pytest.approx(5.0)  # 95 mod 90

    # Pairwise offsets: only same-arm pair (obj_a, obj_b) on left; cross-arm obj_r excluded.
    offsets = compute_pairwise_offsets(poses0, spec.objects)
    assert len(offsets) == 1
    assert offsets[0].object_a == "obj_a"
    assert offsets[0].object_b == "obj_b"
    assert offsets[0].dx == pytest.approx(by_obj0["obj_b"].x - by_obj0["obj_a"].x)
    assert offsets[0].dy == pytest.approx(by_obj0["obj_b"].y - by_obj0["obj_a"].y)

    # Episode 1: stage_b missing grasp -> obj_b skipped and listed, NOT silently dropped.
    ep1_rows = [
        _row(1, "stage_a", 0, 20, contact=None, grasp=None),
        _row(1, "stage_b", 20, 40, contact=None, grasp=None),
    ]
    poses1, skipped1 = extract_episode_object_poses(
        dataset_label="ds",
        episode_index=1,
        ep_rows=ep1_rows,
        arms_by_name=arms,
        objects=spec.objects,
    )
    assert {p.object_name for p in poses1} == {"obj_a", "obj_r"}
    assert len(skipped1) == 1
    assert skipped1[0].object_name == "obj_b"
    assert skipped1[0].episode_index == 1
    assert "grasp" in skipped1[0].reason

    # Trajectories: stage_b on ep0 has grasp-relative segment ending at (0, 0, 0).
    trajs = extract_episode_trajectories(
        dataset_label="ds",
        episode_index=0,
        ep_rows=ep0_rows,
        arms=list(arms.values()),
        max_points=10,
    )
    stage_b_left = next(t for t in trajs if t.subtask == "stage_b" and t.arm == "left")
    assert stage_b_left.grasp_relative_xyz is not None
    assert np.allclose(stage_b_left.grasp_relative_xyz[-1], [0.0, 0.0, 0.0])
    stage_a_left = next(t for t in trajs if t.subtask == "stage_a" and t.arm == "left")
    assert stage_a_left.grasp_relative_xyz is None


def test_coverage_grid_and_suggest_placements() -> None:
    spec = _make_task_spec()
    # Two samples in the same XY cell (ix=2, iy=2) and same folded yaw bin (170° & -10° -> bin 5).
    # One sample in a different XY cell (ix=3, iy=2) and yaw bin 0 (10°).
    from lerobot_check.analysis import ObjectPoseSample

    samples = [
        ObjectPoseSample(
            "ds", 0, "obj_a", "stage_a", "release", "left", 10, 0.11, 0.12, 0.8, 170.0, 170.0
        ),
        ObjectPoseSample(
            "ds", 1, "obj_a", "stage_a", "release", "left", 10, 0.12, 0.13, 0.8, -10.0, 170.0
        ),
        ObjectPoseSample(
            "ds", 2, "obj_a", "stage_a", "release", "left", 10, 0.16, 0.11, 0.8, 10.0, 10.0
        ),
    ]
    cov = compute_object_coverage(spec.objects[0], samples, spec.coverage)
    assert cov.ix_range == (2, 3)
    assert cov.iy_range == (2, 2)
    assert cov.num_yaw_bins == 6
    assert len(cov.bins) == 2 * 1 * 6  # 12 cells
    # Only (2, 2, 5) has count=2 >= sparse_threshold=2; all other 11 cells are sparse!
    assert len(cov.sparse_bins) == 11
    bin_225 = next(b for b in cov.bins if (b.ix, b.iy, b.yaw_bin) == (2, 2, 5))
    assert bin_225.count == 2

    all_covs = compute_all_coverages(samples, spec)
    suggestions, omitted = suggest_placements(all_covs, count=5, seed=123)
    # obj_b and obj_r had zero samples so they are reported as omitted.
    assert set(omitted) == {"obj_b", "obj_r"}
    assert len(suggestions) == 5
    sparse_keys = {(b.ix, b.iy, b.yaw_bin) for b in cov.sparse_bins}
    for s in suggestions:
        assert s.object_name == "obj_a"
        assert s.from_sparse_bin is True
        assert (s.ix, s.iy, s.yaw_bin) in sparse_keys
        assert s.ix * 0.05 <= s.x_m <= (s.ix + 1) * 0.05
        assert s.iy * 0.05 <= s.y_m <= (s.iy + 1) * 0.05
        assert s.yaw_bin * 30.0 <= s.yaw_deg <= (s.yaw_bin + 1) * 30.0
