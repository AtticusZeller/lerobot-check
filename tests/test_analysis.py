"""Unit tests for lerobot_check.analysis (distribution metrics, filtering, coverage) and CLI."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lerobot_check.analysis import (
    PoseXYZYaw,
    circular_yaw_span_deg,
    compute_distribution_from_states,
    compute_endpoint_coverage,
    compute_frame_shares,
    compute_object_coverage,
    compute_trajectory_dispersion_cm,
    fold_yaw_deg,
    parse_episode_filter,
    resample_segment_pose,
    rotation_matrix_to_yaw_deg,
    yaw_bin_index,
)
from lerobot_check.cli import build_parser
from lerobot_check.dataset import ArmState, rotation_6d_to_matrix
from lerobot_check.schema import (
    AnnotationRow,
    CoverageSpec,
    ObjectSpec,
    SubtaskSpec,
    TaskSpec,
)


def _yaw_to_rot6d(yaw_deg: float) -> np.ndarray:
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([c, s, 0.0, -s, c, 0.0], dtype=np.float64)


def _row(
    episode_index: int,
    subtask: str,
    start: int,
    end: int,
    *,
    grasp: int | None = None,
    annotator: str = "human",
) -> AnnotationRow:
    return AnnotationRow(
        repo_id="org/demo",
        revision="rev1",
        episode_index=episode_index,
        subtask=subtask,
        start=start,
        end=end,
        contact=None,
        grasp=grasp,
        success=True,
        annotator=annotator,  # type: ignore[arg-type]
        note="",
    )


def _make_task_spec() -> TaskSpec:
    return TaskSpec(
        name="demo_task",
        subtasks=(
            SubtaskSpec(name="stage_a", prompt="Do stage A."),
            SubtaskSpec(name="stage_b", prompt="Do stage B.", arms=("left", "right")),
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
        ),
        coverage=CoverageSpec(xy_bin_m=0.05, yaw_bin_deg=30.0, sparse_threshold=2),
    )


def test_yaw_extraction_folding_and_circular_span() -> None:
    for deg in (-170.0, -90.0, -10.0, 0.0, 35.0, 90.0, 170.0, 180.0):
        rot = rotation_6d_to_matrix(_yaw_to_rot6d(deg))
        assert rotation_matrix_to_yaw_deg(rot) == pytest.approx(deg, abs=1e-6)

    # 180-deg symmetry folds 170 and -10 into the same bin (bin 5 of 6).
    assert fold_yaw_deg(170.0, 180.0) == pytest.approx(170.0)
    assert fold_yaw_deg(-10.0, 180.0) == pytest.approx(170.0)
    assert yaw_bin_index(170.0, yaw_symmetry_deg=180.0, yaw_bin_deg=30.0) == (5, 6)
    assert fold_yaw_deg(-1e-12, 180.0) == 0.0

    # Circular span handles +/-180 wraparound cleanly.
    assert circular_yaw_span_deg([]) == 0.0
    assert circular_yaw_span_deg([42.0]) == 0.0
    assert circular_yaw_span_deg([175.0, -175.0]) == pytest.approx(10.0)
    assert circular_yaw_span_deg([10.0, 40.0]) == pytest.approx(30.0)
    assert circular_yaw_span_deg([0.0, 90.0, 180.0, -90.0]) == pytest.approx(270.0)


def test_endpoint_coverage_and_trajectory_dispersion_hand_computed() -> None:
    # 3 poses: two in cell (2, 1) and one in cell (3, 1) -> 2 occupied XY cells.
    # Yaws: 170, -170, 180 -> circular span = 20 deg.
    poses = [
        PoseXYZYaw(0.11, 0.06, 0.5, 170.0),
        PoseXYZYaw(0.14, 0.09, 0.5, -170.0),
        PoseXYZYaw(0.16, 0.07, 0.5, 180.0),
    ]
    cov = compute_endpoint_coverage(poses, xy_bin_m=0.05)
    assert cov.occupied_xy_cells == 2
    assert cov.yaw_span_deg == pytest.approx(20.0)

    # Two parallel trajectories shifted by 0.04 m along x and 0.06 m along y at every progress step:
    # Distance from midpoint at every k is hypot(0.02, 0.03) = sqrt(0.0013) m.
    k = 10
    base = np.column_stack([np.linspace(0.0, 1.0, k), np.zeros(k), np.full(k, 0.5)])
    shift = np.array([0.04, 0.06, 0.0])
    disp_cm = compute_trajectory_dispersion_cm([base, base + shift])
    expected_cm = math.hypot(0.02, 0.03) * 100.0
    assert disp_cm == pytest.approx(expected_cm, rel=1e-9)

    assert compute_trajectory_dispersion_cm([base]) == 0.0
    assert compute_trajectory_dispersion_cm([]) is None


def test_resample_segment_pose_interpolates_shortest_yaw_arc() -> None:
    xyz = np.array([[0.0, 0.0, 0.2], [1.0, 0.0, 0.2]], dtype=np.float64)
    rot6d = np.vstack([_yaw_to_rot6d(170.0), _yaw_to_rot6d(-170.0)])
    pts = resample_segment_pose(xyz, rot6d, num_samples=5)
    assert len(pts) == 5
    assert pts[0].progress == pytest.approx(0.0)
    assert pts[-1].progress == pytest.approx(1.0)
    assert pts[0].yaw_deg == pytest.approx(170.0)
    # Midpoint across +170 -> -170 passes through 180 deg, not 0 deg.
    assert pts[2].x_m == pytest.approx(0.5)
    assert pts[2].yaw_deg == pytest.approx(180.0)
    assert pts[-1].yaw_deg == pytest.approx(-170.0)


def test_distribution_hand_computed_and_episode_filter() -> None:
    spec = _make_task_spec()
    # Episode 0 (len=40): stage_a [0, 10), stage_b [10, 40) grasp=20
    # Episode 1 (len=30): skipped (unannotated)
    # Episode 2 (len=60): stage_a [0, 20), stage_b [20, 60) grasp=30
    # Episode 3 (len=20): single-stage supplement stage_b [0, 20) grasp=None (agent)
    rows = [
        _row(0, "stage_a", 0, 10, annotator="human"),
        _row(0, "stage_b", 10, 40, grasp=20, annotator="human"),
        _row(2, "stage_a", 0, 20, annotator="converted"),
        _row(2, "stage_b", 20, 60, grasp=30, annotator="converted"),
        _row(3, "stage_b", 0, 20, grasp=None, annotator="agent"),
    ]
    ep_lengths = {0: 40, 1: 30, 2: 60, 3: 20}

    def _make_arm(
        n: int, x0: float, x1: float, y: float, z: float, yaw0: float, yaw1: float
    ) -> ArmState:
        xyz = np.column_stack([np.linspace(x0, x1, n), np.full(n, y), np.full(n, z)])
        yaws = np.linspace(yaw0, yaw1, n)
        rot6d = np.vstack([_yaw_to_rot6d(float(a)) for a in yaws])
        return ArmState("arm", xyz, rot6d, np.ones(n))

    arm_states = {
        0: {
            "left": _make_arm(40, 0.10, 0.50, 0.10, 0.30, 175.0, 175.0),
            "right": _make_arm(40, 0.20, 0.20, -0.10, 0.30, 0.0, 0.0),
        },
        2: {
            "left": _make_arm(60, 0.10, 0.50, 0.14, 0.30, -175.0, -175.0),
            "right": _make_arm(60, 0.20, 0.20, -0.10, 0.30, 0.0, 0.0),
        },
        3: {
            "left": _make_arm(20, 0.16, 0.56, 0.20, 0.34, 20.0, 20.0),
            "right": _make_arm(20, 0.25, 0.25, -0.10, 0.30, 10.0, 10.0),
        },
    }

    # Check frame shares on all episodes: total_frames = 40 + 60 + 20 = 120.
    # stage_a = 10 + 20 = 30 (25%), stage_b = 30 + 40 + 20 = 90 (75%).
    shares = compute_frame_shares(rows, spec, episode_lengths=ep_lengths)
    assert shares.total_frames == 120
    assert shares.annotated_episodes == (0, 2, 3)
    assert shares.skipped_episodes == (1,)
    by_st = {s.subtask: s for s in shares.subtasks}
    assert by_st["stage_a"].frames == 30
    assert by_st["stage_a"].share == pytest.approx(30 / 120)
    assert by_st["stage_b"].frames == 90
    assert by_st["stage_b"].share == pytest.approx(90 / 120)

    dist = compute_distribution_from_states(
        repo_id="org/demo",
        revision="rev1",
        fps=30.0,
        episode_lengths=ep_lengths,
        arm_states_by_episode=arm_states,
        rows=rows,
        task_spec=spec,
        trajectory_samples=10,
    )
    assert dist["version"] == 1
    assert dist["provenance"]["annotator_counts"] == {
        "agent": 1,
        "human": 2,
        "converted": 2,
        "total": 5,
    }
    assert dist["provenance"]["annotated_episodes"] == [0, 2, 3]
    assert dist["provenance"]["skipped_episodes"] == [1]

    # Participating arms: stage_a defaults to ("left",), stage_b explicitly ("left", "right").
    subtasks_by_name = {s["name"]: s for s in dist["subtasks"]}
    assert subtasks_by_name["stage_a"]["participating_arms"] == ["left"]
    assert subtasks_by_name["stage_b"]["participating_arms"] == ["left", "right"]

    # stage_a left has episodes 0 and 2:
    # Start poses: ep0=(0.10, 0.10, 0.30, 175), ep2=(0.10, 0.14, 0.30, -175)
    # Both fall in XY cell (2, 2) -> occupied_xy_cells == 1, yaw_span == 10 deg.
    sa_left_m = subtasks_by_name["stage_a"]["arms"]["left"]["metrics"]
    assert sa_left_m["episode_count"] == 2
    assert sa_left_m["start_coverage"]["occupied_xy_cells"] == 1
    assert sa_left_m["start_coverage"]["yaw_span_deg"] == pytest.approx(10.0)
    assert len(subtasks_by_name["stage_a"]["arms"]["left"]["episodes"][0]["trajectory"]) == 10

    # Object obj_b skipped on ep 3 (missing grasp), extracted on ep 0 & 2.
    obj_b = next(o for o in dist["objects"] if o["name"] == "obj_b")
    assert len(obj_b["poses"]) == 2
    assert obj_b["skipped"] == [{"episode_index": 3, "reason": "missing grasp frame"}]
    assert obj_b["median_z_m"] == pytest.approx(0.30)

    # Filter to episodes [0, 2]: ep 3 excluded, ep 1 absent from filter so skipped=[].
    dist_02 = compute_distribution_from_states(
        repo_id="org/demo",
        revision="rev1",
        fps=30.0,
        episode_lengths=ep_lengths,
        arm_states_by_episode=arm_states,
        rows=rows,
        task_spec=spec,
        episode_filter=[0, 2],
        trajectory_samples=10,
    )
    assert dist_02["provenance"]["episode_filter"] == [0, 2]
    assert dist_02["provenance"]["annotated_episodes"] == [0, 2]
    assert dist_02["provenance"]["skipped_episodes"] == []
    assert dist_02["provenance"]["annotator_counts"]["agent"] == 0
    assert dist_02["frame_shares"]["total_frames"] == 100
    fs_02 = {s["subtask"]: s for s in dist_02["frame_shares"]["subtasks"]}
    assert fs_02["stage_a"]["frames"] == 30
    assert fs_02["stage_b"]["frames"] == 70
    assert fs_02["stage_b"]["share"] == pytest.approx(0.7)

    # In [0, 2], stage_b right arm is identical (0.20, -0.10, 0.30) -> dispersion == 0.0 cm.
    sb_right_m = next(s for s in dist_02["subtasks"] if s["name"] == "stage_b")["arms"]["right"][
        "metrics"
    ]
    assert sb_right_m["trajectory_dispersion_cm"] == pytest.approx(0.0)


def test_parse_episode_filter_formats() -> None:
    assert parse_episode_filter(["0,30,60"], 121) == [0, 30, 60]
    assert parse_episode_filter(["0-3", "10:12"], 20) == [0, 1, 2, 3, 10, 11]
    with pytest.raises(ValueError, match="outside"):
        parse_episode_filter(["0-20"], 20)
    with pytest.raises(ValueError, match="outside"):
        parse_episode_filter(["20"], 20)


def test_object_coverage_grid_binning() -> None:
    from lerobot_check.analysis import ObjectPoseSample

    spec = _make_task_spec()
    samples = [
        ObjectPoseSample(
            0, "obj_a", "stage_a", "release", "left", 9, 0.11, 0.12, 0.3, 170.0, 170.0
        ),
        ObjectPoseSample(
            1, "obj_a", "stage_a", "release", "left", 9, 0.12, 0.13, 0.3, -10.0, 170.0
        ),
        ObjectPoseSample(2, "obj_a", "stage_a", "release", "left", 9, 0.16, 0.11, 0.3, 10.0, 10.0),
    ]
    cov = compute_object_coverage(spec.objects[0], samples, spec.coverage)
    assert cov.ix_range == (2, 3)
    assert cov.iy_range == (2, 2)
    assert cov.num_yaw_bins == 6
    assert len(cov.bins) == 12
    b225 = next(b for b in cov.bins if (b.ix, b.iy, b.yaw_bin) == (2, 2, 5))
    assert b225.count == 2


def test_cli_subcommands_removed_and_present() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for removed in ("report", "mix"):
        assert f" {removed}" not in help_text
        assert f",{removed}" not in help_text
    for expected in (
        "info",
        "propose",
        "render",
        "validate",
        "relabel",
        "verify-relabel",
        "distribution",
        "suggest",
        "calib-frames",
        "calibrate",
    ):
        assert expected in help_text
