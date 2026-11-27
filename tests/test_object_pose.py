from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from PIL import Image

from lerobot_check.analysis import (
    compute_distribution_from_states,
    compute_object_coverage,
)
from lerobot_check.calib import (
    ArmCalibration,
    HeadCalibration,
    parse_intrinsics,
    project_base_points,
)
from lerobot_check.cli import main
from lerobot_check.dataset import ArmState
from lerobot_check.masks import sha256_bytes
from lerobot_check.object_pose import (
    OBJECT_LAYER_SOURCE,
    OBJECT_POSE_FRAME,
    OBJECT_POSES_SCHEMA_VERSION,
    SKIP_BORDER_TOUCH,
    SKIP_EMPTY_MASK,
    EpisodeObjectMeasurement,
    ObjectPoseMeasurement,
    apply_size_outlier_filter,
    build_object_poses_payload,
    evaluate_mask_for_episode,
    load_object_poses,
    mask_touches_layout_box,
    measure_outline_pose,
    min_area_rect_pose_xy,
    object_yaw_period_deg,
    polygon_centroid_xy,
    render_object_pose_sheets,
)
from lerobot_check.schema import (
    AnnotationRow,
    CoverageSpec,
    ObjectSpec,
    SubtaskSpec,
    TaskSpec,
    load_task_spec,
    parse_task_spec,
    save_task_spec,
)


def _synthetic_calibration() -> HeadCalibration:
    intrinsics = parse_intrinsics(
        {
            "version": 1,
            "camera": "observation.images.head",
            "image_size": [640, 480],
            "fx": 600.0,
            "fy": 600.0,
            "cx": 320.0,
            "cy": 240.0,
            "distortion_model": "inverse_brown_conrady",
            "distortion": [0.05, -0.02, 0.001, -0.001, 0.0],
        }
    )
    tilt = math.radians(20.0)
    r_cam_base = np.array(
        [
            [0.0, -1.0, 0.0],
            [-math.cos(tilt), 0.0, -math.sin(tilt)],
            [math.sin(tilt), 0.0, -math.cos(tilt)],
        ],
        dtype=np.float64,
    )
    cam_pos_base = np.array([0.40, 0.0, 0.65], dtype=np.float64)
    t_cb = -r_cam_base @ cam_pos_base
    T_right = np.eye(4, dtype=np.float64)
    T_right[:3, :3] = r_cam_base
    T_right[:3, 3] = t_cb
    right_cal = ArmCalibration(
        arm="right",
        T_cam_base=T_right,
        tool_offset=np.array([0.0, 0.0, 0.15], dtype=np.float64),
        n_points=10,
        rms_px=1.0,
        loo_rms_px=1.2,
        residuals=(),
    )
    return HeadCalibration(
        repo_id="Org/demo",
        revision="rev1",
        camera="observation.images.head",
        image_size=(640, 480),
        intrinsics=intrinsics,
        arms={"right": right_cal},
    )


def _rotated_rect_xy(
    cx: float, cy: float, length: float, width: float, angle_deg: float
) -> np.ndarray:
    hl, hw = length * 0.5, width * 0.5
    corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=np.float64)
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    rot = np.array([[c, s], [-s, c]], dtype=np.float64)
    return corners @ rot + np.array([cx, cy], dtype=np.float64)


def _make_task_spec(*, with_layout: bool = True) -> TaskSpec:
    return TaskSpec(
        name="demo_task",
        subtasks=(SubtaskSpec(name="pick", prompt="Pick object."),),
        objects=(
            ObjectSpec(
                name="phone",
                subtask="pick",
                event="grasp",
                arm="right",
                yaw_symmetry_deg=360.0,
                layout_box=(100, 100, 300, 300) if with_layout else None,
                object_plane_z_m=-0.42 if with_layout else None,
            ),
            ObjectSpec(
                name="unboxed",
                subtask="pick",
                event="release",
                arm="right",
                yaw_symmetry_deg=180.0,
            ),
        ),
        coverage=CoverageSpec(xy_bin_m=0.05, yaw_bin_deg=30.0, sparse_threshold=2),
    )


def _make_arm_state(n: int = 10) -> ArmState:
    xyz = np.column_stack([np.full(n, 0.8), np.full(n, 0.1), np.full(n, -0.42)])
    rot6d = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64), (n, 1))
    return ArmState("right", xyz, rot6d, np.ones(n))


def _ann_row(ep: int) -> AnnotationRow:
    return AnnotationRow(
        repo_id="Org/demo",
        revision="rev1",
        episode_index=ep,
        subtask="pick",
        start=0,
        end=10,
        contact=None,
        grasp=4,
        success=True,
        annotator="human",
        note="",
    )


def test_task_spec_layout_box_and_plane_z_round_trip_and_validation(tmp_path: Path) -> None:
    spec = _make_task_spec(with_layout=True)
    out = tmp_path / "spec.yaml"
    save_task_spec(spec, out)
    loaded = load_task_spec(out)
    assert loaded == spec
    assert loaded.objects[0].layout_box == (100, 100, 300, 300)
    assert loaded.objects[0].object_plane_z_m == pytest.approx(-0.42)
    assert loaded.objects[1].layout_box is None
    assert loaded.objects[1].object_plane_z_m is None

    legacy = _make_task_spec(with_layout=False)
    legacy_out = tmp_path / "legacy.yaml"
    save_task_spec(legacy, legacy_out)
    legacy_text = legacy_out.read_text(encoding="utf-8")
    assert "layout_box" not in legacy_text
    assert "object_plane_z_m" not in legacy_text
    assert load_task_spec(legacy_out) == legacy


@pytest.mark.parametrize(
    "extra",
    [
        {"layout_box": [10, 20, 30]},
        {"layout_box": [10.5, 20, 30, 40]},
        {"layout_box": [True, 20, 30, 40]},
        {"layout_box": [-1, 20, 30, 40]},
        {"layout_box": [30, 20, 30, 40]},
        {"layout_box": [10, 40, 30, 20]},
        {"layout_box": "10,20,30,40"},
        {"object_plane_z_m": True},
        {"object_plane_z_m": "high"},
        {"object_plane_z_m": float("nan")},
        {"object_plane_z_m": float("inf")},
    ],
)
def test_task_spec_rejects_invalid_layout_box_and_plane_z(extra: dict[str, object]) -> None:
    raw = {
        "name": "bad",
        "subtasks": [{"name": "s", "prompt": "p"}],
        "objects": [
            {
                "name": "obj",
                "subtask": "s",
                "event": "grasp",
                "arm": "right",
                "yaw_symmetry_deg": 180.0,
                **extra,
            }
        ],
    }
    with pytest.raises(ValueError):
        parse_task_spec(raw)


def test_backproject_round_trip_with_synthetic_calibration() -> None:
    calib = _synthetic_calibration()
    plane_z = -0.42
    cx, cy = 0.85, 0.08
    length_m, width_m = 0.14, 0.07
    yaw_deg = 37.5
    rect_xy = _rotated_rect_xy(cx, cy, length_m, width_m, yaw_deg)
    rect_xyz = np.column_stack([rect_xy, np.full(len(rect_xy), plane_z)])
    outline_px = project_base_points(rect_xyz, calib.arms["right"].T_cam_base, calib.intrinsics)

    pose = measure_outline_pose(
        outline_px,
        plane_z_m=plane_z,
        arm_calibration=calib.arms["right"],
        intrinsics=calib.intrinsics,
        yaw_symmetry_deg=360.0,
    )
    assert pose.x_m == pytest.approx(cx, abs=1e-5)
    assert pose.y_m == pytest.approx(cy, abs=1e-5)
    assert pose.z_m == pytest.approx(plane_z)
    assert pose.length_m == pytest.approx(length_m, abs=1e-5)
    assert pose.width_m == pytest.approx(width_m, abs=1e-5)
    assert pose.raw_yaw_deg == pytest.approx(yaw_deg, abs=1e-4)
    assert pose.folded_yaw_deg == pytest.approx(yaw_deg, abs=1e-4)


@pytest.mark.parametrize(
    ("angle_deg", "expected_raw"),
    [
        (0.0, 0.0),
        (90.0, 90.0),
        (179.0, 179.0),
        (180.0, 0.0),
        (-1.0, 179.0),
    ],
)
@pytest.mark.parametrize("yaw_symmetry_deg", [90.0, 180.0, 360.0])
def test_rectangle_yaw_and_folding(
    angle_deg: float, expected_raw: float, yaw_symmetry_deg: float
) -> None:
    calib = _synthetic_calibration()
    plane_z = -0.42
    rect_xy = _rotated_rect_xy(0.82, -0.05, 0.12, 0.06, angle_deg)
    raw_yaw, length_m, width_m = min_area_rect_pose_xy(rect_xy)
    assert raw_yaw == pytest.approx(expected_raw, abs=1e-6)
    assert length_m == pytest.approx(0.12, abs=1e-9)
    assert width_m == pytest.approx(0.06, abs=1e-9)

    rect_xyz = np.column_stack([rect_xy, np.full(len(rect_xy), plane_z)])
    outline_px = project_base_points(rect_xyz, calib.arms["right"].T_cam_base, calib.intrinsics)
    pose = measure_outline_pose(
        outline_px,
        plane_z_m=plane_z,
        arm_calibration=calib.arms["right"],
        intrinsics=calib.intrinsics,
        yaw_symmetry_deg=yaw_symmetry_deg,
    )
    period = object_yaw_period_deg(yaw_symmetry_deg)
    assert period == min(yaw_symmetry_deg, 180.0)
    expected_folded = expected_raw % period
    if abs(expected_folded - period) < 1e-9:
        expected_folded = 0.0
    assert pose.raw_yaw_deg == pytest.approx(expected_raw, abs=1e-4)
    assert pose.folded_yaw_deg == pytest.approx(expected_folded, abs=1e-4)
    assert 0.0 <= pose.folded_yaw_deg < period


def test_polygon_centroid_invariant_to_vertex_density() -> None:
    top_edge = np.column_stack([np.linspace(0.0, 1.0, 20), np.full(20, 0.5)])
    poly = np.vstack([[[0.0, 0.0], [1.0, 0.0]], top_edge[::-1]])
    cx, cy = polygon_centroid_xy(poly)
    assert cx == pytest.approx(0.5, abs=1e-9)
    assert cy == pytest.approx(0.25, abs=1e-9)


def test_skip_on_empty_mask_border_touch_and_missing_arm() -> None:
    calib = _synthetic_calibration()
    box = (200, 150, 400, 350)

    empty = np.zeros((480, 640), dtype=bool)
    res_empty = evaluate_mask_for_episode(
        0,
        empty,
        layout_box=box,
        plane_z_m=-0.42,
        arm_name="right",
        calibration=calib,
        yaw_symmetry_deg=180.0,
    )
    assert res_empty.status == SKIP_EMPTY_MASK
    assert res_empty.outline is None
    assert res_empty.pose is None

    border_mask = np.zeros((480, 640), dtype=bool)
    border_mask[200:260, 200:260] = True
    assert mask_touches_layout_box(border_mask, box) is True
    res_border = evaluate_mask_for_episode(
        1,
        border_mask,
        layout_box=box,
        plane_z_m=-0.42,
        arm_name="right",
        calibration=calib,
        yaw_symmetry_deg=180.0,
    )
    assert res_border.status == SKIP_BORDER_TOUCH
    assert res_border.outline is not None
    assert res_border.pose is None

    clean_mask = np.zeros((480, 640), dtype=bool)
    pts = np.array([[260, 200], [340, 200], [340, 260], [260, 260]], dtype=np.int32)
    cv2.fillPoly(clean_mask.view(np.uint8), [pts], 1)
    assert mask_touches_layout_box(clean_mask, box) is False
    res_ok = evaluate_mask_for_episode(
        2,
        clean_mask,
        layout_box=box,
        plane_z_m=-0.42,
        arm_name="right",
        calibration=calib,
        yaw_symmetry_deg=180.0,
    )
    assert res_ok.status == "ok"
    assert res_ok.outline is not None
    assert res_ok.pose is not None

    res_arm = evaluate_mask_for_episode(
        3,
        clean_mask,
        layout_box=box,
        plane_z_m=-0.42,
        arm_name="left",
        calibration=calib,
        yaw_symmetry_deg=180.0,
    )
    assert "missing calibration for arm" in res_arm.status
    assert res_arm.pose is None


def test_size_outlier_filter_skips_far_from_median() -> None:
    def _rec(ep: int, length: float, width: float) -> EpisodeObjectMeasurement:
        return EpisodeObjectMeasurement(
            episode_index=ep,
            frame=0,
            status="ok",
            outline=[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0]],
            pose=ObjectPoseMeasurement(0.8, 0.1, -0.42, 45.0, 45.0, length, width),
        )

    records = [
        _rec(0, 0.10, 0.05),
        _rec(1, 0.102, 0.051),
        _rec(2, 0.098, 0.049),
        _rec(3, 0.16, 0.05),
    ]
    filtered = apply_size_outlier_filter(records, max_size_deviation=0.30)
    assert [r.status for r in filtered[:3]] == ["ok", "ok", "ok"]
    assert "far from median" in filtered[3].status
    assert filtered[3].pose is None
    assert filtered[3].outline is not None


def test_sidecar_and_object_layer_json_structure_and_regression(tmp_path: Path) -> None:
    spec = _make_task_spec(with_layout=True)
    calib_sha = "a" * 64
    ep_lengths = {0: 10, 1: 10, 2: 10}
    rows = [_ann_row(0), _ann_row(1), _ann_row(2)]
    arm_states = {ep: {"right": _make_arm_state()} for ep in ep_lengths}
    tool_offsets = {"right": np.array([0.0, 0.0, 0.15])}

    baseline_a = compute_distribution_from_states(
        repo_id="Org/demo",
        revision="rev1",
        fps=30.0,
        episode_lengths=ep_lengths,
        arm_states_by_episode=arm_states,
        rows=rows,
        task_spec=spec,
        trajectory_samples=5,
        tool_offsets=tool_offsets,
    )
    baseline_b = compute_distribution_from_states(
        repo_id="Org/demo",
        revision="rev1",
        fps=30.0,
        episode_lengths=ep_lengths,
        arm_states_by_episode=arm_states,
        rows=rows,
        task_spec=_make_task_spec(with_layout=False),
        trajectory_samples=5,
        tool_offsets=tool_offsets,
    )
    assert json.dumps(baseline_a, indent=2) == json.dumps(baseline_b, indent=2)
    assert "object_poses_sha256" not in baseline_a["provenance"]
    assert all("object_layer" not in obj for obj in baseline_a["objects"])

    sidecar = build_object_poses_payload(
        repo_id="Org/demo",
        revision="rev1",
        camera="observation.images.head",
        image_size=(640, 480),
        calibration_sha256=calib_sha,
        objects=spec.objects,
        plane_z_by_object={"phone": -0.42},
        measurements_by_object={
            "phone": [
                EpisodeObjectMeasurement(
                    episode_index=0,
                    frame=0,
                    status="ok",
                    outline=[[250.0, 200.0], [310.0, 200.0], [310.0, 240.0], [250.0, 240.0]],
                    pose=ObjectPoseMeasurement(0.96, 0.23, -0.42, 114.7, 114.7, 0.075, 0.050),
                ),
                EpisodeObjectMeasurement(
                    episode_index=1,
                    frame=0,
                    status=SKIP_BORDER_TOUCH,
                    outline=[[100.0, 100.0], [150.0, 100.0], [150.0, 150.0]],
                    pose=None,
                ),
                EpisodeObjectMeasurement(
                    episode_index=2,
                    frame=0,
                    status="ok",
                    outline=[[260.0, 210.0], [320.0, 210.0], [320.0, 250.0], [260.0, 250.0]],
                    pose=ObjectPoseMeasurement(0.97, 0.24, -0.42, 15.0, 15.0, 0.076, 0.051),
                ),
            ]
        },
    )
    sidecar_path = tmp_path / "object_poses.json"
    sidecar_bytes = (json.dumps(sidecar, indent=2) + "\n").encode("utf-8")
    sidecar_path.write_bytes(sidecar_bytes)
    loaded_sidecar, loaded_sha = load_object_poses(sidecar_path)
    assert loaded_sha == sha256_bytes(sidecar_bytes)
    assert loaded_sidecar["version"] == OBJECT_POSES_SCHEMA_VERSION
    assert set(loaded_sidecar["objects"]) == {"phone"}

    dist = compute_distribution_from_states(
        repo_id="Org/demo",
        revision="rev1",
        fps=30.0,
        episode_lengths=ep_lengths,
        arm_states_by_episode=arm_states,
        rows=rows,
        task_spec=spec,
        trajectory_samples=5,
        tool_offsets=tool_offsets,
        calib_sha256=calib_sha,
        object_poses=loaded_sidecar,
        object_poses_sha256=loaded_sha,
    )
    assert dist["provenance"]["object_poses_sha256"] == loaded_sha
    phone_obj = next(o for o in dist["objects"] if o["name"] == "phone")
    unboxed_obj = next(o for o in dist["objects"] if o["name"] == "unboxed")
    assert "object_layer" not in unboxed_obj
    layer = phone_obj["object_layer"]
    assert layer["source"] == OBJECT_LAYER_SOURCE
    assert layer["frame"] == OBJECT_POSE_FRAME
    assert layer["arm"] == "right"
    assert layer["plane_z_m"] == pytest.approx(-0.42)
    assert layer["yaw_period_deg"] == pytest.approx(180.0)
    assert len(layer["poses"]) == 2
    assert layer["poses"][0] == {
        "episode_index": 0,
        "frame": 0,
        "arm": "right",
        "x_m": 0.96,
        "y_m": 0.23,
        "z_m": -0.42,
        "raw_yaw_deg": 114.7,
        "folded_yaw_deg": 114.7,
        "length_m": 0.075,
        "width_m": 0.050,
    }
    assert layer["skipped"] == [{"episode_index": 1, "reason": SKIP_BORDER_TOUCH}]
    assert layer["coverage"]["sample_count"] == 2
    assert layer["coverage"]["num_yaw_bins"] == 6
    assert set(layer["coverage"]) == set(phone_obj["coverage"])

    dist_filtered = compute_distribution_from_states(
        repo_id="Org/demo",
        revision="rev1",
        fps=30.0,
        episode_lengths=ep_lengths,
        arm_states_by_episode=arm_states,
        rows=rows,
        task_spec=spec,
        episode_filter=[0, 1],
        trajectory_samples=5,
        tool_offsets=tool_offsets,
        calib_sha256=calib_sha,
        object_poses=loaded_sidecar,
        object_poses_sha256=loaded_sha,
    )
    filtered_layer = dist_filtered["objects"][0]["object_layer"]
    assert [p["episode_index"] for p in filtered_layer["poses"]] == [0]
    assert filtered_layer["skipped"] == [{"episode_index": 1, "reason": SKIP_BORDER_TOUCH}]
    assert filtered_layer["coverage"]["sample_count"] == 1
    expected_cov = compute_object_coverage(
        ObjectSpec("phone", "pick", "grasp", "right", 180.0),
        [],
        spec.coverage,
    )
    assert filtered_layer["coverage"]["num_yaw_bins"] == expected_cov.num_yaw_bins


@pytest.mark.parametrize(
    ("override", "calib_sha_arg", "match"),
    [
        ({"repo_id": "Other/ds"}, "a" * 64, "repo_id/revision mismatch"),
        ({"revision": "rev_other"}, "a" * 64, "repo_id/revision mismatch"),
        ({"calibration_sha256": "b" * 64}, "a" * 64, "calibration_sha256"),
        ({}, None, "no head calibration"),
    ],
)
def test_distribution_rejects_mismatched_sidecar(
    override: dict[str, object], calib_sha_arg: str | None, match: str
) -> None:
    spec = _make_task_spec(with_layout=True)
    sidecar = build_object_poses_payload(
        repo_id="Org/demo",
        revision="rev1",
        camera="observation.images.head",
        image_size=(640, 480),
        calibration_sha256="a" * 64,
        objects=spec.objects,
        plane_z_by_object={"phone": -0.42},
        measurements_by_object={"phone": []},
    )
    sidecar.update(override)
    with pytest.raises(ValueError, match=match):
        compute_distribution_from_states(
            repo_id="Org/demo",
            revision="rev1",
            fps=30.0,
            episode_lengths={0: 10},
            arm_states_by_episode={0: {"right": _make_arm_state()}},
            rows=[_ann_row(0)],
            task_spec=spec,
            trajectory_samples=5,
            calib_sha256=calib_sha_arg,
            object_poses=sidecar,
            object_poses_sha256="c" * 64,
        )


def test_render_object_pose_sheets_creates_jpegs(tmp_path: Path) -> None:
    spec = _make_task_spec(with_layout=True)
    sidecar = build_object_poses_payload(
        repo_id="Org/demo",
        revision="rev1",
        camera="observation.images.head",
        image_size=(640, 480),
        calibration_sha256="a" * 64,
        objects=spec.objects,
        plane_z_by_object={"phone": -0.42},
        measurements_by_object={
            "phone": [
                EpisodeObjectMeasurement(
                    episode_index=0,
                    frame=0,
                    status="ok",
                    outline=[[150.0, 150.0], [220.0, 150.0], [220.0, 220.0], [150.0, 220.0]],
                    pose=ObjectPoseMeasurement(0.9, 0.2, -0.42, 45.0, 45.0, 0.08, 0.05),
                ),
                EpisodeObjectMeasurement(
                    episode_index=1,
                    frame=0,
                    status=SKIP_BORDER_TOUCH,
                    outline=[[100.0, 100.0], [150.0, 100.0], [150.0, 150.0]],
                    pose=None,
                ),
            ]
        },
    )
    images = {
        0: Image.new("RGB", (640, 480), color=(30, 30, 30)),
        1: Image.new("RGB", (640, 480), color=(40, 40, 40)),
    }
    written = render_object_pose_sheets(images, sidecar, tmp_path / "sheets", columns=2, per_page=1)
    assert len(written) == 2
    assert all(path.is_file() and path.suffix == ".jpg" for path in written)


def test_cli_object_poses_and_distribution_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["object-poses", "--dataset-root", str(tmp_path / "missing")])
    assert rc == 1
    assert "[dataset]" in capsys.readouterr().err

    ds_dir = tmp_path / "lerobot" / "Org" / "demo"
    ds_dir.mkdir(parents=True)
    spec_path = tmp_path / "no_boxes.yaml"
    save_task_spec(_make_task_spec(with_layout=False), spec_path)

    calib = _synthetic_calibration()
    calib_payload = {
        "version": 1,
        "repo_id": "Org/demo",
        "revision": "rev1",
        "camera": "observation.images.head",
        "image_size": [640, 480],
        "intrinsics": calib.intrinsics.raw,
        "arms": {
            "right": {
                "T_cam_base": calib.arms["right"].T_cam_base.tolist(),
                "tool_offset": calib.arms["right"].tool_offset.tolist(),
                "n_points": 10,
                "rms_px": 1.0,
                "loo_rms_px": 1.2,
                "residuals": [],
            }
        },
    }
    calib_path = tmp_path / "head.json"
    calib_path.write_text(json.dumps(calib_payload), encoding="utf-8")

    fake_ds = SimpleNamespace(
        root=ds_dir.resolve(),
        repo_id="Org/demo",
        revision="rev1",
        fps=30.0,
        total_episodes=1,
        episode_lengths={0: 10},
        camera_keys=("observation.images.head",),
    )
    monkeypatch.setattr("lerobot_check.object_pose.DatasetView", lambda *a, **kw: fake_ds)

    rc = main(
        [
            "object-poses",
            "--dataset-root",
            str(ds_dir),
            "--task-spec",
            str(spec_path),
            "--calib",
            str(tmp_path / "missing_head.json"),
        ]
    )
    assert rc == 1
    assert "[calib]" in capsys.readouterr().err

    monkeypatch.setattr("lerobot_check.object_pose.SamSegmenter", Mock())
    rc = main(
        [
            "object-poses",
            "--dataset-root",
            str(ds_dir),
            "--task-spec",
            str(spec_path),
            "--calib",
            str(calib_path),
        ]
    )
    assert rc == 1
    assert "no objects with layout_box" in capsys.readouterr().err

    monkeypatch.setattr("lerobot_check.cli.DatasetView", lambda *a, **kw: fake_ds)
    ann_path = tmp_path / "ann.csv"
    ann_path.write_text(
        "repo_id,revision,episode_index,subtask,start,end,contact,grasp,success,annotator,note\n"
        "Org/demo,rev1,0,pick,0,10,,4,true,human,\n",
        encoding="utf-8",
    )
    rc = main(
        [
            "distribution",
            "--dataset-root",
            str(ds_dir),
            "--task-spec",
            str(spec_path),
            "--annotations",
            str(ann_path),
            "--object-poses",
            str(tmp_path / "missing_sidecar.json"),
        ]
    )
    assert rc == 1
    assert "[object-poses]" in capsys.readouterr().err
