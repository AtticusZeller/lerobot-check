from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from distribution_fixture import REPO_ID, REVISION, regression_distribution_kwargs
from PIL import Image

from lerobot_check.analysis import compute_distribution_from_states
from lerobot_check.calib import (
    ArmCalibration,
    HeadCalibration,
    parse_intrinsics,
    project_base_points,
)
from lerobot_check.object_pose import (
    SKIP_BORDER_TOUCH,
    SKIP_EMPTY_MASK,
    EpisodeObjectMeasurement,
    ObjectPoseMeasurement,
    apply_size_outlier_filter,
    build_object_poses_payload,
    evaluate_mask_for_episode,
    extract_dataset_object_poses,
    fold_angle_deg,
    measure_outline_pose,
    min_area_rect_pose_xy,
    object_yaw_period_deg,
)
from lerobot_check.schema import CoverageSpec, ObjectSpec, SubtaskSpec, TaskSpec, parse_task_spec

SRC = Path(__file__).resolve().parents[1] / "src" / "lerobot_check"
GOLDEN_SHA256_ALL = "85bcdec38bd1b2249f612de91e20aa7388b7b9200adb028f7127324a78c277ac"
GOLDEN_SHA256_FILTERED = "db7bc2b7b4ac24e4aca7babc992e293420c6411e302c2beb0c85c7d9776344b1"
IMAGE_SIZE = (640, 480)


def _transform(r: np.ndarray, cam_pos_base: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = -r @ cam_pos_base
    return t


def _rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _two_arm_calibration() -> HeadCalibration:
    intrinsics = parse_intrinsics(
        {
            "version": 1,
            "camera": "observation.images.head",
            "image_size": list(IMAGE_SIZE),
            "fx": 520.0,
            "fy": 515.0,
            "cx": 318.0,
            "cy": 243.0,
            "distortion_model": "inverse_brown_conrady",
            "distortion": [0.08, -0.05, 0.002, -0.001, 0.01],
        }
    )
    tilt = math.radians(25.0)
    r_cam_right = np.array(
        [
            [0.0, -1.0, 0.0],
            [-math.cos(tilt), 0.0, -math.sin(tilt)],
            [math.sin(tilt), 0.0, -math.cos(tilt)],
        ]
    )
    right_cam_pos = np.array([0.40, 0.0, 0.65])
    base_yaw = 35.0
    right_to_left = _rot_z(base_yaw)
    left_origin_in_right = np.array([0.05, 0.55, 0.12])
    r_cam_left = r_cam_right @ right_to_left
    left_cam_pos = right_to_left.T @ (right_cam_pos - left_origin_in_right)

    def arm(name: str, transform: np.ndarray) -> ArmCalibration:
        return ArmCalibration(
            arm=name,
            T_cam_base=transform,
            tool_offset=np.array([0.0, 0.0, 0.15]),
            n_points=10,
            rms_px=1.0,
            loo_rms_px=1.2,
            residuals=(),
        )

    return HeadCalibration(
        repo_id=REPO_ID,
        revision=REVISION,
        camera="observation.images.head",
        image_size=IMAGE_SIZE,
        intrinsics=intrinsics,
        arms={
            "right": arm("right", _transform(r_cam_right, right_cam_pos)),
            "left": arm("left", _transform(r_cam_left, left_cam_pos)),
        },
    )


def _rect_xy(cx: float, cy: float, length: float, width: float, yaw_deg: float) -> np.ndarray:
    hl, hw = length / 2, width / 2
    corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    rot = _rot_z(yaw_deg)[:2, :2]
    return corners @ rot.T + np.array([cx, cy])


def _project_rect(
    calib: HeadCalibration, arm: str, rect_xy: np.ndarray, plane_z: float
) -> np.ndarray:
    xyz = np.column_stack([rect_xy, np.full(len(rect_xy), plane_z)])
    return project_base_points(xyz, calib.arms[arm].T_cam_base, calib.intrinsics)


def _circular_diff_deg(a: float, b: float, period: float) -> float:
    d = (a - b) % period
    return min(d, period - d)


def test_backprojection_in_rotated_left_base_frame_with_plane_offset() -> None:
    calib = _two_arm_calibration()
    plane_z = -0.31
    rect = _rect_xy(0.62, -0.38, 0.15, 0.06, 72.0)
    outline_px = _project_rect(calib, "left", rect, plane_z)

    pose = measure_outline_pose(
        outline_px,
        plane_z_m=plane_z,
        arm_calibration=calib.arms["left"],
        intrinsics=calib.intrinsics,
        yaw_symmetry_deg=180.0,
    )
    assert (pose.x_m, pose.y_m, pose.z_m) == pytest.approx((0.62, -0.38, plane_z), abs=1e-5)
    assert pose.raw_yaw_deg == pytest.approx(72.0, abs=1e-4)
    assert (pose.length_m, pose.width_m) == pytest.approx((0.15, 0.06), abs=1e-5)

    same_pixels_in_right_frame = measure_outline_pose(
        outline_px,
        plane_z_m=plane_z - 0.12,
        arm_calibration=calib.arms["right"],
        intrinsics=calib.intrinsics,
        yaw_symmetry_deg=180.0,
    )
    assert _circular_diff_deg(same_pixels_in_right_frame.raw_yaw_deg, 72.0 + 35.0, 180.0) < 1e-3

    shifted = measure_outline_pose(
        outline_px,
        plane_z_m=plane_z - 0.03,
        arm_calibration=calib.arms["left"],
        intrinsics=calib.intrinsics,
        yaw_symmetry_deg=180.0,
    )
    assert shifted.z_m == pytest.approx(plane_z - 0.03)
    assert math.hypot(shifted.x_m - 0.62, shifted.y_m + 0.38) > 0.005


@pytest.mark.parametrize("yaw_deg", [0.0, 15.0, 44.0, 46.0, 89.0, 91.0, 135.0, 179.0])
def test_near_square_rectangle_yaw_follows_long_side(yaw_deg: float) -> None:
    raw, length, width = min_area_rect_pose_xy(_rect_xy(0.5, 0.1, 0.100, 0.098, yaw_deg))
    assert raw == pytest.approx(yaw_deg, abs=1e-6)
    assert (length, width) == pytest.approx((0.100, 0.098), abs=1e-9)


@pytest.mark.parametrize("yaw_deg", [0.0, 10.0, 45.0, 80.0, 100.0, 170.0])
def test_square_yaw_is_stable_modulo_ninety(yaw_deg: float) -> None:
    raw, length, width = min_area_rect_pose_xy(_rect_xy(0.5, 0.1, 0.08, 0.08, yaw_deg))
    assert length == pytest.approx(width)
    assert _circular_diff_deg(fold_angle_deg(raw, 90.0), yaw_deg % 90.0, 90.0) < 1e-6


@pytest.mark.parametrize(
    ("yaw_deg", "expected"),
    [(0.0, 0.0), (90.0, 90.0), (179.0, 179.0), (180.0, 0.0), (-1.0, 179.0), (-179.5, 0.5)],
)
def test_rectangle_raw_yaw_is_in_half_open_range(yaw_deg: float, expected: float) -> None:
    raw, _, _ = min_area_rect_pose_xy(_rect_xy(0.0, 0.0, 0.2, 0.05, yaw_deg))
    assert 0.0 <= raw < 180.0
    assert _circular_diff_deg(raw, expected, 180.0) < 1e-6


@pytest.mark.parametrize(
    ("symmetry", "period"), [(360.0, 180.0), (180.0, 180.0), (90.0, 90.0), (45.0, 45.0)]
)
def test_yaw_period_is_min_symmetry_180(symmetry: float, period: float) -> None:
    assert object_yaw_period_deg(symmetry) == period
    for yaw in (0.0, 30.0, 95.0, 179.999):
        folded = fold_angle_deg(yaw, period)
        assert 0.0 <= folded < period
        assert _circular_diff_deg(folded, yaw, period) < 1e-9


@pytest.mark.parametrize("symmetry", [120.0, 72.0, 100.0])
def test_yaw_period_rejects_symmetry_incompatible_with_rectangle_axis(symmetry: float) -> None:
    with pytest.raises(ValueError, match="180"):
        object_yaw_period_deg(symmetry)


def _ok_record(ep: int, length: float, width: float) -> EpisodeObjectMeasurement:
    return EpisodeObjectMeasurement(
        episode_index=ep,
        frame=0,
        status="ok",
        outline=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        pose=ObjectPoseMeasurement(0.5, 0.1, -0.4, 10.0, 10.0, length, width),
    )


@pytest.mark.parametrize("count", [1, 2])
def test_size_outlier_filter_keeps_records_when_too_few_samples(count: int) -> None:
    records = [_ok_record(0, 0.10, 0.05), _ok_record(1, 0.30, 0.15)][:count]
    assert apply_size_outlier_filter(records) == records


def test_size_outlier_filter_with_three_samples_skips_the_outlier() -> None:
    records = [_ok_record(0, 0.10, 0.05), _ok_record(1, 0.102, 0.051), _ok_record(2, 0.30, 0.15)]
    filtered = apply_size_outlier_filter(records)
    assert [r.status for r in filtered[:2]] == ["ok", "ok"]
    assert filtered[2].status.startswith("rectangle size") and filtered[2].pose is None
    assert filtered[2].outline == records[2].outline


def test_evaluate_mask_skip_reasons() -> None:
    calib = _two_arm_calibration()
    box = (200, 150, 400, 350)
    kwargs = {
        "layout_box": box,
        "plane_z_m": -0.4,
        "calibration": calib,
        "yaw_symmetry_deg": 180.0,
    }
    empty = np.zeros((480, 640), dtype=bool)
    assert evaluate_mask_for_episode(0, empty, arm_name="right", **kwargs).status == (
        SKIP_EMPTY_MASK
    )
    assert evaluate_mask_for_episode(0, None, arm_name="right", **kwargs).status == SKIP_EMPTY_MASK

    for touching in ((slice(150, 200), slice(250, 300)), (slice(300, 349), slice(380, 400))):
        mask = np.zeros((480, 640), dtype=bool)
        mask[touching] = True
        result = evaluate_mask_for_episode(1, mask, arm_name="right", **kwargs)
        assert result.status == SKIP_BORDER_TOUCH
        assert result.outline is not None and result.pose is None

    inside = np.zeros((480, 640), dtype=bool)
    inside[200:260, 250:330] = True
    assert evaluate_mask_for_episode(2, inside, arm_name="right", **kwargs).status == "ok"
    missing = evaluate_mask_for_episode(
        3, inside, arm_name="right", **{**kwargs, "calibration": _single_arm(calib, "left")}
    )
    assert missing.status == "missing calibration for arm 'right'"
    assert missing.pose is None


def _single_arm(calib: HeadCalibration, arm: str) -> HeadCalibration:
    return HeadCalibration(
        repo_id=calib.repo_id,
        revision=calib.revision,
        camera=calib.camera,
        image_size=calib.image_size,
        intrinsics=calib.intrinsics,
        arms={arm: calib.arms[arm]},
    )


def _layout_spec(*objects: ObjectSpec) -> TaskSpec:
    return TaskSpec(
        name="demo",
        subtasks=(SubtaskSpec(name="pick", prompt="Pick."),),
        objects=objects,
        coverage=CoverageSpec(xy_bin_m=0.05, yaw_bin_deg=30.0, sparse_threshold=2),
    )


class _RasterSegmenter:
    def __init__(self, polygons_by_episode: dict[int, dict[tuple[int, ...], np.ndarray]]):
        self.polygons_by_episode = polygons_by_episode
        self.calls: list[int] = []

    def segment_layout_boxes(self, image: Image.Image, boxes) -> list[np.ndarray]:
        ep = int(np.asarray(image)[0, 0, 0])
        self.calls.append(ep)
        masks = []
        for box in boxes:
            mask = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8)
            polygon = self.polygons_by_episode[ep].get(tuple(box))
            if polygon is not None:
                cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
            masks.append(mask.astype(bool))
        return masks


def _bbox_with_margin(px: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    return (
        int(px[:, 0].min()) - margin,
        int(px[:, 1].min()) - margin,
        int(px[:, 0].max()) + margin,
        int(px[:, 1].max()) + margin,
    )


def test_extract_end_to_end_recovers_poses_and_marks_uncalibrated_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calib = _two_arm_calibration()
    plane_z = -0.40
    truths = {0: (0.80, 0.05, 20.0), 1: (0.81, 0.06, 50.0), 2: (0.79, 0.04, 170.0)}
    polygons = {
        ep: _project_rect(calib, "right", _rect_xy(x, y, 0.14, 0.07, yaw), plane_z)
        for ep, (x, y, yaw) in truths.items()
    }
    box = (
        min(_bbox_with_margin(p, 20)[0] for p in polygons.values()),
        min(_bbox_with_margin(p, 20)[1] for p in polygons.values()),
        max(_bbox_with_margin(p, 20)[2] for p in polygons.values()),
        max(_bbox_with_margin(p, 20)[3] for p in polygons.values()),
    )
    phone = ObjectSpec("phone", "pick", "grasp", "right", 360.0, layout_box=box)
    lid = ObjectSpec("lid", "pick", "grasp", "left", 180.0, layout_box=(10, 10, 60, 60))
    segmenter = _RasterSegmenter({ep: {box: poly} for ep, poly in polygons.items()})
    images = {ep: Image.new("RGB", IMAGE_SIZE, color=(ep, 0, 0)) for ep in truths}
    monkeypatch.setattr(
        "lerobot_check.object_pose.decode_frame_zero_images", lambda *_: dict(images)
    )
    dataset = SimpleNamespace(repo_id=REPO_ID, revision=REVISION)

    payload, _ = extract_dataset_object_poses(
        dataset,  # ty: ignore[invalid-argument-type]
        _layout_spec(phone, lid),
        _single_arm(calib, "right"),
        calibration_sha256="a" * 64,
        episodes=sorted(truths),
        plane_z_by_object={"phone": plane_z, "lid": None},
        segmenter=segmenter,  # ty: ignore[invalid-argument-type]
    )
    assert segmenter.calls == [0, 1, 2]
    for entry in payload["objects"]["phone"]["episodes"]:
        x, y, yaw = truths[entry["episode_index"]]
        assert entry["status"] == "ok"
        assert math.hypot(entry["x_m"] - x, entry["y_m"] - y) < 0.004
        assert _circular_diff_deg(entry["folded_yaw_deg"], yaw, 180.0) < 2.0
    assert {e["status"] for e in payload["objects"]["lid"]["episodes"]} == {
        "missing calibration for arm 'left'"
    }


def test_extract_rejects_layout_box_outside_image() -> None:
    calib = _two_arm_calibration()
    obj = ObjectSpec("phone", "pick", "grasp", "right", 180.0, layout_box=(600, 10, 700, 60))
    with pytest.raises(ValueError, match="exceeds calibrated image size"):
        extract_dataset_object_poses(
            SimpleNamespace(repo_id=REPO_ID, revision=REVISION),  # ty: ignore[invalid-argument-type]
            _layout_spec(obj),
            calib,
            calibration_sha256="a" * 64,
            episodes=[0],
            plane_z_by_object={"phone": -0.4},
            segmenter=_RasterSegmenter({}),  # ty: ignore[invalid-argument-type]
        )


def _distribution_text(**overrides) -> str:
    kwargs = {**regression_distribution_kwargs(), **overrides}
    payload = compute_distribution_from_states(**kwargs)
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_distribution_without_sidecar_is_byte_identical_to_pre_object_layer_output() -> None:
    assert _sha(_distribution_text()) == GOLDEN_SHA256_ALL
    assert _sha(_distribution_text(calib_sha256="f" * 64)) == GOLDEN_SHA256_ALL
    assert _sha(_distribution_text(episode_filter=[0, 1, 2])) == GOLDEN_SHA256_FILTERED


def _sidecar(spec: TaskSpec, records: dict[str, list[EpisodeObjectMeasurement]]) -> dict:
    return build_object_poses_payload(
        repo_id=REPO_ID,
        revision=REVISION,
        camera="observation.images.head",
        image_size=IMAGE_SIZE,
        calibration_sha256="a" * 64,
        objects=spec.objects,
        plane_z_by_object={name: -0.4 for name in records},
        measurements_by_object=records,
    )


def _layout_regression_spec() -> TaskSpec:
    base = regression_distribution_kwargs()["task_spec"]
    phone, box = base.objects
    return TaskSpec(
        name=base.name,
        subtasks=base.subtasks,
        objects=(
            ObjectSpec(**{**phone.__dict__, "layout_box": (100, 100, 200, 200)}),
            ObjectSpec(**{**box.__dict__, "layout_box": (300, 100, 400, 200)}),
        ),
        coverage=base.coverage,
    )


def _pose_record(ep: int, raw_yaw: float, x: float = 0.8) -> EpisodeObjectMeasurement:
    return EpisodeObjectMeasurement(
        episode_index=ep,
        frame=0,
        status="ok",
        outline=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        pose=ObjectPoseMeasurement(x, 0.1, -0.4, raw_yaw, raw_yaw % 180.0, 0.14, 0.07),
    )


def _distribution_with_sidecar(sidecar: dict, **overrides) -> dict:
    kwargs = {
        **regression_distribution_kwargs(),
        "task_spec": _layout_regression_spec(),
        "calib_sha256": "a" * 64,
        "object_poses": sidecar,
        "object_poses_sha256": "b" * 64,
        **overrides,
    }
    return compute_distribution_from_states(**kwargs)


def test_object_layer_coverage_uses_yaw_period_not_symmetry() -> None:
    spec = _layout_regression_spec()
    sidecar = _sidecar(spec, {"phone": [_pose_record(0, 170.0), _pose_record(1, 5.0)]})
    phone = _distribution_with_sidecar(sidecar)["objects"][0]
    assert phone["yaw_symmetry_deg"] == 360.0
    assert phone["coverage"]["num_yaw_bins"] == 12
    layer_cov = phone["object_layer"]["coverage"]
    assert layer_cov["num_yaw_bins"] == 6
    occupied = {(b["yaw_bin"], b["yaw_min_deg"], b["yaw_max_deg"]) for b in layer_cov["bins"]}
    assert max(b[2] for b in occupied) == pytest.approx(180.0)
    counts = {b["yaw_bin"]: b["count"] for b in layer_cov["bins"] if b["count"]}
    assert counts == {5: 1, 0: 1}


def test_object_layer_reports_considered_episodes_missing_from_sidecar() -> None:
    spec = _layout_regression_spec()
    sidecar = _sidecar(spec, {"phone": [_pose_record(0, 10.0)], "box": [_pose_record(1, 20.0)]})
    dist = _distribution_with_sidecar(sidecar, episode_filter=[0, 1, 3])
    phone_layer = dist["objects"][0]["object_layer"]
    assert [p["episode_index"] for p in phone_layer["poses"]] == [0]
    assert phone_layer["skipped"] == [
        {"episode_index": 1, "reason": "episode not in object-poses sidecar"},
        {"episode_index": 3, "reason": "episode not in object-poses sidecar"},
    ]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda s: s["objects"]["phone"]["episodes"][0].update(x_m=None), "x_m"),
        (lambda s: s["objects"]["phone"]["episodes"][0].pop("raw_yaw_deg"), "raw_yaw_deg"),
        (lambda s: s["objects"]["phone"].pop("arm"), "arm"),
        (lambda s: s["objects"]["phone"].update(episodes="nope"), "episodes"),
        (lambda s: s["objects"]["phone"]["episodes"][0].pop("status"), "status"),
        (lambda s: s["objects"]["phone"].update(arm="left"), "arm mismatch"),
    ],
)
def test_malformed_sidecar_raises_value_error(mutate, match: str) -> None:
    spec = _layout_regression_spec()
    sidecar = _sidecar(spec, {"phone": [_pose_record(0, 10.0)]})
    mutate(sidecar)
    with pytest.raises(ValueError, match=match):
        _distribution_with_sidecar(sidecar)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("repo_id", "Other/ds", "repo_id/revision"),
        ("revision", "other", "repo_id/revision"),
        ("calibration_sha256", "c" * 64, "calibration_sha256"),
        ("version", 2, "version"),
    ],
)
def test_sidecar_identity_mismatch_is_rejected(field: str, value: object, match: str) -> None:
    spec = _layout_regression_spec()
    sidecar = _sidecar(spec, {"phone": [_pose_record(0, 10.0)]})
    sidecar[field] = value
    with pytest.raises(ValueError, match=match):
        _distribution_with_sidecar(sidecar)


@pytest.mark.parametrize(
    "extra",
    [
        {"layout_box": [0, 0, 10, 10, 20]},
        {"layout_box": [5, 5, 5, 10]},
        {"layout_box": [5, 5, 10, 5]},
        {"layout_box": [5.0, 5, 10, 10]},
        {"layout_box": None, "object_plane_z_m": [1.0]},
        {"object_plane_z_m": "-0.4"},
    ],
)
def test_task_spec_rejects_bad_object_layer_fields(extra: dict[str, object]) -> None:
    raw = {
        "name": "t",
        "subtasks": [{"name": "s", "prompt": "p"}],
        "objects": [
            {"name": "o", "subtask": "s", "event": "grasp", "arm": "right", "yaw_symmetry_deg": 180}
            | extra
        ],
    }
    with pytest.raises(ValueError, match=r"objects\[0\]"):
        parse_task_spec(raw)


def test_task_spec_accepts_integer_plane_z() -> None:
    raw = {
        "name": "t",
        "subtasks": [{"name": "s", "prompt": "p"}],
        "objects": [
            {
                "name": "o",
                "subtask": "s",
                "event": "grasp",
                "arm": "right",
                "yaw_symmetry_deg": 180,
                "layout_box": [0, 0, 10, 10],
                "object_plane_z_m": 0,
            }
        ],
    }
    obj = parse_task_spec(raw).objects[0]
    assert obj.layout_box == (0, 0, 10, 10)
    assert obj.object_plane_z_m == 0.0


_COLOUR_CALLS = {
    "threshold",
    "adaptiveThreshold",
    "inRange",
    "cvtColor",
    "equalizeHist",
    "convert",
    "getextrema",
    "point",
}


@pytest.mark.parametrize("module", ["segmentation.py", "object_pose.py"])
def test_segmentation_path_has_no_pixel_intensity_heuristics(module: str) -> None:
    tree = ast.parse((SRC / module).read_text(encoding="utf-8"))
    segmentation_functions = {
        "segmentation.py": {"segment_layout_boxes", "_interior_grid_points", "_mask_touches_box"},
        "object_pose.py": {
            "evaluate_mask_for_episode",
            "mask_touches_layout_box",
            "extract_dataset_object_poses",
            "measure_outline_pose",
        },
    }[module]
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in segmentation_functions:
            for call in ast.walk(node):
                if isinstance(call, ast.Attribute) and call.attr in _COLOUR_CALLS:
                    found.append((node.name, call.attr))
    assert found == []
