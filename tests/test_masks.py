import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from lerobot_check.calib import (
    ArmCalibration,
    HeadCalibration,
    backproject_pixels_to_plane_z,
    parse_intrinsics,
    project_base_points,
)
from lerobot_check.cli import main
from lerobot_check.dataset import DatasetView, EpisodeRecord, VideoSpan
from lerobot_check.masks import (
    closest_equivalent_yaw,
    make_bundle,
    make_local_outline,
    project_outline,
    read_head_frame,
    rotate_xy,
    template_anchor_layer,
)


@pytest.fixture
def head():
    def arm(name, x):
        transform = np.eye(4)
        transform[0, 3] = x
        transform[2, 3] = 2
        return ArmCalibration(name, transform, np.zeros(3), 4, 0, None, ())

    intrinsics = parse_intrinsics(
        {
            "version": 1,
            "camera": "head",
            "image_size": [640, 480],
            "fx": 520,
            "fy": 530,
            "cx": 320,
            "cy": 240,
            "distortion": [0.1, -0.02, 0.001, 0, 0],
        }
    )
    return HeadCalibration(
        "Org/demo",
        "rev",
        "head",
        (640, 480),
        intrinsics,
        {"right": arm("right", 0), "left": arm("left", 0.15)},
    )


def test_read_uses_episode_offset_and_native_size(monkeypatch):
    image = Image.new("RGB", (640, 480))
    decoder = Mock(return_value={15: image})
    monkeypatch.setattr("lerobot_check.masks.decode_video_frames_pyav", decoder)
    path = Path("shared.mp4")
    episode = EpisodeRecord(4, 30, Path("data.parquet"), {"head": VideoSpan(path, 23.5, 24.5)})
    dataset = Mock(spec=DatasetView, get_episode=Mock(return_value=episode), fps=30)
    assert read_head_frame(dataset, "head", 4, 15) is image
    decoder.assert_called_once_with(path, [(15, 24.0)], fps=30)
    with pytest.raises(ValueError, match="outside episode"):
        read_head_frame(dataset, "head", 4, 30)


def test_reprojects_reference_and_rotates_translates_with_distortion(head):
    pixels = np.array([[300, 202], [311, 196], [330, 207], [316, 219]])
    source = {"arm": "right", "raw_yaw_deg": 170, "x_m": -0.04, "y_m": -0.12}
    local = make_local_outline(pixels, source, 0.21, head)
    template = {"source": source, "outline_xy_m": local, "height_offset_m": 0.01}
    item = {**source, "z_m": 0.20, "yaw_deg": 170}
    np.testing.assert_allclose(project_outline(template, item, 360, head), pixels, atol=0.001)
    changed = {**item, "x_m": 0.1, "yaw_deg": 80, "arm": "left"}
    projected = np.array(project_outline(template, changed, 360, head))
    expected = rotate_xy(local, 80) + np.array([0.1, -0.12])
    actual = backproject_pixels_to_plane_z(
        projected, 0.21, head.arms["left"].T_cam_base, head.intrinsics
    )
    np.testing.assert_allclose(actual[:, :2], expected, atol=3e-6)


def test_yaw_symmetry_keeps_front_back_for_360_only():
    assert closest_equivalent_yaw(5, 170, 180) == 185
    assert closest_equivalent_yaw(5, 170, 360) == 5
    assert closest_equivalent_yaw(359, 1, 360) == -1
    assert closest_equivalent_yaw(90, 0, 180) == -90


def test_bundle_preserves_target_ids_and_explicit_failures(head):
    obj = {"object": "lid", "arm": "right", "z_plane_m": 0.2, "yaw_symmetry_deg": 180}

    def suggestion(x, tier):
        pixel, arrow_end_pixel = project_base_points(
            np.array([[x, 0, 0.2], [x + 0.05, 0, 0.2]]),
            head.arms["right"].T_cam_base,
            head.intrinsics,
        ).tolist()
        return {
            "object": "lid",
            "arm": "right",
            "tier": tier,
            "index": 1,
            "x_m": x,
            "y_m": 0,
            "z_m": 0.2,
            "yaw_deg": 0,
            "pixel": pixel,
            "arrow_end_pixel": arrow_end_pixel,
        }

    item = suggestion(0, "empty")
    suggestions = {
        "version": 1,
        "provenance": {
            "repo_id": "Org/demo",
            "revision": "rev",
            "position_point": "tool_point",
            "arrow_length_m": 0.05,
        },
        "objects": [obj],
        "placements": [item],
        "test_points": [suggestion(1.35, "dense")],
    }
    template = {
        "version": 1,
        "repo_id": "Org/demo",
        "revision": "rev",
        "object": "lid",
        "arm": "right",
        "z_plane_m": 0.2,
        "height_offset_m": 0.0,
        "calibration_sha256": "a",
        "source": {"raw_yaw_deg": 0},
        "outline_xy_m": [[0, 0], [0.1, 0], [0.01, 0.1]],
    }

    def bundle(templates):
        return make_bundle(
            suggestions, head, templates, calibration_sha256="a", suggestions_sha256="b"
        )

    good, outside = bundle([template])["targets"]
    assert good["kind"] == "placement" and good["outline"] is not None
    assert item["tier"] == "empty" and "tier" not in good
    assert good["arrow_end_pixel"] == item["arrow_end_pixel"]
    assert outside["kind"] == "test_point" and outside["outline"] is None
    assert outside["tier"] == "dense"
    assert "beyond" in outside["reason"]
    assert bundle([])["targets"][0]["reason"] == "No reviewed template for this object"
    assert (
        bundle([{**template, "reason": "Unreliable proxy"}])["targets"][0]["reason"]
        == "Unreliable proxy"
    )
    with pytest.raises(ValueError, match="another dataset or calibration"):
        bundle([{**template, "calibration_sha256": "old"}])
    with pytest.raises(ValueError, match="no longer matches suggestions"):
        bundle([{**template, "z_plane_m": 0.3}])
    item["arrow_end_pixel"] = [400, 240]
    with pytest.raises(ValueError, match=r"Suggestions do not match head\.json"):
        bundle([template])


def test_template_anchor_layer_checks_and_legacy_defaults(head):
    obj = {"object": "lid", "arm": "right", "z_plane_m": 0.2, "yaw_symmetry_deg": 180}
    pixel, arrow_end_pixel = project_base_points(
        np.array([[0, 0, 0.2], [0.05, 0, 0.2]]),
        head.arms["right"].T_cam_base,
        head.intrinsics,
    ).tolist()
    item = {
        "object": "lid",
        "arm": "right",
        "tier": "empty",
        "index": 1,
        "x_m": 0,
        "y_m": 0,
        "z_m": 0.2,
        "yaw_deg": 0,
        "pixel": pixel,
        "arrow_end_pixel": arrow_end_pixel,
    }

    def make_suggestions(layer):
        prov = {
            "repo_id": "Org/demo",
            "revision": "rev",
            "position_point": "tool_point",
            "arrow_length_m": 0.05,
        }
        if layer is not None:
            prov["layer"] = layer
        return {
            "version": 1,
            "provenance": prov,
            "objects": [obj],
            "placements": [item],
            "test_points": [],
        }

    legacy_template = {
        "version": 1,
        "repo_id": "Org/demo",
        "revision": "rev",
        "object": "lid",
        "arm": "right",
        "z_plane_m": 0.2,
        "height_offset_m": 0.0,
        "calibration_sha256": "a",
        "source": {"raw_yaw_deg": 0},
        "outline_xy_m": [[0, 0], [0.1, 0], [0.01, 0.1]],
    }
    object_template = {
        **legacy_template,
        "anchor_layer": "object",
        "source": {"raw_yaw_deg": 180, "folded_yaw_deg": 0},
    }
    tool_template = {**legacy_template, "anchor_layer": "tool"}

    assert template_anchor_layer(legacy_template) == "tool"
    assert template_anchor_layer(object_template) == "object"

    assert (
        make_bundle(
            make_suggestions("tool"),
            head,
            [legacy_template],
            calibration_sha256="a",
            suggestions_sha256="b",
        )["targets"][0]["outline"]
        is not None
    )
    assert (
        make_bundle(
            make_suggestions(None),
            head,
            [legacy_template],
            calibration_sha256="a",
            suggestions_sha256="b",
        )["targets"][0]["outline"]
        is not None
    )
    assert (
        make_bundle(
            make_suggestions("object"),
            head,
            [object_template],
            calibration_sha256="a",
            suggestions_sha256="b",
        )["targets"][0]["outline"]
        is not None
    )

    with pytest.raises(ValueError, match=r"Template for lid.*tool.*object"):
        make_bundle(
            make_suggestions("object"),
            head,
            [legacy_template],
            calibration_sha256="a",
            suggestions_sha256="b",
        )
    with pytest.raises(ValueError, match=r"Template for lid.*tool.*object"):
        make_bundle(
            make_suggestions("object"),
            head,
            [tool_template],
            calibration_sha256="a",
            suggestions_sha256="b",
        )
    with pytest.raises(ValueError, match=r"Template for lid.*object.*tool"):
        make_bundle(
            make_suggestions("tool"),
            head,
            [object_template],
            calibration_sha256="a",
            suggestions_sha256="b",
        )
    with pytest.raises(ValueError, match=r"Template for lid.*object.*tool"):
        make_bundle(
            make_suggestions("tool"),
            head,
            [{**object_template, "reason": "No mask", "outline_xy_m": None, "source": None}],
            calibration_sha256="a",
            suggestions_sha256="b",
        )


def test_object_layer_template_uses_folded_yaw_and_period_180(head):
    pixels = np.array([[300, 200], [345, 204], [338, 224], [308, 215]], dtype=np.float64)
    anchor = {
        "arm": "right",
        "x_m": -0.04,
        "y_m": -0.12,
        "raw_yaw_deg": 350.0,
        "folded_yaw_deg": 170.0,
    }
    local = make_local_outline(pixels, anchor, 0.21, head, layer="object")
    template = {
        "anchor_layer": "object",
        "source": anchor,
        "outline_xy_m": local.tolist(),
        "height_offset_m": 0.01,
    }

    roundtrip = {**anchor, "z_m": 0.20, "yaw_deg": 170.0}
    np.testing.assert_allclose(
        project_outline(template, roundtrip, 180.0, head), pixels, atol=0.001
    )

    target = {**anchor, "z_m": 0.20, "yaw_deg": 5.0}
    projected = np.array(project_outline(template, target, 180.0, head))
    actual = backproject_pixels_to_plane_z(
        projected, 0.21, head.arms["right"].T_cam_base, head.intrinsics
    )
    expected_180 = rotate_xy(local, 185.0) + np.array([anchor["x_m"], anchor["y_m"]])
    unexpected_360 = rotate_xy(local, 5.0) + np.array([anchor["x_m"], anchor["y_m"]])
    np.testing.assert_allclose(actual[:, :2], expected_180, atol=3e-5)
    assert not np.allclose(actual[:, :2], unexpected_360, atol=1e-3)


def test_mask_template_cli_records_anchor_layer_and_reject_layer(tmp_path, monkeypatch, head):
    image = Image.new("RGB", (640, 480))
    mask = np.zeros((480, 640), dtype=bool)
    mask[200:240, 300:360] = True
    monkeypatch.setattr("lerobot_check.mask_cli.load_head_calibration", lambda _p: head)
    monkeypatch.setattr("lerobot_check.mask_cli.resolve_dataset_root", lambda _r, _id: tmp_path)
    monkeypatch.setattr(
        "lerobot_check.mask_cli.DatasetView",
        lambda *_a, **_kw: Mock(revision="rev"),
    )
    monkeypatch.setattr("lerobot_check.mask_cli.read_head_frame", lambda *_a, **_kw: image)
    monkeypatch.setattr("lerobot_check.mask_cli.segment", lambda *_a, **_kw: mask)

    calib_path = tmp_path / "head.json"
    calib_path.write_text("{}")
    dist = {
        "version": 1,
        "provenance": {
            "repo_id": "Org/demo",
            "revision": "rev",
            "position_point": "tool_point",
            "calibrated_arms": ["right"],
        },
        "objects": [
            {
                "name": "phone",
                "arm": "right",
                "event": "grasp",
                "median_z_m": 0.15,
                "yaw_symmetry_deg": 360.0,
                "poses": [
                    {
                        "episode_index": 0,
                        "frame": 12,
                        "arm": "right",
                        "x_m": 0.0,
                        "y_m": 0.0,
                        "z_m": 0.15,
                        "raw_yaw_deg": 40.0,
                        "folded_yaw_deg": 40.0,
                    }
                ],
                "object_layer": {
                    "source": "head_mask",
                    "frame": 0,
                    "arm": "right",
                    "plane_z_m": 0.20,
                    "yaw_period_deg": 180.0,
                    "poses": [
                        {
                            "episode_index": 0,
                            "frame": 0,
                            "arm": "right",
                            "x_m": 0.02,
                            "y_m": -0.01,
                            "z_m": 0.20,
                            "raw_yaw_deg": 295.0,
                            "folded_yaw_deg": 115.0,
                            "length_m": 0.08,
                            "width_m": 0.05,
                        }
                    ],
                    "skipped": [],
                    "coverage": {},
                },
            }
        ],
    }
    dist_path = tmp_path / "dist.json"
    dist_path.write_text(json.dumps(dist))

    obj_out = tmp_path / "phone_obj.json"
    preview = tmp_path / "preview.png"
    rc = main(
        [
            "mask-template",
            "--distribution",
            str(dist_path),
            "--calib",
            str(calib_path),
            "--object",
            "phone",
            "--episode",
            "0",
            "--frame",
            "0",
            "--box",
            "295",
            "195",
            "365",
            "245",
            "--preview",
            str(preview),
            "--output",
            str(obj_out),
        ]
    )
    assert rc == 0
    obj_tpl = json.loads(obj_out.read_text())
    assert obj_tpl["anchor_layer"] == "object"
    assert obj_tpl["z_plane_m"] == 0.20
    assert obj_tpl["source"]["folded_yaw_deg"] == 115.0
    assert obj_tpl["source"]["x_m"] == 0.02

    tool_out = tmp_path / "phone_tool.json"
    rc = main(
        [
            "mask-template",
            "--distribution",
            str(dist_path),
            "--calib",
            str(calib_path),
            "--object",
            "phone",
            "--episode",
            "0",
            "--frame",
            "0",
            "--layer",
            "tool",
            "--box",
            "295",
            "195",
            "365",
            "245",
            "--preview",
            str(preview),
            "--output",
            str(tool_out),
        ]
    )
    assert rc == 0
    tool_tpl = json.loads(tool_out.read_text())
    assert tool_tpl["anchor_layer"] == "tool"
    assert tool_tpl["z_plane_m"] == 0.15
    assert tool_tpl["source"]["raw_yaw_deg"] == 40.0

    reject_obj = tmp_path / "reject_obj.json"
    rc = main(
        [
            "mask-template",
            "--distribution",
            str(dist_path),
            "--calib",
            str(calib_path),
            "--object",
            "phone",
            "--reject",
            "Occluded",
            "--output",
            str(reject_obj),
        ]
    )
    assert rc == 0
    assert json.loads(reject_obj.read_text())["anchor_layer"] == "object"

    reject_tool = tmp_path / "reject_tool.json"
    rc = main(
        [
            "mask-template",
            "--distribution",
            str(dist_path),
            "--calib",
            str(calib_path),
            "--object",
            "phone",
            "--layer",
            "tool",
            "--reject",
            "Occluded",
            "--output",
            str(reject_tool),
        ]
    )
    assert rc == 0
    assert json.loads(reject_tool.read_text())["anchor_layer"] == "tool"
