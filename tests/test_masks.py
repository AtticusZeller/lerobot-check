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
from lerobot_check.dataset import DatasetView, EpisodeRecord, VideoSpan
from lerobot_check.masks import (
    closest_equivalent_yaw,
    make_bundle,
    make_local_outline,
    project_outline,
    read_head_frame,
    rotate_xy,
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
    assert good["arrow_end_pixel"] == item["arrow_end_pixel"]
    assert outside["kind"] == "test_point" and outside["outline"] is None
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
