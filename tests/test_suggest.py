"""Unit tests for lerobot_check.suggest (polygon back-projection, sampling, 3-tier test points)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from lerobot_check.calib import (
    ArmCalibration,
    CameraIntrinsics,
    HeadCalibration,
    backproject_pixels_to_plane_z,
    project_base_points,
)
from lerobot_check.suggest import (
    ARROW_LENGTH_M,
    generate_suggestions,
    get_object_layer,
    point_in_polygon,
    write_suggestions_csv,
)


def _make_plausible_calibration(
    *,
    distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
) -> HeadCalibration:
    """Create a plausible overhead head camera looking down at a table near (1.0, 0.1, -0.35)."""
    intr = CameraIntrinsics(
        camera="observation.images.head",
        image_size=(640, 480),
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        distortion_model="inverse_brown_conrady",
        distortion=distortion,
        raw={
            "version": 1,
            "camera": "observation.images.head",
            "image_size": [640, 480],
            "fx": 600.0,
            "fy": 600.0,
            "cx": 320.0,
            "cy": 240.0,
            "distortion_model": "inverse_brown_conrady",
            "distortion": list(distortion),
        },
    )
    # Camera located at base (0.35, 0.05, 0.45), looking toward table center (1.00, 0.05, -0.35).
    cam_pos = np.array([0.35, 0.05, 0.45], dtype=np.float64)
    target = np.array([1.00, 0.05, -0.35], dtype=np.float64)
    z_axis = target - cam_pos
    z_axis /= np.linalg.norm(z_axis)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cross(z_axis, world_up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R_cb = np.vstack([x_axis, y_axis, z_axis])
    t_cb = -(R_cb @ cam_pos)
    T_cb = np.eye(4, dtype=np.float64)
    T_cb[:3, :3] = R_cb
    T_cb[:3, 3] = t_cb

    arm_cal = ArmCalibration(
        arm="right",
        T_cam_base=T_cb,
        tool_offset=np.zeros(3, dtype=np.float64),
        n_points=15,
        rms_px=0.5,
        loo_rms_px=0.7,
        residuals=(),
    )
    return HeadCalibration(
        repo_id="org/demo",
        revision="rev1",
        camera="observation.images.head",
        image_size=(640, 480),
        intrinsics=intr,
        arms={"right": arm_cal},
    )


def _make_distribution_with_tiers(
    *,
    position_point: str = "tool_point",
    calibrated_arms: list[str] | None = None,
) -> dict:
    """Create a distribution with a 3x3 occupied block (dense center, edge ring, empty out)."""
    xy_bin = 0.05
    z_val = -0.35
    poses = []
    ep = 0
    # Occupied cells: ix in {19, 20, 21} (0.95..1.10), iy in {0, 1, 2} (0.00..0.15).
    # Center (20, 1) gets 4 samples (>= sparse_threshold=3) -> interior DENSE.
    # Perimeter 8 cells get 1 sample each and border empty cells -> EDGE.
    for ix in (19, 20, 21):
        for iy in (0, 1, 2):
            reps = 4 if (ix, iy) == (20, 1) else 1
            for r in range(reps):
                poses.append(
                    {
                        "episode_index": ep,
                        "frame": 10,
                        "arm": "right",
                        "x_m": (ix + 0.5) * xy_bin,
                        "y_m": (iy + 0.5) * xy_bin,
                        "z_m": z_val,
                        "raw_yaw_deg": float((r * 30) % 180),
                        "folded_yaw_deg": float((r * 30) % 180),
                        "rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    }
                )
                ep += 1

    return {
        "version": 1,
        "provenance": {
            "repo_id": "org/demo",
            "revision": "rev1",
            "task_name": "demo",
            "annotations_path": None,
            "task_spec_path": None,
            "position_point": position_point,
            "calib_path": "/tmp/fake.head.json" if position_point == "tool_point" else None,
            "calib_mtime": None,
            "calibrated_arms": (
                ["right"]
                if calibrated_arms is None and position_point == "tool_point"
                else (calibrated_arms or [])
            ),
            "fps": 30.0,
            "annotator_counts": {"agent": 0, "human": ep, "converted": 0, "total": ep},
            "dataset_total_episodes": ep,
            "episode_filter": None,
            "annotated_episodes": list(range(ep)),
            "skipped_episodes": [],
        },
        "coverage_config": {
            "xy_bin_m": xy_bin,
            "yaw_bin_deg": 30.0,
            "sparse_threshold": 3,
        },
        "trajectory_samples": 10,
        "frame_shares": {"total_frames": 100, "subtasks": []},
        "episodes": [],
        "objects": [
            {
                "name": "box",
                "subtask": "place",
                "event": "grasp",
                "arm": "right",
                "yaw_symmetry_deg": 180.0,
                "median_z_m": z_val,
                "poses": poses,
                "skipped": [],
                "coverage": {
                    "sample_count": len(poses),
                    "xy_bin_m": xy_bin,
                    "yaw_bin_deg": 30.0,
                    "sparse_threshold": 3,
                    "num_yaw_bins": 6,
                    "ix_range": [19, 21],
                    "iy_range": [0, 2],
                    "bins": [],
                },
            }
        ],
        "subtasks": [],
    }


def test_point_in_polygon_convex_and_concave() -> None:
    square = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    assert point_in_polygon(1.0, 1.0, square)
    assert point_in_polygon(0.0, 1.0, square)  # boundary
    assert not point_in_polygon(2.5, 1.0, square)
    assert not point_in_polygon(-0.1, 0.5, square)

    # Concave L-shape: [0,2]x[0,2] minus (1,2]x(1,2].
    l_shape = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
    assert point_in_polygon(0.5, 1.5, l_shape)
    assert point_in_polygon(1.5, 0.5, l_shape)
    assert not point_in_polygon(1.5, 1.5, l_shape)


@pytest.mark.parametrize(
    "dist",
    [
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (-0.08, 0.03, 0.001, -0.002, 0.0),
    ],
)
def test_polygon_backprojection_round_trips_pixels(
    dist: tuple[float, float, float, float, float],
) -> None:
    cal = _make_plausible_calibration(distortion=dist)
    T = cal.arms["right"].T_cam_base
    z_plane = -0.35
    pixel_poly = np.array([[220.0, 170.0], [420.0, 170.0], [430.0, 310.0], [210.0, 310.0]])
    base_pts = backproject_pixels_to_plane_z(pixel_poly, z_plane, T, cal.intrinsics)
    assert np.allclose(base_pts[:, 2], z_plane)
    reproj = project_base_points(base_pts, T, cal.intrinsics)
    assert np.allclose(reproj, pixel_poly, atol=1e-4)


def test_suggest_with_polygon_prioritizes_empty_and_places_test_tiers(tmp_path: Path) -> None:
    cal = _make_plausible_calibration()
    dist = _make_distribution_with_tiers()
    T = cal.arms["right"].T_cam_base
    z_plane = -0.35

    # Build a pixel polygon covering ix in [18..22], iy in [-1..3] (so it includes
    # empty ring around the 3x3 occupied block, edge ring, and dense center (20, 1)).
    base_corners = np.array(
        [
            [0.90, -0.05, z_plane],
            [1.15, -0.05, z_plane],
            [1.15, 0.20, z_plane],
            [0.90, 0.20, z_plane],
        ],
        dtype=np.float64,
    )
    pixel_corners = project_base_points(base_corners, T, cal.intrinsics).tolist()
    targets = {
        "version": 1,
        "camera": "observation.images.head",
        "image_size": [640, 480],
        "polygon": pixel_corners,
    }

    result = generate_suggestions(
        dist,
        count=12,
        test_per_tier=3,
        targets=targets,
        calibration=cal,
        seed=42,
        layer="tool",
    )
    assert result["version"] == 1
    assert result["provenance"]["region_source"] == "target_polygon"
    assert len(result["placements"]) == 12
    assert len(result["test_points"]) == 9

    base_poly = result["objects"][0]["polygon_base_xy_m"]

    # Every placement is inside both base and pixel polygons, starts with empty cells,
    # spreads across all 6 yaw bins, and has valid pixel + 5 cm arrow projections.
    yaw_bins_seen = set()
    for item in result["placements"]:
        assert item["tier"] == "empty"
        assert item["cell_count"] == 0
        assert point_in_polygon(item["x_m"], item["y_m"], base_poly)
        assert item["pixel"] is not None and item["arrow_end_pixel"] is not None
        assert point_in_polygon(item["pixel"][0], item["pixel"][1], pixel_corners)
        yaw_bins_seen.add(item["yaw_bin"])

        # Verify arrow_end_pixel equals projection of pos + 5 cm along yaw.
        rad = math.radians(item["yaw_deg"])
        expected_arrow = project_base_points(
            np.array(
                [
                    [
                        item["x_m"] + ARROW_LENGTH_M * math.cos(rad),
                        item["y_m"] + ARROW_LENGTH_M * math.sin(rad),
                        item["z_m"],
                    ]
                ]
            ),
            T,
            cal.intrinsics,
        )[0]
        assert item["arrow_end_pixel"] == pytest.approx(expected_arrow.tolist(), abs=1e-6)

    assert yaw_bins_seen == {0, 1, 2, 3, 4, 5}

    # Verify each test tier lands in the exact right grid cells!
    occupied_block = {(ix, iy) for ix in (19, 20, 21) for iy in (0, 1, 2)}
    edge_ring = occupied_block - {(20, 1)}
    by_tier = {"dense": [], "edge": [], "empty": []}
    for tp in result["test_points"]:
        by_tier[tp["tier"]].append((tp["ix"], tp["iy"]))
        assert point_in_polygon(tp["x_m"], tp["y_m"], base_poly)
        assert tp["pixel"] is not None and tp["arrow_end_pixel"] is not None

    assert len(by_tier["dense"]) == 3
    assert all(cell == (20, 1) for cell in by_tier["dense"])
    assert len(by_tier["edge"]) == 3
    assert all(cell in edge_ring for cell in by_tier["edge"])
    assert len(by_tier["empty"]) == 3
    assert all(cell not in occupied_block for cell in by_tier["empty"])

    # CSV export includes header + 12 placements + 9 test points.
    csv_path = tmp_path / "suggestions.csv"
    write_suggestions_csv(result, csv_path)
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 + 12 + 9


def test_suggest_targets_without_calib_errors_and_bbox_fallback_warns() -> None:
    dist = _make_distribution_with_tiers()
    targets = {
        "version": 1,
        "camera": "observation.images.head",
        "image_size": [640, 480],
        "polygon": [[100.0, 100.0], [500.0, 100.0], [500.0, 400.0], [100.0, 400.0]],
    }
    # Explicit target polygon without calibration must error, never silently fall back.
    with pytest.raises(ValueError, match="without head-camera calibration"):
        generate_suggestions(dist, count=5, targets=targets, calibration=None, layer="tool")

    # Without targets and without calibration: warns, uses bbox + 5 cm, pixels are null.
    with pytest.warns(UserWarning, match="bounding box"):
        res = generate_suggestions(
            dist, count=6, test_per_tier=2, targets=None, calibration=None, layer="tool"
        )
    assert res["provenance"]["region_source"] == "data_bbox_plus_5cm"
    assert len(res["placements"]) == 6
    assert all(
        item["pixel"] is None and item["arrow_end_pixel"] is None for item in res["placements"]
    )
    assert all(
        item["pixel"] is None and item["arrow_end_pixel"] is None for item in res["test_points"]
    )


def test_suggest_cli_errors_on_targets_without_calib(tmp_path: Path) -> None:
    import json

    from lerobot_check.cli import main

    dist_path = tmp_path / "dist.json"
    dist_path.write_text(json.dumps(_make_distribution_with_tiers()), encoding="utf-8")
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "version": 1,
                "camera": "observation.images.head",
                "image_size": [640, 480],
                "polygon": [[100.0, 100.0], [500.0, 100.0], [500.0, 400.0], [100.0, 400.0]],
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "suggestions.json"
    rc = main(
        [
            "suggest",
            "--distribution",
            str(dist_path),
            "--targets",
            str(targets_path),
            "--layer",
            "tool",
            "-n",
            "5",
            "-o",
            str(out_path),
        ]
    )
    assert rc == 1
    assert not out_path.exists()


def test_suggest_refuses_tcp_distribution_with_calib_and_uses_tool_point_plane() -> None:
    from lerobot_check.analysis import compute_distribution_from_states
    from lerobot_check.dataset import ArmState
    from lerobot_check.schema import (
        AnnotationRow,
        CoverageSpec,
        ObjectSpec,
        SubtaskSpec,
        TaskSpec,
    )

    cal = _make_plausible_calibration()
    # Replace arm calibration with a realistic 18 cm tool offset along TCP +z.
    T = cal.arms["right"].T_cam_base
    tool_offset = np.array([0.01, -0.02, 0.18], dtype=np.float64)
    cal = HeadCalibration(
        repo_id=cal.repo_id,
        revision=cal.revision,
        camera=cal.camera,
        image_size=cal.image_size,
        intrinsics=cal.intrinsics,
        arms={
            "right": ArmCalibration(
                arm="right",
                T_cam_base=T,
                tool_offset=tool_offset,
                n_points=15,
                rms_px=0.5,
                loo_rms_px=0.7,
                residuals=(),
            )
        },
    )

    # Tilted gripper: 180 deg flip around X (pointing straight down) + 25 deg pitch ->
    # tool point sits below TCP in z and shifted in x/y.
    rx, ry = math.radians(180.0), math.radians(25.0)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    R_grip = Ry @ Rx
    rot6d = np.concatenate([R_grip[:, 0], R_grip[:, 1]])

    raw_tcp = np.array([1.02, 0.08, -0.18], dtype=np.float64)
    expected_tool = raw_tcp + R_grip @ tool_offset

    spec = TaskSpec(
        name="tool_plane_demo",
        subtasks=(SubtaskSpec(name="pick", prompt="Pick."),),
        objects=(ObjectSpec("box", "pick", "grasp", "right", 180.0),),
        coverage=CoverageSpec(xy_bin_m=0.05, yaw_bin_deg=30.0, sparse_threshold=2),
    )
    rows = [AnnotationRow("org/demo", "rev1", 0, "pick", 0, 5, None, 2, True, "human")]
    arm_states = {
        0: {
            "right": ArmState("right", np.tile(raw_tcp, (5, 1)), np.tile(rot6d, (5, 1)), np.ones(5))
        }
    }

    # Distribution without calibration uses raw TCP -> suggest with calibration must refuse.
    dist_tcp = compute_distribution_from_states(
        repo_id="org/demo",
        revision="rev1",
        fps=30.0,
        episode_lengths={0: 5},
        arm_states_by_episode=arm_states,
        rows=rows,
        task_spec=spec,
        trajectory_samples=4,
    )
    with pytest.raises(ValueError, match="raw 'tcp' positions"):
        generate_suggestions(
            dist_tcp, count=4, test_per_tier=1, targets=None, calibration=cal, layer="tool"
        )

    # Distribution with calibration uses tool point -> suggest back-projects onto tool-point z.
    dist_tool = compute_distribution_from_states(
        repo_id="org/demo",
        revision="rev1",
        fps=30.0,
        episode_lengths={0: 5},
        arm_states_by_episode=arm_states,
        rows=rows,
        task_spec=spec,
        trajectory_samples=4,
        tool_offsets={"right": tool_offset},
        calib_path="/tmp/fake.head.json",
    )
    z_tool = float(expected_tool[2])
    assert abs(z_tool - float(raw_tcp[2])) > 0.10
    base_corners = np.array(
        [
            [expected_tool[0] - 0.10, expected_tool[1] - 0.10, z_tool],
            [expected_tool[0] + 0.10, expected_tool[1] - 0.10, z_tool],
            [expected_tool[0] + 0.10, expected_tool[1] + 0.10, z_tool],
            [expected_tool[0] - 0.10, expected_tool[1] + 0.10, z_tool],
        ],
        dtype=np.float64,
    )
    targets = {
        "version": 1,
        "camera": "observation.images.head",
        "image_size": [640, 480],
        "polygon": project_base_points(base_corners, T, cal.intrinsics).tolist(),
    }
    res = generate_suggestions(
        dist_tool, count=5, test_per_tier=1, targets=targets, calibration=cal, layer="tool"
    )
    assert res["provenance"]["position_point"] == "tool_point"
    assert res["provenance"]["layer"] == "tool"
    assert res["objects"][0]["z_plane_m"] == pytest.approx(z_tool)
    assert all(p["z_m"] == pytest.approx(z_tool) for p in res["placements"])


def _attach_object_layer(
    obj: dict,
    *,
    plane_z_m: float = -0.42,
    yaw_period_deg: float = 180.0,
    poses: list[dict] | None = None,
) -> dict:
    if poses is None:
        poses = [
            {
                "episode_index": p["episode_index"],
                "frame": 0,
                "arm": obj["arm"],
                "x_m": p["x_m"] + 0.05,
                "y_m": p["y_m"],
                "z_m": plane_z_m,
                "raw_yaw_deg": float(p["raw_yaw_deg"]) + 180.0,
                "folded_yaw_deg": float(p["raw_yaw_deg"]) % yaw_period_deg,
                "length_m": 0.08,
                "width_m": 0.05,
            }
            for p in obj["poses"]
        ]
    obj["object_layer"] = {
        "source": "head_mask",
        "frame": 0,
        "arm": obj["arm"],
        "plane_z_m": plane_z_m,
        "yaw_period_deg": yaw_period_deg,
        "poses": poses,
        "skipped": [],
        "coverage": {
            "sample_count": len(poses),
            "xy_bin_m": 0.05,
            "yaw_bin_deg": 30.0,
            "sparse_threshold": 3,
            "num_yaw_bins": round(yaw_period_deg / 30.0),
            "ix_range": [20, 22],
            "iy_range": [0, 2],
            "bins": [],
        },
    }
    return obj


def test_get_object_layer_reads_both_layers_and_rejects_missing() -> None:
    dist = _make_distribution_with_tiers()
    obj = dist["objects"][0]
    obj["yaw_symmetry_deg"] = 360.0
    _attach_object_layer(obj, plane_z_m=-0.4254, yaw_period_deg=180.0)

    tool_data = get_object_layer(obj, "tool")
    assert tool_data.poses == obj["poses"]
    assert tool_data.plane_z_m == pytest.approx(-0.35)
    assert tool_data.yaw_period_deg == 360.0
    assert tool_data.coverage == obj["coverage"]

    obj_data = get_object_layer(obj, "object")
    assert obj_data.poses == obj["object_layer"]["poses"]
    assert obj_data.plane_z_m == pytest.approx(-0.4254)
    assert obj_data.yaw_period_deg == 180.0
    assert obj_data.coverage == obj["object_layer"]["coverage"]

    assert get_object_layer(obj) == obj_data

    without_layer = {k: v for k, v in obj.items() if k != "object_layer"}
    with pytest.raises(ValueError, match=r"box.*lerobot-check object-poses.*distribution"):
        get_object_layer(without_layer, "object")
    with pytest.raises(ValueError, match="Unsupported layer"):
        get_object_layer(obj, cast(Any, "invalid"))


def test_suggest_default_object_layer_errors_when_missing_naming_all_objects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from lerobot_check.cli import main

    dist = _make_distribution_with_tiers()
    second = {**dist["objects"][0], "name": "phone"}
    dist["objects"].append(second)

    with pytest.raises(ValueError, match=r"box, phone.*lerobot-check object-poses.*distribution"):
        generate_suggestions(dist, count=4, test_per_tier=1)

    dist_path = tmp_path / "dist.json"
    dist_path.write_text(json.dumps(dist), encoding="utf-8")
    out_path = tmp_path / "suggestions.json"
    rc = main(["suggest", "--distribution", str(dist_path), "-o", str(out_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "box, phone" in err
    assert "lerobot-check object-poses" in err
    assert "distribution" in err
    assert not out_path.exists()


def test_suggest_object_layer_uses_plane_poses_and_yaw_period() -> None:
    cal = _make_plausible_calibration()
    dist = _make_distribution_with_tiers()
    obj = dist["objects"][0]
    obj["yaw_symmetry_deg"] = 360.0
    _attach_object_layer(obj, plane_z_m=-0.4254, yaw_period_deg=180.0)
    T = cal.arms["right"].T_cam_base
    z_obj = -0.4254
    base_corners = np.array(
        [
            [0.95, -0.05, z_obj],
            [1.20, -0.05, z_obj],
            [1.20, 0.20, z_obj],
            [0.95, 0.20, z_obj],
        ],
        dtype=np.float64,
    )
    targets = {
        "version": 1,
        "camera": "observation.images.head",
        "image_size": [640, 480],
        "polygon": project_base_points(base_corners, T, cal.intrinsics).tolist(),
    }

    res = generate_suggestions(dist, count=12, test_per_tier=2, targets=targets, calibration=cal)
    assert res["provenance"]["layer"] == "object"
    assert res["objects"][0]["z_plane_m"] == pytest.approx(z_obj)
    assert res["objects"][0]["yaw_symmetry_deg"] == 180.0
    assert all(p["z_m"] == pytest.approx(z_obj) for p in res["placements"])
    assert all(0.0 <= p["yaw_deg"] < 180.0 for p in res["placements"])
    assert {p["yaw_bin"] for p in res["placements"]} == {0, 1, 2, 3, 4, 5}
    dense_cells = {(tp["ix"], tp["iy"]) for tp in res["test_points"] if tp["tier"] == "dense"}
    assert dense_cells == {(21, 1)}


def test_suggest_tool_layer_matches_existing_fixture_via_cli(tmp_path: Path) -> None:
    import json

    from lerobot_check.calib import write_json
    from lerobot_check.cli import main

    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    dist_fixture = fixtures_dir / "distribution.example.json"
    suggest_fixture = fixtures_dir / "suggestions.example.json"
    fixture_suggest = json.loads(suggest_fixture.read_text(encoding="utf-8"))

    intr_raw = {
        "version": 1,
        "camera": "observation.images.head",
        "image_size": [640, 480],
        "fx": 600.0,
        "fy": 600.0,
        "cx": 320.0,
        "cy": 240.0,
        "distortion_model": "inverse_brown_conrady",
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    cam_pos = np.array([0.35, 0.08, 0.45], dtype=np.float64)
    look_at = np.array([1.05, 0.08, -0.50], dtype=np.float64)
    z_ax = look_at - cam_pos
    z_ax /= np.linalg.norm(z_ax)
    x_ax = np.cross(z_ax, np.array([0.0, 0.0, 1.0]))
    x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)
    R_cb = np.vstack([x_ax, y_ax, z_ax])
    T_cb = np.eye(4, dtype=np.float64)
    T_cb[:3, :3] = R_cb
    T_cb[:3, 3] = -(R_cb @ cam_pos)

    calib_path = tmp_path / "head.json"
    write_json(
        {
            "version": 1,
            "repo_id": "Synthetic/demo",
            "revision": "synthetic0001",
            "camera": "observation.images.head",
            "image_size": [640, 480],
            "intrinsics": intr_raw,
            "arms": {
                "right": {
                    "T_cam_base": T_cb.tolist(),
                    "tool_offset": [0.0, 0.0, -0.15],
                    "n_points": 15,
                    "rms_px": 0.5,
                    "loo_rms_px": 0.7,
                    "residuals": [],
                }
            },
        },
        calib_path,
    )
    targets_path = tmp_path / "targets.json"
    write_json(
        {
            "version": 1,
            "camera": "observation.images.head",
            "image_size": [640, 480],
            "polygon": fixture_suggest["objects"][0]["polygon_pixel"],
        },
        targets_path,
    )
    out_path = tmp_path / "out.json"
    rc = main(
        [
            "suggest",
            "--distribution",
            str(dist_fixture),
            "--targets",
            str(targets_path),
            "--calib",
            str(calib_path),
            "--layer",
            "tool",
            "-n",
            "4",
            "--test-per-tier",
            "2",
            "--seed",
            "0",
            "-o",
            str(out_path),
        ]
    )
    assert rc == 0
    cli_out = json.loads(out_path.read_text(encoding="utf-8"))
    for key in ("version", "objects", "omitted_objects", "placements", "test_points"):
        assert cli_out[key] == fixture_suggest[key]
    assert cli_out["provenance"]["layer"] == "tool"
