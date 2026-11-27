import json
import math
import warnings
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from lerobot_check.calib import ArmCalibration, HeadCalibration, parse_intrinsics
from lerobot_check.cli import main
from lerobot_check.collection_planner import (
    _candidates,
    _coverage_history,
    _placement_candidate,
    _planning_history,
    _select_round,
    build_plan,
    convex_hulls_overlap,
    delete_pending,
)
from lerobot_check.collection_progress import validate_plan
from lerobot_check.masks import sha256_bytes

SHA = "a" * 64


@pytest.fixture
def calibrated_inputs():
    camera = "head"
    intrinsics = parse_intrinsics(
        {
            "version": 1,
            "camera": camera,
            "image_size": [640, 480],
            "fx": 325,
            "fy": 325,
            "cx": 320,
            "cy": 240,
            "distortion": [0, 0, 0, 0, 0],
            "serial_number": "0032",
        }
    )
    identity = np.eye(4)
    identity[2, 3] = 1
    right = ArmCalibration("right", identity, np.zeros(3), 10, 0.5, None, ())
    shifted = identity.copy()
    shifted[0, 3] = 0.15
    left = replace(right, arm="left", T_cam_base=shifted)
    head = HeadCalibration(
        "org/demo", "local", camera, (640, 480), intrinsics, {"right": right, "left": left}
    )
    objects = []
    templates = []
    for name, arm, shift in [
        ("lid", "right", 0),
        ("phone", "left", -0.15),
        ("box_base", "right", 0),
    ]:
        objects.append(
            {
                "name": name,
                "arm": arm,
                "median_z_m": 0,
                "yaw_symmetry_deg": 180,
                "poses": [
                    {"episode_index": 0, "x_m": -0.22 + shift, "y_m": -0.22, "folded_yaw_deg": 0},
                    {"episode_index": 1, "x_m": 0.22 + shift, "y_m": 0.22, "folded_yaw_deg": 0},
                ],
            }
        )
        templates.append(
            {
                "object": name,
                "arm": arm,
                "z_plane_m": 0,
                "height_offset_m": 0.01,
                "version": 1,
                "repo_id": "org/demo",
                "revision": "local",
                "calibration_sha256": SHA,
                "source": {"raw_yaw_deg": 0},
                "outline_xy_m": [[-0.04, -0.03], [0.04, -0.03], [0.04, 0.03], [-0.04, 0.03]],
                "reason": "No reliable pose proxy" if name == "box_base" else None,
            }
        )
    distribution = {
        "provenance": {
            "repo_id": "org/demo",
            "revision": "local",
            "task_name": "demo",
            "position_point": "tool_point",
        },
        "coverage_config": {"xy_bin_m": 0.05, "yaw_bin_deg": 30, "sparse_threshold": 3},
        "objects": objects,
    }
    return distribution, head, templates


def make_plan(inputs, *, round_count=4, objects_per_round=2, **options):
    distribution, head, templates = inputs
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No target polygon provided")
        return build_plan(
            distribution,
            head,
            calibration_sha256=SHA,
            templates=templates,
            source_episode_count=2,
            round_count=round_count,
            objects_per_round=objects_per_round,
            **options,
        )


def test_hull_overlap_includes_touch_and_containment():
    a = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert convex_hulls_overlap(a, a + np.array([0.5, 0]))
    assert convex_hulls_overlap(a, a + np.array([1, 1]))
    assert convex_hulls_overlap(a, a * 0.25 + 0.3)
    assert not convex_hulls_overlap(a, a + np.array([1.01, 0]))


def test_mixed_arm_plan_is_deterministic_and_excludes_unchecked_base(calibrated_inputs):
    plan, bundle = make_plan(calibrated_inputs)
    repeated, second_bundle = make_plan(calibrated_inputs)
    validate_plan(plan)
    assert plan["camera_serial"] == "0032"
    assert plan["rounds"] == repeated["rounds"] and bundle == second_bundle
    assert plan["mask_bundle_sha256"] == sha256_bytes(bundle)
    assert plan["excluded_objects"] == {"box_base": "No reliable pose proxy"}
    for r in plan["rounds"]:
        a, b = r["targets"]
        assert {a["arm"], b["arm"]} == {"right", "left"}
        assert a["outline_plane_z_m"] == b["outline_plane_z_m"] == 0.01
        assert not convex_hulls_overlap(np.array(a["outline"]), np.array(b["outline"]))
    with pytest.raises(ValueError, match=r"only 2 are eligible.*box_base"):
        make_plan(calibrated_inputs, objects_per_round=3)
    single, _ = make_plan(calibrated_inputs, objects_per_round=1, round_count=3)
    base = next(t for r in single["rounds"] for t in r["targets"] if t["object"] == "box_base")
    assert base["outline"] is None and base["outline_plane_z_m"] is None and base["reason"]


def test_generate_reports_all_missing_templates(tmp_path, monkeypatch, capsys, calibrated_inputs):
    distribution, head, templates = calibrated_inputs
    monkeypatch.setattr(
        "lerobot_check.collection_plan_cli.parse_head_calibration", lambda *_a, **_kw: head
    )
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text('{"total_episodes": 2}')
    dist_path = tmp_path / "distribution.json"
    dist_path.write_text(json.dumps({"version": 1, **distribution}))
    calib_path = tmp_path / "head.json"
    calib_path.write_text("{}")
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "box_base.json").write_text(json.dumps(templates[-1]))
    plan_path = tmp_path / "plan.json"

    assert (
        main(
            [
                "collection-plan",
                "generate",
                "--dataset-root",
                str(tmp_path),
                "--repo-id",
                head.repo_id,
                "--viewer-root",
                str(tmp_path),
                "--distribution",
                str(dist_path),
                "--calib",
                str(calib_path),
                "--template-dir",
                str(template_dir),
                "--plan",
                str(plan_path),
                "--rounds",
                "3",
            ]
        )
        == 1
    )
    message = "[collection-plan] Missing reviewed templates for: lid, phone\n"
    assert capsys.readouterr().err == message
    assert not plan_path.exists()
    assert not (tmp_path / "bundles").exists()


def test_object_without_placement_poses_still_requires_a_template(calibrated_inputs):
    distribution, head, templates = calibrated_inputs
    distribution["objects"][0]["poses"] = []
    with pytest.raises(ValueError, match="Missing reviewed templates for: lid"):
        make_plan((distribution, head, templates[1:]), objects_per_round=1)
    plan, _ = make_plan(calibrated_inputs, objects_per_round=1)
    assert plan["excluded_objects"] == {"lid": "No calibrated placement candidates"}


def test_no_collision_solution_reports_shortage(calibrated_inputs):
    distribution, head, templates = calibrated_inputs
    for template in templates:
        template["outline_xy_m"] = [[-0.32, -0.32], [0.32, -0.32], [0.32, 0.32], [-0.32, 0.32]]
    target = {
        "camera": "head",
        "image_size": [640, 480],
        "polygon": [[315, 235], [325, 235], [325, 245], [315, 245]],
    }
    with pytest.raises(ValueError, match="Found 0/4"):
        make_plan((distribution, head, templates), targets=target)


def test_history_augments_coverage_without_double_counting(calibrated_inputs):
    distribution, head, templates = calibrated_inputs
    old, _ = make_plan(calibrated_inputs)
    done, skipped = old["rounds"][:2]
    done["status"], done["episode_index"] = "collected", 2
    skipped["status"] = "skipped"
    first = done["targets"][0]
    real_pose = {
        "episode_index": 2,
        "x_m": first["x_m"] + 0.06,
        "y_m": first["y_m"],
        "folded_yaw_deg": 1.0,
    }
    distribution["objects"][0]["poses"].append(real_pose)
    extra, avoid = _coverage_history(distribution, old["rounds"])
    assert set(extra) == {done["targets"][1]["object"]}
    assert avoid[first["object"]] == [
        (first["x_m"], first["y_m"]),
        (real_pose["x_m"], real_pose["y_m"]),
    ]
    assert skipped["targets"][0]["x_m"] not in {x for x, _ in avoid[first["object"]]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, _, baseline_bundle, _ = _candidates(distribution, head, templates, SHA, None, {}, 4, 0)
        _, _, augmented_bundle, _ = _candidates(
            distribution, head, templates, SHA, None, extra, 4, 0
        )
    assert sha256_bytes(baseline_bundle) != sha256_bytes(augmented_bundle)
    updated, _ = make_plan(calibrated_inputs, previous=old)
    assert updated["rounds"][:2] == [done, skipped]
    for r in updated["rounds"][2:]:
        for t in r["targets"]:
            assert all(
                math.hypot(t["x_m"] - x, t["y_m"] - y) >= 0.025 for x, y in avoid[t["object"]]
            )


def test_placement_candidate_requires_rejection_for_missing_outline(calibrated_inputs):
    _, head, templates = calibrated_inputs
    item = {
        "object": "lid",
        "index": 1,
        "arm": "right",
        "x_m": 0,
        "y_m": 0,
        "z_m": 0,
        "yaw_deg": 0,
        "pixel": [320, 240],
        "arrow_end_pixel": [332, 240],
    }
    mask = {"outline": None, "reason": "Projection failed"}
    args = (180, head.image_size, np.eye(4))
    assert _placement_candidate(item, mask, templates[0], *args) is None

    reason = "No reliable pose proxy"
    rejected = {**templates[0], "reason": reason}
    manual_mask = {"outline": None, "reason": reason}
    manual = _placement_candidate(item, manual_mask, rejected, *args)
    assert manual is not None and manual.target["reason"] == reason
    assert manual.base_hull is None and manual.image_hull is None
    assert _placement_candidate({**item, "pixel": [640, 240]}, manual_mask, rejected, *args) is None


def test_replacement_history_counts_other_pending_rounds_but_avoids_all(calibrated_inputs):
    distribution, _, _ = calibrated_inputs
    original, _ = make_plan(calibrated_inputs, objects_per_round=1)
    done, skipped, replacing, other = original["rounds"]
    done["status"], done["episode_index"] = "collected", 2
    skipped["status"] = "skipped"
    original["settings"]["min_revisit_distance_m"] = 0.11

    history = _planning_history(distribution, original, replacing["id"])

    assert history.distance == 0.11
    assert history.kept == [done, skipped]
    assert history.usage == Counter(r["targets"][0]["object"] for r in (done, other))
    expected = {
        (target["object"], target["x_m"], target["y_m"])
        for round_ in (done, replacing, other)
        for target in round_["targets"]
    }
    assert {(name, x, y) for name, poses in history.avoid.items() for x, y in poses} == expected


def test_round_selection_respects_revisit_usage_and_collision(calibrated_inputs):
    distribution, head, templates = calibrated_inputs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pools, _, _, _ = _candidates(distribution, head, templates, SHA, None, {}, 4, 0)
    lid, phone = pools["lid"], pools["phone"]
    order = {"lid": 0, "phone": 1}
    first_xy = (lid[0].target["x_m"], lid[0].target["y_m"])
    avoid = {"lid": [first_xy]}
    available = {"lid": lid, "phone": phone}
    selected = _select_round(available, avoid, Counter({"lid": 1}), order, 0.025, 1)
    assert selected == [phone[0].target]
    assert _select_round({"lid": lid[:1]}, avoid, Counter(), order, 0.025, 1) is None

    colliding = replace(phone[0], image_hull=lid[0].image_hull)
    assert _select_round({"lid": lid[:1], "phone": [colliding]}, {}, Counter(), order, 0, 2) is None


def test_replacing_or_deleting_one_pending_round_keeps_all_others(calibrated_inputs):
    original, _ = make_plan(calibrated_inputs)
    original["rounds"][0]["status"] = "skipped"
    updated, _ = make_plan(calibrated_inputs, previous=original, replace_round_id=2)
    assert updated["rounds"][1]["id"] == original["next_round_id"]
    assert updated["rounds"][0] == original["rounds"][0]
    assert updated["rounds"][2:] == original["rounds"][2:]
    old_round = original["rounds"][1]
    for target in updated["rounds"][1]["targets"]:
        prior = next(t for t in old_round["targets"] if t["object"] == target["object"])
        assert math.hypot(target["x_m"] - prior["x_m"], target["y_m"] - prior["y_m"]) >= 0.025
    with pytest.raises(ValueError, match="no longer pending"):
        delete_pending(updated, 1)
    delete_pending(updated, 3)
    assert updated["rounds"][2:] == original["rounds"][3:]
    assert updated["generation"] == 3
    validate_plan(updated)
