from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from lerobot_check.calib import HeadCalibration
from lerobot_check.collection_progress import (
    CollectionPlan,
    PlanRound,
    PlanSettings,
    PlanTarget,
    json_bytes,
)
from lerobot_check.masks import (
    make_bundle,
    outline_base_points,
    sha256_bytes,
    template_anchor_layer,
)
from lerobot_check.suggest import (
    DEFAULT_POSE_LAYER,
    PoseLayer,
    generate_suggestions,
    get_object_layer,
    require_object_layers,
    validate_pose_layer,
)


@dataclass(frozen=True)
class _Candidate:
    target: PlanTarget
    base_hull: np.ndarray | None
    image_hull: np.ndarray | None


def _hull(points: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2)
    if len(hull) < 3 or cv2.contourArea(hull) <= 0:
        raise ValueError("Degenerate object outline")
    return hull


def convex_hulls_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    for hull in (a, b):
        edges = np.roll(hull, -1, axis=0) - hull
        axes = np.column_stack((-edges[:, 1], edges[:, 0]))
        projections_a, projections_b = a @ axes.T, b @ axes.T
        if np.any(
            (projections_a.max(axis=0) < projections_b.min(axis=0))
            | (projections_b.max(axis=0) < projections_a.min(axis=0))
        ):
            return False
    return True


def _collides(a: _Candidate, b: _Candidate) -> bool:
    if a.base_hull is None or b.base_hull is None or a.image_hull is None or b.image_hull is None:
        return True
    return convex_hulls_overlap(a.base_hull, b.base_hull) or convex_hulls_overlap(
        a.image_hull, b.image_hull
    )


def _coverage_history(
    distribution: Mapping[str, Any],
    rounds: Sequence[PlanRound],
    *,
    layer: PoseLayer = DEFAULT_POSE_LAYER,
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, list[tuple[float, float]]]]:
    resolved_layer = validate_pose_layer(layer)
    observed = {
        (obj["name"], pose["episode_index"]): pose
        for obj in distribution["objects"]
        for pose in get_object_layer(obj, resolved_layer).poses
    }
    extra: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    avoid: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for round_ in rounds:
        if round_["status"] != "collected":
            continue
        for target in round_["targets"]:
            obj = target["object"]
            avoid[obj].append((target["x_m"], target["y_m"]))
            pose = observed.get((obj, round_["episode_index"]))
            if pose is not None:
                avoid[obj].append((pose["x_m"], pose["y_m"]))
            else:
                extra[obj].append(
                    {
                        "x_m": target["x_m"],
                        "y_m": target["y_m"],
                        "folded_yaw_deg": target["yaw_deg"],
                    }
                )
    return extra, avoid


def _camera_serial(calibration: HeadCalibration) -> str | None:
    raw = calibration.intrinsics.raw
    serial = raw.get("serial_number")
    if isinstance(serial, str) and serial.strip():
        return serial.strip()
    device = raw.get("device")
    match = (
        re.match(r"^RealSense\s+\S+\s+(\d+)(?:\s*,|$)", device) if isinstance(device, str) else None
    )
    return match[1] if match else None


def _candidates(
    distribution: Mapping[str, Any],
    calibration: HeadCalibration,
    templates: Sequence[Mapping[str, Any]],
    calibration_sha256: str,
    targets: Mapping[str, Any] | None,
    additional_poses: Mapping[str, Sequence[Mapping[str, Any]]],
    round_count: int,
    seed: int,
    *,
    layer: PoseLayer = DEFAULT_POSE_LAYER,
) -> tuple[dict[str, list[_Candidate]], str, bytes, dict[str, str]]:
    resolved_layer = validate_pose_layer(layer)
    by_object = {template["object"]: template for template in templates}
    missing_templates = sorted({obj["name"] for obj in distribution["objects"]} - by_object.keys())
    if missing_templates:
        raise ValueError(f"Missing reviewed templates for: {', '.join(missing_templates)}")
    for template in templates:
        name = str(template["object"])
        actual_layer = template_anchor_layer(template)
        if actual_layer != resolved_layer:
            raise ValueError(
                f"Template for {name} uses anchor_layer={actual_layer!r}, "
                f"expected {resolved_layer!r}"
            )
    suggestions = generate_suggestions(
        distribution,
        count=min(2048, max(96, round_count * 5)),
        test_per_tier=0,
        targets=targets,
        calibration=calibration,
        additional_poses=additional_poses,
        seed=seed,
        layer=resolved_layer,
    )
    suggested_names = {obj["object"] for obj in suggestions["objects"]}
    without_placements = set(suggestions["omitted_objects"]) - suggested_names
    bundle = make_bundle(
        suggestions,
        calibration,
        [template for template in templates if template["object"] not in without_placements],
        calibration_sha256=calibration_sha256,
        suggestions_sha256=sha256_bytes(json_bytes(suggestions)),
    )
    pools, exclusions = _build_candidate_pools(
        distribution, calibration, suggestions, bundle, by_object
    )
    return pools, suggestions["provenance"]["region_source"], json_bytes(bundle), exclusions


def _build_candidate_pools(
    distribution: Mapping[str, Any],
    calibration: HeadCalibration,
    suggestions: Mapping[str, Any],
    bundle: Mapping[str, Any],
    templates: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[_Candidate]], dict[str, str]]:
    meta = {obj["object"]: obj for obj in suggestions["objects"]}
    first_arm = min(calibration.arms)
    from_camera = np.linalg.inv(calibration.arms[first_arm].T_cam_base)
    to_common_base = {name: from_camera @ arm.T_cam_base for name, arm in calibration.arms.items()}
    pools: dict[str, list[_Candidate]] = defaultdict(list)
    missing = {
        name: "No calibrated placement candidates" for name in suggestions["omitted_objects"]
    }
    manual: dict[str, str] = {}
    for item, mask in zip(suggestions["placements"], bundle["targets"], strict=True):
        name = item["object"]
        template = templates[name]
        reason = template.get("reason")
        if reason:
            manual[name] = reason
        candidate = _placement_candidate(
            item,
            mask,
            template,
            meta[name]["yaw_symmetry_deg"],
            calibration.image_size,
            to_common_base.get(item["arm"]),
        )
        if candidate is not None:
            pools[name].append(candidate)
    for obj in distribution["objects"]:
        name = obj["name"]
        if name not in pools and name not in missing:
            missing[name] = "No candidate fits the calibrated image"
    return pools, missing | manual


def _placement_candidate(
    item: Mapping[str, Any],
    mask: Mapping[str, Any],
    template: Mapping[str, Any],
    period: float,
    image_size: tuple[int, int],
    to_common_base: np.ndarray | None,
) -> _Candidate | None:
    if not (
        to_common_base is not None
        and item["pixel"] is not None
        and all(0 <= v < bound for v, bound in zip(item["pixel"], image_size, strict=True))
        and item["arrow_end_pixel"] is not None
        and all(map(math.isfinite, item["arrow_end_pixel"]))
    ):
        return None
    target: PlanTarget = {
        "kind": "placement",
        "object": item["object"],
        "index": item["index"],
        "arm": item["arm"],
        "x_m": item["x_m"],
        "y_m": item["y_m"],
        "z_m": item["z_m"],
        "yaw_deg": item["yaw_deg"],
        "yaw_symmetry_deg": period,
        "pixel": item["pixel"],
        "arrow_end_pixel": item["arrow_end_pixel"],
        "outline": mask["outline"],
        "reason": mask["reason"],
        "outline_plane_z_m": None,
    }
    if target["outline"] is None:
        return _Candidate(target, None, None) if template.get("reason") else None
    xyz = outline_base_points(template, item, period)
    base_xy = (xyz @ to_common_base[:3, :3].T + to_common_base[:3, 3])[:, :2]
    try:
        base_hull = _hull(base_xy)
        image_hull = _hull(np.asarray(target["outline"]))
    except ValueError:
        return None
    target["outline_plane_z_m"] = item["z_m"] + template["height_offset_m"]
    return _Candidate(target, base_hull, image_hull)


def _options_for_round(
    names: list[str], pools: Mapping[str, list[_Candidate]], count: int, budget: int
) -> Iterator[list[_Candidate]]:
    def search(start: int, selected: list[_Candidate]) -> Iterator[list[_Candidate]]:
        nonlocal budget
        if len(selected) == count:
            yield selected
            return
        for i in range(start, len(names) - (count - len(selected)) + 1):
            for candidate in pools[names[i]]:
                budget -= 1
                if budget < 0:
                    return
                if not any(_collides(candidate, other) for other in selected):
                    yield from search(i + 1, [*selected, candidate])

    yield from search(0, [])


def _validate_plan_request(
    distribution: Mapping[str, Any],
    calibration: HeadCalibration,
    calibration_sha256: str,
    targets: Mapping[str, Any] | None,
    previous: CollectionPlan | None,
    round_count: int,
    objects_per_round: int,
    seed: int,
    layer: PoseLayer = DEFAULT_POSE_LAYER,
) -> PoseLayer:
    resolved_layer = validate_pose_layer(layer)
    if not calibration.arms or (calibration.camera, calibration.image_size) != (
        calibration.intrinsics.camera,
        calibration.intrinsics.image_size,
    ):
        raise ValueError("Expected head intrinsics and at least one calibrated arm")
    identity = (calibration.repo_id, calibration.revision, calibration_sha256)
    if previous and identity != (
        previous["repo_id"],
        previous["revision"],
        previous["calibration_sha256"],
    ):
        raise ValueError("Existing plan uses another dataset or calibration")
    if previous is not None:
        previous_layer = validate_pose_layer(str(previous.get("layer", "tool")))
        if previous_layer != resolved_layer:
            raise ValueError(
                f"Existing plan uses layer {previous_layer!r}; "
                f"cannot generate with layer {resolved_layer!r}"
            )
    if targets and (targets["camera"], targets["image_size"]) != (
        calibration.camera,
        list(calibration.image_size),
    ):
        raise ValueError("Target polygon does not match head camera calibration")
    if not 1 <= round_count <= 500 or not 1 <= objects_per_round <= len(distribution["objects"]):
        raise ValueError("Request 1–500 rounds and 1–all objects per round")
    if not 0 <= seed <= 2**53 - 1:
        raise ValueError("Seed must be a nonnegative safe integer")
    require_object_layers(distribution["objects"], resolved_layer)
    return resolved_layer


@dataclass(frozen=True)
class _PlanningHistory:
    kept: list[PlanRound]
    additional_poses: dict[str, list[Mapping[str, Any]]]
    avoid: dict[str, list[tuple[float, float]]]
    usage: Counter[str]
    distance: float


def _planning_history(
    distribution: Mapping[str, Any],
    previous: CollectionPlan | None,
    replace_round_id: int | None,
    *,
    layer: PoseLayer = DEFAULT_POSE_LAYER,
) -> _PlanningHistory:
    old_rounds = [] if previous is None else previous["rounds"]
    additional_poses, avoid = _coverage_history(distribution, old_rounds, layer=layer)
    distance = float(distribution["coverage_config"]["xy_bin_m"]) / 2
    kept = [r for r in old_rounds if r["status"] != "pending"]
    usage: Counter[str] = Counter(
        t["object"] for r in kept if r["status"] == "collected" for t in r["targets"]
    )
    if replace_round_id is not None:
        if previous is None:
            raise ValueError("No collection plan to edit")
        _pending_index(previous, replace_round_id)
        distance = previous["settings"]["min_revisit_distance_m"]
        for r in old_rounds:
            if r["status"] == "pending":
                for t in r["targets"]:
                    avoid[t["object"]].append((t["x_m"], t["y_m"]))
                    if r["id"] != replace_round_id:
                        usage[t["object"]] += 1
    return _PlanningHistory(kept, additional_poses, avoid, usage, distance)


def _select_round(
    pools: Mapping[str, list[_Candidate]],
    avoid: Mapping[str, Sequence[tuple[float, float]]],
    usage: Counter[str],
    order: Mapping[str, int],
    distance: float,
    objects_per_round: int,
) -> list[PlanTarget] | None:
    available = {
        name: [
            candidate
            for candidate in pool
            if all(
                math.hypot(candidate.target["x_m"] - x, candidate.target["y_m"] - y) >= distance
                for x, y in avoid.get(name, ())
            )
        ]
        for name, pool in pools.items()
    }
    names = sorted(
        (name for name in available if available[name]), key=lambda n: (usage[n], order[n])
    )
    chosen = next(_options_for_round(names, available, objects_per_round, 24_000), None)
    return None if chosen is None else [candidate.target for candidate in chosen]


def _plan_rounds(
    pools: Mapping[str, list[_Candidate]],
    history: _PlanningHistory,
    order: Mapping[str, int],
    digest: str,
    next_id: int,
    round_count: int,
    objects_per_round: int,
) -> tuple[list[PlanRound], int]:
    planned: list[PlanRound] = []
    for _ in range(round_count):
        selected = _select_round(
            pools, history.avoid, history.usage, order, history.distance, objects_per_round
        )
        if selected is None:
            raise ValueError(
                f"Found {len(planned)}/{round_count} non-overlapping rounds with "
                f"{objects_per_round} objects; try fewer rounds/objects or a larger target region"
            )
        for t in selected:
            history.usage[t["object"]] += 1
            history.avoid[t["object"]].append((t["x_m"], t["y_m"]))
        planned.append(
            {
                "id": next_id,
                "mask_bundle_sha256": digest,
                "targets": selected,
                "status": "pending",
                "episode_index": None,
            }
        )
        next_id += 1
    return planned, next_id


def _eligible_pools(
    pools: dict[str, list[_Candidate]], exclusions: dict[str, str], objects_per_round: int
) -> tuple[dict[str, list[_Candidate]], dict[str, str]]:
    if objects_per_round > 1:
        pools = {name: [c for c in pool if c.base_hull is not None] for name, pool in pools.items()}
        pools = {name: pool for name, pool in pools.items() if pool}
    elif pools:
        exclusions = {name: reason for name, reason in exclusions.items() if name not in pools}
    if len(pools) < objects_per_round:
        reasons = "; ".join(f"{name}: {reason}" for name, reason in exclusions.items())
        raise ValueError(
            f"Requested {objects_per_round} objects per round; "
            f"only {len(pools)} are eligible. {reasons}"
        )
    return pools, exclusions


def build_plan(
    distribution: Mapping[str, Any],
    calibration: HeadCalibration,
    *,
    calibration_sha256: str,
    templates: Sequence[Mapping[str, Any]],
    source_episode_count: int,
    targets: Mapping[str, Any] | None = None,
    previous: CollectionPlan | None = None,
    round_count: int,
    objects_per_round: int,
    seed: int = 0,
    replace_round_id: int | None = None,
    layer: PoseLayer = DEFAULT_POSE_LAYER,
) -> tuple[CollectionPlan, bytes]:
    resolved_layer = _validate_plan_request(
        distribution,
        calibration,
        calibration_sha256,
        targets,
        previous,
        round_count,
        objects_per_round,
        seed,
        layer=layer,
    )
    history = _planning_history(distribution, previous, replace_round_id, layer=resolved_layer)
    if replace_round_id is not None and previous is not None:
        round_count = 1
        objects_per_round = previous["settings"]["objects_per_round"]
    pools, region, bundle, exclusions = _candidates(
        distribution,
        calibration,
        templates,
        calibration_sha256,
        targets,
        history.additional_poses,
        round_count,
        seed,
        layer=resolved_layer,
    )
    pools, exclusions = _eligible_pools(pools, exclusions, objects_per_round)
    digest = sha256_bytes(bundle)
    next_id = 1 if previous is None else previous["next_round_id"]
    order = {obj["name"]: index for index, obj in enumerate(distribution["objects"])}
    planned, next_id = _plan_rounds(
        pools,
        history,
        order,
        digest,
        next_id,
        round_count,
        objects_per_round,
    )
    settings: PlanSettings = {
        "round_count": round_count,
        "objects_per_round": objects_per_round,
        "seed": seed,
        "min_revisit_distance_m": history.distance,
        "region_source": region,
    }
    rounds = history.kept + planned
    if replace_round_id is not None and previous is not None:
        settings = previous["settings"]
        rounds = [planned[0] if r["id"] == replace_round_id else r for r in previous["rounds"]]
    plan: CollectionPlan = {
        "format": "lerobot-collection-plan",
        "version": 1,
        "plan_id": previous["plan_id"] if previous else str(uuid4()),
        "generation": previous["generation"] + 1 if previous else 1,
        "next_round_id": next_id,
        "repo_id": calibration.repo_id,
        "revision": calibration.revision,
        "source_episode_count": previous["source_episode_count"]
        if previous
        else source_episode_count,
        "camera": calibration.camera,
        "image_size": list(calibration.image_size),
        "camera_serial": _camera_serial(calibration),
        "calibration_sha256": calibration_sha256,
        "mask_bundle_sha256": digest,
        "layer": resolved_layer,
        "settings": settings,
        "rounds": rounds,
        "excluded_objects": exclusions,
    }
    return plan, bundle


def _pending_index(plan: CollectionPlan, round_id: int) -> int:
    for index, round_ in enumerate(plan["rounds"]):
        if round_["id"] == round_id and round_["status"] == "pending":
            return index
    raise ValueError(f"Collection round {round_id} is no longer pending")


def delete_pending(plan: CollectionPlan, round_id: int) -> None:
    del plan["rounds"][_pending_index(plan, round_id)]
    plan["generation"] += 1
