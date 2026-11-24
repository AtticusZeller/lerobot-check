"""Collection placement and evaluation test-point suggestions inside target polygons."""

from __future__ import annotations

import csv
import json
import math
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lerobot_check.analysis import yaw_bin_index
from lerobot_check.calib import (
    HeadCalibration,
    backproject_pixels_to_plane_z,
    project_base_points,
)

SUGGESTIONS_SCHEMA_VERSION = 1
TARGETS_SCHEMA_VERSION = 1
DEFAULT_BBOX_MARGIN_M = 0.05
ARROW_LENGTH_M = 0.05
COLLISION_NOTE = (
    "Placements are sampled independently per object; inter-object collisions are not checked."
)
TEST_TIERS: tuple[str, ...] = ("dense", "edge", "empty")


def load_target_polygon(path: Path) -> dict[str, Any]:
    """Load and validate a ``targets/<org>__<name>.json`` pixel-space polygon file.

    Args:
        path: Path to the target polygon JSON file.

    Returns:
        Validated target polygon dictionary.

    Raises:
        ValueError: If the schema version is unsupported or polygon has fewer than 3 vertices.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != TARGETS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported targets JSON at {path} (expected version={TARGETS_SCHEMA_VERSION})"
        )
    poly = raw.get("polygon")
    if not isinstance(poly, list) or len(poly) < 3:
        raise ValueError(f"targets.polygon at {path} must contain at least 3 [u, v] vertices")
    for idx, pt in enumerate(poly):
        if not isinstance(pt, list) or len(pt) != 2:
            raise ValueError(f"targets.polygon[{idx}] must be [u, v], got {pt!r}")
    return raw


def _point_on_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    tol: float = 1e-12,
) -> bool:
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > tol * max(1.0, math.hypot(bx - ax, by - ay)):
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    if dot < -tol:
        return False
    sq_len = (bx - ax) ** 2 + (by - ay) ** 2
    return dot <= sq_len + tol


def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Return ``True`` if 2D point ``(x, y)`` lies inside or on the boundary of ``polygon``.

    Uses the standard ray-casting algorithm and handles both convex and concave
    simple polygons cleanly.

    Args:
        x: Query X coordinate.
        y: Query Y coordinate.
        polygon: Ordered sequence of ``>= 3`` ``(vx, vy)`` vertices.

    Returns:
        ``True`` if ``(x, y)`` is inside or on an edge of ``polygon``.
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    x1, y1 = float(polygon[-1][0]), float(polygon[-1][1])
    for vertex in polygon:
        x2, y2 = float(vertex[0]), float(vertex[1])
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def _data_bbox_polygon(
    poses: Sequence[Mapping[str, Any]],
    margin_m: float = DEFAULT_BBOX_MARGIN_M,
) -> list[list[float]]:
    xs = [float(p["x_m"]) for p in poses]
    ys = [float(p["y_m"]) for p in poses]
    x_min = min(xs) - margin_m
    x_max = max(xs) + margin_m
    y_min = min(ys) - margin_m
    y_max = max(ys) + margin_m
    return [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]


def _cell_anchor_points(
    ix: int,
    iy: int,
    xy_bin_m: float,
    base_polygon: Sequence[Sequence[float]],
    *,
    pixel_polygon: Sequence[Sequence[float]] | None = None,
    z_plane_m: float = 0.0,
    T_cam_base: np.ndarray | None = None,
    intrinsics: Any = None,
) -> list[tuple[float, float]]:
    """Return interior subgrid points of cell ``(ix, iy)`` that lie inside the polygon."""
    x0 = ix * xy_bin_m
    y0 = iy * xy_bin_m
    fracs = (0.5, 0.25, 0.75, 0.1, 0.9)
    candidates: list[tuple[float, float]] = []
    for fx in fracs:
        for fy in fracs:
            x = x0 + fx * xy_bin_m
            y = y0 + fy * xy_bin_m
            if not point_in_polygon(x, y, base_polygon):
                continue
            if pixel_polygon is not None and T_cam_base is not None and intrinsics is not None:
                px = project_base_points(np.array([[x, y, z_plane_m]]), T_cam_base, intrinsics)[0]
                if not point_in_polygon(float(px[0]), float(px[1]), pixel_polygon):
                    continue
            candidates.append((x, y))
    return candidates


def _enumerate_polygon_cells(
    base_polygon: Sequence[Sequence[float]],
    xy_bin_m: float,
    *,
    pixel_polygon: Sequence[Sequence[float]] | None = None,
    z_plane_m: float = 0.0,
    T_cam_base: np.ndarray | None = None,
    intrinsics: Any = None,
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Find all XY grid cells intersecting ``base_polygon`` along with valid interior anchors."""
    xs = [float(v[0]) for v in base_polygon]
    ys = [float(v[1]) for v in base_polygon]
    ix_min = math.floor(min(xs) / xy_bin_m)
    ix_max = math.floor(max(xs) / xy_bin_m)
    iy_min = math.floor(min(ys) / xy_bin_m)
    iy_max = math.floor(max(ys) / xy_bin_m)

    valid_cells: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for ix in range(ix_min, ix_max + 1):
        for iy in range(iy_min, iy_max + 1):
            anchors = _cell_anchor_points(
                ix,
                iy,
                xy_bin_m,
                base_polygon,
                pixel_polygon=pixel_polygon,
                z_plane_m=z_plane_m,
                T_cam_base=T_cam_base,
                intrinsics=intrinsics,
            )
            if anchors:
                valid_cells[(ix, iy)] = anchors
    return valid_cells


def _sample_point_in_cell(
    ix: int,
    iy: int,
    xy_bin_m: float,
    base_polygon: Sequence[Sequence[float]],
    anchors: Sequence[tuple[float, float]],
    rng: np.random.Generator,
    *,
    pixel_polygon: Sequence[Sequence[float]] | None = None,
    z_plane_m: float = 0.0,
    T_cam_base: np.ndarray | None = None,
    intrinsics: Any = None,
) -> tuple[float, float]:
    """Sample a random ``(x, y)`` inside cell ``(ix, iy)`` strictly inside the polygon."""
    x0 = ix * xy_bin_m
    y0 = iy * xy_bin_m
    for _ in range(64):
        x = float(rng.uniform(x0, x0 + xy_bin_m))
        y = float(rng.uniform(y0, y0 + xy_bin_m))
        if not point_in_polygon(x, y, base_polygon):
            continue
        if pixel_polygon is not None and T_cam_base is not None and intrinsics is not None:
            px = project_base_points(np.array([[x, y, z_plane_m]]), T_cam_base, intrinsics)[0]
            if not point_in_polygon(float(px[0]), float(px[1]), pixel_polygon):
                continue
        return x, y
    choice = int(rng.integers(0, len(anchors)))
    return anchors[choice]


def _draw_round_robin_by_yaw(
    candidates: Sequence[tuple[int, int, int, str]],
    needed: int,
    num_yaw_bins: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, str]]:
    """Draw up to ``needed`` ``(ix, iy, yaw_bin, tier)`` items spreading evenly across yaw bins."""
    if not candidates or needed <= 0:
        return []
    by_yaw: dict[int, list[tuple[int, int, int, str]]] = defaultdict(list)
    for item in candidates:
        by_yaw[item[2]].append(item)
    for yb in by_yaw:
        order = rng.permutation(len(by_yaw[yb]))
        by_yaw[yb] = [by_yaw[yb][int(i)] for i in order]

    yaw_order = [int(y) for y in rng.permutation(num_yaw_bins)]
    cursors = dict.fromkeys(yaw_order, 0)
    picked: list[tuple[int, int, int, str]] = []
    while len(picked) < needed:
        progressed = False
        for yb in yaw_order:
            bucket = by_yaw.get(yb)
            if not bucket or cursors[yb] >= len(bucket):
                continue
            picked.append(bucket[cursors[yb]])
            cursors[yb] += 1
            progressed = True
            if len(picked) == needed:
                break
        if not progressed:
            break
    return picked


def _build_suggestion_item(
    *,
    object_name: str,
    arm: str,
    index: int,
    x_m: float,
    y_m: float,
    z_m: float,
    yaw_deg: float,
    ix: int,
    iy: int,
    yaw_bin: int,
    cell_count: int,
    tier: str,
    T_cam_base: np.ndarray | None,
    intrinsics: Any,
) -> dict[str, Any]:
    pixel: list[float] | None = None
    arrow_end_pixel: list[float] | None = None
    if T_cam_base is not None and intrinsics is not None:
        rad = math.radians(yaw_deg)
        pts = np.array(
            [
                [x_m, y_m, z_m],
                [x_m + ARROW_LENGTH_M * math.cos(rad), y_m + ARROW_LENGTH_M * math.sin(rad), z_m],
            ],
            dtype=np.float64,
        )
        proj = project_base_points(pts, T_cam_base, intrinsics)
        pixel = [float(proj[0, 0]), float(proj[0, 1])]
        arrow_end_pixel = [float(proj[1, 0]), float(proj[1, 1])]

    return {
        "object": object_name,
        "arm": arm,
        "index": index,
        "x_m": float(x_m),
        "y_m": float(y_m),
        "z_m": float(z_m),
        "yaw_deg": float(yaw_deg),
        "ix": int(ix),
        "iy": int(iy),
        "yaw_bin": int(yaw_bin),
        "cell_count": int(cell_count),
        "tier": tier,
        "pixel": pixel,
        "arrow_end_pixel": arrow_end_pixel,
    }


def _select_test_cells_by_tier(
    valid_cells: Mapping[tuple[int, int], list[tuple[float, float]]],
    xy_counts: Mapping[tuple[int, int], int],
    sparse_threshold: int,
) -> dict[str, list[tuple[int, int]]]:
    """Partition valid XY cells inside the polygon into ``dense``, ``edge``, and ``empty`` pools."""
    empty_cells = sorted(cell for cell in valid_cells if xy_counts.get(cell, 0) == 0)
    occupied_cells = sorted(cell for cell in valid_cells if xy_counts.get(cell, 0) > 0)

    def _is_edge(cell: tuple[int, int]) -> bool:
        ix, iy = cell
        neighbors = ((ix + 1, iy), (ix - 1, iy), (ix, iy + 1), (ix, iy - 1))
        return any(xy_counts.get(nb, 0) == 0 for nb in neighbors)

    edge_cells = [cell for cell in occupied_cells if _is_edge(cell)]
    interior_occupied = [cell for cell in occupied_cells if not _is_edge(cell)]

    # Prefer interior cells meeting sparse_threshold; gracefully relax on sparse datasets.
    dense_interior = [
        cell for cell in interior_occupied if xy_counts.get(cell, 0) >= sparse_threshold
    ]
    if dense_interior:
        dense_cells = sorted(dense_interior, key=lambda c: (-xy_counts[c], c))
    else:
        dense_any = [cell for cell in occupied_cells if xy_counts.get(cell, 0) >= sparse_threshold]
        if dense_any:
            dense_cells = sorted(dense_any, key=lambda c: (-xy_counts[c], c))
        elif interior_occupied:
            dense_cells = sorted(interior_occupied, key=lambda c: (-xy_counts[c], c))
        else:
            dense_cells = sorted(occupied_cells, key=lambda c: (-xy_counts[c], c))

    return {
        "dense": dense_cells,
        "edge": edge_cells,
        "empty": empty_cells,
    }


def generate_suggestions(
    distribution: Mapping[str, Any],
    *,
    count: int = 10,
    test_per_tier: int = 3,
    targets: Mapping[str, Any] | None = None,
    calibration: HeadCalibration | None = None,
    object_filter: Sequence[str] | None = None,
    seed: int = 0,
    distribution_path: Path | str | None = None,
    targets_path: Path | str | None = None,
    calib_path: Path | str | None = None,
) -> dict[str, Any]:
    """Generate next-batch collection placements and three-tier test points.

    When ``targets`` is provided, ``calibration`` is required so the pixel-space
    polygon can be back-projected onto each object's plane ``z = median_z_m``.
    When ``targets`` is ``None``, falls back to the object's observed data
    bounding box expanded by 5 cm with a warning.

    Args:
        distribution: Loaded ``distribution.json`` dictionary.
        count: Number of collection placements ``N`` to sample per object.
        test_per_tier: Number of evaluation test points per tier (``dense``, ``edge``,
            ``empty``) per object.
        targets: Optional parsed ``targets/<org>__<name>.json`` dictionary.
        calibration: Optional loaded ``HeadCalibration`` from ``head.json``.
        object_filter: Optional subset of object names to generate suggestions for.
        seed: Random seed for reproducible sampling.
        distribution_path: Recorded in provenance when provided.
        targets_path: Recorded in provenance when provided.
        calib_path: Recorded in provenance when provided.

    Returns:
        JSON-serializable ``suggestions.json`` v1 dictionary.

    Raises:
        ValueError: If ``count < 1``, ``test_per_tier < 0``, ``targets`` is supplied
            without calibration, or a requested object/arm cannot be resolved.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if test_per_tier < 0:
        raise ValueError(f"test_per_tier must be >= 0, got {test_per_tier}")
    if targets is not None and calibration is None:
        raise ValueError(
            "Target polygon is in head-image pixels and cannot be back-projected without "
            "head-camera calibration (head.json). Run calibration or pass --calib, "
            "or omit --targets to use the data bounding box fallback."
        )

    if targets is None:
        warnings.warn(
            "No target polygon provided; falling back to observed data bounding box + 5 cm.",
            UserWarning,
            stacklevel=2,
        )
        region_source = "data_bbox_plus_5cm"
        pixel_polygon: list[list[float]] | None = None
    else:
        region_source = "target_polygon"
        pixel_polygon = [[float(pt[0]), float(pt[1])] for pt in targets["polygon"]]

    cov_cfg = distribution["coverage_config"]
    xy_bin_m = float(cov_cfg["xy_bin_m"])
    yaw_bin_deg = float(cov_cfg["yaw_bin_deg"])
    sparse_threshold = int(cov_cfg["sparse_threshold"])

    all_objects: list[Mapping[str, Any]] = list(distribution.get("objects", []))
    if object_filter:
        wanted = set(object_filter)
        known = {str(o["name"]) for o in all_objects}
        missing = sorted(wanted - known)
        if missing:
            raise ValueError(f"Unknown object(s) in --object: {missing}")
        selected_objects = [o for o in all_objects if str(o["name"]) in wanted]
    else:
        selected_objects = all_objects

    rng = np.random.default_rng(seed)
    objects_meta: list[dict[str, Any]] = []
    omitted_objects: list[str] = []
    placements: list[dict[str, Any]] = []
    test_points: list[dict[str, Any]] = []

    for obj in selected_objects:
        obj_name = str(obj["name"])
        arm = str(obj["arm"])
        yaw_sym = float(obj["yaw_symmetry_deg"])
        poses: list[Mapping[str, Any]] = list(obj.get("poses", []))
        median_z_raw = obj.get("median_z_m")
        if not poses or median_z_raw is None:
            omitted_objects.append(obj_name)
            continue
        z_plane_m = float(median_z_raw)

        arm_cal = calibration.arms.get(arm) if calibration is not None else None
        T_cam_base = arm_cal.T_cam_base if arm_cal is not None else None
        intrinsics = calibration.intrinsics if calibration is not None else None

        if pixel_polygon is not None:
            if arm_cal is None or T_cam_base is None or intrinsics is None:
                raise ValueError(
                    f"Object {obj_name!r} uses arm {arm!r}, which is missing from "
                    "head-camera calibration."
                )
            base_pts = backproject_pixels_to_plane_z(
                np.asarray(pixel_polygon, dtype=np.float64),
                z_plane_m,
                T_cam_base,
                intrinsics,
            )
            base_polygon = [[float(p[0]), float(p[1])] for p in base_pts]
        else:
            base_polygon = _data_bbox_polygon(poses, margin_m=DEFAULT_BBOX_MARGIN_M)

        objects_meta.append(
            {
                "object": obj_name,
                "arm": arm,
                "z_plane_m": z_plane_m,
                "yaw_symmetry_deg": yaw_sym,
                "polygon_base_xy_m": base_polygon,
                "polygon_pixel": pixel_polygon,
            }
        )

        valid_cells = _enumerate_polygon_cells(
            base_polygon,
            xy_bin_m,
            pixel_polygon=pixel_polygon,
            z_plane_m=z_plane_m,
            T_cam_base=T_cam_base,
            intrinsics=intrinsics,
        )
        if not valid_cells:
            omitted_objects.append(obj_name)
            continue

        _, num_yaw_bins = yaw_bin_index(0.0, yaw_symmetry_deg=yaw_sym, yaw_bin_deg=yaw_bin_deg)
        xy_counts: dict[tuple[int, int], int] = defaultdict(int)
        xy_yaw_counts: dict[tuple[int, int, int], int] = defaultdict(int)
        for p in poses:
            ix = math.floor(float(p["x_m"]) / xy_bin_m)
            iy = math.floor(float(p["y_m"]) / xy_bin_m)
            yb, _ = yaw_bin_index(
                float(p["folded_yaw_deg"]),
                yaw_symmetry_deg=yaw_sym,
                yaw_bin_deg=yaw_bin_deg,
            )
            xy_counts[(ix, iy)] += 1
            xy_yaw_counts[(ix, iy, yb)] += 1

        # Partition candidate (ix, iy, yaw_bin) cells inside polygon by priority:
        # 0: completely empty XY cell -> 1: empty yaw bin on occupied XY -> 2: sparse -> 3: dense.
        tier_pools: dict[int, list[tuple[int, int, int, str]]] = {0: [], 1: [], 2: [], 3: []}
        for ix, iy in sorted(valid_cells):
            cell_xy_c = xy_counts.get((ix, iy), 0)
            for yb in range(num_yaw_bins):
                c = xy_yaw_counts.get((ix, iy, yb), 0)
                if c == 0 and cell_xy_c == 0:
                    tier_pools[0].append((ix, iy, yb, "empty"))
                elif c == 0:
                    tier_pools[1].append((ix, iy, yb, "empty"))
                elif c < sparse_threshold:
                    tier_pools[2].append((ix, iy, yb, "sparse"))
                else:
                    tier_pools[3].append((ix, iy, yb, "dense"))

        chosen_bins: list[tuple[int, int, int, str]] = []
        for prio in (0, 1, 2, 3):
            if len(chosen_bins) >= count:
                break
            drawn = _draw_round_robin_by_yaw(
                tier_pools[prio], count - len(chosen_bins), num_yaw_bins, rng
            )
            chosen_bins.extend(drawn)

        # If N exceeds total unique (ix, iy, yaw_bin) cells, repeat from the highest-priority pool.
        fallback_pool = next((tier_pools[p] for p in (0, 1, 2, 3) if tier_pools[p]), [])
        while len(chosen_bins) < count and fallback_pool:
            drawn = _draw_round_robin_by_yaw(
                fallback_pool, count - len(chosen_bins), num_yaw_bins, rng
            )
            chosen_bins.extend(drawn)

        for idx_1based, (ix, iy, yb, tier) in enumerate(chosen_bins, start=1):
            x_m, y_m = _sample_point_in_cell(
                ix,
                iy,
                xy_bin_m,
                base_polygon,
                valid_cells[(ix, iy)],
                rng,
                pixel_polygon=pixel_polygon,
                z_plane_m=z_plane_m,
                T_cam_base=T_cam_base,
                intrinsics=intrinsics,
            )
            yaw_lo = yb * yaw_bin_deg
            yaw_hi = min(yaw_sym, (yb + 1) * yaw_bin_deg)
            yaw_deg = float(rng.uniform(yaw_lo, yaw_hi))
            placements.append(
                _build_suggestion_item(
                    object_name=obj_name,
                    arm=arm,
                    index=idx_1based,
                    x_m=x_m,
                    y_m=y_m,
                    z_m=z_plane_m,
                    yaw_deg=yaw_deg,
                    ix=ix,
                    iy=iy,
                    yaw_bin=yb,
                    cell_count=xy_yaw_counts.get((ix, iy, yb), 0),
                    tier=tier,
                    T_cam_base=T_cam_base,
                    intrinsics=intrinsics,
                )
            )

        if test_per_tier > 0:
            test_pools = _select_test_cells_by_tier(valid_cells, xy_counts, sparse_threshold)
            for tier_name in TEST_TIERS:
                cells_pool = test_pools[tier_name]
                if not cells_pool:
                    continue
                selected_xy: list[tuple[int, int]] = []
                while len(selected_xy) < test_per_tier:
                    order = rng.permutation(len(cells_pool))
                    for c_idx in order:
                        selected_xy.append(cells_pool[int(c_idx)])
                        if len(selected_xy) == test_per_tier:
                            break
                for t_idx, (ix, iy) in enumerate(selected_xy, start=1):
                    x_m, y_m = _sample_point_in_cell(
                        ix,
                        iy,
                        xy_bin_m,
                        base_polygon,
                        valid_cells[(ix, iy)],
                        rng,
                        pixel_polygon=pixel_polygon,
                        z_plane_m=z_plane_m,
                        T_cam_base=T_cam_base,
                        intrinsics=intrinsics,
                    )
                    if tier_name == "dense":
                        occ_yaws = [
                            yb
                            for yb in range(num_yaw_bins)
                            if xy_yaw_counts.get((ix, iy, yb), 0) > 0
                        ]
                        yb = (
                            occ_yaws[int(rng.integers(0, len(occ_yaws)))]
                            if occ_yaws
                            else (t_idx - 1) % num_yaw_bins
                        )
                    else:
                        yb = (t_idx - 1) % num_yaw_bins
                    yaw_lo = yb * yaw_bin_deg
                    yaw_hi = min(yaw_sym, (yb + 1) * yaw_bin_deg)
                    yaw_deg = float(rng.uniform(yaw_lo, yaw_hi))
                    test_points.append(
                        _build_suggestion_item(
                            object_name=obj_name,
                            arm=arm,
                            index=t_idx,
                            x_m=x_m,
                            y_m=y_m,
                            z_m=z_plane_m,
                            yaw_deg=yaw_deg,
                            ix=ix,
                            iy=iy,
                            yaw_bin=yb,
                            cell_count=xy_counts.get((ix, iy), 0),
                            tier=tier_name,
                            T_cam_base=T_cam_base,
                            intrinsics=intrinsics,
                        )
                    )

    prov = distribution["provenance"]
    return {
        "version": SUGGESTIONS_SCHEMA_VERSION,
        "provenance": {
            "repo_id": prov["repo_id"],
            "revision": prov["revision"],
            "task_name": prov["task_name"],
            "distribution_path": str(distribution_path) if distribution_path is not None else None,
            "targets_path": str(targets_path) if targets_path is not None else None,
            "calib_path": str(calib_path) if calib_path is not None else None,
            "region_source": region_source,
            "seed": seed,
            "count_per_object": count,
            "test_per_tier": test_per_tier,
            "arrow_length_m": ARROW_LENGTH_M,
            "note": COLLISION_NOTE,
        },
        "objects": objects_meta,
        "omitted_objects": omitted_objects,
        "placements": placements,
        "test_points": test_points,
    }


def write_suggestions_json(payload: Mapping[str, Any], path: Path) -> Path:
    """Write a ``suggestions.json`` payload cleanly with UTF-8 indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_suggestions_csv(payload: Mapping[str, Any], path: Path) -> Path:
    """Export both placements and test points from ``suggestions.json`` to a flat CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "kind",
        "object",
        "arm",
        "index",
        "tier",
        "x_m",
        "y_m",
        "z_m",
        "yaw_deg",
        "ix",
        "iy",
        "yaw_bin",
        "cell_count",
        "pixel_u",
        "pixel_v",
        "arrow_end_u",
        "arrow_end_v",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for kind, key in (("placement", "placements"), ("test_point", "test_points")):
            for item in payload.get(key, []):
                px = item.get("pixel")
                arrow = item.get("arrow_end_pixel")
                writer.writerow(
                    {
                        "kind": kind,
                        "object": item["object"],
                        "arm": item["arm"],
                        "index": item["index"],
                        "tier": item["tier"],
                        "x_m": f"{item['x_m']:.4f}",
                        "y_m": f"{item['y_m']:.4f}",
                        "z_m": f"{item['z_m']:.4f}",
                        "yaw_deg": f"{item['yaw_deg']:.2f}",
                        "ix": item["ix"],
                        "iy": item["iy"],
                        "yaw_bin": item["yaw_bin"],
                        "cell_count": item["cell_count"],
                        "pixel_u": "" if px is None else f"{px[0]:.2f}",
                        "pixel_v": "" if px is None else f"{px[1]:.2f}",
                        "arrow_end_u": "" if arrow is None else f"{arrow[0]:.2f}",
                        "arrow_end_v": "" if arrow is None else f"{arrow[1]:.2f}",
                    }
                )
    return path


def format_suggestions_summary(payload: Mapping[str, Any]) -> str:
    """Format a concise CLI summary of generated placements and test points."""
    prov = payload["provenance"]
    placements = payload.get("placements", [])
    test_points = payload.get("test_points", [])
    lines = [
        f"Suggestions for {prov['repo_id']} @ {prov['revision']} ({prov['region_source']})",
        f"Placements: {len(placements)}, Test points: {len(test_points)}",
        prov["note"],
    ]
    omitted = payload.get("omitted_objects", [])
    if omitted:
        lines.append(f"Omitted objects (no observed poses or valid cells): {', '.join(omitted)}")
    return "\n".join(lines)
