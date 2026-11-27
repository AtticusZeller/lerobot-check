from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lerobot_check.calib import (
    ArmCalibration,
    CameraIntrinsics,
    HeadCalibration,
    backproject_pixels_to_plane_z,
    load_head_calibration,
    write_json,
)
from lerobot_check.dataset import (
    DatasetView,
    default_annotations_path,
    default_calib_head_path,
    default_distribution_path,
    default_object_poses_path,
    default_task_spec_path,
    rotation_6d_to_matrix,
)
from lerobot_check.json_io import read_json, sha256_bytes
from lerobot_check.render import decode_video_frames_pyav
from lerobot_check.schema import ObjectSpec, TaskSpec, load_annotations, load_task_spec
from lerobot_check.segmentation import MODEL, MODEL_REVISION, SamSegmenter, exterior_outline

OBJECT_POSES_SCHEMA_VERSION = 1
OBJECT_POSE_FRAME = 0
OBJECT_LAYER_SOURCE = "head_mask"
DEFAULT_MAX_SIZE_DEVIATION = 0.25
MIN_MEDIAN_SIZE_SAMPLES = 3

SKIP_EMPTY_MASK = "empty mask"
SKIP_DEGENERATE_OUTLINE = "degenerate mask outline"
SKIP_BORDER_TOUCH = "mask touches layout box border"

_OBJECT_COLORS: tuple[tuple[int, int, int], ...] = (
    (34, 211, 238),
    (251, 146, 60),
    (74, 222, 128),
    (232, 121, 249),
    (250, 204, 21),
    (96, 165, 250),
)
_SKIP_COLOR = (239, 68, 68)
_BG_COLOR = (20, 22, 26)
_HEADER_BG = (32, 36, 42)
_TEXT_COLOR = (242, 245, 248)
_OK_TEXT_COLOR = (134, 239, 172)
_WARN_TEXT_COLOR = (252, 165, 165)


@dataclass(frozen=True)
class ObjectPoseMeasurement:
    x_m: float
    y_m: float
    z_m: float
    raw_yaw_deg: float
    folded_yaw_deg: float
    length_m: float
    width_m: float


@dataclass(frozen=True)
class EpisodeObjectMeasurement:
    episode_index: int
    frame: int
    status: str
    outline: list[list[float]] | None
    pose: ObjectPoseMeasurement | None


def object_yaw_period_deg(yaw_symmetry_deg: float) -> float:
    if yaw_symmetry_deg <= 0.0 or yaw_symmetry_deg > 360.0:
        raise ValueError(f"yaw_symmetry_deg must be in (0, 360], got {yaw_symmetry_deg}")
    return min(float(yaw_symmetry_deg), 180.0)


def fold_angle_deg(yaw_deg: float, period_deg: float) -> float:
    if period_deg <= 0.0 or period_deg > 360.0:
        raise ValueError(f"period_deg must be in (0, 360], got {period_deg}")
    folded = float(yaw_deg) % float(period_deg)
    if folded >= period_deg or abs(folded - period_deg) < 1e-9:
        folded = 0.0
    return folded


def polygon_centroid_xy(points_xy: np.ndarray) -> tuple[float, float]:
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
        raise ValueError("Polygon requires at least 3 finite 2D vertices")
    origin = pts[0]
    rel = pts - origin
    nxt = np.roll(rel, -1, axis=0)
    cross = rel[:, 0] * nxt[:, 1] - nxt[:, 0] * rel[:, 1]
    twice_area = float(np.sum(cross))
    if abs(twice_area) <= 1e-12:
        raise ValueError("Polygon has zero area")
    cx = float(np.sum((rel[:, 0] + nxt[:, 0]) * cross) / (3.0 * twice_area)) + float(origin[0])
    cy = float(np.sum((rel[:, 1] + nxt[:, 1]) * cross) / (3.0 * twice_area)) + float(origin[1])
    return cx, cy


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    pts = np.unique(np.asarray(points_xy, dtype=np.float64), axis=0)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
        raise ValueError("Convex hull requires at least 3 distinct finite 2D points")
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    sorted_pts = pts[order]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    hull = np.vstack(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        raise ValueError("Polygon points are collinear")
    return hull


def min_area_rect_pose_xy(points_xy: np.ndarray) -> tuple[float, float, float]:
    hull = _convex_hull_xy(points_xy)
    edges = np.roll(hull, -1, axis=0) - hull
    norms = np.hypot(edges[:, 0], edges[:, 1])
    valid = norms > 1e-12
    if not np.any(valid):
        raise ValueError("Degenerate convex hull edges")
    u = edges[valid] / norms[valid, None]
    v = np.column_stack((-u[:, 1], u[:, 0]))
    proj_u = u @ hull.T
    proj_v = v @ hull.T
    span_u = proj_u.max(axis=1) - proj_u.min(axis=1)
    span_v = proj_v.max(axis=1) - proj_v.min(axis=1)
    areas = span_u * span_v
    best_idx = int(np.argmin(areas))
    su = float(span_u[best_idx])
    sv = float(span_v[best_idx])
    if su <= 1e-12 or sv <= 1e-12:
        raise ValueError("Minimum-area rectangle is degenerate")
    if su >= sv:
        axis = u[best_idx]
        length_m = su
        width_m = sv
    else:
        axis = v[best_idx]
        length_m = sv
        width_m = su
    raw_yaw_deg = fold_angle_deg(math.degrees(math.atan2(float(axis[1]), float(axis[0]))), 180.0)
    return raw_yaw_deg, length_m, width_m


def measure_outline_pose(
    outline_px: np.ndarray,
    *,
    plane_z_m: float,
    arm_calibration: ArmCalibration,
    intrinsics: CameraIntrinsics,
    yaw_symmetry_deg: float,
) -> ObjectPoseMeasurement:
    base_pts = backproject_pixels_to_plane_z(
        outline_px,
        plane_z_m,
        arm_calibration.T_cam_base,
        intrinsics,
    )
    xy = base_pts[:, :2]
    cx, cy = polygon_centroid_xy(xy)
    raw_yaw_deg, length_m, width_m = min_area_rect_pose_xy(xy)
    period_deg = object_yaw_period_deg(yaw_symmetry_deg)
    folded_yaw_deg = fold_angle_deg(raw_yaw_deg, period_deg)
    return ObjectPoseMeasurement(
        x_m=cx,
        y_m=cy,
        z_m=float(plane_z_m),
        raw_yaw_deg=raw_yaw_deg,
        folded_yaw_deg=folded_yaw_deg,
        length_m=length_m,
        width_m=width_m,
    )


def mask_touches_layout_box(
    mask: np.ndarray,
    layout_box: tuple[int, int, int, int],
) -> bool:
    x0, y0, x1, y1 = layout_box
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return False
    return bool(
        (xs <= x0).any() or (xs >= x1 - 1).any() or (ys <= y0).any() or (ys >= y1 - 1).any()
    )


def evaluate_mask_for_episode(
    episode_index: int,
    mask: np.ndarray | None,
    *,
    layout_box: tuple[int, int, int, int],
    plane_z_m: float | None,
    arm_name: str,
    calibration: HeadCalibration,
    yaw_symmetry_deg: float,
    frame: int = OBJECT_POSE_FRAME,
) -> EpisodeObjectMeasurement:
    if arm_name not in calibration.arms:
        return EpisodeObjectMeasurement(
            episode_index=episode_index,
            frame=frame,
            status=f"missing calibration for arm {arm_name!r}",
            outline=None,
            pose=None,
        )
    if plane_z_m is None or not math.isfinite(plane_z_m):
        return EpisodeObjectMeasurement(
            episode_index=episode_index,
            frame=frame,
            status="missing object plane_z_m",
            outline=None,
            pose=None,
        )
    if mask is None or not np.any(mask):
        return EpisodeObjectMeasurement(
            episode_index=episode_index,
            frame=frame,
            status=SKIP_EMPTY_MASK,
            outline=None,
            pose=None,
        )
    try:
        outline_px = exterior_outline(mask)
    except ValueError:
        return EpisodeObjectMeasurement(
            episode_index=episode_index,
            frame=frame,
            status=SKIP_DEGENERATE_OUTLINE,
            outline=None,
            pose=None,
        )
    outline_list = np.round(outline_px, 2).tolist()
    if mask_touches_layout_box(mask, layout_box):
        return EpisodeObjectMeasurement(
            episode_index=episode_index,
            frame=frame,
            status=SKIP_BORDER_TOUCH,
            outline=outline_list,
            pose=None,
        )
    try:
        pose = measure_outline_pose(
            outline_px,
            plane_z_m=plane_z_m,
            arm_calibration=calibration.arms[arm_name],
            intrinsics=calibration.intrinsics,
            yaw_symmetry_deg=yaw_symmetry_deg,
        )
    except ValueError as exc:
        return EpisodeObjectMeasurement(
            episode_index=episode_index,
            frame=frame,
            status=str(exc),
            outline=outline_list,
            pose=None,
        )
    return EpisodeObjectMeasurement(
        episode_index=episode_index,
        frame=frame,
        status="ok",
        outline=outline_list,
        pose=pose,
    )


def apply_size_outlier_filter(
    records: Sequence[EpisodeObjectMeasurement],
    *,
    max_size_deviation: float = DEFAULT_MAX_SIZE_DEVIATION,
    min_samples: int = MIN_MEDIAN_SIZE_SAMPLES,
) -> list[EpisodeObjectMeasurement]:
    if max_size_deviation <= 0.0:
        raise ValueError(f"max_size_deviation must be positive, got {max_size_deviation}")
    ok_poses = [rec.pose for rec in records if rec.status == "ok" and rec.pose is not None]
    if len(ok_poses) < min_samples:
        return list(records)
    med_len = float(np.median([p.length_m for p in ok_poses]))
    med_wid = float(np.median([p.width_m for p in ok_poses]))
    if med_len <= 0.0 or med_wid <= 0.0:
        return list(records)
    filtered: list[EpisodeObjectMeasurement] = []
    for rec in records:
        if rec.status != "ok" or rec.pose is None:
            filtered.append(rec)
            continue
        len_dev = abs(rec.pose.length_m - med_len) / med_len
        wid_dev = abs(rec.pose.width_m - med_wid) / med_wid
        if len_dev > max_size_deviation or wid_dev > max_size_deviation:
            reason = (
                f"rectangle size ({rec.pose.length_m * 100:.1f}x{rec.pose.width_m * 100:.1f} cm) "
                f"far from median ({med_len * 100:.1f}x{med_wid * 100:.1f} cm)"
            )
            filtered.append(
                EpisodeObjectMeasurement(
                    episode_index=rec.episode_index,
                    frame=rec.frame,
                    status=reason,
                    outline=rec.outline,
                    pose=None,
                )
            )
        else:
            filtered.append(rec)
    return filtered


def build_object_poses_payload(
    *,
    repo_id: str,
    revision: str,
    camera: str,
    image_size: tuple[int, int],
    calibration_sha256: str,
    objects: Sequence[ObjectSpec],
    plane_z_by_object: Mapping[str, float | None],
    measurements_by_object: Mapping[str, Sequence[EpisodeObjectMeasurement]],
    model: str = MODEL,
    model_revision: str = MODEL_REVISION,
    frame: int = OBJECT_POSE_FRAME,
) -> dict[str, Any]:
    objects_dict: dict[str, Any] = {}
    for obj in objects:
        if obj.layout_box is None:
            continue
        yaw_period = object_yaw_period_deg(obj.yaw_symmetry_deg)
        records = sorted(
            measurements_by_object.get(obj.name, ()),
            key=lambda item: item.episode_index,
        )
        ep_entries: list[dict[str, Any]] = []
        for rec in records:
            pose = rec.pose if rec.status == "ok" else None
            ep_entries.append(
                {
                    "episode_index": rec.episode_index,
                    "frame": rec.frame,
                    "status": rec.status,
                    "outline": rec.outline,
                    "x_m": pose.x_m if pose is not None else None,
                    "y_m": pose.y_m if pose is not None else None,
                    "z_m": pose.z_m if pose is not None else None,
                    "raw_yaw_deg": pose.raw_yaw_deg if pose is not None else None,
                    "folded_yaw_deg": pose.folded_yaw_deg if pose is not None else None,
                    "length_m": pose.length_m if pose is not None else None,
                    "width_m": pose.width_m if pose is not None else None,
                }
            )
        objects_dict[obj.name] = {
            "arm": obj.arm,
            "layout_box": list(obj.layout_box),
            "plane_z_m": plane_z_by_object.get(obj.name),
            "yaw_period_deg": yaw_period,
            "episodes": ep_entries,
        }
    return {
        "version": OBJECT_POSES_SCHEMA_VERSION,
        "repo_id": repo_id,
        "revision": revision,
        "camera": camera,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "frame": int(frame),
        "calibration_sha256": calibration_sha256,
        "model": model,
        "model_revision": model_revision,
        "objects": objects_dict,
    }


def load_object_poses(path: Path) -> tuple[dict[str, Any], str]:
    raw_bytes = path.read_bytes()
    payload = read_json(path, raw_bytes)
    if payload.get("version") != OBJECT_POSES_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported object-poses JSON at {path} "
            f"(expected version={OBJECT_POSES_SCHEMA_VERSION})"
        )
    if not isinstance(payload.get("objects"), dict):
        raise ValueError(f"Invalid object-poses JSON at {path}: missing 'objects' mapping")
    return payload, sha256_bytes(raw_bytes)


def validate_object_poses_sidecar(
    sidecar: Mapping[str, Any],
    *,
    repo_id: str,
    revision: str,
    calibration_sha256: str | None,
) -> None:
    if sidecar.get("version") != OBJECT_POSES_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported object-poses sidecar version: {sidecar.get('version')!r} "
            f"(expected {OBJECT_POSES_SCHEMA_VERSION})"
        )
    if sidecar.get("repo_id") != repo_id or sidecar.get("revision") != revision:
        raise ValueError(
            "Object-poses sidecar repo_id/revision mismatch: "
            f"sidecar=({sidecar.get('repo_id')!r}, {sidecar.get('revision')!r}), "
            f"dataset=({repo_id!r}, {revision!r})"
        )
    sidecar_calib = sidecar.get("calibration_sha256")
    if not isinstance(sidecar_calib, str) or not sidecar_calib:
        raise ValueError("Object-poses sidecar is missing calibration_sha256")
    if calibration_sha256 is None:
        raise ValueError(
            "Object-poses sidecar requires head calibration matching its calibration_sha256, "
            "but no head calibration was loaded"
        )
    if sidecar_calib != calibration_sha256:
        raise ValueError(
            "Object-poses sidecar calibration_sha256 does not match head calibration "
            f"({sidecar_calib[:12]} != {calibration_sha256[:12]})"
        )
    if not isinstance(sidecar.get("objects"), Mapping):
        raise ValueError("Object-poses sidecar is missing 'objects' mapping")


def decode_frame_zero_images(
    dataset: DatasetView,
    camera: str,
    episodes: Sequence[int],
) -> dict[int, Image.Image]:
    by_video: dict[Path, list[tuple[int, float]]] = {}
    for ep_idx in episodes:
        record = dataset.get_episode(ep_idx)
        if record.length <= OBJECT_POSE_FRAME:
            raise ValueError(f"Episode {ep_idx} has no frame {OBJECT_POSE_FRAME}")
        if camera not in record.videos:
            raise KeyError(f"Episode {ep_idx} has no camera {camera!r}")
        span = record.videos[camera]
        by_video.setdefault(span.path, []).append((ep_idx, span.from_timestamp))
    images: dict[int, Image.Image] = {}
    for video_path, targets in by_video.items():
        decoded = decode_video_frames_pyav(video_path, targets, fps=dataset.fps)
        images.update(decoded)
    return images


def _median_z_from_annotations(
    dataset: DatasetView,
    calibration: HeadCalibration,
    objects: Sequence[ObjectSpec],
    annotations_path: Path,
) -> dict[str, float | None]:
    rows = load_annotations(annotations_path)
    by_ep: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_ep.setdefault(row.episode_index, {})[row.subtask] = row
    zs_by_obj: dict[str, list[float]] = {obj.name: [] for obj in objects}
    for ep_idx in sorted(by_ep):
        ep_rows = by_ep[ep_idx]
        arm_states = {arm.arm: arm for arm in dataset.load_episode_arm_states(ep_idx)}
        for obj in objects:
            row = ep_rows.get(obj.subtask)
            if row is None or obj.arm not in arm_states or obj.arm not in calibration.arms:
                continue
            frame = row.grasp if obj.event == "grasp" else row.end - 1
            if frame is None:
                continue
            arm_state = arm_states[obj.arm]
            rot = rotation_6d_to_matrix(arm_state.tcp_rot6d[frame])
            xyz = arm_state.tcp_xyz[frame] + rot @ calibration.arms[obj.arm].tool_offset
            zs_by_obj[obj.name].append(float(xyz[2]))
    return {name: (float(np.median(vals)) if vals else None) for name, vals in zs_by_obj.items()}


def _median_z_from_distribution(
    distribution_path: Path,
    *,
    repo_id: str,
    revision: str,
) -> dict[str, float | None]:
    payload = read_json(distribution_path, distribution_path.read_bytes())
    prov = payload.get("provenance", {})
    if prov.get("repo_id") != repo_id or prov.get("revision") != revision:
        raise ValueError(
            f"Distribution file {distribution_path} does not match dataset ({repo_id}@{revision})"
        )
    if prov.get("position_point") != "tool_point":
        raise ValueError(
            f"Distribution file {distribution_path} must use calibrated tool_point positions"
        )
    return {
        str(item["name"]): (
            float(item["median_z_m"]) if item.get("median_z_m") is not None else None
        )
        for item in payload.get("objects", [])
    }


def resolve_object_plane_z_map(
    task_spec: TaskSpec,
    *,
    dataset: DatasetView,
    calibration: HeadCalibration,
    annotations_path: Path | None = None,
    distribution_path: Path | None = None,
) -> dict[str, float | None]:
    layout_objects = [obj for obj in task_spec.objects if obj.layout_box is not None]
    result: dict[str, float | None] = {}
    need_default: list[ObjectSpec] = []
    for obj in layout_objects:
        if obj.object_plane_z_m is not None:
            result[obj.name] = obj.object_plane_z_m
        elif obj.arm not in calibration.arms:
            result[obj.name] = None
        else:
            need_default.append(obj)
    if not need_default:
        return result

    fallback_medians: dict[str, float | None] = {}
    if distribution_path is not None:
        fallback_medians = _median_z_from_distribution(
            distribution_path, repo_id=dataset.repo_id, revision=dataset.revision
        )
    else:
        ann_candidate = (
            annotations_path
            if annotations_path is not None
            else default_annotations_path(
                dataset.root, repo_id=dataset.repo_id, revision=dataset.revision
            )
        )
        if ann_candidate.is_file():
            fallback_medians = _median_z_from_annotations(
                dataset, calibration, need_default, ann_candidate
            )
        elif annotations_path is not None:
            raise FileNotFoundError(f"Annotations CSV does not exist: {annotations_path}")
        else:
            dist_candidate = default_distribution_path(
                dataset.root, repo_id=dataset.repo_id, revision=dataset.revision
            )
            if dist_candidate.is_file():
                fallback_medians = _median_z_from_distribution(
                    dist_candidate, repo_id=dataset.repo_id, revision=dataset.revision
                )

    missing: list[str] = []
    for obj in need_default:
        z_val = fallback_medians.get(obj.name)
        if z_val is None:
            missing.append(obj.name)
        else:
            result[obj.name] = z_val
    if missing:
        raise ValueError(
            "Cannot determine default plane_z_m (end-effector median_z_m) for object(s): "
            f"{missing}. Set object_plane_z_m in the task spec or supply --annotations / "
            "--distribution."
        )
    return result


def _load_sheet_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _annotate_episode_frame(
    image: Image.Image,
    object_entries: Sequence[
        tuple[str, tuple[int, int, int], Mapping[str, Any], Mapping[str, Any]]
    ],
    label_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for name, color, obj_meta, ep_entry in object_entries:
        box = obj_meta["layout_box"]
        x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        is_ok = ep_entry["status"] == "ok"
        outline = ep_entry.get("outline")
        if is_ok and outline:
            draw.rectangle([(x0, y0), (x1, y1)], outline=(*color, 110), width=1)
            pts = [(float(p[0]), float(p[1])) for p in outline]
            draw.polygon(pts, fill=(*color, 55), outline=(*color, 255))
            draw.line([*pts, pts[0]], fill=(*color, 255), width=2)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            lx = max(2, int(min(xs)))
            ly = max(2, int(min(ys)) - 16)
            bbox = draw.textbbox((lx, ly), name, font=label_font)
            draw.rectangle(
                [(bbox[0] - 2, bbox[1] - 1), (bbox[2] + 2, bbox[3] + 1)],
                fill=(15, 18, 22, 200),
            )
            draw.text((lx, ly), name, fill=(*color, 255), font=label_font)
        else:
            draw.rectangle([(x0, y0), (x1, y1)], outline=(*_SKIP_COLOR, 230), width=2)
            if outline:
                pts = [(float(p[0]), float(p[1])) for p in outline]
                draw.line([*pts, pts[0]], fill=(*_SKIP_COLOR, 200), width=1)
            tag = f"SKIP {name}"
            lx = max(2, x0 + 2)
            ly = max(2, y0 + 2)
            bbox = draw.textbbox((lx, ly), tag, font=label_font)
            draw.rectangle(
                [(bbox[0] - 2, bbox[1] - 1), (bbox[2] + 2, bbox[3] + 1)],
                fill=(127, 29, 29, 220),
            )
            draw.text((lx, ly), tag, fill=(255, 255, 255, 255), font=label_font)
    return Image.alpha_composite(base, overlay).convert("RGB")


def _wrap_text_lines(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.getlength(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_object_pose_sheets(
    images_by_episode: Mapping[int, Image.Image],
    sidecar: Mapping[str, Any],
    output_dir: Path,
    *,
    columns: int = 3,
    per_page: int = 12,
    tile_height: int = 240,
    jpeg_quality: int = 88,
) -> list[Path]:
    if columns <= 0:
        raise ValueError(f"columns must be positive, got {columns}")
    if per_page <= 0:
        raise ValueError(f"per_page must be positive, got {per_page}")
    if tile_height <= 0:
        raise ValueError(f"tile_height must be positive, got {tile_height}")

    objects_map: Mapping[str, Mapping[str, Any]] = sidecar["objects"]
    object_names = list(objects_map.keys())
    colors = {
        name: _OBJECT_COLORS[idx % len(_OBJECT_COLORS)] for idx, name in enumerate(object_names)
    }
    by_ep: dict[int, dict[str, Mapping[str, Any]]] = {}
    for name in object_names:
        for ep_entry in objects_map[name]["episodes"]:
            by_ep.setdefault(int(ep_entry["episode_index"]), {})[name] = ep_entry

    episodes = sorted(ep for ep in by_ep if ep in images_by_episode)
    if not episodes:
        return []

    label_font = _load_sheet_font(12)
    header_font = _load_sheet_font(13)
    skip_font = _load_sheet_font(11)
    header_h = 22
    skip_line_h = 15
    gap = 8

    prepared: list[tuple[int, Image.Image, int, int, list[str]]] = []
    for ep_idx in episodes:
        raw_img = images_by_episode[ep_idx]
        entries = [
            (name, colors[name], objects_map[name], by_ep[ep_idx][name])
            for name in object_names
            if name in by_ep[ep_idx]
        ]
        annotated = _annotate_episode_frame(raw_img, entries, label_font)
        scale = tile_height / annotated.height
        tile_w = max(1, round(annotated.width * scale))
        resized = annotated.resize((tile_w, tile_height), Image.Resampling.LANCZOS)
        skip_lines: list[str] = []
        skip_count = 0
        for name, _, _, ep_entry in entries:
            if ep_entry["status"] != "ok":
                skip_count += 1
                skip_lines.extend(
                    _wrap_text_lines(
                        f"{name}: {ep_entry['status']}",
                        skip_font,
                        max(40, tile_w - 8),
                    )
                )
        ok_count = len(entries) - skip_count
        prepared.append((ep_idx, resized, ok_count, len(entries), skip_lines))

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pages = [prepared[i : i + per_page] for i in range(0, len(prepared), per_page)]
    for page_num, page_items in enumerate(pages, start=1):
        cell_w = max(item[1].width for item in page_items)
        max_skips = max(len(item[4]) for item in page_items)
        footer_h = max_skips * skip_line_h + (4 if max_skips else 0)
        cell_h = header_h + tile_height + footer_h
        n_cols = min(columns, len(page_items))
        n_rows = math.ceil(len(page_items) / n_cols)
        sheet_w = n_cols * cell_w + (n_cols + 1) * gap
        sheet_h = n_rows * cell_h + (n_rows + 1) * gap
        canvas = Image.new("RGB", (sheet_w, sheet_h), color=_BG_COLOR)
        draw = ImageDraw.Draw(canvas)
        for idx, (ep_idx, tile, ok_count, total_count, skip_lines) in enumerate(page_items):
            row, col = divmod(idx, n_cols)
            x0 = gap + col * (cell_w + gap)
            y0 = gap + row * (cell_h + gap)
            draw.rectangle([(x0, y0), (x0 + cell_w - 1, y0 + header_h - 1)], fill=_HEADER_BG)
            hdr_text = f"ep{ep_idx:03d}  {ok_count}/{total_count} ok"
            hdr_color = _OK_TEXT_COLOR if ok_count == total_count else _WARN_TEXT_COLOR
            draw.text((x0 + 6, y0 + 3), hdr_text, fill=hdr_color, font=header_font)
            canvas.paste(tile, (x0, y0 + header_h))
            if skip_lines:
                fy0 = y0 + header_h + tile_height
                draw.rectangle(
                    [(x0, fy0), (x0 + cell_w - 1, y0 + cell_h - 1)],
                    fill=(45, 22, 24),
                )
                for line_idx, text in enumerate(skip_lines):
                    draw.text(
                        (x0 + 4, fy0 + 2 + line_idx * skip_line_h),
                        text,
                        fill=_WARN_TEXT_COLOR,
                        font=skip_font,
                    )
        dest = output_dir / f"object_poses_p{page_num:02d}.jpg"
        canvas.save(dest, format="JPEG", quality=jpeg_quality)
        written.append(dest)
    return written


def extract_dataset_object_poses(
    dataset: DatasetView,
    task_spec: TaskSpec,
    calibration: HeadCalibration,
    *,
    calibration_sha256: str,
    episodes: Sequence[int],
    plane_z_by_object: Mapping[str, float | None],
    segmenter: SamSegmenter,
    max_size_deviation: float = DEFAULT_MAX_SIZE_DEVIATION,
) -> tuple[dict[str, Any], dict[int, Image.Image]]:
    layout_objects = [obj for obj in task_spec.objects if obj.layout_box is not None]
    if not layout_objects:
        raise ValueError("Task spec has no objects with layout_box configured")

    width, height = calibration.image_size
    for obj in layout_objects:
        assert obj.layout_box is not None
        _, _, x1, y1 = obj.layout_box
        if x1 > width or y1 > height:
            raise ValueError(
                f"Object {obj.name!r} layout_box {list(obj.layout_box)} exceeds calibrated "
                f"image size {list(calibration.image_size)}"
            )

    images = decode_frame_zero_images(dataset, calibration.camera, episodes)
    calibrated_objects = [obj for obj in layout_objects if obj.arm in calibration.arms]
    uncalibrated_objects = [obj for obj in layout_objects if obj.arm not in calibration.arms]

    raw_by_object: dict[str, list[EpisodeObjectMeasurement]] = {
        obj.name: [] for obj in layout_objects
    }
    boxes_for_sam = [obj.layout_box for obj in calibrated_objects if obj.layout_box is not None]

    for ep_idx in episodes:
        image = images[ep_idx]
        if image.size != calibration.image_size:
            raise ValueError(
                f"Episode {ep_idx} frame 0 size {image.size} differs from calibration "
                f"image_size {calibration.image_size}"
            )
        masks = segmenter.segment_layout_boxes(image, boxes_for_sam) if boxes_for_sam else []
        for obj, mask in zip(calibrated_objects, masks, strict=True):
            assert obj.layout_box is not None
            raw_by_object[obj.name].append(
                evaluate_mask_for_episode(
                    ep_idx,
                    mask,
                    layout_box=obj.layout_box,
                    plane_z_m=plane_z_by_object.get(obj.name),
                    arm_name=obj.arm,
                    calibration=calibration,
                    yaw_symmetry_deg=obj.yaw_symmetry_deg,
                )
            )
        for obj in uncalibrated_objects:
            assert obj.layout_box is not None
            raw_by_object[obj.name].append(
                evaluate_mask_for_episode(
                    ep_idx,
                    None,
                    layout_box=obj.layout_box,
                    plane_z_m=plane_z_by_object.get(obj.name),
                    arm_name=obj.arm,
                    calibration=calibration,
                    yaw_symmetry_deg=obj.yaw_symmetry_deg,
                )
            )

    filtered_by_object = {
        name: apply_size_outlier_filter(records, max_size_deviation=max_size_deviation)
        for name, records in raw_by_object.items()
    }
    payload = build_object_poses_payload(
        repo_id=dataset.repo_id,
        revision=dataset.revision,
        camera=calibration.camera,
        image_size=calibration.image_size,
        calibration_sha256=calibration_sha256,
        objects=layout_objects,
        plane_z_by_object=plane_z_by_object,
        measurements_by_object=filtered_by_object,
    )
    return payload, images


def run_object_poses_command(args: argparse.Namespace) -> int:
    from lerobot_check.analysis import parse_episode_filter
    from lerobot_check.dataset import resolve_dataset_root

    try:
        root = resolve_dataset_root(args.dataset_root, args.repo_id)
        dataset = DatasetView(root, repo_id=args.repo_id, revision=args.revision)
    except (OSError, ValueError) as exc:
        print(f"[dataset] {exc}", file=sys.stderr)
        return 1

    spec_path = (
        args.task_spec
        if args.task_spec is not None
        else default_task_spec_path(root, repo_id=dataset.repo_id)
    )
    try:
        task_spec = load_task_spec(spec_path)
    except (OSError, ValueError) as exc:
        print(f"[task-spec] {exc}", file=sys.stderr)
        return 1

    calib_path = (
        args.calib
        if args.calib is not None
        else default_calib_head_path(root, repo_id=dataset.repo_id)
    )
    if not calib_path.is_file():
        print(f"[calib] Head calibration file does not exist: {calib_path}", file=sys.stderr)
        return 1
    try:
        calib_bytes = calib_path.read_bytes()
        calibration = load_head_calibration(calib_path)
        calibration_sha256 = sha256_bytes(calib_bytes)
    except (OSError, ValueError) as exc:
        print(f"[calib] {exc}", file=sys.stderr)
        return 1

    if (calibration.repo_id, calibration.revision) != (dataset.repo_id, dataset.revision):
        print(
            "[calib] Head calibration repo_id/revision differs from dataset: "
            f"calib=({calibration.repo_id}, {calibration.revision}), "
            f"dataset=({dataset.repo_id}, {dataset.revision})",
            file=sys.stderr,
        )
        return 1

    try:
        episodes = (
            parse_episode_filter(args.episodes, dataset.total_episodes)
            if args.episodes
            else list(range(dataset.total_episodes))
        )
    except ValueError as exc:
        print(f"[episodes] {exc}", file=sys.stderr)
        return 1

    try:
        plane_z_by_object = resolve_object_plane_z_map(
            task_spec,
            dataset=dataset,
            calibration=calibration,
            annotations_path=args.annotations,
            distribution_path=args.distribution,
        )
        segmenter = SamSegmenter(args.device)
        payload, images = extract_dataset_object_poses(
            dataset,
            task_spec,
            calibration,
            calibration_sha256=calibration_sha256,
            episodes=episodes,
            plane_z_by_object=plane_z_by_object,
            segmenter=segmenter,
        )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"[object-poses] {exc}", file=sys.stderr)
        return 1

    out_path = (
        args.output
        if args.output is not None
        else default_object_poses_path(root, repo_id=dataset.repo_id, revision=dataset.revision)
    )
    write_json(payload, out_path)
    for name, obj_data in payload["objects"].items():
        eps = obj_data["episodes"]
        ok = sum(1 for item in eps if item["status"] == "ok")
        print(f"{name}: {ok}/{len(eps)} ok (skipped {len(eps) - ok})")
    print(f"Wrote object poses JSON: {out_path}")

    if args.sheet is not None:
        try:
            sheet_paths = render_object_pose_sheets(images, payload, args.sheet)
        except (OSError, ValueError) as exc:
            print(f"[sheet] {exc}", file=sys.stderr)
            return 1
        for path in sheet_paths:
            print(path)
    return 0


def configure_object_poses_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to a local LeRobot v3.0 dataset directory.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "Dataset repo ID (e.g. Xense/pack_iphone_box_1119). "
            "Used to resolve local cache path if --dataset-root is omitted."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Override detected dataset revision.",
    )
    parser.add_argument(
        "--task-spec",
        type=Path,
        default=None,
        help=(
            "Path to the YAML task spec file with layout_box entries "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml)."
        ),
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=None,
        help=(
            "Path to head-camera calibration head.json "
            "(default: <lerobot_root>/.xense-viewer/calib/<org>__<name>.head.json)."
        ),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=(
            "Optional annotation CSV used to compute default plane_z_m when object_plane_z_m is "
            "omitted (default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv)."
        ),
    )
    parser.add_argument(
        "--distribution",
        type=Path,
        default=None,
        help=(
            "Optional calibrated distribution.json supplying default median_z_m planes when "
            "object_plane_z_m is omitted."
        ),
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=None,
        help="Episode indices or ranges (e.g. '0,5,10-20'). Default: all episodes.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device for SAM 2.1 segmentation (default: cpu).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output object-poses sidecar JSON path "
            "(default: <lerobot_root>/.xense-viewer/object-poses/<org>__<name>@<revision>.json)."
        ),
    )
    parser.add_argument(
        "--sheet",
        type=Path,
        default=None,
        help="Optional directory to write JPEG review contact sheets.",
    )
    parser.set_defaults(func=run_object_poses_command)
