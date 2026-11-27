from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
from PIL import Image

from lerobot_check.calib import (
    HeadCalibration,
    backproject_pixels_to_plane_z,
    project_base_points,
)
from lerobot_check.dataset import DatasetView
from lerobot_check.render import decode_video_frames_pyav


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_head_frame(dataset: DatasetView, camera: str, episode: int, frame: int) -> Image.Image:
    record = dataset.get_episode(episode)
    if not 0 <= frame < record.length:
        raise ValueError(f"Frame {frame} outside episode {episode} (length {record.length})")
    span = record.videos[camera]
    timestamp = span.from_timestamp + frame / dataset.fps
    return decode_video_frames_pyav(span.path, [(frame, timestamp)], fps=dataset.fps)[frame]


def exterior_outline(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("Segmentation produced an empty mask")
    outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 0.5, True).reshape(-1, 2)
    if len(outline) < 3 or cv2.contourArea(outline) <= 0:
        raise ValueError("Segmentation produced a degenerate outline")
    return outline.astype(np.float64)


def rotate_xy(points: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    c, s = math.cos(radians), math.sin(radians)
    return points @ np.array([[c, s], [-s, c]])


def closest_equivalent_yaw(target: float, reference: float, period: float) -> float:
    return reference + (target - reference + period / 2) % period - period / 2


def make_local_outline(
    pixels: np.ndarray,
    anchor: Mapping[str, Any],
    plane: float,
    calibration: HeadCalibration,
) -> np.ndarray:
    arm = calibration.arms[anchor["arm"]]
    base = backproject_pixels_to_plane_z(pixels, plane, arm.T_cam_base, calibration.intrinsics)
    return rotate_xy(base[:, :2] - [anchor["x_m"], anchor["y_m"]], -anchor["raw_yaw_deg"])


def outline_base_points(
    template: Mapping[str, Any], item: Mapping[str, Any], period: float
) -> np.ndarray:
    local = np.asarray(template["outline_xy_m"], dtype=np.float64)
    if local.ndim != 2 or local.shape[1] != 2 or len(local) < 3 or not np.isfinite(local).all():
        raise ValueError("Invalid local outline")
    yaw = closest_equivalent_yaw(item["yaw_deg"], template["source"]["raw_yaw_deg"], period)
    xy = rotate_xy(local, yaw) + np.array([item["x_m"], item["y_m"]])
    plane = item["z_m"] + template["height_offset_m"]
    return np.column_stack((xy, np.full(len(xy), plane)))


def project_outline(
    template: Mapping[str, Any],
    item: Mapping[str, Any],
    period: float,
    calibration: HeadCalibration,
) -> list[list[float]]:
    xyz = outline_base_points(template, item, period)
    arm = calibration.arms[item["arm"]]
    pixels = np.round(project_base_points(xyz, arm.T_cam_base, calibration.intrinsics), 3)
    width, height = calibration.image_size
    if not np.isfinite(pixels).all() or ((pixels < 0) | (pixels >= [width, height])).any():
        raise ValueError("Target outline extends beyond the calibrated image")
    return pixels.tolist()


def verify_suggestion_projection(
    item: Mapping[str, Any], arrow_length_m: float, calibration: HeadCalibration
) -> None:
    arm = calibration.arms[item["arm"]]
    origin = np.array([item["x_m"], item["y_m"], item["z_m"]])
    yaw = math.radians(item["yaw_deg"])
    tip = origin + arrow_length_m * np.array([math.cos(yaw), math.sin(yaw), 0.0])
    expected = project_base_points(np.stack([origin, tip]), arm.T_cam_base, calibration.intrinsics)
    actual = np.asarray([item["pixel"], item["arrow_end_pixel"]], dtype=np.float64)
    if actual.shape != (2, 2) or not np.allclose(actual, expected, rtol=0, atol=0.01):
        raise ValueError("Suggestions do not match head.json; regenerate suggestions before masks")


def make_bundle(
    suggestions: Mapping[str, Any],
    calibration: HeadCalibration,
    templates: Sequence[Mapping[str, Any]],
    *,
    calibration_sha256: str,
    suggestions_sha256: str,
) -> dict[str, Any]:
    provenance = suggestions["provenance"]
    repo_id, revision = provenance["repo_id"], provenance["revision"]
    if suggestions.get("version") != 1 or provenance["position_point"] != "tool_point":
        raise ValueError("Expected calibrated v1 suggestions (position_point=tool_point)")
    if (repo_id, revision) != (calibration.repo_id, calibration.revision):
        raise ValueError("Suggestions and calibration refer to different dataset revisions")
    objects = {obj["object"]: obj for obj in suggestions["objects"]}
    by_object: dict[str, Mapping[str, Any]] = {}
    for template in templates:
        name = template["object"]
        if name not in objects or name in by_object:
            raise ValueError(f"Unexpected or duplicate template for {name}")
        expected = (1, repo_id, revision, calibration_sha256)
        actual = tuple(
            template[k] for k in ("version", "repo_id", "revision", "calibration_sha256")
        )
        if actual != expected:
            raise ValueError(f"Template for {name} uses another dataset or calibration")
        if not template.get("reason"):
            obj = objects[name]
            if template["arm"] != obj["arm"] or not math.isclose(
                template["z_plane_m"], obj["z_plane_m"], rel_tol=0, abs_tol=1e-6
            ):
                raise ValueError(f"Template plane/arm for {name} no longer matches suggestions")
        by_object[name] = template

    targets = []
    for kind, items in (
        ("placement", suggestions["placements"]),
        ("test_point", suggestions["test_points"]),
    ):
        for item in items:
            name = item["object"]
            template = by_object.get(name)
            outline = None
            reason = (
                "No reviewed template for this object"
                if template is None
                else template.get("reason")
            )
            if template is not None and not reason:
                verify_suggestion_projection(item, provenance["arrow_length_m"], calibration)
                try:
                    outline = project_outline(
                        template, item, objects[name]["yaw_symmetry_deg"], calibration
                    )
                except (ValueError, KeyError) as exc:
                    reason = str(exc)
            targets.append(
                {
                    "kind": kind,
                    **{k: item[k] for k in ("object", "index", "pixel", "arrow_end_pixel")},
                    **({"tier": item["tier"]} if kind == "test_point" else {}),
                    "outline": outline,
                    "reason": reason,
                }
            )
    return {
        "version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "camera": calibration.camera,
        "image_size": list(calibration.image_size),
        "calibration_sha256": calibration_sha256,
        "suggestions_sha256": suggestions_sha256,
        "sources": {name: template.get("source") for name, template in by_object.items()},
        "targets": targets,
    }
