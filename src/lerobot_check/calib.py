"""Head-camera calibration frame selection, extrinsic+offset solver, and projection geometry."""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from lerobot_check.dataset import DatasetView, rotation_6d_to_matrix

CALIB_SCHEMA_VERSION = 1
DEFAULT_CALIB_CAMERA = "observation.images.head"
DEFAULT_FRAMES_PER_ARM = 15
# Verified on bi_flexiv_rizon4_rt (pack_iphone_box_1119): 0.0 is closed, 1.0 is open;
# gripper <= 0.05 keeps tightly closed fingertips while excluding visibly open jaws (>= 0.06).
DEFAULT_MAX_GRIPPER_POS = 0.05
# Among closed-gripper frames on pack_iphone_box_1119, ~half move at <= 0.03 m/s,
# keeping motion blur and camera/state sync error sub-pixel while preserving pose diversity.
DEFAULT_MAX_SPEED_MPS = 0.03
ROBUST_LOSS_SCALE_PX = 3.0
OUTLIER_MEDIAN_MULTIPLIER = 3.0
OUTLIER_MIN_PX = 15.0
# Weight balancing 10 cm TCP translation with ~29 deg rotation (||R1 - R2||_F = 1.0 at ~41 deg).
_POS_SCALE_M = 0.10
_ROT_SCALE = 0.50


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsic matrix and distortion parameters loaded from ``intrinsics.json``.

    Attributes:
        camera: Camera feature key (e.g. ``"observation.images.head"``).
        image_size: ``(width, height)`` in pixels.
        fx: Horizontal focal length in pixels.
        fy: Vertical focal length in pixels.
        cx: Principal point X coordinate in pixels.
        cy: Principal point Y coordinate in pixels.
        distortion_model: Distortion model label (e.g. ``"inverse_brown_conrady"``).
        distortion: 5-element distortion vector ``(k1, k2, p1, p2, k3)``.
        raw: Original JSON mapping preserved verbatim for copying into ``head.json``.
    """

    camera: str
    image_size: tuple[int, int]
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion: tuple[float, float, float, float, float]
    raw: dict[str, Any]

    @property
    def camera_matrix(self) -> np.ndarray:
        """Return the 3x3 pinhole camera matrix ``K``."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        """Return distortion coefficients as a float64 array of shape ``(5,)``."""
        return np.array(self.distortion, dtype=np.float64)


@dataclass(frozen=True)
class ArmCalibration:
    """Calibrated extrinsic transform and tool-tip offset for a single robot arm.

    Attributes:
        arm: Arm identifier (``"left"`` or ``"right"``).
        T_cam_base: 4x4 homogeneous transform mapping base-frame points into camera frame.
        tool_offset: 3-vector ``(ox, oy, oz)`` in the TCP frame (meters).
        n_points: Number of valid annotated points used to solve this arm.
        rms_px: In-sample Euclidean reprojection RMS error over all points (pixels).
        loo_rms_px: Leave-one-out cross-validation Euclidean RMS error over all points
            (pixels), or ``None`` when fewer than 6 points are available.
        residuals: Per-point reprojection residuals ``(id, du, dv)`` where
            ``du = u_pred - u_obs`` and ``dv = v_pred - v_obs``.
        inlier_rms_px: Euclidean reprojection RMS over non-outlier points (pixels).
        inlier_loo_rms_px: Leave-one-out Euclidean RMS over non-outlier points (pixels).
        n_inliers: Number of non-outlier points.
        outlier_threshold_px: Threshold ``max(3 * median_err, 15.0)`` used to flag outliers.
        outlier_ids: Tuple of frame IDs whose residual magnitude exceeded the threshold.
    """

    arm: str
    T_cam_base: np.ndarray
    tool_offset: np.ndarray
    n_points: int
    rms_px: float
    loo_rms_px: float | None
    residuals: tuple[tuple[str, float, float], ...]
    inlier_rms_px: float | None = None
    inlier_loo_rms_px: float | None = None
    n_inliers: int | None = None
    outlier_threshold_px: float | None = None
    outlier_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class HeadCalibration:
    """Loaded ``head.json`` head-camera calibration for one or more arms.

    Attributes:
        repo_id: Dataset repository ID.
        revision: Dataset revision string.
        camera: Camera feature key.
        image_size: ``(width, height)`` in pixels.
        intrinsics: Camera intrinsic parameters.
        arms: Mapping from arm name to ``ArmCalibration``.
    """

    repo_id: str
    revision: str
    camera: str
    image_size: tuple[int, int]
    intrinsics: CameraIntrinsics
    arms: dict[str, ArmCalibration]


def parse_intrinsics(raw: Mapping[str, Any]) -> CameraIntrinsics:
    """Validate and convert a raw ``intrinsics.json`` mapping into ``CameraIntrinsics``."""
    if raw.get("version") != CALIB_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported intrinsics version: {raw.get('version')!r} "
            f"(expected {CALIB_SCHEMA_VERSION})"
        )
    image_size = raw.get("image_size")
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError(f"intrinsics.image_size must be [width, height], got {image_size!r}")
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"intrinsics.image_size must be positive, got {[width, height]}")

    fx = float(raw["fx"])
    fy = float(raw["fy"])
    cx = float(raw["cx"])
    cy = float(raw["cy"])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Focal lengths fx/fy must be positive, got fx={fx}, fy={fy}")

    raw_dist = raw.get("distortion")
    if not isinstance(raw_dist, list) or len(raw_dist) != 5:
        raise ValueError(f"intrinsics.distortion must be a 5-element list, got {raw_dist!r}")
    dist = (
        float(raw_dist[0]),
        float(raw_dist[1]),
        float(raw_dist[2]),
        float(raw_dist[3]),
        float(raw_dist[4]),
    )
    camera = str(raw.get("camera", "")).strip()
    if not camera:
        raise ValueError("intrinsics.camera must be a non-empty string")
    distortion_model = str(raw.get("distortion_model", "")).strip()
    return CameraIntrinsics(
        camera=camera,
        image_size=(width, height),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion_model=distortion_model,
        distortion=dist,
        raw=dict(raw),
    )


def load_intrinsics(path: Path) -> CameraIntrinsics:
    """Load camera intrinsics strictly from ``path`` (never using hard-coded defaults)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Intrinsics file {path} must contain a JSON object")
    return parse_intrinsics(raw)


def load_head_calibration(path: Path) -> HeadCalibration:
    """Load a calibrated ``head.json`` file into a ``HeadCalibration``."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != CALIB_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported head calibration at {path} (expected version={CALIB_SCHEMA_VERSION})"
        )
    intrinsics = parse_intrinsics(raw["intrinsics"])
    image_size_raw = raw["image_size"]
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
    arms: dict[str, ArmCalibration] = {}
    for arm_name, entry in raw["arms"].items():
        T = np.asarray(entry["T_cam_base"], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T_cam_base for arm {arm_name!r} must be 4x4, got {T.shape}")
        offset = np.asarray(entry["tool_offset"], dtype=np.float64)
        if offset.shape != (3,):
            raise ValueError(f"tool_offset for arm {arm_name!r} must have length 3")
        residuals = tuple(
            (str(r["id"]), float(r["du"]), float(r["dv"])) for r in entry.get("residuals", [])
        )
        outlier_ids = tuple(
            str(r["id"]) for r in entry.get("residuals", []) if r.get("outlier") is True
        )
        loo_val = entry.get("loo_rms_px")
        inlier_rms_val = entry.get("inlier_rms_px")
        inlier_loo_val = entry.get("inlier_loo_rms_px")
        n_inliers_val = entry.get("n_inliers")
        thr_val = entry.get("outlier_threshold_px")
        arms[str(arm_name)] = ArmCalibration(
            arm=str(arm_name),
            T_cam_base=T,
            tool_offset=offset,
            n_points=int(entry["n_points"]),
            rms_px=float(entry["rms_px"]),
            loo_rms_px=float(loo_val) if loo_val is not None else None,
            residuals=residuals,
            inlier_rms_px=float(inlier_rms_val) if inlier_rms_val is not None else None,
            inlier_loo_rms_px=float(inlier_loo_val) if inlier_loo_val is not None else None,
            n_inliers=int(n_inliers_val) if n_inliers_val is not None else None,
            outlier_threshold_px=float(thr_val) if thr_val is not None else None,
            outlier_ids=outlier_ids,
        )
    return HeadCalibration(
        repo_id=str(raw.get("repo_id", "")),
        revision=str(raw.get("revision", "")),
        camera=str(raw.get("camera", intrinsics.camera)),
        image_size=image_size,
        intrinsics=intrinsics,
        arms=arms,
    )


def project_camera_points(points_cam: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Project 3D camera-frame points to distorted pixel coordinates ``(u, v)``.

    Implements the standard Brown-Conrady / RealSense ``inverse_brown_conrady``
    forward distortion model matching ``cv2.projectPoints`` with ``(k1, k2, p1, p2, k3)``.

    Args:
        points_cam: Array of shape ``(N, 3)`` in camera coordinates (``Z > 0``).
        intrinsics: Camera intrinsic parameters.

    Returns:
        Float64 array of shape ``(N, 2)`` containing ``[u, v]`` pixel coordinates.

    Raises:
        ValueError: If any point lies behind or on the camera plane (``Z <= 1e-9``).
    """
    pts = np.asarray(points_cam, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points_cam of shape (N, 3), got {pts.shape}")
    z = pts[:, 2]
    if np.any(z <= 1e-9):
        raise ValueError("Cannot project point with non-positive camera depth Z")

    x = pts[:, 0] / z
    y = pts[:, 1] / z
    k1, k2, p1, p2, k3 = intrinsics.distortion
    if k1 == 0.0 and k2 == 0.0 and p1 == 0.0 and p2 == 0.0 and k3 == 0.0:
        xd, yd = x, y
    else:
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    u = intrinsics.fx * xd + intrinsics.cx
    v = intrinsics.fy * yd + intrinsics.cy
    return np.column_stack([u, v])


def project_base_points(
    points_base: np.ndarray,
    T_cam_base: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Project base-frame 3D points ``(N, 3)`` into head-camera pixel coordinates ``(N, 2)``."""
    pts = np.asarray(points_base, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    T = np.asarray(T_cam_base, dtype=np.float64)
    pts_cam = pts @ T[:3, :3].T + T[:3, 3]
    return project_camera_points(pts_cam, intrinsics)


def backproject_pixels_to_plane_z(
    pixels: np.ndarray,
    z_plane_m: float,
    T_cam_base: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Back-project image pixels onto the horizontal plane ``z = z_plane_m`` in base frame.

    Args:
        pixels: Array of shape ``(N, 2)`` containing ``[u, v]`` pixel coordinates.
        z_plane_m: Target horizontal plane height in the arm's base frame (meters).
        T_cam_base: 4x4 transform mapping base-frame points to camera frame.
        intrinsics: Camera intrinsics and distortion parameters.

    Returns:
        Float64 array of shape ``(N, 3)`` of base-frame intersections ``[x, y, z_plane_m]``.

    Raises:
        ValueError: If a ray is parallel to the plane or intersects behind the camera.
    """
    px = np.asarray(pixels, dtype=np.float64)
    if px.ndim == 1:
        px = px.reshape(1, 2)
    if px.ndim != 2 or px.shape[1] != 2:
        raise ValueError(f"Expected pixels of shape (N, 2), got {px.shape}")

    if all(c == 0.0 for c in intrinsics.distortion):
        norm_x = (px[:, 0] - intrinsics.cx) / intrinsics.fx
        norm_y = (px[:, 1] - intrinsics.cy) / intrinsics.fy
    else:
        undist = cv2.undistortPoints(
            px.reshape(-1, 1, 2),
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
        ).reshape(-1, 2)
        norm_x = undist[:, 0]
        norm_y = undist[:, 1]

    rays_cam = np.column_stack([norm_x, norm_y, np.ones(len(px), dtype=np.float64)])
    T = np.asarray(T_cam_base, dtype=np.float64)
    R_cb = T[:3, :3]
    t_cb = T[:3, 3]
    # Inverse rigid transform: p_base = R_cb^T (p_cam - t_cb).
    cam_origin_base = -(R_cb.T @ t_cb)
    rays_base = rays_cam @ R_cb  # equivalent to (R_cb^T @ rays_cam.T).T

    dz = rays_base[:, 2]
    if np.any(np.abs(dz) <= 1e-9):
        raise ValueError("Camera ray is parallel to the target z-plane")
    t_param = (float(z_plane_m) - float(cam_origin_base[2])) / dz
    if np.any(t_param <= 0.0):
        raise ValueError(
            f"Back-projected ray intersects z={z_plane_m:.4f} m behind the camera "
            "(check calibration extrinsics and polygon coordinates)"
        )

    pts = cam_origin_base[None, :] + t_param[:, None] * rays_base
    pts[:, 2] = float(z_plane_m)
    return pts


def _tcp_pose_features(tcps: np.ndarray) -> np.ndarray:
    """Build scaled 12D ``(xyz, R_flat)`` features for farthest point sampling."""
    xyz_scaled = tcps[:, 0:3] / _POS_SCALE_M
    rots = rotation_6d_to_matrix(tcps[:, 3:9])
    rot_scaled = rots.reshape(-1, 9) / _ROT_SCALE
    return np.concatenate([xyz_scaled, rot_scaled], axis=1)


def _farthest_point_sample(
    features: np.ndarray,
    count: int,
    *,
    seed_features: np.ndarray | None = None,
    excluded_mask: np.ndarray | None = None,
) -> list[int]:
    """Select ``count`` indices by greedy farthest point sampling in feature space.

    When ``seed_features`` is omitted, the first selected point is the eligible
    point farthest from the dataset feature mean so selection is fully deterministic.
    """
    n = len(features)
    if n == 0 or count <= 0:
        return []

    excluded = (
        np.zeros(n, dtype=bool)
        if excluded_mask is None
        else np.asarray(excluded_mask, dtype=bool).copy()
    )
    available = int(np.count_nonzero(~excluded))
    target = min(count, available)
    if target == 0:
        return []

    if seed_features is not None and len(seed_features) > 0:
        min_sq_dist = np.full(n, np.inf, dtype=np.float64)
        for seed in seed_features:
            diff = features - seed[None, :]
            sq = np.einsum("ij,ij->i", diff, diff)
            np.minimum(min_sq_dist, sq, out=min_sq_dist)
        selected: list[int] = []
    else:
        mean_feat = np.mean(features[~excluded], axis=0)
        diff = features - mean_feat[None, :]
        dist_from_mean = np.einsum("ij,ij->i", diff, diff)
        dist_from_mean[excluded] = -1.0
        first_idx = int(np.argmax(dist_from_mean))
        selected = [first_idx]
        excluded[first_idx] = True
        diff_first = features - features[first_idx][None, :]
        min_sq_dist = np.einsum("ij,ij->i", diff_first, diff_first)

    while len(selected) < target:
        scores = min_sq_dist.copy()
        scores[excluded] = -1.0
        next_idx = int(np.argmax(scores))
        if scores[next_idx] < 0.0:
            break
        selected.append(next_idx)
        excluded[next_idx] = True
        diff_next = features - features[next_idx][None, :]
        sq_next = np.einsum("ij,ij->i", diff_next, diff_next)
        np.minimum(min_sq_dist, sq_next, out=min_sq_dist)

    return selected


def _next_arm_frame_index(existing_frames: Sequence[Mapping[str, Any]], arm: str) -> int:
    """Return the next sequential integer suffix for ``f'{arm}-NNN'`` IDs."""
    max_num = -1
    prefix = f"{arm}-"
    for item in existing_frames:
        if item.get("arm") != arm:
            continue
        fid = str(item.get("id", ""))
        if fid.startswith(prefix):
            suffix = fid[len(prefix) :]
            if suffix.isdigit():
                max_num = max(max_num, int(suffix))
    return max_num + 1


def _compute_tcp_speeds_mps(
    ep_indices: np.ndarray,
    frame_indices: np.ndarray,
    xyz: np.ndarray,
    *,
    fps: float,
) -> np.ndarray:
    """Compute per-frame TCP linear speed (m/s) via within-episode central differences.

    Uses central difference ``||p[i+1] - p[i-1]|| / ((f[i+1] - f[i-1]) / fps)`` for
    interior frames of each episode, one-sided differences at episode boundaries, and
    ``NaN`` for 1-frame episodes. Never differentiates across episode boundaries.
    """
    if fps <= 0.0 or not math.isfinite(fps):
        raise ValueError(f"fps must be positive and finite, got {fps}")

    n = len(xyz)
    speeds = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return speeds

    ep_arr = np.asarray(ep_indices, dtype=np.int64)
    fr_arr = np.asarray(frame_indices, dtype=np.int64)
    pos = np.asarray(xyz, dtype=np.float64)

    # Sort row indices by (episode_index, frame_index) so episodes form contiguous runs
    # ordered by frame_index even if input rows are out of order.
    order = np.lexsort((fr_arr, ep_arr))
    sorted_ep = ep_arr[order]
    boundaries = np.flatnonzero(np.diff(sorted_ep) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))

    for s, e in zip(starts, ends, strict=True):
        length = int(e - s)
        if length < 2:
            continue
        idx = order[s:e]
        p = pos[idx]
        f = fr_arr[idx].astype(np.float64)

        ep_speeds = np.full(length, np.nan, dtype=np.float64)

        dt0 = (f[1] - f[0]) / fps
        if dt0 > 0.0:
            ep_speeds[0] = float(np.linalg.norm(p[1] - p[0])) / dt0

        dt_last = (f[-1] - f[-2]) / fps
        if dt_last > 0.0:
            ep_speeds[-1] = float(np.linalg.norm(p[-1] - p[-2])) / dt_last

        if length >= 3:
            dt_mid = (f[2:] - f[:-2]) / fps
            valid = dt_mid > 0.0
            norms = np.linalg.norm(p[2:] - p[:-2], axis=1)
            mid_speeds = np.full(length - 2, np.nan, dtype=np.float64)
            mid_speeds[valid] = norms[valid] / dt_mid[valid]
            ep_speeds[1:-1] = mid_speeds

        speeds[idx] = ep_speeds

    return speeds


def select_calib_frames(
    dataset: DatasetView,
    *,
    camera: str = DEFAULT_CALIB_CAMERA,
    count_per_arm: int = DEFAULT_FRAMES_PER_ARM,
    existing: Mapping[str, Any] | None = None,
    append_per_arm: int | None = None,
    episodes: Sequence[int] | None = None,
    max_gripper_pos: float | None = DEFAULT_MAX_GRIPPER_POS,
    max_speed_mps: float | None = DEFAULT_MAX_SPEED_MPS,
    arms: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Select diverse calibration frames per arm via farthest point sampling.

    By default restricts newly selected frames to closed-gripper
    (``gripper <= max_gripper_pos``) and slow-moving (``speed <= max_speed_mps``)
    frames so the fingertip midpoint is sharp and unaffected by motion blur or
    camera/state timing offset. When ``existing`` and ``append_per_arm`` are
    provided, all existing entries and IDs are preserved verbatim (and still serve
    as FPS seeds) while ``append_per_arm`` new frames are appended for each arm in
    ``arms`` (or all arms when ``arms`` is ``None``).

    Args:
        dataset: Open ``DatasetView``.
        camera: Head camera feature key (default ``"observation.images.head"``).
        count_per_arm: Number of frames to select per arm in fresh mode.
        existing: Previously saved ``frames.json`` mapping when appending.
        append_per_arm: Number of additional frames per arm to append to ``existing``.
        episodes: Optional subset of episodes to sample from (default all episodes).
        max_gripper_pos: Maximum allowed gripper opening for newly selected frames
            (default ``0.05``; pass ``None`` to disable filtering).
        max_speed_mps: Maximum allowed TCP linear speed in m/s for newly selected
            frames (default ``0.03``; pass ``None`` to disable filtering).
        arms: Optional subset of arm names to add new frames for (default all arms).

    Returns:
        JSON-serializable dictionary matching the ``frames.json`` v1 schema.
    """
    if append_per_arm is not None:
        if append_per_arm < 1:
            raise ValueError(f"append_per_arm must be >= 1, got {append_per_arm}")
        if existing is None:
            raise ValueError("existing frames payload is required when append_per_arm is set")
        if existing.get("version") != CALIB_SCHEMA_VERSION:
            raise ValueError(f"Unsupported existing frames version: {existing.get('version')!r}")
        camera = str(existing.get("camera", camera))
        existing_frames = [copy.deepcopy(dict(item)) for item in existing.get("frames", [])]
        to_pick = append_per_arm
    else:
        if count_per_arm < 1:
            raise ValueError(f"count_per_arm must be >= 1, got {count_per_arm}")
        existing_frames = []
        to_pick = count_per_arm

    if max_gripper_pos is not None and (math.isnan(max_gripper_pos) or max_gripper_pos < 0.0):
        raise ValueError(f"max_gripper_pos must be non-negative, got {max_gripper_pos}")
    if max_speed_mps is not None and (math.isnan(max_speed_mps) or max_speed_mps < 0.0):
        raise ValueError(f"max_speed_mps must be non-negative, got {max_speed_mps}")

    width, height = dataset.camera_image_size(camera)
    if hasattr(dataset, "load_arm_calib_table"):
        ep_indices, frame_indices, tcp_by_arm, gripper_by_arm = dataset.load_arm_calib_table(
            episodes=episodes
        )
    else:
        ep_indices, frame_indices, tcp_by_arm = dataset.load_arm_tcp_table(episodes=episodes)
        gripper_by_arm = {}

    available_arms = list(tcp_by_arm.keys())
    if arms is not None:
        requested_arms: list[str] = []
        for name in arms:
            if name not in requested_arms:
                requested_arms.append(name)
        if not requested_arms:
            raise ValueError("arms must not be empty")
        unknown = [name for name in requested_arms if name not in tcp_by_arm]
        if unknown:
            raise ValueError(f"Unknown arm(s): {unknown}; available arms: {available_arms}")
        target_arms = requested_arms
    else:
        target_arms = available_arms

    fps = float(dataset.fps)
    frames_out: list[dict[str, Any]] = list(existing_frames)
    for arm_name in target_arms:
        tcps = tcp_by_arm[arm_name]
        if len(tcps) == 0:
            continue
        features = _tcp_pose_features(tcps)
        speeds = _compute_tcp_speeds_mps(ep_indices, frame_indices, tcps[:, 0:3], fps=fps)

        arm_existing = [item for item in existing_frames if item.get("arm") == arm_name]
        existing_pairs = {
            (int(item["episode_index"]), int(item["frame_index"])) for item in arm_existing
        }
        excluded = np.array(
            [
                (int(e), int(f)) in existing_pairs
                for e, f in zip(ep_indices, frame_indices, strict=True)
            ],
            dtype=bool,
        )
        active_filters: list[str] = []
        if max_gripper_pos is not None and arm_name in gripper_by_arm:
            excluded |= gripper_by_arm[arm_name] > max_gripper_pos
            active_filters.append(f"gripper <= {max_gripper_pos:g}")
        if max_speed_mps is not None:
            excluded |= np.isnan(speeds) | (speeds > max_speed_mps)
            active_filters.append(f"speed <= {max_speed_mps:g} m/s")

        if np.all(excluded):
            filter_desc = " and ".join(active_filters) if active_filters else "current filters"
            raise ValueError(
                f"No eligible frames for arm {arm_name!r} with {filter_desc}; "
                "adjust --max-speed or --max-gripper if needed."
            )

        seed_feats: np.ndarray | None = None
        if arm_existing:
            seed_tcps = np.array([item["tcp"] for item in arm_existing], dtype=np.float64)
            seed_feats = _tcp_pose_features(seed_tcps)

        chosen = _farthest_point_sample(
            features,
            to_pick,
            seed_features=seed_feats,
            excluded_mask=excluded,
        )
        next_num = _next_arm_frame_index(existing_frames, arm_name)
        for offset_i, row_idx in enumerate(chosen):
            speed_val = float(speeds[row_idx])
            frames_out.append(
                {
                    "id": f"{arm_name}-{next_num + offset_i:03d}",
                    "arm": arm_name,
                    "episode_index": int(ep_indices[row_idx]),
                    "frame_index": int(frame_indices[row_idx]),
                    "tcp": [float(v) for v in tcps[row_idx]],
                    "speed_mps": None if math.isnan(speed_val) else round(speed_val, 4),
                }
            )

    return {
        "version": CALIB_SCHEMA_VERSION,
        "repo_id": dataset.repo_id,
        "revision": dataset.revision,
        "camera": camera,
        "image_size": [width, height],
        "frames": frames_out,
    }


def _rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return np.asarray(rot, dtype=np.float64)


def _pose_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rodrigues_to_matrix(rvec)
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def _soft_l1_score(errs_px: np.ndarray, scale_px: float = 10.0) -> float:
    scaled = errs_px / scale_px
    return float(np.sum(2.0 * (np.sqrt(1.0 + scaled**2) - 1.0)))


def _initial_pnp_candidates(
    p_tcp: np.ndarray,
    pixels: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Estimate robust initial ``(rvec, tvec)`` candidates with ``tool_offset = 0``."""
    obj_pts = np.ascontiguousarray(p_tcp, dtype=np.float64).reshape(-1, 1, 3)
    img_pts = np.ascontiguousarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    K = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs

    scored: list[tuple[float, np.ndarray, np.ndarray]] = []

    def _record(ok: bool, rvec: Any, tvec: Any) -> None:
        if not ok or rvec is None or tvec is None:
            return
        rvec_1d = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec_1d = np.asarray(tvec, dtype=np.float64).reshape(3)
        if not (np.all(np.isfinite(rvec_1d)) and np.all(np.isfinite(tvec_1d))):
            return
        R = _rodrigues_to_matrix(rvec_1d)
        pts_cam = p_tcp @ R.T + tvec_1d
        if np.any(pts_cam[:, 2] <= 1e-4):
            return
        pred = project_camera_points(pts_cam, intrinsics)
        errs = np.linalg.norm(pred - pixels, axis=1)
        scored.append((_soft_l1_score(errs, scale_px=10.0), rvec_1d, tvec_1d))

    methods = [cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP]
    if len(p_tcp) >= 6:
        methods.append(cv2.SOLVEPNP_ITERATIVE)

    for flag in methods:
        try:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=flag)
        except cv2.error:
            continue
        _record(bool(ok), rvec, tvec)

    if len(p_tcp) >= 6:
        for reproj_err in (25.0, 50.0):
            try:
                ok, rvec, tvec, _ = cv2.solvePnPRansac(
                    obj_pts,
                    img_pts,
                    K,
                    dist,
                    reprojectionError=reproj_err,
                    iterationsCount=500,
                    confidence=0.999,
                    flags=cv2.SOLVEPNP_EPNP,
                )
            except cv2.error:
                continue
            _record(bool(ok), rvec, tvec)

    if not scored:
        raise ValueError(
            "solvePnP failed to find a valid positive-depth initial pose; "
            "check point diversity, TCP coordinates, and intrinsics."
        )

    scored.sort(key=lambda item: item[0])
    # Deduplicate near-identical initializations while keeping up to 2 distinct starts.
    unique: list[tuple[np.ndarray, np.ndarray]] = []
    for _, rv, tv in scored:
        if any(
            np.linalg.norm(rv - urv) < 1e-2 and np.linalg.norm(tv - utv) < 1e-2
            for urv, utv in unique
        ):
            continue
        unique.append((rv, tv))
        if len(unique) == 2:
            break
    return unique


def _fit_extrinsic_and_offset(
    p_tcp: np.ndarray,
    R_tcp: np.ndarray,
    pixels: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    f_scale: float = ROBUST_LOSS_SCALE_PX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve ``(rvec, tvec, tool_offset)`` via PnP init + ``soft_l1``/``cauchy`` refinement."""
    candidates = _initial_pnp_candidates(p_tcp, pixels, intrinsics)

    def residuals(params: np.ndarray) -> np.ndarray:
        rvec = params[0:3]
        tvec = params[3:6]
        offset = params[6:9]
        R_cb = _rodrigues_to_matrix(rvec)
        pts_base = p_tcp + np.einsum("nij,j->ni", R_tcp, offset)
        pts_cam = pts_base @ R_cb.T + tvec
        # Penalize points drifting behind the camera plane smoothly during TRF steps.
        if np.any(pts_cam[:, 2] <= 1e-4):
            return np.full(len(pixels) * 2, 1e6, dtype=np.float64)
        pred = project_camera_points(pts_cam, intrinsics)
        return (pred - pixels).reshape(-1)

    def _has_positive_depth(params: np.ndarray) -> bool:
        R_cb = _rodrigues_to_matrix(params[0:3])
        pts_base = p_tcp + np.einsum("nij,j->ni", R_tcp, params[6:9])
        pts_cam = pts_base @ R_cb.T + params[3:6]
        return bool(np.all(pts_cam[:, 2] > 1e-4))

    best_sol: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for rvec0, tvec0 in candidates:
        x0 = np.concatenate([rvec0, tvec0, np.zeros(3, dtype=np.float64)])
        # Stage 1: convex soft_l1 warm start for global stability.
        res1 = least_squares(
            residuals,
            x0,
            method="trf",
            loss="soft_l1",
            f_scale=f_scale,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=2000,
        )
        # Stage 2: redescending cauchy refinement removes residual pull from gross outliers
        # while retaining every point in the objective.
        res2 = least_squares(
            residuals,
            res1.x,
            method="trf",
            loss="cauchy",
            f_scale=f_scale,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=2000,
        )
        chosen_x = res2.x if _has_positive_depth(res2.x) else res1.x
        if not _has_positive_depth(chosen_x):
            continue
        cost = float(res2.cost if chosen_x is res2.x else res1.cost)
        if best_sol is None or cost < best_sol[0]:
            best_sol = (cost, chosen_x[0:3], chosen_x[3:6], chosen_x[6:9])

    if best_sol is None:
        raise ValueError("Joint calibration converged to points behind the camera")
    return best_sol[1], best_sol[2], best_sol[3]


def calibrate_arm(
    arm: str,
    tcps: np.ndarray,
    pixels: np.ndarray,
    frame_ids: Sequence[str],
    intrinsics: CameraIntrinsics,
) -> ArmCalibration:
    """Calibrate one arm's ``T_cam_base`` and ``tool_offset`` robustly against bad clicks.

    Model: ``pixel = project(K, dist, T_cam_base @ (p_tcp + R_tcp @ tool_offset))``.
    Initializes with ``solvePnP`` / ``solvePnPRansac`` (``tool_offset = 0``), refines
    jointly with two-stage ``soft_l1`` warm-start + ``cauchy`` robust nonlinear least
    squares (``f_scale = 3.0`` px), flags points with Euclidean residual
    ``> max(3 * median, 15.0)`` px as outliers without dropping them, and reports RMS
    over both all points and inliers.

    Args:
        arm: Arm name (``"left"`` or ``"right"``).
        tcps: Float64 array of shape ``(N, 9)`` containing ``[x, y, z, r1..r6]``.
        pixels: Float64 array of shape ``(N, 2)`` containing observed ``[u, v]``.
        frame_ids: Sequence of ``N`` frame IDs matching rows of ``tcps`` and ``pixels``.
        intrinsics: Head-camera intrinsic parameters.

    Returns:
        Populated ``ArmCalibration`` instance.

    Raises:
        ValueError: If fewer than 5 valid correspondences are provided or inputs mismatch.
    """
    tcp_arr = np.asarray(tcps, dtype=np.float64)
    px_arr = np.asarray(pixels, dtype=np.float64)
    n = len(tcp_arr)
    if tcp_arr.ndim != 2 or tcp_arr.shape[1] != 9:
        raise ValueError(f"Expected tcps of shape (N, 9), got {tcp_arr.shape}")
    if px_arr.shape != (n, 2):
        raise ValueError(f"Expected pixels of shape ({n}, 2), got {px_arr.shape}")
    if len(frame_ids) != n:
        raise ValueError(f"Expected {n} frame_ids, got {len(frame_ids)}")
    if n < 5:
        raise ValueError(
            f"Arm {arm!r} has only {n} valid point(s); at least 5 are required "
            "(>= 6 recommended for leave-one-out validation)."
        )

    p_tcp = tcp_arr[:, 0:3]
    R_tcp = rotation_6d_to_matrix(tcp_arr[:, 3:9])
    rvec, tvec, offset = _fit_extrinsic_and_offset(p_tcp, R_tcp, px_arr, intrinsics)
    T_cam_base = _pose_to_transform(rvec, tvec)

    pts_base = p_tcp + np.einsum("nij,j->ni", R_tcp, offset)
    pred_px = project_base_points(pts_base, T_cam_base, intrinsics)
    diff = pred_px - px_arr
    sq_errs = np.sum(diff**2, axis=1)
    errs_px = np.sqrt(sq_errs)
    rms_px = float(np.sqrt(np.mean(sq_errs)))

    median_err = float(np.median(errs_px))
    outlier_threshold_px = max(OUTLIER_MEDIAN_MULTIPLIER * median_err, OUTLIER_MIN_PX)
    is_outlier = errs_px > outlier_threshold_px
    inlier_mask = ~is_outlier
    n_inliers = int(np.count_nonzero(inlier_mask))
    inlier_rms_px = float(np.sqrt(np.mean(sq_errs[inlier_mask]))) if n_inliers > 0 else None
    outlier_ids = tuple(str(frame_ids[i]) for i in range(n) if bool(is_outlier[i]))

    loo_rms_px: float | None = None
    inlier_loo_rms_px: float | None = None
    if n >= 6:
        loo_sq_errors: list[float] = []
        for hold_idx in range(n):
            mask = np.ones(n, dtype=bool)
            mask[hold_idx] = False
            r_cv, t_cv, o_cv = _fit_extrinsic_and_offset(
                p_tcp[mask], R_tcp[mask], px_arr[mask], intrinsics
            )
            T_cv = _pose_to_transform(r_cv, t_cv)
            pt_hold = (p_tcp[hold_idx] + R_tcp[hold_idx] @ o_cv).reshape(1, 3)
            pred_hold = project_base_points(pt_hold, T_cv, intrinsics)[0]
            err_vec = pred_hold - px_arr[hold_idx]
            loo_sq_errors.append(float(err_vec[0] ** 2 + err_vec[1] ** 2))
        loo_arr = np.asarray(loo_sq_errors, dtype=np.float64)
        loo_rms_px = float(np.sqrt(np.mean(loo_arr)))
        if n_inliers > 0:
            inlier_loo_rms_px = float(np.sqrt(np.mean(loo_arr[inlier_mask])))

    residual_items = tuple(
        (str(fid), float(diff[i, 0]), float(diff[i, 1])) for i, fid in enumerate(frame_ids)
    )
    return ArmCalibration(
        arm=arm,
        T_cam_base=T_cam_base,
        tool_offset=offset,
        n_points=n,
        rms_px=rms_px,
        loo_rms_px=loo_rms_px,
        residuals=residual_items,
        inlier_rms_px=inlier_rms_px,
        inlier_loo_rms_px=inlier_loo_rms_px,
        n_inliers=n_inliers,
        outlier_threshold_px=outlier_threshold_px,
        outlier_ids=outlier_ids,
    )


def calibrate_head_camera(
    points_payload: Mapping[str, Any],
    intrinsics: CameraIntrinsics,
) -> dict[str, Any]:
    """Solve independent head-camera extrinsics for each arm present in ``points.json``.

    Args:
        points_payload: Parsed ``points.json`` dictionary.
        intrinsics: Loaded ``CameraIntrinsics`` from ``intrinsics.json``.

    Returns:
        JSON-serializable ``head.json`` v1 dictionary.

    Raises:
        ValueError: If ``points_payload`` has invalid schema or no solvable arm.
    """
    if points_payload.get("version") != CALIB_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported points version: {points_payload.get('version')!r} "
            f"(expected {CALIB_SCHEMA_VERSION})"
        )
    frames = points_payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("points.json contains no frames")

    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in frames:
        if not isinstance(item, Mapping):
            continue
        if item.get("skipped") is True:
            continue
        pixel = item.get("pixel")
        if not isinstance(pixel, list) or len(pixel) != 2:
            continue
        by_arm[str(item["arm"])].append(item)

    if not by_arm:
        raise ValueError("points.json has no annotated non-skipped frames with pixel coordinates")

    arms_out: dict[str, Any] = {}
    for arm_name in sorted(by_arm):
        entries = by_arm[arm_name]
        tcps = np.array([e["tcp"] for e in entries], dtype=np.float64)
        pixels = np.array([e["pixel"] for e in entries], dtype=np.float64)
        ids = [str(e["id"]) for e in entries]
        cal = calibrate_arm(arm_name, tcps, pixels, ids, intrinsics)
        outlier_set = set(cal.outlier_ids)
        arms_out[arm_name] = {
            "T_cam_base": [[float(v) for v in row] for row in cal.T_cam_base],
            "tool_offset": [float(v) for v in cal.tool_offset],
            "n_points": cal.n_points,
            "n_inliers": cal.n_inliers,
            "rms_px": cal.rms_px,
            "inlier_rms_px": cal.inlier_rms_px,
            "loo_rms_px": cal.loo_rms_px,
            "inlier_loo_rms_px": cal.inlier_loo_rms_px,
            "outlier_threshold_px": cal.outlier_threshold_px,
            "residuals": [
                {"id": fid, "du": du, "dv": dv, "outlier": fid in outlier_set}
                for fid, du, dv in cal.residuals
            ],
        }

    image_size = points_payload.get("image_size", list(intrinsics.image_size))
    return {
        "version": CALIB_SCHEMA_VERSION,
        "repo_id": str(points_payload.get("repo_id", "")),
        "revision": str(points_payload.get("revision", "")),
        "camera": str(points_payload.get("camera", intrinsics.camera)),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "intrinsics": dict(intrinsics.raw),
        "arms": arms_out,
    }


def compare_arm_extrinsics(head_payload: Mapping[str, Any]) -> dict[str, float] | None:
    """Compare ``left`` and ``right`` extrinsics when both arms are calibrated.

    Returns translation difference in millimeters (comparing camera positions in
    each arm's base frame and ``t_cb`` vectors) and relative rotation angle in
    degrees, or ``None`` if fewer than two arms are present.
    """
    arms = head_payload.get("arms", {})
    if "left" not in arms or "right" not in arms:
        return None
    T_l = np.asarray(arms["left"]["T_cam_base"], dtype=np.float64)
    T_r = np.asarray(arms["right"]["T_cam_base"], dtype=np.float64)
    R_rel = T_l[:3, :3] @ T_r[:3, :3].T
    trace_val = float(np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0))
    rot_diff_deg = math.degrees(math.acos(trace_val))
    t_diff_mm = float(np.linalg.norm(T_l[:3, 3] - T_r[:3, 3]) * 1000.0)
    cam_pos_l = -(T_l[:3, :3].T @ T_l[:3, 3])
    cam_pos_r = -(T_r[:3, :3].T @ T_r[:3, 3])
    cam_pos_diff_mm = float(np.linalg.norm(cam_pos_l - cam_pos_r) * 1000.0)
    return {
        "rotation_diff_deg": rot_diff_deg,
        "tvec_diff_mm": t_diff_mm,
        "camera_origin_diff_mm": cam_pos_diff_mm,
    }


def write_json(payload: Mapping[str, Any], path: Path) -> Path:
    """Write a calibration JSON file with UTF-8 indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
