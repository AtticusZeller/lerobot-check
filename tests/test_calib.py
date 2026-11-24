"""Unit tests for lerobot_check.calib (calib-frames FPS/append and synthetic calibration)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from lerobot_check.calib import (
    CameraIntrinsics,
    calibrate_head_camera,
    compare_arm_extrinsics,
    project_base_points,
    project_camera_points,
    select_calib_frames,
)


def _make_intrinsics(
    *,
    distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
) -> CameraIntrinsics:
    raw = {
        "version": 1,
        "camera": "observation.images.head",
        "device": "Synthetic D435",
        "image_size": [640, 480],
        "fx": 606.0,
        "fy": 605.5,
        "cx": 323.0,
        "cy": 244.8,
        "distortion_model": "inverse_brown_conrady",
        "distortion": list(distortion),
    }
    return CameraIntrinsics(
        camera="observation.images.head",
        image_size=(640, 480),
        fx=606.0,
        fy=605.5,
        cx=323.0,
        cy=244.8,
        distortion_model="inverse_brown_conrady",
        distortion=distortion,
        raw=raw,
    )


def _euler_to_rot(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    rx, ry, rz = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _rot_to_6d(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:, 0], R[:, 1]])


def test_project_camera_points_matches_opencv_with_distortion() -> None:
    intr = _make_intrinsics(distortion=(-0.12, 0.05, 0.002, -0.001, 0.01))
    pts_cam = np.array(
        [
            [0.10, -0.05, 0.85],
            [-0.22, 0.14, 1.15],
            [0.05, 0.08, 0.65],
        ],
        dtype=np.float64,
    )
    ours = project_camera_points(pts_cam, intr)
    cv_proj, _ = cv2.projectPoints(
        pts_cam.reshape(-1, 1, 3),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        intr.camera_matrix,
        intr.dist_coeffs,
    )
    assert np.allclose(ours, cv_proj.reshape(-1, 2), atol=1e-10)


def test_synthetic_calibration_recovers_extrinsic_and_tool_offset_under_1px_noise() -> None:
    rng = np.random.default_rng(2026)
    intr = _make_intrinsics()

    # Ground-truth camera mounted ~1 m in front/above workspace looking down.
    cam_pos = np.array([0.30, 0.02, 0.42], dtype=np.float64)
    look_at = np.array([1.00, 0.02, -0.32], dtype=np.float64)
    z_axis = look_at - cam_pos
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, np.array([0.0, 0.0, 1.0]))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_true = np.vstack([x_axis, y_axis, z_axis])
    t_true = -(R_true @ cam_pos)
    T_true = np.eye(4, dtype=np.float64)
    T_true[:3, :3] = R_true
    T_true[:3, 3] = t_true

    # Realistic 18 cm flange-to-fingertip tool offset.
    offset_true = np.array([0.015, -0.022, 0.180], dtype=np.float64)

    # Generate 18 diverse TCP poses (position + roll/pitch/yaw variation) per arm.
    frames = []
    for arm, y_center in (("left", -0.10), ("right", 0.14)):
        for i in range(18):
            p_tcp = np.array(
                [
                    rng.uniform(0.80, 1.20),
                    rng.uniform(y_center - 0.14, y_center + 0.14),
                    rng.uniform(-0.35, -0.05),
                ],
                dtype=np.float64,
            )
            R_tcp = _euler_to_rot(
                float(rng.uniform(-45.0, 45.0)),
                float(rng.uniform(-45.0, 45.0)),
                float(rng.uniform(-120.0, 120.0)),
            )
            p_tip = p_tcp + R_tcp @ offset_true
            uv_clean = project_base_points(p_tip, T_true, intr)[0]
            uv_noisy = uv_clean + rng.normal(0.0, 1.0 / math.sqrt(2.0), size=2)
            tcp_9d = np.concatenate([p_tcp, _rot_to_6d(R_tcp)])
            frames.append(
                {
                    "id": f"{arm}-{i:03d}",
                    "arm": arm,
                    "episode_index": i,
                    "frame_index": i * 10,
                    "tcp": tcp_9d.tolist(),
                    "pixel": uv_noisy.tolist(),
                }
            )

    # Add a skipped frame and an unannotated frame to verify they are ignored cleanly.
    frames.append(
        {
            "id": "left-099",
            "arm": "left",
            "episode_index": 99,
            "frame_index": 0,
            "tcp": [1.0, 0.0, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "skipped": True,
        }
    )

    points_payload = {
        "version": 1,
        "repo_id": "Xense/synthetic",
        "revision": "rev1",
        "camera": "observation.images.head",
        "image_size": [640, 480],
        "frames": frames,
    }
    head = calibrate_head_camera(points_payload, intr)
    assert head["version"] == 1
    assert head["intrinsics"] == intr.raw
    assert set(head["arms"].keys()) == {"left", "right"}

    for arm in ("left", "right"):
        res = head["arms"][arm]
        assert res["n_points"] == 18
        assert len(res["residuals"]) == 18
        T_est = np.asarray(res["T_cam_base"], dtype=np.float64)
        offset_est = np.asarray(res["tool_offset"], dtype=np.float64)

        trans_err_mm = float(np.linalg.norm(T_est[:3, 3] - t_true) * 1000.0)
        offset_err_mm = float(np.linalg.norm(offset_est - offset_true) * 1000.0)
        assert trans_err_mm < 5.0, f"{arm} translation error {trans_err_mm:.2f} mm >= 5 mm"
        assert offset_err_mm < 5.0, f"{arm} tool offset error {offset_err_mm:.2f} mm >= 5 mm"
        assert res["rms_px"] < 2.5
        assert res["loo_rms_px"] is not None and res["loo_rms_px"] < 3.5

    diff = compare_arm_extrinsics(head)
    assert diff is not None
    assert diff["tvec_diff_mm"] < 10.0
    assert diff["rotation_diff_deg"] < 1.0


def test_select_calib_frames_fps_and_append_preserves_ids() -> None:
    rng = np.random.default_rng(7)
    n_frames = 120
    ep_idx = np.repeat(np.arange(4, dtype=np.int64), 30)
    fr_idx = np.tile(np.arange(30, dtype=np.int64), 4)

    def _random_tcps() -> np.ndarray:
        rows = []
        for _ in range(n_frames):
            xyz = rng.uniform([0.8, -0.2, -0.4], [1.2, 0.2, -0.1])
            R = _euler_to_rot(
                float(rng.uniform(-40, 40)),
                float(rng.uniform(-40, 40)),
                float(rng.uniform(-180, 180)),
            )
            rows.append(np.concatenate([xyz, _rot_to_6d(R)]))
        return np.array(rows, dtype=np.float64)

    tcp_by_arm = {"left": _random_tcps(), "right": _random_tcps()}

    class _FakeDataset:
        repo_id = "Xense/demo"
        revision = "abc123"

        def camera_image_size(self, camera_key: str) -> tuple[int, int]:
            assert camera_key == "observation.images.head"
            return (640, 480)

        def load_arm_tcp_table(self, episodes=None):
            return ep_idx, fr_idx, tcp_by_arm

    ds = _FakeDataset()
    initial = select_calib_frames(ds, count_per_arm=5)  # type: ignore[arg-type]
    assert initial["version"] == 1
    assert initial["image_size"] == [640, 480]
    assert len(initial["frames"]) == 10
    left_ids = [f["id"] for f in initial["frames"] if f["arm"] == "left"]
    right_ids = [f["id"] for f in initial["frames"] if f["arm"] == "right"]
    assert left_ids == [f"left-{i:03d}" for i in range(5)]
    assert right_ids == [f"right-{i:03d}" for i in range(5)]

    appended = select_calib_frames(
        ds,  # type: ignore[arg-type]
        existing=initial,
        append_per_arm=3,
    )
    assert len(appended["frames"]) == 16
    # First 10 entries and their IDs remain completely unchanged.
    assert appended["frames"][:10] == initial["frames"]
    new_left = [f for f in appended["frames"][10:] if f["arm"] == "left"]
    new_right = [f for f in appended["frames"][10:] if f["arm"] == "right"]
    assert [f["id"] for f in new_left] == ["left-005", "left-006", "left-007"]
    assert [f["id"] for f in new_right] == ["right-005", "right-006", "right-007"]
    # No duplicate (arm, episode_index, frame_index) triples.
    triples = [(f["arm"], f["episode_index"], f["frame_index"]) for f in appended["frames"]]
    assert len(triples) == len(set(triples))


def test_calibrate_cli_end_to_end(tmp_path) -> None:
    import json

    from lerobot_check.cli import main

    rng = np.random.default_rng(11)
    intr = _make_intrinsics()
    intr_path = tmp_path / "intrinsics.json"
    intr_path.write_text(json.dumps(intr.raw), encoding="utf-8")

    cam_pos = np.array([0.35, 0.0, 0.45], dtype=np.float64)
    look_at = np.array([1.0, 0.0, -0.30], dtype=np.float64)
    z_ax = look_at - cam_pos
    z_ax /= np.linalg.norm(z_ax)
    x_ax = np.cross(z_ax, np.array([0.0, 0.0, 1.0]))
    x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)
    R_true = np.vstack([x_ax, y_ax, z_ax])
    T_true = np.eye(4, dtype=np.float64)
    T_true[:3, :3] = R_true
    T_true[:3, 3] = -(R_true @ cam_pos)
    offset = np.array([0.0, 0.0, 0.12], dtype=np.float64)

    frames = []
    for i in range(10):
        p_tcp = rng.uniform([0.85, -0.15, -0.3], [1.15, 0.15, -0.05])
        R_tcp = _euler_to_rot(
            float(rng.uniform(-35, 35)),
            float(rng.uniform(-35, 35)),
            float(rng.uniform(-90, 90)),
        )
        uv = project_base_points(p_tcp + R_tcp @ offset, T_true, intr)[0]
        frames.append(
            {
                "id": f"right-{i:03d}",
                "arm": "right",
                "episode_index": i,
                "frame_index": 0,
                "tcp": np.concatenate([p_tcp, _rot_to_6d(R_tcp)]).tolist(),
                "pixel": uv.tolist(),
            }
        )
    points_path = tmp_path / "points.json"
    points_path.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_id": "Xense/demo",
                "revision": "rev1",
                "camera": "observation.images.head",
                "image_size": [640, 480],
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "head.json"
    rc = main(
        [
            "calibrate",
            "--points",
            str(points_path),
            "--intrinsics",
            str(intr_path),
            "-o",
            str(out_path),
        ]
    )
    assert rc == 0
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["arms"]["right"]["rms_px"] < 1e-3
