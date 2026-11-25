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
        fps = 30.0

        def camera_image_size(self, camera_key: str) -> tuple[int, int]:
            assert camera_key == "observation.images.head"
            return (640, 480)

        def load_arm_tcp_table(self, episodes=None):
            return ep_idx, fr_idx, tcp_by_arm

    ds = _FakeDataset()
    initial = select_calib_frames(ds, count_per_arm=5, max_speed_mps=None)  # type: ignore[arg-type]
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
        max_speed_mps=None,
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


def test_calib_frames_cli_refuses_existing_without_overwrite(tmp_path, capsys, monkeypatch) -> None:
    import json

    from lerobot_check import cli

    lerobot_root = tmp_path / "lerobot"
    ds_root = lerobot_root / "Xense" / "demo_ds"
    ds_root.mkdir(parents=True)
    frames_path = lerobot_root / ".xense-viewer" / "calib" / "Xense__demo_ds.frames.json"
    frames_path.parent.mkdir(parents=True)
    sentinel = {
        "version": 1,
        "repo_id": "Xense/demo_ds",
        "revision": "local",
        "camera": "observation.images.head",
        "image_size": [640, 480],
        "frames": [
            {
                "id": "left-000",
                "arm": "left",
                "episode_index": 0,
                "frame_index": 0,
                "tcp": [0.9, -0.1, -0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            }
        ],
    }
    frames_path.write_text(json.dumps(sentinel), encoding="utf-8")

    # Without --append or --overwrite, calib-frames must refuse and preserve the file.
    rc = cli.main(
        [
            "calib-frames",
            "--dataset-root",
            str(ds_root),
            "--repo-id",
            "Xense/demo_ds",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "Refusing to overwrite" in err
    assert "--overwrite" in err
    assert json.loads(frames_path.read_text(encoding="utf-8")) == sentinel

    # --append and --overwrite together are rejected.
    rc_conflict = cli.main(
        [
            "calib-frames",
            "--dataset-root",
            str(ds_root),
            "--repo-id",
            "Xense/demo_ds",
            "--append",
            "2",
            "--overwrite",
        ]
    )
    assert rc_conflict == 1
    assert "mutually exclusive" in capsys.readouterr().err

    # With --overwrite, DatasetView is opened and frames.json is regenerated.
    ep_idx = np.array([0, 0, 0], dtype=np.int64)
    fr_idx = np.array([0, 1, 2], dtype=np.int64)
    tcps = {
        "left": np.array(
            [
                [0.9000, -0.10, -0.30, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.9005, -0.10, -0.30, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.9010, -0.10, -0.30, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    }

    class _FakeDatasetView:
        def __init__(self, root, *, repo_id=None, revision=None) -> None:
            self.repo_id = repo_id or "Xense/demo_ds"
            self.revision = revision or "local"
            self.total_episodes = 1
            self.fps = 30.0

        def camera_image_size(self, camera_key: str) -> tuple[int, int]:
            return (640, 480)

        def load_arm_tcp_table(self, episodes=None):
            return ep_idx, fr_idx, tcps

    monkeypatch.setattr(cli, "DatasetView", _FakeDatasetView)
    rc_overwrite = cli.main(
        [
            "calib-frames",
            "--dataset-root",
            str(ds_root),
            "--repo-id",
            "Xense/demo_ds",
            "-n",
            "2",
            "--overwrite",
        ]
    )
    assert rc_overwrite == 0
    regenerated = json.loads(frames_path.read_text(encoding="utf-8"))
    assert len(regenerated["frames"]) == 2
    assert [f["id"] for f in regenerated["frames"]] == ["left-000", "left-001"]


def test_calibrate_cli_missing_points_json_reports_clean_error(tmp_path, capsys) -> None:
    import json

    from lerobot_check.cli import main

    lerobot_root = tmp_path / "lerobot"
    ds_root = lerobot_root / "Xense" / "demo_ds"
    ds_root.mkdir(parents=True)
    calib_dir = lerobot_root / ".xense-viewer" / "calib"
    calib_dir.mkdir(parents=True)
    (calib_dir / "Xense__demo_ds.intrinsics.json").write_text(
        json.dumps(_make_intrinsics().raw), encoding="utf-8"
    )

    rc = main(["calibrate", "--dataset-root", str(ds_root), "--repo-id", "Xense/demo_ds"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "points.json not found" in err
    assert "Xense__demo_ds.points.json" in err


def test_select_calib_frames_filters_open_grippers() -> None:
    import pytest

    ep_idx = np.zeros(20, dtype=np.int64)
    fr_idx = np.arange(20, dtype=np.int64)
    # Frames 0..9 have open grippers (0.20..1.00) with extreme TCP positions that naive FPS
    # would pick first; frames 10..19 have closed grippers (0.00..0.04) moving slowly (0.015 m/s).
    tcps = np.zeros((20, 9), dtype=np.float64)
    for i in range(20):
        if i < 10:
            tcps[i, 0:3] = [1.0 + i * 5.0, 0.0, -0.3]
        else:
            tcps[i, 0:3] = [1.0 + (i - 10) * 0.0005, 0.0, -0.3]
        tcps[i, 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    grippers = np.array([0.2 + 0.08 * i for i in range(10)] + [0.004 * (i + 1) for i in range(10)])

    class _FakeCalibDataset:
        repo_id = "Xense/grip_test"
        revision = "rev1"
        fps = 30.0

        def camera_image_size(self, camera_key: str) -> tuple[int, int]:
            return (640, 480)

        def load_arm_calib_table(self, episodes=None):
            return ep_idx, fr_idx, {"right": tcps}, {"right": grippers}

    ds = _FakeCalibDataset()
    selected = select_calib_frames(ds, count_per_arm=5)  # type: ignore[arg-type]
    chosen_frames = [f["frame_index"] for f in selected["frames"]]
    assert len(chosen_frames) == 5
    assert all(grippers[f] <= 0.05 for f in chosen_frames)

    # Appending preserves an existing open-gripper entry while only adding closed frames.
    existing_with_open = {
        "version": 1,
        "repo_id": "Xense/grip_test",
        "revision": "rev1",
        "camera": "observation.images.head",
        "image_size": [640, 480],
        "frames": [
            {
                "id": "right-000",
                "arm": "right",
                "episode_index": 0,
                "frame_index": 0,
                "tcp": tcps[0].tolist(),
            }
        ],
    }
    appended = select_calib_frames(
        ds,  # type: ignore[arg-type]
        existing=existing_with_open,
        append_per_arm=3,
    )
    assert appended["frames"][0]["frame_index"] == 0
    new_frames = [f["frame_index"] for f in appended["frames"][1:]]
    assert len(new_frames) == 3
    assert all(grippers[f] <= 0.05 for f in new_frames)

    # Error clearly when threshold excludes all frames.
    with pytest.raises(ValueError, match="No eligible frames for arm 'right'"):
        select_calib_frames(ds, count_per_arm=2, max_gripper_pos=0.0)  # type: ignore[arg-type]


def test_robust_calibration_flags_outliers_and_recovers_pose() -> None:
    rng = np.random.default_rng(99)
    intr = _make_intrinsics()

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
    offset_true = np.array([0.015, -0.022, 0.180], dtype=np.float64)

    frames = []
    clean_frames = []
    for i in range(18):
        p_tcp = np.array(
            [
                rng.uniform(0.80, 1.20),
                rng.uniform(-0.02, 0.26),
                rng.uniform(-0.35, -0.05),
            ],
            dtype=np.float64,
        )
        R_tcp = _euler_to_rot(
            float(rng.uniform(-45.0, 45.0)),
            float(rng.uniform(-45.0, 45.0)),
            float(rng.uniform(-120.0, 120.0)),
        )
        uv = project_base_points(p_tcp + R_tcp @ offset_true, T_true, intr)[0]
        uv = uv + rng.normal(0.0, 1.0 / math.sqrt(2.0), size=2)
        base_entry = {
            "id": f"right-{i:03d}",
            "arm": "right",
            "episode_index": i,
            "frame_index": i * 10,
            "tcp": np.concatenate([p_tcp, _rot_to_6d(R_tcp)]).tolist(),
            "pixel": uv.tolist(),
        }
        if i not in (3, 11):
            clean_frames.append(base_entry)
        uv_obs = uv.copy()
        if i == 3:
            uv_obs = uv_obs + np.array([80.0, -20.0])
        elif i == 11:
            uv_obs = uv_obs + np.array([-55.0, 35.0])
        frames.append({**base_entry, "pixel": uv_obs.tolist()})

    head = calibrate_head_camera(
        {
            "version": 1,
            "repo_id": "Xense/robust_test",
            "revision": "rev1",
            "camera": "observation.images.head",
            "image_size": [640, 480],
            "frames": frames,
        },
        intr,
    )
    clean_head = calibrate_head_camera(
        {
            "version": 1,
            "repo_id": "Xense/robust_test",
            "revision": "rev1",
            "camera": "observation.images.head",
            "image_size": [640, 480],
            "frames": clean_frames,
        },
        intr,
    )
    res = head["arms"]["right"]
    clean_res = clean_head["arms"]["right"]
    assert res["n_points"] == 18
    assert res["n_inliers"] == 16
    assert len(res["residuals"]) == 18
    flagged = {r["id"] for r in res["residuals"] if r["outlier"]}
    assert flagged == {"right-003", "right-011"}
    assert res["inlier_rms_px"] < 2.5
    assert res["inlier_loo_rms_px"] is not None and res["inlier_loo_rms_px"] < 3.5
    assert res["rms_px"] > res["inlier_rms_px"]

    T_est = np.asarray(res["T_cam_base"], dtype=np.float64)
    offset_est = np.asarray(res["tool_offset"], dtype=np.float64)
    T_clean = np.asarray(clean_res["T_cam_base"], dtype=np.float64)
    offset_clean = np.asarray(clean_res["tool_offset"], dtype=np.float64)
    assert float(np.linalg.norm(T_est[:3, 3] - t_true) * 1000.0) < 10.0
    assert float(np.linalg.norm(offset_est - offset_true) * 1000.0) < 5.0
    assert float(np.linalg.norm(T_est[:3, 3] - T_clean[:3, 3]) * 1000.0) < 2.0
    assert float(np.linalg.norm(offset_est - offset_clean) * 1000.0) < 2.0


def test_select_calib_frames_speed_filter_and_disable() -> None:
    import pytest

    fps = 30.0
    # Episode 0: 8 fast frames (0.30 m/s = 0.01 m/frame) at extreme x=2.0..2.07.
    # Episode 1: 8 slow frames (0.015 m/s = 0.0005 m/frame) at x=0.90..0.9035.
    # Episode 2: 1-frame episode (speed is NaN -> excluded when speed filter is active).
    ep_idx = np.array([0] * 8 + [1] * 8 + [2], dtype=np.int64)
    fr_idx = np.array(list(range(8)) + list(range(8)) + [0], dtype=np.int64)
    tcps = np.zeros((17, 9), dtype=np.float64)
    tcps[:, 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    for i in range(8):
        tcps[i, 0:3] = [2.0 + i * 0.01, 0.0, -0.2]
    for i in range(8):
        tcps[8 + i, 0:3] = [0.90 + i * 0.0005, 0.1, -0.3]
    tcps[16, 0:3] = [9.0, 9.0, 9.0]  # extreme single-frame episode
    grippers = np.full(17, 0.01, dtype=np.float64)

    class _FakeSpeedDataset:
        repo_id = "Xense/speed_test"
        revision = "rev1"
        fps = 30.0

        def camera_image_size(self, camera_key: str) -> tuple[int, int]:
            return (640, 480)

        def load_arm_calib_table(self, episodes=None):
            return ep_idx, fr_idx, {"left": tcps}, {"left": grippers}

    ds = _FakeSpeedDataset()
    slow_only = select_calib_frames(ds, count_per_arm=5)  # type: ignore[arg-type]
    assert len(slow_only["frames"]) == 5
    for item in slow_only["frames"]:
        assert item["episode_index"] == 1
        assert "speed_mps" in item
        assert item["speed_mps"] <= 0.03
        # Constant velocity 0.0005 m/frame * 30 fps = 0.015 m/s everywhere in episode 1
        # (both one-sided edges and central interior differences).
        assert abs(item["speed_mps"] - (0.0005 * fps)) < 1e-4

    # Disabling the speed filter allows picking fast / 1-frame extreme poses.
    unfiltered = select_calib_frames(
        ds,  # type: ignore[arg-type]
        count_per_arm=5,
        max_speed_mps=None,
    )
    unfiltered_eps = {item["episode_index"] for item in unfiltered["frames"]}
    assert 0 in unfiltered_eps or 2 in unfiltered_eps
    for item in unfiltered["frames"]:
        if item["episode_index"] == 0:
            assert abs(item["speed_mps"] - (0.01 * fps)) < 1e-4
        elif item["episode_index"] == 2:
            assert item["speed_mps"] is None

    with pytest.raises(ValueError, match="max_speed_mps must be non-negative"):
        select_calib_frames(ds, count_per_arm=2, max_speed_mps=-0.01)  # type: ignore[arg-type]


def test_speed_filter_does_not_span_episode_boundaries() -> None:
    # Two consecutive episodes with a huge 5.0 m jump across the episode boundary.
    # Episode 0: stationary at x=0.0 (speed = 0.0 m/s).
    # Episode 1: slow linear motion along y at 0.0004 m/frame * 25 fps = 0.01 m/s starting at x=5.0.
    ep_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    fr_idx = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    tcps = np.zeros((8, 9), dtype=np.float64)
    tcps[:, 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    for i in range(4):
        tcps[i, 0:3] = [0.0, 0.0, -0.3]
    for i in range(4):
        tcps[4 + i, 0:3] = [5.0, i * 0.0004, -0.3]
    grippers = np.zeros(8, dtype=np.float64)

    class _FakeBoundaryDataset:
        repo_id = "Xense/ep_boundary"
        revision = "rev1"
        fps = 25.0

        def camera_image_size(self, camera_key: str) -> tuple[int, int]:
            return (640, 480)

        def load_arm_calib_table(self, episodes=None):
            return ep_idx, fr_idx, {"right": tcps}, {"right": grippers}

    res = select_calib_frames(
        _FakeBoundaryDataset(),  # type: ignore[arg-type]
        count_per_arm=8,
        max_speed_mps=0.03,
    )
    by_key = {(f["episode_index"], f["frame_index"]): f["speed_mps"] for f in res["frames"]}
    assert len(by_key) == 8
    # Boundary frames (ep 0 frame 3 and ep 1 frame 0) must NOT see the 5 m jump.
    assert by_key[(0, 3)] == 0.0
    assert abs(by_key[(1, 0)] - 0.01) < 1e-4


def test_calib_frames_cli_per_arm_append_and_validation(tmp_path, capsys, monkeypatch) -> None:
    import json

    import pytest

    from lerobot_check import cli

    lerobot_root = tmp_path / "lerobot"
    ds_root = lerobot_root / "Xense" / "arm_append_ds"
    ds_root.mkdir(parents=True)
    frames_path = tmp_path / "frames.json"

    ep_idx = np.array([0] * 6, dtype=np.int64)
    fr_idx = np.arange(6, dtype=np.int64)
    # Slow closed-gripper frames (0.0005 m at 30 fps => 0.015 m/s) except fast frame 5.
    left_tcps = np.zeros((6, 9), dtype=np.float64)
    right_tcps = np.zeros((6, 9), dtype=np.float64)
    for i in range(6):
        left_tcps[i, 0:3] = [0.80 + i * 0.0005, -0.10, -0.30]
        left_tcps[i, 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        right_tcps[i, 0:3] = [0.80 + i * 0.0005, 0.10, -0.30]
        right_tcps[i, 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    left_tcps[5, 0] = 1.50  # fast jump at frame 5 (and affects frame 4 central diff)
    grippers = {"left": np.full(6, 0.01), "right": np.full(6, 0.01)}

    existing_entries = [
        {
            "id": "left-000",
            "arm": "left",
            "episode_index": 0,
            "frame_index": 0,
            "tcp": left_tcps[0].tolist(),
            "skipped": True,
        },
        {
            "id": "left-001",
            "arm": "left",
            "episode_index": 0,
            "frame_index": 1,
            "tcp": left_tcps[1].tolist(),
            "pixel": [123.5, 240.0],
        },
        {
            "id": "right-000",
            "arm": "right",
            "episode_index": 0,
            "frame_index": 0,
            "tcp": right_tcps[0].tolist(),
            "pixel": [410.0, 220.0],
        },
    ]
    existing_payload = {
        "version": 1,
        "repo_id": "Xense/arm_append_ds",
        "revision": "rev1",
        "camera": "observation.images.head",
        "image_size": [640, 480],
        "frames": existing_entries,
    }
    existing_json_entries = [json.dumps(e, sort_keys=False) for e in existing_entries]
    frames_path.write_text(json.dumps(existing_payload, indent=2), encoding="utf-8")

    class _FakeArmDataset:
        def __init__(self, root, *, repo_id=None, revision=None) -> None:
            self.repo_id = repo_id or "Xense/arm_append_ds"
            self.revision = revision or "rev1"
            self.total_episodes = 1
            self.fps = 30.0

        def camera_image_size(self, camera_key: str) -> tuple[int, int]:
            return (640, 480)

        def load_arm_calib_table(self, episodes=None):
            return ep_idx, fr_idx, {"left": left_tcps, "right": right_tcps}, grippers

    monkeypatch.setattr(cli, "DatasetView", _FakeArmDataset)

    # --arms without --append is an error.
    rc_no_append = cli.main(
        [
            "calib-frames",
            "--dataset-root",
            str(ds_root),
            "--arms",
            "left",
            "--overwrite",
            "-o",
            str(frames_path),
        ]
    )
    assert rc_no_append == 1
    assert "--arms is only allowed with --append" in capsys.readouterr().err

    # --max-speed and --no-speed-filter are mutually exclusive.
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "calib-frames",
                "--dataset-root",
                str(ds_root),
                "--max-speed",
                "0.05",
                "--no-speed-filter",
                "-o",
                str(frames_path),
            ]
        )
    assert exc_info.value.code == 2
    capsys.readouterr()

    # Unknown arm reports available arms.
    rc_unknown = cli.main(
        [
            "calib-frames",
            "--dataset-root",
            str(ds_root),
            "--append",
            "1",
            "--arms",
            "middle",
            "-o",
            str(frames_path),
        ]
    )
    assert rc_unknown == 1
    err_unknown = capsys.readouterr().err
    assert "Unknown arm" in err_unknown
    assert "left" in err_unknown and "right" in err_unknown

    # Valid --append 1 --arms left adds exactly left-002 and preserves existing entries verbatim.
    rc_append = cli.main(
        [
            "calib-frames",
            "--dataset-root",
            str(ds_root),
            "--append",
            "1",
            "--arms",
            "left",
            "-o",
            str(frames_path),
        ]
    )
    assert rc_append == 0
    out_append = capsys.readouterr().out
    assert "left=+1" in out_append
    updated = json.loads(frames_path.read_text(encoding="utf-8"))
    assert len(updated["frames"]) == 4
    assert [json.dumps(e, sort_keys=False) for e in updated["frames"][:3]] == existing_json_entries
    added = updated["frames"][3]
    assert added["id"] == "left-002"
    assert added["arm"] == "left"
    assert added["frame_index"] in (2, 3)
    assert abs(added["speed_mps"] - 0.015) < 1e-4

    # Requesting more frames than eligible warns on stderr and exits 0 with available frames.
    rc_partial = cli.main(
        [
            "calib-frames",
            "--dataset-root",
            str(ds_root),
            "--append",
            "5",
            "--arms",
            "left",
            "-o",
            str(frames_path),
        ]
    )
    assert rc_partial == 0
    captured = capsys.readouterr()
    assert "Warning:" in captured.err
    assert "requested 5" in captured.err
    assert "left=+1" in captured.out
    assert len(json.loads(frames_path.read_text(encoding="utf-8"))["frames"]) == 5

    # When zero eligible frames remain, error names arm and active filters.
    rc_none = cli.main(
        [
            "calib-frames",
            "--dataset-root",
            str(ds_root),
            "--append",
            "1",
            "--arms",
            "left",
            "-o",
            str(frames_path),
        ]
    )
    assert rc_none == 1
    err_none = capsys.readouterr().err
    assert "left" in err_none
    assert "--max-speed" in err_none and "--max-gripper" in err_none
