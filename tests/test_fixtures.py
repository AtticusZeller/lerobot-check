"""Tests that committed synthetic schema fixtures match current code output."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from lerobot_check.analysis import compute_distribution_from_states
from lerobot_check.calib import (
    ArmCalibration,
    CameraIntrinsics,
    HeadCalibration,
    project_base_points,
)
from lerobot_check.dataset import ArmState
from lerobot_check.schema import (
    AnnotationRow,
    CoverageSpec,
    ObjectSpec,
    SubtaskSpec,
    TaskSpec,
)
from lerobot_check.suggest import generate_suggestions

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DIST_FIXTURE = FIXTURES_DIR / "distribution.example.json"
SUGGEST_FIXTURE = FIXTURES_DIR / "suggestions.example.json"


def _yaw_rot6d(yaw_deg: float) -> np.ndarray:
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([c, s, 0.0, -s, c, 0.0], dtype=np.float64)


def build_synthetic_fixtures() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build deterministic synthetic ``distribution`` and ``suggestions`` fixture payloads."""
    spec = TaskSpec(
        name="synthetic_pick_place",
        subtasks=(
            SubtaskSpec(
                name="pick_block",
                prompt="Pick up the block.",
                cameras=("observation.images.right_wrist",),
            ),
            SubtaskSpec(
                name="place_block",
                prompt="Place the block into the tray.",
                cameras=("observation.images.right_wrist", "observation.images.left_wrist"),
                arms=("right", "left"),
            ),
        ),
        objects=(
            ObjectSpec(
                name="block",
                subtask="pick_block",
                event="grasp",
                arm="right",
                yaw_symmetry_deg=180.0,
            ),
            ObjectSpec(
                name="tray",
                subtask="place_block",
                event="release",
                arm="right",
                yaw_symmetry_deg=180.0,
            ),
        ),
        coverage=CoverageSpec(xy_bin_m=0.05, yaw_bin_deg=30.0, sparse_threshold=2),
    )

    rows = [
        AnnotationRow(
            "Synthetic/demo", "synthetic0001", 0, "pick_block", 0, 20, None, 10, True, "human"
        ),
        AnnotationRow(
            "Synthetic/demo", "synthetic0001", 0, "place_block", 20, 50, None, None, True, "human"
        ),
        AnnotationRow(
            "Synthetic/demo", "synthetic0001", 2, "pick_block", 0, 25, None, 12, True, "agent"
        ),
        AnnotationRow(
            "Synthetic/demo", "synthetic0001", 2, "place_block", 25, 60, None, None, True, "agent"
        ),
    ]
    ep_lengths = {0: 50, 1: 40, 2: 60}

    def _arm_traj(
        n: int, x0: float, x1: float, y0: float, y1: float, z: float, yaw0: float, yaw1: float
    ) -> ArmState:
        xyz = np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n), np.full(n, z)])
        rot6d = np.vstack([_yaw_rot6d(float(a)) for a in np.linspace(yaw0, yaw1, n)])
        return ArmState("arm", xyz, rot6d, np.ones(n))

    arm_states = {
        0: {
            "right": _arm_traj(50, 0.96, 1.08, 0.04, 0.14, -0.35, 10.0, 25.0),
            "left": _arm_traj(50, 0.90, 0.92, -0.15, -0.10, -0.30, -5.0, -5.0),
        },
        2: {
            "right": _arm_traj(60, 1.01, 1.12, 0.09, 0.18, -0.35, 40.0, 55.0),
            "left": _arm_traj(60, 0.88, 0.94, -0.16, -0.08, -0.30, 5.0, 15.0),
        },
    }

    tool_offset_right = np.array([0.0, 0.0, -0.15], dtype=np.float64)
    distribution = compute_distribution_from_states(
        repo_id="Synthetic/demo",
        revision="synthetic0001",
        fps=30.0,
        episode_lengths=ep_lengths,
        arm_states_by_episode=arm_states,
        rows=rows,
        task_spec=spec,
        annotations_path="/path/to/.xense-viewer/subtasks/Synthetic__demo@synthetic0001.csv",
        task_spec_path="/path/to/.xense-viewer/subtasks/Synthetic__demo.task_spec.yaml",
        trajectory_samples=5,
        tool_offsets={"right": tool_offset_right},
        calib_path="/path/to/.xense-viewer/calib/Synthetic__demo.head.json",
        calib_mtime="2026-01-01T00:00:00+00:00",
    )

    intr = CameraIntrinsics(
        camera="observation.images.head",
        image_size=(640, 480),
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        distortion_model="inverse_brown_conrady",
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        raw={
            "version": 1,
            "camera": "observation.images.head",
            "image_size": [640, 480],
            "fx": 600.0,
            "fy": 600.0,
            "cx": 320.0,
            "cy": 240.0,
            "distortion_model": "inverse_brown_conrady",
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )
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

    cal = HeadCalibration(
        repo_id="Synthetic/demo",
        revision="synthetic0001",
        camera="observation.images.head",
        image_size=(640, 480),
        intrinsics=intr,
        arms={
            "right": ArmCalibration(
                arm="right",
                T_cam_base=T_cb,
                tool_offset=tool_offset_right,
                n_points=15,
                rms_px=0.5,
                loo_rms_px=0.7,
                residuals=(),
            )
        },
    )
    z_tool = -0.35 + float(tool_offset_right[2])
    base_corners = np.array(
        [
            [0.90, -0.02, z_tool],
            [1.18, -0.02, z_tool],
            [1.18, 0.24, z_tool],
            [0.90, 0.24, z_tool],
        ],
        dtype=np.float64,
    )
    targets = {
        "version": 1,
        "camera": "observation.images.head",
        "image_size": [640, 480],
        "polygon": project_base_points(base_corners, T_cb, intr).tolist(),
    }

    suggestions = generate_suggestions(
        distribution,
        count=4,
        test_per_tier=2,
        targets=targets,
        calibration=cal,
        seed=0,
        layer="tool",
        distribution_path="/path/to/.xense-viewer/distribution/Synthetic__demo@synthetic0001.json",
        targets_path="/path/to/.xense-viewer/targets/Synthetic__demo.json",
        calib_path="/path/to/.xense-viewer/calib/Synthetic__demo.head.json",
    )
    return distribution, suggestions


def test_committed_fixtures_match_current_schema_output() -> None:
    expected_dist, expected_suggest = build_synthetic_fixtures()
    assert DIST_FIXTURE.is_file(), f"Missing fixture: {DIST_FIXTURE}"
    assert SUGGEST_FIXTURE.is_file(), f"Missing fixture: {SUGGEST_FIXTURE}"

    actual_dist = json.loads(DIST_FIXTURE.read_text(encoding="utf-8"))
    actual_suggest = json.loads(SUGGEST_FIXTURE.read_text(encoding="utf-8"))
    assert actual_dist == expected_dist
    assert actual_suggest == expected_suggest
