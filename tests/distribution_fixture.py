from __future__ import annotations

import math
from typing import Any

import numpy as np

from lerobot_check.dataset import ArmState
from lerobot_check.schema import AnnotationRow, CoverageSpec, ObjectSpec, SubtaskSpec, TaskSpec

REPO_ID = "Org/demo"
REVISION = "rev1"


def regression_task_spec() -> TaskSpec:
    return TaskSpec(
        name="demo_task",
        subtasks=(
            SubtaskSpec(name="pick", prompt="Pick object.", arms=("right",)),
            SubtaskSpec(name="place", prompt="Place object.", arms=("left", "right")),
        ),
        objects=(
            ObjectSpec("phone", "pick", "grasp", "right", 360.0),
            ObjectSpec("box", "place", "release", "left", 180.0),
        ),
        coverage=CoverageSpec(xy_bin_m=0.05, yaw_bin_deg=30.0, sparse_threshold=2),
    )


def _arm_state(arm: str, ep: int, n: int) -> ArmState:
    t = np.linspace(0.0, 1.0, n)
    sign = 1.0 if arm == "right" else -1.0
    xyz = np.column_stack(
        [0.7 + 0.1 * t + 0.01 * ep, sign * (0.1 + 0.05 * t), -0.40 + 0.02 * np.sin(3 * t + ep)]
    )
    yaw = np.radians(20.0 * ep + 30.0 * t)
    rot6d = np.column_stack(
        [np.cos(yaw), np.sin(yaw), np.zeros(n), -np.sin(yaw), np.cos(yaw), np.zeros(n)]
    )
    gripper = 0.5 + 0.5 * np.cos(2 * math.pi * t)
    return ArmState(arm, xyz, rot6d, gripper)


def _row(ep: int, subtask: str, start: int, end: int, grasp: int | None) -> AnnotationRow:
    return AnnotationRow(
        repo_id=REPO_ID,
        revision=REVISION,
        episode_index=ep,
        subtask=subtask,
        start=start,
        end=end,
        contact=None,
        grasp=grasp,
        success=True,
        annotator="human" if ep % 2 else "agent",
        note="",
    )


def regression_distribution_kwargs() -> dict[str, Any]:
    lengths = {0: 20, 1: 24, 2: 22, 3: 18}
    rows = [
        row
        for ep, n in lengths.items()
        if ep != 2
        for row in (_row(ep, "pick", 0, n // 2, 3 + ep), _row(ep, "place", n // 2, n, None))
    ]
    return {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "fps": 30.0,
        "episode_lengths": lengths,
        "arm_states_by_episode": {
            ep: {arm: _arm_state(arm, ep, n) for arm in ("left", "right")}
            for ep, n in lengths.items()
        },
        "rows": rows,
        "task_spec": regression_task_spec(),
        "annotations_path": "/data/ann.csv",
        "task_spec_path": "/data/spec.yaml",
        "trajectory_samples": 6,
        "tool_offsets": {"right": np.array([0.0, 0.01, 0.15]), "left": np.array([0.0, 0.0, 0.14])},
        "calib_path": "/data/head.json",
        "calib_mtime": "2026-01-01T00:00:00+00:00",
    }
