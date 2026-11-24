"""Tests for proprioceptive event proposal, hysteresis, and 6D rotation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lerobot_check.dataset import (
    ArmState,
    parse_arm_feature_indices,
    rotation_6d_to_matrix,
)
from lerobot_check.propose import (
    ProposeConfig,
    compute_tcp_linear_speed,
    detect_gripper_events,
    detect_low_speed_segments,
    load_proposal_event_frames,
    propose_episode_from_arms,
    write_proposals_csv,
)


def test_gripper_hysteresis_suppresses_threshold_jitter() -> None:
    # Two real open->close->open cycles with heavy jitter around both thresholds
    # (close_threshold=0.5, open_threshold=0.7).
    signal = np.array(
        [
            # Start open
            1.0,
            0.95,
            0.60,
            0.52,
            # First real close at frame 4 (<= 0.5)
            0.49,
            # Jitter around 0.50..0.68 must NOT trigger open or extra close events
            0.55,
            0.46,
            0.62,
            0.44,
            0.68,
            0.30,
            0.25,
            # First real open at frame 12 (>= 0.7)
            0.71,
            # Jitter around 0.52..0.75 must NOT trigger close or extra open events
            0.65,
            0.73,
            0.55,
            0.78,
            0.95,
            # Second real close at frame 18 (<= 0.5)
            0.50,
            0.54,
            0.48,
            0.69,
            0.22,
            # Second real open at frame 23 (>= 0.7)
            0.70,
            0.66,
            0.72,
            1.0,
        ],
        dtype=np.float64,
    )
    events = detect_gripper_events(
        signal,
        episode_index=3,
        arm="left",
        fps=30.0,
        close_threshold=0.5,
        open_threshold=0.7,
    )
    assert [(ev.frame, ev.kind) for ev in events] == [
        (4, "close"),
        (12, "open"),
        (18, "close"),
        (23, "open"),
    ]


def test_gripper_starting_closed_does_not_emit_spurious_close() -> None:
    signal = np.array([0.2, 0.25, 0.4, 0.75, 0.8], dtype=np.float64)
    events = detect_gripper_events(signal, episode_index=0, arm="left", fps=30.0)
    assert [(ev.frame, ev.kind) for ev in events] == [(3, "open")]


def test_invalid_hysteresis_thresholds_raise() -> None:
    with pytest.raises(ValueError, match="close_threshold"):
        ProposeConfig(close_threshold=0.7, open_threshold=0.5)


def test_tcp_linear_speed_and_low_speed_segments() -> None:
    fps = 10.0
    # Frames 0..4: stationary (speed 0.0 -> 5 frames = 0.5s >= 0.3s -> segment [0, 5))
    # Frames 5..9: fast 0.1 m/frame -> 1.0 m/s
    # Frames 10..11: brief 2-frame pause (0.2s < 0.3s -> ignored)
    # Frames 12..14: fast 1.0 m/s
    # Frames 15..19: slow 0.001 m/frame -> 0.01 m/s (5 frames = 0.5s -> segment [15, 20))
    xyz = np.zeros((20, 3), dtype=np.float64)
    for t in range(5, 10):
        xyz[t, 0] = xyz[t - 1, 0] + 0.1
    xyz[10] = xyz[9]
    xyz[11] = xyz[10]
    for t in range(12, 15):
        xyz[t, 0] = xyz[t - 1, 0] + 0.1
    for t in range(15, 20):
        xyz[t, 0] = xyz[t - 1, 0] + 0.001

    speeds = compute_tcp_linear_speed(xyz, fps=fps)
    assert speeds[0] == pytest.approx(0.0)
    assert speeds[5] == pytest.approx(1.0)
    assert speeds[15] == pytest.approx(0.01)

    segments = detect_low_speed_segments(
        speeds,
        episode_index=1,
        arm="left",
        fps=fps,
        threshold_mps=0.02,
        min_duration_s=0.3,
    )
    assert [(s.start, s.end) for s in segments] == [(0, 5), (15, 20)]


def test_propose_episode_and_csv_round_trip(tmp_path: Path) -> None:
    n = 30
    left = ArmState(
        arm="left",
        tcp_xyz=np.zeros((n, 3), dtype=np.float64),
        tcp_rot6d=np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (n, 1)),
        gripper=np.array([1.0] * 10 + [0.3] * 10 + [0.9] * 10, dtype=np.float64),
    )
    proposal = propose_episode_from_arms(
        episode_index=2,
        length=n,
        fps=30.0,
        arms=[left],
        config=ProposeConfig(),
    )
    assert proposal.first_close_frame(0, 30) == 10
    assert proposal.first_close_frame(15, 30) is None

    csv_path = tmp_path / "proposals.csv"
    write_proposals_csv([proposal], csv_path)
    loaded_frames = load_proposal_event_frames(csv_path, episode_index=2)
    assert loaded_frames == [10, 20]


def test_rotation_6d_to_matrix_matches_first_two_columns() -> None:
    theta = np.deg2rad(90.0)
    expected = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Column 0 then Column 1 of R, with slight scale/non-orthogonality to exercise Gram-Schmidt.
    col0 = expected[:, 0] * 1.2
    col1 = expected[:, 1] * 0.8 + 0.1 * expected[:, 0]
    r6d = np.concatenate([col0, col1])
    reconstructed = rotation_6d_to_matrix(r6d)
    assert np.allclose(reconstructed, expected, atol=1e-12)
    assert np.allclose(reconstructed.T @ reconstructed, np.eye(3), atol=1e-12)
    assert np.linalg.det(reconstructed) == pytest.approx(1.0)


def test_parse_arm_feature_indices_bimanual() -> None:
    names = [
        "left_tcp.x",
        "left_tcp.y",
        "left_tcp.z",
        "left_tcp.r1",
        "left_tcp.r2",
        "left_tcp.r3",
        "left_tcp.r4",
        "left_tcp.r5",
        "left_tcp.r6",
        "left_gripper.pos",
        "right_tcp.x",
        "right_tcp.y",
        "right_tcp.z",
        "right_tcp.r1",
        "right_tcp.r2",
        "right_tcp.r3",
        "right_tcp.r4",
        "right_tcp.r5",
        "right_tcp.r6",
        "right_gripper.pos",
    ]
    parsed = parse_arm_feature_indices(names)
    assert set(parsed) == {"left", "right"}
    assert parsed["left"].xyz == (0, 1, 2)
    assert parsed["left"].rot6d == (3, 4, 5, 6, 7, 8)
    assert parsed["left"].gripper == 9
    assert parsed["right"].xyz == (10, 11, 12)
    assert parsed["right"].gripper == 19
