"""Tests for relative gripper motion, squeeze, TCP speed, and 6D rotation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lerobot_check.dataset import (
    ArmState,
    parse_action_gripper_indices,
    parse_arm_feature_indices,
    rotation_6d_to_matrix,
)
from lerobot_check.propose import (
    ProposeConfig,
    compute_tcp_linear_speed,
    detect_gripper_events,
    detect_low_speed_segments,
    detect_squeeze_segments,
    load_proposal_event_frames,
    propose_episode_from_arms,
    write_proposals_csv,
)


def test_relative_hysteresis_and_squeeze_mirroring_egg_pattern() -> None:
    """Mirror real egg-pick pattern: ~0.33 plateau with +-0.01 jitter and dips to ~0.24."""
    fps = 30.0
    # Frames 0..11: plateau around 0.33 with +-0.01 jitter (must not trigger)
    # Frames 12..14: close dip 0.30 -> 0.27 -> 0.24
    # Frames 15..24: hold at ~0.24 with +-0.01 jitter; action commands 0.0 during [13, 25)
    # Frames 25..27: open rise 0.27 -> 0.30 -> 0.33
    # Frames 28..37: plateau around 0.33 with +-0.01 jitter
    # Frames 38..40: second close dip 0.30 -> 0.27 -> 0.24
    # Frames 41..48: hold at ~0.24 with +-0.01 jitter; action commands 0.0 during [39, 49)
    # Frames 49..51: second open rise 0.27 -> 0.30 -> 0.33
    # Frames 52..57: plateau around 0.33 with +-0.01 jitter
    state = np.array(
        [
            # 0..11: plateau 0.33 +- 0.01
            0.33,
            0.34,
            0.32,
            0.33,
            0.34,
            0.32,
            0.33,
            0.33,
            0.34,
            0.32,
            0.33,
            0.33,
            # 12..14: first close
            0.30,
            0.27,
            0.24,
            # 15..24: hold ~0.24 +- 0.01
            0.24,
            0.25,
            0.23,
            0.24,
            0.24,
            0.25,
            0.24,
            0.23,
            0.24,
            0.24,
            # 25..27: first open
            0.27,
            0.30,
            0.33,
            # 28..37: plateau 0.33 +- 0.01
            0.33,
            0.34,
            0.32,
            0.33,
            0.34,
            0.33,
            0.32,
            0.33,
            0.34,
            0.33,
            # 38..40: second close
            0.30,
            0.27,
            0.24,
            # 41..48: hold ~0.24 +- 0.01
            0.24,
            0.25,
            0.24,
            0.23,
            0.24,
            0.24,
            0.25,
            0.24,
            # 49..51: second open
            0.27,
            0.30,
            0.33,
            # 52..57: plateau 0.33 +- 0.01
            0.33,
            0.34,
            0.32,
            0.33,
            0.34,
            0.33,
        ],
        dtype=np.float64,
    )
    action = state.copy()
    action[13:25] = 0.0
    action[39:49] = 0.0
    # Brief 3-frame dip (0.1s < min_duration_s=0.2s) must not emit a squeeze.
    action[30:33] = 0.0

    events = detect_gripper_events(state, episode_index=0, arm="left", fps=fps, delta=0.06)
    assert [(ev.kind, ev.frame, ev.end_frame) for ev in events] == [
        ("close", 12, 14),
        ("open", 25, 27),
        ("close", 38, 40),
        ("open", 49, 51),
    ]

    squeezes = detect_squeeze_segments(
        state,
        action,
        episode_index=0,
        arm="left",
        fps=fps,
        gap=0.05,
        min_duration_s=0.2,
    )
    assert [(s.start, s.end) for s in squeezes] == [(13, 25), (39, 49)]
    assert squeezes[0].median_gap == pytest.approx(-0.24, abs=0.01)


def test_relative_hysteresis_full_cycle_and_jitter_immunity() -> None:
    """Verify full 1.0 -> 0.25 -> 1.0 cycle plus +-0.01 plateau jitter."""
    fps = 30.0
    signal = np.array(
        [
            # 0..5: open plateau ~1.0 with +-0.01 jitter
            1.00,
            0.99,
            1.00,
            0.99,
            1.00,
            1.00,
            # 6..9: close ramp 0.80 -> 0.60 -> 0.40 -> 0.25
            0.80,
            0.60,
            0.40,
            0.25,
            # 10..15: closed plateau ~0.25 with +-0.01 jitter
            0.25,
            0.26,
            0.24,
            0.25,
            0.26,
            0.25,
            # 16..19: open ramp 0.45 -> 0.65 -> 0.85 -> 1.00
            0.45,
            0.65,
            0.85,
            1.00,
            # 20..24: open plateau ~1.0 with +-0.01 jitter
            1.00,
            0.99,
            1.00,
            0.99,
            1.00,
        ],
        dtype=np.float64,
    )
    events = detect_gripper_events(signal, episode_index=1, arm="left", fps=fps, delta=0.06)
    assert [(ev.kind, ev.frame, ev.end_frame) for ev in events] == [
        ("close", 6, 9),
        ("open", 16, 19),
    ]
    assert events[0].amplitude == pytest.approx(-0.76, abs=0.02)
    assert events[1].amplitude == pytest.approx(0.76, abs=0.02)


def test_small_jitter_only_produces_zero_events() -> None:
    rng = np.random.default_rng(42)
    signal = 0.33 + rng.uniform(-0.01, 0.01, size=200)
    events = detect_gripper_events(signal, episode_index=0, arm="left", fps=30.0, delta=0.06)
    assert events == []


def test_invalid_propose_config_raises() -> None:
    with pytest.raises(ValueError, match="gripper_delta"):
        ProposeConfig(gripper_delta=0.0)
    with pytest.raises(ValueError, match="squeeze_gap"):
        ProposeConfig(squeeze_gap=-0.05)


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
    state_g = np.array([1.0] * 10 + [0.3] * 10 + [0.9] * 10, dtype=np.float64)
    action_g = state_g.copy()
    action_g[11:19] = 0.0
    left = ArmState(
        arm="left",
        tcp_xyz=np.zeros((n, 3), dtype=np.float64),
        tcp_rot6d=np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (n, 1)),
        gripper=state_g,
        action_gripper=action_g,
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
    assert [(s.start, s.end) for s in proposal.squeeze_segments] == [(11, 19)]

    csv_path = tmp_path / "proposals.csv"
    write_proposals_csv([proposal], csv_path)
    assert load_proposal_event_frames(csv_path, episode_index=2) == [10, 11, 20]
    assert load_proposal_event_frames(csv_path, episode_index=2, event_types=("squeeze",)) == [11]


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
    col0 = expected[:, 0] * 1.2
    col1 = expected[:, 1] * 0.8 + 0.1 * expected[:, 0]
    r6d = np.concatenate([col0, col1])
    reconstructed = rotation_6d_to_matrix(r6d)
    assert np.allclose(reconstructed, expected, atol=1e-12)
    assert np.allclose(reconstructed.T @ reconstructed, np.eye(3), atol=1e-12)
    assert np.linalg.det(reconstructed) == pytest.approx(1.0)


def test_parse_arm_and_action_feature_indices_bimanual() -> None:
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
    assert parse_action_gripper_indices(names) == {"left": 9, "right": 19}
