from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.io_utils import serialize_dict, write_info, write_json, write_tasks
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from scipy.spatial.transform import Rotation

from lerobot_check import cli
from lerobot_check.dataset import rotation_6d_to_matrix
from lerobot_check.tcp_retarget import (
    EXPECTED_TCP_LOCATION_ORDER,
    compute_arm_deltas,
    load_tool_snapshot,
    retarget_calib_payload,
    retarget_tcp_dataset,
    verify_tcp_retarget,
)

BIMANUAL_FEATURE_NAMES = [
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

SINGLE_ARM_FEATURE_NAMES = [
    "tcp.x",
    "tcp.y",
    "tcp.z",
    "tcp.r1",
    "tcp.r2",
    "tcp.r3",
    "tcp.r4",
    "tcp.r5",
    "tcp.r6",
    "gripper.pos",
]


def _write_snapshot(
    path: Path,
    arms: dict[str, dict[str, Any]],
    *,
    order: Sequence[str] = EXPECTED_TCP_LOCATION_ORDER,
    tool_name: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "captured_at": "2026-11-27T00:00:00+00:00",
        "host": "test-host",
        "tcp_location_order": list(order),
        "arms": arms,
    }
    if tool_name is not None:
        payload["tool_name"] = tool_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _make_pure_translation_snapshots(tmp_path: Path) -> tuple[Path, Path]:
    old_path = _write_snapshot(
        tmp_path / "old_tool.json",
        {
            "left": {
                "robot_serial": "Rizon4-063786",
                "tcp_location": [0.000324, 0.011005, 0.191588, 1.0, 0.0, 0.0, 0.0],
            },
            "right": {
                "robot_serial": "Rizon4R-062096",
                "tcp_location": [0.001902, -0.011472, 0.194608, 1.0, 0.0, 0.0, 0.0],
            },
        },
        tool_name="xenseflare",
    )
    new_path = _write_snapshot(
        tmp_path / "new_tool.json",
        {
            "left": {
                "robot_serial": "Rizon4-063786",
                "tcp_location": [-0.002471, 0.035214, 0.222889, 1.0, 0.0, 0.0, 0.0],
            },
            "right": {
                "robot_serial": "Rizon4R-062096",
                "tcp_location": [-0.000480, -0.033730, 0.221160, 1.0, 0.0, 0.0, 0.0],
            },
        },
    )
    return old_path, new_path


def _rot_to_wxyz(rot: Rotation) -> list[float]:
    qx, qy, qz, qw = rot.as_quat()
    return [float(qw), float(qx), float(qy), float(qz)]


def _make_general_se3_snapshots(tmp_path: Path) -> tuple[Path, Path]:
    r_old_l = Rotation.from_euler("xyz", [15.0, -25.0, 40.0], degrees=True)
    r_new_l = Rotation.from_euler("xyz", [-30.0, 12.0, 85.0], degrees=True)
    r_old_r = Rotation.from_euler("zyx", [50.0, 10.0, -20.0], degrees=True)
    r_new_r = Rotation.from_euler("zyx", [-15.0, 35.0, 60.0], degrees=True)

    q_old_l = [v * 2.5 for v in _rot_to_wxyz(r_old_l)]
    old_path = _write_snapshot(
        tmp_path / "se3_old.json",
        {
            "left": {
                "robot_serial": "S-L",
                "tcp_location": [0.01, -0.02, 0.18, *q_old_l],
            },
            "right": {
                "robot_serial": "S-R",
                "tcp_location": [-0.015, 0.008, 0.19, *_rot_to_wxyz(r_old_r)],
            },
        },
    )
    new_path = _write_snapshot(
        tmp_path / "se3_new.json",
        {
            "left": {
                "robot_serial": "S-L",
                "tcp_location": [-0.02, 0.03, 0.24, *_rot_to_wxyz(r_new_l)],
            },
            "right": {
                "robot_serial": "S-R",
                "tcp_location": [0.005, -0.025, 0.23, *_rot_to_wxyz(r_new_r)],
            },
        },
    )
    return old_path, new_path


def _random_bimanual_vectors(rng: np.random.Generator, length: int) -> np.ndarray:
    arr = np.empty((length, 20), dtype=np.float32)
    for offset in (0, 10):
        arr[:, offset : offset + 3] = rng.uniform(-0.6, 0.6, size=(length, 3)).astype(np.float32)
        rots = Rotation.random(length, random_state=rng).as_matrix()
        r6d = np.concatenate([rots[:, :, 0], rots[:, :, 1]], axis=-1).astype(np.float32)
        arr[:, offset + 3 : offset + 9] = r6d
        arr[:, offset + 9] = rng.uniform(0.0, 1.0, size=(length,)).astype(np.float32)
    return arr


def _random_single_arm_vectors(rng: np.random.Generator, length: int) -> np.ndarray:
    arr = np.empty((length, 10), dtype=np.float32)
    arr[:, 0:3] = rng.uniform(-0.6, 0.6, size=(length, 3)).astype(np.float32)
    rots = Rotation.random(length, random_state=rng).as_matrix()
    arr[:, 3:9] = np.concatenate([rots[:, :, 0], rots[:, :, 1]], axis=-1).astype(np.float32)
    arr[:, 9] = rng.uniform(0.0, 1.0, size=(length,)).astype(np.float32)
    return arr


def _make_list_arrow_array(matrix: np.ndarray, arrow_type: pa.DataType) -> pa.Array:
    flat = pa.array(matrix.reshape(-1), type=pa.float32())
    if pa.types.is_fixed_size_list(arrow_type):
        return pa.FixedSizeListArray.from_arrays(flat, type=arrow_type)
    dim = matrix.shape[1]
    offsets = pa.array(np.arange(0, (matrix.shape[0] + 1) * dim, dim, dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, flat, type=arrow_type)


def _build_synthetic_tcp_dataset(
    root: Path,
    episode_lengths: Sequence[int],
    *,
    fps: int = 30,
    episodes_per_file: int = 2,
    bimanual: bool = True,
    fixed_size_list: bool = True,
    include_cache: bool = True,
    seed: int = 7,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    feature_names = BIMANUAL_FEATURE_NAMES if bimanual else SINGLE_ARM_FEATURE_NAMES
    dim = len(feature_names)
    vec_type = (
        pa.list_(pa.field("element", pa.float32()), dim)
        if fixed_size_list
        else pa.list_(pa.field("element", pa.float32()))
    )
    hf_meta = {
        b"huggingface": json.dumps(
            {"info": {"features": {"action": {"_type": "Sequence"}, "observation.state": {}}}}
        ).encode("utf-8")
    }
    data_schema = pa.schema(
        [
            pa.field("action", vec_type),
            pa.field("observation.state", vec_type),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ],
        metadata=hf_meta,
    )

    episode_rows: list[dict[str, object]] = []
    episode_stats_list: list[dict[str, dict[str, np.ndarray]]] = []
    global_idx = 0

    for file_idx, start_ep in enumerate(range(0, len(episode_lengths), episodes_per_file)):
        batch_eps = list(range(start_ep, min(start_ep + episodes_per_file, len(episode_lengths))))
        actions_all: list[np.ndarray] = []
        states_all: list[np.ndarray] = []
        ts_all: list[np.ndarray] = []
        frame_all: list[np.ndarray] = []
        ep_all: list[np.ndarray] = []
        idx_all: list[np.ndarray] = []
        task_all: list[np.ndarray] = []

        for ep in batch_eps:
            length = episode_lengths[ep]
            actions = (
                _random_bimanual_vectors(rng, length)
                if bimanual
                else _random_single_arm_vectors(rng, length)
            )
            states = (
                _random_bimanual_vectors(rng, length)
                if bimanual
                else _random_single_arm_vectors(rng, length)
            )
            timestamps = np.arange(length, dtype=np.float32) / float(fps)
            frames = np.arange(length, dtype=np.int64)
            eps = np.full(length, ep, dtype=np.int64)
            indices = np.arange(global_idx, global_idx + length, dtype=np.int64)
            tasks = np.zeros(length, dtype=np.int64)

            actions_all.append(actions)
            states_all.append(states)
            ts_all.append(timestamps)
            frame_all.append(frames)
            ep_all.append(eps)
            idx_all.append(indices)
            task_all.append(tasks)

            ep_stats = {
                "action": get_feature_stats(actions, axis=0, keepdims=False),
                "observation.state": get_feature_stats(states, axis=0, keepdims=False),
                "timestamp": get_feature_stats(timestamps, axis=0, keepdims=True),
                "frame_index": get_feature_stats(frames, axis=0, keepdims=True),
                "episode_index": get_feature_stats(eps, axis=0, keepdims=True),
                "index": get_feature_stats(indices, axis=0, keepdims=True),
                "task_index": get_feature_stats(tasks, axis=0, keepdims=True),
            }
            episode_stats_list.append(ep_stats)

            ep_row: dict[str, object] = {
                "episode_index": ep,
                "tasks": ["Pack the box."],
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": file_idx,
                "dataset_from_index": global_idx,
                "dataset_to_index": global_idx + length,
                "videos/observation.images.head/chunk_index": 0,
                "videos/observation.images.head/file_index": file_idx,
                "videos/observation.images.head/from_timestamp": 0.0,
                "videos/observation.images.head/to_timestamp": float(length) / float(fps),
            }
            for feat_name, feat_stats in ep_stats.items():
                for stat_name, val in feat_stats.items():
                    ep_row[f"stats/{feat_name}/{stat_name}"] = np.asarray(val).tolist()
            episode_rows.append(ep_row)
            global_idx += length

        data_table = pa.table(
            {
                "action": _make_list_arrow_array(np.vstack(actions_all), vec_type),
                "observation.state": _make_list_arrow_array(np.vstack(states_all), vec_type),
                "timestamp": pa.array(np.concatenate(ts_all), type=pa.float32()),
                "frame_index": pa.array(np.concatenate(frame_all), type=pa.int64()),
                "episode_index": pa.array(np.concatenate(ep_all), type=pa.int64()),
                "index": pa.array(np.concatenate(idx_all), type=pa.int64()),
                "task_index": pa.array(np.concatenate(task_all), type=pa.int64()),
            },
            schema=data_schema,
        )
        data_out = root / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet"
        data_out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(data_table, data_out)

        video_out = (
            root / "videos" / "observation.images.head" / "chunk-000" / f"file-{file_idx:03d}.mp4"
        )
        video_out.parent.mkdir(parents=True, exist_ok=True)
        video_out.write_bytes(f"dummy-mp4-file-{file_idx}".encode())

    ep_table = pa.Table.from_pylist(episode_rows)
    ep_table = ep_table.replace_schema_metadata({b"huggingface": b'{"episodes": true}'})
    ep_out = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(ep_table, ep_out)

    tasks_df = pd.DataFrame({"task_index": [0]}, index=pd.Index(["Pack the box."], name="task"))
    write_tasks(tasks_df, root)

    agg = aggregate_stats(episode_stats_list)
    write_json(serialize_dict(agg), root / "meta" / "stats.json")

    info = {
        "codebase_version": "v3.0",
        "robot_type": "bi_flexiv_rizon4_rt" if bimanual else "flexiv_rizon4_rt",
        "total_episodes": len(episode_lengths),
        "total_frames": global_idx,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps,
        "splits": {"train": f"0:{len(episode_lengths)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [dim], "names": feature_names},
            "observation.state": {"dtype": "float32", "shape": [dim], "names": feature_names},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    write_info(info, root)

    if include_cache:
        cache_file = root / ".cache" / "huggingface" / "trees" / f"{'a' * 40}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("{}", encoding="utf-8")

    return root


def test_pure_translation_preserves_rot6d_bitwise_and_shifts_position(tmp_path: Path) -> None:
    old_tool, new_tool = _make_pure_translation_snapshots(tmp_path)
    src = _build_synthetic_tcp_dataset(tmp_path / "src", [12, 10, 8], episodes_per_file=2)
    dst = tmp_path / "dst"

    result = retarget_tcp_dataset(src, dst, old_tool, new_tool)
    assert result.total_episodes == 3
    assert result.total_frames == 30
    assert result.arms == ("left", "right")
    assert result.data_files_written == 2
    assert result.episode_files_written == 1
    assert not (dst / ".cache").exists()

    old_snap = load_tool_snapshot(old_tool)
    new_snap = load_tool_snapshot(new_tool)
    deltas = compute_arm_deltas(old_snap, new_snap, required_arms=("left", "right"))
    assert deltas["left"].is_pure_translation
    assert deltas["right"].is_pure_translation

    for rel in sorted(p.relative_to(src) for p in (src / "data").glob("*/*.parquet")):
        src_tbl = pq.read_table(src / rel)
        dst_tbl = pq.read_table(dst / rel)
        assert src_tbl.schema.equals(dst_tbl.schema, check_metadata=True)
        assert pa.types.is_fixed_size_list(dst_tbl.schema.field("action").type)
        assert dst_tbl.schema.field("action").type.value_field.name == "element"
        assert dst_tbl.schema.metadata is not None
        assert b"huggingface" in dst_tbl.schema.metadata

        for feat in ("observation.state", "action"):
            src_mat = np.stack(src_tbl[feat].to_numpy()).astype(np.float32)
            dst_mat = np.stack(dst_tbl[feat].to_numpy()).astype(np.float32)
            for arm_name, offset in (("left", 0), ("right", 10)):
                src_rot = np.ascontiguousarray(src_mat[:, offset + 3 : offset + 9])
                dst_rot = np.ascontiguousarray(dst_mat[:, offset + 3 : offset + 9])
                assert np.array_equal(src_rot.view(np.uint32), dst_rot.view(np.uint32))

                r_world = rotation_6d_to_matrix(src_rot)
                expected_xyz = src_mat[:, offset : offset + 3].astype(np.float64) + np.einsum(
                    "nij,j->ni", r_world, deltas[arm_name].translation
                )
                np.testing.assert_allclose(
                    dst_mat[:, offset : offset + 3],
                    expected_xyz.astype(np.float32),
                    rtol=0.0,
                    atol=1e-6,
                )
                src_grip = np.ascontiguousarray(src_mat[:, offset + 9])
                dst_grip = np.ascontiguousarray(dst_mat[:, offset + 9])
                assert np.array_equal(src_grip.view(np.uint32), dst_grip.view(np.uint32))

    ver = verify_tcp_retarget(src, dst, old_tool, new_tool)
    assert ver.ok, ver.errors
    assert ver.max_rot_err == 0.0
    assert ver.max_roundtrip_rot_err == 0.0
    assert ver.max_pos_err_m < 1e-6
    assert ver.max_roundtrip_pos_err_m < 1e-6

    meta = LeRobotDatasetMetadata(repo_id="org/synthetic", root=dst, force_cache_sync=False)
    assert meta.total_episodes == 3
    assert meta.total_frames == 30


def test_general_se3_composition_and_roundtrip(tmp_path: Path) -> None:
    old_tool, new_tool = _make_general_se3_snapshots(tmp_path)
    src = _build_synthetic_tcp_dataset(tmp_path / "src", [14, 9, 11], episodes_per_file=2, seed=19)
    dst = tmp_path / "dst"
    roundtrip = tmp_path / "roundtrip"

    retarget_tcp_dataset(src, dst, old_tool, new_tool)
    ver = verify_tcp_retarget(src, dst, old_tool, new_tool)
    assert ver.ok, ver.errors
    assert ver.max_pos_err_m < 1e-5
    assert ver.max_rot_err < 1e-5
    assert ver.max_roundtrip_pos_err_m < 1e-5
    assert ver.max_roundtrip_rot_err < 1e-5

    old_snap = load_tool_snapshot(old_tool)
    new_snap = load_tool_snapshot(new_tool)
    src_tbl = pq.read_table(src / "data" / "chunk-000" / "file-000.parquet")
    dst_tbl = pq.read_table(dst / "data" / "chunk-000" / "file-000.parquet")
    src_state = np.stack(src_tbl["observation.state"].to_numpy()).astype(np.float64)
    dst_state = np.stack(dst_tbl["observation.state"].to_numpy()).astype(np.float64)

    for arm_name, offset in (("left", 0), ("right", 10)):
        r_old_tool = old_snap.arms[arm_name].rotation
        t_old_tool = old_snap.arms[arm_name].translation
        r_new_tool = new_snap.arms[arm_name].rotation
        t_new_tool = new_snap.arms[arm_name].translation

        r_old_world = rotation_6d_to_matrix(src_state[:, offset + 3 : offset + 9])
        p_old_world = src_state[:, offset : offset + 3]

        r_flange_world = np.einsum("nij,jk->nik", r_old_world, r_old_tool.T)
        p_flange_world = p_old_world - np.einsum("nij,j->ni", r_flange_world, t_old_tool)

        r_expected_world = np.einsum("nij,jk->nik", r_flange_world, r_new_tool)
        p_expected_world = p_flange_world + np.einsum("nij,j->ni", r_flange_world, t_new_tool)

        r_actual_world = rotation_6d_to_matrix(dst_state[:, offset + 3 : offset + 9])
        p_actual_world = dst_state[:, offset : offset + 3]

        np.testing.assert_allclose(p_actual_world, p_expected_world, atol=1e-6)
        np.testing.assert_allclose(r_actual_world, r_expected_world, atol=1e-6)

    retarget_tcp_dataset(dst, roundtrip, new_tool, old_tool)
    ver_back = verify_tcp_retarget(dst, roundtrip, new_tool, old_tool)
    assert ver_back.ok, ver_back.errors

    rt_tbl = pq.read_table(roundtrip / "data" / "chunk-000" / "file-000.parquet")
    rt_state = np.stack(rt_tbl["observation.state"].to_numpy()).astype(np.float64)
    np.testing.assert_allclose(rt_state, src_state, atol=1e-5)


def test_single_arm_and_variable_size_list_schema(tmp_path: Path) -> None:
    old_tool = _write_snapshot(
        tmp_path / "old_single.json",
        {"default": {"tcp_location": [0.0, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0]}},
    )
    new_tool = _write_snapshot(
        tmp_path / "new_single.json",
        {"default": {"tcp_location": [0.01, -0.02, 0.20, 1.0, 0.0, 0.0, 0.0]}},
    )
    src = _build_synthetic_tcp_dataset(
        tmp_path / "src_single",
        [8, 6],
        bimanual=False,
        fixed_size_list=False,
    )
    dst = tmp_path / "dst_single"

    res = retarget_tcp_dataset(src, dst, old_tool, new_tool)
    assert res.arms == ("default",)

    dst_tbl = pq.read_table(dst / "data" / "chunk-000" / "file-000.parquet")
    assert pa.types.is_list(dst_tbl.schema.field("action").type)
    ver = verify_tcp_retarget(src, dst, old_tool, new_tool)
    assert ver.ok, ver.errors


def test_snapshot_validation_errors(tmp_path: Path) -> None:
    bad_order = _write_snapshot(
        tmp_path / "bad_order.json",
        {"left": {"tcp_location": [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]}},
        order=("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"),
    )
    with pytest.raises(ValueError, match="tcp_location_order"):
        load_tool_snapshot(bad_order)

    bad_len = _write_snapshot(
        tmp_path / "bad_len.json",
        {"left": {"tcp_location": [0.0, 0.0, 0.1, 1.0, 0.0, 0.0]}},
    )
    with pytest.raises(ValueError, match="length 7"):
        load_tool_snapshot(bad_len)

    zero_quat = _write_snapshot(
        tmp_path / "zero_quat.json",
        {"left": {"tcp_location": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]}},
    )
    with pytest.raises(ValueError, match="zero or non-finite norm"):
        load_tool_snapshot(zero_quat)

    empty_arms = _write_snapshot(tmp_path / "empty_arms.json", {})
    with pytest.raises(ValueError, match="non-empty 'arms'"):
        load_tool_snapshot(empty_arms)

    serial_a = _write_snapshot(
        tmp_path / "serial_a.json",
        {"left": {"robot_serial": "Rizon4-A", "tcp_location": [0, 0, 0.1, 1, 0, 0, 0]}},
    )
    serial_b = _write_snapshot(
        tmp_path / "serial_b.json",
        {"left": {"robot_serial": "Rizon4-B", "tcp_location": [0, 0, 0.2, 1, 0, 0, 0]}},
    )
    with pytest.raises(ValueError, match="robot_serial mismatch"):
        compute_arm_deltas(load_tool_snapshot(serial_a), load_tool_snapshot(serial_b))

    missing_right = _write_snapshot(
        tmp_path / "missing_right.json",
        {"left": {"robot_serial": "Rizon4-A", "tcp_location": [0, 0, 0.2, 1, 0, 0, 0]}},
    )
    both_arms = _write_snapshot(
        tmp_path / "both_arms.json",
        {
            "left": {"robot_serial": "Rizon4-A", "tcp_location": [0, 0, 0.1, 1, 0, 0, 0]},
            "right": {"robot_serial": "Rizon4-R", "tcp_location": [0, 0, 0.1, 1, 0, 0, 0]},
        },
    )
    with pytest.raises(ValueError, match="missing from new tool snapshot"):
        compute_arm_deltas(
            load_tool_snapshot(both_arms),
            load_tool_snapshot(missing_right),
            required_arms=("left", "right"),
        )


def test_retarget_refuses_existing_dst_and_cleans_up_on_midwrite_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_tool, new_tool = _make_pure_translation_snapshots(tmp_path)
    src = _build_synthetic_tcp_dataset(tmp_path / "src", [8, 6], episodes_per_file=2)

    existing_dst = tmp_path / "existing"
    existing_dst.mkdir()
    marker = existing_dst / "keep.txt"
    marker.write_text("do not touch", encoding="utf-8")

    with pytest.raises(FileExistsError):
        retarget_tcp_dataset(src, existing_dst, old_tool, new_tool)
    assert marker.read_text(encoding="utf-8") == "do not touch"

    partial_dst = tmp_path / "partial"
    real_write_json = write_json

    def failing_write_json(data: Any, path: Path) -> None:
        if path.name == "stats.json":
            raise RuntimeError("simulated write failure")
        real_write_json(data, path)

    monkeypatch.setattr("lerobot_check.tcp_retarget.write_json", failing_write_json)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        retarget_tcp_dataset(src, partial_dst, old_tool, new_tool)
    assert not partial_dst.exists()


def test_verify_tcp_detects_tampering(tmp_path: Path) -> None:
    old_tool, new_tool = _make_pure_translation_snapshots(tmp_path)
    src = _build_synthetic_tcp_dataset(tmp_path / "src", [10, 8], episodes_per_file=2)
    dst = tmp_path / "dst"
    retarget_tcp_dataset(src, dst, old_tool, new_tool)
    assert verify_tcp_retarget(src, dst, old_tool, new_tool).ok

    bad_action = tmp_path / "bad_action"
    shutil.copytree(dst, bad_action, copy_function=os.link)
    data_path = bad_action / "data" / "chunk-000" / "file-000.parquet"
    tbl = pq.read_table(data_path)
    act = np.stack(tbl["action"].to_numpy()).astype(np.float32)
    act[0, 0] += np.float32(0.01)
    pos = tbl.schema.get_field_index("action")
    tbl = tbl.set_column(
        pos,
        tbl.schema.field(pos),
        _make_list_arrow_array(act, tbl.schema.field(pos).type),
    )
    data_path.unlink()
    pq.write_table(tbl, data_path)
    res = verify_tcp_retarget(src, bad_action, old_tool, new_tool)
    assert not res.ok
    assert any("forward position error" in err for err in res.errors)

    bad_gripper = tmp_path / "bad_gripper"
    shutil.copytree(dst, bad_gripper, copy_function=os.link)
    data_path = bad_gripper / "data" / "chunk-000" / "file-000.parquet"
    tbl = pq.read_table(data_path)
    state = np.stack(tbl["observation.state"].to_numpy()).astype(np.float32)
    state[0, 9] += np.float32(0.05)
    pos = tbl.schema.get_field_index("observation.state")
    tbl = tbl.set_column(
        pos,
        tbl.schema.field(pos),
        _make_list_arrow_array(state, tbl.schema.field(pos).type),
    )
    data_path.unlink()
    pq.write_table(tbl, data_path)
    res = verify_tcp_retarget(src, bad_gripper, old_tool, new_tool)
    assert not res.ok
    assert any("non-TCP dimensions" in err for err in res.errors)

    bad_stats = tmp_path / "bad_stats"
    shutil.copytree(dst, bad_stats, copy_function=os.link)
    stats_path = bad_stats / "meta" / "stats.json"
    stats_data = json.loads(stats_path.read_text(encoding="utf-8"))
    stats_data["action"]["mean"][0] += 10.0
    stats_path.unlink()
    stats_path.write_text(json.dumps(stats_data, indent=2), encoding="utf-8")
    res = verify_tcp_retarget(src, bad_stats, old_tool, new_tool)
    assert not res.ok
    assert any("meta/stats.json action/mean" in err for err in res.errors)

    bad_ep_stats = tmp_path / "bad_ep_stats"
    shutil.copytree(dst, bad_ep_stats, copy_function=os.link)
    ep_path = bad_ep_stats / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_tbl = pq.read_table(ep_path)
    ep_rows = ep_tbl.to_pylist()
    ep_rows[0]["stats/observation.state/mean"][0] += 5.0
    ep_path.unlink()
    pq.write_table(pa.Table.from_pylist(ep_rows, schema=ep_tbl.schema), ep_path)
    res = verify_tcp_retarget(src, bad_ep_stats, old_tool, new_tool)
    assert not res.ok
    assert any("stats/observation.state/mean" in err for err in res.errors)

    bad_video = tmp_path / "bad_video"
    shutil.copytree(dst, bad_video, copy_function=shutil.copy2)
    res = verify_tcp_retarget(src, bad_video, old_tool, new_tool)
    assert not res.ok
    assert any("hard link" in err for err in res.errors)


def test_cli_retarget_tcp_and_verify_tcp(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    old_tool, new_tool = _make_general_se3_snapshots(tmp_path)
    src = _build_synthetic_tcp_dataset(tmp_path / "src", [10, 12, 8], episodes_per_file=2)
    dst = tmp_path / "dst"

    code = cli.main(
        [
            "retarget-tcp",
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--old-tool",
            str(old_tool),
            "--new-tool",
            str(new_tool),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "Wrote" in captured.out

    code = cli.main(
        [
            "verify-tcp",
            "--dataset-root",
            str(src),
            "--dst",
            str(dst),
            "--old-tool",
            str(old_tool),
            "--new-tool",
            str(new_tool),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.out + captured.err
    assert "OK" in captured.out

    code_exists = cli.main(
        [
            "retarget-tcp",
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--old-tool",
            str(old_tool),
            "--new-tool",
            str(new_tool),
        ]
    )
    captured = capsys.readouterr()
    assert code_exists == 1
    assert "already exists" in captured.err


def test_retarget_calib_payload_preserves_fields_and_transforms_tcp(tmp_path: Path) -> None:
    old_pure, new_pure = _make_pure_translation_snapshots(tmp_path)
    old_se3, new_se3 = _make_general_se3_snapshots(tmp_path)

    r_l = Rotation.from_euler("xyz", [20.0, 15.0, -35.0], degrees=True).as_matrix()
    r_r = Rotation.from_euler("xyz", [-10.0, 45.0, 30.0], degrees=True).as_matrix()
    payload = {
        "version": 1,
        "repo_id": "Xense/pack_iphone_box_1119",
        "revision": "oldrev",
        "camera": "observation.images.head",
        "image_size": [640, 480],
        "frames": [
            {
                "id": "left-001",
                "arm": "left",
                "episode_index": 0,
                "frame_index": 5,
                "tcp": [0.1, 0.2, 0.3, *r_l[:, 0].tolist(), *r_l[:, 1].tolist()],
                "pixel": [320.5, 240.25],
                "skipped": False,
                "speed_mps": 0.01,
            },
            {
                "id": "right-001",
                "arm": "right",
                "episode_index": 1,
                "frame_index": 12,
                "tcp": [-0.15, -0.05, 0.28, *r_r[:, 0].tolist(), *r_r[:, 1].tolist()],
                "pixel": None,
                "skipped": True,
                "speed_mps": 0.005,
            },
        ],
    }
    original = copy.deepcopy(payload)

    retargeted_pure = retarget_calib_payload(
        payload,
        old_pure,
        new_pure,
        repo_id="Xense/pack_iphone_box_1119_taccap",
        revision="newrev",
    )
    assert payload == original
    assert retargeted_pure["repo_id"] == "Xense/pack_iphone_box_1119_taccap"
    assert retargeted_pure["revision"] == "newrev"
    assert retargeted_pure["frames"][0]["pixel"] == [320.5, 240.25]
    assert retargeted_pure["frames"][0]["id"] == "left-001"
    assert retargeted_pure["frames"][1]["skipped"] is True
    assert retargeted_pure["frames"][0]["tcp"][3:9] == original["frames"][0]["tcp"][3:9]
    assert retargeted_pure["frames"][1]["tcp"][3:9] == original["frames"][1]["tcp"][3:9]

    deltas_pure = compute_arm_deltas(load_tool_snapshot(old_pure), load_tool_snapshot(new_pure))
    expected_p_l = np.array([0.1, 0.2, 0.3]) + r_l @ deltas_pure["left"].translation
    np.testing.assert_allclose(retargeted_pure["frames"][0]["tcp"][0:3], expected_p_l, atol=1e-12)

    retargeted_se3 = retarget_calib_payload(payload, old_se3, new_se3)
    assert retargeted_se3["repo_id"] == "Xense/pack_iphone_box_1119"
    assert retargeted_se3["revision"] == "oldrev"
    roundtrip_se3 = retarget_calib_payload(retargeted_se3, new_se3, old_se3)
    for idx in range(2):
        np.testing.assert_allclose(
            roundtrip_se3["frames"][idx]["tcp"],
            original["frames"][idx]["tcp"],
            atol=1e-12,
        )


def test_lazy_tcp_retarget_exports_do_not_eagerly_import() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

import lerobot_check.alignment

assert 'lerobot_check.tcp_retarget' not in sys.modules
assert 'lerobot.datasets.io_utils' not in sys.modules

import lerobot_check

for name in (
    'RetargetTcpResult',
    'VerifyTcpResult',
    'retarget_tcp_dataset',
    'verify_tcp_retarget',
    'retarget_calib_payload',
):
    assert name in dir(lerobot_check)
    val = getattr(lerobot_check, name)
    import lerobot_check.tcp_retarget as mod
    assert val is getattr(mod, name)
""",
        ],
        check=True,
    )
