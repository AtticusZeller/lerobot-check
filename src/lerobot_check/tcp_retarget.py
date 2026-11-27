from __future__ import annotations

import copy
import math
import os
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.io_utils import serialize_dict, write_json

from lerobot_check.dataset import SUPPORTED_CODEBASE_VERSION, rotation_6d_to_matrix
from lerobot_check.json_io import read_json

EXPECTED_TCP_LOCATION_ORDER = ("x_m", "y_m", "z_m", "qw", "qx", "qy", "qz")
TARGET_FEATURES = ("observation.state", "action")
PURE_TRANSLATION_ATOL = 1e-12
ORTHONORMAL_ATOL = 1e-4
STATS_RTOL = 1e-4
STATS_ATOL = 1e-5


@dataclass(frozen=True)
class ArmToolPose:
    arm: str
    translation: np.ndarray
    rotation: np.ndarray
    robot_serial: str | None = None


@dataclass(frozen=True)
class ToolSnapshot:
    path: Path
    arms: dict[str, ArmToolPose]
    tool_name: str | None = None
    host: str | None = None


@dataclass(frozen=True)
class ArmTcpDelta:
    arm: str
    rotation: np.ndarray
    translation: np.ndarray
    is_pure_translation: bool


@dataclass(frozen=True)
class ArmTcpIndices:
    arm: str
    xyz: tuple[int, int, int]
    rot6d: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class RetargetTcpResult:
    dst: Path
    total_episodes: int
    total_frames: int
    arms: tuple[str, ...]
    data_files_written: int
    episode_files_written: int


@dataclass(frozen=True)
class VerifyTcpResult:
    total_episodes: int
    total_frames: int
    arms: tuple[str, ...]
    max_pos_err_m: float
    max_rot_err: float
    max_roundtrip_pos_err_m: float
    max_roundtrip_rot_err: float
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _quaternion_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Quaternion has zero or non-finite norm")
    w, x, y, z = (float(v) / norm for v in quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _parse_finite_float_vector(raw: Any, *, expected_len: int, context: str) -> np.ndarray:
    if not isinstance(raw, (list, tuple)) or len(raw) != expected_len:
        raise ValueError(f"{context}: expected sequence of length {expected_len}, got {raw!r}")
    values: list[float] = []
    for idx, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{context}[{idx}]: expected finite number, got {item!r}")
        val = float(item)
        if not math.isfinite(val):
            raise ValueError(f"{context}[{idx}]: expected finite number, got {item!r}")
        values.append(val)
    return np.asarray(values, dtype=np.float64)


def load_tool_snapshot(path: Path) -> ToolSnapshot:
    resolved = path.expanduser().resolve()
    raw = read_json(resolved, resolved.read_bytes())

    order = raw.get("tcp_location_order")
    if not isinstance(order, list) or tuple(order) != EXPECTED_TCP_LOCATION_ORDER:
        raise ValueError(
            f"{resolved}: expected tcp_location_order {list(EXPECTED_TCP_LOCATION_ORDER)!r}, "
            f"got {order!r}"
        )

    raw_arms = raw.get("arms")
    if not isinstance(raw_arms, Mapping) or not raw_arms:
        raise ValueError(f"{resolved}: expected non-empty 'arms' mapping")

    arms: dict[str, ArmToolPose] = {}
    for arm_name, entry in raw_arms.items():
        if not isinstance(arm_name, str) or not arm_name:
            raise ValueError(f"{resolved}: invalid arm name {arm_name!r}")
        if not isinstance(entry, Mapping):
            raise ValueError(f"{resolved}: arm {arm_name!r} entry must be a JSON object")
        tcp_loc = _parse_finite_float_vector(
            entry.get("tcp_location"),
            expected_len=7,
            context=f"{resolved}: arms.{arm_name}.tcp_location",
        )
        translation = tcp_loc[0:3].copy()
        try:
            rotation = _quaternion_wxyz_to_matrix(tcp_loc[3:7])
        except ValueError as exc:
            raise ValueError(f"{resolved}: arms.{arm_name}.tcp_location: {exc}") from exc

        raw_serial = entry.get("robot_serial")
        robot_serial: str | None = None
        if raw_serial is not None:
            if not isinstance(raw_serial, str):
                raise ValueError(
                    f"{resolved}: arms.{arm_name}.robot_serial must be a string, got {raw_serial!r}"
                )
            stripped = raw_serial.strip()
            robot_serial = stripped if stripped else None

        arms[arm_name] = ArmToolPose(
            arm=arm_name,
            translation=translation,
            rotation=rotation,
            robot_serial=robot_serial,
        )

    raw_tool_name = raw.get("tool_name")
    tool_name = raw_tool_name if isinstance(raw_tool_name, str) else None
    raw_host = raw.get("host")
    host = raw_host if isinstance(raw_host, str) else None
    return ToolSnapshot(path=resolved, arms=arms, tool_name=tool_name, host=host)


def compute_arm_deltas(
    old_snapshot: ToolSnapshot,
    new_snapshot: ToolSnapshot,
    *,
    required_arms: Sequence[str] | None = None,
) -> dict[str, ArmTcpDelta]:
    common_arms = sorted(set(old_snapshot.arms) & set(new_snapshot.arms))
    if required_arms is not None:
        target_arms = list(dict.fromkeys(required_arms))
        missing_old = [arm for arm in target_arms if arm not in old_snapshot.arms]
        if missing_old:
            raise ValueError(
                f"Arm(s) {missing_old} missing from old tool snapshot {old_snapshot.path}"
            )
        missing_new = [arm for arm in target_arms if arm not in new_snapshot.arms]
        if missing_new:
            raise ValueError(
                f"Arm(s) {missing_new} missing from new tool snapshot {new_snapshot.path}"
            )
    else:
        if not common_arms:
            raise ValueError(f"No common arms between {old_snapshot.path} and {new_snapshot.path}")
        target_arms = common_arms

    for arm in sorted(set(common_arms) | set(target_arms)):
        if arm in old_snapshot.arms and arm in new_snapshot.arms:
            old_serial = old_snapshot.arms[arm].robot_serial
            new_serial = new_snapshot.arms[arm].robot_serial
            if old_serial is not None and new_serial is not None and old_serial != new_serial:
                raise ValueError(
                    f"robot_serial mismatch for arm {arm!r}: "
                    f"old={old_serial!r} ({old_snapshot.path}) vs "
                    f"new={new_serial!r} ({new_snapshot.path})"
                )

    deltas: dict[str, ArmTcpDelta] = {}
    eye3 = np.eye(3, dtype=np.float64)
    for arm in target_arms:
        old_pose = old_snapshot.arms[arm]
        new_pose = new_snapshot.arms[arm]
        r_delta = old_pose.rotation.T @ new_pose.rotation
        t_delta = old_pose.rotation.T @ (new_pose.translation - old_pose.translation)
        is_pure_translation = bool(np.max(np.abs(r_delta - eye3)) <= PURE_TRANSLATION_ATOL)
        deltas[arm] = ArmTcpDelta(
            arm=arm,
            rotation=r_delta,
            translation=t_delta,
            is_pure_translation=is_pure_translation,
        )
    return deltas


def _transform_tcp_poses_float64(
    xyz: np.ndarray,
    rot6d: np.ndarray,
    delta: ArmTcpDelta,
) -> tuple[np.ndarray, np.ndarray]:
    p_old = np.asarray(xyz, dtype=np.float64)
    r_old = rotation_6d_to_matrix(rot6d)
    p_new = p_old + np.einsum("...ij,j->...i", r_old, delta.translation)
    if delta.is_pure_translation:
        return p_new, np.asarray(rot6d, dtype=np.float64)
    r_new = np.einsum("...ij,jk->...ik", r_old, delta.rotation)
    r6d_new = np.concatenate([r_new[..., :, 0], r_new[..., :, 1]], axis=-1)
    return p_new, r6d_new


def _parse_feature_tcp_indices(
    feature_key: str,
    feature_names: Sequence[str],
) -> dict[str, ArmTcpIndices]:
    if len(set(feature_names)) != len(feature_names):
        raise ValueError(f"Feature {feature_key!r} contains duplicate dimension names")
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    arms: dict[str, ArmTcpIndices] = {}

    for prefix, arm_name in (("left_", "left"), ("right_", "right"), ("", "default")):
        xyz_keys = (f"{prefix}tcp.x", f"{prefix}tcp.y", f"{prefix}tcp.z")
        rot_keys = tuple(f"{prefix}tcp.r{i}" for i in range(1, 7))
        all_keys = (*xyz_keys, *rot_keys)
        present = [k for k in all_keys if k in name_to_idx]
        if not present:
            continue
        if len(present) != len(all_keys):
            missing = [k for k in all_keys if k not in name_to_idx]
            raise ValueError(
                f"Feature {feature_key!r} has incomplete TCP columns for arm {arm_name!r}; "
                f"missing {missing}"
            )
        arms[arm_name] = ArmTcpIndices(
            arm=arm_name,
            xyz=(name_to_idx[xyz_keys[0]], name_to_idx[xyz_keys[1]], name_to_idx[xyz_keys[2]]),
            rot6d=(
                name_to_idx[rot_keys[0]],
                name_to_idx[rot_keys[1]],
                name_to_idx[rot_keys[2]],
                name_to_idx[rot_keys[3]],
                name_to_idx[rot_keys[4]],
                name_to_idx[rot_keys[5]],
            ),
        )

    if not arms:
        raise ValueError(
            f"Could not locate tcp.{{x,y,z,r1..r6}} features in {feature_key!r}: "
            f"{list(feature_names)}"
        )
    return arms


def _discover_feature_layouts(
    info: Mapping[str, Any],
) -> tuple[dict[str, dict[str, ArmTcpIndices]], dict[str, int], tuple[str, ...]]:
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("meta/info.json is missing 'features' mapping")

    layouts: dict[str, dict[str, ArmTcpIndices]] = {}
    dims: dict[str, int] = {}
    for key in TARGET_FEATURES:
        feat = features.get(key)
        if not isinstance(feat, Mapping):
            raise ValueError(f"meta/info.json is missing feature {key!r}")
        raw_names = feat.get("names")
        if not isinstance(raw_names, list) or not all(isinstance(n, str) for n in raw_names):
            raise ValueError(f"meta/info.json feature {key!r} must define a list of string names")
        shape = feat.get("shape")
        if isinstance(shape, list) and len(shape) == 1 and int(shape[0]) != len(raw_names):
            raise ValueError(
                f"meta/info.json feature {key!r} shape {shape} != len(names) {len(raw_names)}"
            )
        layouts[key] = _parse_feature_tcp_indices(key, raw_names)
        dims[key] = len(raw_names)

    state_arms = set(layouts["observation.state"])
    action_arms = set(layouts["action"])
    if state_arms != action_arms:
        raise ValueError(
            f"Arm sets differ between observation.state ({sorted(state_arms)}) "
            f"and action ({sorted(action_arms)})"
        )
    ordered_arms = tuple(layouts["observation.state"].keys())
    return layouts, dims, ordered_arms


def _load_v3_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata file: {info_path}")
    info = read_json(info_path, info_path.read_bytes())
    version = info.get("codebase_version")
    if version != SUPPORTED_CODEBASE_VERSION:
        raise ValueError(
            f"Expected LeRobot {SUPPORTED_CODEBASE_VERSION} dataset at {root}, got {version!r}"
        )
    return info


def _load_episode_lengths(root: Path) -> tuple[list[Path], dict[int, int]]:
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {root / 'meta'}")
    lengths: dict[int, int] = {}
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "length"])
        ep_ids = table["episode_index"].to_pylist()
        ep_lens = table["length"].to_pylist()
        for ep, length in zip(ep_ids, ep_lens, strict=True):
            ep_int = int(ep)
            if ep_int in lengths:
                raise ValueError(f"Duplicate episode_index {ep_int} in meta/episodes")
            lengths[ep_int] = int(length)
    return files, lengths


def _validate_source_dataset(
    src_root: Path,
) -> tuple[dict[str, Any], list[Path], list[Path], dict[int, int]]:
    info = _load_v3_info(src_root)
    episode_files, episode_lengths = _load_episode_lengths(src_root)
    total_episodes = int(info["total_episodes"])
    if sorted(episode_lengths) != list(range(total_episodes)):
        raise ValueError(
            f"Episode indices in meta/episodes are not contiguous 0..{total_episodes - 1}"
        )
    data_files = sorted((src_root / "data").glob("*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet files found under {src_root / 'data'}")
    stats_path = src_root / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing dataset stats file: {stats_path}")
    tasks_path = src_root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing dataset tasks file: {tasks_path}")
    return info, data_files, episode_files, episode_lengths


def _arrow_column_to_float32_matrix(column: pa.ChunkedArray, expected_dim: int) -> np.ndarray:
    col_type = column.type
    if len(column) == 0:
        return np.empty((0, expected_dim), dtype=np.float32)

    combined = column.combine_chunks()
    if combined.null_count != 0:
        raise ValueError("Unexpected null rows in vector feature column")

    if pa.types.is_fixed_size_list(col_type):
        if col_type.list_size != expected_dim:
            raise ValueError(f"Expected list_size={expected_dim}, got {col_type.list_size}")
        if not pa.types.is_float32(col_type.value_type):
            raise ValueError(f"Expected float32 list element type, got {col_type.value_type}")
        values = combined.values.slice(
            combined.offset * expected_dim,
            len(combined) * expected_dim,
        )
        if values.null_count != 0:
            raise ValueError("Unexpected null elements in fixed-size list column")
        flat = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32)
        return flat.reshape(len(combined), expected_dim).copy()

    if pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
        if not pa.types.is_float32(col_type.value_type):
            raise ValueError(f"Expected float32 list element type, got {col_type.value_type}")
        offsets = combined.offsets.to_numpy(zero_copy_only=False)
        lengths = np.diff(offsets)
        if not np.all(lengths == expected_dim):
            raise ValueError(f"Expected every list row to have length {expected_dim}")
        start = int(offsets[0])
        end = int(offsets[-1])
        values = combined.values.slice(start, end - start)
        if values.null_count != 0:
            raise ValueError("Unexpected null elements in list column")
        flat = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32)
        return flat.reshape(len(combined), expected_dim).copy()

    raise ValueError(f"Unsupported Arrow column type for vector feature: {col_type}")


def _float32_matrix_to_arrow_array(matrix: np.ndarray, arrow_type: pa.DataType) -> pa.Array:
    arr = np.ascontiguousarray(matrix, dtype=np.float32)
    num_rows, dim = arr.shape
    flat = pa.array(arr.reshape(-1), type=arrow_type.value_type)

    if pa.types.is_fixed_size_list(arrow_type):
        return pa.FixedSizeListArray.from_arrays(flat, type=arrow_type)

    if pa.types.is_list(arrow_type):
        offsets = pa.array(np.arange(0, (num_rows + 1) * dim, dim, dtype=np.int32), type=pa.int32())
        return pa.ListArray.from_arrays(offsets, flat, type=arrow_type)

    if pa.types.is_large_list(arrow_type):
        offsets = pa.array(np.arange(0, (num_rows + 1) * dim, dim, dtype=np.int64), type=pa.int64())
        return pa.LargeListArray.from_arrays(offsets, flat, type=arrow_type)

    raise ValueError(f"Unsupported Arrow column type for vector feature: {arrow_type}")


def _retarget_feature_matrix(
    matrix: np.ndarray,
    arm_indices: Mapping[str, ArmTcpIndices],
    deltas: Mapping[str, ArmTcpDelta],
) -> np.ndarray:
    if matrix.shape[0] == 0:
        return matrix.copy()
    out = matrix.copy()
    for arm_name, idx in arm_indices.items():
        delta = deltas[arm_name]
        xyz_cols = list(idx.xyz)
        rot_cols = list(idx.rot6d)
        p_new, r6d_new = _transform_tcp_poses_float64(
            matrix[:, xyz_cols],
            matrix[:, rot_cols],
            delta,
        )
        out[:, xyz_cols] = p_new.astype(np.float32)
        if not delta.is_pure_translation:
            out[:, rot_cols] = r6d_new.astype(np.float32)
    return out


def _retarget_data_table(
    table: pa.Table,
    layouts: Mapping[str, Mapping[str, ArmTcpIndices]],
    dims: Mapping[str, int],
    deltas: Mapping[str, ArmTcpDelta],
) -> tuple[pa.Table, dict[str, np.ndarray]]:
    updated = table
    matrices: dict[str, np.ndarray] = {}
    for feature_key in TARGET_FEATURES:
        pos = updated.schema.get_field_index(feature_key)
        if pos < 0:
            raise ValueError(f"Data parquet table is missing required column {feature_key!r}")
        field = updated.schema.field(pos)
        src_matrix = _arrow_column_to_float32_matrix(updated[feature_key], dims[feature_key])
        dst_matrix = _retarget_feature_matrix(src_matrix, layouts[feature_key], deltas)
        new_array = _float32_matrix_to_arrow_array(dst_matrix, field.type)
        updated = updated.set_column(pos, field, new_array)
        matrices[feature_key] = dst_matrix
    return updated, matrices


def _collect_episode_feature_arrays(
    episode_ids: np.ndarray,
    matrices: Mapping[str, np.ndarray],
    buffers: dict[int, dict[str, list[np.ndarray]]],
) -> None:
    for raw_ep in np.unique(episode_ids):
        ep = int(raw_ep)
        mask = episode_ids == raw_ep
        ep_buf = buffers[ep]
        for feature_key, matrix in matrices.items():
            ep_buf[feature_key].append(matrix[mask])


def _compute_episode_feature_stats(
    buffers: Mapping[int, Mapping[str, Sequence[np.ndarray]]],
    episode_lengths: Mapping[int, int],
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for ep in sorted(episode_lengths):
        ep_buf = buffers.get(ep)
        if not ep_buf:
            raise ValueError(f"Episode {ep} has no rows in data/*/*.parquet")
        per_feat: dict[str, dict[str, np.ndarray]] = {}
        for feature_key in TARGET_FEATURES:
            chunks = ep_buf.get(feature_key)
            if not chunks:
                raise ValueError(f"Episode {ep} has no {feature_key!r} rows in data/*/*.parquet")
            arr = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
            if len(arr) != episode_lengths[ep]:
                raise ValueError(
                    f"Episode {ep}: data row count {len(arr)} != metadata length "
                    f"{episode_lengths[ep]}"
                )
            per_feat[feature_key] = get_feature_stats(arr, axis=0, keepdims=False)
        episode_stats[ep] = per_feat
    return episode_stats


def _update_episodes_table(
    table: pa.Table,
    episode_stats: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
) -> pa.Table:
    ep_ids = [int(ep) for ep in table["episode_index"].to_pylist()]
    updated = table
    for pos, field in enumerate(updated.schema):
        parsed = _is_target_stat_column(field.name)
        if parsed is None:
            continue
        matched_feat, stat_name = parsed
        col_values: list[Any] = []
        for ep in ep_ids:
            if ep not in episode_stats:
                raise ValueError(f"Episode {ep} missing from recomputed stats")
            feat_stats = episode_stats[ep][matched_feat]
            if stat_name not in feat_stats:
                raise ValueError(f"Stat {stat_name!r} missing from recomputed {matched_feat} stats")
            col_values.append(np.asarray(feat_stats[stat_name]).tolist())
        updated = updated.set_column(pos, field, pa.array(col_values, type=field.type))
    return updated


def _aggregate_target_stats(
    episode_stats: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
) -> dict[str, dict[str, Any]]:
    per_episode_dicts = [
        {feat: dict(feat_stats) for feat, feat_stats in ep_stats.items()}
        for _, ep_stats in sorted(episode_stats.items())
    ]
    aggregated = aggregate_stats(per_episode_dicts)
    return {feat: serialize_dict(dict(aggregated[feat])) for feat in TARGET_FEATURES}


def retarget_tcp_dataset(
    src: Path,
    dst: Path,
    old_tool: Path,
    new_tool: Path,
) -> RetargetTcpResult:
    src_root = src.expanduser().resolve()
    dst_root = dst.expanduser()

    info, data_files, episode_files, episode_lengths = _validate_source_dataset(src_root)
    layouts, dims, ordered_arms = _discover_feature_layouts(info)

    old_snapshot = load_tool_snapshot(old_tool)
    new_snapshot = load_tool_snapshot(new_tool)
    deltas = compute_arm_deltas(old_snapshot, new_snapshot, required_arms=ordered_arms)

    if dst_root.exists():
        raise FileExistsError(f"{dst_root} already exists; refusing to overwrite")

    dst_created = False
    try:
        dst_root.mkdir(parents=True, exist_ok=False)
        dst_created = True

        buffers: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
        total_frames = 0

        for path in data_files:
            src_table = pq.read_table(path)
            if "episode_index" not in src_table.column_names:
                raise ValueError(f"{path}: missing required column 'episode_index'")
            retargeted_table, matrices = _retarget_data_table(src_table, layouts, dims, deltas)
            out_path = dst_root / path.relative_to(src_root)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(retargeted_table, out_path)
            _collect_episode_feature_arrays(
                retargeted_table["episode_index"].to_numpy(),
                matrices,
                buffers,
            )
            total_frames += retargeted_table.num_rows

        episode_stats = _compute_episode_feature_stats(buffers, episode_lengths)

        for path in episode_files:
            ep_table = pq.read_table(path)
            updated_ep = _update_episodes_table(ep_table, episode_stats)
            out_path = dst_root / path.relative_to(src_root)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(updated_ep, out_path)

        stats_src_path = src_root / "meta" / "stats.json"
        dataset_stats = read_json(stats_src_path, stats_src_path.read_bytes())
        agg_target = _aggregate_target_stats(episode_stats)
        for feat in TARGET_FEATURES:
            dataset_stats[feat] = agg_target[feat]
        write_json(dataset_stats, dst_root / "meta" / "stats.json")

        meta_dst = dst_root / "meta"
        meta_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_root / "meta" / "tasks.parquet", meta_dst / "tasks.parquet")
        shutil.copy2(src_root / "meta" / "info.json", meta_dst / "info.json")

        videos_src = src_root / "videos"
        if videos_src.exists():
            shutil.copytree(videos_src, dst_root / "videos", copy_function=os.link)
    except Exception:
        if dst_created and dst_root.exists():
            shutil.rmtree(dst_root, ignore_errors=True)
        raise

    return RetargetTcpResult(
        dst=dst_root,
        total_episodes=len(episode_lengths),
        total_frames=total_frames,
        arms=ordered_arms,
        data_files_written=len(data_files),
        episode_files_written=len(episode_files),
    )


def _check_orthonormality(rot6d: np.ndarray) -> float:
    if rot6d.shape[0] == 0:
        return 0.0
    a1 = rot6d[:, 0:3].astype(np.float64)
    a2 = rot6d[:, 3:6].astype(np.float64)
    norm1_err = float(np.max(np.abs(np.linalg.norm(a1, axis=-1) - 1.0)))
    norm2_err = float(np.max(np.abs(np.linalg.norm(a2, axis=-1) - 1.0)))
    dot_err = float(np.max(np.abs(np.sum(a1 * a2, axis=-1))))
    return max(norm1_err, norm2_err, dot_err)


def _check_data_files_for_retarget(
    src_root: Path,
    dst_root: Path,
    layouts: Mapping[str, Mapping[str, ArmTcpIndices]],
    dims: Mapping[str, int],
    forward_deltas: Mapping[str, ArmTcpDelta],
    inverse_deltas: Mapping[str, ArmTcpDelta],
    *,
    pos_atol: float,
    rot_atol: float,
) -> tuple[
    list[str],
    dict[int, dict[str, list[np.ndarray]]],
    int,
    float,
    float,
    float,
    float,
]:
    errors: list[str] = []
    buffers: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    total_frames = 0
    max_pos_err = 0.0
    max_rot_err = 0.0
    max_rt_pos_err = 0.0
    max_rt_rot_err = 0.0

    src_files = sorted(p.relative_to(src_root) for p in (src_root / "data").glob("*/*.parquet"))
    dst_files = sorted(p.relative_to(dst_root) for p in (dst_root / "data").glob("*/*.parquet"))
    if src_files != dst_files:
        return (
            ["data parquet file lists differ between src and dst"],
            buffers,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    if not dst_files:
        return ["no data parquet files found in dst"], buffers, 0, 0.0, 0.0, 0.0, 0.0

    for rel in src_files:
        src_table = pq.read_table(src_root / rel)
        dst_table = pq.read_table(dst_root / rel)
        total_frames += dst_table.num_rows

        if not src_table.schema.equals(dst_table.schema, check_metadata=True):
            errors.append(f"{rel}: parquet schema or metadata differs between src and dst")

        if src_table.num_rows != dst_table.num_rows or src_table.column_names != (
            dst_table.column_names
        ):
            errors.append(
                f"{rel}: shape/columns mismatch ({dst_table.num_rows} rows / "
                f"{dst_table.column_names} vs {src_table.num_rows} / {src_table.column_names})"
            )
            continue

        for col in src_table.column_names:
            if col not in TARGET_FEATURES and not src_table[col].equals(dst_table[col]):
                errors.append(f"{rel}: non-TCP data column {col!r} differs")

        dst_matrices: dict[str, np.ndarray] = {}
        file_ok_for_stats = True
        for feature_key in TARGET_FEATURES:
            try:
                src_mat = _arrow_column_to_float32_matrix(src_table[feature_key], dims[feature_key])
                dst_mat = _arrow_column_to_float32_matrix(dst_table[feature_key], dims[feature_key])
            except ValueError as exc:
                errors.append(f"{rel}: {feature_key}: {exc}")
                file_ok_for_stats = False
                continue

            dst_matrices[feature_key] = dst_mat
            if dst_mat.shape[0] == 0:
                continue

            if not np.all(np.isfinite(dst_mat)):
                errors.append(f"{rel}: {feature_key} contains non-finite values")
                continue

            arm_map = layouts[feature_key]
            tcp_cols = {c for idx in arm_map.values() for c in (*idx.xyz, *idx.rot6d)}
            non_tcp_cols = [c for c in range(dims[feature_key]) if c not in tcp_cols]
            if non_tcp_cols:
                src_other = np.ascontiguousarray(src_mat[:, non_tcp_cols])
                dst_other = np.ascontiguousarray(dst_mat[:, non_tcp_cols])
                if not np.array_equal(src_other.view(np.uint32), dst_other.view(np.uint32)):
                    errors.append(
                        f"{rel}: {feature_key} non-TCP dimensions {non_tcp_cols} are not "
                        "bitwise identical"
                    )

            for arm_name, idx in arm_map.items():
                xyz_cols = list(idx.xyz)
                rot_cols = list(idx.rot6d)
                fwd_delta = forward_deltas[arm_name]
                inv_delta = inverse_deltas[arm_name]

                src_xyz = src_mat[:, xyz_cols]
                src_rot = src_mat[:, rot_cols]
                dst_xyz = dst_mat[:, xyz_cols]
                dst_rot = dst_mat[:, rot_cols]

                exp_xyz, exp_rot = _transform_tcp_poses_float64(src_xyz, src_rot, fwd_delta)
                pos_err = float(np.max(np.abs(dst_xyz.astype(np.float64) - exp_xyz)))
                rot_err = float(np.max(np.abs(dst_rot.astype(np.float64) - exp_rot)))
                max_pos_err = max(max_pos_err, pos_err)
                max_rot_err = max(max_rot_err, rot_err)

                if pos_err > pos_atol:
                    errors.append(
                        f"{rel}: {feature_key} arm {arm_name!r} forward position error "
                        f"{pos_err:.3e} exceeds {pos_atol:.3e}"
                    )
                if rot_err > rot_atol:
                    errors.append(
                        f"{rel}: {feature_key} arm {arm_name!r} forward rotation error "
                        f"{rot_err:.3e} exceeds {rot_atol:.3e}"
                    )
                if fwd_delta.is_pure_translation:
                    src_rot_bits = np.ascontiguousarray(src_rot).view(np.uint32)
                    dst_rot_bits = np.ascontiguousarray(dst_rot).view(np.uint32)
                    if not np.array_equal(src_rot_bits, dst_rot_bits):
                        errors.append(
                            f"{rel}: {feature_key} arm {arm_name!r} rot6d was not preserved "
                            "bitwise under pure translation"
                        )

                ortho_err = _check_orthonormality(dst_rot)
                if ortho_err >= ORTHONORMAL_ATOL:
                    errors.append(
                        f"{rel}: {feature_key} arm {arm_name!r} rot6d orthonormality error "
                        f"{ortho_err:.3e} >= {ORTHONORMAL_ATOL:.3e}"
                    )

                rt_xyz, rt_rot = _transform_tcp_poses_float64(dst_xyz, dst_rot, inv_delta)
                rt_pos_err = float(np.max(np.abs(rt_xyz - src_xyz.astype(np.float64))))
                rt_rot_err = float(np.max(np.abs(rt_rot - src_rot.astype(np.float64))))
                max_rt_pos_err = max(max_rt_pos_err, rt_pos_err)
                max_rt_rot_err = max(max_rt_rot_err, rt_rot_err)

                if rt_pos_err > pos_atol:
                    errors.append(
                        f"{rel}: {feature_key} arm {arm_name!r} inverse round-trip position "
                        f"error {rt_pos_err:.3e} exceeds {pos_atol:.3e}"
                    )
                if rt_rot_err > rot_atol:
                    errors.append(
                        f"{rel}: {feature_key} arm {arm_name!r} inverse round-trip rotation "
                        f"error {rt_rot_err:.3e} exceeds {rot_atol:.3e}"
                    )

        if file_ok_for_stats and "episode_index" in dst_table.column_names:
            _collect_episode_feature_arrays(
                dst_table["episode_index"].to_numpy(),
                dst_matrices,
                buffers,
            )

    return (
        errors,
        buffers,
        total_frames,
        max_pos_err,
        max_rot_err,
        max_rt_pos_err,
        max_rt_rot_err,
    )


def _is_target_stat_column(col_name: str) -> tuple[str, str] | None:
    for feat in TARGET_FEATURES:
        prefix = f"stats/{feat}/"
        if col_name.startswith(prefix):
            return feat, col_name[len(prefix) :]
    return None


def _check_episodes_files_for_retarget(
    src_root: Path,
    dst_root: Path,
    expected_ep_stats: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
) -> list[str]:
    errors: list[str] = []
    src_files = sorted(
        p.relative_to(src_root) for p in (src_root / "meta" / "episodes").glob("*/*.parquet")
    )
    dst_files = sorted(
        p.relative_to(dst_root) for p in (dst_root / "meta" / "episodes").glob("*/*.parquet")
    )
    if src_files != dst_files:
        return ["meta/episodes parquet file lists differ between src and dst"]

    for rel in src_files:
        src_table = pq.read_table(src_root / rel)
        dst_table = pq.read_table(dst_root / rel)

        if not src_table.schema.equals(dst_table.schema, check_metadata=True):
            errors.append(f"{rel}: episodes table schema or metadata differs")

        if src_table.num_rows != dst_table.num_rows or src_table.column_names != (
            dst_table.column_names
        ):
            errors.append(f"{rel}: episodes table shape/columns differ")
            continue

        for col in src_table.column_names:
            if _is_target_stat_column(col) is None and not src_table[col].equals(dst_table[col]):
                errors.append(f"{rel}: episode column {col!r} differs")

        rows = dst_table.to_pylist()
        for row in rows:
            ep = int(row["episode_index"])
            if ep not in expected_ep_stats:
                continue
            for col in dst_table.column_names:
                parsed = _is_target_stat_column(col)
                if parsed is None:
                    continue
                feat, stat_name = parsed
                exp_val = expected_ep_stats[ep][feat].get(stat_name)
                if exp_val is None:
                    errors.append(f"episode {ep}: unexpected stat column {col!r}")
                    continue
                actual_arr = np.asarray(row[col], dtype=np.float64)
                expected_arr = np.asarray(exp_val, dtype=np.float64)
                if actual_arr.shape != expected_arr.shape or not np.allclose(
                    actual_arr,
                    expected_arr,
                    rtol=STATS_RTOL,
                    atol=STATS_ATOL,
                    equal_nan=True,
                ):
                    errors.append(f"episode {ep}: {col} does not match retargeted data")

    return errors


def _check_meta_files_for_retarget(
    src_root: Path,
    dst_root: Path,
    expected_agg_stats: Mapping[str, Mapping[str, Any]] | None,
) -> list[str]:
    errors: list[str] = []

    info_src_path = src_root / "meta" / "info.json"
    info_dst_path = dst_root / "meta" / "info.json"
    if not info_dst_path.is_file():
        errors.append("missing meta/info.json in dst")
    else:
        info_src = read_json(info_src_path, info_src_path.read_bytes())
        info_dst = read_json(info_dst_path, info_dst_path.read_bytes())
        if info_src != info_dst:
            errors.append("meta/info.json differs between src and dst")

    tasks_src_path = src_root / "meta" / "tasks.parquet"
    tasks_dst_path = dst_root / "meta" / "tasks.parquet"
    if not tasks_dst_path.is_file():
        errors.append("missing meta/tasks.parquet in dst")
    else:
        tasks_src = pq.read_table(tasks_src_path)
        tasks_dst = pq.read_table(tasks_dst_path)
        if not tasks_src.schema.equals(tasks_dst.schema, check_metadata=True) or not (
            tasks_src.equals(tasks_dst)
        ):
            errors.append("meta/tasks.parquet differs between src and dst")

    stats_src_path = src_root / "meta" / "stats.json"
    stats_dst_path = dst_root / "meta" / "stats.json"
    if not stats_dst_path.is_file():
        errors.append("missing meta/stats.json in dst")
    else:
        stats_src = read_json(stats_src_path, stats_src_path.read_bytes())
        stats_dst = read_json(stats_dst_path, stats_dst_path.read_bytes())
        non_target_src = {k: v for k, v in stats_src.items() if k not in TARGET_FEATURES}
        non_target_dst = {k: v for k, v in stats_dst.items() if k not in TARGET_FEATURES}
        if non_target_src != non_target_dst:
            errors.append("meta/stats.json differs outside action and observation.state")
        if expected_agg_stats is not None:
            for feat in TARGET_FEATURES:
                actual_feat = stats_dst.get(feat)
                expected_feat = expected_agg_stats[feat]
                if not isinstance(actual_feat, Mapping):
                    errors.append(f"meta/stats.json is missing {feat!r}")
                    continue
                if set(actual_feat) != set(expected_feat):
                    errors.append(
                        f"meta/stats.json {feat} keys differ: {sorted(actual_feat)} vs "
                        f"{sorted(expected_feat)}"
                    )
                    continue
                for key, exp_val in expected_feat.items():
                    actual_arr = np.asarray(actual_feat[key], dtype=np.float64)
                    expected_arr = np.asarray(exp_val, dtype=np.float64)
                    if actual_arr.shape != expected_arr.shape or not np.allclose(
                        actual_arr,
                        expected_arr,
                        rtol=STATS_RTOL,
                        atol=STATS_ATOL,
                        equal_nan=True,
                    ):
                        errors.append(
                            f"meta/stats.json {feat}/{key} does not match aggregated data"
                        )

    return errors


def _check_video_hardlinks(src_root: Path, dst_root: Path) -> list[str]:
    videos_src = src_root / "videos"
    videos_dst = dst_root / "videos"
    if not videos_src.exists() and not videos_dst.exists():
        return []
    if videos_src.exists() != videos_dst.exists():
        return ["videos directory presence differs between src and dst"]

    src_files = sorted(p.relative_to(src_root) for p in videos_src.rglob("*") if p.is_file())
    dst_files = sorted(p.relative_to(dst_root) for p in videos_dst.rglob("*") if p.is_file())
    if src_files != dst_files:
        return ["video file lists differ between src and dst"]

    errors: list[str] = []
    for rel in src_files:
        s_stat = (src_root / rel).stat()
        d_stat = (dst_root / rel).stat()
        if s_stat.st_ino != d_stat.st_ino or s_stat.st_dev != d_stat.st_dev:
            errors.append(f"video {rel} is not a hard link to src")
    return errors


def verify_tcp_retarget(
    src: Path,
    dst: Path,
    old_tool: Path,
    new_tool: Path,
    *,
    pos_atol: float = 1e-5,
    rot_atol: float = 1e-5,
) -> VerifyTcpResult:
    src_root = src.expanduser().resolve()
    dst_root = dst.expanduser().resolve()

    info, _, _, episode_lengths = _validate_source_dataset(src_root)
    layouts, dims, ordered_arms = _discover_feature_layouts(info)
    old_snapshot = load_tool_snapshot(old_tool)
    new_snapshot = load_tool_snapshot(new_tool)
    forward_deltas = compute_arm_deltas(old_snapshot, new_snapshot, required_arms=ordered_arms)
    inverse_deltas = compute_arm_deltas(new_snapshot, old_snapshot, required_arms=ordered_arms)

    if not dst_root.exists():
        return VerifyTcpResult(
            total_episodes=len(episode_lengths),
            total_frames=0,
            arms=ordered_arms,
            max_pos_err_m=0.0,
            max_rot_err=0.0,
            max_roundtrip_pos_err_m=0.0,
            max_roundtrip_rot_err=0.0,
            errors=(f"dst dataset directory does not exist: {dst_root}",),
        )

    errors: list[str] = []
    (
        data_errors,
        buffers,
        total_frames,
        max_pos_err,
        max_rot_err,
        max_rt_pos_err,
        max_rt_rot_err,
    ) = _check_data_files_for_retarget(
        src_root,
        dst_root,
        layouts,
        dims,
        forward_deltas,
        inverse_deltas,
        pos_atol=pos_atol,
        rot_atol=rot_atol,
    )
    errors.extend(data_errors)

    expected_ep_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    expected_agg_stats: dict[str, dict[str, Any]] | None = None
    if buffers:
        try:
            expected_ep_stats = _compute_episode_feature_stats(buffers, episode_lengths)
            expected_agg_stats = _aggregate_target_stats(expected_ep_stats)
        except ValueError as exc:
            errors.append(str(exc))

    errors.extend(_check_episodes_files_for_retarget(src_root, dst_root, expected_ep_stats))
    errors.extend(_check_meta_files_for_retarget(src_root, dst_root, expected_agg_stats))
    errors.extend(_check_video_hardlinks(src_root, dst_root))

    return VerifyTcpResult(
        total_episodes=len(episode_lengths),
        total_frames=total_frames,
        arms=ordered_arms,
        max_pos_err_m=max_pos_err,
        max_rot_err=max_rot_err,
        max_roundtrip_pos_err_m=max_rt_pos_err,
        max_roundtrip_rot_err=max_rt_rot_err,
        errors=tuple(errors),
    )


def retarget_calib_payload(
    payload: Mapping[str, Any],
    old_tool: Path,
    new_tool: Path,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    frames = result.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Calibration payload must contain a 'frames' list")

    required_arms: list[str] = []
    for idx, entry in enumerate(frames):
        if not isinstance(entry, dict):
            raise ValueError(f"frames[{idx}] must be an object")
        arm = entry.get("arm")
        if not isinstance(arm, str) or not arm:
            raise ValueError(f"frames[{idx}].arm must be a non-empty string")
        if arm not in required_arms:
            required_arms.append(arm)

    old_snapshot = load_tool_snapshot(old_tool)
    new_snapshot = load_tool_snapshot(new_tool)
    deltas = compute_arm_deltas(
        old_snapshot,
        new_snapshot,
        required_arms=required_arms if required_arms else None,
    )

    for idx, entry in enumerate(frames):
        arm = str(entry["arm"])
        tcp = _parse_finite_float_vector(
            entry.get("tcp"),
            expected_len=9,
            context=f"frames[{idx}].tcp",
        )
        delta = deltas[arm]
        p_new, r6d_new = _transform_tcp_poses_float64(tcp[0:3], tcp[3:9], delta)
        if delta.is_pure_translation:
            rot_out = [float(v) for v in entry["tcp"][3:9]]
        else:
            rot_out = [float(v) for v in r6d_new]
        entry["tcp"] = [float(p_new[0]), float(p_new[1]), float(p_new[2]), *rot_out]

    if repo_id is not None:
        result["repo_id"] = repo_id
    if revision is not None:
        result["revision"] = revision
    return result
