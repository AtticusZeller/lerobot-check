"""LeRobot v3.0 dataset metadata, state loading, and 6D rotation utilities.

6D Rotation Convention (confirmed from ``lerobot-xense``):
    Confirmed in ``lerobot-xense/src/lerobot/utils/robot_utils.py``
    (``quaternion_to_rotation_6d`` and ``rotation_6d_to_quaternion``) and used by
    ``BiEliteCS66RT._tcp_rotvec_to_feature_values`` in
    ``lerobot-xense/src/lerobot/robots/bi_elite_cs66_rt/bi_elite_cs66_rt.py``.

    Each arm's 6D rotation ``[r1, r2, r3, r4, r5, r6]`` stores the first two
    columns of the 3x3 rotation matrix ``R`` in the arm's gravity-aligned base
    frame (Zhou et al., 2019):

    - ``a1 = [r1, r2, r3]`` is column 0 of ``R`` (TCP x-axis).
    - ``a2 = [r4, r5, r6]`` is column 1 of ``R`` (TCP y-axis).

    The full right-handed rotation matrix ``R = [b1, b2, b3]`` is reconstructed
    via Gram-Schmidt orthogonalization:
    ``b1 = a1 / ||a1||``,
    ``b2 = normalize(a2 - (b1 . a2) * b1)``,
    ``b3 = cross(b1, b2)``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

SUPPORTED_CODEBASE_VERSION = "v3.0"


@dataclass(frozen=True)
class ArmFeatureIndices:
    """Column indices inside ``observation.state`` for one robot arm.

    Attributes:
        arm: Arm label (``"left"``, ``"right"``, or ``"default"`` for single-arm).
        xyz: Indices of ``(tcp.x, tcp.y, tcp.z)``.
        rot6d: Indices of ``(tcp.r1, ..., tcp.r6)``.
        gripper: Index of ``gripper.pos``.
    """

    arm: str
    xyz: tuple[int, int, int]
    rot6d: tuple[int, int, int, int, int, int]
    gripper: int


@dataclass(frozen=True)
class ArmState:
    """Proprioceptive trajectory for one arm across an episode.

    Attributes:
        arm: Arm identifier (e.g. ``"left"`` or ``"right"``).
        tcp_xyz: TCP positions of shape ``(N, 3)`` in meters.
        tcp_rot6d: 6D TCP orientations of shape ``(N, 6)``, storing the first two
            columns of the 3x3 rotation matrix (see module docstring).
        gripper: Observed gripper position trajectory of shape ``(N,)`` where
            ``1.0`` is open and values toward ``0.0`` are closed on Xense/Elite rigs.
        action_gripper: Optional commanded gripper action trajectory of shape ``(N,)``
            from the dataset's ``action`` feature, used to detect active squeezes
            when ``action_gripper - gripper <= -gap``.
    """

    arm: str
    tcp_xyz: np.ndarray
    tcp_rot6d: np.ndarray
    gripper: np.ndarray
    action_gripper: np.ndarray | None = None


@dataclass(frozen=True)
class VideoSpan:
    """Location and timestamp bounds of one episode inside a camera MP4 file.

    Attributes:
        path: Path to the MP4 file on disk.
        from_timestamp: Episode start timestamp inside the MP4 file (seconds).
        to_timestamp: Episode end timestamp inside the MP4 file (seconds).
    """

    path: Path
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class EpisodeRecord:
    """Metadata needed to load one episode's state and videos without scanning all files.

    Attributes:
        episode_index: Zero-based episode index.
        length: Number of frames in the episode.
        data_path: Path to the parquet file containing this episode's rows.
        videos: Mapping from camera feature key to its ``VideoSpan``.
    """

    episode_index: int
    length: int
    data_path: Path
    videos: dict[str, VideoSpan]


def rotation_6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """Reconstruct 3x3 rotation matrices from 6D rotation vectors via Gram-Schmidt.

    Follows the ``lerobot-xense`` convention (Zhou et al., 2019) where
    ``r6d[..., 0:3]`` is the first column of ``R`` and ``r6d[..., 3:6]`` is the
    second column of ``R``.

    Args:
        r6d: Array of shape ``(..., 6)`` containing ``[r1, r2, r3, r4, r5, r6]``.

    Returns:
        Rotation matrix array of shape ``(..., 3, 3)`` with columns ``[b1, b2, b3]``.

    Raises:
        ValueError: If the last dimension is not 6 or if a vector is degenerate.
    """
    arr = np.asarray(r6d, dtype=np.float64)
    if arr.ndim < 1 or arr.shape[-1] != 6:
        raise ValueError(f"Expected array with last dimension 6, got shape {arr.shape}")

    a1 = arr[..., 0:3]
    a2 = arr[..., 3:6]

    norm1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    if np.any(norm1 <= 1e-12):
        raise ValueError("Degenerate 6D rotation: first column vector has zero norm")
    b1 = a1 / norm1

    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    u2 = a2 - proj
    norm2 = np.linalg.norm(u2, axis=-1, keepdims=True)
    if np.any(norm2 <= 1e-12):
        raise ValueError("Degenerate 6D rotation: second column is collinear with first")
    b2 = u2 / norm2

    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def default_lerobot_cache_root() -> Path:
    """Return the local LeRobot dataset cache root directory."""
    env_home = os.environ.get("HF_LEROBOT_HOME")
    if env_home:
        return Path(env_home).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "lerobot"
    return Path.home() / ".cache" / "huggingface" / "lerobot"


def resolve_dataset_root(
    dataset_root: Path | None,
    repo_id: str | None = None,
) -> Path:
    """Resolve a local LeRobot dataset root from explicit path or cached repo ID.

    Args:
        dataset_root: Explicit local directory path, if provided.
        repo_id: Hugging Face dataset repo ID used to locate the local cache when
            ``dataset_root`` is omitted.

    Returns:
        Resolved local dataset root path.

    Raises:
        ValueError: If neither ``dataset_root`` nor ``repo_id`` is provided.
        FileNotFoundError: If the resolved directory does not exist.
    """
    if dataset_root is not None:
        resolved = dataset_root.expanduser().resolve()
    elif repo_id:
        resolved = (default_lerobot_cache_root() / repo_id).resolve()
    else:
        raise ValueError("Provide at least one of --dataset-root or --repo-id")

    if not resolved.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {resolved}")
    return resolved


def detect_dataset_revision(dataset_root: Path) -> str:
    """Detect a cached Hugging Face commit hash from local metadata, if available.

    Args:
        dataset_root: Local LeRobot dataset directory.

    Returns:
        Commit SHA when a unique tree snapshot exists under ``.cache/huggingface/trees``,
        otherwise ``"local"``.
    """
    trees_dir = dataset_root / ".cache" / "huggingface" / "trees"
    if trees_dir.is_dir():
        stems = sorted(p.stem for p in trees_dir.glob("*.json") if len(p.stem) == 40)
        if len(stems) == 1:
            return stems[0]
    return "local"


def infer_repo_id(dataset_root: Path) -> str:
    """Infer ``<org>/<name>`` from a local dataset path when ``--repo-id`` is omitted."""
    parts = dataset_root.parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return dataset_root.name


def parse_arm_feature_indices(state_names: Sequence[str]) -> dict[str, ArmFeatureIndices]:
    """Extract per-arm TCP and gripper indices from ``observation.state`` feature names.

    Supports bimanual prefixes (``left_*``, ``right_*``) and single-arm unprefixed
    names (``tcp.x``, ..., ``gripper.pos``).

    Args:
        state_names: Ordered feature names from ``meta/info.json``.

    Returns:
        Mapping from arm name to ``ArmFeatureIndices``.

    Raises:
        ValueError: If no recognized arm TCP + gripper schema is present.
    """
    name_to_idx = {name: idx for idx, name in enumerate(state_names)}
    arms: dict[str, ArmFeatureIndices] = {}

    for prefix, arm_name in (("left_", "left"), ("right_", "right"), ("", "default")):
        xyz_keys = (f"{prefix}tcp.x", f"{prefix}tcp.y", f"{prefix}tcp.z")
        rot_keys = tuple(f"{prefix}tcp.r{i}" for i in range(1, 7))
        grip_key = f"{prefix}gripper.pos"
        if all(k in name_to_idx for k in (*xyz_keys, *rot_keys, grip_key)):
            arms[arm_name] = ArmFeatureIndices(
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
                gripper=name_to_idx[grip_key],
            )
            if arm_name == "default":
                break

    if not arms:
        raise ValueError(
            "Could not locate tcp.{x,y,z,r1..r6} and gripper.pos features in "
            f"observation.state names: {list(state_names)}"
        )
    return arms


def parse_action_gripper_indices(action_names: Sequence[str]) -> dict[str, int]:
    """Extract per-arm gripper indices from ``action`` feature names.

    Args:
        action_names: Ordered feature names from ``meta/info.json`` for ``action``.

    Returns:
        Mapping from arm name (``"left"``, ``"right"``, or ``"default"``) to the
        column index of ``gripper.pos`` inside ``action``.
    """
    name_to_idx = {name: idx for idx, name in enumerate(action_names)}
    indices: dict[str, int] = {}
    for prefix, arm_name in (("left_", "left"), ("right_", "right"), ("", "default")):
        grip_key = f"{prefix}gripper.pos"
        if grip_key in name_to_idx:
            indices[arm_name] = name_to_idx[grip_key]
            if arm_name == "default":
                break
    return indices


def _build_episode_records(
    root: Path,
    meta: LeRobotDatasetMetadata,
) -> dict[int, EpisodeRecord]:
    data_template: str = meta.info["data_path"]
    video_template: str = meta.info["video_path"]
    camera_keys = tuple(meta.camera_keys)

    records: dict[int, EpisodeRecord] = {}
    if meta.episodes is None:
        raise ValueError(f"Dataset metadata at {root} does not contain episodes table")

    for row in meta.episodes:
        ep_idx = int(row["episode_index"])
        length = int(row["length"])
        data_rel = data_template.format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        videos: dict[str, VideoSpan] = {}
        for cam_key in camera_keys:
            chunk_col = f"videos/{cam_key}/chunk_index"
            file_col = f"videos/{cam_key}/file_index"
            from_col = f"videos/{cam_key}/from_timestamp"
            to_col = f"videos/{cam_key}/to_timestamp"
            if chunk_col in row and file_col in row:
                video_rel = video_template.format(
                    video_key=cam_key,
                    chunk_index=int(row[chunk_col]),
                    file_index=int(row[file_col]),
                )
                videos[cam_key] = VideoSpan(
                    path=root / video_rel,
                    from_timestamp=float(row[from_col]),
                    to_timestamp=float(row[to_col]),
                )
        records[ep_idx] = EpisodeRecord(
            episode_index=ep_idx,
            length=length,
            data_path=root / data_rel,
            videos=videos,
        )
    return records


class DatasetView:
    """Lightweight read-only handle over a local LeRobot v3.0 dataset.

    Wraps ``LeRobotDatasetMetadata`` so episode lengths, per-episode parquet files,
    and per-camera video spans can be read directly with low memory footprint.
    """

    def __init__(
        self,
        dataset_root: Path,
        *,
        repo_id: str | None = None,
        revision: str | None = None,
    ) -> None:
        """Open a local LeRobot v3.0 dataset without downloading remote files.

        Args:
            dataset_root: Local directory containing ``meta/info.json``.
            repo_id: Optional repository ID; inferred from ``dataset_root`` if omitted.
            revision: Optional revision string; detected from local HF cache metadata
                if omitted.

        Raises:
            ValueError: If the dataset is not LeRobot v3.0.
        """
        self.root = dataset_root.expanduser().resolve()
        self.repo_id = repo_id or infer_repo_id(self.root)
        self.revision = revision or detect_dataset_revision(self.root)
        self.meta = LeRobotDatasetMetadata(
            repo_id=self.repo_id,
            root=self.root,
            force_cache_sync=False,
        )
        codebase_version = self.meta.info.get("codebase_version")
        if codebase_version != SUPPORTED_CODEBASE_VERSION:
            raise ValueError(
                f"Expected LeRobot {SUPPORTED_CODEBASE_VERSION} dataset at {self.root}, "
                f"got {codebase_version!r}"
            )
        self.fps: float = float(self.meta.fps)
        self.total_episodes: int = int(self.meta.total_episodes)
        self.camera_keys: tuple[str, ...] = tuple(self.meta.camera_keys)
        self._episodes: dict[int, EpisodeRecord] = _build_episode_records(self.root, self.meta)

        state_feature: dict[str, Any] = self.meta.features["observation.state"]
        raw_names = state_feature.get("names")
        if not isinstance(raw_names, list):
            raise ValueError("Dataset observation.state feature has no column names")
        self.state_names: tuple[str, ...] = tuple(str(name) for name in raw_names)
        self.arm_indices: dict[str, ArmFeatureIndices] = parse_arm_feature_indices(self.state_names)

        self.action_gripper_indices: dict[str, int] = {}
        action_feature = self.meta.features.get("action")
        if isinstance(action_feature, dict):
            raw_action_names = action_feature.get("names")
            if isinstance(raw_action_names, list):
                self.action_gripper_indices = parse_action_gripper_indices(
                    [str(name) for name in raw_action_names]
                )

    @property
    def episode_lengths(self) -> dict[int, int]:
        """Return ``{episode_index: length}`` for all episodes in metadata."""
        return {ep_idx: rec.length for ep_idx, rec in self._episodes.items()}

    def get_episode(self, episode_index: int) -> EpisodeRecord:
        """Return metadata record for ``episode_index``.

        Raises:
            KeyError: If ``episode_index`` is not present in dataset metadata.
        """
        if episode_index not in self._episodes:
            raise KeyError(
                f"Episode {episode_index} not found (valid range 0..{self.total_episodes - 1})"
            )
        return self._episodes[episode_index]

    def load_episode_state_matrix(self, episode_index: int) -> np.ndarray:
        """Load ``observation.state`` rows for one episode sorted by ``frame_index``.

        Args:
            episode_index: Zero-based episode index.

        Returns:
            Float64 array of shape ``(length, state_dim)``.
        """
        record = self.get_episode(episode_index)
        ep_table = pq.read_table(
            record.data_path,
            columns=["episode_index", "frame_index", "observation.state"],
            filters=[("episode_index", "==", episode_index)],
        )
        frame_indices = ep_table["frame_index"].to_numpy()
        order = np.argsort(frame_indices)
        states = np.stack(ep_table["observation.state"].to_numpy()).astype(np.float64)[order]
        if len(states) != record.length:
            raise ValueError(
                f"Episode {episode_index}: metadata length {record.length} != "
                f"parquet row count {len(states)}"
            )
        return states

    def load_episode_arm_states(self, episode_index: int) -> list[ArmState]:
        """Load structured per-arm TCP and gripper trajectories for one episode.

        Includes ``action_gripper`` whenever the dataset defines matching
        ``action`` gripper columns.

        Args:
            episode_index: Zero-based episode index.

        Returns:
            List of ``ArmState`` objects (ordered by ``arm_indices``).
        """
        record = self.get_episode(episode_index)
        columns = ["episode_index", "frame_index", "observation.state"]
        has_action = bool(self.action_gripper_indices)
        if has_action:
            columns.append("action")

        ep_table = pq.read_table(
            record.data_path,
            columns=columns,
            filters=[("episode_index", "==", episode_index)],
        )
        frame_indices = ep_table["frame_index"].to_numpy()
        order = np.argsort(frame_indices)
        states = np.stack(ep_table["observation.state"].to_numpy()).astype(np.float64)[order]
        if len(states) != record.length:
            raise ValueError(
                f"Episode {episode_index}: metadata length {record.length} != "
                f"parquet row count {len(states)}"
            )

        actions: np.ndarray | None = None
        if has_action:
            actions = np.stack(ep_table["action"].to_numpy()).astype(np.float64)[order]

        result: list[ArmState] = []
        for arm_name, idx in self.arm_indices.items():
            action_grip: np.ndarray | None = None
            if actions is not None and arm_name in self.action_gripper_indices:
                action_grip = actions[:, self.action_gripper_indices[arm_name]]
            result.append(
                ArmState(
                    arm=arm_name,
                    tcp_xyz=states[:, list(idx.xyz)],
                    tcp_rot6d=states[:, list(idx.rot6d)],
                    gripper=states[:, idx.gripper],
                    action_gripper=action_grip,
                )
            )
        return result
