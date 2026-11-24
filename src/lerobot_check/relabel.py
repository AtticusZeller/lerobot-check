"""N-subtask per-frame prompt relabeling and verification for LeRobot v3.0 datasets."""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.io_utils import serialize_dict, write_info, write_json, write_tasks

from lerobot_check.dataset import SUPPORTED_CODEBASE_VERSION
from lerobot_check.schema import AnnotationRow, TaskSpec
from lerobot_check.validate import ValidationViolation, validate_annotations


class RelabelValidationError(ValueError):
    """Raised when an annotation table fails validation before relabeling writes output.

    Attributes:
        violations: Ordered tuple of contract violations that prevented relabeling.
    """

    def __init__(self, violations: Sequence[ValidationViolation]) -> None:
        self.violations: tuple[ValidationViolation, ...] = tuple(violations)
        summary = "\n".join(str(v) for v in self.violations)
        super().__init__(
            f"Annotation validation failed with {len(self.violations)} violation(s):\n{summary}"
        )


@dataclass(frozen=True)
class RelabelResult:
    """Summary of a completed relabeling run.

    Attributes:
        dst: Destination dataset root written to disk.
        total_episodes: Number of episodes relabeled.
        total_frames: Total frames written across all data parquet files.
        tasks: Ordered tuple of task prompt strings (row position equals ``task_index``).
        task_counts: Per-``task_index`` frame counts aligned with ``tasks``.
    """

    dst: Path
    total_episodes: int
    total_frames: int
    tasks: tuple[str, ...]
    task_counts: tuple[int, ...]


@dataclass(frozen=True)
class VerifyResult:
    """Result of verifying a relabeled dataset against its source and annotations.

    Attributes:
        total_frames: Total frames inspected in ``dst`` data parquet files.
        tasks: Expected ordered task prompts from the task specification.
        task_counts: Per-``task_index`` frame counts observed in ``dst``.
        stats_mean: ``mean`` list from ``dst/meta/stats.json["task_index"]``.
        stats_count: ``count`` list from ``dst/meta/stats.json["task_index"]``.
        data_mean: Empirical mean of ``task_index`` across all ``dst`` data rows.
        errors: Tuple of verification error messages; empty means ``OK``.
    """

    total_frames: int
    tasks: tuple[str, ...]
    task_counts: tuple[int, ...]
    stats_mean: list[float]
    stats_count: list[int]
    data_mean: float
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return ``True`` when verification found zero errors."""
        return not self.errors


def _load_info(root: Path) -> dict[str, Any]:
    """Read and check ``meta/info.json`` for a LeRobot v3.0 dataset."""
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata file: {info_path}")
    info: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))
    version = info.get("codebase_version")
    if version != SUPPORTED_CODEBASE_VERSION:
        raise ValueError(
            f"Expected LeRobot {SUPPORTED_CODEBASE_VERSION} dataset at {root}, got {version!r}"
        )
    return info


def load_episode_lengths(root: Path) -> dict[int, int]:
    """Load ``{episode_index: length}`` from ``meta/episodes/*/*.parquet``.

    Args:
        root: Local LeRobot v3.0 dataset root directory.

    Returns:
        Mapping from zero-based ``episode_index`` to frame count ``length``.

    Raises:
        FileNotFoundError: If no episode metadata parquet files are found.
        ValueError: If duplicate episode indices exist.
    """
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
    return lengths


def _resolve_task_prompts(task_spec: TaskSpec) -> tuple[tuple[str, ...], dict[str, int]]:
    """Return ordered unique prompts and ``subtask_name -> task_index`` mapping.

    The position of each prompt in the returned tuple equals its ``task_index`` so
    that ``tasks.parquet`` row order matches ``task_index`` for both positional
    ``iloc`` lookups in LeRobot and column lookups in OpenPI.
    """
    prompts: list[str] = []
    seen_prompts: set[str] = set()
    subtask_to_task_index: dict[str, int] = {}
    for idx, subtask in enumerate(task_spec.subtasks):
        if subtask.prompt in seen_prompts:
            raise ValueError(
                f"Task spec contains duplicate prompt string {subtask.prompt!r}; "
                "each subtask must have a unique prompt for tasks.parquet."
            )
        seen_prompts.add(subtask.prompt)
        prompts.append(subtask.prompt)
        subtask_to_task_index[subtask.name] = idx
    return tuple(prompts), subtask_to_task_index


def validate_annotations_for_relabel(
    rows: Sequence[AnnotationRow],
    task_spec: TaskSpec,
    episode_lengths: Mapping[int, int],
    *,
    expected_repo_id: str | None = None,
    expected_revision: str | None = None,
) -> list[ValidationViolation]:
    """Validate an annotation table with the completeness requirement for relabeling.

    In addition to every rule in :func:`validate_annotations`, relabeling requires
    every episode present in ``episode_lengths`` to be annotated.

    Args:
        rows: Annotation rows to validate.
        task_spec: Task specification defining subtasks and prompts.
        episode_lengths: Dataset episode lengths keyed by ``episode_index``.
        expected_repo_id: Optional repo ID every row must match.
        expected_revision: Optional revision every row must match.

    Returns:
        List of ``ValidationViolation`` items; empty when safe to relabel.
    """
    violations = validate_annotations(
        rows,
        task_spec,
        episode_lengths,
        expected_repo_id=expected_repo_id,
        expected_revision=expected_revision,
    )
    annotated_episodes = {row.episode_index for row in rows}
    for ep in sorted(set(episode_lengths) - annotated_episodes):
        violations.append(
            ValidationViolation(
                episode_index=ep,
                rule="missing_episode",
                message=f"Episode {ep} is missing from the annotation table.",
            )
        )
    return violations


def _build_episode_plans(
    rows: Sequence[AnnotationRow],
    task_spec: TaskSpec,
    episode_lengths: Mapping[int, int],
    subtask_to_task_index: Mapping[str, int],
) -> tuple[dict[int, np.ndarray], dict[int, tuple[str, ...]]]:
    """Compile validated annotation rows into per-episode task-index lookup arrays."""
    by_episode: dict[int, list[AnnotationRow]] = defaultdict(list)
    for row in rows:
        by_episode[row.episode_index].append(row)

    episode_task_indices: dict[int, np.ndarray] = {}
    episode_tasks: dict[int, tuple[str, ...]] = {}
    for ep in sorted(episode_lengths):
        sorted_rows = sorted(by_episode[ep], key=lambda r: (r.start, r.end))
        lookup = np.empty(episode_lengths[ep], dtype=np.int64)
        ep_prompts: list[str] = []
        for row in sorted_rows:
            lookup[row.start : row.end] = subtask_to_task_index[row.subtask]
            prompt = task_spec.prompt_for(row.subtask)
            if prompt not in ep_prompts:
                ep_prompts.append(prompt)
        episode_task_indices[ep] = lookup
        episode_tasks[ep] = tuple(ep_prompts)

    return episode_task_indices, episode_tasks


def _expected_task_index_column(
    table: pa.Table,
    episode_task_indices: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Compute the expected per-frame ``task_index`` array for one data parquet table."""
    ep_ids = table["episode_index"].to_numpy()
    frame_ids = table["frame_index"].to_numpy()
    expected = np.empty(len(ep_ids), dtype=np.int64)

    for raw_ep in np.unique(ep_ids):
        ep = int(raw_ep)
        if ep not in episode_task_indices:
            raise ValueError(f"Data table references unannotated episode_index {ep}")
        mask = ep_ids == raw_ep
        ep_frames = frame_ids[mask]
        lookup = episode_task_indices[ep]
        if ep_frames.size and (int(ep_frames.min()) < 0 or int(ep_frames.max()) >= len(lookup)):
            raise ValueError(
                f"Episode {ep}: frame_index range [{int(ep_frames.min())}, {int(ep_frames.max())}] "
                f"is outside [0, {len(lookup)})"
            )
        expected[mask] = lookup[ep_frames]

    return expected


def relabel_data_table(
    table: pa.Table,
    episode_task_indices: Mapping[int, np.ndarray],
) -> pa.Table:
    """Replace only the ``task_index`` column of a data parquet table.

    Args:
        table: Source LeRobot data parquet table.
        episode_task_indices: Mapping from ``episode_index`` to per-frame ``task_index`` array.

    Returns:
        New ``pa.Table`` identical to ``table`` except for ``task_index``.
    """
    position = table.schema.get_field_index("task_index")
    if position < 0:
        raise ValueError("Data parquet table is missing required column 'task_index'")
    field = table.schema.field(position)
    task_index = _expected_task_index_column(table, episode_task_indices)
    return table.set_column(position, field, pa.array(task_index, type=field.type))


def _collect_episode_task_arrays(
    table: pa.Table,
    buffers: dict[int, list[np.ndarray]],
) -> None:
    """Append per-episode ``task_index`` slices from ``table`` into ``buffers``."""
    ep_ids = table["episode_index"].to_numpy()
    task_ids = table["task_index"].to_numpy()
    for raw_ep in np.unique(ep_ids):
        ep = int(raw_ep)
        buffers[ep].append(task_ids[ep_ids == raw_ep])


def _compute_episode_task_stats(
    buffers: Mapping[int, Sequence[np.ndarray]],
    episode_lengths: Mapping[int, int],
) -> dict[int, dict[str, np.ndarray]]:
    """Compute per-episode LeRobot stats for ``task_index`` across all data files."""
    stats: dict[int, dict[str, np.ndarray]] = {}
    for ep in sorted(episode_lengths):
        chunks = buffers.get(ep)
        if not chunks:
            raise ValueError(f"Episode {ep} has no rows in data/*/*.parquet")
        arr = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        if len(arr) != episode_lengths[ep]:
            raise ValueError(
                f"Episode {ep}: data row count {len(arr)} != metadata length {episode_lengths[ep]}"
            )
        stats[ep] = get_feature_stats(arr, axis=0, keepdims=True)
    return stats


def relabel_episodes_table(
    table: pa.Table,
    episode_tasks: Mapping[int, Sequence[str]],
    stats: Mapping[int, Mapping[str, np.ndarray]],
) -> pa.Table:
    """Update ``tasks`` and ``stats/task_index/*`` columns in an episodes metadata table.

    Args:
        table: Source ``meta/episodes/*/*.parquet`` table.
        episode_tasks: Ordered task prompts present in each episode.
        stats: Per-episode ``task_index`` statistics from :func:`get_feature_stats`.

    Returns:
        Updated ``pa.Table`` with identical schema and all non-task columns untouched.
    """
    schema_names = set(table.schema.names)
    rows = table.to_pylist()
    for row in rows:
        ep = int(row["episode_index"])
        row["tasks"] = list(episode_tasks[ep])
        for stat_name, value in stats[ep].items():
            col_name = f"stats/task_index/{stat_name}"
            if col_name in schema_names:
                row[col_name] = np.asarray(value).tolist()
    return pa.Table.from_pylist(rows, schema=table.schema)


def aggregate_task_index_stats(
    stats: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    """Aggregate per-episode ``task_index`` stats into ``meta/stats.json`` format.

    Args:
        stats: Per-episode ``task_index`` stats keyed by ``episode_index``.

    Returns:
        JSON-serializable statistics dictionary for ``stats["task_index"]``.
    """
    aggregated = aggregate_stats([{"task_index": dict(s)} for _, s in sorted(stats.items())])[
        "task_index"
    ]
    payload: dict[str, Any] = dict(aggregated)
    return serialize_dict(payload)


def relabel_dataset(
    src: Path,
    dst: Path,
    rows: Sequence[AnnotationRow],
    task_spec: TaskSpec,
    *,
    expected_repo_id: str | None = None,
    expected_revision: str | None = None,
) -> RelabelResult:
    """Create a relabeled LeRobot v3.0 dataset copy from an annotation table and task spec.

    Only task-related bookkeeping changes: per-frame ``task_index`` in ``data/``,
    per-episode ``tasks`` and ``stats/task_index/*`` in ``meta/episodes/``,
    ``meta/tasks.parquet`` (row order equal to ``task_index``), ``total_tasks`` in
    ``meta/info.json``, and ``task_index`` in ``meta/stats.json``. Videos are
    hard-linked without re-encoding. Nothing is written if validation fails or if
    ``dst`` already exists.

    Args:
        src: Source LeRobot v3.0 dataset root.
        dst: Destination dataset root (must not exist).
        rows: Parsed annotation rows covering every episode in ``src``.
        task_spec: Task specification with ordered subtasks and prompts.
        expected_repo_id: Optional repo ID enforced against the annotation table.
        expected_revision: Optional revision enforced against the annotation table.

    Returns:
        ``RelabelResult`` summarizing the written dataset.

    Raises:
        RelabelValidationError: If the annotation table is invalid or missing episodes.
        FileExistsError: If ``dst`` already exists.
        FileNotFoundError: If required source metadata or data files are missing.
        ValueError: If the source dataset or task spec violates structural invariants.
    """
    src_root = src.expanduser().resolve()
    dst_root = dst.expanduser()

    info = _load_info(src_root)
    episode_lengths = load_episode_lengths(src_root)
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

    prompts, subtask_to_task_index = _resolve_task_prompts(task_spec)
    violations = validate_annotations_for_relabel(
        rows,
        task_spec,
        episode_lengths,
        expected_repo_id=expected_repo_id,
        expected_revision=expected_revision,
    )
    if violations:
        raise RelabelValidationError(violations)

    if dst_root.exists():
        raise FileExistsError(f"{dst_root} already exists; refusing to overwrite")

    episode_task_indices, episode_tasks = _build_episode_plans(
        rows, task_spec, episode_lengths, subtask_to_task_index
    )

    dst_created = False
    try:
        dst_root.mkdir(parents=True, exist_ok=False)
        dst_created = True

        task_buffers: dict[int, list[np.ndarray]] = defaultdict(list)
        total_frames = 0
        task_counts = np.zeros(len(prompts), dtype=np.int64)

        for path in data_files:
            relabeled = relabel_data_table(pq.read_table(path), episode_task_indices)
            out = dst_root / path.relative_to(src_root)
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(relabeled, out)
            _collect_episode_task_arrays(relabeled, task_buffers)
            total_frames += relabeled.num_rows
            task_counts += np.bincount(
                relabeled["task_index"].to_numpy(),
                minlength=len(prompts),
            )

        episode_stats = _compute_episode_task_stats(task_buffers, episode_lengths)

        for path in sorted((src_root / "meta" / "episodes").glob("*/*.parquet")):
            out = dst_root / path.relative_to(src_root)
            out.parent.mkdir(parents=True, exist_ok=True)
            updated = relabel_episodes_table(pq.read_table(path), episode_tasks, episode_stats)
            pq.write_table(updated, out)

        tasks_df = pd.DataFrame(
            {"task_index": range(len(prompts))},
            index=pd.Index(prompts, name="task"),
        )
        write_tasks(tasks_df, dst_root)

        dataset_stats: dict[str, Any] = json.loads(stats_path.read_text(encoding="utf-8"))
        dataset_stats["task_index"] = aggregate_task_index_stats(episode_stats)
        write_json(dataset_stats, dst_root / "meta" / "stats.json")

        updated_info = dict(info)
        updated_info["total_tasks"] = len(prompts)
        write_info(updated_info, dst_root)

        videos_src = src_root / "videos"
        if videos_src.exists():
            shutil.copytree(videos_src, dst_root / "videos", copy_function=os.link)
    except Exception:
        # Never leave a half-written output directory behind if relabeling aborts mid-way.
        if dst_created and dst_root.exists():
            shutil.rmtree(dst_root, ignore_errors=True)
        raise

    return RelabelResult(
        dst=dst_root,
        total_episodes=total_episodes,
        total_frames=total_frames,
        tasks=prompts,
        task_counts=tuple(int(c) for c in task_counts),
    )


def _check_data_files(
    src_root: Path,
    dst_root: Path,
    episode_task_indices: Mapping[int, np.ndarray],
) -> tuple[list[str], list[pa.Table]]:
    errors: list[str] = []
    src_files = sorted(p.relative_to(src_root) for p in (src_root / "data").glob("*/*.parquet"))
    dst_files = sorted(p.relative_to(dst_root) for p in (dst_root / "data").glob("*/*.parquet"))
    if src_files != dst_files:
        return ["data parquet file lists differ between src and dst"], []
    if not dst_files:
        return ["no data parquet files found in dst"], []

    dst_tables: list[pa.Table] = []
    for rel in src_files:
        src_table = pq.read_table(src_root / rel)
        dst_table = pq.read_table(dst_root / rel)
        dst_tables.append(dst_table)

        if src_table.num_rows != dst_table.num_rows or src_table.column_names != (
            dst_table.column_names
        ):
            errors.append(
                f"{rel}: shape/columns mismatch ({dst_table.num_rows} rows / "
                f"{dst_table.column_names} vs {src_table.num_rows} / {src_table.column_names})"
            )
            continue

        for col in src_table.column_names:
            if col != "task_index" and not src_table[col].equals(dst_table[col]):
                errors.append(f"{rel}: data column {col!r} differs")

        try:
            expected = _expected_task_index_column(dst_table, episode_task_indices)
        except ValueError as exc:
            errors.append(f"{rel}: {exc}")
            continue

        actual = dst_table["task_index"].to_numpy()
        bad = np.flatnonzero(actual != expected)
        if len(bad):
            first = int(bad[0])
            errors.append(
                f"{rel}: task_index has {len(bad)} wrong row(s), first at row {first} "
                f"(got {int(actual[first])}, expected {int(expected[first])})"
            )

    return errors, dst_tables


def _check_episodes_files(
    src_root: Path,
    dst_root: Path,
    episode_tasks: Mapping[int, Sequence[str]],
    expected_stats: Mapping[int, Mapping[str, np.ndarray]],
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
        if src_table.num_rows != dst_table.num_rows or src_table.column_names != (
            dst_table.column_names
        ):
            errors.append(f"{rel}: episodes table shape/columns differ")
            continue

        for col in src_table.column_names:
            if (
                col != "tasks"
                and not col.startswith("stats/task_index/")
                and not src_table[col].equals(dst_table[col])
            ):
                errors.append(f"{rel}: episode column {col!r} differs")

        schema_names = set(dst_table.schema.names)
        for row in dst_table.to_pylist():
            ep = int(row["episode_index"])
            expected_ep_tasks = list(episode_tasks[ep])
            if row["tasks"] != expected_ep_tasks:
                errors.append(
                    f"episode {ep}: tasks={row['tasks']!r}, expected {expected_ep_tasks!r}"
                )
            if ep in expected_stats:
                for stat_name, exp_val in expected_stats[ep].items():
                    col_name = f"stats/task_index/{stat_name}"
                    if col_name in schema_names and not np.allclose(
                        np.asarray(row[col_name], dtype=np.float64),
                        np.asarray(exp_val, dtype=np.float64),
                        equal_nan=True,
                    ):
                        errors.append(f"episode {ep}: {col_name} does not match relabeled data")

    return errors


def _check_meta_files(
    src_root: Path,
    dst_root: Path,
    expected_prompts: Sequence[str],
    expected_agg_stats: Mapping[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    tasks_path = dst_root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        errors.append("missing meta/tasks.parquet in dst")
    else:
        tasks = pd.read_parquet(tasks_path)
        expected_pairs = [(prompt, idx) for idx, prompt in enumerate(expected_prompts)]
        actual_pairs = list(zip(tasks.index.tolist(), tasks["task_index"].tolist(), strict=True))
        if actual_pairs != expected_pairs:
            errors.append(
                f"tasks.parquet mismatch: got {actual_pairs!r}, expected {expected_pairs!r}"
            )

    info_src = json.loads((src_root / "meta" / "info.json").read_text(encoding="utf-8"))
    info_dst_path = dst_root / "meta" / "info.json"
    if not info_dst_path.is_file():
        errors.append("missing meta/info.json in dst")
    else:
        info_dst = json.loads(info_dst_path.read_text(encoding="utf-8"))
        if {**info_src, "total_tasks": len(expected_prompts)} != info_dst:
            errors.append("info.json differs outside total_tasks")

    stats_src = json.loads((src_root / "meta" / "stats.json").read_text(encoding="utf-8"))
    stats_dst_path = dst_root / "meta" / "stats.json"
    task_stats_dst: dict[str, Any] = {}
    if not stats_dst_path.is_file():
        errors.append("missing meta/stats.json in dst")
    else:
        stats_dst = json.loads(stats_dst_path.read_text(encoding="utf-8"))
        if {k: v for k, v in stats_src.items() if k != "task_index"} != {
            k: v for k, v in stats_dst.items() if k != "task_index"
        }:
            errors.append("stats.json differs outside task_index")
        task_stats_dst = stats_dst.get("task_index", {})
        if expected_agg_stats is not None:
            if set(task_stats_dst) != set(expected_agg_stats):
                errors.append(
                    f"stats.json task_index keys differ: {sorted(task_stats_dst)} vs "
                    f"{sorted(expected_agg_stats)}"
                )
            else:
                for key, exp_val in expected_agg_stats.items():
                    if not np.allclose(
                        np.asarray(task_stats_dst[key], dtype=np.float64),
                        np.asarray(exp_val, dtype=np.float64),
                        equal_nan=True,
                    ):
                        errors.append(f"stats.json task_index/{key} does not match aggregated data")

    return errors, task_stats_dst


def _check_videos(src_root: Path, dst_root: Path) -> list[str]:
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


def verify_relabel(
    src: Path,
    dst: Path,
    rows: Sequence[AnnotationRow],
    task_spec: TaskSpec,
    *,
    expected_repo_id: str | None = None,
    expected_revision: str | None = None,
) -> VerifyResult:
    """Verify that ``dst`` differs from ``src`` only in task bookkeeping matching ``rows``.

    Checks every data column except ``task_index``, every episodes metadata column
    except ``tasks`` and ``stats/task_index/*``, ``meta/tasks.parquet`` row order
    and values, ``meta/info.json`` outside ``total_tasks``, ``meta/stats.json``
    outside ``task_index``, ``task_index`` statistics consistency, and hard-linked
    video files.

    Args:
        src: Source dataset root before relabeling.
        dst: Relabeled dataset root to verify.
        rows: Annotation rows used for relabeling.
        task_spec: Task specification used for relabeling.
        expected_repo_id: Optional repo ID enforced against annotations.
        expected_revision: Optional revision enforced against annotations.

    Returns:
        ``VerifyResult`` with observed counts, stats, and any detected errors.
    """
    src_root = src.expanduser().resolve()
    dst_root = dst.expanduser().resolve()
    prompts, subtask_to_task_index = _resolve_task_prompts(task_spec)

    if not dst_root.exists():
        return VerifyResult(
            total_frames=0,
            tasks=prompts,
            task_counts=tuple(0 for _ in prompts),
            stats_mean=[],
            stats_count=[],
            data_mean=0.0,
            errors=(f"dst dataset directory does not exist: {dst_root}",),
        )

    info = _load_info(src_root)
    episode_lengths = load_episode_lengths(src_root)
    total_episodes = int(info["total_episodes"])
    if sorted(episode_lengths) != list(range(total_episodes)):
        return VerifyResult(
            total_frames=0,
            tasks=prompts,
            task_counts=tuple(0 for _ in prompts),
            stats_mean=[],
            stats_count=[],
            data_mean=0.0,
            errors=(
                f"src episode indices in meta/episodes are not contiguous 0..{total_episodes - 1}",
            ),
        )

    violations = validate_annotations_for_relabel(
        rows,
        task_spec,
        episode_lengths,
        expected_repo_id=expected_repo_id,
        expected_revision=expected_revision,
    )
    if violations:
        return VerifyResult(
            total_frames=0,
            tasks=prompts,
            task_counts=tuple(0 for _ in prompts),
            stats_mean=[],
            stats_count=[],
            data_mean=0.0,
            errors=tuple(str(v) for v in violations),
        )

    episode_task_indices, episode_tasks = _build_episode_plans(
        rows, task_spec, episode_lengths, subtask_to_task_index
    )

    errors: list[str] = []
    data_errors, dst_tables = _check_data_files(src_root, dst_root, episode_task_indices)
    errors.extend(data_errors)

    task_buffers: dict[int, list[np.ndarray]] = defaultdict(list)
    all_task_indices: list[np.ndarray] = []
    for table in dst_tables:
        if "episode_index" in table.column_names and "task_index" in table.column_names:
            _collect_episode_task_arrays(table, task_buffers)
            all_task_indices.append(table["task_index"].to_numpy())

    expected_ep_stats: dict[int, dict[str, np.ndarray]] = {}
    expected_agg_stats: dict[str, Any] | None = None
    if dst_tables:
        try:
            expected_ep_stats = _compute_episode_task_stats(task_buffers, episode_lengths)
            expected_agg_stats = aggregate_task_index_stats(expected_ep_stats)
        except ValueError as exc:
            errors.append(str(exc))

    errors.extend(_check_episodes_files(src_root, dst_root, episode_tasks, expected_ep_stats))
    meta_errors, task_stats_dst = _check_meta_files(src_root, dst_root, prompts, expected_agg_stats)
    errors.extend(meta_errors)
    errors.extend(_check_videos(src_root, dst_root))

    task_all = (
        np.concatenate(all_task_indices) if all_task_indices else np.array([], dtype=np.int64)
    )
    total_frames = len(task_all)
    data_mean = float(task_all.mean()) if total_frames > 0 else 0.0
    if total_frames > 0 and int(task_all.min()) >= 0:
        counts = np.bincount(task_all, minlength=len(prompts))
        task_counts = tuple(int(c) for c in counts[: len(prompts)])
    else:
        task_counts = tuple(0 for _ in prompts)

    stats_mean = [float(x) for x in task_stats_dst.get("mean", [])]
    stats_count = [int(x) for x in task_stats_dst.get("count", [])]
    if total_frames > 0 and (
        not stats_mean
        or not stats_count
        or not np.isclose(stats_mean[0], data_mean)
        or stats_count[0] != total_frames
    ):
        errors.append("stats.json task_index mean/count does not match the data")

    return VerifyResult(
        total_frames=total_frames,
        tasks=prompts,
        task_counts=task_counts,
        stats_mean=stats_mean,
        stats_count=stats_count,
        data_mean=data_mean,
        errors=tuple(errors),
    )
