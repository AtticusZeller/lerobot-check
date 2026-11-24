"""Unit tests for N-subtask relabeling and verification."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.io_utils import serialize_dict, write_info, write_json, write_tasks
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from lerobot_check import cli
from lerobot_check.relabel import RelabelValidationError, relabel_dataset, verify_relabel
from lerobot_check.schema import (
    AnnotationRow,
    SubtaskSpec,
    TaskSpec,
    save_annotations,
    save_task_spec,
)

ORIGINAL_PROMPT = "Complete the full multi-stage assembly."


def _make_spec(prompts: Sequence[tuple[str, str]]) -> TaskSpec:
    return TaskSpec(
        name="synthetic_task",
        subtasks=tuple(SubtaskSpec(name=name, prompt=prompt) for name, prompt in prompts),
    )


def _row(
    *,
    episode_index: int,
    subtask: str,
    start: int,
    end: int,
    contact: int | None = None,
    grasp: int | None = None,
    success: bool = True,
    repo_id: str = "org/synthetic",
    revision: str = "local",
) -> AnnotationRow:
    return AnnotationRow(
        repo_id=repo_id,
        revision=revision,
        episode_index=episode_index,
        subtask=subtask,
        start=start,
        end=end,
        contact=contact,
        grasp=grasp,
        success=success,
        annotator="agent",
    )


def _build_synthetic_v3_dataset(
    root: Path,
    episode_lengths: Sequence[int],
    *,
    fps: int = 30,
    episodes_per_file: int = 2,
    revision_sha: str | None = None,
) -> Path:
    """Create a minimal valid LeRobot v3.0 dataset on disk with data, meta, and dummy videos."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    data_schema = pa.schema(
        [
            ("action", pa.list_(pa.float32(), 4)),
            ("observation.state", pa.list_(pa.float32(), 4)),
            ("timestamp", pa.float32()),
            ("frame_index", pa.int64()),
            ("episode_index", pa.int64()),
            ("index", pa.int64()),
            ("task_index", pa.int64()),
        ]
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
            actions = rng.normal(size=(length, 4)).astype(np.float32)
            states = rng.normal(size=(length, 4)).astype(np.float32)
            timestamps = (np.arange(length, dtype=np.float32)) / float(fps)
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
                "tasks": [ORIGINAL_PROMPT],
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
                "action": pa.array(
                    [row.tolist() for row in np.vstack(actions_all)],
                    type=pa.list_(pa.float32(), 4),
                ),
                "observation.state": pa.array(
                    [row.tolist() for row in np.vstack(states_all)],
                    type=pa.list_(pa.float32(), 4),
                ),
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
        video_out.write_bytes(f"dummy-mp4-content-file-{file_idx}".encode())

    ep_out = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episode_rows), ep_out)

    tasks_df = pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index([ORIGINAL_PROMPT], name="task"),
    )
    write_tasks(tasks_df, root)

    agg = aggregate_stats(episode_stats_list)
    write_json(serialize_dict(agg), root / "meta" / "stats.json")

    info = {
        "codebase_version": "v3.0",
        "robot_type": "synthetic",
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
            "action": {"dtype": "float32", "shape": [4], "names": ["a0", "a1", "a2", "a3"]},
            "observation.state": {
                "dtype": "float32",
                "shape": [4],
                "names": ["s0", "s1", "s2", "s3"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    write_info(info, root)

    if revision_sha is not None:
        tree_file = root / ".cache" / "huggingface" / "trees" / f"{revision_sha}.json"
        tree_file.parent.mkdir(parents=True, exist_ok=True)
        tree_file.write_text("{}", encoding="utf-8")

    return root


def test_relabel_four_subtasks_and_single_subtask_supplement(tmp_path: Path) -> None:
    src = _build_synthetic_v3_dataset(tmp_path / "src", [20, 15, 12], episodes_per_file=2)
    dst = tmp_path / "dst"
    spec = _make_spec(
        [
            ("pick_lid", "Pick up the lid."),
            ("scoop_item", "Scoop the item from the bin."),
            ("place_item", "Place the item into the slot."),
            ("close_lid", "Close the lid."),
        ]
    )
    rows = [
        # Episode 0: all 4 subtasks
        _row(episode_index=0, subtask="pick_lid", start=0, end=4),
        _row(episode_index=0, subtask="scoop_item", start=4, end=10, contact=5, grasp=6),
        _row(episode_index=0, subtask="place_item", start=10, end=16),
        _row(episode_index=0, subtask="close_lid", start=16, end=20),
        # Episode 1: single-subtask supplement (only subtask index 2)
        _row(episode_index=1, subtask="place_item", start=0, end=15, contact=3, grasp=5),
        # Episode 2: partial 2-subtask supplement (subtask indices 1 and 2)
        _row(episode_index=2, subtask="scoop_item", start=0, end=5),
        _row(episode_index=2, subtask="place_item", start=5, end=12),
    ]

    result = relabel_dataset(src, dst, rows, spec)
    assert result.total_episodes == 3
    assert result.total_frames == 47
    assert result.task_counts == (4, 11, 28, 4)

    # tasks.parquet row order equals task_index and works with LeRobotDatasetMetadata.iloc
    meta = LeRobotDatasetMetadata(repo_id="org/synthetic", root=dst, force_cache_sync=False)
    assert meta.total_tasks == 4
    for idx, sub in enumerate(spec.subtasks):
        assert meta.tasks.iloc[idx].name == sub.prompt
        assert int(meta.tasks.iloc[idx]["task_index"]) == idx

    # Verify per-frame task_index and prompt at every boundary
    data = pa.concat_tables([pq.read_table(p) for p in sorted((dst / "data").glob("*/*.parquet"))])
    df = data.select(["episode_index", "frame_index", "task_index"]).to_pandas()
    ep0 = df[df["episode_index"] == 0].set_index("frame_index")["task_index"].to_dict()
    assert ep0[3] == 0 and meta.tasks.iloc[ep0[3]].name == "Pick up the lid."
    assert ep0[4] == 1 and meta.tasks.iloc[ep0[4]].name == "Scoop the item from the bin."
    assert ep0[9] == 1 and ep0[10] == 2
    assert ep0[15] == 2 and ep0[16] == 3

    ep1 = df[df["episode_index"] == 1]["task_index"].to_numpy()
    assert np.all(ep1 == 2)

    ep2 = df[df["episode_index"] == 2].set_index("frame_index")["task_index"].to_dict()
    assert ep2[4] == 1 and ep2[5] == 2

    # Episode metadata lists only the tasks present in each episode
    episodes_table = pq.read_table(dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    ep_tasks = {int(r["episode_index"]): r["tasks"] for r in episodes_table.to_pylist()}
    assert ep_tasks[0] == [s.prompt for s in spec.subtasks]
    assert ep_tasks[1] == ["Place the item into the slot."]
    assert ep_tasks[2] == ["Scoop the item from the bin.", "Place the item into the slot."]

    # Videos are hard-linked
    for rel in sorted(p.relative_to(src) for p in (src / "videos").rglob("*") if p.is_file()):
        assert (src / rel).stat().st_ino == (dst / rel).stat().st_ino

    # verify_relabel returns OK
    ver = verify_relabel(src, dst, rows, spec)
    assert ver.ok, ver.errors
    assert ver.total_frames == 47
    assert ver.task_counts == (4, 11, 28, 4)


def test_relabel_pure_single_subtask_supplement_dataset(tmp_path: Path) -> None:
    src = _build_synthetic_v3_dataset(tmp_path / "src", [10, 8], episodes_per_file=2)
    dst = tmp_path / "dst"
    spec = _make_spec(
        [
            ("stage_a", "Do stage A."),
            ("stage_b", "Do stage B."),
            ("stage_c", "Do stage C."),
        ]
    )
    rows = [
        _row(episode_index=0, subtask="stage_b", start=0, end=10),
        _row(episode_index=1, subtask="stage_b", start=0, end=8),
    ]

    result = relabel_dataset(src, dst, rows, spec)
    assert result.task_counts == (0, 18, 0)

    ver = verify_relabel(src, dst, rows, spec)
    assert ver.ok, ver.errors


def test_relabel_rejects_missing_episode_and_writes_nothing(tmp_path: Path) -> None:
    src = _build_synthetic_v3_dataset(tmp_path / "src", [10, 10])
    dst = tmp_path / "dst"
    spec = _make_spec([("a", "Stage A."), ("b", "Stage B."), ("c", "Stage C.")])
    rows = [
        _row(episode_index=0, subtask="a", start=0, end=5),
        _row(episode_index=0, subtask="b", start=5, end=10),
        # Episode 1 intentionally missing
    ]

    with pytest.raises(RelabelValidationError) as exc_info:
        relabel_dataset(src, dst, rows, spec)

    assert [v.rule for v in exc_info.value.violations] == ["missing_episode"]
    assert exc_info.value.violations[0].episode_index == 1
    assert not dst.exists()


@pytest.mark.parametrize(
    ("bad_rows", "expected_rule"),
    [
        (
            [
                _row(episode_index=0, subtask="a", start=0, end=4),
                _row(episode_index=0, subtask="b", start=6, end=10),
            ],
            "missing_frames",
        ),
        (
            [
                _row(episode_index=0, subtask="a", start=0, end=6),
                _row(episode_index=0, subtask="b", start=4, end=10),
            ],
            "overlap",
        ),
        (
            [
                _row(episode_index=0, subtask="a", start=0, end=12),
            ],
            "out_of_range",
        ),
        (
            [
                _row(episode_index=0, subtask="b", start=0, end=5),
                _row(episode_index=0, subtask="a", start=5, end=10),
            ],
            "wrong_subtask_order",
        ),
        (
            [
                _row(episode_index=0, subtask="unknown", start=0, end=10),
            ],
            "unknown_subtask",
        ),
    ],
)
def test_relabel_rejects_invalid_annotations_without_creating_dst(
    tmp_path: Path,
    bad_rows: list[AnnotationRow],
    expected_rule: str,
) -> None:
    src = _build_synthetic_v3_dataset(tmp_path / "src", [10])
    dst = tmp_path / "dst"
    spec = _make_spec([("a", "Stage A."), ("b", "Stage B."), ("c", "Stage C.")])

    with pytest.raises(RelabelValidationError) as exc_info:
        relabel_dataset(src, dst, bad_rows, spec)

    assert expected_rule in {v.rule for v in exc_info.value.violations}
    assert not dst.exists()


def test_relabel_refuses_existing_dst_and_duplicate_prompts(tmp_path: Path) -> None:
    src = _build_synthetic_v3_dataset(tmp_path / "src", [10])
    spec = _make_spec([("a", "Stage A."), ("b", "Stage B.")])
    rows = [
        _row(episode_index=0, subtask="a", start=0, end=5),
        _row(episode_index=0, subtask="b", start=5, end=10),
    ]

    existing_dst = tmp_path / "existing"
    existing_dst.mkdir()
    marker = existing_dst / "keep.txt"
    marker.write_text("do not delete", encoding="utf-8")

    with pytest.raises(FileExistsError):
        relabel_dataset(src, existing_dst, rows, spec)
    assert marker.read_text(encoding="utf-8") == "do not delete"

    dup_spec = _make_spec([("a", "Same prompt."), ("b", "Same prompt.")])
    dst2 = tmp_path / "dst2"
    with pytest.raises(ValueError, match="duplicate prompt"):
        relabel_dataset(src, dst2, rows, dup_spec)
    assert not dst2.exists()


def test_verify_relabel_fails_when_non_task_fields_or_hardlinks_altered(
    tmp_path: Path,
) -> None:
    src = _build_synthetic_v3_dataset(tmp_path / "src", [12, 10], episodes_per_file=2)
    dst = tmp_path / "dst"
    spec = _make_spec([("s0", "First."), ("s1", "Second."), ("s2", "Third.")])
    rows = [
        _row(episode_index=0, subtask="s0", start=0, end=4),
        _row(episode_index=0, subtask="s1", start=4, end=8),
        _row(episode_index=0, subtask="s2", start=8, end=12),
        _row(episode_index=1, subtask="s1", start=0, end=10),
    ]
    relabel_dataset(src, dst, rows, spec)
    assert verify_relabel(src, dst, rows, spec).ok

    # 1. Alter non-task data column (observation.state)
    bad_data = tmp_path / "bad_data"
    shutil.copytree(dst, bad_data, copy_function=shutil.copy2)
    data_file = bad_data / "data" / "chunk-000" / "file-000.parquet"
    tbl = pq.read_table(data_file)
    states = tbl["observation.state"].to_pylist()
    states[0][0] = float(states[0][0]) + 1.0
    pos = tbl.schema.get_field_index("observation.state")
    tbl = tbl.set_column(
        pos, tbl.schema.field(pos), pa.array(states, type=tbl.schema.field(pos).type)
    )
    pq.write_table(tbl, data_file)
    res = verify_relabel(src, bad_data, rows, spec)
    assert not res.ok
    assert any("observation.state" in err for err in res.errors)

    # 1b. Alter non-task episode metadata column (length)
    bad_ep = tmp_path / "bad_ep"
    shutil.copytree(dst, bad_ep, copy_function=os.link)
    ep_file = bad_ep / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_tbl = pq.read_table(ep_file)
    ep_rows = ep_tbl.to_pylist()
    ep_rows[0]["length"] = int(ep_rows[0]["length"]) + 1
    ep_file.unlink()
    pq.write_table(pa.Table.from_pylist(ep_rows, schema=ep_tbl.schema), ep_file)
    res = verify_relabel(src, bad_ep, rows, spec)
    assert not res.ok
    assert any("length" in err for err in res.errors)

    # 2. Alter non-task info.json field (fps)
    bad_info = tmp_path / "bad_info"
    shutil.copytree(dst, bad_info, copy_function=shutil.copy2)
    info_path = bad_info / "meta" / "info.json"
    info_data = json.loads(info_path.read_text(encoding="utf-8"))
    info_data["fps"] = 60
    info_path.write_text(json.dumps(info_data, indent=4), encoding="utf-8")
    res = verify_relabel(src, bad_info, rows, spec)
    assert not res.ok
    assert any("info.json" in err for err in res.errors)

    # 3. Alter non-task stats.json field (action)
    bad_stats = tmp_path / "bad_stats"
    shutil.copytree(dst, bad_stats, copy_function=shutil.copy2)
    stats_path = bad_stats / "meta" / "stats.json"
    stats_data = json.loads(stats_path.read_text(encoding="utf-8"))
    stats_data["action"]["mean"][0] += 99.0
    stats_path.write_text(json.dumps(stats_data, indent=4), encoding="utf-8")
    res = verify_relabel(src, bad_stats, rows, spec)
    assert not res.ok
    assert any("stats.json differs outside task_index" in err for err in res.errors)

    # 4. Break video hard link by replacing with a regular copy (all other files hard-linked)
    bad_video = tmp_path / "bad_video"
    shutil.copytree(dst, bad_video, copy_function=shutil.copy2)
    res = verify_relabel(src, bad_video, rows, spec)
    assert not res.ok
    assert any("hard link" in err for err in res.errors)

    # 5. Swap tasks.parquet row order while keeping hard links intact
    bad_tasks = tmp_path / "bad_tasks"
    shutil.copytree(dst, bad_tasks, copy_function=os.link)
    tasks_path = bad_tasks / "meta" / "tasks.parquet"
    tasks_path.unlink()
    swapped = pd.DataFrame(
        {"task_index": [1, 0, 2]},
        index=pd.Index(["Second.", "First.", "Third."], name="task"),
    )
    swapped.to_parquet(tasks_path)
    res = verify_relabel(src, bad_tasks, rows, spec)
    assert not res.ok
    assert any("tasks.parquet" in err for err in res.errors)


def test_cli_relabel_and_verify_relabel_roundtrip_and_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sha = "a" * 40
    src = _build_synthetic_v3_dataset(tmp_path / "src", [10, 12], revision_sha=sha)
    dst = tmp_path / "dst"
    spec = _make_spec([("s0", "Step zero."), ("s1", "Step one."), ("s2", "Step two.")])
    spec_path = tmp_path / "spec.yaml"
    save_task_spec(spec, spec_path)

    # Wrong revision -> CLI rejects and creates no output
    bad_ann = tmp_path / "bad_rev.csv"
    save_annotations(
        [
            _row(episode_index=0, subtask="s0", start=0, end=10, revision="wrong_rev"),
            _row(episode_index=1, subtask="s1", start=0, end=12, revision="wrong_rev"),
        ],
        bad_ann,
    )
    code = cli.main(
        [
            "relabel",
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--annotations",
            str(bad_ann),
            "--task-spec",
            str(spec_path),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "revision_mismatch" in captured.err
    assert not dst.exists()

    # Valid annotations -> relabel succeeds and verify-relabel prints OK
    good_ann = tmp_path / "good.csv"
    save_annotations(
        [
            _row(episode_index=0, subtask="s0", start=0, end=4, revision=sha),
            _row(episode_index=0, subtask="s1", start=4, end=7, revision=sha),
            _row(episode_index=0, subtask="s2", start=7, end=10, revision=sha),
            _row(episode_index=1, subtask="s2", start=0, end=12, revision=sha),
        ],
        good_ann,
    )
    code = cli.main(
        [
            "relabel",
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--annotations",
            str(good_ann),
            "--task-spec",
            str(spec_path),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "Wrote" in captured.out

    code = cli.main(
        [
            "verify-relabel",
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--annotations",
            str(good_ann),
            "--task-spec",
            str(spec_path),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.out + captured.err
    assert "OK" in captured.out
