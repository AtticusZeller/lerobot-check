"""Unit tests for lerobot_check.report offline HTML generation, summary.json, and formatters."""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_check.analysis import (
    DatasetAnalysis,
    MixEntry,
    ProvenanceCounts,
    compute_all_coverages,
    compute_episode_speed_metrics,
    compute_frame_shares,
    compute_mix,
    compute_pairwise_offsets,
    count_annotators,
    extract_episode_object_poses,
    extract_episode_trajectories,
    parse_episode_groups,
    suggest_placements,
)
from lerobot_check.dataset import ArmState, format_dataset_info
from lerobot_check.report import (
    COLLISION_DISCLAIMER,
    GRASP_RELATIVE_CAPTION,
    SUMMARY_SCHEMA_VERSION,
    build_report_summary,
    format_frame_share_table,
    format_mix_report,
    format_suggestions_table,
    markdown_to_html,
    render_html_report,
    write_html_report,
    write_report_bundle,
    write_suggestions_csv,
)
from lerobot_check.schema import (
    AnnotationRow,
    CoverageSpec,
    ObjectSpec,
    SubtaskSpec,
    TaskSpec,
)


def _yaw_rot6d(yaw_deg: float) -> np.ndarray:
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([c, s, 0.0, -s, c, 0.0], dtype=np.float64)


def _synthetic_analysis(
    label: str,
    *,
    offset_x: float = 0.0,
    include_agent_row: bool = False,
) -> tuple[TaskSpec, DatasetAnalysis]:
    spec = TaskSpec(
        name="synthetic_egg_task",
        subtasks=(
            SubtaskSpec(name="open_carton", prompt="Open the egg carton."),
            SubtaskSpec(name="place_eggs", prompt="Place eggs in tray."),
        ),
        objects=(
            ObjectSpec("egg_carton", "open_carton", "release", "left", 180.0),
            ObjectSpec("egg", "place_eggs", "grasp", "left", 360.0),
        ),
        fine_window_s=1.0,
        coverage=CoverageSpec(xy_bin_m=0.05, yaw_bin_deg=30.0, sparse_threshold=2),
    )

    fps = 10.0
    n = 80
    rows = [
        # Episode 0: both subtasks, place_eggs has grasp.
        AnnotationRow("org/ds", "rev1", 0, "open_carton", 0, 30, None, None, True, "converted"),
        AnnotationRow(
            "org/ds",
            "rev1",
            0,
            "place_eggs",
            30,
            n,
            40,
            45,
            True,
            "agent" if include_agent_row else "human",
        ),
        # Episode 1: skipped (absent from rows).
        # Episode 2: place_eggs missing grasp -> egg event skipped.
        AnnotationRow("org/ds", "rev1", 2, "open_carton", 0, 30, None, None, True, "converted"),
        AnnotationRow("org/ds", "rev1", 2, "place_eggs", 30, n, None, None, True, "converted"),
    ]
    ep_lengths = {0: n, 1: n, 2: n}
    frame_shares = compute_frame_shares(
        rows, spec, fps=fps, episode_lengths=ep_lengths, label=label
    )

    speed_metrics = []
    object_poses = []
    skipped_objects = []
    trajectories = []
    for ep in (0, 2):
        ep_rows = [r for r in rows if r.episode_index == ep]
        left_xyz = np.column_stack(
            [
                np.linspace(0.4 + offset_x, 0.6 + offset_x, n),
                np.linspace(0.0, 0.2, n),
                np.full(n, 0.85),
            ]
        )
        right_xyz = np.zeros((n, 3), dtype=np.float64)
        rot = np.tile(_yaw_rot6d(20.0 + ep * 15.0), (n, 1))
        grip = np.ones(n, dtype=np.float64)
        arms = [ArmState("left", left_xyz, rot, grip), ArmState("right", right_xyz, rot, grip)]
        arms_by_name = {a.arm: a for a in arms}

        speed_metrics.extend(
            compute_episode_speed_metrics(
                dataset_label=label,
                episode_index=ep,
                ep_rows=ep_rows,
                arms=arms,
                fps=fps,
                fine_window_s=spec.fine_window_s,
            )
        )
        poses, skipped = extract_episode_object_poses(
            dataset_label=label,
            episode_index=ep,
            ep_rows=ep_rows,
            arms_by_name=arms_by_name,
            objects=spec.objects,
        )
        object_poses.extend(poses)
        skipped_objects.extend(skipped)
        trajectories.extend(
            extract_episode_trajectories(
                dataset_label=label,
                episode_index=ep,
                ep_rows=ep_rows,
                arms=arms,
                max_points=20,
            )
        )

    analysis = DatasetAnalysis(
        label=label,
        repo_id="org/ds",
        revision="rev1",
        fps=fps,
        arms=("left", "right"),
        frame_shares=frame_shares,
        speed_metrics=tuple(speed_metrics),
        object_poses=tuple(object_poses),
        skipped_objects=tuple(skipped_objects),
        pairwise_offsets=tuple(compute_pairwise_offsets(object_poses, spec.objects)),
        trajectories=tuple(trajectories),
        provenance=count_annotators(rows),
    )
    return spec, analysis


def test_render_html_report_chinese_offline_and_multi_group(tmp_path: Path) -> None:
    spec, base_analysis = _synthetic_analysis("原数据", offset_x=0.0)
    _, supp_analysis = _synthetic_analysis("补录", offset_x=0.08)

    out_path = tmp_path / "report.html"
    write_html_report([base_analysis, supp_analysis], spec, out_path)
    html_text = out_path.read_text(encoding="utf-8")

    # Completely self-contained offline HTML: no external script/stylesheet links.
    assert re.search(r"<script[^>]+src\s*=\s*['\"]https?://", html_text, re.IGNORECASE) is None
    assert re.search(r"<link[^>]+href\s*=\s*['\"]https?://", html_text, re.IGNORECASE) is None
    assert '<html lang="zh-CN">' in html_text
    assert "Plotly" in html_text

    # Chinese section titles, table headers, group labels, skipped reporting, and grasp caption.
    assert "数据概览与跳过项" in html_text
    assert "1. 帧占比与精细窗口" in html_text
    assert "2. 精细窗口速度检查" in html_text
    assert "3. 物体初始位置与相对位移" in html_text
    assert "4. 3D TCP 轨迹" in html_text
    assert "5. 工作空间覆盖网格与稀疏格子" in html_text
    assert "分组 / 数据集" in html_text
    assert "帧占比" in html_text
    assert "精细窗口帧数" in html_text
    assert "全部合计" in html_text
    assert "原数据" in html_text
    assert "补录" in html_text
    assert "跳过的 Episode" in html_text
    assert "跳过的物体事件" in html_text
    assert "missing grasp frame" in html_text
    assert html.escape(GRASP_RELATIVE_CAPTION) in html_text


def test_report_provenance_counts_and_unreviewed_agent_warning() -> None:
    spec, reviewed = _synthetic_analysis("已复核组", include_agent_row=False)
    assert reviewed.provenance == ProvenanceCounts(agent=0, human=1, converted=3)

    reviewed_html = render_html_report([reviewed], spec)
    assert "标注来源统计" in reviewed_html
    assert "human（人工）1" in reviewed_html
    assert "converted（转换）3" in reviewed_html
    assert "agent（未复核）0" in reviewed_html
    assert 'class="alert-unreviewed"' not in reviewed_html

    _, unreviewed = _synthetic_analysis("含未复核组", include_agent_row=True)
    assert unreviewed.provenance == ProvenanceCounts(agent=1, human=0, converted=3)

    unreviewed_html = render_html_report([unreviewed], spec)
    assert "agent（未复核）1" in unreviewed_html
    assert 'class="alert-unreviewed"' in unreviewed_html
    assert "存在 1 条未经人工复核的" in unreviewed_html


def test_report_summary_json_fields_and_bundle(tmp_path: Path) -> None:
    spec, base_analysis = _synthetic_analysis("原数据", offset_x=0.0)
    _, supp_analysis = _synthetic_analysis("补录", offset_x=0.08, include_agent_row=True)

    # Directory output writes index.html and summary.json.
    report_dir = tmp_path / "report_dir"
    html_path, summary_path, summary = write_report_bundle(
        [base_analysis, supp_analysis],
        spec,
        report_dir,
        advice_markdown="# 建议\n\n- 优先补采 `open_carton`。",
    )
    assert html_path == report_dir / "index.html"
    assert summary_path == report_dir / "summary.json"
    assert html_path.is_file()
    assert summary_path.is_file()

    disk_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert disk_summary == summary
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert summary["task_name"] == "synthetic_egg_task"
    assert summary["fine_window_s"] == pytest.approx(1.0)
    assert summary["coverage_config"] == {
        "xy_bin_m": 0.05,
        "yaw_bin_deg": 30.0,
        "sparse_threshold": 2,
    }
    assert summary["provenance"] == {
        "total_rows": 8,
        "counts": {"agent": 1, "human": 1, "converted": 6},
        "has_unreviewed_agent": True,
    }
    assert summary["overall"]["annotated_episodes"] == 4
    assert summary["overall"]["total_frames"] == 320
    subtask_map = {item["subtask"]: item for item in summary["overall"]["subtasks"]}
    assert subtask_map["open_carton"]["frames"] == 120
    assert subtask_map["place_eggs"]["frames"] == 200
    assert subtask_map["open_carton"]["mean_frames_per_episode"] == pytest.approx(30.0)
    assert len(summary["groups"]) == 2
    assert summary["groups"][0]["label"] == "原数据"
    assert summary["groups"][1]["label"] == "补录"
    assert "speed_check" in summary
    assert "by_arm" in summary["speed_check"]
    assert len(summary["objects"]) == 2
    carton_obj = next(o for o in summary["objects"] if o["name"] == "egg_carton")
    assert carton_obj["sample_count"] == 4
    assert carton_obj["coverage"]["total_cells"] > 0
    assert isinstance(carton_obj["coverage"]["sparse_cells"], list)
    assert "原数据" in carton_obj["by_group"]
    assert "补录" in carton_obj["by_group"]
    assert len(summary["pairwise_offsets"]) == 1

    # Explicit .html path writes sibling summary.json.
    explicit_html = tmp_path / "custom" / "my_report.html"
    h2, s2, _ = write_report_bundle([base_analysis], spec, explicit_html)
    assert h2 == explicit_html
    assert s2 == explicit_html.parent / "summary.json"
    base_summary = build_report_summary([base_analysis], spec)
    assert base_summary["provenance"]["has_unreviewed_agent"] is False


def test_report_advice_embedding_and_safe_markdown() -> None:
    spec, analysis = _synthetic_analysis("原数据")
    advice_md = (
        "## 补采建议\n\n"
        "- 优先补采 **open_carton**（当前 `60` 帧）。\n"
        "- <script>alert('xss')</script>\n\n"
        "| 指标 | 数值 |\n"
        "| --- | --- |\n"
        "| 总帧数 | 160 |\n"
    )
    rendered = render_html_report([analysis], spec, advice_markdown=advice_md)
    assert 'class="advice-section"' in rendered
    assert '<span class="badge-agent">Agent 建议</span>' in rendered
    assert "<strong>open_carton</strong>" in rendered
    assert "<code>60</code>" in rendered
    assert "<table><thead><tr><th>指标</th><th>数值</th></tr></thead>" in rendered
    assert "<script>alert('xss')</script>" not in rendered
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in rendered
    assert "<h3>标题</h3>" in markdown_to_html("# 标题")


def test_episode_group_parsing_and_validation() -> None:
    groups_incl = parse_episode_groups(["原数据=0-120", "补录=121-173"], total_episodes=174)
    assert len(groups_incl) == 2
    assert groups_incl[0].name == "原数据"
    assert groups_incl[0].episodes[0] == 0 and groups_incl[0].episodes[-1] == 120
    assert len(groups_incl[0].episodes) == 121
    assert groups_incl[1].episodes[0] == 121 and groups_incl[1].episodes[-1] == 173
    assert len(groups_incl[1].episodes) == 53

    groups_half = parse_episode_groups(["原数据=0:121", "补录=121:174"], total_episodes=174)
    assert groups_half == groups_incl

    with pytest.raises(ValueError, match="multiple groups"):
        parse_episode_groups(["a=0-10", "b=10-20"], total_episodes=30)
    with pytest.raises(ValueError, match="Duplicate episode group name"):
        parse_episode_groups(["a=0-5", "a=6-10"], total_episodes=20)
    with pytest.raises(ValueError, match="outside"):
        parse_episode_groups(["a=0-20"], total_episodes=20)


def test_format_dataset_info(tmp_path: Path) -> None:
    ds_root = tmp_path / "lerobot" / "Xense" / "demo_ds"
    ds_root.mkdir(parents=True)
    fake_dataset = SimpleNamespace(
        root=ds_root.resolve(),
        repo_id="Xense/demo_ds",
        revision="a" * 40,
        fps=30.0,
        total_episodes=2,
        camera_keys=("observation.images.head", "observation.images.right_wrist"),
        episode_lengths={0: 300, 1: 450},
    )
    info_text = format_dataset_info(fake_dataset)  # type: ignore[arg-type]
    assert "repo_id: Xense/demo_ds" in info_text
    assert f"revision: {'a' * 40}" in info_text
    assert "fps: 30" in info_text
    assert "episodes: 2" in info_text
    assert "total_frames: 750" in info_text
    assert "observation.images.right_wrist" in info_text
    assert f".xense-viewer/subtasks/Xense__demo_ds@{'a' * 40}.csv" in info_text
    assert "0\t300\t10.00" in info_text
    assert "1\t450\t15.00" in info_text


def test_text_and_csv_formatters(tmp_path: Path) -> None:
    spec, analysis = _synthetic_analysis("base_ds")

    share_text = format_frame_share_table(analysis.frame_shares)
    assert "open_carton" in share_text
    assert "place_eggs" in share_text
    assert "TOTAL" in share_text
    assert "skipped: [1]" in share_text

    rows = [
        AnnotationRow("org/ds", "rev1", 0, "open_carton", 0, 30, None, None, True, "converted"),
        AnnotationRow("org/ds", "rev1", 0, "place_eggs", 30, 80, 40, 45, True, "human"),
    ]
    per_input, combined = compute_mix(
        [
            MixEntry("ds", rows, fps=10.0, total_episodes=1, episode_lengths={0: 80}, repeat=1),
            MixEntry("ds", rows, fps=10.0, total_episodes=1, episode_lengths={0: 80}, repeat=1),
        ],
        spec,
    )
    mix_text = format_mix_report(per_input, combined)
    assert "ds x1" in mix_text
    assert "160" in mix_text

    covs = compute_all_coverages(analysis.object_poses, spec)
    suggestions, omitted = suggest_placements(covs, count=3, seed=7)
    assert omitted == []
    sug_text = format_suggestions_table(suggestions, omitted_objects=omitted)
    assert COLLISION_DISCLAIMER in sug_text
    assert "egg_carton" in sug_text
    assert "egg" in sug_text

    csv_path = tmp_path / "suggestions.csv"
    write_suggestions_csv(suggestions, csv_path)
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 + 3 * 2  # header + 3 batch items * 2 objects

    # Also ensure single-dataset render works cleanly.
    single_html = render_html_report([analysis], spec)
    assert "synthetic_egg_task" in single_html
