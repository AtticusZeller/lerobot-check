"""Unit tests for lerobot_check.report offline HTML generation and text formatters."""

from __future__ import annotations

import html
import math
import re
from pathlib import Path

import numpy as np

from lerobot_check.analysis import (
    DatasetAnalysis,
    MixEntry,
    compute_all_coverages,
    compute_episode_speed_metrics,
    compute_frame_shares,
    compute_mix,
    compute_pairwise_offsets,
    extract_episode_object_poses,
    extract_episode_trajectories,
    suggest_placements,
)
from lerobot_check.dataset import ArmState
from lerobot_check.report import (
    COLLISION_DISCLAIMER,
    GRASP_RELATIVE_CAPTION,
    format_frame_share_table,
    format_mix_report,
    format_suggestions_table,
    render_html_report,
    write_html_report,
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


def _synthetic_analysis(label: str, offset_x: float = 0.0) -> tuple[TaskSpec, DatasetAnalysis]:
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
        AnnotationRow("org/ds", "rev1", 0, "place_eggs", 30, n, 40, 45, True, "human"),
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
    )
    return spec, analysis


def test_render_html_report_offline_and_multi_dataset(tmp_path: Path) -> None:
    spec, base_analysis = _synthetic_analysis("base_ds", offset_x=0.0)
    _, supp_analysis = _synthetic_analysis("supplement_ds", offset_x=0.08)

    out_path = tmp_path / "report.html"
    write_html_report([base_analysis, supp_analysis], spec, out_path)
    html_text = out_path.read_text(encoding="utf-8")

    # Must be completely self-contained offline HTML: no external script/stylesheet links.
    assert re.search(r"<script[^>]+src\s*=\s*['\"]https?://", html_text, re.IGNORECASE) is None
    assert re.search(r"<link[^>]+href\s*=\s*['\"]https?://", html_text, re.IGNORECASE) is None
    assert "Plotly" in html_text

    # All 5 sections, both dataset labels, skipped episode/object reporting, and grasp caption.
    assert "1. Frame Shares" in html_text
    assert "2. Speed Check" in html_text
    assert "3. Object Initial Positions" in html_text
    assert "4. 3D TCP Trajectories" in html_text
    assert "5. Workspace Coverage Grid" in html_text
    assert "base_ds" in html_text
    assert "supplement_ds" in html_text
    assert "Skipped Episodes" in html_text
    assert "Skipped Object Events" in html_text
    assert "missing grasp frame" in html_text
    assert html.escape(GRASP_RELATIVE_CAPTION) in html_text


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
