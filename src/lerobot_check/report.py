"""Offline static HTML report generation, summary.json export, and CLI formatters."""

from __future__ import annotations

import csv
import html
import io
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.offline
from plotly.subplots import make_subplots

from lerobot_check.analysis import (
    DEFAULT_MIN_REST_SPEED_MPS,
    DEFAULT_SPEED_RATIO_THRESHOLD,
    DatasetAnalysis,
    EpisodeSpeedMetric,
    FrameShareSummary,
    ObjectCoverage,
    ObjectPoseSample,
    PairwiseRelativeOffset,
    PlacementSuggestion,
    ProvenanceCounts,
    SkippedObjectEvent,
    TrajectorySegment,
    compute_all_coverages,
    yaw_bin_index,
)
from lerobot_check.schema import ObjectSpec, TaskSpec

SUMMARY_SCHEMA_VERSION = 1

COLLISION_DISCLAIMER = "说明：各物体从各自的稀疏覆盖格中独立抽样，未检查或消除物体间碰撞。"

GRASP_RELATIVE_CAPTION = (
    "以 grasp 帧 TCP 为原点的接近轨迹将每条 episode 的 [start, grasp] 平移至 (0, 0, 0)。"
    "去除工作空间绝对平移后，该视图仅反映接近方向的多样性，不代表物体绝对位置覆盖。"
)

DATASET_COLORS: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
)

# Distinct sequential colorscales for multi-dataset / multi-group 3D trajectory overlays
# so each group keeps a distinct hue while shade still encodes normalized time t in [0, 1].
DATASET_COLORSCALES: tuple[list[list[Any]], ...] = (
    [[0.0, "#c6dbef"], [0.5, "#4292c6"], [1.0, "#08306b"]],
    [[0.0, "#fdd0a2"], [0.5, "#f16913"], [1.0, "#7f2704"]],
    [[0.0, "#c7e9c0"], [0.5, "#41ab5d"], [1.0, "#00441b"]],
    [[0.0, "#fcbba1"], [0.5, "#ef3b2c"], [1.0, "#67000d"]],
    [[0.0, "#dadaeb"], [0.5, "#807dba"], [1.0, "#3f007d"]],
    [[0.0, "#d9c7b8"], [0.5, "#8c564b"], [1.0, "#3b1f1a"]],
)

SUGGESTION_CSV_COLUMNS: tuple[str, ...] = (
    "batch_index",
    "object_name",
    "arm",
    "x_m",
    "y_m",
    "yaw_deg",
    "ix",
    "iy",
    "yaw_bin",
    "current_count",
    "from_sparse_bin",
)

_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_ORDERED_ITEM_RE = re.compile(r"^\s*\d+\.\s+(.*)$")
_UNORDERED_ITEM_RE = re.compile(r"^\s*[-*]\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _dataset_color(idx: int) -> str:
    return DATASET_COLORS[idx % len(DATASET_COLORS)]


def _dataset_colorscale(idx: int, total_datasets: int) -> str | list[list[Any]]:
    if total_datasets == 1:
        return "Viridis"
    return DATASET_COLORSCALES[idx % len(DATASET_COLORSCALES)]


def format_frame_share_table(summary: FrameShareSummary) -> str:
    """Format a ``FrameShareSummary`` as an aligned plain-text table."""
    headers = (
        "Subtask",
        "Episodes",
        "Frames",
        "Share",
        "Fine Frames",
        "Fine / Subtask",
        "Fine / Total",
    )
    rows: list[tuple[str, ...]] = []
    for st in summary.subtasks:
        rows.append(
            (
                st.subtask,
                str(st.episodes),
                f"{st.frames:,}",
                f"{st.frame_share * 100:.2f}%",
                f"{st.fine_frames:,}",
                f"{st.fine_share_of_subtask * 100:.2f}%",
                f"{st.fine_share_of_total * 100:.2f}%",
            )
        )
    rows.append(
        (
            "TOTAL",
            str(summary.annotated_episodes),
            f"{summary.total_frames:,}",
            "100.00%" if summary.total_frames > 0 else "0.00%",
            f"{summary.total_fine_frames:,}",
            f"{summary.total_fine_share * 100:.2f}%",
            f"{summary.total_fine_share * 100:.2f}%",
        )
    )

    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]

    def _fmt_row(values: tuple[str, ...]) -> str:
        cells = [
            values[0].ljust(widths[0]),
            *(values[i].rjust(widths[i]) for i in range(1, len(values))),
        ]
        return " | ".join(cells)

    lines = [
        f"Dataset: {summary.label} "
        f"(annotated {summary.annotated_episodes}/{summary.total_episodes} episodes"
        + (f", skipped: {list(summary.skipped_episodes)}" if summary.skipped_episodes else "")
        + ")",
        _fmt_row(headers),
        "-+-".join("-" * w for w in widths),
        *(_fmt_row(r) for r in rows[:-1]),
        "-+-".join("-" * w for w in widths),
        _fmt_row(rows[-1]),
    ]
    return "\n".join(lines)


def format_mix_report(
    per_input: Sequence[tuple[FrameShareSummary, int]],
    combined: FrameShareSummary,
) -> str:
    """Format a multi-dataset ``mix`` result showing input weights and combined shares."""
    lines = ["Inputs:"]
    for summary, repeat in per_input:
        skipped_note = (
            f", skipped episodes: {list(summary.skipped_episodes)}"
            if summary.skipped_episodes
            else ""
        )
        lines.append(
            f"  - {summary.label} x{repeat}: {summary.annotated_episodes} annotated ep(s), "
            f"{summary.total_frames:,} frames ({summary.total_frames * repeat:,} weighted), "
            f"{summary.total_fine_frames:,} fine frames "
            f"({summary.total_fine_frames * repeat:,} weighted){skipped_note}"
        )
    lines.append("")
    lines.append(format_frame_share_table(combined))
    return "\n".join(lines)


def format_suggestions_table(
    suggestions: Sequence[PlacementSuggestion],
    *,
    omitted_objects: Sequence[str] = (),
) -> str:
    """Format next-batch placement suggestions as an aligned text table."""
    lines = [COLLISION_DISCLAIMER]
    if omitted_objects:
        lines.append("已跳过（无观测样本）的物体：" + ", ".join(sorted(omitted_objects)))
    if not suggestions:
        lines.append("无可用的摆放建议。")
        return "\n".join(lines)

    headers = (
        "Batch#",
        "Object",
        "Arm",
        "X (m)",
        "Y (m)",
        "Yaw (deg)",
        "Grid (ix,iy,yaw)",
        "Count",
        "Source",
    )
    rows = [
        (
            str(item.batch_index),
            item.object_name,
            item.arm,
            f"{item.x_m:.3f}",
            f"{item.y_m:.3f}",
            f"{item.yaw_deg:.1f}",
            f"({item.ix}, {item.iy}, {item.yaw_bin})",
            str(item.current_count),
            "sparse" if item.from_sparse_bin else "lowest",
        )
        for item in suggestions
    ]
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]

    def _fmt(vals: tuple[str, ...]) -> str:
        return " | ".join(
            vals[i].ljust(widths[i]) if i in (1, 2, 6, 8) else vals[i].rjust(widths[i])
            for i in range(len(vals))
        )

    lines.append("")
    lines.append(_fmt(headers))
    lines.append("-+-".join("-" * w for w in widths))
    lines.extend(_fmt(r) for r in rows)
    return "\n".join(lines)


def write_suggestions_csv(
    suggestions: Sequence[PlacementSuggestion],
    path: Path,
) -> None:
    """Write placement suggestions to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=SUGGESTION_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for item in suggestions:
        writer.writerow(
            {
                "batch_index": item.batch_index,
                "object_name": item.object_name,
                "arm": item.arm,
                "x_m": f"{item.x_m:.4f}",
                "y_m": f"{item.y_m:.4f}",
                "yaw_deg": f"{item.yaw_deg:.2f}",
                "ix": item.ix,
                "iy": item.iy,
                "yaw_bin": item.yaw_bin,
                "current_count": item.current_count,
                "from_sparse_bin": "true" if item.from_sparse_bin else "false",
            }
        )
    path.write_text(buf.getvalue(), encoding="utf-8")


def _fig_to_div(fig: go.Figure) -> str:
    return str(
        fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"responsive": True, "displaylogo": False},
        )
    )


def _escape(text: object) -> str:
    return html.escape(str(text))


def _format_inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    code_placeholders: list[str] = []

    def _replace_code(match: re.Match[str]) -> str:
        token = f"\x00CODE{len(code_placeholders)}\x00"
        code_placeholders.append(f"<code>{match.group(1)}</code>")
        return token

    escaped = _INLINE_CODE_RE.sub(_replace_code, escaped)
    escaped = _BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    escaped = _ITALIC_RE.sub(r"<em>\1</em>", escaped)
    for idx, snippet in enumerate(code_placeholders):
        escaped = escaped.replace(f"\x00CODE{idx}\x00", snippet)
    return escaped


def _split_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def markdown_to_html(md_text: str) -> str:
    """Convert standard Markdown (headings, lists, tables, quotes, code) to safe HTML."""
    lines = md_text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    idx = 0
    n = len(lines)

    while idx < n:
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            idx += 1
            continue

        if stripped.startswith("```"):
            code_lines: list[str] = []
            idx += 1
            while idx < n and not lines[idx].strip().startswith("```"):
                code_lines.append(lines[idx])
                idx += 1
            if idx < n:
                idx += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading_match:
            # Shift heading levels down inside the advice section so they nest under <h2>.
            level = min(6, len(heading_match.group(1)) + 2)
            out.append(f"<h{level}>{_format_inline_markdown(heading_match.group(2))}</h{level}>")
            idx += 1
            continue

        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while idx < n and lines[idx].strip().startswith(">"):
                quote_lines.append(lines[idx].strip().lstrip(">").strip())
                idx += 1
            body = "<br>".join(_format_inline_markdown(q) for q in quote_lines)
            out.append(f"<blockquote>{body}</blockquote>")
            continue

        if "|" in line and idx + 1 < n and _TABLE_SEP_RE.match(lines[idx + 1]):
            headers = _split_markdown_table_row(line)
            idx += 2
            body_rows: list[list[str]] = []
            while idx < n and "|" in lines[idx] and lines[idx].strip():
                body_rows.append(_split_markdown_table_row(lines[idx]))
                idx += 1
            th = "".join(f"<th>{_format_inline_markdown(h)}</th>" for h in headers)
            trs: list[str] = []
            for row in body_rows:
                padded = row + [""] * max(0, len(headers) - len(row))
                tds = "".join(
                    f"<td>{_format_inline_markdown(cell)}</td>" for cell in padded[: len(headers)]
                )
                trs.append(f"<tr>{tds}</tr>")
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>")
            continue

        if _UNORDERED_ITEM_RE.match(line):
            items: list[str] = []
            while idx < n:
                m = _UNORDERED_ITEM_RE.match(lines[idx])
                if not m:
                    break
                items.append(f"<li>{_format_inline_markdown(m.group(1))}</li>")
                idx += 1
            out.append(f"<ul>{''.join(items)}</ul>")
            continue

        if _ORDERED_ITEM_RE.match(line):
            items = []
            while idx < n:
                m = _ORDERED_ITEM_RE.match(lines[idx])
                if not m:
                    break
                items.append(f"<li>{_format_inline_markdown(m.group(1))}</li>")
                idx += 1
            out.append(f"<ol>{''.join(items)}</ol>")
            continue

        para_lines: list[str] = []
        while idx < n and lines[idx].strip():
            cur = lines[idx]
            cur_s = cur.strip()
            if (
                cur_s.startswith("```")
                or cur_s.startswith(">")
                or re.match(r"^#{1,4}\s+", cur_s)
                or _UNORDERED_ITEM_RE.match(cur)
                or _ORDERED_ITEM_RE.match(cur)
                or ("|" in cur and idx + 1 < n and _TABLE_SEP_RE.match(lines[idx + 1]))
            ):
                break
            para_lines.append(cur_s)
            idx += 1
        out.append(f"<p>{_format_inline_markdown(' '.join(para_lines))}</p>")

    return "\n".join(out)


def _render_html_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    numeric_cols: set[int] | None = None,
    scrollable: bool = False,
) -> str:
    num_cols = numeric_cols or set()
    th_cells = "".join(
        f'<th class="{"num" if i in num_cols else ""}">{_escape(h)}</th>'
        for i, h in enumerate(headers)
    )
    body_rows: list[str] = []
    for row in rows:
        td_cells = "".join(
            f'<td class="{"num" if i in num_cols else ""}">{_escape(val)}</td>'
            for i, val in enumerate(row)
        )
        body_rows.append(f"<tr>{td_cells}</tr>")
    table_html = (
        f"<table><thead><tr>{th_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    )
    if scrollable:
        return f'<div class="table-scroll">{table_html}</div>'
    return table_html


def _aggregate_provenance(analyses: Sequence[DatasetAnalysis]) -> ProvenanceCounts:
    return ProvenanceCounts(
        agent=sum(a.provenance.agent for a in analyses),
        human=sum(a.provenance.human for a in analyses),
        converted=sum(a.provenance.converted for a in analyses),
    )


def _numeric_stats(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
    }


def _summarize_speed_metrics(metrics: Sequence[EpisodeSpeedMetric]) -> dict[str, Any]:
    flagged = [m for m in metrics if m.not_slowed_down]
    idle_count = sum(1 for m in metrics if m.idle)
    flagged_eps = sorted({(m.dataset_label, m.episode_index) for m in flagged})

    arms_seen: list[str] = []
    for m in metrics:
        if m.arm not in arms_seen:
            arms_seen.append(m.arm)

    by_arm: dict[str, dict[str, Any]] = {}
    for arm in arms_seen:
        arm_ms = [m for m in metrics if m.arm == arm]
        active = [m for m in arm_ms if not m.idle and m.ratio is not None]
        arm_flagged = [m for m in active if m.not_slowed_down]
        fine_vals = [float(m.fine_median_mps) for m in active if m.fine_median_mps is not None]
        rest_vals = [float(m.rest_median_mps) for m in active if m.rest_median_mps is not None]
        ratios = [float(m.ratio) for m in active if m.ratio is not None]
        by_arm[arm] = {
            "active_episodes": len(active),
            "idle_episodes": sum(1 for m in arm_ms if m.idle),
            "flagged_episodes": len(arm_flagged),
            "median_fine_mps": round(float(np.median(fine_vals)), 6) if fine_vals else None,
            "median_rest_mps": round(float(np.median(rest_vals)), 6) if rest_vals else None,
            "median_ratio": round(float(np.median(ratios)), 6) if ratios else None,
        }

    return {
        "flagged_episode_count": len(flagged_eps),
        "flagged_arm_records": len(flagged),
        "idle_arm_records": idle_count,
        "by_arm": by_arm,
    }


def build_report_summary(
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
    *,
    speed_ratio_threshold: float = DEFAULT_SPEED_RATIO_THRESHOLD,
    min_rest_speed_mps: float = DEFAULT_MIN_REST_SPEED_MPS,
) -> dict[str, Any]:
    """Build the machine-readable ``summary.json`` payload for report and agent advice.

    Args:
        analyses: One or more ``DatasetAnalysis`` objects (datasets or episode groups).
        task_spec: Task specification shared by all inputs.
        speed_ratio_threshold: Speed ratio cutoff used for "not slowed down" checks.
        min_rest_speed_mps: Idle-arm median speed cutoff.

    Returns:
        JSON-serializable dictionary covering provenance, frame shares, fine windows,
        speed checks, object poses, workspace coverage, sparse cells, and pairwise offsets.

    Raises:
        ValueError: If ``analyses`` is empty.
    """
    if not analyses:
        raise ValueError("build_report_summary requires at least one DatasetAnalysis")

    total_prov = _aggregate_provenance(analyses)

    overall_frames = sum(a.frame_shares.total_frames for a in analyses)
    overall_fine = sum(a.frame_shares.total_fine_frames for a in analyses)
    overall_annotated = sum(a.frame_shares.annotated_episodes for a in analyses)
    overall_total_eps = sum(a.frame_shares.total_episodes for a in analyses)
    overall_skipped = sum(len(a.frame_shares.skipped_episodes) for a in analyses)

    overall_subtasks: list[dict[str, Any]] = []
    for subtask_spec in task_spec.subtasks:
        name = subtask_spec.name
        st_eps = 0
        st_frames = 0
        st_fine = 0
        for a in analyses:
            for st in a.frame_shares.subtasks:
                if st.subtask == name:
                    st_eps += st.episodes
                    st_frames += st.frames
                    st_fine += st.fine_frames
        overall_subtasks.append(
            {
                "subtask": name,
                "episodes": st_eps,
                "frames": st_frames,
                "frame_share": round(st_frames / overall_frames, 6) if overall_frames > 0 else 0.0,
                "mean_frames_per_episode": round(st_frames / st_eps, 2) if st_eps > 0 else 0.0,
                "fine_frames": st_fine,
                "fine_share_of_subtask": round(st_fine / st_frames, 6) if st_frames > 0 else 0.0,
                "fine_share_of_total": round(st_fine / overall_frames, 6)
                if overall_frames > 0
                else 0.0,
            }
        )

    groups_payload: list[dict[str, Any]] = []
    for a in analyses:
        fs = a.frame_shares
        group_speed = _summarize_speed_metrics(a.speed_metrics)
        flagged_records = sorted(
            [m for m in a.speed_metrics if m.not_slowed_down],
            key=lambda m: (m.arm, -(m.ratio or 0.0), m.episode_index),
        )
        group_speed["flagged_episodes"] = sorted({m.episode_index for m in flagged_records})
        group_speed["flagged_records"] = [
            {
                "episode_index": m.episode_index,
                "arm": m.arm,
                "fine_median_mps": round(float(m.fine_median_mps), 6)
                if m.fine_median_mps is not None
                else None,
                "rest_median_mps": round(float(m.rest_median_mps), 6)
                if m.rest_median_mps is not None
                else None,
                "ratio": round(float(m.ratio), 6) if m.ratio is not None else None,
            }
            for m in flagged_records
        ]
        groups_payload.append(
            {
                "label": a.label,
                "repo_id": a.repo_id,
                "revision": a.revision,
                "fps": a.fps,
                "arms": list(a.arms),
                "provenance": {
                    "agent": a.provenance.agent,
                    "human": a.provenance.human,
                    "converted": a.provenance.converted,
                },
                "frame_shares": {
                    "annotated_episodes": fs.annotated_episodes,
                    "total_episodes": fs.total_episodes,
                    "skipped_episodes": list(fs.skipped_episodes),
                    "total_frames": fs.total_frames,
                    "total_fine_frames": fs.total_fine_frames,
                    "total_fine_share": round(fs.total_fine_share, 6),
                    "subtasks": [
                        {
                            "subtask": st.subtask,
                            "episodes": st.episodes,
                            "frames": st.frames,
                            "frame_share": round(st.frame_share, 6),
                            "mean_frames_per_episode": round(st.frames / st.episodes, 2)
                            if st.episodes > 0
                            else 0.0,
                            "fine_frames": st.fine_frames,
                            "fine_share_of_subtask": round(st.fine_share_of_subtask, 6),
                            "fine_share_of_total": round(st.fine_share_of_total, 6),
                        }
                        for st in fs.subtasks
                    ],
                },
                "speed_check": group_speed,
                "skipped_objects_count": len(a.skipped_objects),
            }
        )

    all_speed_metrics = [m for a in analyses for m in a.speed_metrics]
    all_poses = [s for a in analyses for s in a.object_poses]
    all_skipped_objects = [s for a in analyses for s in a.skipped_objects]
    coverages = compute_all_coverages(all_poses, task_spec)
    cov_by_name = {cov.spec.name: cov for cov in coverages}

    objects_payload: list[dict[str, Any]] = []
    for obj in task_spec.objects:
        samples = [s for s in all_poses if s.object_name == obj.name]
        skipped_cnt = sum(1 for s in all_skipped_objects if s.object_name == obj.name)
        cov = cov_by_name[obj.name]
        total_cells = len(cov.bins)
        occupied_cells = sum(1 for b in cov.bins if b.count > 0)
        sparse_count = len(cov.sparse_bins)

        by_group: dict[str, Any] = {}
        for a in analyses:
            g_samples = [s for s in a.object_poses if s.object_name == obj.name]
            by_group[a.label] = {
                "sample_count": len(g_samples),
                "x_m": _numeric_stats([s.x for s in g_samples]),
                "y_m": _numeric_stats([s.y for s in g_samples]),
                "folded_yaw_deg": _numeric_stats([s.folded_yaw_deg for s in g_samples]),
            }

        objects_payload.append(
            {
                "name": obj.name,
                "subtask": obj.subtask,
                "event": obj.event,
                "arm": obj.arm,
                "yaw_symmetry_deg": obj.yaw_symmetry_deg,
                "sample_count": len(samples),
                "skipped_count": skipped_cnt,
                "x_m": _numeric_stats([s.x for s in samples]),
                "y_m": _numeric_stats([s.y for s in samples]),
                "z_m": _numeric_stats([s.z for s in samples]),
                "folded_yaw_deg": _numeric_stats([s.folded_yaw_deg for s in samples]),
                "by_group": by_group,
                "coverage": {
                    "total_cells": total_cells,
                    "occupied_cells": occupied_cells,
                    "sparse_cell_count": sparse_count,
                    "sparse_cell_share": round(sparse_count / total_cells, 6)
                    if total_cells > 0
                    else 0.0,
                    "ix_range": list(cov.ix_range) if cov.ix_range is not None else None,
                    "iy_range": list(cov.iy_range) if cov.iy_range is not None else None,
                    "num_yaw_bins": cov.num_yaw_bins,
                    "sparse_cells": [
                        {
                            "ix": b.ix,
                            "iy": b.iy,
                            "yaw_bin": b.yaw_bin,
                            "x_range_m": [round(b.x_min, 4), round(b.x_max, 4)],
                            "y_range_m": [round(b.y_min, 4), round(b.y_max, 4)],
                            "yaw_range_deg": [round(b.yaw_min_deg, 2), round(b.yaw_max_deg, 2)],
                            "count": b.count,
                        }
                        for b in cov.sparse_bins
                    ],
                },
            }
        )

    all_offsets = [o for a in analyses for o in a.pairwise_offsets]
    pairwise_payload: list[dict[str, Any]] = []
    for obj_a, obj_b in combinations(task_spec.objects, 2):
        if obj_a.arm != obj_b.arm:
            continue
        pair = [o for o in all_offsets if o.object_a == obj_a.name and o.object_b == obj_b.name]
        pairwise_payload.append(
            {
                "object_a": obj_a.name,
                "object_b": obj_b.name,
                "arm": obj_a.arm,
                "count": len(pair),
                "dx_m": _numeric_stats([o.dx for o in pair]),
                "dy_m": _numeric_stats([o.dy for o in pair]),
                "dist_xy_m": _numeric_stats([o.dist_xy for o in pair]),
            }
        )

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "task_name": task_spec.name,
        "fine_window_s": task_spec.fine_window_s,
        "speed_ratio_threshold": speed_ratio_threshold,
        "min_rest_speed_mps": min_rest_speed_mps,
        "coverage_config": {
            "xy_bin_m": task_spec.coverage.xy_bin_m,
            "yaw_bin_deg": task_spec.coverage.yaw_bin_deg,
            "sparse_threshold": task_spec.coverage.sparse_threshold,
        },
        "provenance": {
            "total_rows": total_prov.total,
            "counts": {
                "agent": total_prov.agent,
                "human": total_prov.human,
                "converted": total_prov.converted,
            },
            "has_unreviewed_agent": total_prov.has_unreviewed_agent,
        },
        "overall": {
            "annotated_episodes": overall_annotated,
            "total_episodes": overall_total_eps,
            "skipped_episodes_count": overall_skipped,
            "total_frames": overall_frames,
            "total_fine_frames": overall_fine,
            "total_fine_share": round(overall_fine / overall_frames, 6)
            if overall_frames > 0
            else 0.0,
            "subtasks": overall_subtasks,
        },
        "groups": groups_payload,
        "speed_check": _summarize_speed_metrics(all_speed_metrics),
        "objects": objects_payload,
        "pairwise_offsets": pairwise_payload,
    }


def write_report_summary(summary: dict[str, Any], output_path: Path) -> Path:
    """Write ``summary.json`` formatted with UTF-8 encoding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _build_frame_share_figure(
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
) -> go.Figure:
    subtask_names = list(task_spec.subtask_names)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("各 Subtask 帧占比（占总帧数 %）", "精细窗口帧占比（占总帧数 %）"),
    )
    for ds_idx, analysis in enumerate(analyses):
        color = _dataset_color(ds_idx)
        by_name = {st.subtask: st for st in analysis.frame_shares.subtasks}
        shares = [by_name[name].frame_share * 100.0 for name in subtask_names]
        fine_shares = [by_name[name].fine_share_of_total * 100.0 for name in subtask_names]
        frames_text = [f"{by_name[name].frames:,} 帧" for name in subtask_names]
        fine_text = [f"{by_name[name].fine_frames:,} 精细帧" for name in subtask_names]

        fig.add_trace(
            go.Bar(
                name=analysis.label,
                legendgroup=analysis.label,
                x=subtask_names,
                y=shares,
                marker_color=color,
                hovertext=frames_text,
                hovertemplate=(
                    "%{x}<br>帧占比: %{y:.2f}%<br>%{hovertext}<extra>%{fullData.name}</extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name=analysis.label,
                legendgroup=analysis.label,
                showlegend=False,
                x=subtask_names,
                y=fine_shares,
                marker_color=color,
                hovertext=fine_text,
                hovertemplate=(
                    "%{x}<br>精细/总帧: %{y:.2f}%<br>%{hovertext}<extra>%{fullData.name}</extra>"
                ),
            ),
            row=1,
            col=2,
        )
    fig.update_layout(
        barmode="group",
        height=380,
        margin={"t": 50, "b": 50, "l": 55, "r": 25},
        yaxis_title="占比 (%)",
        yaxis2_title="占比 (%)",
    )
    return fig


def _build_speed_arm_figure(
    analyses: Sequence[DatasetAnalysis],
    arm: str,
    speed_ratio_threshold: float,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"机械臂 {arm}：精细窗口 vs 非精细段 TCP 中位速度",
            f"机械臂 {arm}：精细 / 非精细速度比值分布",
        ),
    )
    max_rest = 0.0
    max_fine = 0.0
    for ds_idx, analysis in enumerate(analyses):
        color = _dataset_color(ds_idx)
        arm_metrics = [
            m
            for m in analysis.speed_metrics
            if m.arm == arm and m.rest_median_mps is not None and m.fine_median_mps is not None
        ]
        active = [m for m in arm_metrics if not m.idle and m.ratio is not None]
        if active:
            rest_vals = [float(m.rest_median_mps) for m in active if m.rest_median_mps is not None]
            fine_vals = [float(m.fine_median_mps) for m in active if m.fine_median_mps is not None]
            ratios = [float(m.ratio) for m in active if m.ratio is not None]
            max_rest = max(max_rest, max(rest_vals))
            max_fine = max(max_fine, max(fine_vals))
            symbols = ["x" if m.not_slowed_down else "circle" for m in active]
            hover = [
                f"ep {m.episode_index}<br>精细段={m.fine_median_mps:.4f} m/s<br>"
                f"非精细段={m.rest_median_mps:.4f} m/s<br>比值={m.ratio:.3f}"
                + (" [未放慢]" if m.not_slowed_down else "")
                for m in active
            ]
            fig.add_trace(
                go.Scatter(
                    x=rest_vals,
                    y=fine_vals,
                    mode="markers",
                    name=analysis.label,
                    legendgroup=analysis.label,
                    marker={"color": color, "symbol": symbols, "size": 7, "opacity": 0.8},
                    text=hover,
                    hovertemplate="%{text}<extra>%{fullData.name}</extra>",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Histogram(
                    x=ratios,
                    name=analysis.label,
                    legendgroup=analysis.label,
                    showlegend=False,
                    marker_color=color,
                    opacity=0.65,
                    nbinsx=25,
                ),
                row=1,
                col=2,
            )

    axis_max = max(max_rest, max_fine, 0.05) * 1.05
    fig.add_trace(
        go.Scatter(
            x=[0.0, axis_max],
            y=[0.0, axis_max * speed_ratio_threshold],
            mode="lines",
            name=f"阈值 ({speed_ratio_threshold:.2f}x)",
            line={"color": "#444444", "dash": "dash", "width": 1.5},
        ),
        row=1,
        col=1,
    )
    fig.add_vline(
        x=speed_ratio_threshold,
        line_dash="dash",
        line_color="#d62728",
        row=1,
        col=2,
    )
    fig.update_layout(
        barmode="overlay",
        height=390,
        margin={"t": 50, "b": 50, "l": 60, "r": 25},
        xaxis_title="非精细段中位速度 (m/s)",
        yaxis_title="精细窗口中位速度 (m/s)",
        xaxis2_title="速度比值 (fine / rest)",
        yaxis2_title="Episode 数",
    )
    return fig


def _build_object_pose_figure(
    obj: ObjectSpec,
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
) -> go.Figure:
    yaw_step = task_spec.coverage.yaw_bin_deg
    _, num_yaw_bins = yaw_bin_index(
        0.0,
        yaw_symmetry_deg=obj.yaw_symmetry_deg,
        yaw_bin_deg=yaw_step,
    )
    bin_labels = [
        f"[{i * yaw_step:.0f}, {min(obj.yaw_symmetry_deg, (i + 1) * yaw_step):.0f})°"
        for i in range(num_yaw_bins)
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"{obj.name} XY 位置分布（arm={obj.arm}, event={obj.event}）",
            f"{obj.name} 折叠 Yaw 分布（mod {obj.yaw_symmetry_deg:.0f}°）",
        ),
    )

    for ds_idx, analysis in enumerate(analyses):
        color = _dataset_color(ds_idx)
        samples = [s for s in analysis.object_poses if s.object_name == obj.name]
        if not samples:
            continue
        hover = [
            f"ep {s.episode_index} (f={s.frame})<br>"
            f"xyz=({s.x:.3f}, {s.y:.3f}, {s.z:.3f}) m<br>"
            f"原始 yaw={s.raw_yaw_deg:.1f}°，折叠={s.folded_yaw_deg:.1f}°"
            for s in samples
        ]
        fig.add_trace(
            go.Scatter(
                x=[s.x for s in samples],
                y=[s.y for s in samples],
                mode="markers",
                name=analysis.label,
                legendgroup=analysis.label,
                marker={"color": color, "size": 7, "opacity": 0.78},
                text=hover,
                hovertemplate="%{text}<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=1,
        )

        bin_counts = [0] * num_yaw_bins
        for s in samples:
            idx, _ = yaw_bin_index(
                s.folded_yaw_deg,
                yaw_symmetry_deg=obj.yaw_symmetry_deg,
                yaw_bin_deg=yaw_step,
            )
            bin_counts[idx] += 1

        fig.add_trace(
            go.Bar(
                x=bin_labels,
                y=bin_counts,
                name=analysis.label,
                legendgroup=analysis.label,
                showlegend=False,
                marker_color=color,
                opacity=0.82,
            ),
            row=1,
            col=2,
        )

    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_layout(
        barmode="group",
        height=390,
        margin={"t": 50, "b": 60, "l": 60, "r": 25},
        xaxis_title=f"{obj.arm} 基座 X (m)",
        yaxis_title=f"{obj.arm} 基座 Y (m)",
        xaxis2_title="折叠 Yaw 分箱",
        yaxis2_title="样本数",
    )
    return fig


def _build_pairwise_figure(
    obj_a: ObjectSpec,
    obj_b: ObjectSpec,
    analyses: Sequence[DatasetAnalysis],
) -> go.Figure:
    fig = go.Figure()
    for ds_idx, analysis in enumerate(analyses):
        color = _dataset_color(ds_idx)
        offsets = [
            o
            for o in analysis.pairwise_offsets
            if o.object_a == obj_a.name and o.object_b == obj_b.name
        ]
        if not offsets:
            continue
        hover = [
            f"ep {o.episode_index}<br>dx={o.dx:.3f} m, dy={o.dy:.3f} m, dz={o.dz:.3f} m<br>"
            f"平面距离={o.dist_xy:.3f} m"
            for o in offsets
        ]
        fig.add_trace(
            go.Scatter(
                x=[o.dx for o in offsets],
                y=[o.dy for o in offsets],
                mode="markers",
                name=analysis.label,
                marker={"color": color, "size": 7, "opacity": 0.78},
                text=hover,
                hovertemplate="%{text}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=f"相对位移：{obj_b.name} - {obj_a.name}（arm={obj_a.arm}）",
        xaxis_title="dx (m)",
        yaxis_title="dy (m)",
        height=360,
        margin={"t": 50, "b": 50, "l": 60, "r": 25},
    )
    return fig


def _build_trajectory_3d_figure(
    analyses: Sequence[DatasetAnalysis],
    *,
    subtask: str,
    arm: str,
    grasp_relative: bool,
) -> tuple[go.Figure, int, int]:
    """Build a 3D TCP trajectory plot for one ``(subtask, arm)``."""
    fig = go.Figure()
    total_datasets = len(analyses)
    plotted_eps = 0
    missing_grasp = 0

    for ds_idx, analysis in enumerate(analyses):
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        colors: list[float] = []
        texts: list[str] = []

        segs = [t for t in analysis.trajectories if t.subtask == subtask and t.arm == arm]
        for seg in segs:
            if grasp_relative:
                if seg.grasp_relative_xyz is None or seg.grasp_relative_t_norm is None:
                    missing_grasp += 1
                    continue
                pts = seg.grasp_relative_xyz
                ts = seg.grasp_relative_t_norm
            else:
                pts = seg.xyz
                ts = seg.t_norm

            if len(pts) == 0:
                continue
            plotted_eps += 1
            for pt, t_val in zip(pts, ts, strict=True):
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
                zs.append(float(pt[2]))
                colors.append(float(t_val))
                texts.append(
                    f"{analysis.label} | ep {seg.episode_index} | t={t_val:.2f}<br>"
                    f"({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}) m"
                )
            xs.append(None)
            ys.append(None)
            zs.append(None)
            colors.append(0.0)
            texts.append("")

        if not xs:
            continue

        line_cfg: dict[str, Any] = {
            "color": colors,
            "colorscale": _dataset_colorscale(ds_idx, total_datasets),
            "cmin": 0.0,
            "cmax": 1.0,
            "width": 2.5,
        }
        if total_datasets == 1:
            line_cfg["colorbar"] = {"title": "归一化时间", "len": 0.75}

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=analysis.label,
                line=line_cfg,
                text=texts,
                hovertemplate="%{text}<extra></extra>",
                opacity=0.72,
            )
        )

    if grasp_relative and plotted_eps > 0:
        fig.add_trace(
            go.Scatter3d(
                x=[0.0],
                y=[0.0],
                z=[0.0],
                mode="markers",
                name="抓取原点 (0, 0, 0)",
                marker={"size": 7, "color": "#d62728", "symbol": "diamond"},
                hovertemplate="抓取帧 TCP 原点 (0, 0, 0)<extra></extra>",
            )
        )

    frame_label = "抓取原点相对轨迹 [start, grasp]" if grasp_relative else f"{arm} 基座坐标系"
    fig.update_layout(
        title=f"{subtask} — 机械臂 {arm}（{frame_label}）",
        scene={
            "xaxis_title": "dX (m)" if grasp_relative else "X (m)",
            "yaxis_title": "dY (m)" if grasp_relative else "Y (m)",
            "zaxis_title": "dZ (m)" if grasp_relative else "Z (m)",
            "aspectmode": "data",
        },
        height=470,
        margin={"t": 50, "b": 15, "l": 10, "r": 10},
    )
    return fig, plotted_eps, missing_grasp


def _build_coverage_heatmap(cov: ObjectCoverage) -> go.Figure:
    assert cov.ix_range is not None and cov.iy_range is not None
    ix_min, ix_max = cov.ix_range
    iy_min, iy_max = cov.iy_range
    step = cov.coverage_spec.xy_bin_m

    nx = ix_max - ix_min + 1
    ny = iy_max - iy_min + 1
    grid = np.zeros((ny, nx), dtype=np.int64)
    per_xy_bins: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for b in cov.bins:
        per_xy_bins[(b.ix, b.iy)].append(b)
        grid[b.iy - iy_min, b.ix - ix_min] += b.count

    x_centers = [(ix + 0.5) * step for ix in range(ix_min, ix_max + 1)]
    y_centers = [(iy + 0.5) * step for iy in range(iy_min, iy_max + 1)]

    hover_text: list[list[str]] = []
    for iy in range(iy_min, iy_max + 1):
        row_text: list[str] = []
        for ix in range(ix_min, ix_max + 1):
            cell_bins = sorted(per_xy_bins[(ix, iy)], key=lambda item: item.yaw_bin)
            total = sum(b.count for b in cell_bins)
            sparse_n = sum(1 for b in cell_bins if b.count < cov.coverage_spec.sparse_threshold)
            yaw_detail = "<br>".join(
                f"  [{b.yaw_min_deg:.0f}, {b.yaw_max_deg:.0f})°: {b.count}" for b in cell_bins
            )
            row_text.append(
                f"格子 ({ix}, {iy})<br>"
                f"X: [{ix * step:.3f}, {(ix + 1) * step:.3f}) m<br>"
                f"Y: [{iy * step:.3f}, {(iy + 1) * step:.3f}) m<br>"
                f"总样本: {total} | 稀疏 Yaw 箱: {sparse_n}/{cov.num_yaw_bins}<br>"
                f"{yaw_detail}"
            )
        hover_text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=grid.tolist(),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorscale="YlGnBu",
            colorbar={"title": "样本数"},
        )
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=(
            f"{cov.spec.name} XY 覆盖热力图（格边长={step * 100:.1f} cm，"
            f"Yaw 分箱={cov.coverage_spec.yaw_bin_deg:.0f}°，"
            f"稀疏阈值 &lt; {cov.coverage_spec.sparse_threshold}）"
        ),
        xaxis_title=f"{cov.spec.arm} 基座 X (m)",
        yaxis_title=f"{cov.spec.arm} 基座 Y (m)",
        height=390,
        margin={"t": 50, "b": 50, "l": 60, "r": 25},
    )
    return fig


def _render_provenance_banner(analyses: Sequence[DatasetAnalysis]) -> str:
    prov = _aggregate_provenance(analyses)
    parts = [
        '<div class="provenance-bar">',
        "  <strong>标注来源统计：</strong>",
        f'  <span class="pill">总行数 {prov.total}</span>',
        f'  <span class="pill pill-human">human（人工）{prov.human}</span>',
        f'  <span class="pill pill-converted">converted（转换）{prov.converted}</span>',
        f'  <span class="pill {"pill-agent-warn" if prov.agent > 0 else "pill-agent"}">'
        f"agent（未复核）{prov.agent}</span>",
        "</div>",
    ]
    if prov.has_unreviewed_agent:
        parts.append(
            '<div class="alert-unreviewed">'
            f"<strong>注意：存在 {prov.agent} 条未经人工复核的 <code>agent</code> 标注行。</strong>"
            " 请先在 xense-lerobot-viewer 中完成人工复核，再据此做最终补采决策或 prompt 重标。"
            "</div>"
        )
    return "\n".join(parts)


def _render_advice_section(advice_markdown: str | None) -> str:
    if not advice_markdown or not advice_markdown.strip():
        return ""
    body_html = markdown_to_html(advice_markdown)
    return (
        '<section class="advice-section">\n'
        '  <h2>Agent 建议 <span class="badge-agent">Agent 建议</span></h2>\n'
        '  <p class="note">以下补采与采集建议由 Agent 根据 <code>summary.json</code> 与 '
        "<code>mix</code> / <code>suggest</code> 结果生成，请结合现场实际条件核对后执行。</p>\n"
        f'  <div class="advice-body">{body_html}</div>\n'
        "</section>"
    )


def _render_overview_and_skipped(
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
) -> str:
    parts = ["<section><h2>数据概览与跳过项</h2>"]
    ds_rows = [
        [
            a.label,
            a.repo_id,
            a.revision,
            f"{a.fps:.1f}",
            ", ".join(a.arms),
            f"{a.frame_shares.annotated_episodes}/{a.frame_shares.total_episodes}",
            len(a.frame_shares.skipped_episodes),
            f"{a.frame_shares.total_frames:,}",
            f"{a.provenance.human} / {a.provenance.converted} / {a.provenance.agent}",
        ]
        for a in analyses
    ]
    parts.append(
        _render_html_table(
            (
                "分组 / 数据集",
                "Repo ID",
                "Revision",
                "FPS",
                "机械臂",
                "已标注 / 总 Episode",
                "跳过 Episode",
                "总帧数",
                "标注来源 (human / converted / agent)",
            ),
            ds_rows,
            numeric_cols={3, 5, 6, 7},
        )
    )

    skipped_ep_items: list[str] = []
    for a in analyses:
        if a.frame_shares.skipped_episodes:
            ep_list = ", ".join(str(ep) for ep in a.frame_shares.skipped_episodes)
            skipped_ep_items.append(
                f"<li><strong>{_escape(a.label)}</strong> "
                f"（跳过 {len(a.frame_shares.skipped_episodes)} 条）：{_escape(ep_list)}</li>"
            )
    if skipped_ep_items:
        parts.append("<h3>跳过的 Episode（标注表中未出现）</h3>")
        parts.append(f"<ul>{''.join(skipped_ep_items)}</ul>")
    else:
        parts.append("<p class='note'>输入数据中没有跳过的 Episode。</p>")

    all_skipped_objects = [s for a in analyses for s in a.skipped_objects]
    if all_skipped_objects:
        parts.append(
            f"<h3>跳过的物体事件（{len(all_skipped_objects)}）</h3>"
            "<p class='note'>以下物体因事件帧（如 <code>grasp</code>）为空或所属 subtask "
            "未出现而被跳过，不计入位置分布与覆盖网格。</p>"
        )
        obj_rows = [
            [
                s.dataset_label,
                s.episode_index,
                s.object_name,
                s.subtask,
                s.event,
                s.arm,
                s.reason,
            ]
            for s in all_skipped_objects
        ]
        parts.append(
            _render_html_table(
                ("分组 / 数据集", "Episode", "物体", "Subtask", "事件", "机械臂", "原因"),
                obj_rows,
                numeric_cols={1},
                scrollable=True,
            )
        )
    elif task_spec.objects:
        parts.append("<p class='note'>没有跳过的物体事件。</p>")

    parts.append("</section>")
    return "\n".join(parts)


def _render_frame_share_section(
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
) -> str:
    parts = ["<section><h2>1. 帧占比与精细窗口</h2>"]
    parts.append(
        f"<p class='note'>精细窗口长度 &Delta; = {task_spec.fine_window_s:.2f} s，定义为 "
        "<code>[max(start, contact - &Delta;), grasp] &cup; "
        "[max(start, end - &Delta;), end)</code>"
        "（缺少 <code>contact</code> 或 <code>grasp</code> 时仅取释放尾段）。</p>"
    )
    parts.append(_fig_to_div(_build_frame_share_figure(analyses, task_spec)))

    headers = (
        "分组 / 数据集",
        "Subtask",
        "Episode 数",
        "帧数",
        "帧占比",
        "精细窗口帧数",
        "精细 / Subtask",
        "精细 / 总帧数",
    )
    table_rows: list[list[object]] = []
    for analysis in analyses:
        fs = analysis.frame_shares
        for st in fs.subtasks:
            table_rows.append(
                [
                    analysis.label,
                    st.subtask,
                    st.episodes,
                    f"{st.frames:,}",
                    f"{st.frame_share * 100:.2f}%",
                    f"{st.fine_frames:,}",
                    f"{st.fine_share_of_subtask * 100:.2f}%",
                    f"{st.fine_share_of_total * 100:.2f}%",
                ]
            )
        table_rows.append(
            [
                analysis.label,
                "小计",
                fs.annotated_episodes,
                f"{fs.total_frames:,}",
                "100.00%" if fs.total_frames > 0 else "0.00%",
                f"{fs.total_fine_frames:,}",
                f"{fs.total_fine_share * 100:.2f}%",
                f"{fs.total_fine_share * 100:.2f}%",
            ]
        )

    if len(analyses) > 1:
        total_frames = sum(a.frame_shares.total_frames for a in analyses)
        total_fine = sum(a.frame_shares.total_fine_frames for a in analyses)
        total_eps = sum(a.frame_shares.annotated_episodes for a in analyses)
        for subtask_spec in task_spec.subtasks:
            name = subtask_spec.name
            st_eps = sum(
                st.episodes
                for a in analyses
                for st in a.frame_shares.subtasks
                if st.subtask == name
            )
            st_frames = sum(
                st.frames for a in analyses for st in a.frame_shares.subtasks if st.subtask == name
            )
            st_fine = sum(
                st.fine_frames
                for a in analyses
                for st in a.frame_shares.subtasks
                if st.subtask == name
            )
            table_rows.append(
                [
                    "全部合计",
                    name,
                    st_eps,
                    f"{st_frames:,}",
                    f"{(st_frames / total_frames * 100):.2f}%" if total_frames > 0 else "0.00%",
                    f"{st_fine:,}",
                    f"{(st_fine / st_frames * 100):.2f}%" if st_frames > 0 else "0.00%",
                    f"{(st_fine / total_frames * 100):.2f}%" if total_frames > 0 else "0.00%",
                ]
            )
        table_rows.append(
            [
                "全部合计",
                "总计",
                total_eps,
                f"{total_frames:,}",
                "100.00%" if total_frames > 0 else "0.00%",
                f"{total_fine:,}",
                f"{(total_fine / total_frames * 100):.2f}%" if total_frames > 0 else "0.00%",
                f"{(total_fine / total_frames * 100):.2f}%" if total_frames > 0 else "0.00%",
            ]
        )

    parts.append(_render_html_table(headers, table_rows, numeric_cols={2, 3, 4, 5, 6, 7}))
    parts.append("</section>")
    return "\n".join(parts)


def _render_speed_section(
    analyses: Sequence[DatasetAnalysis],
    *,
    speed_ratio_threshold: float,
    min_rest_speed_mps: float,
) -> str:
    all_arms: list[str] = []
    for analysis in analyses:
        for arm in analysis.arms:
            if arm not in all_arms:
                all_arms.append(arm)

    all_metrics: list[EpisodeSpeedMetric] = [
        m for analysis in analyses for m in analysis.speed_metrics
    ]
    flagged = [m for m in all_metrics if m.not_slowed_down]
    idle_count = sum(1 for m in all_metrics if m.idle)
    flagged_unique_eps = sorted({(m.dataset_label, m.episode_index) for m in flagged})

    parts = ["<section><h2>2. 精细窗口速度检查</h2>"]
    parts.append(
        f"<p class='note'>当 <code>median_speed(fine) / median_speed(rest) &gt; "
        f"{speed_ratio_threshold:.2f}</code> 时，该 episode 被标记为<strong>未放慢</strong>。"
        f"非精细段中位速度低于 <code>{min_rest_speed_mps:.4f} m/s</code> 的机械臂"
        f"（共 {idle_count} 条臂-episode 记录）记为静止臂，不参与未放慢判定。</p>"
    )
    parts.append(
        f"<p><strong>标记为未放慢：</strong>"
        f"{len(flagged_unique_eps)} 条独立 Episode"
        f"（{len(flagged)} 条臂-episode 记录）。</p>"
    )

    for arm in all_arms:
        parts.append(_fig_to_div(_build_speed_arm_figure(analyses, arm, speed_ratio_threshold)))

    if flagged:
        flagged_sorted = sorted(
            flagged,
            key=lambda m: (m.dataset_label, m.arm, -(m.ratio or 0.0), m.episode_index),
        )
        rows = [
            [
                m.dataset_label,
                m.episode_index,
                m.arm,
                f"{m.fine_median_mps:.4f}" if m.fine_median_mps is not None else "-",
                f"{m.rest_median_mps:.4f}" if m.rest_median_mps is not None else "-",
                f"{m.ratio:.3f}" if m.ratio is not None else "-",
            ]
            for m in flagged_sorted
        ]
        parts.append("<h3>未放慢的 Episode 列表</h3>")
        parts.append(
            _render_html_table(
                (
                    "分组 / 数据集",
                    "Episode",
                    "机械臂",
                    "精细段中位速度 (m/s)",
                    "非精细段中位速度 (m/s)",
                    "速度比值",
                ),
                rows,
                numeric_cols={1, 3, 4, 5},
                scrollable=True,
            )
        )
    else:
        parts.append("<p>没有 Episode 超过速度比值阈值。</p>")

    parts.append("</section>")
    return "\n".join(parts)


def _render_objects_section(
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
) -> str:
    parts = ["<section><h2>3. 物体初始位置与相对位移</h2>"]
    if not task_spec.objects:
        parts.append("<p>task spec 中未配置物体定义。</p></section>")
        return "\n".join(parts)

    parts.append(
        "<p class='note'>物体位姿取自对应事件帧的机械臂 TCP 位姿（<code>event=grasp</code> 取 "
        "<code>grasp</code> 帧，<code>event=release</code> 取 <code>end - 1</code> 帧），"
        "在各自机械臂基座坐标系下绘制；两臂无外参标定，不跨臂合并。</p>"
    )

    all_samples: list[ObjectPoseSample] = [s for a in analyses for s in a.object_poses]
    all_skipped: list[SkippedObjectEvent] = [s for a in analyses for s in a.skipped_objects]

    for obj in task_spec.objects:
        obj_samples = [s for s in all_samples if s.object_name == obj.name]
        obj_skipped = [s for s in all_skipped if s.object_name == obj.name]
        parts.append(
            f"<h3>物体：{_escape(obj.name)}"
            f"（subtask={_escape(obj.subtask)}，event={_escape(obj.event)}，"
            f"arm={_escape(obj.arm)}，对称周期={obj.yaw_symmetry_deg:.0f}&deg;）</h3>"
        )
        parts.append(
            f"<p>有效样本数：<strong>{len(obj_samples)}</strong>"
            + (f" ｜ 跳过：<strong>{len(obj_skipped)}</strong>" if obj_skipped else "")
            + "</p>"
        )
        if obj_samples:
            parts.append(_fig_to_div(_build_object_pose_figure(obj, analyses, task_spec)))
        else:
            parts.append(
                "<p class='warn'>该物体暂无可用的事件帧位姿样本（原因见上方跳过列表）。</p>"
            )

    same_arm_pairs = [(a, b) for a, b in combinations(task_spec.objects, 2) if a.arm == b.arm]
    cross_arm_pairs = [(a, b) for a, b in combinations(task_spec.objects, 2) if a.arm != b.arm]
    if same_arm_pairs:
        parts.append("<h3>同臂物体两两相对位移</h3>")
        summary_rows: list[list[object]] = []
        all_offsets: list[PairwiseRelativeOffset] = [
            o for a in analyses for o in a.pairwise_offsets
        ]
        for obj_a, obj_b in same_arm_pairs:
            pair_offsets = [
                o for o in all_offsets if o.object_a == obj_a.name and o.object_b == obj_b.name
            ]
            if not pair_offsets:
                summary_rows.append([f"{obj_b.name} - {obj_a.name}", obj_a.arm, 0, "-", "-", "-"])
                continue
            dxs = np.array([o.dx for o in pair_offsets], dtype=np.float64)
            dys = np.array([o.dy for o in pair_offsets], dtype=np.float64)
            dists = np.array([o.dist_xy for o in pair_offsets], dtype=np.float64)
            summary_rows.append(
                [
                    f"{obj_b.name} - {obj_a.name}",
                    obj_a.arm,
                    len(pair_offsets),
                    f"{np.mean(dxs):.3f} ± {np.std(dxs):.3f}",
                    f"{np.mean(dys):.3f} ± {np.std(dys):.3f}",
                    f"{np.mean(dists):.3f} ± {np.std(dists):.3f}",
                ]
            )
            parts.append(_fig_to_div(_build_pairwise_figure(obj_a, obj_b, analyses)))

        parts.append(
            _render_html_table(
                (
                    "物体对",
                    "机械臂",
                    "Episode 数",
                    "dx 均值 ± 标准差 (m)",
                    "dy 均值 ± 标准差 (m)",
                    "平面距离 均值 ± 标准差 (m)",
                ),
                summary_rows,
                numeric_cols={2, 3, 4, 5},
            )
        )

    if cross_arm_pairs:
        excluded_names = "、".join(f"{b.name} - {a.name}" for a, b in cross_arm_pairs)
        parts.append(
            f"<p class='note'>已略去跨臂物体对（分属不同机械臂基座坐标系）："
            f"{_escape(excluded_names)}。</p>"
        )

    parts.append("</section>")
    return "\n".join(parts)


def _render_trajectories_section(
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
) -> str:
    all_arms: list[str] = []
    for analysis in analyses:
        for arm in analysis.arms:
            if arm not in all_arms:
                all_arms.append(arm)

    parts = ["<section><h2>4. 3D TCP 轨迹</h2>"]
    parts.append(
        "<p class='note'>两条机械臂在各自基座坐标系下分别绘制，曲线深浅按 subtask 内归一化时间 "
        "<code>t &isin; [0, 1]</code> 着色；多组叠加时按组分配色系。</p>"
    )

    all_segs: list[TrajectorySegment] = [t for a in analyses for t in a.trajectories]
    for subtask_spec in task_spec.subtasks:
        subtask = subtask_spec.name
        parts.append(f"<h3>Subtask：{_escape(subtask)}</h3>")
        parts.append('<div class="plot-grid">')
        for arm in all_arms:
            fig, plotted, _ = _build_trajectory_3d_figure(
                analyses, subtask=subtask, arm=arm, grasp_relative=False
            )
            if plotted > 0:
                parts.append(f'<div class="plot-card">{_fig_to_div(fig)}</div>')
            else:
                parts.append(
                    f'<div class="plot-card"><p class="note">'
                    f"{_escape(subtask)} / 机械臂 {_escape(arm)} 暂无轨迹数据。</p></div>"
                )
        parts.append("</div>")

        has_any_grasp = any(
            t.subtask == subtask and t.grasp_relative_xyz is not None for t in all_segs
        )
        if has_any_grasp:
            parts.append(f"<h4>{_escape(subtask)} — 以 grasp 为原点的接近轨迹</h4>")
            parts.append(
                f"<p class='note'><strong>图注：</strong>{_escape(GRASP_RELATIVE_CAPTION)}</p>"
            )
            parts.append('<div class="plot-grid">')
            for arm in all_arms:
                fig, plotted, missing = _build_trajectory_3d_figure(
                    analyses, subtask=subtask, arm=arm, grasp_relative=True
                )
                if plotted > 0:
                    note = (
                        f"<p class='caption'>已绘制 {plotted} 条含 grasp 标注的 Episode"
                        + (f"（另有 {missing} 条缺少 grasp 帧未绘制）。" if missing else "。")
                        + "</p>"
                    )
                    parts.append(f'<div class="plot-card">{_fig_to_div(fig)}{note}</div>')
            parts.append("</div>")

    parts.append("</section>")
    return "\n".join(parts)


def _render_coverage_section(coverages: Sequence[ObjectCoverage]) -> str:
    parts = ["<section><h2>5. 工作空间覆盖网格与稀疏格子</h2>"]
    if not coverages:
        parts.append("<p>task spec 中未配置物体定义。</p></section>")
        return "\n".join(parts)

    for cov in coverages:
        parts.append(
            f"<h3>覆盖统计：{_escape(cov.spec.name)}（arm={_escape(cov.spec.arm)}，"
            f"xy 格边长={cov.coverage_spec.xy_bin_m * 100:.1f} cm，"
            f"yaw 分箱={cov.coverage_spec.yaw_bin_deg:.0f}&deg;，"
            f"稀疏阈值 &lt; {cov.coverage_spec.sparse_threshold}）</h3>"
        )
        if cov.sample_count == 0 or not cov.bins:
            parts.append("<p class='warn'>该物体暂无可用于构建覆盖网格的样本。</p>")
            continue

        total_cells = len(cov.bins)
        sparse_count = len(cov.sparse_bins)
        occupied_cells = sum(1 for b in cov.bins if b.count > 0)
        parts.append(
            f"<p>总样本：<strong>{cov.sample_count}</strong> ｜ "
            f"总格子数 (XY &times; yaw)：<strong>{total_cells}</strong> ｜ "
            f"非空格子：<strong>{occupied_cells}</strong> ｜ "
            f"稀疏格子 (&lt; {cov.coverage_spec.sparse_threshold})："
            f"<strong>{sparse_count}</strong>（{sparse_count / total_cells * 100:.1f}%）</p>"
        )
        parts.append(_fig_to_div(_build_coverage_heatmap(cov)))

        if cov.sparse_bins:
            sparse_rows = [
                [
                    b.ix,
                    b.iy,
                    b.yaw_bin,
                    f"[{b.x_min:.3f}, {b.x_max:.3f})",
                    f"[{b.y_min:.3f}, {b.y_max:.3f})",
                    f"[{b.yaw_min_deg:.0f}, {b.yaw_max_deg:.0f})°",
                    b.count,
                ]
                for b in cov.sparse_bins
            ]
            parts.append(f"<h4>{_escape(cov.spec.name)} 稀疏格子列表（{sparse_count}）</h4>")
            parts.append(
                _render_html_table(
                    ("ix", "iy", "yaw_bin", "X 范围 (m)", "Y 范围 (m)", "Yaw 范围", "样本数"),
                    sparse_rows,
                    numeric_cols={0, 1, 2, 6},
                    scrollable=True,
                )
            )
        else:
            parts.append("<p>全部格子均达到稀疏阈值。</p>")

    parts.append("</section>")
    return "\n".join(parts)


_REPORT_CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Noto Sans CJK SC",
    "Microsoft YaHei", "Segoe UI", Roboto, sans-serif;
  margin: 0;
  padding: 24px 32px;
  color: #1f2937;
  background: #f8fafc;
  line-height: 1.6;
}
header {
  border-bottom: 2px solid #cbd5e1;
  padding-bottom: 14px;
  margin-bottom: 20px;
}
h1 { margin: 0 0 6px 0; font-size: 1.7rem; }
h2 { margin-top: 0; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; font-size: 1.32rem; }
h3 { margin-top: 20px; font-size: 1.08rem; color: #334155; }
h4 { margin-top: 14px; font-size: 0.98rem; color: #475569; }
section {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 20px 24px;
  margin-bottom: 24px;
}
.provenance-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
  font-size: 0.92rem;
}
.pill {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  background: #e2e8f0;
  color: #1e293b;
  font-weight: 500;
}
.pill-human { background: #dcfce7; color: #166534; }
.pill-converted { background: #e0f2fe; color: #075985; }
.pill-agent { background: #f1f5f9; color: #475569; }
.pill-agent-warn { background: #fef3c7; color: #92400e; border: 1px solid #f59e0b; }
.alert-unreviewed {
  margin-top: 12px;
  padding: 12px 16px;
  border-radius: 6px;
  background: #fffbeb;
  border: 1px solid #f59e0b;
  border-left: 5px solid #d97706;
  color: #92400e;
  font-size: 0.95rem;
}
.advice-section {
  border-color: #93c5fd;
  border-left: 5px solid #2563eb;
  background: #f8fbff;
}
.badge-agent {
  display: inline-block;
  margin-left: 8px;
  padding: 2px 9px;
  border-radius: 999px;
  background: #2563eb;
  color: #ffffff;
  font-size: 0.78rem;
  font-weight: 600;
  vertical-align: middle;
}
.advice-body blockquote {
  margin: 10px 0;
  padding: 8px 14px;
  border-left: 3px solid #93c5fd;
  background: #eff6ff;
  color: #1e3a8a;
}
.advice-body pre {
  background: #0f172a;
  color: #e2e8f0;
  padding: 10px 14px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 0.86rem;
}
.advice-body pre code {
  background: transparent;
  color: inherit;
  padding: 0;
}
.note { color: #475569; font-size: 0.92rem; }
.caption { color: #475569; font-size: 0.85rem; margin-top: 4px; }
.warn { color: #b45309; background: #fffbeb; padding: 8px 12px; border-radius: 6px; }
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 0.88rem;
  margin: 12px 0;
}
th, td {
  border: 1px solid #e2e8f0;
  padding: 6px 10px;
  text-align: left;
}
th { background: #f1f5f9; font-weight: 600; position: sticky; top: 0; }
th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; }
.table-scroll { max-height: 320px; overflow-y: auto; border: 1px solid #e2e8f0; }
.table-scroll table { margin: 0; border: none; }
.plot-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
  gap: 16px;
}
.plot-card {
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 8px;
  background: #ffffff;
}
code {
  background: #f1f5f9;
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 0.88em;
}
"""


def render_html_report(
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
    *,
    speed_ratio_threshold: float = DEFAULT_SPEED_RATIO_THRESHOLD,
    min_rest_speed_mps: float = DEFAULT_MIN_REST_SPEED_MPS,
    title: str | None = None,
    advice_markdown: str | None = None,
) -> str:
    """Render a self-contained offline Chinese HTML report with embedded Plotly JS.

    Args:
        analyses: One or more ``DatasetAnalysis`` results to render/overlay.
        task_spec: Task specification shared by the analyzed datasets.
        speed_ratio_threshold: Threshold used to flag "not slowed down" episodes.
        min_rest_speed_mps: Idle-arm median speed cutoff.
        title: Optional page title; defaults to a Chinese title with ``task_spec.name``.
        advice_markdown: Optional Markdown advice text embedded as an "Agent 建议" section.

    Returns:
        Complete HTML document string with inline Plotly JS and no external
        network references.

    Raises:
        ValueError: If ``analyses`` is empty.
    """
    if not analyses:
        raise ValueError("render_html_report requires at least one DatasetAnalysis")

    page_title = title or f"lerobot-check 数据分布分析报告：{task_spec.name or 'dataset'}"
    all_poses = [s for a in analyses for s in a.object_poses]
    coverages = compute_all_coverages(all_poses, task_spec)

    # Embed plotly.js once inline so the HTML opens offline with zero network calls.
    plotly_js = plotly.offline.get_plotlyjs()

    sections = [
        _render_advice_section(advice_markdown),
        _render_overview_and_skipped(analyses, task_spec),
        _render_frame_share_section(analyses, task_spec),
        _render_speed_section(
            analyses,
            speed_ratio_threshold=speed_ratio_threshold,
            min_rest_speed_mps=min_rest_speed_mps,
        ),
        _render_objects_section(analyses, task_spec),
        _render_trajectories_section(analyses, task_spec),
        _render_coverage_section(coverages),
    ]
    non_empty_sections = [s for s in sections if s]

    return (
        "<!DOCTYPE html>\n"
        '<html lang="zh-CN">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{_escape(page_title)}</title>\n"
        f"  <style>{_REPORT_CSS}</style>\n"
        f'  <script type="text/javascript">{plotly_js}</script>\n'
        "</head>\n"
        "<body>\n"
        "  <header>\n"
        f"    <h1>{_escape(page_title)}</h1>\n"
        f"    <p class='note'>任务：<strong>{_escape(task_spec.name or '(未命名)')}</strong> ｜ "
        f"Subtasks：{_escape(', '.join(task_spec.subtask_names))} ｜ "
        f"分组 / 数据集数：{len(analyses)}</p>\n"
        f"    {_render_provenance_banner(analyses)}\n"
        "  </header>\n" + "\n".join(non_empty_sections) + "\n</body>\n</html>\n"
    )


def write_html_report(
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
    output_path: Path,
    *,
    speed_ratio_threshold: float = DEFAULT_SPEED_RATIO_THRESHOLD,
    min_rest_speed_mps: float = DEFAULT_MIN_REST_SPEED_MPS,
    title: str | None = None,
    advice_markdown: str | None = None,
) -> Path:
    """Render and write a self-contained offline HTML report to ``output_path``."""
    html_text = render_html_report(
        analyses,
        task_spec,
        speed_ratio_threshold=speed_ratio_threshold,
        min_rest_speed_mps=min_rest_speed_mps,
        title=title,
        advice_markdown=advice_markdown,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def resolve_report_output_paths(output_path: Path) -> tuple[Path, Path]:
    """Resolve ``(html_path, summary_json_path)`` from an HTML file or directory path."""
    if output_path.suffix.lower() in {".html", ".htm"}:
        return output_path, output_path.parent / "summary.json"
    return output_path / "index.html", output_path / "summary.json"


def write_report_bundle(
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
    output_path: Path,
    *,
    speed_ratio_threshold: float = DEFAULT_SPEED_RATIO_THRESHOLD,
    min_rest_speed_mps: float = DEFAULT_MIN_REST_SPEED_MPS,
    title: str | None = None,
    advice_markdown: str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write both the offline HTML report and ``summary.json`` into the target location.

    When ``output_path`` is a directory (no ``.html``/``.htm`` suffix), writes
    ``index.html`` and ``summary.json`` inside it. When ``output_path`` ends
    with ``.html``/``.htm``, writes the HTML there and ``summary.json`` next to it.

    Returns:
        Tuple ``(html_path, summary_path, summary_dict)``.
    """
    html_path, summary_path = resolve_report_output_paths(output_path)
    summary = build_report_summary(
        analyses,
        task_spec,
        speed_ratio_threshold=speed_ratio_threshold,
        min_rest_speed_mps=min_rest_speed_mps,
    )
    write_html_report(
        analyses,
        task_spec,
        html_path,
        speed_ratio_threshold=speed_ratio_threshold,
        min_rest_speed_mps=min_rest_speed_mps,
        title=title,
        advice_markdown=advice_markdown,
    )
    write_report_summary(summary, summary_path)
    return html_path, summary_path, summary
