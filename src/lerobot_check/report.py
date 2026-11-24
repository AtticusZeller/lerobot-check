"""Offline static HTML report generation and CLI text/CSV formatters."""

from __future__ import annotations

import csv
import html
import io
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
    SkippedObjectEvent,
    TrajectorySegment,
    compute_all_coverages,
    yaw_bin_index,
)
from lerobot_check.schema import ObjectSpec, TaskSpec

COLLISION_DISCLAIMER = (
    "Note: Objects are sampled independently from sparse coverage bins; "
    "inter-object collisions are not checked or resolved."
)

GRASP_RELATIVE_CAPTION = (
    "Grasp-relative trajectories shift each episode's [start, grasp] segment so the "
    "grasp-frame TCP lies at (0, 0, 0). Removing workspace translation means this view "
    "only reflects approach-direction diversity (not absolute object position coverage)."
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

# Distinct sequential colorscales for multi-dataset 3D trajectory overlays so each
# dataset keeps a distinct hue while shade still encodes normalized time t in [0, 1].
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
        lines.append("Omitted objects (no observed samples): " + ", ".join(sorted(omitted_objects)))
    if not suggestions:
        lines.append("No placement suggestions available.")
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
    """Write placement suggestions to CSV with a comment header noting no collision check."""
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


def _build_frame_share_figure(
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
) -> go.Figure:
    subtask_names = list(task_spec.subtask_names)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Subtask Frame Share (% of total)", "Fine-Window Share (% of total)"),
    )
    for ds_idx, analysis in enumerate(analyses):
        color = _dataset_color(ds_idx)
        by_name = {st.subtask: st for st in analysis.frame_shares.subtasks}
        shares = [by_name[name].frame_share * 100.0 for name in subtask_names]
        fine_shares = [by_name[name].fine_share_of_total * 100.0 for name in subtask_names]
        frames_text = [f"{by_name[name].frames:,} frames" for name in subtask_names]
        fine_text = [f"{by_name[name].fine_frames:,} fine frames" for name in subtask_names]

        fig.add_trace(
            go.Bar(
                name=analysis.label,
                legendgroup=analysis.label,
                x=subtask_names,
                y=shares,
                marker_color=color,
                hovertext=frames_text,
                hovertemplate=(
                    "%{x}<br>Share: %{y:.2f}%<br>%{hovertext}<extra>%{fullData.name}</extra>"
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
                    "%{x}<br>Fine/Total: %{y:.2f}%<br>%{hovertext}<extra>%{fullData.name}</extra>"
                ),
            ),
            row=1,
            col=2,
        )
    fig.update_layout(
        barmode="group",
        height=380,
        margin={"t": 50, "b": 50, "l": 55, "r": 25},
        yaxis_title="Share (%)",
        yaxis2_title="Share (%)",
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
            f"Arm '{arm}': Fine vs Rest Median TCP Speed",
            f"Arm '{arm}': Fine / Rest Speed Ratio Distribution",
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
                f"ep {m.episode_index}<br>fine={m.fine_median_mps:.4f} m/s<br>"
                f"rest={m.rest_median_mps:.4f} m/s<br>ratio={m.ratio:.3f}"
                + (" [NOT SLOWED]" if m.not_slowed_down else "")
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
            name=f"threshold ({speed_ratio_threshold:.2f}x)",
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
        xaxis_title="Non-fine median speed (m/s)",
        yaxis_title="Fine-window median speed (m/s)",
        xaxis2_title="Ratio (fine / rest)",
        yaxis2_title="Episodes",
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
            f"{obj.name} XY Position (arm={obj.arm}, event={obj.event})",
            f"{obj.name} Folded Yaw (mod {obj.yaw_symmetry_deg:.0f}°)",
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
            f"raw_yaw={s.raw_yaw_deg:.1f}°, folded={s.folded_yaw_deg:.1f}°"
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
        xaxis_title=f"{obj.arm} base X (m)",
        yaxis_title=f"{obj.arm} base Y (m)",
        xaxis2_title="Folded yaw bin",
        yaxis2_title="Count",
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
            f"dist_xy={o.dist_xy:.3f} m"
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
        title=f"Relative offset: {obj_b.name} - {obj_a.name} (arm={obj_a.arm})",
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
    """Build a 3D TCP trajectory plot for one ``(subtask, arm)``.

    Returns ``(figure, plotted_episodes, missing_grasp_episodes)``.
    """
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
            # Break line segments between episodes while keeping numeric color entries.
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
            line_cfg["colorbar"] = {"title": "Norm. time", "len": 0.75}

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
                name="Grasp origin (0, 0, 0)",
                marker={"size": 7, "color": "#d62728", "symbol": "diamond"},
                hovertemplate="Grasp TCP origin (0, 0, 0)<extra></extra>",
            )
        )

    frame_label = "Grasp-relative [start, grasp]" if grasp_relative else f"{arm} base frame"
    fig.update_layout(
        title=f"{subtask} — arm '{arm}' ({frame_label})",
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
                f"Cell ({ix}, {iy})<br>"
                f"X: [{ix * step:.3f}, {(ix + 1) * step:.3f}) m<br>"
                f"Y: [{iy * step:.3f}, {(iy + 1) * step:.3f}) m<br>"
                f"Total: {total} | Sparse yaw bins: {sparse_n}/{cov.num_yaw_bins}<br>"
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
            colorbar={"title": "Samples"},
        )
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=(
            f"{cov.spec.name} XY Coverage Grid (cell={step * 100:.1f} cm, "
            f"yaw bin={cov.coverage_spec.yaw_bin_deg:.0f}°, "
            f"sparse < {cov.coverage_spec.sparse_threshold})"
        ),
        xaxis_title=f"{cov.spec.arm} base X (m)",
        yaxis_title=f"{cov.spec.arm} base Y (m)",
        height=390,
        margin={"t": 50, "b": 50, "l": 60, "r": 25},
    )
    return fig


def _render_frame_share_section(
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
) -> str:
    parts = ["<section><h2>1. Frame Shares &amp; Fine Windows</h2>"]
    parts.append(
        f"<p class='note'>Fine window duration &Delta; = {task_spec.fine_window_s:.2f} s: "
        "<code>[max(start, contact - &Delta;), grasp] &cup; "
        "[max(start, end - &Delta;), end)</code> "
        "(release tail only when <code>contact</code> or <code>grasp</code> is missing).</p>"
    )
    parts.append(_fig_to_div(_build_frame_share_figure(analyses, task_spec)))

    headers = (
        "Dataset",
        "Subtask",
        "Episodes",
        "Frames",
        "Share",
        "Fine Frames",
        "Fine / Subtask",
        "Fine / Total",
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
                "TOTAL",
                fs.annotated_episodes,
                f"{fs.total_frames:,}",
                "100.00%" if fs.total_frames > 0 else "0.00%",
                f"{fs.total_fine_frames:,}",
                f"{fs.total_fine_share * 100:.2f}%",
                f"{fs.total_fine_share * 100:.2f}%",
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

    parts = ["<section><h2>2. Speed Check (Fine Window vs Rest)</h2>"]
    parts.append(
        f"<p class='note'>Episodes where <code>median_speed(fine) / median_speed(rest) &gt; "
        f"{speed_ratio_threshold:.2f}</code> are flagged as <strong>not slowed down</strong>. "
        f"Arm-episodes with <code>rest_median &lt; {min_rest_speed_mps:.4f} m/s</code> "
        f"({idle_count} arm-episode(s)) are marked idle and excluded from flagging.</p>"
    )
    parts.append(
        f"<p><strong>Flagged as not slowed down:</strong> "
        f"{len(flagged_unique_eps)} unique episode(s) "
        f"({len(flagged)} arm-episode record(s)).</p>"
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
        parts.append("<h3>Episodes Flagged as Not Slowed Down</h3>")
        parts.append(
            _render_html_table(
                ("Dataset", "Episode", "Arm", "Fine Median (m/s)", "Rest Median (m/s)", "Ratio"),
                rows,
                numeric_cols={1, 3, 4, 5},
                scrollable=True,
            )
        )
    else:
        parts.append("<p>No episodes exceeded the speed ratio threshold.</p>")

    parts.append("</section>")
    return "\n".join(parts)


def _render_objects_section(
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
) -> str:
    parts = ["<section><h2>3. Object Initial Positions &amp; Pairwise Relative Offsets</h2>"]
    if not task_spec.objects:
        parts.append("<p>No objects defined in task spec.</p></section>")
        return "\n".join(parts)

    parts.append(
        "<p class='note'>Poses are taken from the event frame (<code>grasp</code> frame for "
        "<code>event=grasp</code>, <code>end - 1</code> for <code>event=release</code>) in each "
        "arm's own base frame. Arms are plotted separately without assuming inter-arm "
        "extrinsics.</p>"
    )

    all_samples: list[ObjectPoseSample] = [s for a in analyses for s in a.object_poses]
    all_skipped: list[SkippedObjectEvent] = [s for a in analyses for s in a.skipped_objects]

    for obj in task_spec.objects:
        obj_samples = [s for s in all_samples if s.object_name == obj.name]
        obj_skipped = [s for s in all_skipped if s.object_name == obj.name]
        parts.append(
            f"<h3>Object: {_escape(obj.name)} "
            f"(subtask={_escape(obj.subtask)}, event={_escape(obj.event)}, "
            f"arm={_escape(obj.arm)}, symmetry={obj.yaw_symmetry_deg:.0f}&deg;)</h3>"
        )
        parts.append(
            f"<p>Extracted samples: <strong>{len(obj_samples)}</strong>"
            + (f" | Skipped: <strong>{len(obj_skipped)}</strong>" if obj_skipped else "")
            + "</p>"
        )
        if obj_samples:
            parts.append(_fig_to_div(_build_object_pose_figure(obj, analyses, task_spec)))
        else:
            parts.append(
                "<p class='warn'>No event-frame poses available for this object "
                "(see skipped list).</p>"
            )

    same_arm_pairs = [(a, b) for a, b in combinations(task_spec.objects, 2) if a.arm == b.arm]
    cross_arm_pairs = [(a, b) for a, b in combinations(task_spec.objects, 2) if a.arm != b.arm]
    if same_arm_pairs:
        parts.append("<h3>Pairwise Relative Displacements (Same-Arm Base Frame)</h3>")
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
                    "Pair",
                    "Arm",
                    "Episodes",
                    "dx mean ± std (m)",
                    "dy mean ± std (m)",
                    "dist_xy mean ± std (m)",
                ),
                summary_rows,
                numeric_cols={2, 3, 4, 5},
            )
        )

    if cross_arm_pairs:
        excluded_names = ", ".join(f"{b.name}-{a.name}" for a, b in cross_arm_pairs)
        parts.append(
            f"<p class='note'>Cross-arm object pairs excluded (separate arm base frames): "
            f"{_escape(excluded_names)}.</p>"
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

    parts = ["<section><h2>4. 3D TCP Trajectories</h2>"]
    parts.append(
        "<p class='note'>Each arm is plotted separately in its own base frame, colored by "
        "normalized time <code>t &isin; [0, 1]</code> within the subtask.</p>"
    )

    all_segs: list[TrajectorySegment] = [t for a in analyses for t in a.trajectories]
    for subtask_spec in task_spec.subtasks:
        subtask = subtask_spec.name
        parts.append(f"<h3>Subtask: {_escape(subtask)}</h3>")
        parts.append('<div class="plot-grid">')
        for arm in all_arms:
            fig, plotted, _ = _build_trajectory_3d_figure(
                analyses, subtask=subtask, arm=arm, grasp_relative=False
            )
            if plotted > 0:
                parts.append(f'<div class="plot-card">{_fig_to_div(fig)}</div>')
            else:
                parts.append(
                    f'<div class="plot-card"><p class="note">No trajectories for '
                    f"{_escape(subtask)} / arm {_escape(arm)}.</p></div>"
                )
        parts.append("</div>")

        has_any_grasp = any(
            t.subtask == subtask and t.grasp_relative_xyz is not None for t in all_segs
        )
        if has_any_grasp:
            parts.append(f"<h4>{_escape(subtask)} — Grasp-Relative Approach Trajectories</h4>")
            parts.append(
                f"<p class='note'><strong>Note:</strong> {_escape(GRASP_RELATIVE_CAPTION)}</p>"
            )
            parts.append('<div class="plot-grid">')
            for arm in all_arms:
                fig, plotted, missing = _build_trajectory_3d_figure(
                    analyses, subtask=subtask, arm=arm, grasp_relative=True
                )
                if plotted > 0:
                    note = (
                        f"<p class='caption'>Plotted {plotted} episode(s) with grasp"
                        + (
                            f" ({missing} episode(s) omitted due to missing grasp)."
                            if missing
                            else "."
                        )
                        + "</p>"
                    )
                    parts.append(f'<div class="plot-card">{_fig_to_div(fig)}{note}</div>')
            parts.append("</div>")

    parts.append("</section>")
    return "\n".join(parts)


def _render_coverage_section(coverages: Sequence[ObjectCoverage]) -> str:
    parts = ["<section><h2>5. Workspace Coverage Grid &amp; Sparse Cells</h2>"]
    if not coverages:
        parts.append("<p>No objects defined in task spec.</p></section>")
        return "\n".join(parts)

    for cov in coverages:
        parts.append(
            f"<h3>Coverage: {_escape(cov.spec.name)} (arm={_escape(cov.spec.arm)}, "
            f"xy_bin={cov.coverage_spec.xy_bin_m * 100:.1f} cm, "
            f"yaw_bin={cov.coverage_spec.yaw_bin_deg:.0f}&deg;, "
            f"sparse &lt; {cov.coverage_spec.sparse_threshold})</h3>"
        )
        if cov.sample_count == 0 or not cov.bins:
            parts.append("<p class='warn'>No samples available for coverage grid.</p>")
            continue

        total_cells = len(cov.bins)
        sparse_count = len(cov.sparse_bins)
        occupied_cells = sum(1 for b in cov.bins if b.count > 0)
        parts.append(
            f"<p>Samples: <strong>{cov.sample_count}</strong> | "
            f"Grid cells (XY &times; yaw): <strong>{total_cells}</strong> | "
            f"Non-empty cells: <strong>{occupied_cells}</strong> | "
            f"Sparse cells (&lt; {cov.coverage_spec.sparse_threshold}): "
            f"<strong>{sparse_count}</strong> ({sparse_count / total_cells * 100:.1f}%)</p>"
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
            parts.append(f"<h4>Sparse Cells for {_escape(cov.spec.name)} ({sparse_count})</h4>")
            parts.append(
                _render_html_table(
                    ("ix", "iy", "yaw_bin", "X range (m)", "Y range (m)", "Yaw range", "Count"),
                    sparse_rows,
                    numeric_cols={0, 1, 2, 6},
                    scrollable=True,
                )
            )
        else:
            parts.append("<p>All grid cells meet the sparse threshold.</p>")

    parts.append("</section>")
    return "\n".join(parts)


def _render_overview_and_skipped(
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
) -> str:
    parts = ["<section><h2>Overview &amp; Skipped Items</h2>"]
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
        ]
        for a in analyses
    ]
    parts.append(
        _render_html_table(
            (
                "Label",
                "Repo ID",
                "Revision",
                "FPS",
                "Arms",
                "Annotated / Total Ep",
                "Skipped Ep",
                "Total Frames",
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
                f"({len(a.frame_shares.skipped_episodes)} skipped): {_escape(ep_list)}</li>"
            )
    if skipped_ep_items:
        parts.append("<h3>Skipped Episodes (Absent from Annotation Table)</h3>")
        parts.append(f"<ul>{''.join(skipped_ep_items)}</ul>")
    else:
        parts.append("<p class='note'>No episodes were skipped across input datasets.</p>")

    all_skipped_objects = [s for a in analyses for s in a.skipped_objects]
    if all_skipped_objects:
        parts.append(
            f"<h3>Skipped Object Events ({len(all_skipped_objects)})</h3>"
            "<p class='note'>Objects whose event frame (e.g. <code>grasp</code>) or subtask was "
            "missing are listed below and excluded from pose/coverage calculations.</p>"
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
                ("Dataset", "Episode", "Object", "Subtask", "Event", "Arm", "Reason"),
                obj_rows,
                numeric_cols={1},
                scrollable=True,
            )
        )
    elif task_spec.objects:
        parts.append("<p class='note'>No object events were skipped.</p>")

    parts.append("</section>")
    return "\n".join(parts)


_REPORT_CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0;
  padding: 24px 32px;
  color: #1f2937;
  background: #f8fafc;
  line-height: 1.5;
}
header {
  border-bottom: 2px solid #cbd5e1;
  padding-bottom: 12px;
  margin-bottom: 24px;
}
h1 { margin: 0 0 6px 0; font-size: 1.75rem; }
h2 { margin-top: 0; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; font-size: 1.35rem; }
h3 { margin-top: 20px; font-size: 1.08rem; color: #334155; }
h4 { margin-top: 14px; font-size: 0.98rem; color: #475569; }
section {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 20px 24px;
  margin-bottom: 24px;
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
) -> str:
    """Render a self-contained offline HTML report with embedded Plotly JS.

    Args:
        analyses: One or more ``DatasetAnalysis`` results to render/overlay.
        task_spec: Task specification shared by the analyzed datasets.
        speed_ratio_threshold: Threshold used to flag "not slowed down" episodes.
        min_rest_speed_mps: Idle-arm median speed cutoff.
        title: Optional page title; defaults to ``f"lerobot-check Report: {task_spec.name}"``.

    Returns:
        Complete HTML document string with inline Plotly JS and no external
        network references.

    Raises:
        ValueError: If ``analyses`` is empty.
    """
    if not analyses:
        raise ValueError("render_html_report requires at least one DatasetAnalysis")

    page_title = title or f"lerobot-check Report: {task_spec.name or 'dataset'}"
    all_poses = [s for a in analyses for s in a.object_poses]
    coverages = compute_all_coverages(all_poses, task_spec)

    # Embed plotly.js once inline so the HTML opens offline with zero network calls.
    plotly_js = plotly.offline.get_plotlyjs()

    sections = [
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

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
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
        f"    <p class='note'>Task: <strong>{_escape(task_spec.name or '(unnamed)')}</strong> | "
        f"Subtasks: {_escape(', '.join(task_spec.subtask_names))} | "
        f"Datasets: {len(analyses)}</p>\n"
        "  </header>\n" + "\n".join(sections) + "\n</body>\n</html>\n"
    )


def write_html_report(
    analyses: Sequence[DatasetAnalysis],
    task_spec: TaskSpec,
    output_path: Path,
    *,
    speed_ratio_threshold: float = DEFAULT_SPEED_RATIO_THRESHOLD,
    min_rest_speed_mps: float = DEFAULT_MIN_REST_SPEED_MPS,
    title: str | None = None,
) -> Path:
    """Render and write a self-contained offline HTML report to ``output_path``."""
    html_text = render_html_report(
        analyses,
        task_spec,
        speed_ratio_threshold=speed_ratio_threshold,
        min_rest_speed_mps=min_rest_speed_mps,
        title=title,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path
