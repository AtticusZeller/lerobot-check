"""Command-line interface for ``lerobot-check``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lerobot_check.analysis import (
    DEFAULT_MAX_TRAJ_POINTS,
    DEFAULT_MIN_REST_SPEED_MPS,
    DEFAULT_SPEED_RATIO_THRESHOLD,
    MixEntry,
    analyze_dataset,
    analyze_dataset_groups,
    compute_all_coverages,
    compute_mix,
    parse_episode_groups,
    suggest_placements,
)
from lerobot_check.dataset import (
    DatasetView,
    default_annotations_path,
    default_report_dir,
    default_task_spec_path,
    detect_dataset_revision,
    format_dataset_info,
    infer_repo_id,
    resolve_dataset_root,
)
from lerobot_check.propose import (
    DEFAULT_RENDER_EVENT_TYPES,
    ProposeConfig,
    format_proposals_csv,
    load_proposal_event_frames,
    propose_episode,
    write_proposals_csv,
)
from lerobot_check.relabel import (
    RelabelValidationError,
    relabel_dataset,
    verify_relabel,
)
from lerobot_check.render import (
    parse_frame_spec,
    render_episode_sheets,
    resolve_render_cameras,
    select_frames_around_events,
)
from lerobot_check.report import (
    format_frame_share_table,
    format_mix_report,
    format_suggestions_table,
    write_report_bundle,
    write_suggestions_csv,
)
from lerobot_check.schema import (
    AnnotationRow,
    TaskSpec,
    load_annotations,
    load_task_spec,
)
from lerobot_check.validate import validate_annotations


def _parse_episode_list(values: Sequence[str] | None, total_episodes: int) -> list[int]:
    if not values:
        return list(range(total_episodes))
    episodes: set[int] = set()
    for raw in values:
        for token in (part.strip() for part in raw.split(",")):
            if not token:
                continue
            if ":" in token:
                parts = token.split(":")
                if len(parts) != 2:
                    raise ValueError(f"Invalid episode range: {token!r}")
                start, end = int(parts[0]), int(parts[1])
                episodes.update(range(start, end))
            else:
                episodes.add(int(token))
    return sorted(episodes)


def _parse_event_types(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_RENDER_EVENT_TYPES
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not items:
        return DEFAULT_RENDER_EVENT_TYPES
    return items


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to a local LeRobot v3.0 dataset directory.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "Dataset repo ID (e.g. Xense/bi_elite_cs66_rt_egg_carton_to_tray_1016_fixed). "
            "Used to resolve local cache path if --dataset-root is omitted."
        ),
    )


def _cmd_info(args: argparse.Namespace) -> int:
    try:
        root = resolve_dataset_root(args.dataset_root, args.repo_id)
        dataset = DatasetView(root, repo_id=args.repo_id, revision=args.revision)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(format_dataset_info(dataset))
    return 0


def _cmd_propose(args: argparse.Namespace) -> int:
    root = resolve_dataset_root(args.dataset_root, args.repo_id)
    dataset = DatasetView(root, repo_id=args.repo_id)
    episodes = _parse_episode_list(args.episodes, dataset.total_episodes)
    config = ProposeConfig(
        gripper_delta=args.gripper_delta,
        squeeze_gap=args.squeeze_gap,
        min_squeeze_s=args.min_squeeze_s,
        low_speed_threshold_mps=args.low_speed_threshold,
        min_low_speed_s=args.min_low_speed_s,
    )
    proposals = [propose_episode(dataset, ep, config=config) for ep in episodes]
    if args.output is not None:
        write_proposals_csv(proposals, args.output)
        total_gripper = sum(len(p.gripper_events) for p in proposals)
        total_squeeze = sum(len(p.squeeze_segments) for p in proposals)
        total_slow = sum(len(p.low_speed_segments) for p in proposals)
        print(
            f"Wrote {args.output}: {len(proposals)} episode(s), "
            f"{total_gripper} gripper motion(s), {total_squeeze} squeeze interval(s), "
            f"{total_slow} low-speed segment(s)."
        )
    else:
        sys.stdout.write(format_proposals_csv(proposals))
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    root = resolve_dataset_root(args.dataset_root, args.repo_id)
    dataset = DatasetView(root, repo_id=args.repo_id)
    record = dataset.get_episode(args.episode)

    task_spec: TaskSpec | None = None
    if args.subtask is not None and not args.cameras:
        spec_path = (
            args.task_spec
            if args.task_spec is not None
            else default_task_spec_path(root, repo_id=dataset.repo_id)
        )
        try:
            task_spec = load_task_spec(spec_path)
        except (OSError, ValueError) as exc:
            print(f"[task-spec] {exc}", file=sys.stderr)
            return 1

    try:
        cameras = resolve_render_cameras(
            explicit_cameras=args.cameras,
            subtask=args.subtask,
            task_spec=task_spec,
            columns=args.columns,
        )
    except (KeyError, ValueError) as exc:
        print(f"[cameras] {exc}", file=sys.stderr)
        return 1

    highlight_frames: list[int] = []
    if args.frames is not None:
        frame_indices = parse_frame_spec(args.frames, record.length)
    else:
        event_types = _parse_event_types(args.event_types)
        event_frames: list[int] = []
        if args.events is not None:
            event_frames.extend(parse_frame_spec(args.events, record.length))
        if args.proposals is not None:
            event_frames.extend(
                load_proposal_event_frames(args.proposals, args.episode, event_types=event_types)
            )
        if not event_frames:
            proposal = propose_episode(
                dataset,
                args.episode,
                config=ProposeConfig(
                    gripper_delta=args.gripper_delta,
                    squeeze_gap=args.squeeze_gap,
                    min_squeeze_s=args.min_squeeze_s,
                ),
            )
            if "gripper_close" in event_types:
                event_frames.extend(
                    ev.frame for ev in proposal.gripper_events if ev.kind == "close"
                )
            if "gripper_open" in event_types:
                event_frames.extend(ev.frame for ev in proposal.gripper_events if ev.kind == "open")
            if "squeeze" in event_types:
                event_frames.extend(seg.start for seg in proposal.squeeze_segments)
        if not event_frames:
            print(
                f"Episode {args.episode} has no candidate events matching {event_types}; "
                "pass --frames or --events explicitly.",
                file=sys.stderr,
            )
            return 1
        highlight_frames = sorted(set(event_frames))
        window_frames = (
            args.window_frames
            if args.window_frames is not None
            else round(args.window_s * dataset.fps)
        )
        frame_indices = select_frames_around_events(
            highlight_frames,
            episode_length=record.length,
            window_frames=window_frames,
            stride=args.stride,
        )

    if args.max_frames_per_image is not None:
        max_frames_per_image = args.max_frames_per_image
    elif args.columns is not None:
        max_frames_per_image = args.columns * 8
    else:
        max_frames_per_image = 12

    written = render_episode_sheets(
        dataset,
        args.episode,
        frame_indices,
        args.output,
        cameras=cameras,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        max_frames_per_image=max_frames_per_image,
        highlight_frames=highlight_frames,
        columns=args.columns,
    )
    for path in written:
        print(f"Wrote {path} ({len(frame_indices)} total frames across {len(written)} sheet(s))")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        root = resolve_dataset_root(args.dataset_root, args.repo_id)
        dataset = DatasetView(root, repo_id=args.repo_id, revision=args.revision)
    except (OSError, ValueError) as exc:
        print(f"[dataset] {exc}", file=sys.stderr)
        return 1

    spec_path = (
        args.task_spec
        if args.task_spec is not None
        else default_task_spec_path(root, repo_id=dataset.repo_id)
    )
    ann_path = (
        args.annotations
        if args.annotations is not None
        else default_annotations_path(root, repo_id=dataset.repo_id, revision=dataset.revision)
    )

    try:
        task_spec = load_task_spec(spec_path)
    except (OSError, ValueError) as exc:
        print(f"[task-spec] {exc}", file=sys.stderr)
        return 1

    try:
        rows = load_annotations(ann_path)
    except (OSError, ValueError) as exc:
        print(f"[annotations] {exc}", file=sys.stderr)
        return 1

    expected_revision: str | None = None
    if dataset.revision == "local":
        print(
            "Warning: dataset revision could not be detected ('local'); "
            "annotation revision was not verified.",
            file=sys.stderr,
        )
    else:
        expected_revision = dataset.revision

    violations = validate_annotations(
        rows,
        task_spec,
        dataset.episode_lengths,
        expected_repo_id=args.repo_id,
        expected_revision=expected_revision,
    )
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        print(f"Validation failed with {len(violations)} violation(s).", file=sys.stderr)
        return 1

    annotated_episodes = {row.episode_index for row in rows}
    skipped = sorted(set(dataset.episode_lengths) - annotated_episodes)
    print(
        f"OK: {len(rows)} rows covering {len(annotated_episodes)}/{dataset.total_episodes} "
        f"episodes (omitted: {len(skipped)})."
    )
    return 0


def _resolve_expected_revision(src_root: Path, explicit_revision: str | None) -> str | None:
    if explicit_revision is not None:
        return explicit_revision
    detected = detect_dataset_revision(src_root)
    if detected == "local":
        print(
            "Warning: dataset revision could not be detected ('local'); "
            "annotation revision was not verified.",
            file=sys.stderr,
        )
        return None
    return detected


def _cmd_relabel(args: argparse.Namespace) -> int:
    try:
        src_root = resolve_dataset_root(args.src, args.repo_id)
    except (OSError, ValueError) as exc:
        print(f"[src] {exc}", file=sys.stderr)
        return 1

    resolved_repo_id = args.repo_id or infer_repo_id(src_root)
    resolved_rev = args.revision or detect_dataset_revision(src_root)
    spec_path = (
        args.task_spec
        if args.task_spec is not None
        else default_task_spec_path(src_root, repo_id=resolved_repo_id)
    )
    ann_path = (
        args.annotations
        if args.annotations is not None
        else default_annotations_path(src_root, repo_id=resolved_repo_id, revision=resolved_rev)
    )

    try:
        task_spec = load_task_spec(spec_path)
    except (OSError, ValueError) as exc:
        print(f"[task-spec] {exc}", file=sys.stderr)
        return 1

    try:
        rows = load_annotations(ann_path)
    except (OSError, ValueError) as exc:
        print(f"[annotations] {exc}", file=sys.stderr)
        return 1

    expected_revision = _resolve_expected_revision(src_root, args.revision)

    try:
        result = relabel_dataset(
            src_root,
            args.dst,
            rows,
            task_spec,
            expected_repo_id=args.repo_id,
            expected_revision=expected_revision,
        )
    except RelabelValidationError as exc:
        for violation in exc.violations:
            print(violation, file=sys.stderr)
        print(
            f"Relabel aborted with {len(exc.violations)} violation(s); nothing written.",
            file=sys.stderr,
        )
        return 1
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {result.dst}: {result.total_episodes} episodes, {result.total_frames} frames, "
        f"{len(result.tasks)} tasks, task_index counts {list(result.task_counts)}"
    )
    return 0


def _cmd_verify_relabel(args: argparse.Namespace) -> int:
    try:
        src_root = resolve_dataset_root(args.src, args.repo_id)
    except (OSError, ValueError) as exc:
        print(f"[src] {exc}", file=sys.stderr)
        return 1

    resolved_repo_id = args.repo_id or infer_repo_id(src_root)
    resolved_rev = args.revision or detect_dataset_revision(src_root)
    spec_path = (
        args.task_spec
        if args.task_spec is not None
        else default_task_spec_path(src_root, repo_id=resolved_repo_id)
    )
    ann_path = (
        args.annotations
        if args.annotations is not None
        else default_annotations_path(src_root, repo_id=resolved_repo_id, revision=resolved_rev)
    )

    try:
        task_spec = load_task_spec(spec_path)
    except (OSError, ValueError) as exc:
        print(f"[task-spec] {exc}", file=sys.stderr)
        return 1

    try:
        rows = load_annotations(ann_path)
    except (OSError, ValueError) as exc:
        print(f"[annotations] {exc}", file=sys.stderr)
        return 1

    expected_revision = _resolve_expected_revision(src_root, args.revision)

    try:
        result = verify_relabel(
            src_root,
            args.dst,
            rows,
            task_spec,
            expected_repo_id=args.repo_id,
            expected_revision=expected_revision,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1

    print(f"output: {result.total_frames} frames, task_index counts {list(result.task_counts)}")
    if result.stats_mean or result.stats_count:
        print(
            f"stats.json task_index mean={result.stats_mean} count={result.stats_count} "
            f"(data mean={result.data_mean:.6f})"
        )
    for err in result.errors:
        print(f"ERROR {err}")
    print("OK" if result.ok else f"{len(result.errors)} error(s)")
    return 0 if result.ok else 1


def _resolve_multi_dataset_roots(
    *,
    dataset_roots: Sequence[Path] | None,
    repo_ids: Sequence[str] | None,
    annotation_paths: Sequence[Path] | None,
) -> list[tuple[Path | None, str | None]] | None:
    pairs: list[tuple[Path | None, str | None]]
    if dataset_roots and repo_ids:
        if len(dataset_roots) != len(repo_ids):
            print(
                f"Expected equal counts for --dataset-root ({len(dataset_roots)}) "
                f"and --repo-id ({len(repo_ids)}).",
                file=sys.stderr,
            )
            return None
        pairs = list(zip(dataset_roots, repo_ids, strict=True))
    elif dataset_roots:
        pairs = [(p, None) for p in dataset_roots]
    elif repo_ids:
        pairs = [(None, r) for r in repo_ids]
    else:
        print("Provide at least one --dataset-root or --repo-id.", file=sys.stderr)
        return None

    if annotation_paths is not None and len(annotation_paths) != len(pairs):
        print(
            f"Expected one --annotations per dataset "
            f"(got {len(annotation_paths)} vs {len(pairs)}).",
            file=sys.stderr,
        )
        return None
    return pairs


def _load_validated_inputs(
    *,
    dataset_roots: Sequence[Path] | None,
    repo_ids: Sequence[str] | None,
    annotation_paths: Sequence[Path] | None,
    labels: Sequence[str] | None,
    task_spec_path: Path | None,
) -> tuple[TaskSpec, list[tuple[DatasetView, list[AnnotationRow], str]]] | None:
    """Resolve, open, and validate one or more ``(dataset, annotations)`` inputs."""
    pairs = _resolve_multi_dataset_roots(
        dataset_roots=dataset_roots,
        repo_ids=repo_ids,
        annotation_paths=annotation_paths,
    )
    if pairs is None:
        return None

    n = len(pairs)
    if labels is not None and len(labels) != n:
        print(
            f"Expected one --label per dataset (got {len(labels)} vs {n}).",
            file=sys.stderr,
        )
        return None

    datasets: list[tuple[DatasetView, str | None]] = []
    for idx, (root_arg, repo_arg) in enumerate(pairs):
        try:
            root = resolve_dataset_root(root_arg, repo_arg)
            dataset = DatasetView(root, repo_id=repo_arg)
        except (OSError, ValueError) as exc:
            print(f"[dataset #{idx + 1}] {exc}", file=sys.stderr)
            return None
        datasets.append((dataset, repo_arg))

    first_dataset = datasets[0][0]
    resolved_spec_path = (
        task_spec_path
        if task_spec_path is not None
        else default_task_spec_path(first_dataset.root, repo_id=first_dataset.repo_id)
    )
    try:
        task_spec = load_task_spec(resolved_spec_path)
    except (OSError, ValueError) as exc:
        print(f"[task-spec] {exc}", file=sys.stderr)
        return None

    raw_labels: list[str] = []
    loaded: list[tuple[DatasetView, list[AnnotationRow]]] = []
    for idx, (dataset, explicit_repo) in enumerate(datasets):
        ann_path = (
            annotation_paths[idx]
            if annotation_paths is not None
            else default_annotations_path(
                dataset.root, repo_id=dataset.repo_id, revision=dataset.revision
            )
        )
        try:
            rows = load_annotations(ann_path)
        except (OSError, ValueError) as exc:
            print(f"[annotations #{idx + 1}] {exc}", file=sys.stderr)
            return None

        expected_revision: str | None = None
        if dataset.revision == "local":
            print(
                f"Warning: dataset #{idx + 1} ({dataset.repo_id}) revision could not be detected "
                "('local'); annotation revision was not verified.",
                file=sys.stderr,
            )
        else:
            expected_revision = dataset.revision

        violations = validate_annotations(
            rows,
            task_spec,
            dataset.episode_lengths,
            expected_repo_id=explicit_repo,
            expected_revision=expected_revision,
        )
        if violations:
            for violation in violations:
                print(violation, file=sys.stderr)
            print(
                f"Validation failed for {ann_path} with {len(violations)} violation(s).",
                file=sys.stderr,
            )
            return None

        loaded.append((dataset, rows))
        raw_labels.append(labels[idx] if labels is not None else dataset.repo_id)

    # Disambiguate duplicate default labels so legends and tables remain distinct.
    label_counts: dict[str, int] = {}
    for lbl in raw_labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    seen: dict[str, int] = {}
    result: list[tuple[DatasetView, list[AnnotationRow], str]] = []
    for (dataset, rows), lbl in zip(loaded, raw_labels, strict=True):
        if label_counts[lbl] > 1:
            seen[lbl] = seen.get(lbl, 0) + 1
            final_label = f"{lbl}#{seen[lbl]}"
        else:
            final_label = lbl
        result.append((dataset, rows, final_label))
    return task_spec, result


def _add_multi_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task-spec",
        type=Path,
        default=None,
        help=(
            "Path to the YAML task spec file "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml)."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        default=None,
        help="Local LeRobot v3.0 dataset directory (repeat once per input dataset).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        action="append",
        default=None,
        help="Dataset repo ID (repeat once per input dataset when --dataset-root is omitted).",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        action="append",
        default=None,
        help=(
            "Annotation CSV path paired with each dataset "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv)."
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        action="append",
        default=None,
        help="Optional display label for each input dataset (repeat to match datasets).",
    )


def _cmd_report(args: argparse.Namespace) -> int:
    loaded = _load_validated_inputs(
        dataset_roots=args.dataset_root,
        repo_ids=args.repo_id,
        annotation_paths=args.annotations,
        labels=args.label,
        task_spec_path=args.task_spec,
    )
    if loaded is None:
        return 1
    task_spec, inputs = loaded

    advice_markdown: str | None = None
    if args.advice is not None:
        try:
            advice_markdown = args.advice.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"[advice] {exc}", file=sys.stderr)
            return 1

    if args.group:
        if len(inputs) != 1:
            print(
                "--group only applies when analyzing a single dataset input.",
                file=sys.stderr,
            )
            return 1
        dataset, rows, _ = inputs[0]
        try:
            groups = parse_episode_groups(args.group, dataset.total_episodes)
        except ValueError as exc:
            print(f"[group] {exc}", file=sys.stderr)
            return 1
        analyses = analyze_dataset_groups(
            dataset,
            rows,
            task_spec,
            groups,
            speed_ratio_threshold=args.speed_ratio_threshold,
            min_rest_speed_mps=args.min_rest_speed,
            max_traj_points=args.max_traj_points,
        )
    else:
        analyses = [
            analyze_dataset(
                dataset,
                rows,
                task_spec,
                label=label,
                speed_ratio_threshold=args.speed_ratio_threshold,
                min_rest_speed_mps=args.min_rest_speed,
                max_traj_points=args.max_traj_points,
            )
            for dataset, rows, label in inputs
        ]

    first_dataset = inputs[0][0]
    output_target = (
        args.output
        if args.output is not None
        else default_report_dir(
            first_dataset.root,
            repo_id=first_dataset.repo_id,
            revision=first_dataset.revision,
        )
    )
    html_path, summary_path, _ = write_report_bundle(
        analyses,
        task_spec,
        output_target,
        speed_ratio_threshold=args.speed_ratio_threshold,
        min_rest_speed_mps=args.min_rest_speed,
        title=args.title,
        advice_markdown=advice_markdown,
    )

    for analysis in analyses:
        print(format_frame_share_table(analysis.frame_shares))
        flagged_eps = sorted({m.episode_index for m in analysis.speed_metrics if m.not_slowed_down})
        flagged_arm_records = sum(1 for m in analysis.speed_metrics if m.not_slowed_down)
        idle_arm_records = sum(1 for m in analysis.speed_metrics if m.idle)
        print(
            f"Speed check (ratio > {args.speed_ratio_threshold:.2f}): "
            f"{len(flagged_eps)} episode(s) flagged as not slowed down "
            f"({flagged_arm_records} arm-episode record(s); "
            f"{idle_arm_records} idle arm-episode(s))."
        )
        if analysis.skipped_objects:
            print(f"Skipped object events: {len(analysis.skipped_objects)} (listed in report).")
        print()

    print(f"Wrote offline HTML report: {html_path}")
    print(f"Wrote summary JSON: {summary_path}")
    return 0


def _cmd_mix(args: argparse.Namespace) -> int:
    loaded = _load_validated_inputs(
        dataset_roots=args.dataset_root,
        repo_ids=args.repo_id,
        annotation_paths=args.annotations,
        labels=args.label,
        task_spec_path=args.task_spec,
    )
    if loaded is None:
        return 1
    task_spec, inputs = loaded

    n = len(inputs)
    if args.repeat is None:
        repeats = [1] * n
    elif len(args.repeat) != n:
        print(
            f"Expected one --repeat per dataset (got {len(args.repeat)} vs {n}).",
            file=sys.stderr,
        )
        return 1
    else:
        repeats = list(args.repeat)

    if any(r < 1 for r in repeats):
        print(f"All --repeat values must be >= 1, got {repeats}.", file=sys.stderr)
        return 1

    entries = [
        MixEntry(
            label=label,
            rows=rows,
            fps=dataset.fps,
            total_episodes=dataset.total_episodes,
            episode_lengths=dataset.episode_lengths,
            repeat=repeat,
        )
        for (dataset, rows, label), repeat in zip(inputs, repeats, strict=True)
    ]
    per_input, combined = compute_mix(entries, task_spec)
    print(format_mix_report(per_input, combined))
    return 0


def _cmd_suggest(args: argparse.Namespace) -> int:
    if args.count < 1:
        print(f"--count / -n must be >= 1, got {args.count}.", file=sys.stderr)
        return 1

    loaded = _load_validated_inputs(
        dataset_roots=args.dataset_root,
        repo_ids=args.repo_id,
        annotation_paths=args.annotations,
        labels=args.label,
        task_spec_path=args.task_spec,
    )
    if loaded is None:
        return 1
    task_spec, inputs = loaded

    analyses = [
        analyze_dataset(
            dataset,
            rows,
            task_spec,
            label=label,
            include_trajectories=False,
        )
        for dataset, rows, label in inputs
    ]
    all_poses = [sample for analysis in analyses for sample in analysis.object_poses]
    all_skipped_objects = [item for analysis in analyses for item in analysis.skipped_objects]
    coverages = compute_all_coverages(all_poses, task_spec)
    suggestions, omitted_objects = suggest_placements(coverages, args.count, seed=args.seed)

    if all_skipped_objects:
        print(
            f"Skipped object events: {len(all_skipped_objects)} "
            "(missing event frame or absent subtask)."
        )
    print(format_suggestions_table(suggestions, omitted_objects=omitted_objects))

    if args.output is not None:
        write_suggestions_csv(suggestions, args.output)
        print(f"\nWrote suggestions CSV: {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for ``lerobot-check``."""
    parser = argparse.ArgumentParser(
        prog="lerobot-check",
        description=(
            "Subtask annotation, validation, relabeling, and analysis CLI for LeRobot v3.0."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser(
        "info",
        help="Print dataset repo_id, revision, fps, cameras, episode count, and episode lengths.",
    )
    _add_dataset_args(info_parser)
    info_parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Override detected dataset revision.",
    )
    info_parser.set_defaults(func=_cmd_info)

    propose_parser = subparsers.add_parser(
        "propose",
        help="Propose gripper close/open motions, active squeezes, and TCP low-speed segments.",
    )
    _add_dataset_args(propose_parser)
    propose_parser.add_argument(
        "--episodes",
        nargs="+",
        default=None,
        help="Episode indices or start:end ranges (comma or space separated). Default: all.",
    )
    propose_parser.add_argument(
        "--gripper-delta",
        type=float,
        default=0.06,
        help="Relative gripper amplitude change required to emit close/open (default: 0.06).",
    )
    propose_parser.add_argument(
        "--squeeze-gap",
        type=float,
        default=0.05,
        help="Gap threshold (action - state <= -gap) for squeeze intervals (default: 0.05).",
    )
    propose_parser.add_argument(
        "--min-squeeze-s",
        type=float,
        default=0.2,
        help="Minimum squeeze interval duration in seconds (default: 0.2).",
    )
    propose_parser.add_argument(
        "--low-speed-threshold",
        type=float,
        default=0.02,
        help="TCP linear speed threshold in m/s for low-speed segments (default: 0.02).",
    )
    propose_parser.add_argument(
        "--min-low-speed-s",
        type=float,
        default=0.3,
        help="Minimum low-speed segment duration in seconds (default: 0.3).",
    )
    propose_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Prints CSV to stdout if omitted.",
    )
    propose_parser.set_defaults(func=_cmd_propose)

    render_parser = subparsers.add_parser(
        "render",
        help="Render tiled camera contact sheets with frame numbers to PNG.",
    )
    _add_dataset_args(render_parser)
    render_parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Zero-based episode index to render.",
    )
    render_parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help=(
            "Explicit frames or half-open start:end[:step] ranges [start, end), "
            "e.g. '250,283,310' or '600:900:30'."
        ),
    )
    render_parser.add_argument(
        "--events",
        type=str,
        default=None,
        help="Center event frames to expand with --window-frames / --window-s and --stride.",
    )
    render_parser.add_argument(
        "--proposals",
        type=Path,
        default=None,
        help="Proposals CSV from `lerobot-check propose`; expands events for --episode.",
    )
    render_parser.add_argument(
        "--event-types",
        type=str,
        default=None,
        help=(
            "Comma-separated proposal event types to render "
            "(default: gripper_close,gripper_open,squeeze)."
        ),
    )
    render_parser.add_argument(
        "--window-frames",
        type=int,
        default=None,
        help="Half-window radius around events in frames (overrides --window-s).",
    )
    render_parser.add_argument(
        "--window-s",
        type=float,
        default=1.5,
        help="Half-window radius around events in seconds (default: 1.5).",
    )
    render_parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Sampling stride in frames around events (default: 15).",
    )
    render_parser.add_argument(
        "--gripper-delta",
        type=float,
        default=0.06,
        help="Relative gripper delta used if auto-proposing events (default: 0.06).",
    )
    render_parser.add_argument(
        "--squeeze-gap",
        type=float,
        default=0.05,
        help="Squeeze gap used if auto-proposing events (default: 0.05).",
    )
    render_parser.add_argument(
        "--min-squeeze-s",
        type=float,
        default=0.2,
        help="Minimum squeeze duration (s) used if auto-proposing events (default: 0.2).",
    )
    render_parser.add_argument(
        "--subtask",
        type=str,
        default=None,
        help="Subtask name whose configured task-spec cameras are used by default.",
    )
    render_parser.add_argument(
        "--task-spec",
        type=Path,
        default=None,
        help=(
            "Path to the YAML task spec file used with --subtask "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml)."
        ),
    )
    render_parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help=(
            "Explicit camera keys or short names (overrides --subtask; default without "
            "--subtask is head, left_wrist, right_wrist, or head alone with --columns)."
        ),
    )
    render_parser.add_argument(
        "--columns",
        type=int,
        default=None,
        help=(
            "Single-camera grid layout column count for coarse scans "
            "(default: multi-camera strip with 1 frame per row)."
        ),
    )
    render_parser.add_argument(
        "--tile-width",
        type=int,
        default=320,
        help="Width of each camera tile in pixels (default: 320).",
    )
    render_parser.add_argument(
        "--tile-height",
        type=int,
        default=None,
        help="Height of each camera tile in pixels (default: preserve aspect ratio).",
    )
    render_parser.add_argument(
        "--max-frames-per-image",
        type=int,
        default=None,
        help=(
            "Maximum frames per PNG before splitting pages "
            "(default: 12 for strip layout, columns*8 for grid layout)."
        ),
    )
    render_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output PNG file path (outside the repository).",
    )
    render_parser.set_defaults(func=_cmd_render)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate an annotation CSV against a task spec and dataset metadata.",
    )
    _add_dataset_args(validate_parser)
    validate_parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Expected dataset revision (overrides auto-detected local cache revision).",
    )
    validate_parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=(
            "Path to the annotation CSV file "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv)."
        ),
    )
    validate_parser.add_argument(
        "--task-spec",
        type=Path,
        default=None,
        help=(
            "Path to the YAML task spec file "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml)."
        ),
    )
    validate_parser.set_defaults(func=_cmd_validate)

    relabel_parser = subparsers.add_parser(
        "relabel",
        help="Write a relabeled LeRobot v3.0 dataset with per-frame subtask task_index.",
    )
    relabel_parser.add_argument(
        "--src",
        "--dataset-root",
        dest="src",
        type=Path,
        default=None,
        help="Source LeRobot v3.0 dataset root.",
    )
    relabel_parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "Dataset repo ID (resolves local cache path if --src is omitted and enforces "
            "annotation repo_id)."
        ),
    )
    relabel_parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Expected dataset revision (overrides auto-detected local cache revision).",
    )
    relabel_parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Destination dataset root; must not exist.",
    )
    relabel_parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=(
            "Path to the annotation CSV file "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv)."
        ),
    )
    relabel_parser.add_argument(
        "--task-spec",
        type=Path,
        default=None,
        help=(
            "Path to the YAML task spec file "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml)."
        ),
    )
    relabel_parser.set_defaults(func=_cmd_relabel)

    verify_parser = subparsers.add_parser(
        "verify-relabel",
        help="Verify that a relabeled dataset changed only task bookkeeping matching annotations.",
    )
    verify_parser.add_argument(
        "--src",
        "--dataset-root",
        dest="src",
        type=Path,
        default=None,
        help="Source LeRobot v3.0 dataset root before relabeling.",
    )
    verify_parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "Dataset repo ID (resolves local cache path if --src is omitted and enforces "
            "annotation repo_id)."
        ),
    )
    verify_parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Expected dataset revision (overrides auto-detected local cache revision).",
    )
    verify_parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Relabeled LeRobot v3.0 dataset root to verify.",
    )
    verify_parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=(
            "Path to the annotation CSV file "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv)."
        ),
    )
    verify_parser.add_argument(
        "--task-spec",
        type=Path,
        default=None,
        help=(
            "Path to the YAML task spec file "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml)."
        ),
    )
    verify_parser.set_defaults(func=_cmd_verify_relabel)

    report_parser = subparsers.add_parser(
        "report",
        help="Generate an offline Chinese HTML analysis report and summary.json.",
    )
    _add_multi_dataset_args(report_parser)
    report_parser.add_argument(
        "--group",
        type=str,
        action="append",
        default=None,
        metavar="NAME=EPISODES",
        help=(
            "Partition a single dataset into colored episode groups "
            "(e.g. --group 原数据=0-120 --group 补录=121-173 or 0:121 / 121:174)."
        ),
    )
    report_parser.add_argument(
        "--advice",
        type=Path,
        default=None,
        help="Optional Markdown file (e.g. advice.md) to embed as the 'Agent 建议' section.",
    )
    report_parser.add_argument(
        "--speed-ratio-threshold",
        type=float,
        default=DEFAULT_SPEED_RATIO_THRESHOLD,
        help=(
            "Flag episodes as 'not slowed down' when fine_median / rest_median exceeds this "
            f"threshold (default: {DEFAULT_SPEED_RATIO_THRESHOLD})."
        ),
    )
    report_parser.add_argument(
        "--min-rest-speed",
        type=float,
        default=DEFAULT_MIN_REST_SPEED_MPS,
        help=(
            "Non-fine median TCP speed (m/s) below which an arm is treated as idle "
            f"(default: {DEFAULT_MIN_REST_SPEED_MPS})."
        ),
    )
    report_parser.add_argument(
        "--max-traj-points",
        type=int,
        default=DEFAULT_MAX_TRAJ_POINTS,
        help=(
            "Maximum sampled points per episode trajectory segment in 3D plots "
            f"(default: {DEFAULT_MAX_TRAJ_POINTS})."
        ),
    )
    report_parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom HTML report title.",
    )
    report_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output report directory or .html path "
            "(default: <lerobot_root>/.xense-viewer/reports/<org>__<name>@<revision>/)."
        ),
    )
    report_parser.set_defaults(func=_cmd_report)

    mix_parser = subparsers.add_parser(
        "mix",
        help="Compute combined subtask and fine-window frame shares across repeated datasets.",
    )
    _add_multi_dataset_args(mix_parser)
    mix_parser.add_argument(
        "--repeat",
        type=int,
        action="append",
        default=None,
        help="Positive integer repetition weight for each input dataset (default: 1 each).",
    )
    mix_parser.set_defaults(func=_cmd_mix)

    suggest_parser = subparsers.add_parser(
        "suggest",
        help="Suggest next-batch object placements (xy, yaw) sampled from sparse coverage cells.",
    )
    _add_multi_dataset_args(suggest_parser)
    suggest_parser.add_argument(
        "-n",
        "--count",
        type=int,
        required=True,
        help="Number of batch episodes N to suggest.",
    )
    suggest_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible placement sampling (default: 0).",
    )
    suggest_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path to write placement suggestions (outside the repository).",
    )
    suggest_parser.set_defaults(func=_cmd_suggest)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``lerobot-check`` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
