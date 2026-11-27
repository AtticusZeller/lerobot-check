"""Command-line interface for ``lerobot-check``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from lerobot_check.analysis import (
    DEFAULT_TRAJECTORY_SAMPLES,
    compute_distribution,
    format_distribution_summary,
    load_distribution_json,
    parse_episode_filter,
    write_distribution_json,
)
from lerobot_check.calib import (
    DEFAULT_CALIB_CAMERA,
    DEFAULT_FRAMES_PER_ARM,
    DEFAULT_MAX_GRIPPER_POS,
    DEFAULT_MAX_SPEED_MPS,
    HeadCalibration,
    calibrate_head_camera,
    compare_arm_extrinsics,
    load_head_calibration,
    load_intrinsics,
    select_calib_frames,
    write_json,
)
from lerobot_check.dataset import (
    DatasetView,
    default_annotations_path,
    default_calib_frames_path,
    default_calib_head_path,
    default_calib_intrinsics_path,
    default_calib_points_path,
    default_distribution_path,
    default_suggestions_path,
    default_targets_path,
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
    DEFAULT_CAMERAS,
    _resolve_camera_keys,
    load_windows_csv,
    parse_boundary_spec,
    parse_crop_specs,
    parse_episode_spec,
    parse_frame_spec,
    parse_window_seconds,
    render_boundary_exemplars,
    render_boundary_sheets,
    render_episode_sheets,
    render_window_sheets,
    resolve_boundary_subtasks,
    resolve_crop_keys,
    resolve_exemplar_cameras,
    resolve_render_cameras,
    select_boundary_exemplars,
    select_end_frames,
    select_frames_around_events,
)
from lerobot_check.schema import (
    TaskSpec,
    load_annotations,
    load_task_spec,
)
from lerobot_check.suggest import (
    format_suggestions_summary,
    generate_suggestions,
    load_target_polygon,
    write_suggestions_csv,
    write_suggestions_json,
)
from lerobot_check.validate import validate_annotations


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
            "Dataset repo ID (e.g. Xense/pack_iphone_box_1119). "
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
    episodes = (
        parse_episode_filter(args.episodes, dataset.total_episodes)
        if args.episodes
        else list(range(dataset.total_episodes))
    )
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


def _cmd_exemplars(args: argparse.Namespace) -> int:
    try:
        boundary = parse_boundary_spec(args.boundary)
    except ValueError as exc:
        print(f"[boundary] {exc}", file=sys.stderr)
        return 1

    try:
        root = resolve_dataset_root(args.dataset_root, args.repo_id)
        dataset = DatasetView(root, repo_id=args.repo_id, revision=args.revision)
    except (OSError, ValueError) as exc:
        print(f"[dataset] {exc}", file=sys.stderr)
        return 1

    task_spec: TaskSpec | None = None
    spec_path = (
        args.task_spec
        if args.task_spec is not None
        else default_task_spec_path(root, repo_id=dataset.repo_id)
    )
    if args.task_spec is not None or spec_path.is_file():
        try:
            task_spec = load_task_spec(spec_path)
        except (OSError, ValueError) as exc:
            print(f"[task-spec] {exc}", file=sys.stderr)
            return 1

    try:
        cameras = resolve_exemplar_cameras(
            boundary,
            explicit_cameras=args.cameras,
            task_spec=task_spec,
        )
    except (KeyError, ValueError) as exc:
        print(f"[cameras] {exc}", file=sys.stderr)
        return 1

    ann_path = (
        args.annotations
        if args.annotations is not None
        else default_annotations_path(root, repo_id=dataset.repo_id, revision=dataset.revision)
    )
    if args.annotations is None and not ann_path.is_file():
        print(
            f"No human exemplars found for boundary '{boundary.raw}' "
            f"(annotation CSV not found: {ann_path})."
        )
        return 0

    try:
        rows = load_annotations(ann_path)
    except (OSError, ValueError) as exc:
        print(f"[annotations] {exc}", file=sys.stderr)
        return 1

    try:
        exemplars = select_boundary_exemplars(
            rows,
            boundary,
            max_episodes=args.max_episodes,
        )
    except ValueError as exc:
        print(f"[exemplars] {exc}", file=sys.stderr)
        return 1

    if not exemplars:
        print(f"No human exemplars found for boundary '{boundary.raw}' in {ann_path}.")
        return 0

    try:
        rendered = render_boundary_exemplars(
            dataset,
            exemplars,
            args.output,
            cameras=cameras,
            window_s=args.window_s,
            stride=args.stride,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[exemplars] {exc}", file=sys.stderr)
        return 1

    cam_names = ", ".join(cam.removeprefix("observation.images.") for cam in cameras)
    print(f"boundary={boundary.raw} cameras=[{cam_names}] count={len(rendered)}")
    for ex, png_path in rendered:
        print(f"episode={ex.episode_index} frame={ex.frame} path={png_path}")
    return 0


def _cmd_boundary_sheet(args: argparse.Namespace) -> int:
    if args.tile_height <= 0:
        print(
            f"[boundary-sheet] --tile-height must be positive, got {args.tile_height}",
            file=sys.stderr,
        )
        return 1
    if args.columns <= 0:
        print(f"[boundary-sheet] --columns must be positive, got {args.columns}", file=sys.stderr)
        return 1
    if args.per_page < 0:
        print(f"[boundary-sheet] --per-page must be >= 0, got {args.per_page}", file=sys.stderr)
        return 1

    try:
        crops = parse_crop_specs(args.crop)
    except ValueError as exc:
        print(f"[crop] {exc}", file=sys.stderr)
        return 1

    episode_filter: list[int] | None = None
    if args.episodes is not None:
        try:
            episode_filter = parse_episode_spec(args.episodes)
        except ValueError as exc:
            print(f"[episodes] {exc}", file=sys.stderr)
            return 1

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
    try:
        task_spec = load_task_spec(spec_path)
    except (OSError, ValueError) as exc:
        print(f"[task-spec] {exc}", file=sys.stderr)
        return 1

    try:
        target_subtasks = resolve_boundary_subtasks(task_spec, args.subtasks)
    except ValueError as exc:
        print(f"[subtasks] {exc}", file=sys.stderr)
        return 1

    ann_path = (
        args.annotations
        if args.annotations is not None
        else default_annotations_path(root, repo_id=dataset.repo_id, revision=dataset.revision)
    )
    try:
        rows = load_annotations(ann_path)
    except (OSError, ValueError) as exc:
        print(f"[annotations] {exc}", file=sys.stderr)
        return 1

    cameras = tuple(args.cameras) if args.cameras else DEFAULT_CAMERAS
    dataset_camera_keys = getattr(dataset, "camera_keys", ())
    resolved_cameras = (
        _resolve_camera_keys(cameras, dict.fromkeys(dataset_camera_keys))
        if dataset_camera_keys
        else cameras
    )
    try:
        resolved_crops = resolve_crop_keys(crops, resolved_cameras)
    except ValueError as exc:
        print(f"[crop] {exc}", file=sys.stderr)
        return 1

    episode_lengths = getattr(dataset, "episode_lengths", None)

    for subtask_pos, subtask_name in target_subtasks:
        try:
            end_frames = select_end_frames(
                rows,
                subtask_name,
                offset=args.offset,
                episodes=episode_filter,
                episode_lengths=episode_lengths,
            )
        except ValueError as exc:
            print(f"[boundary-sheet] {exc}", file=sys.stderr)
            return 1

        if not end_frames:
            print(
                f"No annotated episodes found for subtask '{subtask_name}' in {ann_path}.",
                file=sys.stderr,
            )
            continue

        try:
            written = render_boundary_sheets(
                dataset,
                end_frames,
                args.output,
                subtask_position=subtask_pos,
                subtask=subtask_name,
                cameras=resolved_cameras,
                crops=resolved_crops,
                tile_height=args.tile_height,
                columns=args.columns,
                per_page=args.per_page,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"[boundary-sheet] {exc}", file=sys.stderr)
            return 1

        for path in written:
            print(path)

    return 0


def _cmd_window_sheet(args: argparse.Namespace) -> int:
    if (args.window_s is None) == (args.windows is None):
        print(
            "[window-sheet] Provide exactly one of --window-s or --windows.",
            file=sys.stderr,
        )
        return 1
    if args.samples < 2:
        print(f"[window-sheet] --samples must be >= 2, got {args.samples}", file=sys.stderr)
        return 1
    if args.tile_height <= 0:
        print(
            f"[window-sheet] --tile-height must be positive, got {args.tile_height}",
            file=sys.stderr,
        )
        return 1
    if args.per_page < 0:
        print(f"[window-sheet] --per-page must be >= 0, got {args.per_page}", file=sys.stderr)
        return 1
    if not args.name or not args.name.strip():
        print("[window-sheet] --name must not be empty.", file=sys.stderr)
        return 1

    try:
        crops = parse_crop_specs(args.crop)
    except ValueError as exc:
        print(f"[crop] {exc}", file=sys.stderr)
        return 1

    episode_filter: list[int] | None = None
    if args.episodes is not None:
        try:
            episode_filter = parse_episode_spec(args.episodes)
        except ValueError as exc:
            print(f"[episodes] {exc}", file=sys.stderr)
            return 1

    window_seconds: tuple[float, float] | None = None
    windows_by_ep: dict[int, tuple[int, int]] | None = None
    if args.window_s is not None:
        try:
            window_seconds = parse_window_seconds(args.window_s)
        except ValueError as exc:
            print(f"[window-s] {exc}", file=sys.stderr)
            return 1
    else:
        try:
            windows_by_ep = load_windows_csv(args.windows)
        except (OSError, ValueError) as exc:
            print(f"[windows] {exc}", file=sys.stderr)
            return 1

    try:
        root = resolve_dataset_root(args.dataset_root, args.repo_id)
        dataset = DatasetView(root, repo_id=args.repo_id)
    except (OSError, ValueError) as exc:
        print(f"[dataset] {exc}", file=sys.stderr)
        return 1

    episode_lengths = getattr(dataset, "episode_lengths", None)
    if episode_lengths is not None:
        known_episodes = set(episode_lengths)
        all_episodes = sorted(episode_lengths)
    else:
        total = getattr(dataset, "total_episodes", 0)
        known_episodes = set(range(total))
        all_episodes = list(range(total))

    if episode_filter is not None:
        unknown_filter = [ep for ep in episode_filter if ep not in known_episodes]
        if unknown_filter:
            print(
                f"[episodes] Episode index not in dataset: {unknown_filter}",
                file=sys.stderr,
            )
            return 1

    if window_seconds is not None:
        lo_s, hi_s = window_seconds
        lo_frame = round(lo_s * dataset.fps)
        hi_frame = round(hi_s * dataset.fps)
        target_episodes = episode_filter if episode_filter is not None else all_episodes
        episode_windows = [(ep, lo_frame, hi_frame) for ep in target_episodes]
    else:
        assert windows_by_ep is not None
        if episode_filter is not None:
            missing_rows = [ep for ep in episode_filter if ep not in windows_by_ep]
            if missing_rows:
                print(
                    f"[windows] --episodes entries with no row in {args.windows}: {missing_rows}",
                    file=sys.stderr,
                )
                return 1
            filter_set = set(episode_filter)
            episode_windows = [
                (ep, lo, hi) for ep, (lo, hi) in windows_by_ep.items() if ep in filter_set
            ]
        else:
            episode_windows = [(ep, lo, hi) for ep, (lo, hi) in windows_by_ep.items()]
        unknown_win = [ep for ep, _, _ in episode_windows if ep not in known_episodes]
        if unknown_win:
            print(
                f"[windows] Episode index not in dataset: {unknown_win}",
                file=sys.stderr,
            )
            return 1

    if not episode_windows:
        print("[window-sheet] No episodes selected to render.", file=sys.stderr)
        return 1

    cameras = tuple(args.cameras) if args.cameras else (DEFAULT_CAMERAS[0],)
    dataset_camera_keys = getattr(dataset, "camera_keys", ())
    resolved_cameras = (
        _resolve_camera_keys(cameras, dict.fromkeys(dataset_camera_keys))
        if dataset_camera_keys
        else cameras
    )
    if dataset_camera_keys:
        missing_cams = [cam for cam in resolved_cameras if cam not in dataset_camera_keys]
        if missing_cams:
            print(
                f"[cameras] Unknown camera(s): {missing_cams}; "
                f"available: {list(dataset_camera_keys)}.",
                file=sys.stderr,
            )
            return 1
    try:
        resolved_crops = resolve_crop_keys(crops, resolved_cameras)
    except ValueError as exc:
        print(f"[crop] {exc}", file=sys.stderr)
        return 1

    try:
        written = render_window_sheets(
            dataset,
            episode_windows,
            args.output,
            samples=args.samples,
            cameras=resolved_cameras,
            crops=resolved_crops,
            tile_height=args.tile_height,
            per_page=args.per_page,
            name=args.name,
        )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"[window-sheet] {exc}", file=sys.stderr)
        return 1

    for path in written:
        print(path)
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


def _cmd_distribution(args: argparse.Namespace) -> int:
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

    episode_filter: list[int] | None = None
    if args.episodes:
        try:
            episode_filter = parse_episode_filter(args.episodes, dataset.total_episodes)
        except ValueError as exc:
            print(f"[episodes] {exc}", file=sys.stderr)
            return 1

    calib_path: Path | None = args.calib
    if calib_path is not None:
        if not calib_path.is_file():
            print(f"[calib] Head calibration file does not exist: {calib_path}", file=sys.stderr)
            return 1
    else:
        candidate_calib = default_calib_head_path(root, repo_id=dataset.repo_id)
        if candidate_calib.is_file():
            calib_path = candidate_calib

    calibration: HeadCalibration | None = None
    calib_mtime: str | None = None
    if calib_path is not None:
        try:
            calibration = load_head_calibration(calib_path)
            calib_mtime = datetime.fromtimestamp(calib_path.stat().st_mtime, tz=UTC).isoformat()
        except (OSError, ValueError) as exc:
            print(f"[calib] {exc}", file=sys.stderr)
            return 1

    try:
        payload = compute_distribution(
            dataset,
            rows,
            task_spec,
            annotations_path=ann_path.resolve(),
            task_spec_path=spec_path.resolve(),
            episode_filter=episode_filter,
            trajectory_samples=args.trajectory_samples,
            calibration=calibration,
            calib_path=calib_path.resolve() if calib_path is not None else None,
            calib_mtime=calib_mtime,
        )
    except (OSError, ValueError) as exc:
        print(f"[distribution] {exc}", file=sys.stderr)
        return 1

    out_path = (
        args.output
        if args.output is not None
        else default_distribution_path(root, repo_id=dataset.repo_id, revision=dataset.revision)
    )
    write_distribution_json(payload, out_path)
    print(format_distribution_summary(payload))
    print(f"\nWrote distribution JSON: {out_path}")
    return 0


def _cmd_suggest(args: argparse.Namespace) -> int:
    has_dataset_ref = args.dataset_root is not None or args.repo_id is not None
    root: Path | None = None
    repo_id: str | None = args.repo_id
    revision: str | None = args.revision

    if has_dataset_ref:
        try:
            root = resolve_dataset_root(args.dataset_root, args.repo_id)
        except (OSError, ValueError) as exc:
            print(f"[dataset] {exc}", file=sys.stderr)
            return 1
        repo_id = repo_id or infer_repo_id(root)
        revision = revision or detect_dataset_revision(root)

    if args.distribution is not None:
        dist_path = args.distribution
    elif root is not None:
        dist_path = default_distribution_path(root, repo_id=repo_id, revision=revision)
    else:
        print(
            "Provide --distribution or one of --dataset-root / --repo-id.",
            file=sys.stderr,
        )
        return 1

    try:
        distribution = load_distribution_json(dist_path)
    except (OSError, ValueError) as exc:
        print(f"[distribution] {exc}", file=sys.stderr)
        return 1

    targets_path: Path | None = args.targets
    if targets_path is not None:
        if not targets_path.is_file():
            print(f"[targets] Target polygon file does not exist: {targets_path}", file=sys.stderr)
            return 1
    elif root is not None:
        candidate = default_targets_path(root, repo_id=repo_id)
        if candidate.is_file():
            targets_path = candidate

    calib_path: Path | None = args.calib
    if calib_path is not None:
        if not calib_path.is_file():
            print(f"[calib] Head calibration file does not exist: {calib_path}", file=sys.stderr)
            return 1
    elif root is not None:
        candidate = default_calib_head_path(root, repo_id=repo_id)
        if candidate.is_file():
            calib_path = candidate

    targets_payload = None
    if targets_path is not None:
        try:
            targets_payload = load_target_polygon(targets_path)
        except (OSError, ValueError) as exc:
            print(f"[targets] {exc}", file=sys.stderr)
            return 1

    calibration: HeadCalibration | None = None
    if calib_path is not None:
        try:
            calibration = load_head_calibration(calib_path)
        except (OSError, ValueError) as exc:
            print(f"[calib] {exc}", file=sys.stderr)
            return 1

    try:
        payload = generate_suggestions(
            distribution,
            count=args.count,
            test_per_tier=args.test_per_tier,
            targets=targets_payload,
            calibration=calibration,
            object_filter=args.object,
            seed=args.seed,
            distribution_path=dist_path.resolve(),
            targets_path=targets_path.resolve() if targets_path is not None else None,
            calib_path=calib_path.resolve() if calib_path is not None else None,
        )
    except ValueError as exc:
        print(f"[suggest] {exc}", file=sys.stderr)
        return 1

    if args.output is not None:
        out_path = args.output
    elif root is not None:
        out_path = default_suggestions_path(root, repo_id=repo_id, revision=revision)
    else:
        print("Provide -o/--output when --dataset-root and --repo-id are omitted.", file=sys.stderr)
        return 1

    write_suggestions_json(payload, out_path)
    print(format_suggestions_summary(payload))
    print(f"Wrote suggestions JSON: {out_path}")
    if args.csv is not None:
        write_suggestions_csv(payload, args.csv)
        print(f"Wrote suggestions CSV: {args.csv}")
    return 0


def _cmd_calib_frames(args: argparse.Namespace) -> int:
    if args.append is not None and args.overwrite:
        print(
            "[calib-frames] --append and --overwrite are mutually exclusive.",
            file=sys.stderr,
        )
        return 1

    parsed_arms: list[str] | None = None
    if args.arms is not None:
        if args.append is None:
            print(
                "[calib-frames] --arms is only allowed with --append.",
                file=sys.stderr,
            )
            return 1
        parsed_arms = [part.strip() for part in args.arms.split(",") if part.strip()]
        if not parsed_arms:
            print(
                "[calib-frames] --arms must specify at least one arm name.",
                file=sys.stderr,
            )
            return 1

    max_speed_mps = (
        None
        if args.no_speed_filter
        else (DEFAULT_MAX_SPEED_MPS if args.max_speed is None else args.max_speed)
    )

    try:
        root = resolve_dataset_root(args.dataset_root, args.repo_id)
    except (OSError, ValueError) as exc:
        print(f"[dataset] {exc}", file=sys.stderr)
        return 1

    resolved_repo_id = args.repo_id or infer_repo_id(root)
    out_path = (
        args.output
        if args.output is not None
        else default_calib_frames_path(root, repo_id=resolved_repo_id)
    )

    existing = None
    if args.append is not None:
        if not out_path.is_file():
            print(
                f"[calib-frames] --append requires an existing frames file at {out_path}",
                file=sys.stderr,
            )
            return 1
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"[calib-frames] Failed to read {out_path}: {exc}", file=sys.stderr)
            return 1
    elif out_path.exists() and not args.overwrite:
        print(
            f"[calib-frames] Refusing to overwrite existing frames file {out_path} "
            "(regenerating restarts frame IDs and would orphan existing clicks). "
            "Pass --append K to add frames or --overwrite to regenerate.",
            file=sys.stderr,
        )
        return 1

    try:
        dataset = DatasetView(root, repo_id=args.repo_id, revision=args.revision)
    except (OSError, ValueError) as exc:
        print(f"[dataset] {exc}", file=sys.stderr)
        return 1

    episodes = None
    if args.episodes:
        try:
            episodes = parse_episode_filter(args.episodes, dataset.total_episodes)
        except ValueError as exc:
            print(f"[episodes] {exc}", file=sys.stderr)
            return 1

    before_per_arm: dict[str, int] = {}
    if existing is not None:
        for item in existing.get("frames", []):
            arm = str(item.get("arm", ""))
            before_per_arm[arm] = before_per_arm.get(arm, 0) + 1

    try:
        payload = select_calib_frames(
            dataset,
            camera=args.camera,
            count_per_arm=args.count,
            existing=existing,
            append_per_arm=args.append,
            episodes=episodes,
            max_gripper_pos=args.max_gripper,
            max_speed_mps=max_speed_mps,
            arms=parsed_arms,
        )
    except (KeyError, ValueError) as exc:
        print(f"[calib-frames] {exc}", file=sys.stderr)
        return 1

    write_json(payload, out_path)
    per_arm: dict[str, int] = {}
    for item in payload["frames"]:
        arm = str(item["arm"])
        per_arm[arm] = per_arm.get(arm, 0) + 1

    requested_count = args.append if args.append is not None else args.count
    added_per_arm: dict[str, int] = {}
    target_arm_list = (
        list(dict.fromkeys(parsed_arms)) if parsed_arms is not None else sorted(per_arm)
    )
    for arm in target_arm_list:
        added = per_arm.get(arm, 0) - before_per_arm.get(arm, 0)
        added_per_arm[arm] = added
        if added < requested_count:
            print(
                f"Warning: arm={arm!r} requested {requested_count} frame(s), "
                f"only {added} eligible frame(s) available.",
                file=sys.stderr,
            )

    counts_str = ", ".join(f"{arm}={cnt}" for arm, cnt in sorted(per_arm.items()))
    added_str = ", ".join(f"{arm}=+{cnt}" for arm, cnt in added_per_arm.items())
    print(f"Wrote {out_path}: {len(payload['frames'])} frame(s) ({counts_str}; added: {added_str})")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    has_dataset_ref = args.dataset_root is not None or args.repo_id is not None
    root: Path | None = None
    repo_id: str | None = args.repo_id
    if has_dataset_ref:
        try:
            root = resolve_dataset_root(args.dataset_root, args.repo_id)
        except (OSError, ValueError) as exc:
            print(f"[dataset] {exc}", file=sys.stderr)
            return 1
        repo_id = repo_id or infer_repo_id(root)

    if args.points is not None:
        points_path = args.points
    elif root is not None:
        points_path = default_calib_points_path(root, repo_id=repo_id)
    else:
        print("Provide --points or one of --dataset-root / --repo-id.", file=sys.stderr)
        return 1

    if args.intrinsics is not None:
        intrinsics_path = args.intrinsics
    elif root is not None:
        intrinsics_path = default_calib_intrinsics_path(root, repo_id=repo_id)
    else:
        print("Provide --intrinsics or one of --dataset-root / --repo-id.", file=sys.stderr)
        return 1

    if args.output is not None:
        out_path = args.output
    elif root is not None:
        out_path = default_calib_head_path(root, repo_id=repo_id)
    else:
        print("Provide -o/--output or one of --dataset-root / --repo-id.", file=sys.stderr)
        return 1

    if not intrinsics_path.is_file():
        print(f"[intrinsics] intrinsics.json not found: {intrinsics_path}", file=sys.stderr)
        return 1

    if not points_path.is_file():
        print(f"[calibrate] points.json not found: {points_path}", file=sys.stderr)
        return 1

    try:
        intrinsics = load_intrinsics(intrinsics_path)
    except (OSError, ValueError) as exc:
        print(f"[intrinsics] {exc}", file=sys.stderr)
        return 1

    try:
        points_payload = json.loads(points_path.read_text(encoding="utf-8"))
        head_payload = calibrate_head_camera(points_payload, intrinsics)
    except (OSError, ValueError) as exc:
        print(f"[calibrate] {exc}", file=sys.stderr)
        return 1

    write_json(head_payload, out_path)
    for arm_name, entry in head_payload["arms"].items():
        offset_mm = [v * 1000.0 for v in entry["tool_offset"]]
        loo = entry["loo_rms_px"]
        loo_str = "n/a" if loo is None else f"{loo:.2f} px"
        inlier_rms = entry.get("inlier_rms_px")
        inlier_rms_str = "n/a" if inlier_rms is None else f"{inlier_rms:.2f} px"
        inlier_loo = entry.get("inlier_loo_rms_px")
        inlier_loo_str = "n/a" if inlier_loo is None else f"{inlier_loo:.2f} px"
        n_inliers = entry.get("n_inliers", entry["n_points"])
        flagged = [r["id"] for r in entry["residuals"] if r.get("outlier")]
        outliers_str = ", ".join(flagged) if flagged else "none"
        print(
            f"arm={arm_name}: n={entry['n_points']} (inliers={n_inliers}), "
            f"rms={entry['rms_px']:.2f} px (inlier={inlier_rms_str}), "
            f"loo_rms={loo_str} (inlier={inlier_loo_str}), "
            f"tool_offset=[{offset_mm[0]:.1f}, {offset_mm[1]:.1f}, {offset_mm[2]:.1f}] mm, "
            f"outliers=[{outliers_str}]"
        )
    diff = compare_arm_extrinsics(head_payload)
    if diff is not None:
        print(
            "left vs right extrinsic diff: "
            f"rot={diff['rotation_diff_deg']:.2f} deg, "
            f"tvec={diff['tvec_diff_mm']:.1f} mm, "
            f"camera_origin={diff['camera_origin_diff_mm']:.1f} mm"
        )
    print(f"Wrote head calibration JSON: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for ``lerobot-check``."""
    from lerobot_check.collection_plan_cli import configure_plan_parser
    from lerobot_check.mask_cli import configure_overlays_parser, configure_template_parser

    parser = argparse.ArgumentParser(
        prog="lerobot-check",
        description=(
            "Subtask annotation, validation, relabeling, distribution, and calibration CLI "
            "for LeRobot v3.0."
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
        help="Episode indices or ranges (comma or space separated). Default: all.",
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

    exemplars_parser = subparsers.add_parser(
        "exemplars",
        help="Render boundary or grasp strip exemplars from human-reviewed annotation rows.",
    )
    _add_dataset_args(exemplars_parser)
    exemplars_parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Override detected dataset revision when resolving default annotations path.",
    )
    exemplars_parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=(
            "Path to the annotation CSV file "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv)."
        ),
    )
    exemplars_parser.add_argument(
        "--task-spec",
        type=Path,
        default=None,
        help=(
            "Path to the YAML task spec file "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml)."
        ),
    )
    exemplars_parser.add_argument(
        "--boundary",
        type=str,
        required=True,
        help="Target boundary '<prev_subtask>:<next_subtask>' or '<subtask>:grasp'.",
    )
    exemplars_parser.add_argument(
        "--max-episodes",
        type=int,
        default=3,
        help="Maximum number of human-reviewed exemplar episodes to render (default: 3).",
    )
    exemplars_parser.add_argument(
        "--window-s",
        type=float,
        default=1.0,
        help="Half-window radius around the boundary frame in seconds (default: 1.0).",
    )
    exemplars_parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Sampling stride in frames within the window (default: 3).",
    )
    exemplars_parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help=(
            "Explicit camera keys or short names "
            "(overrides the union of subtask cameras from the task spec)."
        ),
    )
    exemplars_parser.add_argument(
        "--tile-width",
        type=int,
        default=320,
        help="Width of each camera tile in pixels (default: 320).",
    )
    exemplars_parser.add_argument(
        "--tile-height",
        type=int,
        default=None,
        help="Height of each camera tile in pixels (default: preserve aspect ratio).",
    )
    exemplars_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory for exemplar PNGs (outside the repository).",
    )
    exemplars_parser.set_defaults(func=_cmd_exemplars)

    boundary_sheet_parser = subparsers.add_parser(
        "boundary-sheet",
        help="Render per-subtask contact sheets of end frames across episodes.",
    )
    _add_dataset_args(boundary_sheet_parser)
    boundary_sheet_parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Override detected dataset revision when resolving default annotations path.",
    )
    boundary_sheet_parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=(
            "Path to the annotation CSV file "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv)."
        ),
    )
    boundary_sheet_parser.add_argument(
        "--task-spec",
        type=Path,
        default=None,
        help=(
            "Path to the YAML task spec file "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml)."
        ),
    )
    boundary_sheet_parser.add_argument(
        "--subtasks",
        nargs="+",
        default=None,
        help=("Subtask names to render (default: every subtask in the task spec except the last)."),
    )
    boundary_sheet_parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Optional episode filter with inclusive ranges, e.g. '5,9,20-28'. Default: all.",
    )
    boundary_sheet_parser.add_argument(
        "--offset",
        type=int,
        default=-1,
        help="Frame offset relative to row end (default: -1, the subtask's last frame).",
    )
    boundary_sheet_parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help=(
            "Camera keys or short names to place side by side "
            "(default: head left_wrist right_wrist)."
        ),
    )
    boundary_sheet_parser.add_argument(
        "--crop",
        action="append",
        default=None,
        metavar="CAM=x0,y0,x1,y1",
        help="Repeatable crop box in native pixels applied to CAM before resizing.",
    )
    boundary_sheet_parser.add_argument(
        "--tile-height",
        type=int,
        default=240,
        help="Height of each camera image in pixels, keeping aspect ratio (default: 240).",
    )
    boundary_sheet_parser.add_argument(
        "--columns",
        type=int,
        default=2,
        help="Number of episode cells per row (default: 2).",
    )
    boundary_sheet_parser.add_argument(
        "--per-page",
        type=int,
        default=12,
        help="Episodes per JPEG image, or 0 for all episodes on one image (default: 12).",
    )
    boundary_sheet_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory for boundary-sheet JPEGs (outside the repository).",
    )
    boundary_sheet_parser.set_defaults(func=_cmd_boundary_sheet)

    window_sheet_parser = subparsers.add_parser(
        "window-sheet",
        help="Render multi-episode contact sheets sampled across a time or frame window.",
    )
    _add_dataset_args(window_sheet_parser)
    window_sheet_parser.add_argument(
        "--window-s",
        type=str,
        default=None,
        metavar="LO:HI",
        help="Absolute time window in seconds (LO:HI) applied to every selected episode.",
    )
    window_sheet_parser.add_argument(
        "--windows",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "CSV file with header episode_index,lo,hi (inclusive frame numbers), "
            "one row per episode."
        ),
    )
    window_sheet_parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Optional episode filter with inclusive ranges, e.g. '0,15,30-35'. Default: all.",
    )
    window_sheet_parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="Number of evenly spaced frames sampled per window (default: 8, must be >= 2).",
    )
    window_sheet_parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Camera keys or short names stacked vertically per cell (default: head).",
    )
    window_sheet_parser.add_argument(
        "--crop",
        action="append",
        default=None,
        metavar="CAM=x0,y0,x1,y1",
        help="Repeatable crop box in native pixels applied to CAM before resizing.",
    )
    window_sheet_parser.add_argument(
        "--tile-height",
        type=int,
        default=200,
        help="Height of each camera image in pixels, keeping aspect ratio (default: 200).",
    )
    window_sheet_parser.add_argument(
        "--per-page",
        type=int,
        default=8,
        help="Episodes per JPEG image, or 0 for all episodes on one image (default: 8).",
    )
    window_sheet_parser.add_argument(
        "--name",
        type=str,
        default="window",
        help="Output filename prefix; writes <name>_p<page:02d>.jpg (default: window).",
    )
    window_sheet_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory for window-sheet JPEGs (outside the repository).",
    )
    window_sheet_parser.set_defaults(func=_cmd_window_sheet)

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

    dist_parser = subparsers.add_parser(
        "distribution",
        help="Compute machine-readable distribution.json for xense-lerobot-viewer.",
    )
    _add_dataset_args(dist_parser)
    dist_parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Override detected dataset revision.",
    )
    dist_parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=(
            "Path to the annotation CSV file "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv)."
        ),
    )
    dist_parser.add_argument(
        "--task-spec",
        type=Path,
        default=None,
        help=(
            "Path to the YAML task spec file "
            "(default: <lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml)."
        ),
    )
    dist_parser.add_argument(
        "--episodes",
        nargs="+",
        default=None,
        help=(
            "Restrict output to episode indices/ranges (e.g. '0,30,60', '0-20', '0:21'). "
            "Default: all episodes."
        ),
    )
    dist_parser.add_argument(
        "--trajectory-samples",
        type=int,
        default=DEFAULT_TRAJECTORY_SAMPLES,
        help=(
            "Number of progress-resampled points per trajectory from 0 to 1 "
            f"(default: {DEFAULT_TRAJECTORY_SAMPLES})."
        ),
    )
    dist_parser.add_argument(
        "--calib",
        type=Path,
        default=None,
        help=(
            "Optional head-camera calibration head.json used to compute calibrated "
            "tool-point positions p_tcp + R_tcp @ tool_offset "
            "(default: <lerobot_root>/.xense-viewer/calib/<org>__<name>.head.json if present)."
        ),
    )
    dist_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output distribution JSON path "
            "(default: <lerobot_root>/.xense-viewer/distribution/<org>__<name>@<revision>.json)."
        ),
    )
    dist_parser.set_defaults(func=_cmd_distribution)

    suggest_parser = subparsers.add_parser(
        "suggest",
        help="Generate collection placement suggestions and 3-tier test points.",
    )
    _add_dataset_args(suggest_parser)
    suggest_parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Override detected dataset revision when resolving default paths.",
    )
    suggest_parser.add_argument(
        "--distribution",
        type=Path,
        default=None,
        help=(
            "Path to distribution.json "
            "(default: <lerobot_root>/.xense-viewer/distribution/<org>__<name>@<revision>.json)."
        ),
    )
    suggest_parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help=(
            "Optional target polygon JSON in head-image pixels "
            "(default: <lerobot_root>/.xense-viewer/targets/<org>__<name>.json if present)."
        ),
    )
    suggest_parser.add_argument(
        "--calib",
        type=Path,
        default=None,
        help=(
            "Optional head-camera calibration head.json "
            "(default: <lerobot_root>/.xense-viewer/calib/<org>__<name>.head.json if present)."
        ),
    )
    suggest_parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=10,
        help="Number of collection placements N per object (default: 10).",
    )
    suggest_parser.add_argument(
        "--test-per-tier",
        type=int,
        default=3,
        help="Number of test points per tier (dense, edge, empty) per object (default: 3).",
    )
    suggest_parser.add_argument(
        "--object",
        nargs="+",
        default=None,
        help="Optional subset of object names from distribution.json to generate for.",
    )
    suggest_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling (default: 0).",
    )
    suggest_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output suggestions JSON path "
            "(default: <lerobot_root>/.xense-viewer/suggestions/<org>__<name>@<revision>.json)."
        ),
    )
    suggest_parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV export path for placements and test points.",
    )
    suggest_parser.set_defaults(func=_cmd_suggest)

    calib_frames_parser = subparsers.add_parser(
        "calib-frames",
        help="Select diverse TCP frames per arm for head-camera calibration.",
    )
    _add_dataset_args(calib_frames_parser)
    calib_frames_parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Override detected dataset revision.",
    )
    calib_frames_parser.add_argument(
        "--camera",
        type=str,
        default=DEFAULT_CALIB_CAMERA,
        help=f"Head camera feature key (default: {DEFAULT_CALIB_CAMERA}).",
    )
    calib_frames_parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=DEFAULT_FRAMES_PER_ARM,
        help=f"Number of frames to select per arm (default: {DEFAULT_FRAMES_PER_ARM}).",
    )
    calib_frames_parser.add_argument(
        "--append",
        type=int,
        default=None,
        help="Append K additional frames per arm to an existing frames.json, keeping existing IDs.",
    )
    calib_frames_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Allow overwriting an existing frames.json (restarts frame IDs).",
    )
    calib_frames_parser.add_argument(
        "--max-gripper",
        type=float,
        default=DEFAULT_MAX_GRIPPER_POS,
        help=(
            "Maximum gripper opening allowed for newly selected frames "
            f"(default: {DEFAULT_MAX_GRIPPER_POS:g}, closed fingertips)."
        ),
    )
    speed_group = calib_frames_parser.add_mutually_exclusive_group()
    speed_group.add_argument(
        "--max-speed",
        type=float,
        default=None,
        help=(
            "Maximum TCP linear speed in m/s for newly selected frames "
            f"(default: {DEFAULT_MAX_SPEED_MPS:g})."
        ),
    )
    speed_group.add_argument(
        "--no-speed-filter",
        action="store_true",
        default=False,
        help="Disable TCP speed filtering (mutually exclusive with --max-speed).",
    )
    calib_frames_parser.add_argument(
        "--arms",
        type=str,
        default=None,
        help=(
            "Comma-separated arm names to append frames for (e.g. 'left' or 'left,right'). "
            "Allowed only with --append."
        ),
    )
    calib_frames_parser.add_argument(
        "--episodes",
        nargs="+",
        default=None,
        help="Optional episode filter (indices or ranges). Default: all episodes.",
    )
    calib_frames_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output frames JSON path "
            "(default: <lerobot_root>/.xense-viewer/calib/<org>__<name>.frames.json)."
        ),
    )
    calib_frames_parser.set_defaults(func=_cmd_calib_frames)

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Solve per-arm head-camera extrinsics and tool offset from points.json.",
    )
    _add_dataset_args(calibrate_parser)
    calibrate_parser.add_argument(
        "--points",
        type=Path,
        default=None,
        help=(
            "Path to annotated points JSON "
            "(default: <lerobot_root>/.xense-viewer/calib/<org>__<name>.points.json)."
        ),
    )
    calibrate_parser.add_argument(
        "--intrinsics",
        type=Path,
        default=None,
        help=(
            "Path to camera intrinsics JSON "
            "(default: <lerobot_root>/.xense-viewer/calib/<org>__<name>.intrinsics.json)."
        ),
    )
    calibrate_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output head calibration JSON path "
            "(default: <lerobot_root>/.xense-viewer/calib/<org>__<name>.head.json)."
        ),
    )
    calibrate_parser.set_defaults(func=_cmd_calibrate)

    configure_template_parser(
        subparsers.add_parser(
            "mask-template", help="Prepare a reviewed head-camera object outline."
        )
    )
    configure_overlays_parser(
        subparsers.add_parser(
            "mask-overlays", help="Project reviewed outlines onto collection targets."
        )
    )
    configure_plan_parser(
        subparsers.add_parser("collection-plan", help="Generate and review collection rounds.")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``lerobot-check`` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
