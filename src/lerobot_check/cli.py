"""Command-line interface for ``lerobot-check`` (propose, render, validate, relabel)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lerobot_check.dataset import DatasetView, detect_dataset_revision, resolve_dataset_root
from lerobot_check.propose import (
    DEFAULT_RENDER_EVENT_TYPES,
    ProposeConfig,
    format_proposals_csv,
    load_proposal_event_frames,
    propose_episode,
    write_proposals_csv,
)
from lerobot_check.relabel import RelabelValidationError, relabel_dataset, verify_relabel
from lerobot_check.render import (
    DEFAULT_CAMERAS,
    parse_frame_spec,
    render_episode_sheets,
    select_frames_around_events,
)
from lerobot_check.schema import load_annotations, load_task_spec
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

    if args.cameras:
        cameras = tuple(args.cameras)
    elif args.columns is not None:
        cameras = (DEFAULT_CAMERAS[0],)
    else:
        cameras = DEFAULT_CAMERAS

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
        task_spec = load_task_spec(args.task_spec)
    except (OSError, ValueError) as exc:
        print(f"[task-spec] {exc}", file=sys.stderr)
        return 1

    try:
        rows = load_annotations(args.annotations)
    except (OSError, ValueError) as exc:
        print(f"[annotations] {exc}", file=sys.stderr)
        return 1

    root = resolve_dataset_root(args.dataset_root, args.repo_id)
    dataset = DatasetView(root, repo_id=args.repo_id, revision=args.revision)

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
        task_spec = load_task_spec(args.task_spec)
    except (OSError, ValueError) as exc:
        print(f"[task-spec] {exc}", file=sys.stderr)
        return 1

    try:
        rows = load_annotations(args.annotations)
    except (OSError, ValueError) as exc:
        print(f"[annotations] {exc}", file=sys.stderr)
        return 1

    try:
        src_root = resolve_dataset_root(args.src, args.repo_id)
    except (OSError, ValueError) as exc:
        print(f"[src] {exc}", file=sys.stderr)
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
        task_spec = load_task_spec(args.task_spec)
    except (OSError, ValueError) as exc:
        print(f"[task-spec] {exc}", file=sys.stderr)
        return 1

    try:
        rows = load_annotations(args.annotations)
    except (OSError, ValueError) as exc:
        print(f"[annotations] {exc}", file=sys.stderr)
        return 1

    try:
        src_root = resolve_dataset_root(args.src, args.repo_id)
    except (OSError, ValueError) as exc:
        print(f"[src] {exc}", file=sys.stderr)
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


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for ``lerobot-check``."""
    parser = argparse.ArgumentParser(
        prog="lerobot-check",
        description=(
            "Subtask annotation proposal, frame rendering, and validation for LeRobot v3.0."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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
        help="Render tiled multi-camera contact sheets with frame numbers to PNG.",
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
        help="Explicit frames or start:end[:step] ranges, e.g. '250,283,310' or '600:900:30'.",
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
        "--cameras",
        nargs="+",
        default=None,
        help=(
            "Camera keys or short names (default: head, left_wrist, right_wrist; "
            "or head alone when --columns is set)."
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
        required=True,
        help="Path to the annotation CSV file.",
    )
    validate_parser.add_argument(
        "--task-spec",
        type=Path,
        required=True,
        help="Path to the YAML task spec file.",
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
        required=True,
        help="Path to the annotation CSV file.",
    )
    relabel_parser.add_argument(
        "--task-spec",
        type=Path,
        required=True,
        help="Path to the YAML task spec file.",
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
        required=True,
        help="Path to the annotation CSV file.",
    )
    verify_parser.add_argument(
        "--task-spec",
        type=Path,
        required=True,
        help="Path to the YAML task spec file.",
    )
    verify_parser.set_defaults(func=_cmd_verify_relabel)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``lerobot-check`` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
