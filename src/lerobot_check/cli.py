"""Command-line interface for ``lerobot-check`` (propose, render, validate)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lerobot_check.dataset import DatasetView, resolve_dataset_root
from lerobot_check.propose import (
    ProposeConfig,
    format_proposals_csv,
    load_proposal_event_frames,
    propose_episode,
    write_proposals_csv,
)
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
        close_threshold=args.close_threshold,
        open_threshold=args.open_threshold,
        low_speed_threshold_mps=args.low_speed_threshold,
        min_low_speed_s=args.min_low_speed_s,
    )
    proposals = [propose_episode(dataset, ep, config=config) for ep in episodes]
    if args.output is not None:
        write_proposals_csv(proposals, args.output)
        total_gripper = sum(len(p.gripper_events) for p in proposals)
        total_slow = sum(len(p.low_speed_segments) for p in proposals)
        print(
            f"Wrote {args.output}: {len(proposals)} episode(s), "
            f"{total_gripper} gripper event(s), {total_slow} low-speed segment(s)."
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
        event_frames: list[int] = []
        if args.events is not None:
            event_frames.extend(parse_frame_spec(args.events, record.length))
        if args.proposals is not None:
            event_frames.extend(load_proposal_event_frames(args.proposals, args.episode))
        if not event_frames:
            proposal = propose_episode(
                dataset,
                args.episode,
                config=ProposeConfig(
                    close_threshold=args.close_threshold,
                    open_threshold=args.open_threshold,
                ),
            )
            event_frames = [ev.frame for ev in proposal.gripper_events]
        if not event_frames:
            print(
                f"Episode {args.episode} has no candidate gripper events; "
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

    cameras = tuple(args.cameras) if args.cameras else DEFAULT_CAMERAS
    written = render_episode_sheets(
        dataset,
        args.episode,
        frame_indices,
        args.output,
        cameras=cameras,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        max_frames_per_image=args.max_frames_per_image,
        highlight_frames=highlight_frames,
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
    dataset = DatasetView(root, repo_id=args.repo_id)
    violations = validate_annotations(
        rows,
        task_spec,
        dataset.episode_lengths,
        expected_repo_id=args.repo_id,
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
        help="Propose per-arm gripper close/open events and TCP low-speed segments.",
    )
    _add_dataset_args(propose_parser)
    propose_parser.add_argument(
        "--episodes",
        nargs="+",
        default=None,
        help="Episode indices or start:end ranges (comma or space separated). Default: all.",
    )
    propose_parser.add_argument(
        "--close-threshold",
        type=float,
        default=0.5,
        help="Gripper position <= threshold triggers a close event (default: 0.5).",
    )
    propose_parser.add_argument(
        "--open-threshold",
        type=float,
        default=0.7,
        help="Gripper position >= threshold triggers an open event (default: 0.7).",
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
        help="Proposals CSV from `lerobot-check propose`; expands gripper events for --episode.",
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
        "--close-threshold",
        type=float,
        default=0.5,
        help="Gripper close threshold used if auto-proposing events (default: 0.5).",
    )
    render_parser.add_argument(
        "--open-threshold",
        type=float,
        default=0.7,
        help="Gripper open threshold used if auto-proposing events (default: 0.7).",
    )
    render_parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Camera feature keys to tile (default: head, left_wrist, right_wrist).",
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
        default=12,
        help="Maximum frames stacked per PNG before splitting pages (default: 12).",
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
        help="Validate an annotation CSV against a task spec and dataset episode lengths.",
    )
    _add_dataset_args(validate_parser)
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``lerobot-check`` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
