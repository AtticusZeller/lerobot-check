from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lerobot_check.analysis import load_distribution_json
from lerobot_check.calib import parse_head_calibration
from lerobot_check.collection_planner import build_plan, delete_pending
from lerobot_check.collection_progress import (
    CollectionPlan,
    PlanBusyError,
    atomic_write,
    import_progress,
    plan_session,
    validate_plan,
)
from lerobot_check.dataset import (
    dataset_lerobot_root,
    dataset_slug,
    detect_dataset_revision,
    infer_repo_id,
    resolve_dataset_root,
)
from lerobot_check.json_io import read_json
from lerobot_check.masks import sha256_bytes
from lerobot_check.suggest import DEFAULT_POSE_LAYER, POSE_LAYERS, load_target_polygon


def configure_plan_parser(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="plan_command", required=True)
    generate = commands.add_parser("generate", help="Generate pending collection rounds.")
    generate.add_argument("--dataset-root", type=Path)
    generate.add_argument("--repo-id")
    generate.add_argument("--viewer-root", type=Path)
    generate.add_argument("--distribution", type=Path)
    generate.add_argument("--calib", type=Path)
    generate.add_argument("--targets", type=Path)
    generate.add_argument("--template-dir", type=Path)
    generate.add_argument("--template", type=Path, action="append", default=[])
    generate.add_argument("--plan", type=Path)
    generate.add_argument("--rounds", type=int, default=10)
    generate.add_argument("--objects-per-round", type=int, default=1)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--layer", choices=POSE_LAYERS, default=DEFAULT_POSE_LAYER)
    generate.add_argument("--replace-round", type=int)
    generate.add_argument("--expect-missing", action="store_true")
    generate.add_argument("--expect-plan-id")
    generate.add_argument("--expect-generation", type=int)
    delete = commands.add_parser("delete-round", help="Remove one pending round.")
    delete.add_argument("--plan", type=Path, required=True)
    delete.add_argument("--round-id", required=True, type=int)
    delete.add_argument("--expect-plan-id", required=True)
    delete.add_argument("--expect-generation", required=True, type=int)
    merge = commands.add_parser("import-progress", help="Merge progress from a returned plan.")
    merge.add_argument("--plan", type=Path, required=True)
    merge.add_argument("--returned", type=Path, required=True)
    confirm = commands.add_parser("mark-round", help="Record a verified saved episode or skip.")
    confirm.add_argument("--plan", type=Path, required=True)
    confirm.add_argument("--round-id", required=True, type=int)
    result = confirm.add_mutually_exclusive_group(required=True)
    result.add_argument("--episode-index", type=int)
    result.add_argument("--skip", action="store_true")
    parser.set_defaults(func=run_collection_plan)


def _check_expectation(args: argparse.Namespace, plan: CollectionPlan | None) -> None:
    if args.expect_plan_id is None and args.expect_generation is None:
        return
    if not plan or (plan["plan_id"], plan["generation"]) != (
        args.expect_plan_id,
        args.expect_generation,
    ):
        raise ValueError("Collection plan changed; refresh and review the new generation")


def _generate(args: argparse.Namespace) -> None:
    root = resolve_dataset_root(args.dataset_root, args.repo_id)
    repo_id = args.repo_id or infer_repo_id(root)
    revision = detect_dataset_revision(root)
    viewer_dir = args.viewer_root or dataset_lerobot_root(root) / ".xense-viewer"
    slug = dataset_slug(repo_id)
    key = f"{slug}@{revision}"
    plan_path = args.plan or viewer_dir / "plans" / f"{key}.json"
    dist_path = args.distribution or viewer_dir / "distribution" / f"{key}.json"
    calib_path = args.calib or viewer_dir / "calib" / f"{slug}.head.json"
    targets_path = args.targets or viewer_dir / "targets" / f"{slug}.json"
    templates = args.template or sorted(
        (args.template_dir or viewer_dir / "mask-templates" / key).glob("*.json")
    )
    with plan_session(plan_path) as session:
        try:
            previous = session.read()
        except FileNotFoundError:
            previous = None
        if args.expect_missing and previous is not None:
            raise ValueError("Collection plan already exists; refresh and review it")
        _check_expectation(args, previous)
        calib_bytes = calib_path.read_bytes()
        calibration = parse_head_calibration(calib_bytes, source=str(calib_path))
        if (calibration.repo_id, calibration.revision) != (repo_id, revision):
            raise ValueError("Head calibration belongs to another dataset revision")
        info = root / "meta" / "info.json"
        count = read_json(info, info.read_bytes())["total_episodes"]
        plan, bundle = build_plan(
            load_distribution_json(dist_path),
            calibration,
            calibration_sha256=sha256_bytes(calib_bytes),
            templates=[read_json(path, path.read_bytes()) for path in templates],
            source_episode_count=count,
            targets=load_target_polygon(targets_path)
            if args.targets or targets_path.is_file()
            else None,
            previous=previous,
            round_count=args.rounds,
            objects_per_round=args.objects_per_round,
            seed=args.seed,
            replace_round_id=args.replace_round,
            layer=args.layer,
        )
        validate_plan(plan)
        atomic_write(plan_path.parent / "bundles" / f"{plan['mask_bundle_sha256']}.json", bundle)
        session.write(plan)
    print(f"Wrote collection plan generation {plan['generation']}: {plan_path}")


def run_collection_plan(args: argparse.Namespace) -> int:
    try:
        if args.plan_command == "generate":
            _generate(args)
        elif args.plan_command == "import-progress":
            plan = import_progress(args.plan, args.returned)
            collected = sum(r["status"] == "collected" for r in plan["rounds"])
            print(f"Merged progress: {collected} collected rounds")
        else:
            with plan_session(args.plan) as session:
                if args.plan_command == "delete-round":
                    plan = session.read()
                    _check_expectation(args, plan)
                    delete_pending(plan, args.round_id)
                    session.write(plan)
                    print(f"Removed collection round {args.round_id}")
                elif args.skip:
                    session.mark_skipped(args.round_id)
                else:
                    session.mark_collected(args.round_id, args.episode_index)
    except PlanBusyError as exc:
        print(f"[collection-plan] {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[collection-plan] {exc}", file=sys.stderr)
        return 1
    return 0
