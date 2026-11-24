"""Convert legacy egg-carton ``cuts.csv`` into a lerobot-check annotation CSV.

Reads the original 122-episode ``Xense/bi_elite_cs66_rt_egg_carton_to_tray_1016_fixed``
dataset (pre-deletion episode numbering). Episode 108 is excluded, each cut frame
is computed as ``round(cut_s * fps)`` (30 fps), and ``grasp`` is set to the first
gripper close-motion start frame reported by ``propose`` inside each subtask (for
the arm defined by ``task_spec.objects`` for that subtask, or any arm if omitted).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Collection, Mapping
from pathlib import Path

from lerobot_check.dataset import DatasetView, resolve_dataset_root
from lerobot_check.propose import EpisodeProposal, ProposeConfig, propose_episode
from lerobot_check.schema import AnnotationRow, TaskSpec, load_task_spec, save_annotations
from lerobot_check.validate import validate_annotations

DEFAULT_REPO_ID = "Xense/bi_elite_cs66_rt_egg_carton_to_tray_1016_fixed"
DEFAULT_TASK_SPEC = Path(__file__).resolve().parent / "task_spec.yaml"
DEFAULT_SKIP_EPISODES = (108,)


def load_cuts_csv(path: Path) -> dict[int, float]:
    """Read ``episode_index,cut_s`` rows from a legacy ``cuts.csv`` file.

    Args:
        path: Path to ``cuts.csv``.

    Returns:
        Mapping from pre-deletion ``episode_index`` to cut time in seconds.

    Raises:
        ValueError: If duplicate episodes or malformed values are encountered.
    """
    cuts: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"episode_index", "cut_s"} <= set(reader.fieldnames):
            raise ValueError(
                f"{path} must have 'episode_index' and 'cut_s' columns, got {reader.fieldnames}"
            )
        for line_no, raw in enumerate(reader, start=2):
            try:
                ep = int(raw["episode_index"])
                cut_s = float(raw["cut_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Line {line_no}: cannot parse row {raw!r}") from exc
            if ep in cuts:
                raise ValueError(f"Line {line_no}: duplicate episode_index {ep}")
            cuts[ep] = cut_s
    return cuts


def _arm_for_subtask(task_spec: TaskSpec, subtask: str) -> str | None:
    """Resolve the single arm configured in ``task_spec.objects`` for ``subtask``, if any."""
    matching = [obj for obj in task_spec.objects if obj.subtask == subtask]
    grasp_arms = {obj.arm for obj in matching if obj.event == "grasp"}
    if len(grasp_arms) == 1:
        return next(iter(grasp_arms))
    all_arms = {obj.arm for obj in matching}
    if len(all_arms) == 1:
        return next(iter(all_arms))
    return None


def build_annotations_from_cuts(
    cuts: Mapping[int, float],
    *,
    task_spec: TaskSpec,
    episode_lengths: Mapping[int, int],
    proposals_by_episode: Mapping[int, EpisodeProposal],
    fps: float,
    repo_id: str,
    revision: str,
    skip_episodes: Collection[int] = DEFAULT_SKIP_EPISODES,
) -> list[AnnotationRow]:
    """Convert two-stage cut times and proposals into validated ``AnnotationRow`` entries.

    Args:
        cuts: Mapping from pre-deletion ``episode_index`` to cut time (seconds).
        task_spec: Two-subtask task specification (e.g. ``open_carton``, ``place_eggs``).
        episode_lengths: Dataset episode lengths keyed by ``episode_index``.
        proposals_by_episode: ``EpisodeProposal`` for each kept episode.
        fps: Dataset frame rate (30 Hz for the egg-carton dataset).
        repo_id: Dataset repo ID recorded in each row.
        revision: Dataset revision recorded in each row.
        skip_episodes: Episodes to exclude from output (episode 108 by default).

    Returns:
        Ordered list of ``AnnotationRow`` records.

    Raises:
        ValueError: If ``task_spec`` does not define exactly 2 subtasks or if any
            generated row violates the annotation contract.
    """
    if len(task_spec.subtasks) != 2:
        raise ValueError(
            f"cuts.csv converter requires a 2-subtask task spec, got {len(task_spec.subtasks)}"
        )

    first_subtask = task_spec.subtasks[0].name
    second_subtask = task_spec.subtasks[1].name
    first_arm = _arm_for_subtask(task_spec, first_subtask)
    second_arm = _arm_for_subtask(task_spec, second_subtask)

    skipped = set(skip_episodes)
    rows: list[AnnotationRow] = []

    for episode_index in sorted(cuts):
        if episode_index in skipped:
            continue
        if episode_index not in episode_lengths:
            raise ValueError(f"Episode {episode_index} from cuts.csv not found in dataset")

        length = episode_lengths[episode_index]
        cut_frame = round(cuts[episode_index] * fps)
        if not 0 < cut_frame < length:
            raise ValueError(
                f"Episode {episode_index}: cut_s={cuts[episode_index]} -> "
                f"cut_frame={cut_frame} is outside (0, {length})"
            )

        proposal = proposals_by_episode[episode_index]
        first_grasp = proposal.first_close_frame(0, cut_frame, arm=first_arm)
        second_grasp = proposal.first_close_frame(cut_frame, length, arm=second_arm)

        rows.append(
            AnnotationRow(
                repo_id=repo_id,
                revision=revision,
                episode_index=episode_index,
                subtask=first_subtask,
                start=0,
                end=cut_frame,
                contact=None,
                grasp=first_grasp,
                success=True,
                annotator="converted",
                note="",
            )
        )
        rows.append(
            AnnotationRow(
                repo_id=repo_id,
                revision=revision,
                episode_index=episode_index,
                subtask=second_subtask,
                start=cut_frame,
                end=length,
                contact=None,
                grasp=second_grasp,
                success=True,
                annotator="converted",
                note="",
            )
        )

    violations = validate_annotations(
        rows,
        task_spec,
        episode_lengths,
        expected_repo_id=repo_id,
        expected_revision=revision,
    )
    if violations:
        details = "\n".join(str(v) for v in violations)
        raise ValueError(f"Converted annotations failed validation:\n{details}")

    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for ``convert_cuts.py``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to the original 122-episode egg-carton dataset.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Dataset repo ID (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Dataset revision (auto-detected from local cache metadata if omitted).",
    )
    parser.add_argument(
        "--cuts",
        type=Path,
        required=True,
        help="Path to legacy cuts.csv (pre-deletion episode numbering).",
    )
    parser.add_argument(
        "--task-spec",
        type=Path,
        default=DEFAULT_TASK_SPEC,
        help=f"Path to egg-carton task_spec.yaml (default: {DEFAULT_TASK_SPEC}).",
    )
    parser.add_argument(
        "--skip-episodes",
        type=int,
        nargs="*",
        default=list(DEFAULT_SKIP_EPISODES),
        help="Pre-deletion episode indices to omit (default: 108).",
    )
    parser.add_argument(
        "--gripper-delta",
        type=float,
        default=0.06,
        help="Relative gripper delta passed to propose (default: 0.06).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output annotation CSV path (outside the repository).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for converting legacy egg-carton ``cuts.csv``."""
    args = parse_args(argv)
    root = resolve_dataset_root(args.dataset_root, args.repo_id)
    dataset = DatasetView(root, repo_id=args.repo_id, revision=args.revision)
    task_spec = load_task_spec(args.task_spec)
    cuts = load_cuts_csv(args.cuts)
    skip_set = set(args.skip_episodes)

    propose_config = ProposeConfig(gripper_delta=args.gripper_delta)
    proposals_by_episode = {
        ep: propose_episode(dataset, ep, config=propose_config)
        for ep in sorted(cuts)
        if ep not in skip_set
    }

    rows = build_annotations_from_cuts(
        cuts,
        task_spec=task_spec,
        episode_lengths=dataset.episode_lengths,
        proposals_by_episode=proposals_by_episode,
        fps=dataset.fps,
        repo_id=dataset.repo_id,
        revision=dataset.revision,
        skip_episodes=skip_set,
    )
    save_annotations(rows, args.output)

    episodes_covered = sorted({row.episode_index for row in rows})
    with_grasp = sum(1 for row in rows if row.grasp is not None)
    print(
        f"Wrote {args.output}: {len(rows)} rows covering {len(episodes_covered)} episodes "
        f"(skipped: {sorted(skip_set)}; rows with grasp: {with_grasp}/{len(rows)})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
