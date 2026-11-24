"""Validation rules for subtask annotation tables."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

from lerobot_check.schema import VALID_ANNOTATORS, AnnotationRow, TaskSpec


@dataclass(frozen=True)
class ValidationViolation:
    """One contract violation discovered while validating an annotation table.

    Attributes:
        episode_index: Episode index associated with the violation, or ``None`` for
            table-level violations.
        rule: Short machine-readable rule identifier.
        message: Human-readable description of the violation.
    """

    episode_index: int | None
    rule: str
    message: str

    def __str__(self) -> str:
        prefix = "table" if self.episode_index is None else f"episode {self.episode_index}"
        return f"[{prefix}] ({self.rule}) {self.message}"


def _validate_table_header_consistency(
    rows: Sequence[AnnotationRow],
    *,
    expected_repo_id: str | None,
    expected_revision: str | None,
) -> list[ValidationViolation]:
    violations: list[ValidationViolation] = []
    if not rows:
        violations.append(
            ValidationViolation(
                episode_index=None,
                rule="empty_table",
                message="Annotation table contains no rows.",
            )
        )
        return violations

    dataset_keys: set[tuple[str, str]] = set()
    for row in rows:
        if not row.repo_id:
            violations.append(
                ValidationViolation(
                    episode_index=row.episode_index,
                    rule="missing_repo_id",
                    message="Row has empty repo_id.",
                )
            )
        if not row.revision:
            violations.append(
                ValidationViolation(
                    episode_index=row.episode_index,
                    rule="missing_revision",
                    message="Row has empty revision.",
                )
            )
        if row.repo_id and row.revision:
            dataset_keys.add((row.repo_id, row.revision))

    if len(dataset_keys) > 1:
        violations.append(
            ValidationViolation(
                episode_index=None,
                rule="inconsistent_dataset",
                message=(
                    "Annotation table must belong to a single (repo_id, revision), "
                    f"found {sorted(dataset_keys)}."
                ),
            )
        )

    if expected_repo_id is not None:
        mismatched_repos = sorted({r.repo_id for r in rows if r.repo_id != expected_repo_id})
        if mismatched_repos:
            violations.append(
                ValidationViolation(
                    episode_index=None,
                    rule="repo_id_mismatch",
                    message=(f"Expected repo_id {expected_repo_id!r}, found {mismatched_repos}."),
                )
            )

    if expected_revision is not None:
        mismatched_revs = sorted({r.revision for r in rows if r.revision != expected_revision})
        if mismatched_revs:
            violations.append(
                ValidationViolation(
                    episode_index=None,
                    rule="revision_mismatch",
                    message=(f"Expected revision {expected_revision!r}, found {mismatched_revs}."),
                )
            )

    return violations


def _validate_episode_rows(
    episode_index: int,
    ep_rows: Sequence[AnnotationRow],
    *,
    episode_length: int | None,
    spec_index: Mapping[str, int],
) -> list[ValidationViolation]:
    violations: list[ValidationViolation] = []
    if episode_length is None:
        violations.append(
            ValidationViolation(
                episode_index=episode_index,
                rule="unknown_episode",
                message=f"Episode {episode_index} does not exist in dataset metadata.",
            )
        )

    sorted_rows = sorted(ep_rows, key=lambda r: (r.start, r.end))
    all_subtasks_known = True

    for row in sorted_rows:
        if row.annotator not in VALID_ANNOTATORS:
            violations.append(
                ValidationViolation(
                    episode_index=episode_index,
                    rule="invalid_annotator",
                    message=(
                        f"Row subtask={row.subtask!r} has invalid annotator {row.annotator!r}; "
                        f"expected one of {sorted(VALID_ANNOTATORS)}."
                    ),
                )
            )

        if row.subtask not in spec_index:
            all_subtasks_known = False
            violations.append(
                ValidationViolation(
                    episode_index=episode_index,
                    rule="unknown_subtask",
                    message=f"Subtask {row.subtask!r} is not defined in task spec.",
                )
            )

        if row.start < 0:
            violations.append(
                ValidationViolation(
                    episode_index=episode_index,
                    rule="out_of_range",
                    message=f"Subtask {row.subtask!r} has negative start frame {row.start}.",
                )
            )

        if episode_length is not None and (row.end > episode_length or row.start >= episode_length):
            violations.append(
                ValidationViolation(
                    episode_index=episode_index,
                    rule="out_of_range",
                    message=(
                        f"Subtask {row.subtask!r} interval [{row.start}, {row.end}) "
                        f"exceeds episode length {episode_length}."
                    ),
                )
            )

        if row.start >= row.end:
            violations.append(
                ValidationViolation(
                    episode_index=episode_index,
                    rule="invalid_interval",
                    message=(
                        f"Subtask {row.subtask!r} interval [{row.start}, {row.end}) "
                        "must satisfy start < end."
                    ),
                )
            )

        if row.contact is not None and not (row.start <= row.contact < row.end):
            violations.append(
                ValidationViolation(
                    episode_index=episode_index,
                    rule="contact_out_of_range",
                    message=(
                        f"Subtask {row.subtask!r} contact={row.contact} is outside "
                        f"[{row.start}, {row.end})."
                    ),
                )
            )

        if row.grasp is not None and not (row.start <= row.grasp < row.end):
            violations.append(
                ValidationViolation(
                    episode_index=episode_index,
                    rule="grasp_out_of_range",
                    message=(
                        f"Subtask {row.subtask!r} grasp={row.grasp} is outside "
                        f"[{row.start}, {row.end})."
                    ),
                )
            )

        if row.contact is not None and row.grasp is not None and row.contact > row.grasp:
            violations.append(
                ValidationViolation(
                    episode_index=episode_index,
                    rule="contact_after_grasp",
                    message=(
                        f"Subtask {row.subtask!r} has contact={row.contact} > grasp={row.grasp}."
                    ),
                )
            )

    if sorted_rows[0].start > 0:
        violations.append(
            ValidationViolation(
                episode_index=episode_index,
                rule="missing_frames",
                message=(
                    f"Episode does not start at frame 0; frames [0, {sorted_rows[0].start}) "
                    "are uncovered."
                ),
            )
        )

    for prev, curr in pairwise(sorted_rows):
        if curr.start > prev.end:
            violations.append(
                ValidationViolation(
                    episode_index=episode_index,
                    rule="missing_frames",
                    message=(
                        f"Gap between {prev.subtask!r} (end={prev.end}) and "
                        f"{curr.subtask!r} (start={curr.start}): frames [{prev.end}, {curr.start}) "
                        "are uncovered."
                    ),
                )
            )
        elif curr.start < prev.end:
            overlap_end = min(prev.end, curr.end)
            violations.append(
                ValidationViolation(
                    episode_index=episode_index,
                    rule="overlap",
                    message=(
                        f"Overlap between {prev.subtask!r} [{prev.start}, {prev.end}) and "
                        f"{curr.subtask!r} [{curr.start}, {curr.end}) on "
                        f"[{curr.start}, {overlap_end})."
                    ),
                )
            )

    if episode_length is not None and sorted_rows[-1].end < episode_length:
        violations.append(
            ValidationViolation(
                episode_index=episode_index,
                rule="missing_frames",
                message=(
                    f"Episode ends at {sorted_rows[-1].end} < length {episode_length}; "
                    f"frames [{sorted_rows[-1].end}, {episode_length}) are uncovered."
                ),
            )
        )

    if all_subtasks_known:
        # Subtasks within an episode must form a contiguous slice of task_spec.subtasks
        # in order, which permits single-stage or partial-stage supplement episodes while
        # rejecting reversed, duplicated, or skipped stages.
        for prev, curr in pairwise(sorted_rows):
            prev_idx = spec_index[prev.subtask]
            curr_idx = spec_index[curr.subtask]
            if curr_idx != prev_idx + 1:
                violations.append(
                    ValidationViolation(
                        episode_index=episode_index,
                        rule="wrong_subtask_order",
                        message=(
                            f"Subtask {curr.subtask!r} (spec index {curr_idx}) cannot follow "
                            f"{prev.subtask!r} (spec index {prev_idx}); subtasks must follow "
                            "task spec order as a contiguous subsequence."
                        ),
                    )
                )

    return violations


def validate_annotations(
    rows: Sequence[AnnotationRow],
    task_spec: TaskSpec,
    episode_lengths: Mapping[int, int],
    *,
    expected_repo_id: str | None = None,
    expected_revision: str | None = None,
) -> list[ValidationViolation]:
    """Validate annotation rows against a task spec and dataset episode lengths.

    Episodes absent from ``rows`` are allowed (downstream analysis skips them while
    relabeling checks completeness separately).

    Args:
        rows: Annotation rows to check.
        task_spec: Task specification defining legal subtasks and their order.
        episode_lengths: Mapping from ``episode_index`` to episode length in frames.
        expected_repo_id: Optional repo ID every row must match.
        expected_revision: Optional revision every row must match.

    Returns:
        List of ``ValidationViolation`` objects; empty when the table is valid.
    """
    violations = _validate_table_header_consistency(
        rows,
        expected_repo_id=expected_repo_id,
        expected_revision=expected_revision,
    )
    if not rows:
        return violations

    spec_index = task_spec.subtask_index
    by_episode: dict[int, list[AnnotationRow]] = defaultdict(list)
    for row in rows:
        by_episode[row.episode_index].append(row)

    for episode_index in sorted(by_episode):
        violations.extend(
            _validate_episode_rows(
                episode_index,
                by_episode[episode_index],
                episode_length=episode_lengths.get(episode_index),
                spec_index=spec_index,
            )
        )

    return violations
