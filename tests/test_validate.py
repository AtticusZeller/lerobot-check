"""Tests for every annotation validation rule in the contract."""

from __future__ import annotations

from lerobot_check.schema import AnnotationRow, SubtaskSpec, TaskSpec
from lerobot_check.validate import validate_annotations


def _make_spec(names: tuple[str, ...] = ("open_carton", "place_eggs")) -> TaskSpec:
    return TaskSpec(
        name="test_task",
        subtasks=tuple(SubtaskSpec(name=n, prompt=f"Prompt for {n}.") for n in names),
    )


def _row(
    *,
    episode_index: int = 0,
    subtask: str = "open_carton",
    start: int = 0,
    end: int = 100,
    contact: int | None = None,
    grasp: int | None = None,
    success: bool = True,
    repo_id: str = "org/ds",
    revision: str = "rev1",
) -> AnnotationRow:
    return AnnotationRow(
        repo_id=repo_id,
        revision=revision,
        episode_index=episode_index,
        subtask=subtask,
        start=start,
        end=end,
        contact=contact,
        grasp=grasp,
        success=success,
        annotator="agent",
    )


def test_valid_full_and_omitted_episodes() -> None:
    spec = _make_spec()
    rows = [
        _row(episode_index=0, subtask="open_carton", start=0, end=100, contact=40, grasp=50),
        _row(episode_index=0, subtask="place_eggs", start=100, end=250, contact=120, grasp=130),
    ]
    # Episode 1 is intentionally omitted; validate must allow missing episodes.
    violations = validate_annotations(rows, spec, {0: 250, 1: 300})
    assert violations == []


def test_valid_single_subtask_supplement_episode() -> None:
    spec = _make_spec(("stage_a", "stage_b", "stage_c"))
    rows = [
        # Single-stage supplement covering the entire episode for stage_b only.
        _row(episode_index=0, subtask="stage_b", start=0, end=180, contact=30, grasp=45),
        # Contiguous partial supplement covering stage_b -> stage_c.
        _row(episode_index=1, subtask="stage_b", start=0, end=90),
        _row(episode_index=1, subtask="stage_c", start=90, end=200),
    ]
    violations = validate_annotations(rows, spec, {0: 180, 1: 200})
    assert violations == []


def test_missing_frames_start_middle_and_end() -> None:
    spec = _make_spec()
    rows = [
        # Episode 0: missing leading frames [0, 10)
        _row(episode_index=0, subtask="open_carton", start=10, end=100),
        _row(episode_index=0, subtask="place_eggs", start=100, end=200),
        # Episode 1: gap in the middle [100, 115)
        _row(episode_index=1, subtask="open_carton", start=0, end=100),
        _row(episode_index=1, subtask="place_eggs", start=115, end=200),
        # Episode 2: missing trailing frames [180, 200)
        _row(episode_index=2, subtask="open_carton", start=0, end=100),
        _row(episode_index=2, subtask="place_eggs", start=100, end=180),
    ]
    violations = validate_annotations(rows, spec, {0: 200, 1: 200, 2: 200})
    by_ep = {v.episode_index: v.rule for v in violations}
    assert by_ep == {0: "missing_frames", 1: "missing_frames", 2: "missing_frames"}


def test_overlap_violation() -> None:
    spec = _make_spec()
    rows = [
        _row(episode_index=0, subtask="open_carton", start=0, end=120),
        _row(episode_index=0, subtask="place_eggs", start=100, end=200),
    ]
    violations = validate_annotations(rows, spec, {0: 200})
    assert [v.rule for v in violations] == ["overlap"]


def test_out_of_range_violations() -> None:
    spec = _make_spec()
    rows = [
        # Negative start
        _row(episode_index=0, subtask="open_carton", start=-5, end=100),
        # End exceeds episode length
        _row(episode_index=1, subtask="open_carton", start=0, end=150),
        # Episode index does not exist in dataset metadata
        _row(episode_index=99, subtask="open_carton", start=0, end=50),
        # grasp == end (illegal because [start, end) is half-open)
        _row(episode_index=2, subtask="open_carton", start=0, end=100, grasp=100),
        # contact < start
        _row(episode_index=3, subtask="open_carton", start=0, end=100, contact=-1),
    ]
    violations = validate_annotations(rows, spec, {0: 100, 1: 120, 2: 100, 3: 100})
    rules_by_ep = {(v.episode_index, v.rule) for v in violations}
    assert (0, "out_of_range") in rules_by_ep
    assert (1, "out_of_range") in rules_by_ep
    assert (99, "unknown_episode") in rules_by_ep
    assert (2, "grasp_out_of_range") in rules_by_ep
    assert (3, "contact_out_of_range") in rules_by_ep


def test_contact_after_grasp_violation() -> None:
    spec = _make_spec()
    rows = [
        _row(episode_index=0, subtask="open_carton", start=0, end=200, contact=80, grasp=70),
    ]
    violations = validate_annotations(rows, spec, {0: 200})
    assert [v.rule for v in violations] == ["contact_after_grasp"]


def test_unknown_subtask_violation() -> None:
    spec = _make_spec()
    rows = [
        _row(episode_index=0, subtask="mystery_stage", start=0, end=200),
    ]
    violations = validate_annotations(rows, spec, {0: 200})
    assert [v.rule for v in violations] == ["unknown_subtask"]


def test_wrong_subtask_order_reversed_duplicate_and_skipped() -> None:
    spec = _make_spec(("stage_a", "stage_b", "stage_c"))
    rows = [
        # Reversed order
        _row(episode_index=0, subtask="stage_b", start=0, end=100),
        _row(episode_index=0, subtask="stage_a", start=100, end=200),
        # Duplicate subtask
        _row(episode_index=1, subtask="stage_a", start=0, end=100),
        _row(episode_index=1, subtask="stage_a", start=100, end=200),
        # Non-contiguous skip (stage_a -> stage_c)
        _row(episode_index=2, subtask="stage_a", start=0, end=100),
        _row(episode_index=2, subtask="stage_c", start=100, end=200),
    ]
    violations = validate_annotations(rows, spec, {0: 200, 1: 200, 2: 200})
    assert [(v.episode_index, v.rule) for v in violations] == [
        (0, "wrong_subtask_order"),
        (1, "wrong_subtask_order"),
        (2, "wrong_subtask_order"),
    ]


def test_empty_table_and_inconsistent_dataset_violations() -> None:
    spec = _make_spec()
    assert [v.rule for v in validate_annotations([], spec, {0: 100})] == ["empty_table"]

    rows = [
        _row(episode_index=0, subtask="open_carton", start=0, end=100, revision="rev1"),
        _row(episode_index=1, subtask="open_carton", start=0, end=100, revision="rev2"),
    ]
    violations = validate_annotations(rows, spec, {0: 100, 1: 100})
    assert [v.rule for v in violations] == ["inconsistent_dataset"]
