"""Tests for examples/egg_carton/convert_cuts.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from lerobot_check.propose import EpisodeProposal, GripperEvent
from lerobot_check.schema import SubtaskSpec, TaskSpec, load_task_spec
from lerobot_check.validate import validate_annotations

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "examples" / "egg_carton" / "task_spec.yaml"
CONVERTER_PATH = REPO_ROOT / "examples" / "egg_carton" / "convert_cuts.py"


def _load_converter_module():
    spec = importlib.util.spec_from_file_location("convert_cuts", CONVERTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _close_ev(ep: int, arm: str, start: int, end: int) -> GripperEvent:
    return GripperEvent(
        episode_index=ep,
        arm=arm,
        kind="close",
        frame=start,
        end_frame=end,
        time_s=start / 30.0,
        end_time_s=end / 30.0,
        start_pos=0.40,
        end_pos=0.25,
    )


def test_build_annotations_from_cuts_skips_108_and_uses_task_spec_arm() -> None:
    converter = _load_converter_module()
    task_spec = load_task_spec(SPEC_PATH)

    cuts = {0: 30.0, 1: 20.0, 108: 15.0}
    lengths = {0: 2000, 1: 1500, 108: 1361}
    proposals = {
        0: EpisodeProposal(
            episode_index=0,
            length=2000,
            fps=30.0,
            gripper_events=(
                # Right-arm close before left-arm close must be ignored because
                # egg_carton task_spec objects specify arm="left".
                _close_ev(0, "right", 100, 110),
                _close_ev(0, "left", 277, 285),
                _close_ev(0, "left", 737, 745),
                _close_ev(0, "left", 1048, 1054),
                _close_ev(0, "left", 1369, 1373),
            ),
        ),
        1: EpisodeProposal(
            episode_index=1,
            length=1500,
            fps=30.0,
            gripper_events=(_close_ev(1, "left", 813, 862),),
        ),
    }

    rows = converter.build_annotations_from_cuts(
        cuts,
        task_spec=task_spec,
        episode_lengths=lengths,
        proposals_by_episode=proposals,
        fps=30.0,
        repo_id="Xense/bi_elite_cs66_rt_egg_carton_to_tray_1016_fixed",
        revision="rev1",
    )

    assert len(rows) == 4
    assert {r.episode_index for r in rows} == {0, 1}
    assert all(r.annotator == "converted" for r in rows)
    assert all(r.contact is None for r in rows)

    assert (rows[0].subtask, rows[0].start, rows[0].end, rows[0].grasp) == (
        "open_carton",
        0,
        900,
        277,
    )
    assert (rows[1].subtask, rows[1].start, rows[1].end, rows[1].grasp) == (
        "place_eggs",
        900,
        2000,
        1048,
    )
    assert (rows[2].subtask, rows[2].start, rows[2].end, rows[2].grasp) == (
        "open_carton",
        0,
        600,
        None,
    )
    assert (rows[3].subtask, rows[3].start, rows[3].end, rows[3].grasp) == (
        "place_eggs",
        600,
        1500,
        813,
    )

    assert validate_annotations(rows, task_spec, lengths, expected_revision="rev1") == []


def test_converter_falls_back_to_any_arm_when_objects_omitted() -> None:
    converter = _load_converter_module()
    spec_no_objects = TaskSpec(
        name="generic",
        subtasks=(
            SubtaskSpec(name="stage_a", prompt="A."),
            SubtaskSpec(name="stage_b", prompt="B."),
        ),
    )
    proposals = {
        0: EpisodeProposal(
            episode_index=0,
            length=600,
            fps=30.0,
            gripper_events=(
                _close_ev(0, "right", 120, 130),
                _close_ev(0, "left", 400, 410),
            ),
        )
    }
    rows = converter.build_annotations_from_cuts(
        {0: 10.0},
        task_spec=spec_no_objects,
        episode_lengths={0: 600},
        proposals_by_episode=proposals,
        fps=30.0,
        repo_id="org/ds",
        revision="rev1",
    )
    assert [r.grasp for r in rows] == [120, 400]
