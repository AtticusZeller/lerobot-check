"""Tests for examples/egg_carton/convert_cuts.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from lerobot_check.propose import EpisodeProposal, GripperEvent
from lerobot_check.schema import load_task_spec
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


def test_build_annotations_from_cuts_skips_108_and_sets_first_grasp() -> None:
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
                GripperEvent(0, "left", "close", 283, 283 / 30.0, 0.48),
                GripperEvent(0, "left", "open", 404, 404 / 30.0, 0.72),
                GripperEvent(0, "left", "close", 737, 737 / 30.0, 0.49),
                GripperEvent(0, "left", "close", 1084, 1084 / 30.0, 0.45),
                GripperEvent(0, "left", "close", 1185, 1185 / 30.0, 0.40),
            ),
            low_speed_segments=(),
        ),
        1: EpisodeProposal(
            episode_index=1,
            length=1500,
            fps=30.0,
            gripper_events=(
                # No close inside [0, 600); first close in place_eggs is 820.
                GripperEvent(1, "left", "close", 820, 820 / 30.0, 0.44),
            ),
            low_speed_segments=(),
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

    # Episode 0: cut_frame = 900; first close in [0, 900) is 283, in [900, 2000) is 1084.
    assert (rows[0].subtask, rows[0].start, rows[0].end, rows[0].grasp) == (
        "open_carton",
        0,
        900,
        283,
    )
    assert (rows[1].subtask, rows[1].start, rows[1].end, rows[1].grasp) == (
        "place_eggs",
        900,
        2000,
        1084,
    )

    # Episode 1: cut_frame = 600; no close in [0, 600) -> grasp None; in [600, 1500) -> 820.
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
        820,
    )

    assert validate_annotations(rows, task_spec, lengths) == []
