"""Tests for TaskSpec and AnnotationRow schema I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from lerobot_check.schema import (
    AnnotationRow,
    CoverageSpec,
    ObjectSpec,
    SubtaskSpec,
    TaskSpec,
    load_annotations,
    load_task_spec,
    parse_task_spec,
    save_annotations,
    save_task_spec,
)

EGG_SPEC_PATH = Path(__file__).resolve().parents[1] / "examples" / "egg_carton" / "task_spec.yaml"


def test_egg_carton_task_spec_has_verbatim_prompts() -> None:
    spec = load_task_spec(EGG_SPEC_PATH)
    assert spec.subtask_names == ("open_carton", "place_eggs")
    assert spec.prompt_for("open_carton") == "Open the egg carton."
    assert (
        spec.prompt_for("place_eggs")
        == "Take out the eggs one by one and place them in the egg tray."
    )
    assert spec.fine_window_s == pytest.approx(1.0)
    assert spec.coverage.xy_bin_m == pytest.approx(0.05)
    assert spec.coverage.yaw_bin_deg == pytest.approx(30.0)


def test_task_spec_yaml_round_trip(tmp_path: Path) -> None:
    spec = TaskSpec(
        name="demo_task",
        subtasks=(
            SubtaskSpec(name="stage_a", prompt="Do stage A."),
            SubtaskSpec(name="stage_b", prompt="Do stage B."),
        ),
        objects=(
            ObjectSpec(
                name="cube",
                subtask="stage_a",
                event="grasp",
                arm="left",
                yaw_symmetry_deg=90.0,
            ),
        ),
        fine_window_s=1.25,
        coverage=CoverageSpec(xy_bin_m=0.04, yaw_bin_deg=15.0, sparse_threshold=5),
    )
    out = tmp_path / "spec.yaml"
    save_task_spec(spec, out)
    loaded = load_task_spec(out)
    assert loaded == spec


@pytest.mark.parametrize(
    "bad_raw",
    [
        "not_a_mapping",
        {"subtasks": []},
        {"subtasks": [{"name": "a", "prompt": "p"}, {"name": "a", "prompt": "q"}]},
        {
            "subtasks": [{"name": "a", "prompt": "p"}],
            "objects": [
                {
                    "name": "obj",
                    "subtask": "missing",
                    "event": "grasp",
                    "arm": "left",
                    "yaw_symmetry_deg": 180,
                }
            ],
        },
        {
            "subtasks": [{"name": "a", "prompt": "p"}],
            "objects": [
                {
                    "name": "obj",
                    "subtask": "a",
                    "event": "invalid",
                    "arm": "left",
                    "yaw_symmetry_deg": 180,
                }
            ],
        },
        {"subtasks": [{"name": "a", "prompt": "p"}], "fine_window_s": 0.0},
    ],
)
def test_parse_task_spec_rejects_invalid_inputs(bad_raw: object) -> None:
    with pytest.raises(ValueError):
        parse_task_spec(bad_raw)


def test_annotations_csv_round_trip(tmp_path: Path) -> None:
    rows = [
        AnnotationRow(
            repo_id="org/ds",
            revision="abc123",
            episode_index=0,
            subtask="open_carton",
            start=0,
            end=100,
            contact=None,
            grasp=None,
            success=True,
            annotator="converted",
            note="",
        ),
        AnnotationRow(
            repo_id="org/ds",
            revision="abc123",
            episode_index=0,
            subtask="place_eggs",
            start=100,
            end=300,
            contact=120,
            grasp=135,
            success=False,
            annotator="agent",
            note="dropped egg",
        ),
    ]
    out = tmp_path / "annotations.csv"
    save_annotations(rows, out)
    loaded = load_annotations(out)
    assert loaded == rows


def test_load_annotations_rejects_missing_column(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("repo_id,episode_index,subtask\norg/ds,0,a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_annotations(bad_csv)


def test_load_annotations_rejects_invalid_annotator(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad_annotator.csv"
    bad_csv.write_text(
        "repo_id,revision,episode_index,subtask,start,end,contact,grasp,success,annotator,note\n"
        "org/ds,rev1,0,open_carton,0,100,,,true,robot,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="annotator"):
        load_annotations(bad_csv)
