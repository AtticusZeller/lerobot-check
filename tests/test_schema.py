"""Tests for TaskSpec, AnnotationRow schema I/O, and shared default paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot_check.dataset import (
    dataset_lerobot_root,
    default_annotations_path,
    default_calib_frames_path,
    default_calib_head_path,
    default_calib_intrinsics_path,
    default_calib_points_path,
    default_distribution_path,
    default_object_poses_path,
    default_suggestions_path,
    default_targets_path,
    default_task_spec_path,
    format_dataset_info,
)
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
    assert spec.cameras_for("open_carton") == ("observation.images.right_wrist",)
    assert spec.cameras_for("place_eggs") == ("observation.images.left_wrist",)
    assert spec.arms_for("open_carton") == ("right",)
    assert spec.arms_for("place_eggs") == ("left",)
    assert spec.coverage.xy_bin_m == pytest.approx(0.05)
    assert spec.coverage.yaw_bin_deg == pytest.approx(30.0)


def test_task_spec_yaml_round_trip(tmp_path: Path) -> None:
    spec = TaskSpec(
        name="demo_task",
        subtasks=(
            SubtaskSpec(name="stage_a", prompt="Do stage A.", arms=("left", "right")),
            SubtaskSpec(name="stage_b", prompt="Do stage B."),
        ),
        objects=(
            ObjectSpec(
                name="cube",
                subtask="stage_b",
                event="grasp",
                arm="left",
                yaw_symmetry_deg=90.0,
            ),
        ),
        coverage=CoverageSpec(xy_bin_m=0.04, yaw_bin_deg=15.0, sparse_threshold=5),
    )
    out = tmp_path / "spec.yaml"
    save_task_spec(spec, out)
    loaded = load_task_spec(out)
    assert loaded == spec
    assert loaded.arms_for("stage_a") == ("left", "right")
    assert loaded.arms_for("stage_b") == ("left",)


def test_deprecated_fine_window_s_warns_without_failing() -> None:
    with pytest.warns(UserWarning, match="fine_window_s"):
        spec = parse_task_spec(
            {
                "name": "legacy_task",
                "subtasks": [{"name": "a", "prompt": "Do A."}],
                "fine_window_s": 1.0,
            }
        )
    assert spec.name == "legacy_task"
    assert spec.subtask_names == ("a",)


def test_task_spec_arms_validation() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        parse_task_spec({"subtasks": [{"name": "a", "prompt": "p", "arms": "left"}]})
    with pytest.raises(ValueError, match="must be one of"):
        parse_task_spec({"subtasks": [{"name": "a", "prompt": "p", "arms": ["middle"]}]})
    with pytest.raises(ValueError, match="duplicate arm"):
        parse_task_spec({"subtasks": [{"name": "a", "prompt": "p", "arms": ["left", "left"]}]})


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
            contact=None,
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


def test_task_spec_per_subtask_cameras_round_trip_and_backward_compat(tmp_path: Path) -> None:
    legacy_spec = parse_task_spec(
        {
            "name": "legacy",
            "subtasks": [
                {"name": "stage_a", "prompt": "Do A."},
                {"name": "stage_b", "prompt": "Do B."},
            ],
        }
    )
    assert legacy_spec.cameras_for("stage_a") == ()
    assert legacy_spec.cameras_for("stage_b") == ()

    spec = TaskSpec(
        name="with_cameras",
        subtasks=(
            SubtaskSpec(
                name="open_carton",
                prompt="Open the egg carton.",
                cameras=("observation.images.right_wrist",),
            ),
            SubtaskSpec(
                name="place_eggs",
                prompt="Place eggs.",
                cameras=("observation.images.left_wrist", "observation.images.head"),
            ),
        ),
    )
    out = tmp_path / "with_cams.yaml"
    save_task_spec(spec, out)
    loaded = load_task_spec(out)
    assert loaded == spec

    with pytest.raises(ValueError, match="must be a list"):
        parse_task_spec(
            {
                "subtasks": [
                    {"name": "a", "prompt": "p", "cameras": "observation.images.head"},
                ]
            }
        )
    with pytest.raises(ValueError, match="duplicate camera"):
        parse_task_spec(
            {
                "subtasks": [
                    {
                        "name": "a",
                        "prompt": "p",
                        "cameras": ["observation.images.head", "observation.images.head"],
                    },
                ]
            }
        )


def test_default_xense_viewer_paths_and_format_info(tmp_path: Path) -> None:
    lerobot_root = tmp_path / "lerobot"
    ds_root = lerobot_root / "Xense" / "my_dataset"
    sha = "fd0f68ff3e923887f23b153412cd3faa8c441717"
    trees_dir = ds_root / ".cache" / "huggingface" / "trees"
    trees_dir.mkdir(parents=True)
    (trees_dir / f"{sha}.json").write_text("{}", encoding="utf-8")

    viewer_root = lerobot_root.resolve() / ".xense-viewer"
    assert dataset_lerobot_root(ds_root) == lerobot_root.resolve()
    assert default_task_spec_path(ds_root) == (
        viewer_root / "subtasks" / "Xense__my_dataset.task_spec.yaml"
    )
    assert default_annotations_path(ds_root) == (
        viewer_root / "subtasks" / f"Xense__my_dataset@{sha}.csv"
    )
    assert default_distribution_path(ds_root) == (
        viewer_root / "distribution" / f"Xense__my_dataset@{sha}.json"
    )
    assert default_object_poses_path(ds_root) == (
        viewer_root / "object-poses" / f"Xense__my_dataset@{sha}.json"
    )
    assert default_suggestions_path(ds_root) == (
        viewer_root / "suggestions" / f"Xense__my_dataset@{sha}.json"
    )
    assert default_targets_path(ds_root) == viewer_root / "targets" / "Xense__my_dataset.json"
    assert default_calib_intrinsics_path(ds_root) == (
        viewer_root / "calib" / "Xense__my_dataset.intrinsics.json"
    )
    assert default_calib_frames_path(ds_root) == (
        viewer_root / "calib" / "Xense__my_dataset.frames.json"
    )
    assert default_calib_points_path(ds_root) == (
        viewer_root / "calib" / "Xense__my_dataset.points.json"
    )
    assert default_calib_head_path(ds_root) == (
        viewer_root / "calib" / "Xense__my_dataset.head.json"
    )

    fake_dataset = SimpleNamespace(
        root=ds_root.resolve(),
        repo_id="Xense/my_dataset",
        revision=sha,
        fps=30.0,
        total_episodes=2,
        camera_keys=("observation.images.head", "observation.images.right_wrist"),
        episode_lengths={0: 300, 1: 450},
    )
    info_text = format_dataset_info(fake_dataset)  # type: ignore[arg-type]
    assert "repo_id: Xense/my_dataset" in info_text
    assert f".xense-viewer/distribution/Xense__my_dataset@{sha}.json" in info_text
    assert f".xense-viewer/suggestions/Xense__my_dataset@{sha}.json" in info_text
    assert ".xense-viewer/calib/Xense__my_dataset.head.json" in info_text
    assert "default_report_dir" not in info_text
