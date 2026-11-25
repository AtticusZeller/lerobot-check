"""Tests for frame selection and contact-sheet (strip & grid) rendering."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lerobot_check.cli import main
from lerobot_check.render import (
    _BOUNDARY_BORDER,
    DEFAULT_CAMERAS,
    parse_boundary_spec,
    parse_frame_spec,
    render_contact_sheet,
    resolve_exemplar_cameras,
    resolve_render_cameras,
    select_boundary_exemplars,
    select_frames_around_events,
    select_frames_in_window,
)
from lerobot_check.schema import AnnotationRow, SubtaskSpec, TaskSpec, save_annotations


def test_parse_frame_spec_half_open_semantics() -> None:
    # Half-open [start, end): 20:30:5 excludes 30; :60:30 excludes 60.
    assert parse_frame_spec("10, 5, 20:30:5, 10", episode_length=100) == [5, 10, 20, 25]
    assert parse_frame_spec("::30", episode_length=100) == [0, 30, 60, 90]
    assert parse_frame_spec("0::30", episode_length=100) == [0, 30, 60, 90]
    assert parse_frame_spec(":60:30", episode_length=100) == [0, 30]
    # Exact episode_length is legal as the exclusive upper bound.
    assert parse_frame_spec("90:100:3", episode_length=100) == [90, 93, 96, 99]
    assert parse_frame_spec("0:100:25", episode_length=100) == [0, 25, 50, 75]
    # Empty half-open range and out-of-bounds endpoints are rejected.
    with pytest.raises(ValueError, match="empty or outside"):
        parse_frame_spec("10:10", episode_length=100)
    with pytest.raises(ValueError, match="empty or outside"):
        parse_frame_spec("0:101", episode_length=100)
    with pytest.raises(ValueError, match="outside"):
        parse_frame_spec("100", episode_length=100)
    with pytest.raises(ValueError, match="step"):
        parse_frame_spec("10:20:0", episode_length=100)


def test_resolve_render_cameras_per_subtask_and_overrides() -> None:
    spec = TaskSpec(
        name="egg_task",
        subtasks=(
            SubtaskSpec(
                name="open_carton",
                prompt="Open.",
                cameras=("observation.images.right_wrist",),
            ),
            SubtaskSpec(
                name="place_eggs",
                prompt="Place.",
                cameras=("observation.images.left_wrist", "observation.images.head"),
            ),
            SubtaskSpec(name="legacy_stage", prompt="Legacy."),
        ),
    )

    assert resolve_render_cameras(subtask="open_carton", task_spec=spec) == (
        "observation.images.right_wrist",
    )
    assert resolve_render_cameras(subtask="open_carton", task_spec=spec, columns=6) == (
        "observation.images.right_wrist",
    )
    assert resolve_render_cameras(subtask="place_eggs", task_spec=spec) == (
        "observation.images.left_wrist",
        "observation.images.head",
    )
    # Backward compatibility: subtask without configured cameras falls back to defaults.
    assert resolve_render_cameras(subtask="legacy_stage", task_spec=spec) == DEFAULT_CAMERAS
    assert resolve_render_cameras(subtask="legacy_stage", task_spec=spec, columns=4) == (
        DEFAULT_CAMERAS[0],
    )
    # Explicit --cameras always wins over --subtask.
    assert resolve_render_cameras(
        explicit_cameras=("head", "left_wrist", "right_wrist"),
        subtask="open_carton",
        task_spec=spec,
    ) == ("head", "left_wrist", "right_wrist")

    with pytest.raises(ValueError, match="--subtask requires"):
        resolve_render_cameras(subtask="open_carton", task_spec=None)
    with pytest.raises(KeyError):
        resolve_render_cameras(subtask="missing_subtask", task_spec=spec)


def test_select_frames_around_events_clamps_and_includes_center() -> None:
    frames = select_frames_around_events(
        [10, 95],
        episode_length=100,
        window_frames=20,
        stride=15,
    )
    assert 10 in frames
    assert 95 in frames
    assert frames == [10, 25, 80, 95]


def test_render_contact_sheet_strip_dimensions_and_badge_drawing() -> None:
    cams = DEFAULT_CAMERAS
    frames_by_index = {
        30: {
            cams[0]: Image.new("RGB", (640, 480), color=(40, 80, 120)),
            cams[1]: Image.new("RGB", (640, 480), color=(80, 120, 40)),
            cams[2]: Image.new("RGB", (640, 480), color=(120, 40, 80)),
        },
        60: {
            cams[0]: Image.new("RGB", (640, 480), color=(40, 80, 120)),
            cams[1]: Image.new("RGB", (640, 480), color=(80, 120, 40)),
            cams[2]: Image.new("RGB", (640, 480), color=(120, 40, 80)),
        },
    }
    sheet = render_contact_sheet(
        frames_by_index,
        cameras=cams,
        fps=30.0,
        tile_width=160,
        highlight_frames=(60,),
    )
    # 3 tiles of width 160 + 2 dividers of 2 px = 484; height = 28 header + 2*120 + 2 divider = 270
    assert sheet.size == (484, 270)
    arr = np.asarray(sheet)
    row0_badge = arr[32:54, 4:100]
    row1_badge = arr[154:176, 4:100]
    assert not np.array_equal(row0_badge, row1_badge)

    custom = render_contact_sheet(
        frames_by_index,
        cameras=cams,
        fps=30.0,
        tile_width=200,
        tile_height=100,
    )
    assert custom.size == (3 * 200 + 4, 28 + 2 * 100 + 2)


def test_render_contact_sheet_single_camera_grid_layout() -> None:
    cam = DEFAULT_CAMERAS[0]
    # 5 frames in a 3-column grid -> 2 rows (3 tiles on row 0, 2 tiles + 1 blank on row 1)
    frames_by_index = {
        i * 30: {cam: Image.new("RGB", (640, 480), color=(40 + i * 20, 90, 140))} for i in range(5)
    }
    grid = render_contact_sheet(
        frames_by_index,
        cameras=(cam,),
        fps=30.0,
        tile_width=160,
        columns=3,
        highlight_frames=(60,),
    )
    # width = 3*160 + 2*2 = 484; height = 28 header + 2*120 + 1*2 = 270
    assert grid.size == (484, 270)
    arr = np.asarray(grid)
    # Every occupied tile has a badge drawn over its top-left (differs from solid fill color).
    # Cell (0, 1) -> frame 30, fill (60, 90, 140)
    col1_x0 = 160 + 2
    col1_badge = arr[28 + 4 : 28 + 22, col1_x0 + 4 : col1_x0 + 80]
    assert not np.all(col1_badge == np.array([60, 90, 140]))
    # Cell (0, 2) -> frame 60 is highlighted (differs from non-highlighted cell (0, 1) badge)
    col2_x0 = 2 * (160 + 2)
    col2_badge = arr[28 + 4 : 28 + 22, col2_x0 + 4 : col2_x0 + 80]
    assert not np.array_equal(col1_badge, col2_badge)
    # Empty trailing cell (1, 2) remains background color (20, 20, 20)
    row1_y0 = 28 + 120 + 2
    empty_cell = arr[row1_y0 + 10 : row1_y0 + 50, col2_x0 + 10 : col2_x0 + 50]
    assert np.all(empty_cell == np.array([20, 20, 20]))


def test_render_contact_sheet_grid_rejects_multi_camera_or_non_positive_columns() -> None:
    cams = DEFAULT_CAMERAS
    frames_by_index = {
        0: {cam: Image.new("RGB", (320, 240), color=(0, 0, 0)) for cam in cams},
    }
    with pytest.raises(ValueError, match="single camera"):
        render_contact_sheet(frames_by_index, cameras=cams, fps=30.0, columns=4)
    with pytest.raises(ValueError, match="columns must be positive"):
        render_contact_sheet(frames_by_index, cameras=(cams[0],), fps=30.0, columns=0)


def _row(
    ep: int,
    subtask: str,
    start: int,
    end: int,
    *,
    annotator: str = "agent",
    grasp: int | None = None,
) -> AnnotationRow:
    return AnnotationRow(
        repo_id="Xense/test",
        revision="rev1",
        episode_index=ep,
        subtask=subtask,
        start=start,
        end=end,
        contact=None,
        grasp=grasp,
        success=True,
        annotator=annotator,  # type: ignore[arg-type]
    )


def test_parse_boundary_spec_transition_and_grasp() -> None:
    trans = parse_boundary_spec("place_phone:move_lid")
    assert trans.prev_subtask == "place_phone"
    assert trans.next_subtask == "move_lid"
    assert trans.is_grasp is False
    assert trans.raw == "place_phone:move_lid"
    assert trans.label == "BOUNDARY"

    grasp = parse_boundary_spec("place_phone:grasp")
    assert grasp.prev_subtask == "place_phone"
    assert grasp.next_subtask is None
    assert grasp.is_grasp is True
    assert grasp.raw == "place_phone:grasp"
    assert grasp.label == "GRASP"

    for bad in ("place_phone", ":grasp", "place_phone:", "a:b:c", " : "):
        with pytest.raises(ValueError, match="Invalid boundary"):
            parse_boundary_spec(bad)


def test_select_boundary_exemplars_human_only_ordering_and_max() -> None:
    rows = [
        # ep 12: both human, adjacent -> qualifies
        _row(12, "place_phone", 100, 1200, annotator="human"),
        _row(12, "move_lid", 1200, 1500, annotator="human"),
        # ep 0: prev human, next agent -> excluded
        _row(0, "place_phone", 100, 989, annotator="human"),
        _row(0, "move_lid", 989, 1191, annotator="agent"),
        # ep 2: prev agent, next human -> excluded
        _row(2, "place_phone", 100, 900, annotator="agent"),
        _row(2, "move_lid", 900, 1100, annotator="human"),
        # ep 5: both human, non-adjacent -> excluded
        _row(5, "place_phone", 100, 500, annotator="human"),
        _row(5, "move_lid", 510, 800, annotator="human"),
        # ep 10: both human, adjacent -> qualifies
        _row(10, "place_phone", 100, 1000, annotator="human"),
        _row(10, "move_lid", 1000, 1300, annotator="human"),
        # ep 7: both human, adjacent -> qualifies
        _row(7, "place_phone", 100, 700, annotator="human"),
        _row(7, "move_lid", 700, 950, annotator="human"),
        # ep 15: both human, adjacent -> qualifies
        _row(15, "place_phone", 100, 1500, annotator="human"),
        _row(15, "move_lid", 1500, 1800, annotator="human"),
    ]
    boundary = parse_boundary_spec("place_phone:move_lid")

    top3 = select_boundary_exemplars(rows, boundary, max_episodes=3)
    assert [(ex.episode_index, ex.frame) for ex in top3] == [
        (7, 700),
        (10, 1000),
        (12, 1200),
    ]

    top2 = select_boundary_exemplars(rows, boundary, max_episodes=2)
    assert [(ex.episode_index, ex.frame) for ex in top2] == [(7, 700), (10, 1000)]

    all_matches = select_boundary_exemplars(rows, boundary, max_episodes=10)
    assert [(ex.episode_index, ex.frame) for ex in all_matches] == [
        (7, 700),
        (10, 1000),
        (12, 1200),
        (15, 1500),
    ]

    with pytest.raises(ValueError, match="max_episodes"):
        select_boundary_exemplars(rows, boundary, max_episodes=0)


def test_select_boundary_exemplars_grasp_mode() -> None:
    rows = [
        _row(8, "place_phone", 100, 900, annotator="human", grasp=420),
        _row(1, "place_phone", 100, 900, annotator="agent", grasp=350),
        _row(3, "place_phone", 100, 900, annotator="human", grasp=None),
        _row(2, "place_phone", 100, 900, annotator="human", grasp=280),
        _row(5, "place_phone", 100, 900, annotator="human", grasp=310),
    ]
    boundary = parse_boundary_spec("place_phone:grasp")

    selected = select_boundary_exemplars(rows, boundary, max_episodes=2)
    assert [(ex.episode_index, ex.frame) for ex in selected] == [(2, 280), (5, 310)]

    all_selected = select_boundary_exemplars(rows, boundary, max_episodes=5)
    assert [(ex.episode_index, ex.frame) for ex in all_selected] == [
        (2, 280),
        (5, 310),
        (8, 420),
    ]


def test_resolve_exemplar_cameras_union_and_override() -> None:
    spec = TaskSpec(
        name="pack_iphone",
        subtasks=(
            SubtaskSpec(
                name="place_phone",
                prompt="Place phone.",
                cameras=("observation.images.right_wrist", "observation.images.left_wrist"),
            ),
            SubtaskSpec(
                name="move_lid",
                prompt="Move lid.",
                cameras=("observation.images.right_wrist", "observation.images.head"),
            ),
            SubtaskSpec(name="bare_stage", prompt="Bare."),
        ),
    )

    assert resolve_exemplar_cameras(
        parse_boundary_spec("place_phone:move_lid"), task_spec=spec
    ) == (
        "observation.images.right_wrist",
        "observation.images.left_wrist",
        "observation.images.head",
    )
    assert resolve_exemplar_cameras(parse_boundary_spec("place_phone:grasp"), task_spec=spec) == (
        "observation.images.right_wrist",
        "observation.images.left_wrist",
    )
    assert (
        resolve_exemplar_cameras(parse_boundary_spec("bare_stage:grasp"), task_spec=spec)
        == DEFAULT_CAMERAS
    )
    assert resolve_exemplar_cameras(
        parse_boundary_spec("place_phone:move_lid"),
        explicit_cameras=("head",),
        task_spec=spec,
    ) == ("head",)

    with pytest.raises(KeyError):
        resolve_exemplar_cameras(parse_boundary_spec("place_phone:unknown"), task_spec=spec)


def test_select_frames_in_window_half_open_and_centered() -> None:
    frames = select_frames_in_window(989, episode_length=2000, window_frames=30, stride=3)
    assert frames[0] == 959
    assert 989 in frames
    assert frames[-1] == 1016
    assert 1019 not in frames
    assert all(b - a == 3 for a, b in pairwise(frames))

    near_start = select_frames_in_window(2, episode_length=100, window_frames=10, stride=3)
    assert near_start == [2, 5, 8, 11]

    near_end = select_frames_in_window(98, episode_length=100, window_frames=10, stride=3)
    assert near_end == [89, 92, 95, 98]


def test_render_contact_sheet_draws_boundary_border_and_label() -> None:
    cams = DEFAULT_CAMERAS[:2]
    frames_by_index = {
        57: {cam: Image.new("RGB", (320, 240), color=(40, 80, 120)) for cam in cams},
        60: {cam: Image.new("RGB", (320, 240), color=(40, 80, 120)) for cam in cams},
        63: {cam: Image.new("RGB", (320, 240), color=(40, 80, 120)) for cam in cams},
    }
    sheet = render_contact_sheet(
        frames_by_index,
        cameras=cams,
        fps=30.0,
        tile_width=160,
        boundary_frames={60: "BOUNDARY"},
    )
    arr = np.asarray(sheet)
    # Row 1 (frame 60) starts at y0 = 28 + 120 + 2 = 150, height = 120.
    y0 = 28 + 120 + 2
    border_rgb = np.array(_BOUNDARY_BORDER, dtype=np.uint8)
    # Bottom edge of row 1 carries the boundary border across the full strip width.
    assert np.all(arr[y0 + 119, :] == border_rgb)
    # Row 0 (frame 57) does not have the red boundary border.
    assert not np.any(np.all(arr[28 + 119, :] == border_rgb, axis=-1))


def test_cli_exemplars_no_human_exemplars_exits_zero_with_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ds_root = tmp_path / "Xense" / "demo_ds"
    ds_root.mkdir(parents=True)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "name: demo\nsubtasks:\n"
        "  - name: stage_a\n    prompt: A\n    cameras: [observation.images.head]\n"
        "  - name: stage_b\n    prompt: B\n    cameras: [observation.images.right_wrist]\n",
        encoding="utf-8",
    )
    ann_path = tmp_path / "annotations.csv"
    save_annotations(
        [
            _row(0, "stage_a", 0, 50, annotator="human", grasp=20),
            _row(0, "stage_b", 50, 100, annotator="agent", grasp=None),
        ],
        ann_path,
    )

    class _FakeDataset:
        def __init__(self, root: Path, repo_id: str | None = None, revision: str | None = None):
            self.root = root
            self.repo_id = repo_id or "Xense/demo_ds"
            self.revision = revision or "rev1"
            self.fps = 30.0

    monkeypatch.setattr("lerobot_check.cli.DatasetView", _FakeDataset)

    out_dir = tmp_path / "out_exemplars"
    exit_code = main(
        [
            "exemplars",
            "--dataset-root",
            str(ds_root),
            "--annotations",
            str(ann_path),
            "--task-spec",
            str(spec_path),
            "--boundary",
            "stage_a:stage_b",
            "-o",
            str(out_dir),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No human exemplars found" in captured.out
    assert "stage_a:stage_b" in captured.out
    assert not out_dir.exists()
