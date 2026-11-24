"""Tests for frame selection and contact-sheet (strip & grid) rendering."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from lerobot_check.render import (
    DEFAULT_CAMERAS,
    parse_frame_spec,
    render_contact_sheet,
    resolve_render_cameras,
    select_frames_around_events,
)
from lerobot_check.schema import SubtaskSpec, TaskSpec


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
