"""Tests for frame selection and multi-camera contact-sheet rendering."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from lerobot_check.render import (
    DEFAULT_CAMERAS,
    parse_frame_spec,
    render_contact_sheet,
    select_frames_around_events,
)


def test_parse_frame_spec_handles_points_and_ranges() -> None:
    assert parse_frame_spec("10, 5, 20:30:5, 10", episode_length=100) == [5, 10, 20, 25, 30]
    with pytest.raises(ValueError, match="outside"):
        parse_frame_spec("100", episode_length=100)
    with pytest.raises(ValueError, match="step"):
        parse_frame_spec("10:20:0", episode_length=100)


def test_select_frames_around_events_clamps_and_includes_center() -> None:
    # window_frames=20 with stride=15 is not a multiple of stride; center must still appear.
    frames = select_frames_around_events(
        [10, 95],
        episode_length=100,
        window_frames=20,
        stride=15,
    )
    assert 10 in frames
    assert 95 in frames
    assert frames == [10, 25, 80, 95]


def test_render_contact_sheet_dimensions_and_badge_drawing() -> None:
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
    # Highlight badge on row 1 (y >= 28 + 120 + 2 = 150) should differ from non-highlight row 0.
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
