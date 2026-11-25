"""Multi-camera episode frame decoding and contact-sheet PNG rendering."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from lerobot_check.dataset import DatasetView, EpisodeRecord
from lerobot_check.schema import AnnotationRow, Annotator, TaskSpec

DEFAULT_CAMERAS: tuple[str, ...] = (
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)

_HEADER_HEIGHT = 28
_DIVIDER_PX = 2
_BG_COLOR = (20, 20, 20)
_HEADER_BG = (32, 32, 36)
_DIVIDER_COLOR = (65, 65, 72)
_BADGE_BG = (16, 16, 20)
_HIGHLIGHT_BG = (196, 132, 0)
_BOUNDARY_BG = (220, 38, 38)
_BOUNDARY_BORDER = (239, 68, 68)
_BOUNDARY_BORDER_PX = 4
_TEXT_COLOR = (255, 255, 255)
_BOUNDARY_LABEL_HEIGHT = 26
_BOUNDARY_CAM_GAP_PX = 2
_BOUNDARY_CELL_GAP_PX = 6
_HUMAN_MARKER_BG = (22, 163, 74)
_HUMAN_MARKER_FG = (255, 255, 255)


@dataclass(frozen=True)
class BoundarySpec:
    """Parsed ``--boundary`` selector for human-reviewed exemplar rendering.

    Attributes:
        prev_subtask: Leading subtask name (or the target subtask in grasp mode).
        next_subtask: Following subtask name, or ``None`` in ``<subtask>:grasp`` mode.
        is_grasp: ``True`` for ``<subtask>:grasp`` exemplars, ``False`` for transitions.
    """

    prev_subtask: str
    next_subtask: str | None
    is_grasp: bool = False

    @property
    def raw(self) -> str:
        """Canonical ``<left>:<right>`` boundary string."""
        if self.is_grasp:
            return f"{self.prev_subtask}:grasp"
        return f"{self.prev_subtask}:{self.next_subtask}"

    @property
    def label(self) -> str:
        """Short visual marker label drawn on the target frame."""
        return "GRASP" if self.is_grasp else "BOUNDARY"


@dataclass(frozen=True)
class BoundaryExemplar:
    """One human-reviewed exemplar frame selected from an annotation CSV."""

    episode_index: int
    frame: int
    boundary: BoundarySpec


def parse_boundary_spec(spec: str) -> BoundarySpec:
    """Parse ``<prev_subtask>:<next_subtask>`` or ``<subtask>:grasp``.

    Args:
        spec: Raw ``--boundary`` CLI argument.

    Returns:
        Parsed ``BoundarySpec``.

    Raises:
        ValueError: If ``spec`` does not contain exactly two non-empty colon-separated parts.
    """
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid boundary {spec!r}; expected '<prev_subtask>:<next_subtask>' "
            "or '<subtask>:grasp'."
        )
    left, right = parts
    if right == "grasp":
        return BoundarySpec(prev_subtask=left, next_subtask=None, is_grasp=True)
    return BoundarySpec(prev_subtask=left, next_subtask=right, is_grasp=False)


def resolve_exemplar_cameras(
    boundary: BoundarySpec,
    *,
    explicit_cameras: Sequence[str] | None = None,
    task_spec: TaskSpec | None = None,
) -> tuple[str, ...]:
    """Resolve cameras for a boundary exemplar strip.

    Validates subtask names against ``task_spec`` when ``task_spec`` is provided.
    Explicit ``--cameras`` overrides task-spec cameras; otherwise takes the ordered,
    deduplicated union of the relevant subtask camera lists, falling back to
    ``DEFAULT_CAMERAS`` when none are configured.
    """
    spec_cams: list[str] = []
    seen_short: set[str] = set()
    if task_spec is not None:
        subtasks_to_check = (
            (boundary.prev_subtask,)
            if boundary.is_grasp
            else (boundary.prev_subtask, str(boundary.next_subtask))
        )
        for name in subtasks_to_check:
            for cam in task_spec.cameras_for(name):
                short = _short_camera_name(cam)
                if short not in seen_short:
                    seen_short.add(short)
                    spec_cams.append(cam)

    if explicit_cameras:
        return tuple(explicit_cameras)
    if spec_cams:
        return tuple(spec_cams)
    return DEFAULT_CAMERAS


def select_boundary_exemplars(
    rows: Sequence[AnnotationRow],
    boundary: BoundarySpec,
    *,
    max_episodes: int = 3,
) -> list[BoundaryExemplar]:
    """Select human-reviewed boundary or grasp exemplars in ascending episode order.

    For transition boundaries ``prev:next``, an episode qualifies only when BOTH
    the ``prev`` and ``next`` rows have ``annotator == "human"`` and satisfy
    ``prev.end == next.start``. For ``subtask:grasp``, an episode qualifies when
    that subtask's row has ``annotator == "human"`` and a non-null ``grasp`` frame.
    """
    if max_episodes < 1:
        raise ValueError(f"max_episodes must be >= 1, got {max_episodes}")

    by_episode: dict[int, list[AnnotationRow]] = defaultdict(list)
    for row in rows:
        by_episode[row.episode_index].append(row)

    selected: list[BoundaryExemplar] = []
    for ep_idx in sorted(by_episode):
        ep_rows = by_episode[ep_idx]
        if boundary.is_grasp:
            for row in ep_rows:
                if (
                    row.subtask == boundary.prev_subtask
                    and row.annotator == "human"
                    and row.grasp is not None
                ):
                    selected.append(
                        BoundaryExemplar(
                            episode_index=ep_idx,
                            frame=row.grasp,
                            boundary=boundary,
                        )
                    )
                    break
        else:
            prev_human = [
                r for r in ep_rows if r.subtask == boundary.prev_subtask and r.annotator == "human"
            ]
            next_human = [
                r for r in ep_rows if r.subtask == boundary.next_subtask and r.annotator == "human"
            ]
            matched_frame: int | None = None
            for prev_row in prev_human:
                for next_row in next_human:
                    if prev_row.end == next_row.start:
                        matched_frame = prev_row.end
                        break
                if matched_frame is not None:
                    break
            if matched_frame is not None:
                selected.append(
                    BoundaryExemplar(
                        episode_index=ep_idx,
                        frame=matched_frame,
                        boundary=boundary,
                    )
                )

        if len(selected) >= max_episodes:
            break

    return selected


def select_frames_in_window(
    center: int,
    *,
    episode_length: int,
    window_frames: int,
    stride: int,
) -> list[int]:
    """Sample frames in half-open ``[center - window_frames, center + window_frames)``.

    Steps symmetrically at ``stride`` from ``center`` so ``center`` itself is
    always included, clamped to ``[0, episode_length)``.
    """
    if episode_length <= 0:
        raise ValueError(f"episode_length must be positive, got {episode_length}")
    if not 0 <= center < episode_length:
        raise ValueError(f"center frame {center} is outside [0, {episode_length - 1}]")
    if window_frames < 0:
        raise ValueError(f"window_frames must be >= 0, got {window_frames}")
    if stride <= 0:
        raise ValueError(f"stride must be >= 1, got {stride}")

    low = max(0, center - window_frames)
    high = min(episode_length, center + window_frames)
    selected: set[int] = {center}

    offset = stride
    while center - offset >= low:
        selected.add(center - offset)
        offset += stride

    offset = stride
    while center + offset < high:
        selected.add(center + offset)
        offset += stride

    return sorted(selected)


def parse_frame_spec(spec: str, episode_length: int) -> list[int]:
    """Parse a comma-separated frame specification into sorted unique frame indices.

    Supports single frames (``"120"``) and half-open slices ``[start, end)``
    (``"start:end"``, ``"start:end:step"``, or open-ended slices such as
    ``"::30"`` / ``"0::30"`` where empty start defaults to ``0`` and empty end
    defaults to ``episode_length``), matching annotation table intervals.

    Args:
        spec: Comma-separated frame numbers or half-open slices, e.g.
            ``"100,200,300:360:15"`` or ``"::30"``.
        episode_length: Total frame count of the episode.

    Returns:
        Sorted list of unique frame indices within ``[0, episode_length)``.

    Raises:
        ValueError: If syntax is invalid, step <= 0, a range is empty, or any
            frame is out of bounds.
    """
    if episode_length <= 0:
        raise ValueError(f"episode_length must be positive, got {episode_length}")

    frames: set[int] = set()
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if ":" in token:
            parts = token.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid frame range token: {token!r}")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else episode_length
            step = int(parts[2]) if (len(parts) == 3 and parts[2]) else 1
            if step <= 0:
                raise ValueError(f"Frame range step must be positive in {token!r}")
            if start < 0 or end > episode_length or start >= end:
                raise ValueError(f"Frame range {token!r} is empty or outside [0, {episode_length})")
            frames.update(range(start, end, step))
        else:
            idx = int(token)
            if not 0 <= idx < episode_length:
                raise ValueError(f"Frame {idx} is outside [0, {episode_length - 1}]")
            frames.add(idx)

    if not frames:
        raise ValueError("Frame specification produced no frames")
    return sorted(frames)


def resolve_render_cameras(
    *,
    explicit_cameras: Sequence[str] | None = None,
    subtask: str | None = None,
    task_spec: TaskSpec | None = None,
    columns: int | None = None,
) -> tuple[str, ...]:
    """Select cameras for rendering based on CLI overrides and subtask spec.

    Priority:
    1. Explicit ``--cameras`` always wins.
    2. When ``--subtask`` is provided and the task spec defines ``cameras`` for
       that subtask, use them.
    3. Otherwise fall back to ``(DEFAULT_CAMERAS[0],)`` in single-camera grid
       mode (``columns`` set) or ``DEFAULT_CAMERAS`` in multi-camera strip mode.

    Args:
        explicit_cameras: Explicit camera keys/names from ``--cameras``.
        subtask: Optional subtask name from ``--subtask``.
        task_spec: Task spec used to look up per-subtask cameras.
        columns: Optional grid column count from ``--columns``.

    Returns:
        Ordered tuple of camera keys or short names.

    Raises:
        ValueError: If ``subtask`` is given without ``task_spec``.
        KeyError: If ``subtask`` is not defined in ``task_spec``.
    """
    if explicit_cameras:
        return tuple(explicit_cameras)
    if subtask is not None:
        if task_spec is None:
            raise ValueError("--subtask requires a task spec (--task-spec or default task spec)")
        subtask_cams = task_spec.cameras_for(subtask)
        if subtask_cams:
            return subtask_cams
    if columns is not None:
        return (DEFAULT_CAMERAS[0],)
    return DEFAULT_CAMERAS


def select_frames_around_events(
    event_frames: Sequence[int],
    *,
    episode_length: int,
    window_frames: int,
    stride: int,
) -> list[int]:
    """Expand candidate event frames into a deduplicated, sorted frame list.

    For each event ``e``, samples ``e + k * stride`` within
    ``[e - window_frames, e + window_frames]`` and always includes ``e`` itself,
    clamped to ``[0, episode_length - 1]``.

    Args:
        event_frames: Center frames (e.g. gripper close/open events).
        episode_length: Total frame count of the episode.
        window_frames: Half-window radius in frames (must be >= 0).
        stride: Sampling step in frames (must be >= 1).

    Returns:
        Sorted unique frame indices ready for decoding and rendering.

    Raises:
        ValueError: If parameters are invalid or ``event_frames`` is empty.
    """
    if episode_length <= 0:
        raise ValueError(f"episode_length must be positive, got {episode_length}")
    if window_frames < 0:
        raise ValueError(f"window_frames must be >= 0, got {window_frames}")
    if stride <= 0:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if not event_frames:
        raise ValueError("event_frames must not be empty")

    selected: set[int] = set()
    max_frame = episode_length - 1
    for center in event_frames:
        clamped_center = min(max(center, 0), max_frame)
        selected.add(clamped_center)
        low = max(0, clamped_center - window_frames)
        high = min(max_frame, clamped_center + window_frames)
        # Step symmetrically outward from the event center so the event itself
        # always lands on the sampling grid.
        offset = stride
        while clamped_center - offset >= low:
            selected.add(clamped_center - offset)
            offset += stride
        offset = stride
        while clamped_center + offset <= high:
            selected.add(clamped_center + offset)
            offset += stride

    return sorted(selected)


def _short_camera_name(camera_key: str) -> str:
    prefix = "observation.images."
    return camera_key.removeprefix(prefix)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "DejaVuSansMono-Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _resize_tile(
    image: Image.Image,
    tile_width: int,
    tile_height: int | None,
) -> Image.Image:
    if tile_height is None:
        src_w, src_h = image.size
        computed_h = max(1, round(src_h * tile_width / src_w))
    else:
        computed_h = tile_height
    if image.size == (tile_width, computed_h):
        return image
    return image.resize((tile_width, computed_h), Image.Resampling.BILINEAR)


def _draw_frame_badge(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    frame_idx: int,
    fps: float,
    highlighted: bool,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    boundary_label: str | None = None,
) -> None:
    badge_text = f"f={frame_idx:05d} ({frame_idx / fps:.2f}s)"
    if boundary_label is not None:
        badge_text += f"  {boundary_label}"
        bg = _BOUNDARY_BG
    elif highlighted:
        badge_text += " *"
        bg = _HIGHLIGHT_BG
    else:
        bg = _BADGE_BG
    text_bbox = draw.textbbox((0, 0), badge_text, font=font)
    pad_x, pad_y = 6, 4
    box_w = (text_bbox[2] - text_bbox[0]) + 2 * pad_x
    box_h = (text_bbox[3] - text_bbox[1]) + 2 * pad_y
    draw.rectangle([(x + 4, y + 4), (x + 4 + box_w, y + 4 + box_h)], fill=bg)
    draw.text(
        (x + 4 + pad_x - text_bbox[0], y + 4 + pad_y - text_bbox[1]),
        badge_text,
        fill=_TEXT_COLOR,
        font=font,
    )


def _render_single_camera_grid(
    frames_by_index: Mapping[int, Mapping[str, Image.Image]],
    *,
    camera: str,
    columns: int,
    fps: float,
    tile_width: int,
    resolved_h: int,
    highlight_set: set[int],
    boundary_map: Mapping[int, str],
) -> Image.Image:
    sorted_indices = sorted(frames_by_index)
    num_rows = math.ceil(len(sorted_indices) / columns)
    sheet_w = columns * tile_width + (columns - 1) * _DIVIDER_PX
    sheet_h = _HEADER_HEIGHT + num_rows * resolved_h + max(0, num_rows - 1) * _DIVIDER_PX

    canvas = Image.new("RGB", (sheet_w, sheet_h), color=_BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    header_font = _load_font(max(12, min(18, tile_width // 18)))
    badge_font = _load_font(max(11, min(18, tile_width // 18)))

    draw.rectangle([(0, 0), (sheet_w, _HEADER_HEIGHT)], fill=_HEADER_BG)
    header_text = (
        f"{_short_camera_name(camera)}  |  {len(sorted_indices)} frames "
        f"(f={sorted_indices[0]}..{sorted_indices[-1]})"
    )
    draw.text((8, 6), header_text, fill=_TEXT_COLOR, font=header_font)

    for col_idx in range(1, columns):
        x0 = col_idx * (tile_width + _DIVIDER_PX)
        draw.rectangle(
            [(x0 - _DIVIDER_PX, _HEADER_HEIGHT), (x0 - 1, sheet_h)],
            fill=_DIVIDER_COLOR,
        )
    for row_idx in range(1, num_rows):
        y0 = _HEADER_HEIGHT + row_idx * (resolved_h + _DIVIDER_PX)
        draw.rectangle(
            [(0, y0 - _DIVIDER_PX), (sheet_w, y0 - 1)],
            fill=_DIVIDER_COLOR,
        )

    for cell_idx, frame_idx in enumerate(sorted_indices):
        cam_map = frames_by_index[frame_idx]
        if camera not in cam_map:
            raise ValueError(f"Missing image for frame {frame_idx}, camera {camera!r}")
        row_idx, col_idx = divmod(cell_idx, columns)
        x0 = col_idx * (tile_width + _DIVIDER_PX)
        y0 = _HEADER_HEIGHT + row_idx * (resolved_h + _DIVIDER_PX)
        tile = _resize_tile(cam_map[camera].convert("RGB"), tile_width, resolved_h)
        canvas.paste(tile, (x0, y0))
        boundary_label = boundary_map.get(frame_idx)
        if boundary_label is not None:
            draw.rectangle(
                [(x0, y0), (x0 + tile_width - 1, y0 + resolved_h - 1)],
                outline=_BOUNDARY_BORDER,
                width=_BOUNDARY_BORDER_PX,
            )
        _draw_frame_badge(
            draw,
            x=x0,
            y=y0,
            frame_idx=frame_idx,
            fps=fps,
            highlighted=frame_idx in highlight_set,
            font=badge_font,
            boundary_label=boundary_label,
        )

    return canvas


def render_contact_sheet(
    frames_by_index: Mapping[int, Mapping[str, Image.Image]],
    *,
    cameras: Sequence[str],
    fps: float,
    tile_width: int = 320,
    tile_height: int | None = None,
    highlight_frames: Collection[int] = (),
    columns: int | None = None,
    boundary_frames: Mapping[int, str] | None = None,
) -> Image.Image:
    """Tile RGB frames into a single contact sheet with frame labels.

    By default (``columns=None``), each frame occupies one horizontal row with
    all ``cameras`` side by side. When ``columns`` is set (coarse grid mode),
    ``cameras`` must contain exactly one camera and frames are placed in an
    ``N``-column grid with frame number and timestamp drawn on every tile.

    Args:
        frames_by_index: Nested mapping ``frame_index -> camera_key -> PIL.Image``.
        cameras: Ordered camera keys to display.
        fps: Dataset frame rate in Hz used for timestamp labels.
        tile_width: Target width (pixels) of each individual camera tile.
        tile_height: Optional target height (pixels); preserves source aspect ratio
            when ``None``.
        highlight_frames: Optional set of frame indices (e.g. candidate events)
            whose frame label badge is highlighted.
        columns: Optional grid column count for single-camera coarse contact sheets.
        boundary_frames: Optional mapping ``frame_index -> label`` (e.g. ``"BOUNDARY"``
            or ``"GRASP"``) drawn with a coloured full-row border and prominent badge.

    Returns:
        Rendered RGB ``PIL.Image``.

    Raises:
        ValueError: If arguments are empty/invalid, if ``columns`` is used with
            multiple cameras, or if a camera image is missing for any frame.
    """
    if not frames_by_index:
        raise ValueError("frames_by_index must not be empty")
    if not cameras:
        raise ValueError("cameras must not be empty")
    if tile_width <= 0 or (tile_height is not None and tile_height <= 0):
        raise ValueError("tile_width and tile_height must be positive")
    if columns is not None:
        if columns <= 0:
            raise ValueError(f"columns must be positive, got {columns}")
        if len(cameras) != 1:
            raise ValueError(
                f"Grid layout (--columns {columns}) requires a single camera, "
                f"got {len(cameras)} cameras: {list(cameras)}"
            )

    sorted_indices = sorted(frames_by_index)
    first_cam_img = frames_by_index[sorted_indices[0]][cameras[0]]
    resolved_h = (
        tile_height
        if tile_height is not None
        else max(1, round(first_cam_img.size[1] * tile_width / first_cam_img.size[0]))
    )
    highlight_set = set(highlight_frames)
    boundary_map = dict(boundary_frames) if boundary_frames else {}

    if columns is not None:
        return _render_single_camera_grid(
            frames_by_index,
            camera=cameras[0],
            columns=columns,
            fps=fps,
            tile_width=tile_width,
            resolved_h=resolved_h,
            highlight_set=highlight_set,
            boundary_map=boundary_map,
        )

    num_cols = len(cameras)
    num_rows = len(sorted_indices)
    sheet_w = num_cols * tile_width + (num_cols - 1) * _DIVIDER_PX
    sheet_h = _HEADER_HEIGHT + num_rows * resolved_h + max(0, num_rows - 1) * _DIVIDER_PX

    canvas = Image.new("RGB", (sheet_w, sheet_h), color=_BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    header_font = _load_font(max(12, min(18, tile_width // 20)))
    badge_font = _load_font(max(13, min(20, tile_width // 18)))

    draw.rectangle([(0, 0), (sheet_w, _HEADER_HEIGHT)], fill=_HEADER_BG)
    for col_idx, cam_key in enumerate(cameras):
        x0 = col_idx * (tile_width + _DIVIDER_PX)
        label = _short_camera_name(cam_key)
        draw.text((x0 + 8, 6), label, fill=_TEXT_COLOR, font=header_font)
        if col_idx > 0:
            draw.rectangle(
                [(x0 - _DIVIDER_PX, 0), (x0 - 1, sheet_h)],
                fill=_DIVIDER_COLOR,
            )

    for row_idx, frame_idx in enumerate(sorted_indices):
        y0 = _HEADER_HEIGHT + row_idx * (resolved_h + _DIVIDER_PX)
        if row_idx > 0:
            draw.rectangle(
                [(0, y0 - _DIVIDER_PX), (sheet_w, y0 - 1)],
                fill=_DIVIDER_COLOR,
            )

        cam_map = frames_by_index[frame_idx]
        for col_idx, cam_key in enumerate(cameras):
            if cam_key not in cam_map:
                raise ValueError(f"Missing image for frame {frame_idx}, camera {cam_key!r}")
            x0 = col_idx * (tile_width + _DIVIDER_PX)
            tile = _resize_tile(cam_map[cam_key].convert("RGB"), tile_width, resolved_h)
            canvas.paste(tile, (x0, y0))

        boundary_label = boundary_map.get(frame_idx)
        if boundary_label is not None:
            draw.rectangle(
                [(0, y0), (sheet_w - 1, y0 + resolved_h - 1)],
                outline=_BOUNDARY_BORDER,
                width=_BOUNDARY_BORDER_PX,
            )

        _draw_frame_badge(
            draw,
            x=0,
            y=y0,
            frame_idx=frame_idx,
            fps=fps,
            highlighted=frame_idx in highlight_set,
            font=badge_font,
            boundary_label=boundary_label,
        )

    return canvas


def decode_video_frames_pyav(
    video_path: Path,
    frame_timestamps: Sequence[tuple[int, float]],
    *,
    fps: float,
    tile_width: int | None = None,
    tile_height: int | None = None,
    seek_gap_s: float = 2.0,
) -> dict[int, Image.Image]:
    """Decode only the requested frames from an MP4 file using PyAV.

    Seeks near each cluster of target timestamps and resizes frames immediately
    after decoding so memory stays bounded even for long episodes.

    Args:
        video_path: Path to the H.264/MP4 video file.
        frame_timestamps: Sequence of ``(frame_index, timestamp_in_video_s)`` pairs.
        fps: Video frame rate in Hz used for timestamp matching tolerance.
        tile_width: Optional immediate resize width to minimize memory use.
        tile_height: Optional immediate resize height.
        seek_gap_s: Gap (seconds) above which the decoder performs a keyframe seek
            instead of stepping sequentially.

    Returns:
        Mapping ``frame_index -> PIL.Image`` for every requested frame.

    Raises:
        RuntimeError: If any requested frame cannot be decoded from ``video_path``.
    """
    if not frame_timestamps:
        return {}

    targets = sorted(frame_timestamps, key=lambda item: item[1])
    tolerance_s = (0.5 / fps) + 1e-3
    decoded: dict[int, Image.Image] = {}

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        if time_base is None:
            raise RuntimeError(f"Video stream in {video_path} has no time_base")

        # Group targets into clusters that are close enough to decode in one forward pass.
        clusters: list[list[tuple[int, float]]] = []
        for item in targets:
            if not clusters or (item[1] - clusters[-1][-1][1]) > seek_gap_s:
                clusters.append([item])
            else:
                clusters[-1].append(item)

        for cluster in clusters:
            first_ts = max(0.0, cluster[0][1] - 0.1)
            seek_pts = int(first_ts / float(time_base))
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

            target_idx = 0
            best_img: Image.Image | None = None
            best_err = float("inf")

            def _commit_target(frame_id: int, img: Image.Image | None) -> None:
                if img is None:
                    raise RuntimeError(f"Failed to decode frame {frame_id} from {video_path}")
                if tile_width is not None:
                    decoded[frame_id] = _resize_tile(img, tile_width, tile_height)
                else:
                    decoded[frame_id] = img

            for av_frame in container.decode(stream):
                if av_frame.pts is None:
                    continue
                pts_s = float(av_frame.pts * time_base)

                while target_idx < len(cluster):
                    frame_id, target_ts = cluster[target_idx]
                    err = abs(pts_s - target_ts)

                    if pts_s < target_ts - tolerance_s:
                        break

                    if err <= best_err:
                        best_err = err
                        best_img = av_frame.to_image()

                    # Once decoded PTS passes target_ts, the closest frame has been seen.
                    if pts_s >= target_ts:
                        _commit_target(frame_id, best_img)
                        target_idx += 1
                        best_img = None
                        best_err = float("inf")
                        # Re-evaluate the current frame against the next target in case
                        # two targets fall near the same timestamp.
                        continue
                    break

                if target_idx >= len(cluster):
                    break

            while target_idx < len(cluster):
                frame_id, _ = cluster[target_idx]
                if best_img is not None and best_err <= (1.0 / fps) + 1e-2:
                    _commit_target(frame_id, best_img)
                    target_idx += 1
                    best_img = None
                    best_err = float("inf")
                else:
                    raise RuntimeError(
                        f"Reached end of {video_path} before decoding frame {frame_id} "
                        f"(target timestamp {cluster[target_idx][1]:.3f}s)"
                    )

    return decoded


def _resolve_camera_keys(
    cameras: Sequence[str],
    available: Mapping[str, object],
) -> tuple[str, ...]:
    resolved: list[str] = []
    for cam in cameras:
        if cam in available:
            resolved.append(cam)
        elif f"observation.images.{cam}" in available:
            resolved.append(f"observation.images.{cam}")
        else:
            resolved.append(cam)
    return tuple(resolved)


def render_episode_sheets(
    dataset: DatasetView,
    episode_index: int,
    frame_indices: Sequence[int],
    output_path: Path,
    *,
    cameras: Sequence[str] = DEFAULT_CAMERAS,
    tile_width: int = 320,
    tile_height: int | None = None,
    max_frames_per_image: int | None = None,
    highlight_frames: Collection[int] = (),
    columns: int | None = None,
    boundary_frames: Mapping[int, str] | None = None,
) -> list[Path]:
    """Decode requested frames for an episode and write one or more tiled PNGs.

    Args:
        dataset: Open ``DatasetView``.
        episode_index: Zero-based episode index.
        frame_indices: Sorted frame indices to render.
        output_path: Destination PNG path (or prefix when split across multiple sheets).
        cameras: Camera keys (or short names such as ``"head"``) to render.
        tile_width: Width of each camera tile in pixels.
        tile_height: Optional height of each camera tile in pixels.
        max_frames_per_image: Optional page size; when exceeded, writes numbered PNGs
            (``<stem>_001.png``, ``<stem>_002.png``, ...).
        highlight_frames: Optional event frames to visually highlight in badges.
        columns: Optional grid column count for single-camera coarse scans.
        boundary_frames: Optional mapping ``frame_index -> label`` rendered with a
            coloured full-row border and prominent label badge.

    Returns:
        List of written PNG file paths.
    """
    record: EpisodeRecord = dataset.get_episode(episode_index)
    unique_frames = sorted(set(frame_indices))
    if not unique_frames:
        raise ValueError("frame_indices must not be empty")
    for frame_idx in unique_frames:
        if not 0 <= frame_idx < record.length:
            raise ValueError(
                f"Frame {frame_idx} out of range for episode {episode_index} "
                f"(length={record.length})"
            )

    resolved_cameras = _resolve_camera_keys(cameras, record.videos)
    if columns is not None:
        if columns <= 0:
            raise ValueError(f"columns must be positive, got {columns}")
        if len(resolved_cameras) != 1:
            raise ValueError(
                f"Grid layout (--columns {columns}) requires a single camera, "
                f"got {len(resolved_cameras)} cameras: {list(resolved_cameras)}"
            )

    missing_cams = [cam for cam in resolved_cameras if cam not in record.videos]
    if missing_cams:
        raise ValueError(
            f"Episode {episode_index} does not contain requested cameras: {missing_cams}"
        )

    decoded_per_cam: dict[str, dict[int, Image.Image]] = {}
    for cam in resolved_cameras:
        span = record.videos[cam]
        targets = [(f, span.from_timestamp + f / dataset.fps) for f in unique_frames]
        decoded_per_cam[cam] = decode_video_frames_pyav(
            span.path,
            targets,
            fps=dataset.fps,
            tile_width=tile_width,
            tile_height=tile_height,
        )

    frames_by_index: dict[int, dict[str, Image.Image]] = {
        f: {cam: decoded_per_cam[cam][f] for cam in resolved_cameras} for f in unique_frames
    }

    if max_frames_per_image is not None and max_frames_per_image <= 0:
        raise ValueError(f"max_frames_per_image must be positive, got {max_frames_per_image}")

    chunks: list[list[int]]
    if max_frames_per_image is None or len(unique_frames) <= max_frames_per_image:
        chunks = [unique_frames]
    else:
        chunks = [
            unique_frames[i : i + max_frames_per_image]
            for i in range(0, len(unique_frames), max_frames_per_image)
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for page_idx, chunk in enumerate(chunks, start=1):
        chunk_map = {f: frames_by_index[f] for f in chunk}
        sheet = render_contact_sheet(
            chunk_map,
            cameras=resolved_cameras,
            fps=dataset.fps,
            tile_width=tile_width,
            tile_height=tile_height,
            highlight_frames=highlight_frames,
            columns=columns,
            boundary_frames=boundary_frames,
        )
        if len(chunks) == 1:
            dest = output_path
        else:
            dest = output_path.with_name(f"{output_path.stem}_{page_idx:03d}{output_path.suffix}")
        sheet.save(dest, format="PNG")
        written.append(dest)

    return written


def render_boundary_exemplars(
    dataset: DatasetView,
    exemplars: Sequence[BoundaryExemplar],
    output_dir: Path,
    *,
    cameras: Sequence[str] = DEFAULT_CAMERAS,
    window_s: float = 1.0,
    stride: int = 3,
    tile_width: int = 320,
    tile_height: int | None = None,
) -> list[tuple[BoundaryExemplar, Path]]:
    """Render one marked strip PNG per selected human boundary exemplar.

    Each PNG shows frames in ``[b - window, b + window)`` at ``stride`` with the
    target boundary or grasp frame visibly outlined and labeled.
    """
    if window_s < 0.0:
        raise ValueError(f"window_s must be >= 0, got {window_s}")
    if stride <= 0:
        raise ValueError(f"stride must be >= 1, got {stride}")

    output_dir.mkdir(parents=True, exist_ok=True)
    window_frames = round(window_s * dataset.fps)
    results: list[tuple[BoundaryExemplar, Path]] = []
    for ex in exemplars:
        record = dataset.get_episode(ex.episode_index)
        frame_indices = select_frames_in_window(
            ex.frame,
            episode_length=record.length,
            window_frames=window_frames,
            stride=stride,
        )
        safe_boundary = ex.boundary.raw.replace(":", "__").replace("/", "_")
        dest = output_dir / f"ep{ex.episode_index:04d}_{safe_boundary}_f{ex.frame:05d}.png"
        written = render_episode_sheets(
            dataset,
            ex.episode_index,
            frame_indices,
            dest,
            cameras=cameras,
            tile_width=tile_width,
            tile_height=tile_height,
            max_frames_per_image=None,
            boundary_frames={ex.frame: ex.boundary.label},
        )
        results.append((ex, written[0]))
    return results


@dataclass(frozen=True)
class EndFrame:
    """Selected boundary check frame for one episode."""

    episode_index: int
    frame: int
    annotator: Annotator


@dataclass(frozen=True)
class BoundaryCell:
    """Decoded camera images and metadata for one episode cell on a boundary sheet."""

    episode_index: int
    frame: int
    annotator: Annotator
    images: tuple[Image.Image, ...]


def parse_episode_spec(spec: str) -> list[int]:
    """Parse an episode filter specification with inclusive ranges, e.g. ``5,9,20-28``.

    Args:
        spec: Comma-separated non-negative episode indices or ``start-end`` inclusive ranges.

    Returns:
        Sorted list of unique episode indices.

    Raises:
        ValueError: If ``spec`` is empty or contains malformed or reversed tokens.
    """
    if not spec or not spec.strip():
        raise ValueError("Episode specification must not be empty.")

    episodes: set[int] = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"Empty token in episode specification: {spec!r}")
        if "-" in token:
            parts = [part.strip() for part in token.split("-")]
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError(
                    f"Invalid episode range {token!r}; expected non-negative 'start-end'."
                )
            start_ep, end_ep = int(parts[0]), int(parts[1])
            if start_ep > end_ep:
                raise ValueError(f"Episode range start > end in {token!r}.")
            episodes.update(range(start_ep, end_ep + 1))
        else:
            if not token.isdigit():
                raise ValueError(
                    f"Invalid episode index {token!r}; expected a non-negative integer."
                )
            episodes.add(int(token))

    return sorted(episodes)


def parse_crop_specs(specs: Sequence[str] | None) -> dict[str, tuple[int, int, int, int]]:
    """Parse repeatable ``CAM=x0,y0,x1,y1`` crop specifications.

    Camera names are normalized to short names (stripping ``observation.images.``)
    so crops match whether the user passes short names or full feature keys.

    Args:
        specs: Optional sequence of ``CAM=x0,y0,x1,y1`` strings.

    Returns:
        Mapping ``short_camera_name -> (x0, y0, x1, y1)``.

    Raises:
        ValueError: If syntax is invalid, coordinates are non-integer or negative,
            or ``x0 < x1`` / ``y0 < y1`` is violated.
    """
    if not specs:
        return {}

    crops: dict[str, tuple[int, int, int, int]] = {}
    for raw in specs:
        if "=" not in raw:
            raise ValueError(f"Invalid crop spec {raw!r}; expected 'CAM=x0,y0,x1,y1'.")
        cam_raw, coords_raw = raw.split("=", 1)
        cam = _short_camera_name(cam_raw.strip())
        if not cam:
            raise ValueError(f"Empty camera name in crop spec {raw!r}.")

        parts = [part.strip() for part in coords_raw.split(",")]
        if len(parts) != 4:
            raise ValueError(
                f"Invalid crop box in {raw!r}; expected four comma-separated integers x0,y0,x1,y1."
            )
        try:
            x0, y0, x1, y1 = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"Non-integer crop coordinate in {raw!r}.") from exc

        if x0 < 0 or y0 < 0:
            raise ValueError(f"Crop coordinates must be non-negative in {raw!r}.")
        if not (x0 < x1 and y0 < y1):
            raise ValueError(f"Crop box must satisfy x0 < x1 and y0 < y1 in {raw!r}.")
        crops[cam] = (x0, y0, x1, y1)

    return crops


def resolve_boundary_subtasks(
    task_spec: TaskSpec,
    requested: Sequence[str] | None = None,
) -> list[tuple[int, str]]:
    """Resolve and validate target subtasks for ``boundary-sheet``.

    Args:
        task_spec: Task specification defining subtask order.
        requested: Optional requested subtask names. When omitted, selects every
            subtask except the last one.

    Returns:
        Ordered list of ``(position_1based, subtask_name)`` pairs.

    Raises:
        ValueError: If any requested subtask is unknown, is the last subtask in the
            spec, or if the spec has fewer than 2 subtasks.
    """
    if not task_spec.subtasks:
        raise ValueError("Task spec defines no subtasks.")

    last_name = task_spec.subtasks[-1].name
    if requested is None:
        if len(task_spec.subtasks) < 2:
            raise ValueError(
                f"Task spec only defines one subtask ({last_name!r}); "
                "its end is the episode end and has no interior boundary."
            )
        return [(idx + 1, item.name) for idx, item in enumerate(task_spec.subtasks[:-1])]

    index_map = task_spec.subtask_index
    resolved: list[tuple[int, str]] = []
    seen: set[str] = set()
    for name in requested:
        if name not in index_map:
            valid = list(task_spec.subtask_names[:-1])
            raise ValueError(
                f"Unknown subtask {name!r}; expected one of {valid} "
                f"(all spec subtasks: {list(task_spec.subtask_names)})."
            )
        if name == last_name:
            raise ValueError(
                f"Subtask {name!r} is the last subtask in the task spec; "
                "its end is the episode end and cannot be rendered with boundary-sheet."
            )
        if name not in seen:
            seen.add(name)
            resolved.append((index_map[name] + 1, name))
    return resolved


def select_end_frames(
    rows: Sequence[AnnotationRow],
    subtask: str,
    *,
    offset: int = -1,
    episodes: Collection[int] | None = None,
    episode_lengths: Mapping[int, int] | None = None,
) -> list[EndFrame]:
    """Select one boundary check frame per episode for ``subtask`` in ascending episode order.

    Computes ``row.end + offset`` clamped into ``[row.start, episode_length)``. When
    ``episode_lengths`` is not provided, each episode's length is inferred from the
    maximum ``end`` across that episode's rows. Episodes without ``subtask`` are skipped.

    Args:
        rows: Annotation rows from an annotation CSV.
        subtask: Target subtask name.
        offset: Frame offset relative to ``row.end`` (default ``-1``, the last frame).
        episodes: Optional collection of episode indices to include.
        episode_lengths: Optional mapping ``episode_index -> episode_length``.

    Returns:
        List of ``EndFrame`` records ordered by ascending ``episode_index``.
    """
    by_episode: dict[int, list[AnnotationRow]] = defaultdict(list)
    for row in rows:
        by_episode[row.episode_index].append(row)

    episode_filter = set(episodes) if episodes is not None else None
    selected: list[EndFrame] = []
    for ep_idx in sorted(by_episode):
        if episode_filter is not None and ep_idx not in episode_filter:
            continue
        ep_rows = by_episode[ep_idx]
        target_row = next((row for row in ep_rows if row.subtask == subtask), None)
        if target_row is None:
            continue

        if episode_lengths is not None and ep_idx in episode_lengths:
            ep_length = episode_lengths[ep_idx]
        else:
            ep_length = max(row.end for row in ep_rows)
        if ep_length <= 0:
            raise ValueError(f"Episode {ep_idx} has non-positive length {ep_length}.")

        frame = min(max(target_row.end + offset, target_row.start), ep_length - 1)
        selected.append(
            EndFrame(
                episode_index=ep_idx,
                frame=frame,
                annotator=target_row.annotator,
            )
        )

    return selected


def _crop_and_resize_height(
    image: Image.Image,
    *,
    crop: tuple[int, int, int, int] | None,
    tile_height: int | None,
) -> Image.Image:
    rgb = image.convert("RGB")
    if crop is not None:
        x0, y0, x1, y1 = crop
        w, h = rgb.size
        cx1 = min(x1, w)
        cy1 = min(y1, h)
        if x0 >= cx1 or y0 >= cy1:
            raise ValueError(f"Crop box {crop} falls outside image size {w}x{h}.")
        rgb = rgb.crop((x0, y0, cx1, cy1))

    if tile_height is not None:
        if tile_height <= 0:
            raise ValueError(f"tile_height must be positive, got {tile_height}.")
        src_w, src_h = rgb.size
        if src_h != tile_height:
            new_w = max(1, round(src_w * tile_height / src_h))
            rgb = rgb.resize((new_w, tile_height), Image.Resampling.BILINEAR)
    return rgb


def compose_boundary_sheets(
    cells: Sequence[BoundaryCell],
    *,
    columns: int = 2,
    per_page: int = 12,
    tile_height: int | None = None,
) -> list[Image.Image]:
    """Compose episode boundary cells into one or more RGB contact-sheet images.

    Each episode cell stacks a label bar (``ep<idx> f<frame>`` plus a coloured ``H``
    badge when ``annotator == "human"``) above the episode's camera images placed
    side by side.

    Args:
        cells: Ordered sequence of ``BoundaryCell`` items to tile.
        columns: Number of episode cells per row (must be >= 1).
        per_page: Maximum episode cells per page, or ``0`` to put all cells on one page.
        tile_height: Optional height to resize every camera image to while preserving
            aspect ratio. When ``None``, images keep their input height.

    Returns:
        List of composed RGB ``PIL.Image`` pages.

    Raises:
        ValueError: If layout parameters are invalid or a cell has no images.
    """
    if columns <= 0:
        raise ValueError(f"columns must be positive, got {columns}.")
    if per_page < 0:
        raise ValueError(f"per_page must be >= 0, got {per_page}.")
    if tile_height is not None and tile_height <= 0:
        raise ValueError(f"tile_height must be positive, got {tile_height}.")
    if not cells:
        return []

    prepared: list[tuple[BoundaryCell, tuple[Image.Image, ...]]] = []
    max_cam_w = 0
    max_cam_h = 0
    for cell in cells:
        if not cell.images:
            raise ValueError(f"Episode {cell.episode_index} cell has no images.")
        target_h = tile_height if tile_height is not None else cell.images[0].height
        resized = tuple(
            _crop_and_resize_height(img, crop=None, tile_height=target_h) for img in cell.images
        )
        cam_w = sum(im.width for im in resized) + max(0, len(resized) - 1) * _BOUNDARY_CAM_GAP_PX
        cam_h = max(im.height for im in resized)
        max_cam_w = max(max_cam_w, cam_w)
        max_cam_h = max(max_cam_h, cam_h)
        prepared.append((cell, resized))

    cell_w = max_cam_w
    cell_h = _BOUNDARY_LABEL_HEIGHT + max_cam_h

    if per_page == 0:
        pages = [prepared]
    else:
        pages = [prepared[i : i + per_page] for i in range(0, len(prepared), per_page)]

    font = _load_font(14)
    sheets: list[Image.Image] = []
    for page_items in pages:
        num_rows = math.ceil(len(page_items) / columns)
        sheet_w = columns * cell_w + (columns - 1) * _BOUNDARY_CELL_GAP_PX
        sheet_h = num_rows * cell_h + (num_rows - 1) * _BOUNDARY_CELL_GAP_PX
        canvas = Image.new("RGB", (sheet_w, sheet_h), color=_BG_COLOR)
        draw = ImageDraw.Draw(canvas)

        for idx, (cell, imgs) in enumerate(page_items):
            row_idx, col_idx = divmod(idx, columns)
            x0 = col_idx * (cell_w + _BOUNDARY_CELL_GAP_PX)
            y0 = row_idx * (cell_h + _BOUNDARY_CELL_GAP_PX)

            draw.rectangle(
                [(x0, y0), (x0 + cell_w - 1, y0 + _BOUNDARY_LABEL_HEIGHT - 1)],
                fill=_HEADER_BG,
            )

            label = f"ep{cell.episode_index} f{cell.frame}"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = x0 + 6
            text_y = y0 + (_BOUNDARY_LABEL_HEIGHT - text_h) // 2 - bbox[1]
            draw.text((text_x, text_y), label, fill=_TEXT_COLOR, font=font)

            if cell.annotator == "human":
                marker = "H"
                mbbox = draw.textbbox((0, 0), marker, font=font)
                mw = mbbox[2] - mbbox[0]
                mh = mbbox[3] - mbbox[1]
                pad_x = 5
                pad_y = 2
                pill_w = mw + 2 * pad_x
                pill_h = mh + 2 * pad_y
                pill_x0 = text_x + text_w + 8
                if pill_x0 + pill_w > x0 + cell_w - 2:
                    pill_x0 = max(x0 + 2, x0 + cell_w - 2 - pill_w)
                pill_y0 = y0 + max(2, (_BOUNDARY_LABEL_HEIGHT - pill_h) // 2)
                pill_x1 = min(x0 + cell_w - 2, pill_x0 + pill_w)
                pill_y1 = min(y0 + _BOUNDARY_LABEL_HEIGHT - 2, pill_y0 + pill_h)
                draw.rectangle([(pill_x0, pill_y0), (pill_x1, pill_y1)], fill=_HUMAN_MARKER_BG)
                draw.text(
                    (pill_x0 + pad_x - mbbox[0], pill_y0 + pad_y - mbbox[1]),
                    marker,
                    fill=_HUMAN_MARKER_FG,
                    font=font,
                )

            cx = x0
            img_y0 = y0 + _BOUNDARY_LABEL_HEIGHT
            for im in imgs:
                canvas.paste(im, (cx, img_y0))
                cx += im.width + _BOUNDARY_CAM_GAP_PX

        sheets.append(canvas)

    return sheets


def render_boundary_sheets(
    dataset: DatasetView,
    end_frames: Sequence[EndFrame],
    output_dir: Path,
    *,
    subtask_position: int,
    subtask: str,
    cameras: Sequence[str] = DEFAULT_CAMERAS,
    crops: Mapping[str, tuple[int, int, int, int]] | None = None,
    tile_height: int = 240,
    columns: int = 2,
    per_page: int = 12,
    jpeg_quality: int = 85,
) -> list[Path]:
    """Decode one boundary frame per camera per episode and write JPEG contact sheets.

    Args:
        dataset: Open ``DatasetView``.
        end_frames: Ordered ``EndFrame`` records selected for ``subtask``.
        output_dir: Destination directory for JPEG files.
        subtask_position: 1-based subtask index in the task spec.
        subtask: Subtask name used in output filenames.
        cameras: Camera keys or short names (default: head, left_wrist, right_wrist).
        crops: Optional mapping ``camera_name -> (x0, y0, x1, y1)`` applied before resizing.
        tile_height: Target camera tile height in pixels (default 240).
        columns: Episode cells per row (default 2).
        per_page: Episodes per JPEG image, or ``0`` for all episodes on one image.
        jpeg_quality: JPEG encoding quality (default 85).

    Returns:
        List of written JPEG file paths.
    """
    if subtask_position < 1:
        raise ValueError(f"subtask_position must be >= 1, got {subtask_position}.")
    if tile_height <= 0:
        raise ValueError(f"tile_height must be positive, got {tile_height}.")
    if columns <= 0:
        raise ValueError(f"columns must be positive, got {columns}.")
    if per_page < 0:
        raise ValueError(f"per_page must be >= 0, got {per_page}.")
    if not end_frames:
        return []

    cells: list[BoundaryCell] = []
    for ef in end_frames:
        record = dataset.get_episode(ef.episode_index)
        if not 0 <= ef.frame < record.length:
            raise ValueError(
                f"Frame {ef.frame} out of range for episode {ef.episode_index} "
                f"(length={record.length})."
            )
        resolved_cameras = _resolve_camera_keys(cameras, record.videos)
        missing_cams = [cam for cam in resolved_cameras if cam not in record.videos]
        if missing_cams:
            raise ValueError(
                f"Episode {ef.episode_index} does not contain requested cameras: {missing_cams}"
            )

        cam_images: list[Image.Image] = []
        for cam in resolved_cameras:
            span = record.videos[cam]
            target_ts = span.from_timestamp + ef.frame / dataset.fps
            decoded = decode_video_frames_pyav(
                span.path,
                [(ef.frame, target_ts)],
                fps=dataset.fps,
            )
            crop_box = None
            if crops:
                crop_box = crops.get(cam) or crops.get(_short_camera_name(cam))
            cam_images.append(
                _crop_and_resize_height(
                    decoded[ef.frame],
                    crop=crop_box,
                    tile_height=tile_height,
                )
            )
        cells.append(
            BoundaryCell(
                episode_index=ef.episode_index,
                frame=ef.frame,
                annotator=ef.annotator,
                images=tuple(cam_images),
            )
        )

    sheets = compose_boundary_sheets(
        cells,
        columns=columns,
        per_page=per_page,
        tile_height=tile_height,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for page_idx, sheet in enumerate(sheets, start=1):
        dest = output_dir / f"{subtask_position}_{subtask}_end_p{page_idx:02d}.jpg"
        sheet.save(dest, format="JPEG", quality=jpeg_quality)
        written.append(dest)
    return written
