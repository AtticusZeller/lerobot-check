"""Multi-camera episode frame decoding and contact-sheet PNG rendering."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from lerobot_check.dataset import DatasetView, EpisodeRecord

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
_TEXT_COLOR = (255, 255, 255)


def parse_frame_spec(spec: str, episode_length: int) -> list[int]:
    """Parse a comma-separated frame specification into sorted unique frame indices.

    Supports single frames (``"120"``) and inclusive-end slices
    (``"start:end"`` or ``"start:end:step"``).

    Args:
        spec: Comma-separated frame numbers or slices, e.g. ``"100,200,300:360:15"``.
        episode_length: Total frame count of the episode.

    Returns:
        Sorted list of unique frame indices within ``[0, episode_length - 1]``.

    Raises:
        ValueError: If syntax is invalid, step <= 0, or any frame is out of bounds.
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
            start = int(parts[0])
            end = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            if step <= 0:
                raise ValueError(f"Frame range step must be positive in {token!r}")
            if start < 0 or end >= episode_length or start > end:
                raise ValueError(f"Frame range {token!r} is outside [0, {episode_length - 1}]")
            frames.update(range(start, end + 1, step))
        else:
            idx = int(token)
            if not 0 <= idx < episode_length:
                raise ValueError(f"Frame {idx} is outside [0, {episode_length - 1}]")
            frames.add(idx)

    if not frames:
        raise ValueError("Frame specification produced no frames")
    return sorted(frames)


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


def render_contact_sheet(
    frames_by_index: Mapping[int, Mapping[str, Image.Image]],
    *,
    cameras: Sequence[str],
    fps: float,
    tile_width: int = 320,
    tile_height: int | None = None,
    highlight_frames: Collection[int] = (),
) -> Image.Image:
    """Tile multi-camera RGB frames into a single contact sheet with frame labels.

    Each frame occupies one horizontal row containing the requested ``cameras``
    side by side, with a high-contrast frame number and timestamp badge drawn on
    the top-left of the row.

    Args:
        frames_by_index: Nested mapping ``frame_index -> camera_key -> PIL.Image``.
        cameras: Ordered camera keys to display left-to-right.
        fps: Dataset frame rate in Hz used for timestamp labels.
        tile_width: Target width (pixels) of each individual camera tile.
        tile_height: Optional target height (pixels); preserves source aspect ratio
            when ``None``.
        highlight_frames: Optional set of frame indices (e.g. candidate events)
            whose frame label badge is highlighted.

    Returns:
        Rendered RGB ``PIL.Image``.

    Raises:
        ValueError: If ``frames_by_index`` or ``cameras`` is empty, or if a camera
            image is missing for any requested frame.
    """
    if not frames_by_index:
        raise ValueError("frames_by_index must not be empty")
    if not cameras:
        raise ValueError("cameras must not be empty")
    if tile_width <= 0 or (tile_height is not None and tile_height <= 0):
        raise ValueError("tile_width and tile_height must be positive")

    sorted_indices = sorted(frames_by_index)
    first_cam_img = frames_by_index[sorted_indices[0]][cameras[0]]
    resolved_h = (
        tile_height
        if tile_height is not None
        else max(1, round(first_cam_img.size[1] * tile_width / first_cam_img.size[0]))
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

    highlight_set = set(highlight_frames)
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

        badge_text = f"f={frame_idx:05d} ({frame_idx / fps:.2f}s)"
        if frame_idx in highlight_set:
            badge_text += " *"
        text_bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
        pad_x, pad_y = 6, 4
        box_w = (text_bbox[2] - text_bbox[0]) + 2 * pad_x
        box_h = (text_bbox[3] - text_bbox[1]) + 2 * pad_y
        bg = _HIGHLIGHT_BG if frame_idx in highlight_set else _BADGE_BG
        draw.rectangle([(4, y0 + 4), (4 + box_w, y0 + 4 + box_h)], fill=bg)
        draw.text(
            (4 + pad_x - text_bbox[0], y0 + 4 + pad_y - text_bbox[1]),
            badge_text,
            fill=_TEXT_COLOR,
            font=badge_font,
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
) -> list[Path]:
    """Decode requested frames for an episode and write one or more tiled PNGs.

    Args:
        dataset: Open ``DatasetView``.
        episode_index: Zero-based episode index.
        frame_indices: Sorted frame indices to render.
        output_path: Destination PNG path (or prefix when split across multiple sheets).
        cameras: Camera keys to include per frame row.
        tile_width: Width of each camera tile in pixels.
        tile_height: Optional height of each camera tile in pixels.
        max_frames_per_image: Optional page size; when exceeded, writes numbered PNGs
            (``<stem>_001.png``, ``<stem>_002.png``, ...).
        highlight_frames: Optional event frames to visually highlight in badges.

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

    missing_cams = [cam for cam in cameras if cam not in record.videos]
    if missing_cams:
        raise ValueError(
            f"Episode {episode_index} does not contain requested cameras: {missing_cams}"
        )

    decoded_per_cam: dict[str, dict[int, Image.Image]] = {}
    for cam in cameras:
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
        f: {cam: decoded_per_cam[cam][f] for cam in cameras} for f in unique_frames
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
            cameras=cameras,
            fps=dataset.fps,
            tile_width=tile_width,
            tile_height=tile_height,
            highlight_frames=highlight_frames,
        )
        if len(chunks) == 1:
            dest = output_path
        else:
            dest = output_path.with_name(f"{output_path.stem}_{page_idx:03d}{output_path.suffix}")
        sheet.save(dest, format="PNG")
        written.append(dest)

    return written
