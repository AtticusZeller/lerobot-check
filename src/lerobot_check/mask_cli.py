from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lerobot_check.calib import load_head_calibration, write_json
from lerobot_check.dataset import DatasetView, resolve_dataset_root
from lerobot_check.json_io import read_json
from lerobot_check.masks import (
    make_bundle,
    make_local_outline,
    read_head_frame,
    sha256_bytes,
)
from lerobot_check.segmentation import MODEL, MODEL_REVISION, SamSegmenter, exterior_outline
from lerobot_check.suggest import (
    DEFAULT_POSE_LAYER,
    POSE_LAYERS,
    get_object_layer,
    validate_pose_layer,
)


def segment(
    image: Image.Image,
    box: list[float],
    positive: list[list[float]],
    negative: list[list[float]],
    device: str,
) -> np.ndarray:
    return SamSegmenter(device).predict(image, [box], positive=positive, negative=negative)[0]


def run_template(args: argparse.Namespace) -> None:
    layer = validate_pose_layer(args.layer)
    calibration = load_head_calibration(args.calib)
    distribution = read_json(args.distribution, args.distribution.read_bytes())
    prov = distribution["provenance"]
    if prov["position_point"] != "tool_point" or (prov["repo_id"], prov["revision"]) != (
        calibration.repo_id,
        calibration.revision,
    ):
        raise ValueError(
            "Distribution must use tool points and the same dataset revision as head.json"
        )
    obj = next((item for item in distribution["objects"] if item["name"] == args.object), None)
    if obj is None:
        raise ValueError(f"Unknown object: {args.object}")
    if obj["arm"] not in prov["calibrated_arms"] or obj["arm"] not in calibration.arms:
        raise ValueError(f"Missing calibrated arm for {args.object}")
    outline_xy_m = None
    if args.reject is not None:
        if not args.reject.strip() or args.output is None or args.preview_only or args.box:
            raise ValueError("--reject requires a reason and --output instead of segmentation")
        source = None
        raw_layer = obj.get("object_layer") if layer == "object" else None
        plane_z = (
            raw_layer.get("plane_z_m")
            if isinstance(raw_layer, Mapping)
            else (None if layer == "object" else obj.get("median_z_m"))
        )
    else:
        if args.episode is None or args.frame is None or args.preview is None:
            raise ValueError("Supply --episode, --frame and --preview for the source image")
        root = resolve_dataset_root(args.dataset_root, prov["repo_id"])
        dataset = DatasetView(root, repo_id=prov["repo_id"])
        if dataset.revision != prov["revision"]:
            raise ValueError("Local dataset and distribution have different revisions")
        image = read_head_frame(dataset, calibration.camera, args.episode, args.frame)
        if image.size != calibration.image_size:
            raise ValueError("Source frame dimensions differ from head calibration")
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        if args.preview_only:
            image.save(args.preview)
            return
        if args.box is None or args.output is None:
            raise ValueError("Supply --box and --output to generate a template")
        x1, y1, x2, y2 = args.box
        width, height = image.size
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("--box must be a nonempty rectangle inside the calibrated image")
        if not math.isfinite(args.height_offset_m):
            raise ValueError("--height-offset-m must be finite")
        layer_data = get_object_layer(obj, layer)
        anchor = next((p for p in layer_data.poses if p["episode_index"] == args.episode), None)
        if anchor is None:
            raise ValueError(f"Episode {args.episode} has no {args.object} {layer}-layer pose")
        if layer_data.plane_z_m is None:
            raise ValueError(f"Object {args.object!r} has no {layer}-layer plane_z_m")
        plane_z = layer_data.plane_z_m
        pixels = exterior_outline(
            segment(image, args.box, args.positive, args.negative, args.device)
        )
        plane = plane_z + args.height_offset_m
        outline_xy_m = make_local_outline(pixels, anchor, plane, calibration, layer=layer).tolist()
        vertices = [tuple(point) for point in pixels.tolist()]
        tinted = Image.new("RGBA", image.size)
        ImageDraw.Draw(tinted).polygon(
            vertices, fill=(34, 211, 238, 65), outline=(34, 211, 238, 255)
        )
        Image.alpha_composite(image.convert("RGBA"), tinted).convert("RGB").save(args.preview)
        source = {
            "episode": args.episode,
            "image_frame": args.frame,
            "event": (
                obj["object_layer"].get("source", "head_mask")
                if layer == "object"
                else obj["event"]
            ),
            "event_frame": anchor["frame"],
            "x_m": anchor["x_m"],
            "y_m": anchor["y_m"],
            "raw_yaw_deg": anchor["raw_yaw_deg"],
            **({"folded_yaw_deg": anchor["folded_yaw_deg"]} if layer == "object" else {}),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "box": args.box,
            "positive": args.positive,
            "negative": args.negative,
        }
    payload = {
        "version": 1,
        "repo_id": prov["repo_id"],
        "revision": prov["revision"],
        "object": args.object,
        "arm": obj["arm"],
        "anchor_layer": layer,
        "z_plane_m": plane_z,
        "height_offset_m": args.height_offset_m,
        "calibration_sha256": sha256_bytes(args.calib.read_bytes()),
        "source": source,
        "outline_xy_m": outline_xy_m,
        "reason": args.reject,
    }
    print(write_json(payload, args.output))


def run_overlays(args: argparse.Namespace) -> None:
    suggestions = args.suggestions.read_bytes()
    result = make_bundle(
        read_json(args.suggestions, suggestions),
        load_head_calibration(args.calib),
        [read_json(path, path.read_bytes()) for path in args.template],
        calibration_sha256=sha256_bytes(args.calib.read_bytes()),
        suggestions_sha256=sha256_bytes(suggestions),
    )
    ready = sum(target["outline"] is not None for target in result["targets"])
    print(f"{write_json(result, args.output)}: {ready}/{len(result['targets'])} projected targets")


def run_mask_command(args: argparse.Namespace) -> int:
    try:
        (run_template if args.command == "mask-template" else run_overlays)(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"[{args.command}] {exc}", file=sys.stderr)
        return 1
    return 0


def configure_template_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--distribution", required=True, type=Path)
    parser.add_argument("--calib", required=True, type=Path)
    parser.add_argument("--object", required=True)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--frame", type=int)
    parser.add_argument("--preview", type=Path)
    parser.add_argument(
        "--preview-only", action="store_true", help="Extract the source RGB without SAM."
    )
    parser.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--positive", type=float, nargs=2, action="append", default=[])
    parser.add_argument("--negative", type=float, nargs=2, action="append", default=[])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--layer", choices=POSE_LAYERS, default=DEFAULT_POSE_LAYER)
    parser.add_argument("--height-offset-m", type=float, default=0.0)
    parser.add_argument(
        "--reject", help="Write a no-mask template with this reason instead of segmenting."
    )
    parser.add_argument("--output", type=Path)
    parser.set_defaults(func=run_mask_command)


def configure_overlays_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suggestions", required=True, type=Path)
    parser.add_argument("--calib", required=True, type=Path)
    parser.add_argument("--template", type=Path, action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.set_defaults(func=run_mask_command)
