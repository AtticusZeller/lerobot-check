from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np

from lerobot_check.collection_progress import CollectionPlan, PlanTarget
from lerobot_check.segmentation import Precision, SamSegmenter, exterior_outline

DEFAULT_MIN_IOU = 0.60


class Segmenter(Protocol):
    def predict(self, image: np.ndarray, boxes: Sequence[Sequence[float]]) -> np.ndarray: ...


@dataclass(frozen=True)
class AlignmentResult:
    """One target's native-image contour, overlap, decision and display-ready reason."""

    object: str
    index: int
    status: Literal["aligned", "misaligned", "undetermined", "manual"]
    reason: str
    passed: bool = False
    observed_outline: list[list[float]] | None = None
    iou: float | None = None


@dataclass(frozen=True)
class _Target:
    object: str
    index: int
    mask: np.ndarray
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class PreparedRound:
    """Reusable native-image masks and prompts for one collection round."""

    image_size: tuple[int, int]
    targets: tuple[_Target | AlignmentResult, ...]


class AlignmentEvaluator:
    """Evaluate a round's visible 2D overlap without access to a camera or a dataset."""

    def __init__(self, segmenter: Segmenter, *, min_iou: float = DEFAULT_MIN_IOU) -> None:
        if not math.isfinite(min_iou) or not 0 < min_iou <= 1:
            raise ValueError("min_iou must be in (0, 1]")
        self.segmenter = segmenter
        self.min_iou = min_iou

    @classmethod
    def from_files(
        cls,
        thresholds_json: Path | None = None,
        *,
        device: str = "cuda",
        precision: Precision = "fp32",
    ) -> AlignmentEvaluator:
        """Load an optional v1 threshold override and initialize SAM from the local cache."""
        threshold = DEFAULT_MIN_IOU if thresholds_json is None else _read_threshold(thresholds_json)
        return cls(SamSegmenter(device, precision, local_files_only=True), min_iou=threshold)

    def prepare_round(self, plan: CollectionPlan, targets: Sequence[PlanTarget]) -> PreparedRound:
        """Prepare a round from a validated v1 plan without opening files."""
        width, height = plan["image_size"]
        image_size = (width, height)
        return PreparedRound(image_size, tuple(_prepare(target, image_size) for target in targets))

    def evaluate(self, rgb: np.ndarray, prepared: PreparedRound) -> tuple[AlignmentResult, ...]:
        """Return one result per target in input order for a native H-by-W RGB uint8 frame."""
        width, height = prepared.image_size
        if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"Expected native {width}×{height} RGB uint8 frame")
        active = [target for target in prepared.targets if isinstance(target, _Target)]
        if not active:
            return tuple(item for item in prepared.targets if isinstance(item, AlignmentResult))
        masks = np.asarray(self.segmenter.predict(rgb, [t.box for t in active]), dtype=bool)
        if masks.shape != (len(active), height, width):
            raise RuntimeError("Segmenter did not return one native mask per target")
        measured = iter(self._measure(t, mask) for t, mask in zip(active, masks, strict=True))
        return tuple(
            item if isinstance(item, AlignmentResult) else next(measured)
            for item in prepared.targets
        )

    def _measure(self, target: _Target, mask: np.ndarray) -> AlignmentResult:
        try:
            outline = exterior_outline(mask)
        except ValueError:
            return AlignmentResult(
                target.object, target.index, "undetermined", "No usable object mask found"
            )
        iou = np.count_nonzero(target.mask & mask) / np.count_nonzero(target.mask | mask)
        passed = iou >= self.min_iou
        return AlignmentResult(
            target.object,
            target.index,
            "aligned" if passed else "misaligned",
            (
                f"Mask IoU {iou:.1%} meets {self.min_iou:.1%}"
                if passed
                else f"Mask IoU {iou:.1%} below {self.min_iou:.1%}; move toward the target"
            ),
            passed,
            outline.tolist(),
            iou,
        )


def _read_threshold(path: Path) -> float:
    try:
        config = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read alignment thresholds {path}: {exc}") from exc
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError(f"{path}: expected alignment thresholds version 1")
    threshold = config.get("min_iou")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 < threshold <= 1
    ):
        raise ValueError(f"{path}: min_iou must be in (0, 1]")
    return float(threshold)


def _prepare(target: PlanTarget, size: tuple[int, int]) -> _Target | AlignmentResult:
    name, index = target["object"], target["index"]
    outline = target["outline"]
    if name == "box_base" or outline is None:
        return AlignmentResult(
            name,
            index,
            "manual",
            target["reason"] or "No reliable target outline; confirm manually",
        )
    width, height = size
    pixels = np.asarray(outline, dtype=float)
    if (
        pixels.ndim != 2
        or pixels.shape[1] != 2
        or len(pixels) < 3
        or not np.isfinite(pixels).all()
        or np.any((pixels < 0) | (pixels >= [width, height]))
        or cv2.contourArea(pixels.astype(np.float32)) <= 0
    ):
        raise ValueError(f"{name}: invalid target contour")
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(pixels * 256).astype(np.int32)], 1, shift=8)
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    padding = np.maximum(8.0, 0.20 * (high - low))
    x0, y0 = np.maximum(low - padding, 0).tolist()
    x1, y1 = np.minimum(high + padding, [width - 1, height - 1]).tolist()
    return _Target(name, index, mask.view(bool), (x0, y0, x1, y1))
