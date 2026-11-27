from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from importlib import import_module
from typing import Literal

import cv2
import numpy as np
from PIL import Image

MODEL = "facebook/sam2.1-hiera-tiny"
MODEL_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
Precision = Literal["fp32", "bf16"]


def exterior_outline(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("Segmentation produced an empty mask")
    outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 0.5, True).reshape(-1, 2)
    if len(outline) < 3 or cv2.contourArea(outline) <= 0:
        raise ValueError("Segmentation produced a degenerate outline")
    return outline.astype(np.float64)


class SamSegmenter:
    """Reuse the image model, and encode each frame once for all its box prompts."""

    def __init__(
        self, device: str, precision: Precision = "fp32", *, local_files_only: bool = False
    ) -> None:
        try:
            transformers = import_module("transformers")
        except ModuleNotFoundError as exc:
            if exc.name != "transformers":
                raise
            raise ValueError("Install SAM with: uv sync --extra mask-template") from exc
        self.torch = import_module("torch")
        self.device = device
        self.precision = precision
        if precision == "bf16" and (
            not device.startswith("cuda") or not self.torch.cuda.is_bf16_supported()
        ):
            raise ValueError("BF16 segmentation requires a BF16-capable CUDA device")
        self.processor = transformers.Sam2Processor.from_pretrained(
            MODEL, revision=MODEL_REVISION, local_files_only=local_files_only
        )
        model = transformers.Sam2Model.from_pretrained(
            MODEL, revision=MODEL_REVISION, local_files_only=local_files_only
        )
        self.model = model.to(device).eval()

    def predict(
        self,
        image: Image.Image | np.ndarray,
        boxes: Sequence[Sequence[float]],
        *,
        positive: list[list[float]] | None = None,
        negative: list[list[float]] | None = None,
    ) -> np.ndarray:
        """Return one highest model-scoring boolean mask per prompt, at native resolution."""
        pos, neg = positive or [], negative or []
        prompts = (
            {"input_points": [[pos + neg]], "input_labels": [[[1] * len(pos) + [0] * len(neg)]]}
            if pos or neg
            else {}
        )
        if isinstance(image, np.ndarray) and not image.flags.writeable:
            image = image.copy()
        inputs = self.processor(
            images=image, input_boxes=[[list(box) for box in boxes]], return_tensors="pt", **prompts
        )
        sizes = inputs.pop("original_sizes")
        precision = (
            self.torch.autocast("cuda", dtype=self.torch.bfloat16)
            if self.precision == "bf16"
            else nullcontext()
        )
        with self.torch.inference_mode(), precision:
            result = self.model(**inputs.to(self.device))
            best = result.iou_scores[0].argmax(-1)
            chosen = result.pred_masks[0, range(len(boxes)), best][:, None]
            return self.processor.post_process_masks([chosen.float().cpu()], sizes)[0][:, 0].numpy()
