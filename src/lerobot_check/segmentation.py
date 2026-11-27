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

    def segment_layout_boxes(
        self,
        image: Image.Image | np.ndarray,
        boxes: Sequence[Sequence[float]],
        *,
        grid_size: int = 4,
        min_iou: float = 0.80,
        prompt_chunk_size: int = 16,
    ) -> list[np.ndarray]:
        if not boxes:
            return []
        if grid_size < 1:
            raise ValueError(f"grid_size must be >= 1, got {grid_size}")
        if prompt_chunk_size < 1:
            raise ValueError(f"prompt_chunk_size must be >= 1, got {prompt_chunk_size}")
        if isinstance(image, np.ndarray) and not image.flags.writeable:
            image = image.copy()
        encoded = self.processor(images=image, return_tensors="pt")
        sizes = encoded["original_sizes"]
        pixel_values = encoded["pixel_values"].to(self.device)
        precision = (
            self.torch.autocast("cuda", dtype=self.torch.bfloat16)
            if self.precision == "bf16"
            else nullcontext()
        )
        results: list[np.ndarray] = []
        with self.torch.inference_mode(), precision:
            image_embeddings = self.model.get_image_embeddings(pixel_values)
            for box in boxes:
                x0, y0, x1, y1 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                points = _interior_grid_points(x0, y0, x1, y1, grid_size)
                best: tuple[int, float, np.ndarray] | None = None
                for start in range(0, len(points), prompt_chunk_size):
                    chunk = points[start : start + prompt_chunk_size]
                    prompt_inputs = self.processor(
                        input_points=[chunk],
                        input_labels=[[[1]] * len(chunk)],
                        original_sizes=sizes,
                        return_tensors="pt",
                    )
                    prompt_inputs.pop("original_sizes", None)
                    prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
                    out = self.model(image_embeddings=image_embeddings, **prompt_inputs)
                    ious = out.iou_scores[0].float().cpu().numpy()
                    masks = self.processor.post_process_masks(
                        [out.pred_masks[0].float().cpu()], sizes
                    )[0].numpy()
                    for pi in range(len(chunk)):
                        for mi in range(masks.shape[1]):
                            mask = np.asarray(masks[pi, mi], dtype=bool)
                            iou = float(ious[pi, mi])
                            area = int(mask.sum())
                            if (
                                area > 0
                                and iou >= min_iou
                                and not _mask_touches_box(mask, x0, y0, x1, y1)
                                and (best is None or (area, iou) > (best[0], best[1]))
                            ):
                                best = (area, iou, mask)
                if best is not None:
                    results.append(best[2])
                    continue
                box_inputs = self.processor(
                    input_boxes=[[[x0, y0, x1, y1]]],
                    original_sizes=sizes,
                    return_tensors="pt",
                )
                box_inputs.pop("original_sizes", None)
                box_inputs = {k: v.to(self.device) for k, v in box_inputs.items()}
                box_out = self.model(image_embeddings=image_embeddings, **box_inputs)
                best_idx = int(box_out.iou_scores[0, 0].argmax().item())
                chosen = box_out.pred_masks[0, :, best_idx : best_idx + 1].float().cpu()
                fallback_mask = self.processor.post_process_masks([chosen], sizes)[0][0, 0].numpy()
                results.append(np.asarray(fallback_mask, dtype=bool))
        return results


def _interior_grid_points(
    x0: float, y0: float, x1: float, y1: float, grid_size: int
) -> list[list[list[float]]]:
    margin_x = (x1 - x0) * 0.15
    margin_y = (y1 - y0) * 0.15
    if grid_size == 1:
        xs = np.array([(x0 + x1) * 0.5], dtype=np.float64)
        ys = np.array([(y0 + y1) * 0.5], dtype=np.float64)
    else:
        xs = np.linspace(x0 + margin_x, x1 - margin_x, grid_size, dtype=np.float64)
        ys = np.linspace(y0 + margin_y, y1 - margin_y, grid_size, dtype=np.float64)
    return [[[float(x), float(y)]] for y in ys for x in xs]


def _mask_touches_box(mask: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> bool:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return False
    return bool(
        (xs <= x0).any() or (xs >= x1 - 1).any() or (ys <= y0).any() or (ys >= y1 - 1).any()
    )
