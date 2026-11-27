from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from PIL import Image

from lerobot_check.segmentation import MODEL, MODEL_REVISION, SamSegmenter


def test_loads_sam_once_and_decodes_boxes_in_one_forward(monkeypatch):
    model = Mock()
    model.to.return_value.eval.return_value = model
    predictions = torch.arange(6.0).reshape(1, 2, 3, 1, 1)
    model.return_value = SimpleNamespace(
        pred_masks=predictions, iou_scores=torch.tensor([[[0.1, 0.9, 0.2], [0.0, 0.1, 0.9]]])
    )
    processor = Mock()
    inputs = Mock()
    inputs.pop.return_value = [[24, 32]]
    inputs.to.return_value = {"pixel_values": object()}
    processor.return_value = inputs
    processor.post_process_masks.side_effect = lambda masks, _: masks
    build_model = Mock(return_value=model)
    build_processor = Mock(return_value=processor)
    transformers = SimpleNamespace(
        Sam2Processor=SimpleNamespace(from_pretrained=build_processor),
        Sam2Model=SimpleNamespace(from_pretrained=build_model),
    )
    fake_torch = SimpleNamespace(inference_mode=nullcontext)
    monkeypatch.setattr(
        "lerobot_check.segmentation.import_module",
        lambda module: {"transformers": transformers, "torch": fake_torch}[module],
    )
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    boxes = [(0, 0, 8, 10), (13, 10, 22, 20)]
    segmenter = SamSegmenter("cpu", local_files_only=True)
    np.testing.assert_array_equal(segmenter.predict(rgb, boxes).ravel(), [1, 5])
    processor.assert_called_once_with(
        images=rgb, input_boxes=[[[0, 0, 8, 10], [13, 10, 22, 20]]], return_tensors="pt"
    )
    model.assert_called_once_with(**inputs.to.return_value)
    processor.post_process_masks.assert_called_once()
    build_model.assert_called_once_with(MODEL, revision=MODEL_REVISION, local_files_only=True)
    build_processor.assert_called_once_with(MODEL, revision=MODEL_REVISION, local_files_only=True)

    model.return_value = SimpleNamespace(
        pred_masks=predictions[:, :1], iou_scores=torch.tensor([[[0.1, 0.9, 0.2]]])
    )
    image = Image.new("RGB", (32, 24))
    segmenter.predict(image, [boxes[0]], positive=[[3, 3]], negative=[[6, 6]])
    processor.assert_called_with(
        images=image,
        input_boxes=[[[0, 0, 8, 10]]],
        input_points=[[[[3, 3], [6, 6]]]],
        input_labels=[[[1, 0]]],
        return_tensors="pt",
    )
    assert build_model.call_count == build_processor.call_count == 1


def test_segment_layout_boxes_encodes_image_once(monkeypatch):
    model = Mock()
    model.to.return_value.eval.return_value = model
    embeddings = object()
    model.get_image_embeddings.return_value = embeddings

    interior_mask = np.zeros((24, 32), dtype=bool)
    interior_mask[4:8, 4:8] = True
    border_mask = np.zeros((24, 32), dtype=bool)
    border_mask[10:20, 12:22] = True
    fallback_mask = np.zeros((24, 32), dtype=bool)
    fallback_mask[10:20, 12:22] = True

    call_idx = {"n": 0}

    def fake_model(image_embeddings=None, **kwargs):
        assert image_embeddings is embeddings
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx == 0:
            return SimpleNamespace(
                pred_masks=torch.zeros((1, 4, 3, 24, 32)),
                iou_scores=torch.full((1, 4, 3), 0.95),
            )
        if idx == 1:
            return SimpleNamespace(
                pred_masks=torch.zeros((1, 4, 3, 24, 32)),
                iou_scores=torch.full((1, 4, 3), 0.95),
            )
        return SimpleNamespace(
            pred_masks=torch.zeros((1, 1, 3, 24, 32)),
            iou_scores=torch.tensor([[[0.2, 0.85, 0.1]]]),
        )

    model.side_effect = fake_model

    post_call = {"n": 0}

    def fake_post_process(masks, sizes):
        idx = post_call["n"]
        post_call["n"] += 1
        if idx == 0:
            out = np.zeros((4, 3, 24, 32), dtype=bool)
            out[0, 1] = interior_mask
            return [torch.from_numpy(out)]
        if idx == 1:
            out = np.zeros((4, 3, 24, 32), dtype=bool)
            out[:, :] = border_mask
            return [torch.from_numpy(out)]
        return [torch.from_numpy(fallback_mask[None, None])]

    processor = Mock()

    def fake_processor(**kwargs):
        if "images" in kwargs:
            return {"pixel_values": torch.zeros((1, 3, 24, 32)), "original_sizes": [[24, 32]]}
        if "input_points" in kwargs:
            return {
                "input_points": torch.zeros((1, 4, 1, 2)),
                "input_labels": torch.ones((1, 4, 1)),
                "original_sizes": kwargs["original_sizes"],
            }
        return {
            "input_boxes": torch.zeros((1, 1, 4)),
            "original_sizes": kwargs["original_sizes"],
        }

    processor.side_effect = fake_processor
    processor.post_process_masks.side_effect = fake_post_process

    transformers = SimpleNamespace(
        Sam2Processor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
        Sam2Model=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )
    fake_torch = SimpleNamespace(inference_mode=nullcontext)
    monkeypatch.setattr(
        "lerobot_check.segmentation.import_module",
        lambda module: {"transformers": transformers, "torch": fake_torch}[module],
    )

    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    boxes = [(1, 1, 11, 11), (12, 10, 22, 20)]
    segmenter = SamSegmenter("cpu", local_files_only=True)
    masks = segmenter.segment_layout_boxes(rgb, boxes, grid_size=2)
    assert len(masks) == 2
    model.get_image_embeddings.assert_called_once()
    np.testing.assert_array_equal(masks[0], interior_mask)
    np.testing.assert_array_equal(masks[1], fallback_mask)
