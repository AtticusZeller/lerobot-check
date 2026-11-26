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
