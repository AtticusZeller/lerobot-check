import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot_check.alignment import AlignmentEvaluator
from lerobot_check.collection_progress import CollectionPlan, PlanTarget, load_plan, next_pending

PLAN_PATH = Path(__file__).parent / "fixtures" / "collection-plan.example.json"
FILE_PLAN = load_plan(PLAN_PATH)
PLAN: CollectionPlan = {**FILE_PLAN, "image_size": [32, 24]}
OUTLINE: list[list[float]] = [[10, 8], [13, 8], [13, 11], [10, 11]]


def target(
    name: str = "paper_insert",
    index: int = 1,
    outline: list[list[float]] | None = OUTLINE,
    reason: str | None = None,
) -> PlanTarget:
    result = FILE_PLAN["rounds"][0]["targets"][0].copy()
    result.update(object=name, index=index, outline=outline, reason=reason)
    return result


def test_evaluate_selected_round_from_plan_file():
    plan = load_plan(PLAN_PATH)
    pending = next_pending(plan)
    assert pending is not None
    width, height = plan["image_size"]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    masks = np.zeros((1, height, width), dtype=bool)
    masks[0, 137:169, 110:154] = True
    segmenter = Mock(predict=Mock(return_value=masks))
    evaluator = AlignmentEvaluator(segmenter)

    prepared = evaluator.prepare_round(plan, pending["targets"])
    (result,) = evaluator.evaluate(rgb, prepared)

    assert (result.object, result.index) == ("lid", 3)
    assert result.passed and result.iou is not None
    assert result.observed_outline is not None
    segmenter.predict.assert_called_once()


def test_outline_order_batch_iou_and_threshold():
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    masks = np.zeros((3, 24, 32), dtype=bool)
    masks[0, 8:12, 10:14] = True
    masks[1, 8:12, 11:15] = True
    masks[2, 8:12, 24:28] = True
    segmenter = Mock(predict=Mock(return_value=masks))
    evaluator = AlignmentEvaluator(segmenter, min_iou=0.60)
    prepared = evaluator.prepare_round(
        PLAN,
        [target(), target("box_base", 2), target("iphone", 3), target("box_lid", 4)],
    )
    first, manual, boundary, distant = evaluator.evaluate(rgb, prepared)
    segmenter.predict.assert_called_once_with(rgb, [(2.0, 0.0, 21.0, 19.0)] * 3)
    assert [r.index for r in (first, manual, boundary, distant)] == [1, 2, 3, 4]
    assert first.iou == 1.0 and first.passed and first.status == "aligned"
    assert first.observed_outline is not None
    assert {tuple(p) for p in first.observed_outline} == {tuple(p) for p in OUTLINE}
    assert manual.status == "manual" and not manual.passed and manual.iou is None
    assert boundary.iou == 0.60 and boundary.passed
    assert distant.iou == 0.0 and not distant.passed and distant.status == "misaligned"
    assert "below" in distant.reason and "0.0%" in distant.reason


def test_iou_uses_full_mask_not_rendered_exterior_ring():
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    masks = np.zeros((1, 24, 32), dtype=bool)
    masks[0, 8:12, 10:14] = True
    masks[0, 9:11, 11:13] = False
    masks[0, 20:22, 20:22] = True
    evaluator = AlignmentEvaluator(Mock(predict=Mock(return_value=masks)))
    (result,) = evaluator.evaluate(rgb, evaluator.prepare_round(PLAN, [target()]))
    assert result.iou == 12 / 20
    assert result.observed_outline is not None
    assert len(result.observed_outline) == 4


def test_no_mask_and_manual_only_round():
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    segmenter = Mock(predict=Mock(return_value=np.zeros((1, 24, 32), dtype=bool)))
    evaluator = AlignmentEvaluator(segmenter)
    manual = target("paper_insert", outline=None, reason="Projection unavailable")
    base = target("box_base", 2, reason="Box pose is not reliable")
    prepared = evaluator.prepare_round(PLAN, [manual, base])
    a, b = evaluator.evaluate(rgb, prepared)
    assert (a.status, a.reason, b.status, b.reason) == (
        "manual",
        "Projection unavailable",
        "manual",
        "Box pose is not reliable",
    )
    assert not a.passed and not b.passed
    assert a.observed_outline is None and b.observed_outline is None
    segmenter.predict.assert_not_called()
    (missing,) = evaluator.evaluate(rgb, evaluator.prepare_round(PLAN, [target()]))
    assert missing.status == "undetermined" and not missing.passed
    assert missing.iou is None and missing.observed_outline is None
    assert "No usable object mask" in missing.reason


@pytest.mark.parametrize("rgb", [np.zeros((24, 31, 3), np.uint8), np.zeros((24, 32, 3))])
def test_rejects_non_native_input(rgb):
    segmenter = Mock()
    evaluator = AlignmentEvaluator(segmenter)
    prepared = evaluator.prepare_round(PLAN, [target()])
    with pytest.raises(ValueError, match="native"):
        evaluator.evaluate(rgb, prepared)
    segmenter.predict.assert_not_called()


@pytest.mark.parametrize("outline", [[], [[0, 0], [1, 1], [2, 2]], [[0, 0], [32, 0], [0, 7]]])
def test_rejects_invalid_plan_contours(outline):
    with pytest.raises(ValueError, match="invalid target contour"):
        AlignmentEvaluator(Mock()).prepare_round(PLAN, [target(outline=outline)])


def test_threshold_override_and_default(monkeypatch, tmp_path):
    constructor = Mock()
    monkeypatch.setattr("lerobot_check.alignment.SamSegmenter", constructor)
    config = tmp_path / "thresholds.json"
    config.write_text(json.dumps({"version": 1, "min_iou": 0.75}))
    assert AlignmentEvaluator.from_files(config, device="cpu").min_iou == 0.75
    constructor.assert_called_once_with("cpu", "fp32", local_files_only=True)
    assert AlignmentEvaluator.from_files().min_iou == 0.60


@pytest.mark.parametrize("value", [0, 1.1, True, "0.6", None])
def test_invalid_config_fails_before_model_load(value, monkeypatch, tmp_path):
    constructor = Mock()
    monkeypatch.setattr("lerobot_check.alignment.SamSegmenter", constructor)
    config = tmp_path / "thresholds.json"
    config.write_text(json.dumps({"version": 1, "min_iou": value}))
    with pytest.raises(ValueError, match="min_iou"):
        AlignmentEvaluator.from_files(config)
    constructor.assert_not_called()


def test_alignment_import_does_not_load_model_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import lerobot_check; loaded = set(sys.modules); "
            "import lerobot_check.alignment; "
            "assert not ({'transformers', 'torch'} & (set(sys.modules) - loaded))",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
