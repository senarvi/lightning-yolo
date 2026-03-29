import pytest
import torch

from lightning_yolo.loss import YOLOLoss
from lightning_yolo.target_matching import (
    HighestIoUMatching,
    IoUThresholdMatching,
    SimOTAMatching,
    SizeRatioMatching,
    TALMatching,
    _probability_of_labels,
    _sim_ota_match,
    _tal_match,
)


def test_highest_iou_matching() -> None:
    matcher = HighestIoUMatching(prior_shapes=[(10, 10), (20, 20), (30, 30)], prior_shape_idxs=[1])
    matched_targets, matched_anchors = matcher.match(torch.tensor([[20.0, 20.0], [30.0, 30.0]]))

    assert torch.equal(matched_targets, torch.tensor([True, False]))
    assert torch.equal(matched_anchors, torch.tensor([0]))


def test_iou_threshold_matching() -> None:
    matcher = IoUThresholdMatching(prior_shapes=[(10, 10), (20, 20)], prior_shape_idxs=[0, 1], threshold=0.5)
    matched = matcher.match(torch.tensor([[10.0, 10.0], [30.0, 30.0]]))

    # The first target matches the first anchor.
    assert torch.equal(matched, torch.tensor([[0], [0]]))


def test_size_ratio_matching() -> None:
    matcher = SizeRatioMatching(prior_shapes=[(10, 10), (20, 20)], prior_shape_idxs=[0, 1], threshold=1.5)
    matched = matcher.match(torch.tensor([[10.0, 10.0], [20.0, 20.0]]))

    # The first target matches the first anchor and the second target matches the second anchor.
    assert torch.equal(matched, torch.tensor([[0, 1], [0, 1]]))


def test_sim_ota_match() -> None:
    # For each of the two targets, k will be the sum of the IoUs. 2 and 1 predictions will be selected for the first and
    # the second target respectively.
    ious = torch.tensor([[0.1, 0.2], [0.1, 0.3], [0.9, 0.4], [0.9, 0.1]])
    # Costs will determine that the first and the last prediction will be selected for the first target, and the first
    # prediction will be selected for the second target. The first prediction was selected for two targets, but it will
    # be matched to the best target only (the second one).
    costs = torch.tensor([[0.3, 0.1], [0.5, 0.2], [0.4, 0.5], [0.3, 0.3]])
    matched_preds, matched_targets = _sim_ota_match(costs, ious)

    # The first and the last prediction were matched.
    assert len(matched_preds) == 4
    assert matched_preds[0]
    assert not matched_preds[1]
    assert not matched_preds[2]
    assert matched_preds[3]

    # The first prediction was matched to the target 1 and the last prediction was matched to target 0.
    assert len(matched_targets) == 2
    assert matched_targets[0] == 1
    assert matched_targets[1] == 0


def test_tal_match() -> None:
    align_metric = torch.tensor(
        [
            [0.2, 0.9],
            [0.8, 0.1],
            [0.7, 0.6],
        ]
    )
    ious = torch.tensor(
        [
            [0.3, 0.8],
            [0.9, 0.2],
            [0.5, 0.7],
        ]
    )
    inside_selector = torch.tensor(
        [
            [True, True],
            [True, False],
            [True, True],
        ]
    )
    pred_mask, target_selector = _tal_match(align_metric, ious, inside_selector, topk=1)

    assert torch.equal(pred_mask, torch.tensor([True, True, False]))
    assert torch.equal(target_selector, torch.tensor([1, 0]))


def test_probability_of_labels() -> None:
    pred_probs = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.8, 0.3, 0.4],
        ]
    )
    target_labels = torch.tensor([1, 5], dtype=torch.int64)
    probs = _probability_of_labels(pred_probs, target_labels)

    # For the first target, the label is class 1, so the probabilities are [0.9, 0.3]. For the second target, the label
    # is mapped to class 2, so the probabilities are [0.2, 0.4].
    expected = torch.tensor(
        [
            [0.9, 0.2],
            [0.3, 0.4],
        ]
    )
    torch.testing.assert_close(probs, expected)


def test_probability_of_labels_multiclass() -> None:
    pred_probs = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.8, 0.3, 0.4],
        ]
    )
    target_labels = torch.tensor(
        [
            [True, False, True],
            [False, True, True],
        ]
    )
    probs = _probability_of_labels(pred_probs, target_labels)

    # For the first target, the label mask is [class 0, class 2], so the probabilities are [0.1 + 0.2, 0.8 + 0.4]. For
    # the second target, the label mask is [class 1, class 2], so the probabilities are [0.9 + 0.2, 0.3 + 0.4].
    expected = torch.tensor(
        [
            [0.3, 1.1],
            [1.2, 0.7],
        ]
    )

    torch.testing.assert_close(probs, expected)


def test_sim_ota_matching() -> None:
    matcher = SimOTAMatching(
        prior_shapes=[(2, 2)],
        prior_shape_idxs=[0],
        loss_func=YOLOLoss("iou"),
        spatial_range=1.0,
        size_range=4.0,
    )
    preds = {
        "boxes": torch.tensor([[[[0.0, 0.0, 2.0, 2.0]]]]),
        "confidences": torch.tensor([[[0.0]]]),
        "classprobs": torch.tensor([[[[0.0]]]]),
    }
    targets = {
        "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
        "labels": torch.tensor([0], dtype=torch.int64),
    }
    pred_selector, background_mask, target_selector = matcher(
        preds,
        targets,
        image_size=torch.tensor([2.0, 2.0]),
        input_is_normalized=False,
    )

    # The only prediction matches the only target.
    assert torch.equal(pred_selector, torch.tensor([[[True]]]))
    assert torch.equal(background_mask, torch.tensor([[[False]]]))
    assert torch.equal(target_selector, torch.tensor([0]))


@pytest.mark.parametrize("input_is_normalized", [False, True])
@pytest.mark.parametrize(
    "target_labels",
    [
        torch.tensor([0, 1]),
        torch.tensor([[True, False], [False, True]]),
    ],
    ids=["integer-labels", "boolean-class-mask"],
)
def test_tal_matching(input_is_normalized: bool, target_labels: torch.Tensor) -> None:
    matcher = TALMatching(prior_shapes=[[2, 2]], prior_shape_idxs=[0], topk=1, alpha=0.5, beta=6.0)
    class_logits = torch.tensor([[[[8.0, -8.0]], [[-8.0, 8.0]]]])
    classprobs = class_logits.sigmoid() if input_is_normalized else class_logits
    preds = {
        "boxes": torch.tensor([[[[0.0, 0.0, 1.0, 1.0]], [[1.0, 0.0, 2.0, 1.0]]]]),
        "confidences": torch.zeros((1, 2, 1)),
        "classprobs": classprobs,
    }
    targets = {
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 2.0, 1.0]]),
        "labels": target_labels,
        "polygons": torch.empty((0, 8)),
    }
    image_size = torch.tensor([2.0, 1.0])
    pred_selector, background_mask, target_selector = matcher(
        preds,
        targets,
        image_size,
        input_is_normalized=input_is_normalized,
    )

    # The first prediction matches the first target and the second prediction matches the second target, because they
    # have the same IoU and the same probability of the target labels, but the first prediction has a smaller center
    # distance to the first target and the second prediction has a smaller center distance to the second target.
    anchor_y, anchor_x, anchor_idx = pred_selector
    assert torch.equal(anchor_y, torch.tensor([0, 0]))
    assert torch.equal(anchor_x, torch.tensor([0, 1]))
    assert torch.equal(anchor_idx, torch.tensor([0, 0]))
    assert torch.equal(target_selector, torch.tensor([0, 1]))
    assert background_mask.shape == torch.Size([1, 2, 1])
    assert not background_mask[0, 0, 0]
    assert not background_mask[0, 1, 0]
