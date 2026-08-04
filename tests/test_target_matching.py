import pytest
import torch
from torchvision.ops import box_iou, complete_box_iou

from lightning_yolo.batching import pack_targets
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
    pred_mask, target_selector, assignment_weights = _tal_match(
        align_metric.unsqueeze(0),
        ious.unsqueeze(0),
        inside_selector.unsqueeze(0),
        torch.tensor([[True, True]]),
        topk=1,
    )

    assert torch.equal(pred_mask[0], torch.tensor([True, True, False]))
    assert torch.equal(target_selector[0][pred_mask[0]], torch.tensor([1, 0]))
    assert assignment_weights.sum().item() == pytest.approx(1.7)


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
        prior_shape_idxs=[[0]],
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
    empty_targets = {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty(0, dtype=torch.int64),
    }
    # One feature level with two images in the batch.
    result = matcher(
        [[preds, preds]],
        pack_targets([targets, empty_targets]),
        image_size=torch.tensor([2.0, 2.0]),
        input_is_normalized=False,
    )

    matched, empty = result.images
    # The only prediction matches the only target.
    assert torch.equal(matched.foreground, torch.tensor([0]))
    assert torch.equal(matched.background, torch.tensor([False]))
    assert torch.equal(matched.target_boxes, targets["boxes"])
    assert torch.equal(matched.target_labels, targets["labels"])
    # The second image has no targets, so nothing is matched and every anchor is background.
    assert empty.foreground.numel() == 0
    assert torch.equal(empty.background, torch.tensor([True]))
    assert result.assignment_weight_sum == 1


def test_sim_ota_matching_pools_levels() -> None:
    matcher = SimOTAMatching(
        prior_shapes=[(2, 2)],
        prior_shape_idxs=[[0], [0]],
        loss_func=YOLOLoss("iou"),
        spatial_range=1.0,
        size_range=4.0,
    )
    level_preds = {
        "boxes": torch.tensor([[[[0.0, 0.0, 2.0, 2.0]]]]),
        "confidences": torch.tensor([[[0.0]]]),
        "classprobs": torch.tensor([[[[0.0]]]]),
    }
    targets = {
        "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
        "labels": torch.tensor([0], dtype=torch.int64),
    }

    # Two levels each contribute one candidate that perfectly overlaps the target. Dynamic-k is computed over the
    # pooled candidates (k = clipped sum of the top IoUs = 2), so both pooled anchors are matched.
    result = matcher(
        [[level_preds], [level_preds]],
        pack_targets([targets]),
        image_size=torch.tensor([2.0, 2.0]),
    )

    (image,) = result.images
    # Foreground indices address the concatenated anchors of both levels (level 1's anchor is offset by one).
    assert torch.equal(image.foreground.sort().values, torch.tensor([0, 1]))
    assert torch.equal(image.background, torch.tensor([False, False]))
    assert image.target_boxes.shape == (2, 4)
    assert result.assignment_weight_sum == 2


def test_tal_matching() -> None:
    matcher = TALMatching(topk=1, alpha=0.5, beta=6.0)
    top5_matcher = TALMatching(topk=5, alpha=0.5, beta=6.0)
    top10_matcher = TALMatching(topk=10, alpha=0.5, beta=6.0)

    one_target_boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    one_target = {
        "boxes": one_target_boxes,
        "labels": torch.tensor([0], dtype=torch.int64),
    }
    two_inside_points = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    two_identical_boxes = torch.tensor([[[[0.0, 0.0, 4.0, 4.0]], [[0.0, 0.0, 4.0, 4.0]]]])
    two_point_preds = {
        "boxes": two_identical_boxes,
        "confidences": torch.ones((1, 2, 1)),
        "classprobs": torch.full((1, 2, 1, 1), 8.0),
    }

    # Unmatched valid points should be supervised as background.
    background_result = matcher(
        [
            {
                "boxes": two_identical_boxes,
                "confidences": torch.ones((1, 2, 1)),
                "classprobs": torch.tensor([[[[8.0]], [[7.0]]]]),
            }
        ],
        pack_targets([one_target]),
        anchor_points=two_inside_points,
    )
    assert background_result.foreground.sum() == 1
    assert torch.equal(background_result.background, ~background_result.foreground)

    # Complete IoU should be clamped and assignment outputs detached from autograd.
    clamp_result = matcher(
        [
            {
                "boxes": torch.tensor([[[[100.0, 100.0, 104.0, 104.0]]]], requires_grad=True),
                "confidences": torch.ones((1, 1, 1)),
                "classprobs": torch.tensor([[[[8.0]]]], requires_grad=True),
            }
        ],
        pack_targets([one_target]),
        anchor_points=torch.tensor([[2.0, 2.0]]),
    )
    assert clamp_result.foreground.item()
    torch.testing.assert_close(clamp_result.assignment_weights, torch.zeros_like(clamp_result.assignment_weights))
    assert not clamp_result.assignment_weights.requires_grad
    assert not clamp_result.target_boxes.requires_grad
    assert not clamp_result.target_labels.requires_grad

    # Only points strictly inside the target box can become foreground.
    inside_result = top5_matcher(
        [
            {
                "boxes": torch.tensor([[[[0.0, 0.0, 4.0, 4.0]]]]).expand(1, 5, 1, 4).clone(),
                "confidences": torch.ones((1, 5, 1)),
                "classprobs": torch.full((1, 5, 1, 1), 8.0),
            }
        ],
        pack_targets([one_target]),
        anchor_points=torch.tensor([[0.0, 2.0], [2.0, 0.0], [4.0, 2.0], [2.0, 4.0], [2.0, 2.0]]),
    )
    torch.testing.assert_close(inside_result.foreground, torch.tensor([[False, False, False, False, True]]))
    assert torch.equal(inside_result.background, ~inside_result.foreground)

    # When there are fewer valid points than top-k, every valid point is selected.
    topk_result = top10_matcher(
        [two_point_preds],
        pack_targets([one_target]),
        anchor_points=two_inside_points,
    )
    torch.testing.assert_close(topk_result.foreground, torch.tensor([[True, True]]))
    torch.testing.assert_close(topk_result.assignment_weights, torch.ones((1, 2)))

    # A real (non-padded) target with no points inside it must yield only background and finite, zero weights. This
    # guards the normalization by the target's best alignment score, which is zero when the target has no matches.
    no_inside_result = top5_matcher(
        [
            {
                "boxes": torch.tensor([[[[0.0, 0.0, 4.0, 4.0]]]]).expand(1, 3, 1, 4).clone(),
                "confidences": torch.ones((1, 3, 1)),
                "classprobs": torch.full((1, 3, 1, 1), 8.0),
            }
        ],
        pack_targets([one_target]),
        anchor_points=torch.tensor([[10.0, 10.0], [12.0, 8.0], [6.0, 6.0]]),
    )
    assert not no_inside_result.foreground.any()
    assert no_inside_result.background.all()
    torch.testing.assert_close(no_inside_result.assignment_weights, torch.zeros((1, 3)))
    assert torch.isfinite(no_inside_result.assignment_weights).all()

    # Ranking should follow complete IoU (not plain IoU) when selecting candidates.
    ranking_pred_boxes = torch.tensor([[[-4.0, -4.0, 2.0, 4.0], [1.5, 1.5, 2.5, 2.5]]])
    ordinary_overlaps = box_iou(ranking_pred_boxes[0], one_target_boxes).squeeze(1)
    complete_overlaps = complete_box_iou(ranking_pred_boxes[0], one_target_boxes).squeeze(1)
    ranking_result = matcher(
        [
            {
                "boxes": ranking_pred_boxes,
                "confidences": torch.ones((1, 2, 1)),
                "classprobs": torch.full((1, 2, 1), 8.0),
            }
        ],
        pack_targets([one_target]),
        anchor_points=torch.tensor([[1.0, 2.0], [2.0, 2.0]]),
    )
    assert ordinary_overlaps[0] > ordinary_overlaps[1]
    assert complete_overlaps[1] > complete_overlaps[0]
    torch.testing.assert_close(ranking_result.foreground, torch.tensor([[False, True]]))

    # If one prediction is eligible for multiple targets, resolve it by best overlap target.
    conflicting_targets = {
        "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 6.0, 6.0]]),
        "labels": torch.tensor([0, 1], dtype=torch.int64),
    }
    conflict_result = matcher(
        [
            {
                "boxes": torch.tensor([[[[0.0, 0.0, 4.0, 4.0]]]]),
                "confidences": torch.ones((1, 1, 1)),
                "classprobs": torch.tensor([[[[8.0, 8.0]]]]),
            }
        ],
        pack_targets([conflicting_targets]),
        anchor_points=torch.tensor([[2.0, 2.0]]),
    )
    assert conflict_result.foreground.item()
    assert conflict_result.target_labels.item() == 0
    torch.testing.assert_close(conflict_result.target_boxes[0, 0], conflicting_targets["boxes"][0])


def test_tal_matching_empty_targets() -> None:
    matcher = TALMatching(topk=1, alpha=0.5, beta=6.0)

    empty_targets = {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty(0, dtype=torch.int64),
    }
    one_target = {
        "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
        "labels": torch.tensor([0], dtype=torch.int64),
    }

    # Empty targets should mark everything as background with zero assignment weight.
    empty_result = matcher(
        [
            {
                "boxes": torch.tensor([[[[0.0, 0.0, 1.0, 1.0]]]]),
                "confidences": torch.zeros((1, 1, 1)),
                "classprobs": torch.zeros((1, 1, 1, 2)),
            }
        ],
        pack_targets([empty_targets]),
        anchor_points=torch.tensor([[0.5, 0.5]]),
    )
    assert not empty_result.foreground.any()
    assert empty_result.background.all()
    assert empty_result.assignment_weights.sum() == 0

    # Batched matching should handle mixed non-empty and empty images.
    mixed_batch_preds = {
        "boxes": torch.tensor([[[[0.0, 0.0, 4.0, 4.0]], [[5.0, 5.0, 8.0, 8.0]]]]),
        "confidences": torch.ones((1, 2, 1)),
        "classprobs": torch.full((1, 2, 1, 1), 8.0),
    }
    mixed_batch_result = matcher(
        [mixed_batch_preds, mixed_batch_preds],
        pack_targets([one_target, empty_targets]),
        anchor_points=torch.tensor([[2.0, 2.0], [6.0, 6.0]]),
    )
    torch.testing.assert_close(mixed_batch_result.foreground[0], torch.tensor([True, False]))
    assert not mixed_batch_result.foreground[1].any()
    assert mixed_batch_result.background[1].all()


@pytest.mark.parametrize("input_is_normalized", [False, True])
@pytest.mark.parametrize(
    "target_labels",
    [
        torch.tensor([0, 1]),
        torch.tensor([[True, False], [False, True]]),
    ],
    ids=["integer-labels", "boolean-class-mask"],
)
def test_tal_matching_input_is_normalized(input_is_normalized: bool, target_labels: torch.Tensor) -> None:
    matcher = TALMatching(topk=1, alpha=0.5, beta=6.0)
    class_logits = torch.tensor([[[[8.0, -8.0]], [[-8.0, 8.0]]]], requires_grad=True)
    classprobs = class_logits.sigmoid() if input_is_normalized else class_logits
    preds = {
        "boxes": torch.tensor([[[[0.0, 0.0, 1.0, 1.0]], [[1.0, 0.0, 2.0, 1.0]]]]),
        "confidences": torch.zeros((1, 2, 1)),
        "classprobs": classprobs,
    }
    target_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 2.0, 1.0]])
    result = matcher(
        [preds],
        pack_targets(
            [
                {
                    "boxes": target_boxes,
                    "labels": target_labels,
                    "polygons": torch.empty((0, 8)),
                }
            ]
        ),
        input_is_normalized=input_is_normalized,
        anchor_points=torch.tensor([[0.5, 0.5], [1.5, 0.5]]),
    )

    assert torch.equal(result.foreground, torch.tensor([[True, True]]))
    assert torch.equal(result.target_boxes[0], target_boxes)
    assert torch.equal(result.target_labels[0], target_labels)
    assert result.assignment_weights.sum() == 2.0
    assert not result.assignment_weights.requires_grad
    assert result.background.shape == torch.Size([1, 2])
    assert not result.background.any()
