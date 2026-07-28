import pytest
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torchvision.ops import box_iou

from lightning_yolo.loss import (
    DistributionalDistanceLoss,
    YOLOLoss,
    _background_class_loss,
    _background_confidence_loss,
    _foreground_confidence_loss,
    _pairwise_confidence_loss,
    _size_compensation,
    _target_labels_to_probs,
    box_iou_loss,
    dfl_loss,
)
from lightning_yolo.matching_result import DenseMatchingResult, ImageMatch, SparseMatchingResult
from lightning_yolo.types import DistributionalDistancePredictions


class FixedMatcher:
    def __call__(
        self,
        predictions: object,
        targets: object,
        *,
        anchor_points: torch.Tensor,
        input_is_normalized: bool,
    ) -> DenseMatchingResult:
        del predictions, targets, anchor_points, input_is_normalized
        return DenseMatchingResult(
            foreground=torch.tensor([[True, True]]),
            background=torch.tensor([[False, False]]),
            target_boxes=torch.tensor([[[0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 5.0, 5.0]]]),
            target_labels=torch.tensor([[0, 1]], dtype=torch.int64),
            assignment_weights=torch.tensor([[0.5, 1.0]]),
        )


def _distributional_predictions(
    class_logits: torch.Tensor,
    dfl_logits: torch.Tensor | None = None,
    pixel_boxes: torch.Tensor | None = None,
) -> DistributionalDistancePredictions:
    if dfl_logits is None:
        dfl_logits = torch.zeros((*class_logits.shape[:2], 16), dtype=class_logits.dtype, device=class_logits.device)
    if pixel_boxes is None:
        pixel_boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 5.0, 5.0]]], dtype=class_logits.dtype)
    return DistributionalDistancePredictions(
        detections=torch.empty((*class_logits.shape[:2], class_logits.shape[2] + 5), dtype=class_logits.dtype),
        pixel_boxes=pixel_boxes,
        grid_boxes=pixel_boxes,
        dfl_logits=dfl_logits,
        class_logits=class_logits,
        anchor_points=torch.tensor([[1.0, 1.0], [4.0, 4.0]], dtype=class_logits.dtype),
        strides=torch.ones((2, 1), dtype=class_logits.dtype),
    )


def _distributional_matching(
    foreground: torch.Tensor,
    target_labels: torch.Tensor,
    assignment_weights: torch.Tensor,
) -> DenseMatchingResult:
    return DenseMatchingResult(
        foreground=foreground,
        background=~foreground,
        target_boxes=torch.tensor([[[0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 5.0, 5.0]]]),
        target_labels=target_labels,
        assignment_weights=assignment_weights,
    )


def test_box_iou_loss():
    boxes1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]])
    boxes2 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
    result = box_iou_loss(boxes1, boxes2)
    expected = torch.tensor([0.0, 6.0 / 7.0])

    torch.testing.assert_close(result, expected)


def test_size_compensation():
    targets = torch.tensor([[0.0, 0.0, 10.0, 20.0], [0.0, 0.0, 40.0, 20.0]])
    image_size = torch.tensor([100.0, 100.0])
    result = _size_compensation(targets, image_size)
    expected = torch.tensor([1.98, 1.92])

    torch.testing.assert_close(result, expected)


def test_pairwise_confidence_loss():
    preds = torch.tensor([0.1, -0.4])
    overlap = torch.tensor([[0.5, -0.2], [1.0, 0.0]])
    result = _pairwise_confidence_loss(preds, overlap, binary_cross_entropy_with_logits, predict_overlap=0.5)
    preds = preds.unsqueeze(1).expand(overlap.shape)
    targets = 0.5 + 0.5 * overlap.clamp(min=0)
    expected = binary_cross_entropy_with_logits(preds, targets, reduction="none")

    torch.testing.assert_close(result, expected)


def test_foreground_confidence_loss():
    preds = torch.tensor([0.2, -0.3, 1.0])
    overlap = torch.tensor([0.7, -0.5, 0.2])
    result = _foreground_confidence_loss(preds, overlap, binary_cross_entropy_with_logits, predict_overlap=0.25)
    targets = 0.75 + 0.25 * overlap.clamp(min=0)
    expected = binary_cross_entropy_with_logits(preds, targets, reduction="sum")

    torch.testing.assert_close(result, expected)


def test_background_confidence_loss():
    preds = torch.tensor([0.2, -0.3, 1.0])
    result = _background_confidence_loss(preds, binary_cross_entropy_with_logits)
    expected = binary_cross_entropy_with_logits(preds, torch.zeros_like(preds), reduction="sum")

    torch.testing.assert_close(result, expected)


def test_background_class_loss():
    preds = torch.tensor([[0.2, -0.3], [1.0, 0.5]])
    result = _background_class_loss(preds, binary_cross_entropy_with_logits)
    expected = binary_cross_entropy_with_logits(preds, torch.zeros_like(preds), reduction="sum")

    torch.testing.assert_close(result, expected)


def test_target_labels_to_probs():
    # In case a label is greater than the number of predicted classes, it will be mapped to the last class.
    labels = torch.tensor([0, 2, 3])
    result = _target_labels_to_probs(labels, num_classes=3, dtype=torch.float32, label_smoothing=0.2)
    base = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    expected = 0.1 + 0.8 * base

    torch.testing.assert_close(result, expected)


def test_target_labels_to_probs_class_probabilities():
    probs = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
    result = _target_labels_to_probs(probs, num_classes=2, dtype=torch.float32)

    assert torch.equal(result, probs)


def test_dfl_loss() -> None:
    # Fractional targets should interpolate between adjacent distance bins.
    logits = torch.tensor([[[0.5, 2.0, -1.0], [1.0, -0.5, 0.25], [-0.2, 0.7, 1.5], [1.2, -0.4, 0.3]]])
    targets = torch.tensor([[1.25, 0.5, 1.75, 0.0]])

    result = dfl_loss(logits, targets)

    flat_logits = logits.reshape(-1, 3)
    left_bins = torch.tensor([1, 0, 1, 0])
    right_bins = torch.tensor([2, 1, 2, 1])
    right_weights = torch.tensor([0.25, 0.5, 0.75, 0.0])
    expected = (
        (
            torch.nn.functional.cross_entropy(flat_logits, left_bins, reduction="none") * (1 - right_weights)
            + torch.nn.functional.cross_entropy(flat_logits, right_bins, reduction="none") * right_weights
        )
        .mean()
        .unsqueeze(0)
    )

    torch.testing.assert_close(result, expected)

    # Uniform logits should produce the uniform-loss value, clip targets in the computation, and backpropagate.
    uniform_logits = torch.zeros((2, 4, 5), requires_grad=True)
    clipped_targets = torch.tensor([[-2.0, 1.0, 2.0, 12.0], [1.5, 2.5, 3.0, 0.5]])
    original_targets = clipped_targets.clone()
    uniform_result = dfl_loss(uniform_logits, clipped_targets)

    assert uniform_result.shape == (2,)
    torch.testing.assert_close(clipped_targets, original_targets)
    torch.testing.assert_close(uniform_result, torch.full((2,), torch.log(torch.tensor(5.0))))
    uniform_result.sum().backward()

    assert uniform_logits.grad is not None
    assert torch.isfinite(uniform_logits.grad).all()


def test_yolo_loss() -> None:
    loss_func = YOLOLoss("ciou")
    pred_boxes = torch.tensor(
        [
            [[0.0, 0.0, 2.0, 2.0], [2.0, 0.0, 4.0, 2.0], [4.0, 0.0, 6.0, 2.0]],
            [[0.0, 0.0, 2.0, 2.0], [2.0, 0.0, 4.0, 2.0], [4.0, 0.0, 6.0, 2.0]],
        ],
        requires_grad=True,
    )
    pred_classprobs = torch.tensor(
        [
            [[2.0, -1.0], [-1.0, 2.0], [0.5, -0.5]],
            [[0.2, -0.3], [-0.4, 0.6], [1.0, -1.5]],
        ],
        requires_grad=True,
    )
    image_size = torch.tensor([6.0, 2.0])

    # The first image has two foreground anchors; the second image has no targets, so dense TAL padding leaves
    # zero-area target boxes behind the background anchors.
    target_boxes = torch.tensor(
        [
            [[0.0, 0.0, 2.0, 2.0], [2.0, 0.0, 4.0, 2.0], [0.0, 0.0, 2.0, 2.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    target_labels = torch.tensor([[0, 1, 0], [0, 0, 0]])
    foreground = torch.tensor([[True, True, False], [False, False, False]])
    dense = DenseMatchingResult(
        foreground=foreground,
        background=torch.tensor([[False, False, True], [True, True, True]]),
        target_boxes=target_boxes,
        target_labels=target_labels,
        assignment_weights=foreground.float(),
    )
    sparse = SparseMatchingResult(
        [
            ImageMatch(
                foreground=torch.tensor([0, 1]),
                background=torch.tensor([False, False, True]),
                target_boxes=target_boxes[0, :2],
                target_labels=target_labels[0, :2],
            ),
            ImageMatch(
                foreground=torch.empty(0, dtype=torch.int64),
                background=torch.tensor([True, True, True]),
                target_boxes=torch.empty((0, 4)),
                target_labels=torch.empty(0, dtype=torch.int64),
            ),
        ]
    )
    preds = [
        {"boxes": boxes, "confidences": torch.ones(3), "classprobs": classprobs}
        for boxes, classprobs in zip(pred_boxes, pred_classprobs, strict=True)
    ]

    dense_sums = loss_func(dense, preds, False, image_size).sums
    sparse_sums = loss_func(sparse, preds, False, image_size).sums

    # Overlap loss covers only the foreground anchors.
    expected_overlap = loss_func._elementwise_overlap_loss(target_boxes[0, :2], pred_boxes[0, :2])
    expected_overlap = (
        expected_overlap * (2 - target_boxes[0, :2, 2] / image_size[0] * target_boxes[0, :2, 3] / image_size[1])
    ).sum()
    torch.testing.assert_close(dense_sums[0], expected_overlap * loss_func.overlap_multiplier)

    # Confidence loss is non-zero because the loss function always predicts confidence.
    assert float(dense_sums[1]) > 0.0
    expected_foreground_class = binary_cross_entropy_with_logits(
        pred_classprobs[0:1, :2], torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), reduction="sum"
    )
    torch.testing.assert_close(dense_sums[2], expected_foreground_class)

    # Background anchors do not contribute to the overlap gradient; all classification gradients are finite.
    grads = torch.autograd.grad(dense_sums[0] + dense_sums[2], (pred_boxes, pred_classprobs))
    assert torch.equal(grads[0][0, 2], torch.zeros_like(grads[0][0, 2]))
    assert torch.equal(grads[0][1], torch.zeros_like(grads[0][1]))
    assert torch.isfinite(grads[0]).all()
    assert torch.isfinite(grads[1]).all()

    # Dense and sparse matching produce identical loss sums.
    torch.testing.assert_close(dense_sums, sparse_sums)

    weighted_dense = DenseMatchingResult(
        foreground=foreground,
        background=torch.tensor([[False, False, True], [True, True, True]]),
        target_boxes=target_boxes,
        target_labels=target_labels,
        assignment_weights=torch.tensor([[0.25, 0.75, 0.0], [0.0, 0.0, 0.0]]),
    )
    weighted_sums = loss_func(weighted_dense, preds, False, image_size).sums
    expected_weighted_overlap = loss_func._elementwise_overlap_loss(target_boxes[0, :2], pred_boxes[0, :2])
    expected_weighted_overlap = (
        expected_weighted_overlap
        * (2 - target_boxes[0, :2, 2] / image_size[0] * target_boxes[0, :2, 3] / image_size[1])
        * torch.tensor([0.25, 0.75])
    ).sum()
    expected_weighted_foreground_class = binary_cross_entropy_with_logits(
        pred_classprobs[0:1, :2], torch.tensor([[[0.25, 0.0], [0.0, 0.75]]]), reduction="sum"
    )
    torch.testing.assert_close(weighted_sums[0], expected_weighted_overlap * loss_func.overlap_multiplier)
    torch.testing.assert_close(weighted_sums[2], expected_weighted_foreground_class)

    all_empty_dense = DenseMatchingResult(
        foreground=torch.tensor([[False, False, False]]),
        background=torch.tensor([[True, True, True]]),
        target_boxes=torch.zeros((1, 3, 4)),
        target_labels=torch.zeros((1, 3), dtype=torch.int64),
        assignment_weights=torch.zeros((1, 3)),
    )
    all_empty_sums = loss_func(all_empty_dense, preds[1:], False, image_size).sums
    all_empty_grads = torch.autograd.grad(
        all_empty_sums[0] + all_empty_sums[2],
        (pred_boxes, pred_classprobs),
        allow_unused=True,
    )
    assert all_empty_grads[0] is not None
    assert torch.equal(all_empty_grads[0], torch.zeros_like(pred_boxes))
    assert all_empty_grads[1] is not None
    assert torch.isfinite(all_empty_grads[1]).all()


def test_yolo_loss_multilabel() -> None:
    # A boolean class mask with one active class per box must match the equivalent integer class indices.
    loss_func = YOLOLoss("ciou")
    image_size = torch.tensor([6.0, 2.0])
    pred_boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0], [2.0, 0.0, 4.0, 2.0]]])
    pred_classprobs = torch.tensor([[[2.0, -1.0], [-1.0, 2.0]]])
    preds = [{"boxes": pred_boxes[0], "confidences": torch.ones(2), "classprobs": pred_classprobs[0]}]
    target_boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0], [2.0, 0.0, 4.0, 2.0]]])
    foreground = torch.tensor([[True, True]])
    background = torch.tensor([[False, False]])
    assignment_weights = foreground.float()

    index_matching = DenseMatchingResult(
        foreground=foreground,
        background=background,
        target_boxes=target_boxes,
        target_labels=torch.tensor([[0, 1]]),
        assignment_weights=assignment_weights,
    )
    mask_matching = DenseMatchingResult(
        foreground=foreground,
        background=background,
        target_boxes=target_boxes,
        target_labels=torch.tensor([[[True, False], [False, True]]]),
        assignment_weights=assignment_weights,
    )

    index_sums = loss_func(index_matching, preds, False, image_size).sums
    mask_sums = loss_func(mask_matching, preds, False, image_size).sums

    torch.testing.assert_close(index_sums, mask_sums)


@pytest.mark.parametrize(
    ("loss", "preds", "targets", "input_is_normalized", "expected_shape"),
    [
        pytest.param(
            YOLOLoss("iou", overlap_multiplier=2.0, confidence_multiplier=3.0, class_multiplier=4.0),
            {
                "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]),
                "confidences": torch.tensor([0.1, -0.2]),
                "classprobs": torch.tensor([[0.3, -0.7], [0.6, -0.4]]),
            },
            {
                "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 4.0, 4.0]]),
                "labels": torch.tensor([0, 1]),
            },
            False,
            (2, 2),
            id="logits",
        ),
        pytest.param(
            YOLOLoss("iou"),
            {
                "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
                "confidences": torch.tensor([0.8]),
                "classprobs": torch.tensor([[0.7, 0.2]]),
            },
            {
                "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
                "labels": torch.tensor([1]),
            },
            True,
            (1, 1),
            id="normalized",
        ),
    ],
)
def test_yolo_loss_pairwise_costs(loss, preds, targets, input_is_normalized, expected_shape):
    losses, overlap = loss.pairwise_costs(preds, targets, input_is_normalized=input_is_normalized)

    assert overlap.shape == expected_shape
    torch.testing.assert_close(overlap, box_iou(preds["boxes"], targets["boxes"]))
    assert losses.overlap.shape == expected_shape
    assert losses.confidence.shape == expected_shape
    assert losses.classification.shape == expected_shape
    assert torch.isfinite(losses.overlap).all()
    assert torch.isfinite(losses.confidence).all()
    assert torch.isfinite(losses.classification).all()


def test_distributional_distance_loss() -> None:
    loss_func = DistributionalDistanceLoss(num_dfl_bins=4)

    # A matched foreground anchor should produce classification and DFL loss with finite gradients.
    class_logits = torch.zeros((1, 2, 2), requires_grad=True)
    dfl_logits = torch.zeros((1, 2, 16), requires_grad=True)
    pixel_boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 5.0, 5.0]]], requires_grad=True)
    preds = _distributional_predictions(class_logits, dfl_logits, pixel_boxes)
    matching = _distributional_matching(
        foreground=torch.tensor([[True, False]]),
        target_labels=torch.tensor([[0, 0]], dtype=torch.int64),
        assignment_weights=torch.tensor([[1.0, 0.0]]),
    )

    record = loss_func(preds, matching)
    losses = record.sums / record.normalizers

    expected_class = binary_cross_entropy_with_logits(
        class_logits.detach(), torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]), reduction="sum"
    )
    torch.testing.assert_close(losses[0], torch.tensor(0.0))
    torch.testing.assert_close(losses[1], expected_class * 0.5)
    torch.testing.assert_close(losses[2], torch.log(torch.tensor(4.0)) * 1.5)

    losses.sum().backward()
    assert class_logits.grad is not None
    assert dfl_logits.grad is not None
    assert pixel_boxes.grad is not None
    assert torch.isfinite(class_logits.grad).all()
    assert torch.isfinite(dfl_logits.grad).all()
    assert torch.isfinite(pixel_boxes.grad).all()

    # Fractional assignment weights should scale the class targets and the shared denominator.
    class_logits = torch.tensor([[[1.0, -1.0], [0.5, -0.5]]])
    preds = _distributional_predictions(class_logits)
    matching = FixedMatcher()(preds, object(), anchor_points=preds.anchor_points, input_is_normalized=False)

    record = loss_func(preds, matching)
    losses = record.sums / record.normalizers

    expected_denominator = torch.tensor(1.5)
    expected_class_sum = binary_cross_entropy_with_logits(
        class_logits,
        torch.tensor([[[0.5, 0.0], [0.0, 1.0]]]),
        reduction="sum",
    )

    torch.testing.assert_close(record.normalizers, expected_denominator.expand_as(record.normalizers))
    torch.testing.assert_close(losses[0], torch.tensor(0.0))
    torch.testing.assert_close(losses[1], expected_class_sum * 0.5 / expected_denominator)
    torch.testing.assert_close(losses[2], torch.log(torch.tensor(4.0)) * 1.5)

    # With no foreground matches, only the classification branch should receive gradients.
    class_logits = torch.zeros((1, 2, 2), requires_grad=True)
    dfl_logits = torch.zeros((1, 2, 16), requires_grad=True)
    pixel_boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 5.0, 5.0]]], requires_grad=True)
    preds = _distributional_predictions(class_logits, dfl_logits, pixel_boxes)
    matching = _distributional_matching(
        foreground=torch.tensor([[False, False]]),
        target_labels=torch.tensor([[0, 0]], dtype=torch.int64),
        assignment_weights=torch.tensor([[0.0, 0.0]]),
    )

    record = loss_func(preds, matching)
    losses = record.sums / record.normalizers

    torch.testing.assert_close(losses[0], torch.tensor(0.0))
    torch.testing.assert_close(losses[2], torch.tensor(0.0))
    losses.sum().backward()
    assert class_logits.grad is not None
    assert dfl_logits.grad is not None
    assert pixel_boxes.grad is not None
    assert torch.isfinite(class_logits.grad).all()
    torch.testing.assert_close(dfl_logits.grad, torch.zeros_like(dfl_logits))
    torch.testing.assert_close(pixel_boxes.grad, torch.zeros_like(pixel_boxes))


def test_distributional_distance_loss_label_smoothing() -> None:
    class_logits = torch.zeros((1, 2, 2))
    preds = _distributional_predictions(class_logits)
    loss_func = DistributionalDistanceLoss(label_smoothing=0.2, num_dfl_bins=4)
    matching = FixedMatcher()(preds, object(), anchor_points=preds.anchor_points, input_is_normalized=False)

    record = loss_func(preds, matching)
    losses = record.sums / record.normalizers

    expected_targets = torch.tensor([[[0.45, 0.05], [0.05, 0.9]]])
    expected_class_sum = binary_cross_entropy_with_logits(class_logits, expected_targets, reduction="sum")
    torch.testing.assert_close(record.normalizers, torch.tensor([1.5, 1.5, 1.5]))
    torch.testing.assert_close(losses[1], expected_class_sum * 0.5 / 1.5)
