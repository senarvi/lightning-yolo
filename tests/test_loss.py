import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torchvision.ops import box_iou

from lightning_yolo.loss import (
    YOLOLoss,
    _background_confidence_loss,
    _foreground_confidence_loss,
    _pairwise_confidence_loss,
    _size_compensation,
    _target_labels_to_probs,
    box_iou_loss,
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


def test_yolo_loss_pairwise_shapes_and_overlap_values():
    loss = YOLOLoss("iou", overlap_multiplier=2.0, confidence_multiplier=3.0, class_multiplier=4.0)
    preds = {
        "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]),
        "confidences": torch.tensor([0.1, -0.2]),
        "classprobs": torch.tensor([[0.3, -0.7], [0.6, -0.4]]),
    }
    targets = {
        "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 4.0, 4.0]]),
        "labels": torch.tensor([0, 1]),
    }

    losses, overlap = loss.pairwise(preds, targets, input_is_normalized=False)

    assert overlap.shape == (2, 2)
    torch.testing.assert_close(overlap, box_iou(preds["boxes"], targets["boxes"]))
    assert losses.overlap.shape == (2, 2)
    assert losses.confidence.shape == (2, 2)
    assert losses.classification.shape == (2, 2)
    assert torch.isfinite(losses.overlap).all()
    assert torch.isfinite(losses.confidence).all()
    assert torch.isfinite(losses.classification).all()


def test_yolo_loss_pairwise():
    loss = YOLOLoss("iou")
    preds = {
        "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
        "confidences": torch.tensor([0.8]),
        "classprobs": torch.tensor([[0.7, 0.2]]),
    }
    targets = {
        "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
        "labels": torch.tensor([1]),
    }
    losses, overlap = loss.pairwise(preds, targets, input_is_normalized=True)

    assert overlap.shape == (1, 1)
    assert losses.overlap.shape == (1, 1)
    assert losses.confidence.shape == (1, 1)
    assert losses.classification.shape == (1, 1)


def test_yolo_loss_elementwise_sums():
    loss = YOLOLoss("iou")
    preds = {
        "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
        "confidences": torch.tensor([0.7]),
        "bg_confidences": torch.tensor([0.1, 0.2]),
        "classprobs": torch.tensor([[0.8, 0.1]]),
    }
    targets = {
        "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
        "labels": torch.tensor([1]),
    }
    result = loss.elementwise_sums(
        preds,
        targets,
        input_is_normalized=True,
        image_size=torch.tensor([64.0, 64.0]),
    )

    assert torch.isfinite(result.overlap)
    assert torch.isfinite(result.confidence)
    assert torch.isfinite(result.classification)
