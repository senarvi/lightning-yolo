import torch
import torch.nn as nn

from lightning_yolo.batching import pack_targets
from lightning_yolo.config import LossConfig, MatchingConfig
from lightning_yolo.heads import (
    DFLExpectation,
    DistributionalDistanceDetectionHead,
    PriorShapeDetectionLayer,
    create_distributional_distance_detection_head,
    create_prior_shape_detection_head,
    create_prior_shape_detection_head_with_aux,
)
from lightning_yolo.loss import YOLOLoss
from lightning_yolo.target_matching import HighestIoUMatching, SimOTAMatching


class FixedOutput(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_buffer("output").to(device=x.device, dtype=x.dtype)


def test_prior_shape_detection_layer() -> None:
    prior_shapes = [(10, 12), (20, 24)]
    layer = PriorShapeDetectionLayer(
        num_classes=2,
        prior_shapes=prior_shapes,
        matching_func=HighestIoUMatching(prior_shapes, [0, 1]),
        loss_func=YOLOLoss("ciou"),
    )
    x = torch.randn(1, 14, 2, 2)
    image_size = torch.tensor([64, 64])
    level = layer(x, image_size)
    preds = level.as_grid()

    assert level.detections.shape == (1, 8, 7)
    assert len(preds) == 1
    assert preds[0]["boxes"].shape == (2, 2, 2, 4)
    assert preds[0]["confidences"].shape == (2, 2, 2)
    assert preds[0]["classprobs"].shape == (2, 2, 2, 2)
    assert torch.isfinite(level.detections).all()


def test_create_prior_shape_detection_head() -> None:
    head = create_prior_shape_detection_head(
        [(8, 8), (16, 16)],
        [range(0, 1), range(1, 2)],
        num_classes=2,
        matching=MatchingConfig(algorithm="simota"),
    )
    assert isinstance(head.matching_func, SimOTAMatching)

    # One anchor per cell and two classes give (5 + 2) = 7 output channels per level.
    features = [torch.randn(1, 7, 4, 4, requires_grad=True), torch.randn(1, 7, 2, 2, requires_grad=True)]
    image_size = torch.tensor([32.0, 32.0])
    targets = pack_targets(
        [{"boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0]]), "labels": torch.tensor([1], dtype=torch.int64)}]
    )
    detections, losses = head(features, image_size, targets)

    # Global SimOTA assigns once across both levels, so the head returns a single loss record.
    assert len(detections) == 2
    assert len(losses) == 1
    assert torch.isfinite(losses[0].sums).all()
    sum(loss.sums.sum() for loss in losses).backward()
    assert all(feature.grad is not None and torch.isfinite(feature.grad).all() for feature in features)


def test_create_prior_shape_detection_head_with_aux() -> None:
    head = create_prior_shape_detection_head_with_aux(
        [(8, 8), (16, 16)],
        [range(0, 1), range(1, 2)],
        num_classes=2,
        matching=MatchingConfig(algorithm="simota"),
    )

    # One anchor per cell and two classes give (5 + 2) = 7 output channels per level.
    features = [torch.randn(1, 7, 4, 4, requires_grad=True), torch.randn(1, 7, 2, 2, requires_grad=True)]
    image_size = torch.tensor([32.0, 32.0])
    targets = pack_targets(
        [{"boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0]]), "labels": torch.tensor([1], dtype=torch.int64)}]
    )
    aux_features = [torch.randn(1, 7, 4, 4, requires_grad=True), torch.randn(1, 7, 2, 2, requires_grad=True)]
    detections, losses = head(features, aux_features, image_size, targets)

    # One global assignment for the lead head and one for the auxiliary head, not one per level.
    assert len(detections) == 2
    assert len(losses) == 2
    assert all(torch.isfinite(loss.sums).all() for loss in losses)
    sum(loss.sums.sum() for loss in losses).backward()
    assert all(feature.grad is not None and torch.isfinite(feature.grad).all() for feature in features)
    assert all(feature.grad is not None and torch.isfinite(feature.grad).all() for feature in aux_features)

    # Standard YOLOv8 feature map sizes with learned branches.
    head = DistributionalDistanceDetectionHead([8, 8, 8], num_classes=2, num_dfl_bins=4)

    assert len(head.box_branches) == 3
    assert len(head.class_branches) == 3
    for box_branch, class_branch in zip(head.box_branches, head.class_branches, strict=True):
        assert box_branch[-1].out_channels == 16
        assert class_branch[-1].out_channels == 2

    features = [
        torch.randn(1, 8, 80, 80, requires_grad=True),
        torch.randn(1, 8, 40, 40, requires_grad=True),
        torch.randn(1, 8, 20, 20, requires_grad=True),
    ]
    output = head.predict(features, torch.tensor([640.0, 640.0]))

    assert output.detections.shape == (1, 8400, 7)
    assert output.dfl_logits.shape == (1, 8400, 16)
    assert output.class_logits.shape == (1, 8400, 2)
    assert output.anchor_points.shape == (8400, 2)
    assert output.strides.shape == (8400, 1)
    torch.testing.assert_close(output.detections[..., 4], torch.ones_like(output.detections[..., 4]))
    torch.testing.assert_close(output.detections[..., 5:], output.class_logits.sigmoid())
    assert torch.isfinite(output.detections).all()
    (output.detections.sum() + output.dfl_logits.sum()).backward()
    assert all(feature.grad is not None for feature in features)
    assert all(torch.isfinite(feature.grad).all() for feature in features if feature.grad is not None)

    # Fixed branch outputs should be concatenated in spatial order across levels.
    flattened_head = DistributionalDistanceDetectionHead([1, 1, 1], num_classes=1, num_dfl_bins=2)
    raw_boxes = [
        torch.arange(32.0).reshape(1, 8, 2, 2),
        torch.arange(200.0, 208.0).reshape(1, 8, 1, 1),
        torch.arange(300.0, 308.0).reshape(1, 8, 1, 1),
    ]
    raw_classes = [
        torch.arange(100.0, 104.0).reshape(1, 1, 2, 2),
        torch.tensor([[[[400.0]]]]),
        torch.tensor([[[[500.0]]]]),
    ]
    flattened_head.box_branches = nn.ModuleList([FixedOutput(output) for output in raw_boxes])
    flattened_head.class_branches = nn.ModuleList([FixedOutput(output) for output in raw_classes])

    flattened_output = flattened_head.predict(
        [torch.zeros((1, 1, 2, 2)), torch.zeros((1, 1, 1, 1)), torch.zeros((1, 1, 1, 1))], torch.tensor([16.0, 16.0])
    )

    expected_boxes = torch.cat([level.permute(0, 2, 3, 1).reshape(1, -1, 8) for level in raw_boxes], dim=1)
    expected_classes = torch.cat([level.permute(0, 2, 3, 1).reshape(1, -1, 1) for level in raw_classes], dim=1)
    torch.testing.assert_close(flattened_output.dfl_logits, expected_boxes)
    torch.testing.assert_close(flattened_output.class_logits, expected_classes)


def test_distributional_distance_detection_head_initialize_output_biases() -> None:
    head = DistributionalDistanceDetectionHead([16, 32, 64], num_classes=5, num_dfl_bins=4)

    head.initialize_output_biases([8.0, 16.0, 32.0])

    for stride, box_branch, class_branch in zip([8.0, 16.0, 32.0], head.box_branches, head.class_branches, strict=True):
        torch.testing.assert_close(box_branch[-1].bias, torch.ones_like(box_branch[-1].bias))
        class_bias_value = torch.log(torch.tensor(5 / 5 / (640 / stride) ** 2)).item()
        expected_class_bias = torch.full_like(class_branch[-1].bias, class_bias_value)
        torch.testing.assert_close(class_branch[-1].bias, expected_class_bias)


def test_create_distributional_distance_detection_head() -> None:
    loss = LossConfig(overlap_multiplier=2.0, class_multiplier=3.0, dfl_multiplier=4.0, num_dfl_bins=8)

    head = create_distributional_distance_detection_head([16, 32, 64], num_classes=5, loss=loss)

    assert isinstance(head, DistributionalDistanceDetectionHead)
    assert head.num_dfl_bins == 8
    assert head.loss_func.overlap_multiplier == 2.0
    assert head.loss_func.class_multiplier == 3.0
    assert head.loss_func.dfl_multiplier == 4.0
    assert head.loss_func.num_dfl_bins == 8


def test_distributional_distance_detection_head_geometry_cache() -> None:
    head = DistributionalDistanceDetectionHead([8, 8, 8], num_classes=2, num_dfl_bins=4)
    head.eval()
    image_size = torch.tensor([16.0, 16.0])
    features = [torch.randn(1, 8, 2, 2), torch.randn(1, 8, 1, 1), torch.randn(1, 8, 1, 1)]
    changed_features = [torch.randn(1, 8, 4, 4), torch.randn(1, 8, 2, 2), torch.randn(1, 8, 1, 1)]

    first = head.predict(features, image_size)
    second = head.predict(features, image_size)
    third = head.predict(changed_features, torch.tensor([32.0, 32.0]))

    assert first.anchor_points is second.anchor_points
    assert third.anchor_points.shape[0] == 21


def test_dfl_expectation() -> None:
    decoder = DFLExpectation(num_dfl_bins=4)
    logits = torch.full((1, 2, 16), -20.0, requires_grad=True)
    logits.data[..., 1] = 20.0
    logits.data[..., 6] = 20.0
    logits.data[..., 11] = 20.0
    logits.data[..., 12] = 20.0

    distances = decoder(logits)
    uniform_distances = decoder(torch.zeros((2, 16)))

    torch.testing.assert_close(distances, torch.tensor([[[1.0, 2.0, 3.0, 0.0], [1.0, 2.0, 3.0, 0.0]]]))
    torch.testing.assert_close(uniform_distances, torch.full((2, 4), 1.5))
    assert distances.device == logits.device
    distances.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
