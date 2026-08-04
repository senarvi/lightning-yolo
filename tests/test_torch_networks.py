import pytest
import torch
from torch.optim import SGD

from lightning_yolo.batching import pack_targets
from lightning_yolo.config import LossConfig
from lightning_yolo.torch_networks import create_network
from lightning_yolo.types import PackedTargetDict


def _training_fixture() -> tuple[torch.Tensor, PackedTargetDict]:
    images = torch.linspace(0.0, 1.0, steps=3 * 64 * 64).reshape(1, 3, 64, 64)
    targets = pack_targets([
        {"boxes": torch.tensor([[16.0, 16.0, 48.0, 48.0]]), "labels": torch.tensor([1], dtype=torch.int64)}
    ])
    return images, targets


def _assert_finite_nonzero_grad(model: torch.nn.Module, prefix: str) -> None:
    grads = [parameter.grad for name, parameter in model.named_parameters() if name.startswith(prefix)]
    assert grads
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
    assert any(grad is not None and grad.abs().sum() > 0 for grad in grads)


@pytest.mark.parametrize(
    ("architecture", "expected_loss_records"),
    [
        (architecture, None)
        for architecture in [
            "yolov4",
            "yolov4-tiny",
            "yolov4-p6",
            "yolov5n",
            "yolov5s",
            "yolov5m",
            "yolov5l",
            "yolov5x",
            "yolox-tiny",
            "yolox-s",
            "yolox-m",
            "yolox-l",
        ]
    ]
    + [("yolov7-w6", 8)]
    + [(architecture, 1) for architecture in ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]],
)
@pytest.mark.parametrize("in_channels", [1, 3])
def test_create_network(architecture: str, expected_loss_records: int | None, in_channels: int) -> None:
    num_classes = 2
    model = create_network(architecture=architecture, num_classes=num_classes, in_channels=in_channels)
    model.eval()

    images = torch.rand(1, in_channels, 128, 128)
    with torch.no_grad():
        detections, losses = model(images, targets=None)

    assert len(detections) > 0
    assert losses == []

    for output in detections:
        assert output.shape[0] == 1
        assert output.shape[2] == (5 + num_classes)
        assert torch.isfinite(output).all()

    targets = pack_targets([
        {"boxes": torch.tensor([[24.0, 32.0, 88.0, 96.0]]), "labels": torch.tensor([1], dtype=torch.int64)}
    ])
    detections, losses = model(images, targets)

    assert len(detections) > 0
    if expected_loss_records is None:
        assert len(losses) > 0
    else:
        assert len(losses) == expected_loss_records
    assert all(torch.isfinite(loss.sums).all() for loss in losses)
    assert all(torch.isfinite(record.normalizers).all() for record in losses)

    sum(loss.sums.sum() for loss in losses).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_create_network_yolov8n() -> None:
    model = create_network("yolov8n", num_classes=80)
    model.eval()

    images = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        detections, losses = model(images, targets=None)

    assert losses == []
    assert len(detections) == 1
    assert detections[0].shape == (1, 8400, 85)
    assert detections[0][..., 4].eq(1).all()
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_157_184


@pytest.mark.parametrize(
    ("loss", "expected_loss_settings"),
    [
        (None, (7.5, 0.5, 1.5, 16)),
        (
            LossConfig(overlap_multiplier=2.0, class_multiplier=3.0, dfl_multiplier=4.0, num_dfl_bins=8),
            (2.0, 3.0, 4.0, 8),
        ),
        (LossConfig(overlap_func="iou"), (7.5, 0.5, 1.5, 16)),
        (LossConfig(label_smoothing=0.1), (7.5, 0.5, 1.5, 16)),
    ],
)
def test_yolo_forward(loss: LossConfig | None, expected_loss_settings: tuple[float, float, float, int]) -> None:
    model = create_network("yolov8n", num_classes=2, loss=loss)
    images, targets = _training_fixture()

    assert (
        model.detect.loss_func.overlap_multiplier,
        model.detect.loss_func.class_multiplier,
        model.detect.loss_func.dfl_multiplier,
        model.detect.loss_func.num_dfl_bins,
    ) == expected_loss_settings
    assert model.detect.num_dfl_bins == expected_loss_settings[-1]

    _, loss_records = model(images, targets)

    assert len(loss_records) == 1
    assert torch.isfinite(loss_records[0].sums).all()


@pytest.mark.parametrize(
    ("loss", "expected_nonzero_grad_prefixes", "expected_positive_loss_index"),
    [
        (None, ("detect.box_branches", "detect.class_branches", "backbone", "pan3"), None),
        (LossConfig(overlap_multiplier=1.0, class_multiplier=0.0, dfl_multiplier=0.0), ("detect.box_branches",), 0),
        (LossConfig(overlap_multiplier=0.0, class_multiplier=0.0, dfl_multiplier=1.0), ("detect.box_branches",), 2),
        (LossConfig(overlap_multiplier=0.0, class_multiplier=1.0, dfl_multiplier=0.0), ("detect.class_branches",), 1),
    ],
)
def test_yolo_forward_backward(
    loss: LossConfig | None, expected_nonzero_grad_prefixes: tuple[str, ...], expected_positive_loss_index: int | None
) -> None:
    torch.manual_seed(0)
    model = create_network("yolov8n", num_classes=2, loss=loss)
    model.train()
    images, targets = _training_fixture()
    optimizer = SGD(model.parameters(), lr=1e-3)

    _, loss_records = model(images, targets)
    assert len(loss_records) == 1
    loss_record = loss_records[0]
    losses = loss_record.sums / loss_record.normalizers
    total_loss = losses.sum()

    assert loss_record.names == ("overlap", "classification", "dfl")
    assert torch.isfinite(losses).all()
    assert torch.isfinite(total_loss)
    if expected_positive_loss_index is not None:
        assert losses[expected_positive_loss_index] > 0

    optimizer.zero_grad()
    total_loss.backward()
    for prefix in expected_nonzero_grad_prefixes:
        _assert_finite_nonzero_grad(model, prefix)
    optimizer.step()
