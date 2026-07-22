import pytest
import torch

from lightning_yolo.config import MatchingConfig
from lightning_yolo.torch_networks import create_network


@pytest.mark.parametrize(
    ("architecture", "predict_confidence", "matching_algorithm", "expected_loss_records"),
    [
        (architecture, True, None, None)
        for architecture in [
            "yolov4",
            "yolov4-tiny",
            "yolov4-p6",
            "yolov5n",
            "yolov5s",
            "yolov5m",
            "yolov5l",
            "yolov5x",
            "yolov7-w6",
            "yolov8n",
            "yolov8s",
            "yolov8m",
            "yolov8l",
            "yolov8x",
            "yolox-tiny",
            "yolox-s",
            "yolox-m",
            "yolox-l",
        ]
    ]
    + [("yolov8n", False, None, None), ("yolox-tiny", False, None, None)]
    + [("yolov8n", True, "tal", 1), ("yolov8n", True, "maxiou", 3)],
)
@pytest.mark.parametrize("in_channels", [1, 3])
def test_create_network(
    architecture: str,
    predict_confidence: bool,
    matching_algorithm: str | None,
    expected_loss_records: int | None,
    in_channels: int,
) -> None:
    num_classes = 2
    model = create_network(
        architecture=architecture,
        num_classes=num_classes,
        in_channels=in_channels,
        predict_confidence=predict_confidence,
        matching=MatchingConfig(algorithm=matching_algorithm) if matching_algorithm is not None else None,
    )
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

    targets = [{"boxes": torch.tensor([[24.0, 32.0, 88.0, 96.0]]), "labels": torch.tensor([1], dtype=torch.int64)}]
    detections, losses = model(images, targets)

    assert len(detections) > 0
    if expected_loss_records is None:
        assert len(losses) > 0
    else:
        assert len(losses) == expected_loss_records
        assert len(detections) == 3
    assert all(torch.isfinite(record.loss_sums).all() for record in losses)
    assert all(torch.isfinite(record.normalizers).all() for record in losses)

    sum(record.loss_sums.sum() for record in losses).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
