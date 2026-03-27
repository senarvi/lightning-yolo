import pytest
import torch

from lightning_yolo.torch_networks import create_network


@pytest.mark.parametrize(
    "architecture",
    [
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
    ],
)
def test_create_network(architecture):
    num_classes = 2
    model = create_network(architecture=architecture, num_classes=num_classes)
    model.eval()

    images = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        detections, losses, hits = model(images, targets=None)

    assert len(detections) > 0
    assert losses == []
    assert hits == []

    for output in detections:
        assert output.shape[0] == 1
        assert output.shape[2] == (5 + num_classes)
        assert torch.isfinite(output).all()
