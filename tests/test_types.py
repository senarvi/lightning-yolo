import torch

from lightning_yolo.types import DetectionLossRecord


def test_detection_loss_record():
    record = DetectionLossRecord(loss_sums=torch.tensor([2.0, 4.0, 6.0]), normalizers=torch.tensor([1.0, 2.0, 3.0]))
    scaled = record.scaled(0.25)

    assert torch.allclose(scaled.loss_sums, torch.tensor([0.5, 1.0, 1.5]))
    assert torch.allclose(scaled.normalizers, torch.tensor([0.25, 0.5, 0.75]))
