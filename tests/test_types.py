import torch

from lightning_yolo.types import DetectionLoss, DetectionLossContribution


def test_detection_loss_contribution_scaled() -> None:
    contribution = DetectionLossContribution(
        sums=torch.tensor([2.0, 4.0, 6.0]), normalizers=torch.tensor([1.0, 2.0, 3.0])
    )
    scaled = contribution.scaled(0.25)

    assert torch.allclose(scaled.sums, torch.tensor([0.5, 1.0, 1.5]))
    assert torch.allclose(scaled.normalizers, torch.tensor([0.25, 0.5, 0.75]))
    assert scaled.names == ("overlap", "confidence", "class")


def test_detection_loss_from_contributions() -> None:
    contributions = [
        DetectionLossContribution(
            sums=torch.tensor([2.0, 6.0]), normalizers=torch.tensor([1.0, 2.0]), names=("box", "class")
        ),
        DetectionLossContribution(
            sums=torch.tensor([4.0, 10.0]), normalizers=torch.tensor([2.0, 5.0]), names=("box", "class")
        ),
    ]

    combined = DetectionLoss.from_contributions(contributions)

    torch.testing.assert_close(combined.values, torch.tensor([6.0 / 3.0, 16.0 / 7.0]))
    torch.testing.assert_close(combined.total, combined.values.sum())
