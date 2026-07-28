import torch

from lightning_yolo.callbacks import EMAWeightAveraging


def test_ema_weight_averaging() -> None:
    callback = EMAWeightAveraging(decay=0.99, warmup_updates=100.0)

    ema_tensors = [torch.tensor([0.0])]
    current_tensors = [torch.tensor([1.0])]
    callback._multi_avg_fn(ema_tensors, current_tensors, torch.tensor(0))
    expected_decay = 0.99 * (1.0 - torch.exp(torch.tensor(-1.0 / 100.0)))
    torch.testing.assert_close(ema_tensors[0], (1.0 - expected_decay).unsqueeze(0))
