import torch
from lightning.pytorch.callbacks import WeightAveraging
from torch import Tensor


class EMAWeightAveraging(WeightAveraging):
    """Average model weights with the ramped EMA schedule used by Ultralytics.

    Args:
        decay: Maximum EMA decay.
        warmup_updates: Number of updates over which EMA decay ramps toward its maximum.

    """

    def __init__(self, decay: float = 0.9999, warmup_updates: float = 2000.0) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in the interval (0, 1).")
        if warmup_updates <= 0.0:
            raise ValueError("warmup_updates must be > 0.")
        self.decay = decay
        self.warmup_updates = warmup_updates
        super().__init__(use_buffers=True, multi_avg_fn=self._multi_avg_fn)

    @torch.no_grad()
    def _multi_avg_fn(self, ema_tensors: list[Tensor], current_tensors: list[Tensor], num_averaged: Tensor) -> None:
        """Update one device and dtype group using the ramped EMA decay.

        Args:
            ema_tensors: Current averaged tensors.
            current_tensors: Current model tensors.
            num_averaged: Number of completed averaging updates.

        """
        updates = num_averaged + 1
        decay = self.decay * (1.0 - torch.exp(-updates / self.warmup_updates))
        if torch.is_floating_point(ema_tensors[0]) or torch.is_complex(ema_tensors[0]):
            torch._foreach_lerp_(ema_tensors, current_tensors, 1.0 - decay)  # type: ignore[call-overload]
        else:
            for ema_tensor, current_tensor in zip(ema_tensors, current_tensors, strict=True):
                ema_tensor.copy_(ema_tensor * decay + current_tensor * (1.0 - decay))
