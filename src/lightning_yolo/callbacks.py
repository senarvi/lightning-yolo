import time
from typing import Any, override

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class EpochLogger(Callback):
    """Log training throughput and validation mAP at the end of each epoch.

    After each training and validation cycle, emits a single ``[epoch]`` line, showing the epoch number, elapsed
    training seconds, training throughput, and validation mAP. The elapsed time covers only the training pass, not
    validation.

    """

    def __init__(self) -> None:
        """Initialise per-epoch accumulators."""
        self._start: float | None = None
        self._train_elapsed: float | None = None
        self._measured_samples: int = 0
        self._measured_batches: int = 0

    @staticmethod
    def _now() -> float:
        """Return the current wall-clock time in seconds, after synchronising the CUDA device.

        Synchronisation ensures that all pending GPU kernels have finished before the timestamp is recorded.
        Without it, ``time.perf_counter()`` would capture the host-side submission time rather than the actual
        GPU completion time, causing the measured elapsed time to be underestimated.

        Returns:
            Wall-clock time in seconds.

        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    @override
    def on_train_epoch_start(self, trainer: Trainer, _pl_module: LightningModule) -> None:
        """Record the epoch start time and reset sample counters.

        Args:
            trainer: The Lightning trainer.
            _pl_module: The Lightning module (unused).

        """
        self._start = self._now()
        self._measured_samples = 0
        self._measured_batches = 0

    @override
    def on_train_batch_end(
        self, trainer: Trainer, _pl_module: LightningModule, _outputs: Any, batch: Any, _batch_idx: int
    ) -> None:
        """Accumulate the number of training samples seen in this epoch.

        Args:
            trainer: The Lightning trainer (unused).
            _pl_module: The Lightning module (unused).
            _outputs: Batch outputs (unused).
            batch: The current batch as ``(images, targets)``; the first element determines the batch size.
            _batch_idx: Batch index within the epoch (unused).

        """
        self._measured_samples += len(batch[0])
        self._measured_batches += 1

    @override
    def on_train_epoch_end(self, trainer: Trainer, _pl_module: LightningModule) -> None:
        """Snapshot the training-only elapsed time when the training pass finishes.

        The snapshot is taken here rather than in ``on_validation_epoch_end`` so that validation
        time is excluded from the reported training throughput.

        Args:
            trainer: The Lightning trainer (unused).
            _pl_module: The Lightning module (unused).

        """
        if self._start is not None:
            self._train_elapsed = self._now() - self._start

    @override
    def on_validation_epoch_end(self, trainer: Trainer, _pl_module: LightningModule) -> None:
        """Emit a single ``[epoch]`` log line combining throughput and validation metrics.

        Skipped silently for the sanity-check validation that runs before the first training epoch,
        because no training elapsed time has been recorded yet.

        Args:
            trainer: The Lightning trainer; provides the current epoch number and callback metrics.
            _pl_module: The Lightning module (unused).

        """
        if self._train_elapsed is None:
            return
        elapsed = self._train_elapsed
        samples_per_second = self._measured_samples / elapsed if elapsed else 0.0

        metrics = trainer.callback_metrics
        val_map = metrics.get("val/map")
        val_map_50 = metrics.get("val/map_50")

        parts = [
            f"epoch={trainer.current_epoch}",
            f"elapsed_s={elapsed:.0f}",
            f"img/s={samples_per_second:.1f}",
        ]
        if val_map is not None:
            parts.append(f"val/map={float(val_map):.6f}")
        if val_map_50 is not None:
            parts.append(f"val/map_50={float(val_map_50):.6f}")

        rank_zero_info("[epoch] " + "  ".join(parts))
