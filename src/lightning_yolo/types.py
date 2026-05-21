from dataclasses import dataclass
from typing import TypedDict

from torch import Tensor


@dataclass(frozen=True)
class DetectionLossRecord:
    """Loss sums and corresponding normalizers for one detection head."""

    loss_sums: Tensor
    normalizers: Tensor

    def scaled(self, scale: float) -> DetectionLossRecord:
        """Scales the loss sums and normalizers by the same factor.

        Args:
            scale: Multiplier applied to this record's contribution to the global loss average.

        """
        return DetectionLossRecord(loss_sums=self.loss_sums * scale, normalizers=self.normalizers * scale)


class PredictionDict(TypedDict):
    boxes: Tensor
    confidences: Tensor
    classprobs: Tensor


class TargetDict(TypedDict):
    boxes: Tensor
    labels: Tensor


class MatchedPredictionDict(TypedDict):
    boxes: Tensor
    confidences: Tensor
    bg_confidences: Tensor
    classprobs: Tensor


class MatchedTargetDict(TypedDict):
    boxes: Tensor
    labels: Tensor


IMAGES = tuple[Tensor, ...] | list[Tensor]
PREDICTIONS = tuple[PredictionDict, ...] | list[PredictionDict]
PRIOR_SHAPES = list[tuple[int, int]]
TARGETS = tuple[TargetDict, ...] | list[TargetDict]
BATCH = tuple[IMAGES, TARGETS]
NETWORK_OUTPUT = tuple[list[Tensor], list[DetectionLossRecord]]
