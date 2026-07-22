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


@dataclass(frozen=True)
class LevelPredictions:
    """Decoded predictions and geometry for one detection feature level.

    Attributes:
        detections: Public normalized detections shaped ``[B, N, C + 5]``.
        boxes: Decoded boxes shaped ``[B, N, 4]``.
        confidences: Confidence logits, or normalized probabilities when the input is normalized, shaped ``[B, N]``.
        classprobs: Classification logits, or normalized probabilities when the input is normalized, shaped
            ``[B, N, C]``.
        anchor_points: Candidate center points in image coordinates shaped ``[N, 2]``.
        spatial_shape: Feature grid height, width, and candidates per cell.

    """

    detections: Tensor
    boxes: Tensor
    confidences: Tensor
    classprobs: Tensor
    anchor_points: Tensor
    spatial_shape: tuple[int, int, int]

    def as_grid(self) -> list[PredictionDict]:
        """Return per-image prediction dictionaries with tensors shaped by grid cell and candidate."""
        height, width, candidates_per_cell = self.spatial_shape
        return [
            {
                "boxes": boxes.view(height, width, candidates_per_cell, 4),
                "confidences": confidences.view(height, width, candidates_per_cell),
                "classprobs": classprobs.view(height, width, candidates_per_cell, -1),
            }
            for boxes, confidences, classprobs in zip(self.boxes, self.confidences, self.classprobs, strict=True)
        ]


class PredictionDict(TypedDict):
    boxes: Tensor
    confidences: Tensor
    classprobs: Tensor


class TargetDict(TypedDict):
    boxes: Tensor
    labels: Tensor


IMAGES = tuple[Tensor, ...] | list[Tensor]
PREDICTIONS = tuple[PredictionDict, ...] | list[PredictionDict]
PRIOR_SHAPES = list[tuple[int, int]]
TARGETS = tuple[TargetDict, ...] | list[TargetDict]
BATCH = tuple[IMAGES, TARGETS]
NETWORK_OUTPUT = tuple[list[Tensor], list[DetectionLossRecord]]
