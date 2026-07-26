from collections.abc import Sequence
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
    """Decoded predictions for a single image.

    Attributes:
        boxes: Predicted boxes shaped ``[N, 4]`` in ``xyxy`` coordinates.
        confidences: Confidence logits, or normalized probabilities when the input is normalized, shaped ``[N]``.
        classprobs: Class logits, or normalized probabilities when the input is normalized, shaped ``[N, C]``.

    """

    boxes: Tensor
    confidences: Tensor
    classprobs: Tensor


class TargetDict(TypedDict):
    """Ground-truth detection targets for a single image.

    Attributes:
        boxes: Target boxes shaped ``[N, 4]`` in ``xyxy`` pixel coordinates.
        labels: Class labels shaped ``[N]``.

    """

    boxes: Tensor
    labels: Tensor


class PackedTargetDict(TypedDict):
    """Targets for a whole batch concatenated into flat tensors.

    Packing avoids Python-level looping over images and removes the need for padding to the largest target count.
    The data module produces this format via :func:`collate_packed_batch` and the model consumes it directly.

    Attributes:
        boxes: Concatenated boxes shaped ``[T, 4]`` in ``xyxy`` pixel coordinates, where ``T`` is the total number of
            targets across the batch.
        labels: Concatenated class labels shaped ``[T]``.
        batch_indices: Image index for each target row, shaped ``[T]``. Used for scatter-style operations over the
            packed tensors.
        counts: Number of targets per image, one entry per image in the batch. ``len(counts)`` is the batch size.
            Unlike ``batch_indices``, this encodes images with zero targets and is used to split the packed tensors
            back into per-image lists via ``Tensor.split(counts)``.

    """

    boxes: Tensor
    labels: Tensor
    batch_indices: Tensor
    counts: list[int]


PREDICTIONS = Sequence[PredictionDict]
PRIOR_SHAPES = list[tuple[int, int]]
BATCH = tuple[Tensor, PackedTargetDict]
NETWORK_OUTPUT = tuple[list[Tensor], list[DetectionLossRecord]]
