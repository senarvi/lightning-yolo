from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import Tensor


@dataclass(frozen=True)
class DetectionLossContribution:
    """Unnormalized detection loss contribution from one detection head.

    Contributions remain additive across feature levels and auxiliary heads: their sums and normalizers must be
    combined before normalization to preserve the correct weighting.

    Attributes:
        sums: Scaled loss sums shaped ``[components]``.
        normalizers: Normalizers for ``sums``, shaped ``[components]``.
        names: Component names in the same order as ``sums`` and ``normalizers``.

    """

    sums: Tensor
    normalizers: Tensor
    names: tuple[str, ...] = ("overlap", "confidence", "class")

    def scaled(self, scale: float) -> DetectionLossContribution:
        """Scales the loss sums and normalizers by the same factor.

        Args:
            scale: Multiplier applied to this contribution in the global loss average.

        """
        return DetectionLossContribution(sums=self.sums * scale, normalizers=self.normalizers * scale, names=self.names)


@dataclass(frozen=True)
class DetectionLoss:
    """Normalized detection loss components used for optimization and logging.

    Attributes:
        values: Normalized, scaled loss components shaped ``[components]``.
        names: Component names in the same order as ``values``.

    """

    values: Tensor
    names: tuple[str, ...]

    @property
    def total(self) -> Tensor:
        """Returns the sum of the normalized loss components."""
        return self.values.sum()

    @classmethod
    def from_contributions(cls, contributions: Sequence[DetectionLossContribution]) -> DetectionLoss:
        """Combines and normalizes detection-head contributions.

        Args:
            contributions: Contributions with matching component names.

        Returns:
            The normalized detection loss.

        """
        if not contributions:
            raise ValueError("Expected at least one detection loss contribution.")

        names = contributions[0].names
        if any(contribution.names != names for contribution in contributions):
            raise ValueError("Cannot aggregate detection loss contributions with different component names.")

        return cls(
            values=torch.stack([contribution.sums for contribution in contributions]).sum(0)
            / torch.stack([contribution.normalizers for contribution in contributions]).sum(0).clamp_min(1),
            names=names,
        )


@dataclass(frozen=True)
class PriorShapeLevelPredictions:
    """Decoded predictions and geometry for one detection feature level.

    Attributes:
        detections: Public normalized detections shaped ``[batch_size, N, classes + 5]``.
        boxes: Decoded boxes shaped ``[batch_size, N, 4]``.
        confidences: Confidence logits, or normalized probabilities when the input is normalized, shaped
            ``[batch_size, N]``.
        classprobs: Classification logits, or normalized probabilities when the input is normalized, shaped
            ``[batch_size, N, classes]``.
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


@dataclass(frozen=True)
class DistributionalDistancePredictions:
    """Decoded anchor predictions with distributional distance-regression logits.

    Attributes:
        detections: Public pixel-space detections shaped ``[batch_size, N, classes + 5]``.
        pixel_boxes: Decoded corner-format boxes in image coordinates shaped ``[batch_size, N, 4]``.
        grid_boxes: Decoded corner-format boxes in feature-grid coordinates shaped ``[batch_size, N, 4]``.
        dfl_logits: Raw distance-bin logits shaped ``[batch_size, N, 4 * bins]``.
        class_logits: Raw class logits shaped ``[batch_size, N, classes]``.
        anchor_points: Candidate center points in grid coordinates shaped ``[N, 2]``.
        strides: Per-point image stride values shaped ``[N, 1]``.

    """

    detections: Tensor
    pixel_boxes: Tensor
    grid_boxes: Tensor
    dfl_logits: Tensor
    class_logits: Tensor
    anchor_points: Tensor
    strides: Tensor


class PredictionDict(TypedDict):
    """Decoded predictions for a single image.

    Attributes:
        boxes: Predicted boxes shaped ``[N, 4]`` in ``xyxy`` coordinates.
        confidences: Confidence logits, or normalized probabilities when the input is normalized, shaped ``[N]``.
        classprobs: Class logits, or normalized probabilities when the input is normalized, shaped
            ``[N, classes]``.

    """

    boxes: Tensor
    confidences: Tensor
    classprobs: Tensor


class TargetDict(TypedDict):
    """Ground-truth detection targets for a single image.

    Attributes:
        boxes: Target boxes shaped ``[N, 4]`` in ``xyxy`` pixel coordinates.
        labels: Class indices shaped ``[N]``, or a boolean class mask shaped ``[N, classes]`` for multi-label
            targets.

    """

    boxes: Tensor
    labels: Tensor


class PackedTargetDict(TypedDict):
    """Targets for a whole batch concatenated into flat tensors.

    Packing avoids Python-level looping over images and removes the need for padding to the largest target count.
    The data module produces this format via :func:`collate_packed_batch` and the model consumes it directly.

    Attributes:
        boxes: Concatenated boxes shaped ``[targets, 4]`` in ``xyxy`` pixel coordinates, where ``targets`` is
            the total number of targets across the batch.
        labels: Concatenated class indices shaped ``[targets]``, or a boolean class mask shaped
            ``[targets, classes]`` for multi-label targets.
        sample_idxs: Image index for each target row, shaped ``[targets]``. Used for scatter-style operations
            over the packed tensors.
        counts: Number of targets per image, one entry per image in the batch. ``len(counts)`` is the batch size.
            Unlike ``sample_idxs``, this encodes images with zero targets and is used to split the packed tensors
            back into per-image lists via ``Tensor.split(counts)``.

    """

    boxes: Tensor
    labels: Tensor
    sample_idxs: Tensor
    counts: list[int]


PREDICTIONS = Sequence[PredictionDict]
PRIOR_SHAPES = list[tuple[int, int]]
BATCH = tuple[Tensor, PackedTargetDict]
NETWORK_OUTPUT = tuple[list[Tensor], list[DetectionLossContribution]]
