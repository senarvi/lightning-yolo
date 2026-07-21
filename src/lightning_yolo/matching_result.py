from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


class MatchingResult(ABC):
    """Assignment of predictions to targets for one training batch.

    Two concrete layouts implement this interface, so the loss can be computed in the form that suits each matcher.
    ``DenseMatchingResult`` keeps fixed-shape ``[B, N]`` tensors, which lets the task-aligned matcher compute the loss
    with masking and avoid host synchronization. ``SparseMatchingResult`` keeps only the matched predictions per image,
    which lets the hard-assignment matchers compute the loss over their few foreground anchors instead of every anchor.

    """

    @property
    @abstractmethod
    def assignment_weight_sum(self) -> Tensor:
        """Sum of assignment weights, used to normalize the detection losses."""


@dataclass(frozen=True)
class DenseMatchingResult(MatchingResult):
    """Fixed-shape batched assignments produced by the task-aligned matcher.

    Args:
        foreground: Boolean ``[B, N]`` mask selecting predictions assigned to targets.
        background: Boolean ``[B, N]`` mask selecting predictions supervised as background.
        target_boxes: Matched target boxes shaped ``[B, N, 4]``.
        target_labels: Matched class indices ``[B, N]`` or class masks ``[B, N, C]``.
        assignment_weights: Per-prediction matching weights shaped ``[B, N]``.

    """

    foreground: Tensor
    background: Tensor
    target_boxes: Tensor
    target_labels: Tensor
    assignment_weights: Tensor

    @property
    def assignment_weight_sum(self) -> Tensor:
        return self.assignment_weights.sum()


@dataclass(frozen=True)
class ImageMatch:
    """Foreground assignments and background mask for a single image.

    Args:
        foreground: Flat indices ``[M]`` of the matched predictions, ordered to match ``target_boxes``.
        background: Boolean ``[N]`` mask selecting predictions supervised as background.
        target_boxes: Matched target boxes shaped ``[M, 4]``.
        target_labels: Matched class indices ``[M]`` or class masks ``[M, C]``.

    """

    foreground: Tensor
    background: Tensor
    target_boxes: Tensor
    target_labels: Tensor


@dataclass(frozen=True)
class SparseMatchingResult(MatchingResult):
    """Per-image assignments produced by hard-assignment matchers.

    Hard matchers assign a small number of foreground anchors per image, so the loss only needs those predictions
    rather than a dense mask over every anchor.

    Args:
        images: One :class:`ImageMatch` per image in the batch.

    """

    images: list[ImageMatch]

    @property
    def assignment_weight_sum(self) -> Tensor:
        # Hard matchers assign unit weight to each foreground anchor, so the weight sum is the foreground count.
        total = sum(image.foreground.numel() for image in self.images)
        return torch.tensor(total, device=self.images[0].background.device)
