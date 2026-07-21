from collections.abc import Callable
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Literal

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits, one_hot
from torchvision.ops import (
    box_iou,
    complete_box_iou,
    complete_box_iou_loss,
    distance_box_iou,
    distance_box_iou_loss,
    generalized_box_iou,
    generalized_box_iou_loss,
)

from .matching_result import DenseMatchingResult, MatchingResult, SparseMatchingResult
from .types import PREDICTIONS, PredictionDict, TargetDict

_PredKey = Literal["boxes", "confidences", "classprobs"]


def _flatten_predictions(preds: PREDICTIONS) -> list[PredictionDict]:
    """Flattens the spatial and anchor dimensions of each image's predictions into a single axis.

    Args:
        preds: Predictions for each image, as returned by a detection layer.

    Returns:
        One ``PredictionDict`` per image with boxes shaped ``[N, 4]``, confidences shaped ``[N]``, and classprobs
        shaped ``[N, C]``, where ``N`` is the total number of anchors across all spatial locations.

    """
    return [
        {
            "boxes": pred["boxes"].reshape(-1, 4),
            "confidences": pred["confidences"].reshape(-1),
            "classprobs": pred["classprobs"].reshape(-1, pred["classprobs"].shape[-1]),
        }
        for pred in preds
    ]


def _stack_predictions(preds: PREDICTIONS) -> PredictionDict:
    """Flattens and stacks predictions from all images into batched tensors.

    Args:
        preds: Predictions for each image, as returned by a detection layer.

    Returns:
        A single ``PredictionDict`` with boxes shaped ``[B, N, 4]``, confidences shaped ``[B, N]``, and classprobs
        shaped ``[B, N, C]``.

    """
    flat_preds = _flatten_predictions(preds)
    return {
        "boxes": torch.stack([pred["boxes"] for pred in flat_preds]),
        "confidences": torch.stack([pred["confidences"] for pred in flat_preds]),
        "classprobs": torch.stack([pred["classprobs"] for pred in flat_preds]),
    }


def _gather_predictions(
    preds: list[PredictionDict], selectors: list[Tensor], keys: tuple[_PredKey, ...]
) -> dict[str, Tensor]:
    """Selects predictions from each image by index and concatenates them across the batch.

    Args:
        preds: Flattened predictions for each image.
        selectors: Per-image index or boolean tensors selecting which predictions to gather.
        keys: Which prediction fields to include in the result.

    Returns:
        A dictionary with the requested fields concatenated across all images.

    """
    return {
        key: torch.cat([pred[key][selector] for pred, selector in zip(preds, selectors, strict=True)]) for key in keys
    }


def box_iou_loss(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    return 1.0 - box_iou(boxes1, boxes2).diagonal()


_iou_and_loss_functions = {
    "iou": (box_iou, box_iou_loss),
    "giou": (generalized_box_iou, generalized_box_iou_loss),
    "diou": (distance_box_iou, distance_box_iou_loss),
    "ciou": (complete_box_iou, complete_box_iou_loss),
}


def _get_iou_and_loss_functions(name: str) -> tuple[Callable, Callable]:
    """Returns functions for calculating the IoU and the IoU loss, given the IoU variant name.

    Args:
        name: Name of the IoU variant. Either "iou", "giou", "diou", or "ciou".

    Returns:
        A tuple of two functions. The first function calculates the pairwise IoU and the second function calculates the
        elementwise loss.

    """
    if name not in _iou_and_loss_functions:
        raise ValueError(f"Unknown IoU function '{name}'.")
    iou_func, loss_func = _iou_and_loss_functions[name]
    if not callable(iou_func):
        raise ValueError(f"The IoU function '{name}' is not supported by the installed version of Torchvision.")
    assert callable(loss_func)
    return iou_func, loss_func


def _size_compensation(targets: Tensor, image_size: Tensor) -> Tensor:
    """Calculates the size compensation factor for the overlap loss.

    The overlap losses for each target should be multiplied by the returned weight. The returned value is
    `2 - (unit_width * unit_height)`, which is large for small boxes (the maximum value is 2) and small for large boxes
    (the minimum value is 1).

    Args:
        targets: An ``[N, 4]`` matrix of target `(x1, y1, x2, y2)` coordinates.
        image_size: Image size, which is used to scale the target boxes to unit coordinates.

    Returns:
        The size compensation factor.

    """
    unit_wh = targets[:, 2:] / image_size
    return 2 - (unit_wh[:, 0] * unit_wh[:, 1])


def _pairwise_confidence_loss(
    preds: Tensor, overlap: Tensor, bce_func: Callable, predict_overlap: float | None
) -> Tensor:
    """Calculates the confidence loss for every pair of a foreground anchor and a target.

    If ``predict_overlap`` is ``None``, the target confidence will be 1. If ``predict_overlap`` is 1.0, ``overlap`` will
    be used as the target confidence. Otherwise this parameter defines a balance between these two targets. The method
    returns a vector of losses for each foreground anchor.

    Args:
        preds: An ``[N]`` vector of predicted confidences.
        overlap: An ``[N, M]`` matrix of overlaps between all predicted and target bounding boxes.
        bce_func: A function for calculating binary cross entropy.
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the overlap.

    Returns:
        An ``[N, M]`` matrix of confidence losses between all predictions and targets.

    """
    if predict_overlap is not None:
        # When predicting overlap, target confidence is different for each pair of a prediction and a target. The
        # tensors have to be broadcasted to [N, M].
        preds = preds.unsqueeze(1).expand(overlap.shape)
        targets = torch.ones_like(preds) - predict_overlap
        # Distance-IoU may return negative "overlaps", so we have to make sure that the targets are not negative.
        targets += predict_overlap * overlap.detach().clamp(min=0)
        return bce_func(preds, targets, reduction="none")

    # When not predicting overlap, target confidence is the same for every prediction,
    # but we should still return a matrix.
    targets = torch.ones_like(preds)
    return bce_func(preds, targets, reduction="none").unsqueeze(1).expand(overlap.shape)


def _foreground_confidence_loss(
    preds: Tensor, overlap: Tensor, bce_func: Callable, predict_overlap: float | None
) -> Tensor:
    """Calculates the sum of the confidence losses for foreground anchors and their matched targets.

    If ``predict_overlap`` is ``None``, the target confidence will be 1. If ``predict_overlap`` is 1.0, ``overlap`` will
    be used as the target confidence. Otherwise this parameter defines a balance between these two targets. The method
    returns a vector of losses for each foreground anchor.

    Args:
        preds: A vector of predicted confidences.
        overlap: A vector of overlaps between matched target and predicted bounding boxes.
        bce_func: A function for calculating binary cross entropy.
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1, and 1.0 means that the target confidence is the overlap.

    Returns:
        The sum of the confidence losses for foreground anchors.

    """
    targets = torch.ones_like(preds)
    if predict_overlap is not None:
        targets -= predict_overlap
        # Distance-IoU may return negative "overlaps", so we have to make sure that the targets are not negative.
        targets += predict_overlap * overlap.detach().clamp(min=0)
    return bce_func(preds, targets, reduction="sum")


def _background_confidence_loss(preds: Tensor, bce_func: Callable) -> Tensor:
    """Calculates the sum of the confidence losses for background anchors.

    Args:
        preds: A vector of predicted confidences for background anchors.
        bce_func: A function for calculating binary cross entropy.

    Returns:
        The sum of the background confidence losses.

    """
    targets = torch.zeros_like(preds)
    return bce_func(preds, targets, reduction="sum")


def _background_class_loss(preds: Tensor, bce_func: Callable) -> Tensor:
    """Calculates the sum of the classification losses for background anchors.

    Confidence-free heads (as in YOLOv8) have no separate confidence output, so background anchors are instead
    supervised by driving all of their class probabilities toward zero. This replaces the background confidence loss.

    Args:
        preds: An ``[N, C]`` matrix of predicted class scores for background anchors.
        bce_func: A function for calculating binary cross entropy.

    Returns:
        The sum of the background classification losses.

    """
    targets = torch.zeros_like(preds)
    return bce_func(preds, targets, reduction="sum")


def _target_labels_to_probs(
    targets: Tensor, num_classes: int, dtype: torch.dtype, label_smoothing: float | None = None
) -> Tensor:
    """If ``targets`` is a vector of class labels, converts it to a matrix of one-hot class probabilities.

    If label smoothing is disabled, the returned target probabilities will be binary. If label smoothing is enabled, the
    target probabilities will be, ``(label_smoothing / 2)`` or ``(label_smoothing / 2) + (1.0 - label_smoothing)``. That
    corresponds to label smoothing with two categories, since the YOLO model does multi-label classification.

    Args:
        targets: An ``[M, C]`` matrix of target class probabilities or an ``[M]`` vector of class labels.
        num_classes: The number of classes (C dimension) for the new targets. If ``targets`` is already two-dimensional,
            checks that the length of the second dimension matches this number.
        dtype: Floating-point data type to be used for the one-hot targets.
        label_smoothing: The epsilon parameter (weight) for label smoothing. 0.0 means no smoothing (binary targets),
            and 1.0 means that the target probabilities are always 0.5.

    Returns:
        An ``[M, C]`` matrix of target class probabilities.

    """
    if targets.ndim == 1:
        if torch.is_floating_point(targets):
            raise ValueError("Class-index targets must use an integer dtype.")
        # The data may contain a different number of classes than what the model predicts. In case a label is
        # greater than the number of predicted classes, it will be mapped to the last class.
        last_class = torch.tensor(num_classes - 1, device=targets.device)
        targets = torch.min(targets, last_class)
        targets = one_hot(targets.to(dtype=torch.int64), num_classes)  # one_hot() only supports int64 dtype.
    elif targets.shape[-1] != num_classes:
        raise ValueError(
            f"The number of classes in the data ({targets.shape[-1]}) doesn't match the number of classes "
            f"predicted by the model ({num_classes})."
        )
    targets = targets.to(dtype=dtype)
    if label_smoothing is not None:
        targets = (label_smoothing / 2) + targets * (1.0 - label_smoothing)
    return targets


@dataclass
class YOLOLosses:
    overlap: Tensor
    confidence: Tensor
    classification: Tensor


class YOLOLoss:
    """A class for calculating the YOLO losses from predictions and targets.

    If label smoothing is enabled, the target class probabilities will be ``(label_smoothing / 2)`` or
    ``(label_smoothing / 2) + (1.0 - label_smoothing)``, instead of 0 or 1. That corresponds to label smoothing with two
    categories, since the YOLO model does multi-label classification.

    Args:
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.

    """

    def __init__(
        self,
        overlap_func: str | Callable = "ciou",
        predict_overlap: float | None = None,
        label_smoothing: float | None = None,
        overlap_multiplier: float = 5.0,
        confidence_multiplier: float = 1.0,
        class_multiplier: float = 1.0,
        predict_confidence: bool = True,
    ):
        if callable(overlap_func):
            self._pairwise_overlap = overlap_func
            self._elementwise_overlap_loss = lambda boxes1, boxes2: 1.0 - overlap_func(boxes1, boxes2).diagonal()
        else:
            self._pairwise_overlap, self._elementwise_overlap_loss = _get_iou_and_loss_functions(overlap_func)

        self.predict_overlap = predict_overlap
        self.label_smoothing = label_smoothing
        self.overlap_multiplier = overlap_multiplier
        self.confidence_multiplier = confidence_multiplier
        self.class_multiplier = class_multiplier
        self.predict_confidence = predict_confidence

    def pairwise(
        self,
        preds: PredictionDict,
        targets: TargetDict,
        input_is_normalized: bool,
    ) -> tuple[YOLOLosses, Tensor]:
        """Calculates matrices containing the losses for all prediction/target pairs.

        This method is called for obtaining costs for SimOTA matching.

        Args:
            preds: A dictionary of predictions, containing "boxes", "confidences", and "classprobs". Each tensor
                contains `N` rows.
            targets: A dictionary of training targets, containing "boxes" and "labels". Each tensor contains `M` rows.
            input_is_normalized: If ``False``, input is logits, if ``True``, input is normalized to `0..1`.

        Returns:
            Loss matrices and an overlap matrix. Each matrix is shaped ``[N, M]``.

        """
        loss_shape = torch.Size([len(preds["boxes"]), len(targets["boxes"])])

        bce_func: Callable[..., Tensor] = (
            binary_cross_entropy if input_is_normalized else binary_cross_entropy_with_logits
        )

        overlap = self._pairwise_overlap(preds["boxes"], targets["boxes"])
        assert overlap.shape == loss_shape

        overlap_loss = 1.0 - overlap
        assert overlap_loss.shape == loss_shape

        confidence_loss = _pairwise_confidence_loss(preds["confidences"], overlap, bce_func, self.predict_overlap)
        assert confidence_loss.shape == loss_shape

        pred_probs = preds["classprobs"].unsqueeze(1)  # [N, 1, classes]
        target_probs = _target_labels_to_probs(
            targets["labels"], pred_probs.shape[-1], pred_probs.dtype, self.label_smoothing
        )
        target_probs = target_probs.unsqueeze(0)  # [1, M, classes]
        pred_probs, target_probs = torch.broadcast_tensors(pred_probs, target_probs)
        class_loss = bce_func(pred_probs, target_probs, reduction="none").sum(-1)
        assert class_loss.shape == loss_shape

        losses = YOLOLosses(
            overlap_loss * self.overlap_multiplier,
            confidence_loss * self.confidence_multiplier,
            class_loss * self.class_multiplier,
        )

        return losses, overlap

    @singledispatchmethod
    def sums(
        self,
        matching: MatchingResult,
        preds: PREDICTIONS,  # noqa: ARG002
        input_is_normalized: bool,  # noqa: ARG002
        image_size: Tensor,  # noqa: ARG002
    ) -> YOLOLosses:
        """Calculates loss sums for the predictions matched by ``matching``.

        The reduction is specialized by the matching result layout: dense results are reduced with masking over every
        anchor, while sparse results are reduced over only the matched foreground anchors.

        Args:
            matching: The target assignments produced by a matcher.
            preds: Predicted boxes, confidences, and class scores for each image.
            input_is_normalized: If ``False``, input is logits, if ``True``, input is normalized to `0..1`.
            image_size: Width and height used for box-size weighting.

        Returns:
            Loss sums for overlap, confidence, and classification.

        """
        raise TypeError(f"Unsupported matching result type: {type(matching)!r}")

    @sums.register(DenseMatchingResult)
    def _dense_sums(
        self,
        matching: DenseMatchingResult,
        preds: PREDICTIONS,
        input_is_normalized: bool,
        image_size: Tensor,
    ) -> YOLOLosses:
        pred_batch = _stack_predictions(preds)
        foreground = matching.foreground
        background = matching.background
        assignment_weights = matching.assignment_weights.to(pred_batch["boxes"].dtype)
        bce_func: Callable[..., Tensor] = (
            binary_cross_entropy if input_is_normalized else binary_cross_entropy_with_logits
        )
        target_boxes = matching.target_boxes.flatten(0, 1)
        pred_boxes = pred_batch["boxes"].flatten(0, 1)
        target_boxes = torch.where(foreground.flatten().unsqueeze(-1), target_boxes, pred_boxes.detach())
        overlap_loss_values = self._elementwise_overlap_loss(target_boxes, pred_boxes).view_as(foreground)
        overlap = 1.0 - overlap_loss_values
        box_weights = _size_compensation(target_boxes, image_size).view_as(foreground)
        overlap_loss = torch.where(foreground, overlap_loss_values * box_weights * assignment_weights, 0.0).sum()

        if self.predict_confidence:
            confidence_targets = assignment_weights
            if self.predict_overlap is not None:
                overlap_targets = assignment_weights * self.predict_overlap * overlap.detach().clamp(min=0)
                confidence_targets = confidence_targets * (1 - self.predict_overlap) + overlap_targets
            confidence_values = bce_func(pred_batch["confidences"], confidence_targets, reduction="none")
            background_values = bce_func(
                pred_batch["confidences"], torch.zeros_like(pred_batch["confidences"]), reduction="none"
            )
            confidence_loss = torch.where(foreground, confidence_values, 0.0).sum()
            confidence_loss = confidence_loss + torch.where(background, background_values, 0.0).sum()
        else:
            confidence_loss = overlap_loss.new_zeros(())

        target_labels = matching.target_labels
        flat_target_labels = target_labels.flatten(0, 1) if target_labels.ndim == 3 else target_labels.flatten()
        target_probs = _target_labels_to_probs(
            flat_target_labels,
            pred_batch["classprobs"].shape[-1],
            pred_batch["classprobs"].dtype,
            self.label_smoothing,
        ).view_as(pred_batch["classprobs"])
        class_targets = torch.where(foreground.unsqueeze(-1), target_probs * assignment_weights.unsqueeze(-1), 0.0)
        class_values = bce_func(pred_batch["classprobs"], class_targets, reduction="none")
        class_mask = foreground | (background if not self.predict_confidence else torch.zeros_like(background))
        class_loss = torch.where(class_mask.unsqueeze(-1), class_values, 0.0).sum()

        return YOLOLosses(
            overlap_loss * self.overlap_multiplier,
            confidence_loss * self.confidence_multiplier,
            class_loss * self.class_multiplier,
        )

    @sums.register(SparseMatchingResult)
    def _sparse_sums(
        self,
        matching: SparseMatchingResult,
        preds: PREDICTIONS,
        input_is_normalized: bool,
        image_size: Tensor,
    ) -> YOLOLosses:
        flat_preds = _flatten_predictions(preds)
        foreground = _gather_predictions(
            flat_preds, [image.foreground for image in matching.images], ("boxes", "confidences", "classprobs")
        )
        background = _gather_predictions(
            flat_preds, [image.background for image in matching.images], ("confidences", "classprobs")
        )
        target_boxes = torch.cat([image.target_boxes for image in matching.images])
        target_labels = torch.cat([image.target_labels for image in matching.images])

        bce_func: Callable[..., Tensor] = (
            binary_cross_entropy if input_is_normalized else binary_cross_entropy_with_logits
        )

        overlap_loss_values = self._elementwise_overlap_loss(target_boxes, foreground["boxes"])
        overlap = 1.0 - overlap_loss_values
        overlap_loss = (overlap_loss_values * _size_compensation(target_boxes, image_size)).sum()

        if self.predict_confidence:
            confidence_loss = _foreground_confidence_loss(
                foreground["confidences"], overlap, bce_func, self.predict_overlap
            )
            confidence_loss = confidence_loss + _background_confidence_loss(background["confidences"], bce_func)
        else:
            confidence_loss = overlap_loss.new_zeros(())

        target_probs = _target_labels_to_probs(
            target_labels, foreground["classprobs"].shape[-1], foreground["classprobs"].dtype, self.label_smoothing
        )
        class_loss = bce_func(foreground["classprobs"], target_probs, reduction="sum")
        if not self.predict_confidence:
            class_loss = class_loss + _background_class_loss(background["classprobs"], bce_func)

        return YOLOLosses(
            overlap_loss * self.overlap_multiplier,
            confidence_loss * self.confidence_multiplier,
            class_loss * self.class_multiplier,
        )
