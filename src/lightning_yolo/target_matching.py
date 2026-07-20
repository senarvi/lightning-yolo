from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou

from .loss import YOLOLoss
from .types import PREDICTIONS, PRIOR_SHAPES, TARGETS, PredictionDict, TargetDict
from .utils import aligned_iou, box_size_ratio, grid_centers, iou_below, is_inside_box

# A selector for matched predictions. Matchers may return either:
# 1) a tuple of index tensors (y_idxs, x_idxs, anchor_idxs), or
# 2) a boolean mask tensor that can be used directly for indexing.
PredSelector = tuple[Tensor, Tensor, Tensor] | Tensor


@dataclass(frozen=True)
class MatchingResult:
    """Predictions and targets selected by a target matcher."""

    pred_selector: PredSelector
    background_selector: Tensor
    target_selector: Tensor
    assignment_weight_sum: int | float


# A matching function takes batched predictions and targets, image size, and a boolean indicating whether the
# probabilities are normalized, and returns one matching result per image.
MatchingFn = Callable[[PREDICTIONS, TARGETS, Tensor, bool], list[MatchingResult]]


class ShapeMatching(ABC):
    """Selects which anchors are used to predict each target, by comparing the shape of the target box to a set of prior
    shapes.

    Most YOLO variants match targets to anchors based on prior shapes that are assigned to the anchors in the model
    configuration. The subclasses of ``ShapeMatching`` implement matching rules that compare the width and height of
    the targets to each prior shape (regardless of the location where the target is). When the model includes multiple
    detection layers, different shapes are defined for each layer. Usually there are three detection layers and three
    prior shapes per layer.

    The assignment weight sum returned by these hard-assignment matchers is the number of selected foreground anchors.
    It is used as the confidence loss normalizer.

    Args:
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.

    """

    def __init__(self, ignore_bg_threshold: float = 0.7) -> None:
        self.ignore_bg_threshold = ignore_bg_threshold

    def __call__(
        self,
        preds: PREDICTIONS,
        targets: TARGETS,
        image_size: Tensor,
        input_is_normalized: bool = False,  # noqa: ARG002
    ) -> list[MatchingResult]:
        """For each target, selects predictions from the same grid cell, where the center of the target box is.

        Typically there are three predictions per grid cell. Subclasses implement ``match()``, which selects the
        predictions within the grid cell.

        Args:
            preds: Predictions for each image.
            targets: Training targets for each image.
            image_size: Input image width and height.
            input_is_normalized: The predicted confidences and class probabilities have been normalized by logistic
                activation. This is used by the Darknet configurations of Scaled-YOLOv4.

        Returns:
            One matching result per image.

        """
        return [
            self._match_image(image_preds, image_targets, image_size)
            for image_preds, image_targets in zip(preds, targets, strict=True)
        ]

    def _match_image(self, preds: PredictionDict, targets: TargetDict, image_size: Tensor) -> MatchingResult:
        height, width = preds["boxes"].shape[:2]
        # This multiplier scales image coordinates to feature map coordinates.
        grid_size = torch.tensor([width, height], device=preds["boxes"].device)
        image_to_grid = torch.true_divide(grid_size, image_size)

        # Bounding box center coordinates are converted to the feature map dimensions so that the whole number tells the
        # cell index and the fractional part tells the location inside the cell.
        xywh = box_convert(targets["boxes"], in_fmt="xyxy", out_fmt="cxcywh")
        grid_xy = xywh[:, :2] * image_to_grid
        cell_i = grid_xy[:, 0].to(torch.int64).clamp(0, width - 1)
        cell_j = grid_xy[:, 1].to(torch.int64).clamp(0, height - 1)

        target_selector, anchor_selector = self.match(xywh[:, 2:])
        cell_i = cell_i[target_selector]
        cell_j = cell_j[target_selector]

        # Background mask is used to select anchors that are not responsible for predicting any object, for
        # calculating the part of the confidence loss with zero as the target confidence. It is set to False, if a
        # predicted box overlaps any target significantly, or if a prediction is matched to a target.
        background_mask = iou_below(preds["boxes"], targets["boxes"], self.ignore_bg_threshold)
        background_mask[cell_j, cell_i, anchor_selector] = False

        pred_selector = (cell_j, cell_i, anchor_selector)

        # Shape-based matchers make hard assignments, so each matched foreground anchor contributes one normalizer unit.
        assignment_weight_sum = cell_i.numel()

        return MatchingResult(pred_selector, background_mask, target_selector, assignment_weight_sum)

    @abstractmethod
    def match(self, wh: Tensor) -> tuple[Tensor, Tensor] | Tensor:
        """Selects anchors for each target based on the predicted shapes. The subclasses implement this method.

        Args:
            wh: A matrix of predicted width and height values.

        Returns:
            matched_targets, matched_anchors: Two vectors or a `2xN` matrix. The first vector is used to select the
            targets that this layer matched and the second one lists the matching anchors within the grid cell.

        """
        pass


class HighestIoUMatching(ShapeMatching):
    """For each target, select the prior shape that gives the highest IoU.

    This is the original YOLO matching rule.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain (width, height) tuples in the
            network input resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.

    """

    def __init__(
        self,
        prior_shapes: PRIOR_SHAPES,
        prior_shape_idxs: Sequence[int],
        ignore_bg_threshold: float = 0.7,
    ) -> None:
        super().__init__(ignore_bg_threshold)
        self.prior_shapes = prior_shapes
        # anchor_map maps each global anchor index to an anchor in this layer, or to -1 if the anchor belongs to a
        # different layer.
        # This layer ignores the target if all the selected anchors are in another layer.
        self.anchor_map = [
            prior_shape_idxs.index(idx) if idx in prior_shape_idxs else -1 for idx in range(len(prior_shapes))
        ]

    def match(self, wh: Tensor) -> tuple[Tensor, Tensor] | Tensor:
        prior_wh = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)
        anchor_map = torch.tensor(self.anchor_map, dtype=torch.int64, device=wh.device)

        ious = aligned_iou(wh, prior_wh)
        highest_iou_anchors = ious.max(1).indices
        highest_iou_anchors = anchor_map[highest_iou_anchors]
        matched_targets = highest_iou_anchors >= 0
        matched_anchors = highest_iou_anchors[matched_targets]
        return matched_targets, matched_anchors


class IoUThresholdMatching(ShapeMatching):
    """For each target, select all prior shapes that give a high enough IoU.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain (width, height) tuples in the
            network input resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        threshold: IoU threshold for matching.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor will not be taken into account when
            calculating the confidence loss.

    """

    def __init__(
        self,
        prior_shapes: PRIOR_SHAPES,
        prior_shape_idxs: Sequence[int],
        threshold: float,
        ignore_bg_threshold: float = 0.7,
    ) -> None:
        super().__init__(ignore_bg_threshold)
        self.prior_shapes = [prior_shapes[idx] for idx in prior_shape_idxs]
        self.threshold = threshold

    def match(self, wh: Tensor) -> tuple[Tensor, Tensor] | Tensor:
        prior_wh = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)

        ious = aligned_iou(wh, prior_wh)
        above_threshold = (ious > self.threshold).nonzero()
        return above_threshold.T


class SizeRatioMatching(ShapeMatching):
    """For each target, select those prior shapes, whose width and height relative to the target is below given ratio.

    This is the matching rule used by Ultralytics YOLOv5 implementation.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain (width, height) tuples in the
            network input resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        threshold: Size ratio threshold for matching.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor will not be taken into account when
            calculating the confidence loss.

    """

    def __init__(
        self,
        prior_shapes: PRIOR_SHAPES,
        prior_shape_idxs: Sequence[int],
        threshold: float,
        ignore_bg_threshold: float = 0.7,
    ) -> None:
        super().__init__(ignore_bg_threshold)
        self.prior_shapes = [prior_shapes[idx] for idx in prior_shape_idxs]
        self.threshold = threshold

    def match(self, wh: Tensor) -> tuple[Tensor, Tensor] | Tensor:
        prior_wh = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)
        return (box_size_ratio(wh, prior_wh) < self.threshold).nonzero().T


def _sim_ota_match(costs: Tensor, ious: Tensor) -> tuple[Tensor, Tensor]:
    """Implements the SimOTA matching rule.

    The number of units supplied by each supplier (training target) needs to be decided in the Optimal Transport
    problem. "Dynamic k Estimation" uses the sum of the top 10 IoU values (casted to int) between the target and the
    predicted boxes.

    Args:
        costs: A ``[predictions, targets]`` matrix of losses.
        ious: A ``[predictions, targets]`` matrix of IoUs.

    Returns:
        A mask of matched predictions and the target index selected for each matched prediction.

    """
    num_preds, _num_targets = ious.shape
    matching_matrix = torch.zeros_like(costs, dtype=torch.bool)

    if ious.numel() > 0:
        # For each target, k is the sum of the 10 highest IoUs.
        top10_iou = torch.topk(ious, min(10, num_preds), dim=0).values.sum(0)
        ks = torch.clip(top10_iou.int(), min=1)

        # For each target, select k predictions with the lowest cost.
        sorted_preds = torch.argsort(costs, dim=0)
        selected_ranks = torch.arange(num_preds, device=costs.device).unsqueeze(1) < ks.unsqueeze(0)
        matching_matrix.scatter_(0, sorted_preds, selected_ranks)

        # If there's more than one match for some prediction, match it with the best target. Now we consider all
        # targets, regardless of whether they were originally matched with the prediction or not.
        more_than_one_match = matching_matrix.sum(1) > 1
        best_targets = costs[more_than_one_match, :].argmin(1)
        matching_matrix[more_than_one_match, :] = False
        matching_matrix[more_than_one_match, best_targets] = True

    # For each matched prediction, select the index of the assigned target.
    pred_mask = matching_matrix.sum(1) > 0
    target_selector = (
        matching_matrix[pred_mask, :].int().argmax(1)
        if matching_matrix.shape[1] > 0
        else torch.empty(0, dtype=torch.int64, device=costs.device)
    )
    return pred_mask, target_selector


def _tal_match(
    align_metric: Tensor,
    ious: Tensor,
    inside_selector: Tensor,
    target_mask: Tensor,
    topk: int,
    eps: float = 1e-9,
) -> tuple[Tensor, Tensor, Tensor]:
    """Implements the TAL matching rule.

    For each target, this method considers only anchors whose center point is inside the target box, ranks them using
    the TAL alignment score, and marks the ``topk`` highest-scoring anchors as matched. If an anchor is selected by
    more than one target, it is assigned to the target with the highest IoU.

    Args:
        align_metric: ``[B, N, T_max]`` alignment scores.
        ious: ``[B, N, T_max]`` IoU values.
        inside_selector: ``[B, N, T_max]`` bool — True when an anchor centre is inside the target box.
        target_mask: ``[B, T_max]`` bool — True for real targets, False for padding.
        topk: Maximum number of anchors to select per target.

    Returns:
        pred_mask ``[B, N]``, target_selector ``[B, N]`` (meaningful only where pred_mask is True), and
        assignment_weights ``[B, N]`` used for loss normalisation.

    """
    batch_size, num_preds, max_targets = align_metric.shape

    # Anchors outside the target box and padded targets are excluded from top-k selection.
    masked_metric = align_metric.clone()
    masked_metric[~inside_selector] = -1.0
    masked_metric[~target_mask[:, None, :].expand(batch_size, num_preds, max_targets)] = -1.0

    # For each target, select top-k anchors by the alignment metric among anchors that are inside the target box.
    masked_t = masked_metric.permute(0, 2, 1)  # [B, T_max, N]
    k = min(topk, num_preds)
    _, topk_indices = torch.topk(masked_t, k=k, dim=2)  # [B, T_max, k]

    valid = torch.gather(inside_selector.permute(0, 2, 1), 2, topk_indices)

    # Scatter topk selections back into the [B, N, T_max] matching matrix.
    # topk_indices: [B, T_max, k] → transpose to [B, k, T_max] for scatter on dim 1.
    idx = topk_indices.permute(0, 2, 1)  # [B, k, T_max]
    val = valid.permute(0, 2, 1)  # [B, k, T_max]
    matching_matrix = torch.zeros_like(align_metric, dtype=torch.bool)
    matching_matrix.scatter_(1, idx, val)

    # If there is more than one match for some prediction, match it with the target that has the highest IoU.
    multiple = matching_matrix.sum(dim=2) > 1  # [B, N]
    best_targets = ious.argmax(dim=2, keepdim=True)
    best_matches = torch.zeros_like(matching_matrix).scatter_(2, best_targets, True)
    matching_matrix = torch.where(multiple.unsqueeze(-1), best_matches, matching_matrix)

    # For those predictions that were matched, get the index of the target.
    pred_mask = matching_matrix.any(dim=2)  # [B, N]
    target_selector = matching_matrix.int().argmax(dim=2)  # [B, N]

    # Normalize each matched alignment score by the best matched alignment score for the same target, then scale it by
    # that target's best matched IoU.
    matched_metric = align_metric * matching_matrix.float()
    best_align = matched_metric.amax(dim=1, keepdim=True).clamp(min=eps)  # [B, 1, T_max]
    best_iou = (ious * matching_matrix.float()).amax(dim=1, keepdim=True)  # [B, 1, T_max]
    assignment_weights = (matched_metric * best_iou / best_align).amax(dim=2)  # [B, N]

    return pred_mask, target_selector, assignment_weights


def _probability_of_labels(pred_probs: Tensor, target_labels: Tensor) -> Tensor:
    """Computes a ``[predictions, targets]`` matrix of probabilities predicted for the ground-truth labels.

    The returned matrix is used as the class scores by TAL. For single-label targets, returns a matrix of the predicted
    probabilities for the target class. For multi-label targets, each prediction/target pair uses the sum of the
    predicted probabilities among the classes assigned to that target. TAL performs the top-k selection independently
    per target, so this is equivalent to using the average predicted probability.

    Args:
        pred_probs: Predicted class probabilities in a matrix shaped ``[predictions, num_classes]``.
        target_labels: Target labels either as a vector of class indices or a boolean mask shaped
            ``[targets, num_classes]``.

    Returns:
        A ``[predictions, targets]`` matrix of probabilities.

    """
    num_classes = pred_probs.shape[-1]

    if target_labels.ndim == 1:
        if torch.is_floating_point(target_labels):
            raise ValueError("Class-index targets must use an integer dtype.")

        # The data may contain a different number of classes than what the model predicts. In case a label is
        # greater than the number of predicted classes, it will be mapped to the last class.
        last_class = torch.tensor(num_classes - 1, device=target_labels.device)
        target_labels = torch.min(target_labels, last_class)
        return pred_probs[:, target_labels]

    if target_labels.ndim == 2:
        if target_labels.dtype != torch.bool:
            raise ValueError("Class-mask targets must use the bool dtype.")

        if target_labels.shape[-1] != num_classes:
            raise ValueError(
                f"The number of classes in the data ({target_labels.shape[-1]}) doesn't match the number of classes "
                f"predicted by the model ({num_classes})."
            )

        # For each prediction/target pair, take the predicted probability of every class that is assigned to the target.
        class_mask = target_labels.unsqueeze(0)  # [1, targets, num_classes]
        masked_probs = pred_probs.unsqueeze(1) * class_mask  # [predictions, targets, num_classes]
        return masked_probs.sum(-1)

    raise ValueError(f"Expected target labels to have shape [N] or [N, num_classes], got {list(target_labels.shape)}.")


class SimOTAMatching:
    """Selects which anchors are used to predict each target using the SimOTA matching rule.

    This is the matching rule used by YOLOX.

    The assignment weight sum is the number of anchors selected by dynamic-k matching after conflict resolution, with
    one unit per matched foreground anchor.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain (width, height) tuples in the
            network input resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        loss_func: A ``YOLOLoss`` object that can be used to calculate the pairwise costs.
        spatial_range: For each target, restrict to the anchors that are within an `N × N` grid cell are centered at the
            target, where `N` is the value of this parameter.
        size_range: For each target, restrict to the anchors whose prior dimensions are not larger than the target
            dimensions multiplied by this value and not smaller than the target dimensions divided by this value.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the predicted box has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.

    """

    def __init__(
        self,
        prior_shapes: PRIOR_SHAPES,
        prior_shape_idxs: Sequence[int],
        loss_func: YOLOLoss,
        spatial_range: float,
        size_range: float,
        ignore_bg_threshold: float = 0.7,
    ) -> None:
        self.prior_shapes = [prior_shapes[idx] for idx in prior_shape_idxs]
        self.loss_func = loss_func
        self.spatial_range = spatial_range
        self.size_range = size_range
        self.ignore_bg_threshold = ignore_bg_threshold

    def __call__(
        self,
        preds: PREDICTIONS,
        targets: TARGETS,
        image_size: Tensor,
        input_is_normalized: bool = False,
    ) -> list[MatchingResult]:
        """For each target, selects predictions using the SimOTA matching rule.

        Args:
            preds: Predictions for each image.
            targets: Training targets for each image.
            image_size: Input image width and height.
            input_is_normalized: The predicted confidences and class probabilities have been normalized by logistic
                activation. This is used by the Darknet configurations of Scaled-YOLOv4.

        Returns:
            One matching result per image.

        """
        return [
            self._match_image(image_preds, image_targets, image_size, input_is_normalized)
            for image_preds, image_targets in zip(preds, targets, strict=True)
        ]

    def _match_image(
        self,
        preds: PredictionDict,
        targets: TargetDict,
        image_size: Tensor,
        input_is_normalized: bool,
    ) -> MatchingResult:
        height, width, boxes_per_cell, _ = preds["boxes"].shape
        prior_mask, anchor_inside_target = self._get_prior_mask(targets, image_size, width, height, boxes_per_cell)
        prior_preds: PredictionDict = {
            "boxes": preds["boxes"][prior_mask],
            "confidences": preds["confidences"][prior_mask],
            "classprobs": preds["classprobs"][prior_mask],
        }

        losses, ious = self.loss_func.pairwise(prior_preds, targets, input_is_normalized=input_is_normalized)
        costs = losses.overlap + losses.confidence + losses.classification
        costs += 100000.0 * ~anchor_inside_target
        pred_mask, target_selector = _sim_ota_match(costs, ious)

        # SimOTA makes hard assignments, so each selected foreground anchor contributes one normalizer unit.
        assignment_weight_sum = int(pred_mask.sum().item())

        # Replace the candidate-prior True values with the actual SimOTA matches.
        prior_mask[prior_mask.nonzero(as_tuple=True)] = pred_mask

        # Background mask is used to select anchors that are not responsible for predicting any object, for
        # calculating the part of the confidence loss with zero as the target confidence. It is set to False, if a
        # predicted box overlaps any target significantly, or if a prediction is matched to a target.
        background_mask = iou_below(preds["boxes"], targets["boxes"], self.ignore_bg_threshold)
        background_mask[prior_mask] = False

        return MatchingResult(prior_mask, background_mask, target_selector, assignment_weight_sum)

    def _get_prior_mask(
        self,
        targets: TargetDict,
        image_size: Tensor,
        grid_width: int,
        grid_height: int,
        boxes_per_cell: int,
    ) -> tuple[Tensor, Tensor]:
        """Creates a mask for selecting the "center prior" anchors.

        In the first step we restrict ourselves to the grid cells whose center is inside or close enough to one or more
        targets.

        Args:
            targets: Training targets for a single image.
            image_size: Input image width and height.
            grid_width: Width of the feature grid.
            grid_height: Height of the feature grid.
            boxes_per_cell: Number of boxes that will be predicted per feature grid cell.

        Returns:
            Two masks, a ``[grid_height, grid_width, boxes_per_cell]`` mask for selecting anchors that are close and
            similar in shape to a target, and an ``[anchors, targets]`` matrix that indicates which targets are inside
            those anchors.

        """
        # This multiplier scales feature map coordinates to image coordinates.
        grid_size = torch.tensor([grid_width, grid_height], device=targets["boxes"].device)
        grid_to_image = torch.true_divide(image_size, grid_size)

        # Convert target boxes to center coordinates and dimensions.
        xywh = box_convert(targets["boxes"], in_fmt="xyxy", out_fmt="cxcywh")
        xy = xywh[:, :2]
        wh = xywh[:, 2:]

        # Create a [boxes_per_cell, targets] tensor for selecting prior shapes that are close enough to the target
        # dimensions.
        prior_wh = torch.tensor(self.prior_shapes, device=targets["boxes"].device)
        shape_selector = box_size_ratio(prior_wh, wh) < self.size_range

        # Create a [grid_cells, targets] tensor for selecting spatial locations that are inside target bounding boxes.
        centers = grid_centers(grid_size).view(-1, 2) * grid_to_image
        inside_selector = is_inside_box(centers, targets["boxes"])

        # Combine the above selectors into a [grid_cells, boxes_per_cell, targets] tensor for selecting anchors that are
        # inside target bounding boxes and close enough shape.
        inside_selector = inside_selector[:, None, :].repeat(1, boxes_per_cell, 1)
        inside_selector = torch.logical_and(inside_selector, shape_selector)

        # Set the width and height of all target bounding boxes to self.range grid cells and create a selector for
        # anchors that are now inside the boxes. If a small target has no anchors inside its bounding box, it will be
        # matched to one of these anchors, but a high penalty will ensure that anchors that are inside the bounding box
        # will be preferred.
        wh = self.spatial_range * grid_to_image * torch.ones_like(xy)
        xywh = torch.cat((xy, wh), -1)
        boxes = box_convert(xywh, in_fmt="cxcywh", out_fmt="xyxy")
        close_selector = is_inside_box(centers, boxes)

        # Create a [grid_cells, boxes_per_cell, targets] tensor for selecting anchors that are spatially close to a
        # target and whose shape is close enough to the target.
        close_selector = close_selector[:, None, :].repeat(1, boxes_per_cell, 1)
        close_selector = torch.logical_and(close_selector, shape_selector)

        mask = torch.logical_or(inside_selector, close_selector).sum(-1) > 0
        mask = mask.view(grid_height, grid_width, boxes_per_cell)
        inside_selector = inside_selector.view(grid_height, grid_width, boxes_per_cell, -1)
        return mask, inside_selector[mask]


class TALMatching:
    """Selects which anchors are used to predict each target using task-aligned matching.

    This matcher uses the same alignment idea as Ultralytics YOLOv8 TAL: class confidence and IoU are combined into a
    task-aligned score. For each target, top-k anchors by alignment score are selected from anchors whose center point
    is inside the target box.

    The assignment weight sum is the sum of normalized alignment scores for the matched foreground anchors.

    Args:
        prior_shapes: A list of all the prior box dimensions. Included for API compatibility with other matchers.
        prior_shape_idxs: List of indices to ``prior_shapes`` that this layer uses. Included for API compatibility.
        topk: Number of anchors to select per target.
        alpha: Exponent for the classification score in the alignment metric.
        beta: Exponent for IoU in the alignment metric.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the predicted box has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.

    """

    def __init__(
        self,
        prior_shapes: PRIOR_SHAPES,  # noqa: ARG002
        prior_shape_idxs: Sequence[int],  # noqa: ARG002
        topk: int = 10,
        alpha: float = 0.5,
        beta: float = 6.0,
        ignore_bg_threshold: float = 0.7,
        eps: float = 1e-9,
    ) -> None:
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.ignore_bg_threshold = ignore_bg_threshold
        self.eps = eps

    def __call__(
        self,
        preds: PREDICTIONS,
        targets: TARGETS,
        image_size: Tensor,
        input_is_normalized: bool = False,
    ) -> list[MatchingResult]:
        """Selects predictions for a batch using task-aligned matching.

        Args:
            preds: Predictions for each image.
            targets: Training targets for each image.
            image_size: Input image width and height.
            input_is_normalized: The predicted confidences and class probabilities have been normalized by logistic
                activation. This is used by the Darknet configurations of Scaled-YOLOv4.

        Returns:
            One matching result per image.

        """
        batch_size = len(preds)
        device = preds[0]["boxes"].device
        grid_height, grid_width, boxes_per_cell, _ = preds[0]["boxes"].shape
        num_preds = grid_height * grid_width * boxes_per_cell
        num_classes = preds[0]["classprobs"].shape[-1]

        # This multiplier scales feature map coordinates to image coordinates.
        grid_size = torch.tensor([grid_width, grid_height], device=device)
        grid_to_image = torch.true_divide(image_size, grid_size)

        # Create the anchor centers in image coordinates. The centers are the same for all images in the batch.
        centers = grid_centers(grid_size).view(-1, 2) * grid_to_image  # [grid_cells, 2]
        centers = centers[:, None, :].repeat(1, boxes_per_cell, 1).view(-1, 2)  # [num_preds, 2]

        # Flatten the spatial and anchor dimensions, then stack predictions so that TAL can process the whole batch.
        pred_boxes_batch = torch.stack([pred["boxes"].reshape(-1, 4) for pred in preds])
        pred_probs_batch = torch.stack([pred["classprobs"].reshape(-1, num_classes) for pred in preds])
        if not input_is_normalized:
            pred_probs_batch = pred_probs_batch.sigmoid()

        # Pad targets to the maximum target count in the batch.
        target_counts = [target["boxes"].shape[0] for target in targets]
        max_targets = max(max(target_counts, default=0), 1)
        padded_boxes = pred_boxes_batch.new_zeros(batch_size, max_targets, 4)
        target_mask = torch.zeros(batch_size, max_targets, dtype=torch.bool, device=device)
        for image_idx, target in enumerate(targets):
            num_targets = target["boxes"].shape[0]
            padded_boxes[image_idx, :num_targets] = target["boxes"]
            target_mask[image_idx, :num_targets] = True

        # Create a [batch, predictions, targets] tensor that indicates which anchor centers are inside each target box.
        # The centers and padded boxes are broadcast across the batch and prediction dimensions, respectively.
        pts = centers[None, :, None, :]  # [1, N, 1, 2]
        lt = pts[..., :2] - padded_boxes[:, None, :, :2]  # [B, N, T_max, 2]
        rb = padded_boxes[:, None, :, 2:] - pts[..., :2]  # [B, N, T_max, 2]
        inside_selector = torch.cat((lt, rb), dim=-1).amin(dim=-1) > 0.0  # [B, N, T_max]
        inside_selector = inside_selector & target_mask[:, None, :]

        # Calculate the TAL alignment metric from the target-class probabilities and IoUs. Padded targets are masked so
        # that they cannot contribute to matching.
        ious = box_iou(pred_boxes_batch, padded_boxes)  # [B, N, T_max]
        ious = ious * target_mask[:, None, :].float()

        # For each prediction-target pair, select the predicted probability of that target's class. Each image is
        # handled separately because its target labels and target count may differ.
        class_scores = pred_boxes_batch.new_zeros(batch_size, num_preds, max_targets)
        for image_idx, target in enumerate(targets):
            num_targets = target["boxes"].shape[0]
            class_scores[image_idx, :, :num_targets] = _probability_of_labels(
                pred_probs_batch[image_idx], target["labels"]
            )

        align_metric = class_scores.pow(self.alpha) * ious.pow(self.beta)

        # Run TAL matching for the whole batch.
        pred_mask_b, target_sel_b, weights_b = _tal_match(
            align_metric, ious, inside_selector, target_mask, self.topk, self.eps
        )  # [B, N], [B, N], [B, N]

        # The background IoU mask for confidence suppression has shape [B, grid_h, grid_w, boxes_per_cell].
        best_iou = ious.amax(dim=2).view(batch_size, grid_height, grid_width, boxes_per_cell)

        results: list[MatchingResult] = []
        for image_idx in range(batch_size):
            # Convert flat matched indices back to grid-row, grid-column, and anchor selectors.
            flat_idx = pred_mask_b[image_idx].nonzero().squeeze(-1)  # 0...(grid_cells * boxes_per_cell - 1)
            spatial_idx = flat_idx // boxes_per_cell  # 0...(grid_cells - 1)
            anchor_idx = flat_idx % boxes_per_cell  # 0...(boxes_per_cell - 1)
            anchor_y = spatial_idx // grid_width  # 0...(grid_height - 1)
            anchor_x = spatial_idx % grid_width  # 0...(grid_width - 1)

            # The background mask selects anchors that are not responsible for a target. Anchors are excluded when a
            # predicted box overlaps a target significantly or when TAL assigns the anchor to a target.
            background_mask = best_iou[image_idx] <= self.ignore_bg_threshold
            background_mask[anchor_y, anchor_x, anchor_idx] = False

            results.append(
                MatchingResult(
                    pred_selector=(anchor_y, anchor_x, anchor_idx),
                    background_selector=background_mask,
                    target_selector=target_sel_b[image_idx][pred_mask_b[image_idx]],
                    assignment_weight_sum=float(weights_b[image_idx].sum().item()),
                )
            )

        return results
