from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou

from .loss import YOLOLoss
from .types import PRIOR_SHAPES, PredictionDict, TargetDict
from .utils import aligned_iou, box_size_ratio, grid_centers, iou_below, is_inside_box

# A selector for matched predictions. Matchers may return either:
# 1) a tuple of index tensors (y_idxs, x_idxs, anchor_idxs), or
# 2) a boolean mask tensor that can be used directly for indexing.
PredSelector = tuple[Tensor, Tensor, Tensor] | Tensor

# A matching function takes predictions, targets, image size, and a boolean indicating whether the probabilities are
# normalized, and returns a selector for the matched predictions, background mask, and the indices of matched targets.
MatchingFn = Callable[[PredictionDict, TargetDict, Tensor, bool], tuple[PredSelector, Tensor, Tensor]]


class ShapeMatching(ABC):
    """Selects which anchors are used to predict each target, by comparing the shape of the target box to a set of prior
    shapes.

    Most YOLO variants match targets to anchors based on prior shapes that are assigned to the anchors in the model
    configuration. The subclasses of ``ShapeMatching`` implement matching rules that compare the width and height of
    the targets to each prior shape (regardless of the location where the target is). When the model includes multiple
    detection layers, different shapes are defined for each layer. Usually there are three detection layers and three
    prior shapes per layer.

    Args:
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.

    """

    def __init__(self, ignore_bg_threshold: float = 0.7) -> None:
        self.ignore_bg_threshold = ignore_bg_threshold

    def __call__(
        self,
        preds: PredictionDict,
        targets: TargetDict,
        image_size: Tensor,
        input_is_normalized: bool = False,
    ) -> tuple[PredSelector, Tensor, Tensor]:
        """For each target, selects predictions from the same grid cell, where the center of the target box is.

        Typically there are three predictions per grid cell. Subclasses implement ``match()``, which selects the
        predictions within the grid cell.

        Args:
            preds: Predictions for a single image.
            targets: Training targets for a single image.
            image_size: Input image width and height.
            input_is_normalized: The predicted confidences and class probabilities have been normalized by logistic
                activation. This is used by the Darknet configurations of Scaled-YOLOv4.

        Returns:
            The indices of the matched predictions, background mask, and a mask for selecting the matched targets.

        """
        height, width = preds["boxes"].shape[:2]
        device = preds["boxes"].device

        # A multiplier for scaling image coordinates to feature map coordinates
        grid_size = torch.tensor([width, height], device=device)
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

        return pred_selector, background_mask, target_selector

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
        # anchor_map maps the anchor indices to anchors in this layer, or to -1 if it's not an anchor of this layer.
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
        A mask of predictions that were matched, and the indices of the matched targets. The latter contains as many
        elements as there are ``True`` values in the mask.

    """
    num_preds, num_targets = ious.shape

    matching_matrix = torch.zeros_like(costs, dtype=torch.bool)

    if ious.numel() > 0:
        # For each target, define k as the sum of the 10 highest IoUs.
        top10_iou = torch.topk(ious, min(10, num_preds), dim=0).values.sum(0)
        ks = torch.clip(top10_iou.int(), min=1)
        assert len(ks) == num_targets

        # For each target, select k predictions with the lowest cost.
        for target_idx, (target_costs, k) in enumerate(zip(costs.T, ks, strict=True)):
            pred_idx = torch.topk(target_costs, k, largest=False).indices
            matching_matrix[pred_idx, target_idx] = True

        # If there's more than one match for some prediction, match it with the best target. Now we consider all
        # targets, regardless of whether they were originally matched with the prediction or not.
        more_than_one_match = matching_matrix.sum(1) > 1
        best_targets = costs[more_than_one_match, :].argmin(1)
        matching_matrix[more_than_one_match, :] = False
        matching_matrix[more_than_one_match, best_targets] = True

    # For those predictions that were matched, get the index of the target.
    pred_mask = matching_matrix.sum(1) > 0
    target_selector = matching_matrix[pred_mask, :].int().argmax(1)
    return pred_mask, target_selector


def _tal_match(align_metric: Tensor, ious: Tensor, inside_selector: Tensor, topk: int) -> tuple[Tensor, Tensor]:
    """Implements the TAL matching rule.

    For each target, this method considers only anchors whose center point is inside the target box, ranks them using
    the TAL alignment score, and marks the ``topk`` highest-scoring anchors as matched. The ``k`` is a fixed parameter.

    Args:
        align_metric: A ``[predictions, targets]`` matrix of TAL alignment scores.
        ious: A ``[predictions, targets]`` matrix of IoUs.
        inside_selector: A ``[predictions, targets]`` boolean matrix that indicates which anchor centers are inside
            target boxes.
        topk: Number of top-scoring anchors to select per target.

    Returns:
        A mask of predictions that were matched, and the indices of the matched targets. The latter contains as many
        elements as there are ``True`` values in the mask.

    """
    matching_matrix = torch.zeros_like(ious, dtype=torch.bool, device=ious.device)

    # For each target, select top-k anchors by the alignment metric among anchors that are inside the target box.
    for target_idx in range(align_metric.shape[1]):
        valid_mask = inside_selector[:, target_idx]
        valid_indices = valid_mask.nonzero().squeeze(-1)
        if valid_indices.numel() == 0:
            continue

        target_scores = align_metric[valid_indices, target_idx]
        k = min(topk, target_scores.numel())
        topk_indices = torch.topk(target_scores, k=k, largest=True).indices
        matching_matrix[valid_indices[topk_indices], target_idx] = True

    # If there's more than one match for some prediction, match it with the best target. Now we consider all
    # targets, regardless of whether they were originally matched with the prediction or not.
    more_than_one_match = matching_matrix.sum(1) > 1
    best_targets = ious[more_than_one_match, :].argmax(1)
    matching_matrix[more_than_one_match, :] = False
    matching_matrix[more_than_one_match, best_targets] = True

    # For those predictions that were matched, get the index of the target.
    pred_mask = matching_matrix.sum(1) > 0
    target_selector = matching_matrix[pred_mask, :].int().argmax(1)
    return pred_mask, target_selector


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
        preds: PredictionDict,
        targets: TargetDict,
        image_size: Tensor,
        input_is_normalized: bool = False,
    ) -> tuple[PredSelector, Tensor, Tensor]:
        """For each target, selects predictions using the SimOTA matching rule.

        Args:
            preds: Predictions for a single image.
            targets: Training targets for a single image.
            image_size: Input image width and height.
            input_is_normalized: The predicted confidences and class probabilities have been normalized by logistic
                activation. This is used by the Darknet configurations of Scaled-YOLOv4.

        Returns:
            A mask of predictions that were matched, background mask, and the indices of the matched targets. The last
            tensor contains as many elements as there are ``True`` values in the first mask.

        """
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

        # Replace True values with the results of the actual SimOTA matching.
        prior_mask[prior_mask.nonzero(as_tuple=True)] = pred_mask

        # Background mask is used to select anchors that are not responsible for predicting any object, for
        # calculating the part of the confidence loss with zero as the target confidence. It is set to False, if a
        # predicted box overlaps any target significantly, or if a prediction is matched to a target.
        background_mask = iou_below(preds["boxes"], targets["boxes"], self.ignore_bg_threshold)
        background_mask[prior_mask] = False

        return prior_mask, background_mask, target_selector

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
        # A multiplier for scaling feature map coordinates to image coordinates
        grid_size = torch.tensor([grid_width, grid_height], device=targets["boxes"].device)
        grid_to_image = torch.true_divide(image_size, grid_size)

        # Get target center coordinates and dimensions.
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
        prior_shapes: PRIOR_SHAPES,
        prior_shape_idxs: Sequence[int],
        topk: int = 10,
        alpha: float = 0.5,
        beta: float = 6.0,
        ignore_bg_threshold: float = 0.7,
    ) -> None:
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.ignore_bg_threshold = ignore_bg_threshold

    def __call__(
        self,
        preds: PredictionDict,
        targets: TargetDict,
        image_size: Tensor,
        input_is_normalized: bool = False,
    ) -> tuple[PredSelector, Tensor, Tensor]:
        """For each target, selects predictions using task-aligned matching.

        Args:
            preds: Predictions for a single image.
            targets: Training targets for a single image.
            image_size: Input image width and height.
            input_is_normalized: The predicted confidences and class probabilities have been normalized by logistic
                activation. This is used by the Darknet configurations of Scaled-YOLOv4.

        Returns:
            The indices of the matched predictions, background mask, and the indices of matched targets.

        """
        grid_height, grid_width, boxes_per_cell, _ = preds["boxes"].shape
        num_classes = preds["classprobs"].shape[-1]

        # A multiplier for scaling feature map coordinates to image coordinates
        grid_size = torch.tensor([grid_width, grid_height], device=targets["boxes"].device)
        grid_to_image = torch.true_divide(image_size, grid_size)

        # Flatten the predictions to [grid_cells * boxes_per_cell, ...].
        pred_boxes = preds["boxes"].reshape(-1, 4)
        pred_probs = preds["classprobs"] if input_is_normalized else preds["classprobs"].sigmoid()
        pred_probs = pred_probs.reshape(-1, num_classes)

        # Create a [grid_cells * boxes_per_cell, targets] tensor for selecting spatial locations that are inside target
        # bounding boxes.
        centers = grid_centers(grid_size).view(-1, 2) * grid_to_image  # [grid_cells, 2]
        centers = centers[:, None, :].repeat(1, boxes_per_cell, 1).view(-1, 2)  # [grid_cells * boxes_per_cell, 2]
        inside_selector = is_inside_box(centers, targets["boxes"])  # [grid_cells * boxes_per_cell, targets]

        # Calculate the TAL alignment metric for anchors that are inside the target box and select the top-k anchors.
        ious = box_iou(pred_boxes, targets["boxes"])
        class_scores = _probability_of_labels(pred_probs, targets["labels"])
        align_metric = class_scores.pow(self.alpha) * ious.pow(self.beta)
        pred_mask, target_selector = _tal_match(align_metric, ious, inside_selector, self.topk)

        # Add the anchor dimension to the mask and replace True values with the results of the actual TAL matching.
        flat_idx = pred_mask.nonzero().squeeze(-1)  # 0...(grid_cells * boxes_per_cell - 1)
        spatial_idx = flat_idx // boxes_per_cell  # 0...(grid_cells - 1)
        anchor_idx = flat_idx % boxes_per_cell  # 0...(boxes_per_cell - 1)
        anchor_y = spatial_idx // grid_width  # 0...(grid_height - 1)
        anchor_x = spatial_idx % grid_width  # 0...(grid_width - 1)

        # Background mask is used to select anchors that are not responsible for predicting any object, for
        # calculating the part of the confidence loss with zero as the target confidence. It is set to False, if a
        # predicted box overlaps any target significantly, or if a prediction is matched to a target.
        background_mask = iou_below(preds["boxes"], targets["boxes"], self.ignore_bg_threshold)
        background_mask[anchor_y, anchor_x, anchor_idx] = False

        pred_selector = (anchor_y, anchor_x, anchor_idx)

        return pred_selector, background_mask, target_selector
