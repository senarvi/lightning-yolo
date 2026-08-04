from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torchvision.ops import box_convert, complete_box_iou

from .batching import split_targets
from .loss import YOLOLoss
from .matching_result import DenseMatchingResult, ImageMatch, MatchingResult, SparseMatchingResult
from .types import PREDICTIONS, PRIOR_SHAPES, PackedTargetDict, PredictionDict, TargetDict
from .utils import aligned_iou, box_size_ratio, grid_centers, iou_below, is_inside_box

# A matching function takes batched predictions and targets, image size, and a boolean indicating whether the
# probabilities are normalized, and returns the assignment of predictions to targets for the whole batch.
MatchingFn = Callable[[PREDICTIONS, PackedTargetDict, Tensor, bool], MatchingResult]


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
        targets: PackedTargetDict,
        image_size: Tensor,
        input_is_normalized: bool = False,  # noqa: ARG002
    ) -> MatchingResult:
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
            Per-image foreground assignments for the whole batch.

        """
        image_targets = split_targets(targets)
        images = [
            self._match_image(image_preds, image_targets, image_size)
            for image_preds, image_targets in zip(preds, image_targets, strict=True)
        ]
        return SparseMatchingResult(images)

    def _match_image(self, preds: PredictionDict, targets: TargetDict, image_size: Tensor) -> ImageMatch:
        height, width, boxes_per_cell = preds["boxes"].shape[:3]
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

        # Flatten the grid selectors into anchor indices, ordered to match the selected targets.
        foreground = (cell_j * width + cell_i) * boxes_per_cell + anchor_selector

        return ImageMatch(
            foreground=foreground,
            background=background_mask.reshape(-1),
            target_boxes=targets["boxes"][target_selector],
            target_labels=targets["labels"][target_selector],
        )

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

    The rule, originally used by Ultralytics YOLOv5, compares target dimensions with anchor dimensions and assigns
    anchors whose largest width/height ratio is below the configured threshold.

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
        align_metric: ``[batch_size, N, max_targets]`` alignment scores.
        ious: ``[batch_size, N, max_targets]`` IoU values.
        inside_selector: ``[batch_size, N, max_targets]`` bool — True when an anchor centre is inside the target box.
        target_mask: ``[batch_size, max_targets]`` bool — True for real targets, False for padding.
        topk: Maximum number of anchors to select per target.

    Returns:
        pred_mask ``[batch_size, N]``, target_selector ``[batch_size, N]`` (meaningful only where pred_mask is True),
        and assignment_weights ``[batch_size, N]`` used for loss normalisation.

    """
    batch_size, num_preds, max_targets = align_metric.shape

    # Keep masked alignment metrics at zero, then filter the selected top-k indices below.
    masked_metric = align_metric.clone()
    masked_metric[~inside_selector] = 0.0
    masked_metric[~target_mask[:, None, :].expand(batch_size, num_preds, max_targets)] = 0.0

    # For each target, select top-k anchors by the alignment metric among anchors that are inside the target box.
    masked_t = masked_metric.permute(0, 2, 1)  # [batch_size, max_targets, N]
    k = min(topk, num_preds)
    _, topk_indices = torch.topk(masked_t, k=k, dim=2)  # [batch_size, max_targets, k]

    # This gather is what makes zeroing (rather than setting to -1) the masked metrics above safe: top-k may pick
    # outside/padded anchors when fewer than k anchors are inside a target (or the target is padding), so re-check
    # ``inside_selector`` here and drop those selections before they enter the matching matrix.
    valid = torch.gather(inside_selector.permute(0, 2, 1), 2, topk_indices)

    # Scatter topk selections back into the [batch_size, N, max_targets] matching matrix.
    # topk_indices: [batch_size, max_targets, k] → transpose to [batch_size, k, max_targets] for scatter on dim 1.
    idx = topk_indices.permute(0, 2, 1)  # [batch_size, k, max_targets]
    val = valid.permute(0, 2, 1)  # [batch_size, k, max_targets]
    matching_matrix = torch.zeros_like(align_metric, dtype=torch.bool)
    matching_matrix.scatter_(1, idx, val)

    # If there is more than one match for some prediction, match it with the target that has the highest IoU.
    multiple = matching_matrix.sum(dim=2) > 1  # [batch_size, N]
    best_targets = ious.argmax(dim=2, keepdim=True)
    best_matches = torch.zeros_like(matching_matrix).scatter_(2, best_targets, True)
    matching_matrix = torch.where(multiple.unsqueeze(-1), best_matches, matching_matrix)

    # For those predictions that were matched, get the index of the target.
    pred_mask = matching_matrix.any(dim=2)  # [batch_size, N]
    target_selector = matching_matrix.int().argmax(dim=2)  # [batch_size, N]

    # Normalize each matched alignment score by the best matched alignment score for the same target, then scale it by
    # that target's best matched IoU.
    matched_metric = align_metric * matching_matrix.float()
    best_align = matched_metric.amax(dim=1, keepdim=True)  # [batch_size, 1, max_targets]
    best_iou = (ious * matching_matrix.float()).amax(dim=1, keepdim=True)  # [batch_size, 1, max_targets]
    assignment_weights = (matched_metric * best_iou / (best_align + eps)).amax(dim=2)  # [batch_size, N]

    return pred_mask, target_selector, assignment_weights


def _probability_of_labels(pred_probs: Tensor, target_labels: Tensor) -> Tensor:
    """Computes a ``[predictions, targets]`` matrix of probabilities predicted for the ground-truth labels.

    The returned matrix is used as the class scores by TAL. For single-label targets, returns a matrix of the predicted
    probabilities for the target class. For multi-label targets, each prediction/target pair uses the sum of the
    predicted probabilities among the classes assigned to that target. TAL performs the top-k selection independently
    per target, so this is equivalent to using the average predicted probability.

    Args:
        pred_probs: Predicted class probabilities in a matrix shaped ``[predictions, classes]``.
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

    SimOTA was introduced by YOLOX. It does one global assignment per image over the pooled candidates of every feature
    level, so dynamic-k selection ranks candidates across all levels jointly instead of one level at a time. A native
    detection head decodes the per-level grid predictions and calls this matcher once; it returns a single batched
    assignment whose foreground indices address the concatenated anchors of all levels in level order, and the head
    computes the loss with one call over the pooled predictions.

    The assignment weight sum is the number of anchors selected by dynamic-k matching after conflict resolution, with
    one unit per matched foreground anchor.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain (width, height) tuples in the
            network input resolution.
        prior_shape_idxs: For each feature level, the indices into ``prior_shapes`` that the level uses.
        loss_func: A ``YOLOLoss`` object that can be used to calculate the pairwise costs.
        spatial_range: For each target, restrict to the anchors that are within an `N × N` grid cell area centered at
            the target, where `N` is the value of this parameter.
        size_range: For each target, restrict to the anchors whose prior dimensions are not larger than the target
            dimensions multiplied by this value and not smaller than the target dimensions divided by this value.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the predicted box has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.

    """

    def __init__(
        self,
        prior_shapes: PRIOR_SHAPES,
        prior_shape_idxs: Sequence[Sequence[int]],
        loss_func: YOLOLoss,
        spatial_range: float,
        size_range: float,
        ignore_bg_threshold: float = 0.7,
    ) -> None:
        self.prior_shapes_per_level = [[prior_shapes[idx] for idx in idxs] for idxs in prior_shape_idxs]
        self.loss_func = loss_func
        self.spatial_range = spatial_range
        self.size_range = size_range
        self.ignore_bg_threshold = ignore_bg_threshold

    def __call__(
        self,
        preds: Sequence[list[PredictionDict]],
        targets: PackedTargetDict,
        image_size: Tensor,
        input_is_normalized: bool = False,
    ) -> SparseMatchingResult:
        """Assigns targets across all feature levels for every image in the batch.

        Args:
            preds: For each detection level, the per-image grid predictions returned by
                :meth:`PriorShapeLevelPredictions.as_grid`.
            targets: Packed training targets for the whole batch.
            image_size: Input image width and height.
            input_is_normalized: The predicted confidences and class probabilities have been normalized by logistic
                activation. This is used by the Darknet configurations of Scaled-YOLOv4.

        Returns:
            One assignment per image, whose foreground indices address the concatenated anchors of all levels.

        """
        image_targets = split_targets(targets)
        images = [
            self._match_image([level[image_idx] for level in preds], targets_i, image_size, input_is_normalized)
            for image_idx, targets_i in enumerate(image_targets)
        ]
        return SparseMatchingResult(images)

    def _match_image(
        self,
        level_preds: Sequence[PredictionDict],
        targets: TargetDict,
        image_size: Tensor,
        input_is_normalized: bool,
    ) -> ImageMatch:
        candidate_boxes: list[Tensor] = []
        candidate_confidences: list[Tensor] = []
        candidate_classprobs: list[Tensor] = []
        inside_matrices: list[Tensor] = []
        candidate_indices: list[Tensor] = []
        level_boxes: list[Tensor] = []
        offset = 0
        for prior_shapes, preds in zip(self.prior_shapes_per_level, level_preds, strict=True):
            height, width, boxes_per_cell, _ = preds["boxes"].shape
            prior_mask, anchor_inside_target = self._get_prior_mask(
                prior_shapes, targets, image_size, width, height, boxes_per_cell
            )
            candidate_boxes.append(preds["boxes"][prior_mask])
            candidate_confidences.append(preds["confidences"][prior_mask])
            candidate_classprobs.append(preds["classprobs"][prior_mask])
            inside_matrices.append(anchor_inside_target)
            candidate_indices.append(prior_mask.reshape(-1).nonzero(as_tuple=True)[0] + offset)
            level_boxes.append(preds["boxes"].reshape(-1, 4))
            offset += preds["confidences"].numel()

        candidate_preds: PredictionDict = {
            "boxes": torch.cat(candidate_boxes),
            "confidences": torch.cat(candidate_confidences),
            "classprobs": torch.cat(candidate_classprobs),
        }
        anchor_inside_target = torch.cat(inside_matrices)
        pooled_indices = torch.cat(candidate_indices)
        pooled_boxes = torch.cat(level_boxes)

        losses, ious = self.loss_func.pairwise_costs(candidate_preds, targets, input_is_normalized=input_is_normalized)
        costs = losses.overlap + losses.confidence + losses.classification
        costs += 100000.0 * ~anchor_inside_target
        pred_mask, target_selector = _sim_ota_match(costs, ious)

        # The matched anchors in pooled flat order match the target selector returned by SimOTA.
        foreground = pooled_indices[pred_mask]

        # Every anchor of every level is supervised as background unless it overlaps a target significantly or was
        # matched to a target.
        background = iou_below(pooled_boxes, targets["boxes"], self.ignore_bg_threshold)
        background[foreground] = False

        return ImageMatch(
            foreground=foreground,
            background=background,
            target_boxes=targets["boxes"][target_selector],
            target_labels=targets["labels"][target_selector],
        )

    def _get_prior_mask(
        self,
        prior_shapes: PRIOR_SHAPES,
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
            prior_shapes: The prior box dimensions used by this feature level.
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
        prior_wh = torch.tensor(prior_shapes, device=targets["boxes"].device)
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

    Task-aligned matching selects the top candidates across all feature levels at once, so it is global: native
    architectures prepare the concatenated predictions and anchor points in their detection head before calling this
    matcher. It is therefore not usable in the per-level path of Darknet configurations, which process one detection
    layer at a time.

    The assignment weight sum is the sum of normalized alignment scores for the matched foreground anchors.

    Args:
        topk: Number of anchors to select per target.
        alpha: Exponent for the classification score in the alignment metric.
        beta: Exponent for IoU in the alignment metric.
        overlap_func: Function that computes the pairwise ``[predictions, targets]`` overlaps used both for matching
            and for the background decision. Defaults to IoU; the distributional-distance head passes complete IoU.
        ignore_bg_threshold: A non-foreground point is supervised as background unless its best overlap with a target
            exceeds this threshold. Defaults to 1.0, meaning that every non-foreground point is supervised as
            background.
        eps: Small constant added to the denominator when normalizing the alignment weights.

    """

    def __init__(
        self,
        topk: int = 10,
        alpha: float = 0.5,
        beta: float = 6.0,
        overlap_func: Callable[[Tensor, Tensor], Tensor] = complete_box_iou,
        ignore_bg_threshold: float = 1.0,
        eps: float = 1e-9,
    ) -> None:
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.overlap_func = overlap_func
        self.ignore_bg_threshold = ignore_bg_threshold
        self.eps = eps

    def __call__(
        self,
        preds: PREDICTIONS,
        targets: PackedTargetDict,
        anchor_points: Tensor,
        input_is_normalized: bool = False,
    ) -> MatchingResult:
        """Selects predictions for a batch using task-aligned matching.

        Task-aligned matching is global, so ``anchor_points`` (the candidate centers of every feature level) must be
        given by the caller. Native architectures prepare them in their detection head. Per-level matching, as used by
        the Darknet path, is not supported.

        Args:
            preds: Predictions for each image, concatenated across all feature levels.
            targets: Training targets for each image.
            anchor_points: Candidate centers in image coordinates for every prediction.
            input_is_normalized: The predicted confidences and class probabilities have been normalized by logistic
                activation. If ``False``, class probabilities are logits and are normalized with sigmoid for TAL
                assignment scores only; the original logits remain available for loss calculation.

        Returns:
            Fixed-shape assignments for the whole batch.

        """
        batch_size = len(preds)
        device = preds[0]["boxes"].device
        num_preds = preds[0]["boxes"].numel() // 4
        num_classes = preds[0]["classprobs"].shape[-1]

        if anchor_points.shape != (num_preds, 2):
            raise ValueError(f"Expected {num_preds} anchor points, got shape {tuple(anchor_points.shape)}.")

        # Flatten the spatial and anchor dimensions, then stack predictions so that TAL can process the whole batch.
        pred_boxes = torch.stack([pred["boxes"].reshape(-1, 4) for pred in preds])
        pred_probs = torch.stack([pred["classprobs"].reshape(-1, num_classes) for pred in preds])
        if not input_is_normalized:
            pred_probs = pred_probs.sigmoid()
        # Matching is a non-differentiable assignment. Detaching the predictions keeps gradients out of the discrete
        # top-k selection and frees the large [batch_size, N, max_targets] intermediates before backward.
        pred_boxes = pred_boxes.detach()
        pred_probs = pred_probs.detach()

        # Scatter the packed targets into a [batch_size, max_targets] padded layout without a Python per-image loop.
        # Each packed row knows its image via sample_idxs; per-image counts give its position within that image.
        counts = targets["counts"]
        packed_boxes = targets["boxes"]
        packed_labels = targets["labels"]
        sample_idxs = targets["sample_idxs"].to(device)
        multilabel = packed_labels.ndim == 2

        max_targets = max(max(counts), 1)
        padded_boxes = pred_boxes.new_zeros(batch_size, max_targets, 4)
        target_mask = torch.zeros(batch_size, max_targets, dtype=torch.bool, device=device)
        if multilabel:
            padded_labels = torch.zeros(batch_size, max_targets, num_classes, dtype=packed_labels.dtype, device=device)
        else:
            padded_labels = torch.zeros(batch_size, max_targets, dtype=packed_labels.dtype, device=device)

        num_targets = packed_boxes.shape[0]
        if num_targets:
            counts_tensor = torch.as_tensor(counts, device=device)
            image_starts = counts_tensor.cumsum(0) - counts_tensor
            within_image = torch.arange(num_targets, device=device) - image_starts.repeat_interleave(counts_tensor)
            padded_boxes[sample_idxs, within_image] = packed_boxes.to(padded_boxes.dtype)
            padded_labels[sample_idxs, within_image] = packed_labels
            target_mask[sample_idxs, within_image] = True

        # Create a [batch, predictions, targets] tensor that indicates which anchor centers are inside each target box.
        # The centers and padded boxes are broadcast across the batch and prediction dimensions, respectively.
        pts = anchor_points[None, :, None, :]  # [1, N, 1, 2]
        lt = pts[..., :2] - padded_boxes[:, None, :, :2]  # [batch_size, N, max_targets, 2]
        rb = padded_boxes[:, None, :, 2:] - pts[..., :2]  # [batch_size, N, max_targets, 2]
        inside_selector = torch.cat((lt, rb), dim=-1).amin(dim=-1) > 0.0  # [batch_size, N, max_targets]
        inside_selector = inside_selector & target_mask[:, None, :]

        # Calculate the TAL alignment metric from the target-class probabilities and overlaps. Padded targets are masked
        # so that they cannot contribute to matching.
        overlaps = self.overlap_func(pred_boxes, padded_boxes)
        overlaps = torch.where(target_mask[:, None, :], overlaps, torch.zeros_like(overlaps)).clamp(min=0)

        # Gather each target's predicted class score vectorized across the batch.
        if padded_labels.ndim == 2:
            label_indices = padded_labels.clamp(max=num_classes - 1)
            class_scores = torch.gather(
                pred_probs,
                2,
                label_indices[:, None, :].expand(-1, num_preds, -1),
            )
        else:
            class_scores = torch.matmul(pred_probs, padded_labels.transpose(1, 2).to(pred_probs.dtype))

        align_metric = class_scores.pow(self.alpha) * overlaps.pow(self.beta)

        # Run TAL matching for the whole batch.
        pred_mask, target_sel, weights = _tal_match(
            align_metric, overlaps, inside_selector, target_mask, self.topk, self.eps
        )  # [batch_size, N], [batch_size, N], [batch_size, N]

        # A non-foreground point is background unless it overlaps a target more than the threshold. With a threshold of
        # 1.0 every non-foreground point becomes background, because the clamped overlaps never exceed one.
        best_iou = overlaps.amax(dim=2)
        background = (best_iou <= self.ignore_bg_threshold) & ~pred_mask
        target_boxes = torch.gather(padded_boxes, 1, target_sel.unsqueeze(-1).expand(-1, -1, 4))

        if not multilabel:
            target_labels = torch.gather(padded_labels, 1, target_sel)
        else:
            target_labels = torch.gather(padded_labels, 1, target_sel.unsqueeze(-1).expand(-1, -1, num_classes))

        return DenseMatchingResult(
            foreground=pred_mask,
            background=background,
            target_boxes=target_boxes,
            target_labels=target_labels,
            assignment_weights=weights,
        )
