from collections.abc import Sequence
from dataclasses import replace
from math import log
from typing import cast

import torch
from torch import Tensor, nn
from torchvision.ops import box_convert

from .config import LossConfig, MatchingConfig
from .layers import Conv
from .loss import DistributionalDistanceLoss, YOLOLoss
from .matching_result import DenseMatchingResult
from .target_matching import (
    HighestIoUMatching,
    IoUThresholdMatching,
    MatchingFn,
    ShapeMatching,
    SimOTAMatching,
    SizeRatioMatching,
    TALMatching,
)
from .types import (
    PREDICTIONS,
    PRIOR_SHAPES,
    DetectionLossContribution,
    DistributionalDistancePredictions,
    PackedTargetDict,
    PriorShapeLevelPredictions,
)
from .utils import anchor_points_and_strides, distance_offsets_to_boxes, global_xy, grid_centers


class PriorShapeDetectionLayer(nn.Module):
    """A YOLO detection layer.

    A YOLO model has usually 1 - 3 detection layers at different resolutions. The loss is summed from all of them.

    Args:
        num_classes: Number of different classes that this layer predicts.
        prior_shapes: A list of prior box dimensions for this layer, used for scaling the predicted dimensions. The list
            should contain (width, height) tuples in the network input resolution.
        matching_func: The matching algorithm used for assigning targets to anchors, or ``None`` when a head-level
            matcher (such as task-aligned matching) assigns targets across all levels instead.
        loss_func: ``YOLOLoss`` object for calculating the losses.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
        input_is_normalized: The input is normalized by logistic activation in the previous layer. In this case the
            detection layer will not take the sigmoid of the coordinate and probability predictions, and the width and
            height are scaled up so that the maximum value is four times the anchor dimension. This is used by the
            Darknet configurations of Scaled-YOLOv4.

    """

    def __init__(
        self,
        num_classes: int,
        prior_shapes: PRIOR_SHAPES,
        matching_func: MatchingFn | None,
        loss_func: YOLOLoss,
        xy_scale: float = 1.0,
        input_is_normalized: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.prior_shapes = prior_shapes
        self.matching_func = matching_func
        self.loss_func = loss_func
        self.xy_scale = xy_scale
        self.input_is_normalized = input_is_normalized

    def forward(self, x: Tensor, image_size: Tensor) -> PriorShapeLevelPredictions:
        """Decode one feature level into a structured prediction representation.

        Maps cell-local coordinates to global coordinates in the image space, scales the bounding boxes with the
        anchors, converts the center coordinates to corner coordinates, and maps public detections to the `]0, 1[`
        probability range using sigmoid. The returned training predictions preserve unnormalized confidence and class
        logits when ``input_is_normalized`` is ``False``, so the loss can use ``binary_cross_entropy_with_logits``.

        Args:
            x: The output from the previous layer. The size of this tensor has to be
                ``[batch_size, anchors_per_cell * (num_classes + 5), height, width]``.
            image_size: Image width and height in a vector (defines the scale of the predicted and target coordinates).

        Returns:
            Decoded, flattened predictions and candidate geometry for this feature level.

        """
        batch_size, num_features, height, width = x.shape
        box_attrs = 5
        num_attrs = self.num_classes + box_attrs
        anchors_per_cell = num_features // num_attrs
        if anchors_per_cell != len(self.prior_shapes):
            raise ValueError(
                f"The model predicts {anchors_per_cell} bounding boxes per spatial location, but "
                f"{len(self.prior_shapes)} prior box dimensions are defined for this layer."
            )

        # Reshape the output to have the bounding box attributes of each grid cell on its own row.
        x = x.permute(0, 2, 3, 1)  # [batch_size, height, width, anchors_per_cell * num_attrs]
        x = x.view(batch_size, height, width, anchors_per_cell, num_attrs)

        # Take the sigmoid of the bounding box coordinates, confidence score, and class probabilities, unless the input
        # is normalized by the previous layer activation. Confidence and class losses use the unnormalized values when
        # available, so YOLOLoss can apply binary_cross_entropy_with_logits.
        norm_x = x if self.input_is_normalized else torch.sigmoid(x)
        xy = norm_x[..., :2]
        wh = x[..., 2:4]
        confidence = x[..., 4]
        norm_confidence = norm_x[..., 4]
        classprob = x[..., 5:]
        norm_classprob = norm_x[..., 5:]

        # Eliminate grid sensitivity. The previous layer should output extremely high values for the sigmoid to produce
        # x/y coordinates close to one. YOLOv4 solves this by scaling the x/y coordinates.
        xy = xy * self.xy_scale - 0.5 * (self.xy_scale - 1)

        image_xy = global_xy(xy, image_size)
        prior_shapes = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)
        image_wh = 4 * torch.square(wh) * prior_shapes if self.input_is_normalized else torch.exp(wh) * prior_shapes
        box = torch.cat((image_xy, image_wh), -1)
        box = box_convert(box, in_fmt="cxcywh", out_fmt="xyxy")
        output = torch.cat((box, norm_confidence.unsqueeze(-1), norm_classprob), -1)
        output = output.reshape(batch_size, height * width * anchors_per_cell, self.num_classes + 5)

        grid_size = torch.tensor([width, height], device=x.device)
        grid_to_image = torch.true_divide(image_size, grid_size)
        anchor_points = grid_centers(grid_size).view(-1, 2) * grid_to_image
        anchor_points = anchor_points[:, None, :].expand(-1, anchors_per_cell, -1).reshape(-1, 2)

        return PriorShapeLevelPredictions(
            detections=output,
            boxes=box.reshape(batch_size, -1, 4),
            confidences=confidence.reshape(batch_size, -1),
            classprobs=classprob.reshape(batch_size, -1, self.num_classes),
            anchor_points=anchor_points,
            spatial_shape=(height, width, anchors_per_cell),
        )


class PriorShapeDetectionHead(nn.Module):
    """Detection head for native YOLO architectures with multiple feature levels.

    Owns all detection layers and applies the appropriate assignment strategy. Task-aligned matching operates on the
    whole head: a single matcher is passed as ``matching_func`` and assigns targets across the concatenated candidates
    of every level. All other matchers are per-level and live in the individual detection layers, whose
    ``matching_func`` is used to match each level independently with its own prior shapes and grid geometry.

    This class bundles what would otherwise be separate per-level ``PriorShapeDetectionLayer`` modules plus an external
    criterion, removing the need for a separate ``DetectionCriterion`` object in the network.

    Args:
        layers: The per-level detection layers.
        matching_func: A head-level matcher that assigns targets across all levels at once. When ``None``, each layer
            uses its own ``matching_func`` instead.

    """

    def __init__(self, layers: Sequence[PriorShapeDetectionLayer], matching_func: TALMatching | None = None) -> None:
        super().__init__()
        if not layers:
            raise ValueError("At least one detection layer is required.")
        self.layers = cast(Sequence[PriorShapeDetectionLayer], nn.ModuleList(layers))
        self.matching_func = matching_func

    def forward(
        self,
        features: Sequence[Tensor],
        image_size: Tensor,
        targets: PackedTargetDict | None,
    ) -> tuple[list[Tensor], list[DetectionLossContribution]]:
        """Decodes all feature levels and computes detection losses if targets are provided.

        Args:
            features: Feature tensors for each detection level, one per layer.
            image_size: Image width and height.
            targets: Training targets, or ``None`` during inference.

        Returns:
            Decoded detections and loss contributions (empty list during inference).

        """
        levels = [layer(feat, image_size) for layer, feat in zip(self.layers, features, strict=True)]
        detections = [level.detections for level in levels]

        if targets is None:
            return detections, []

        first = self.layers[0]
        if self.matching_func is not None:
            predictions: PREDICTIONS = [
                {
                    "boxes": torch.cat([level.boxes[image_idx] for level in levels], dim=0),
                    "confidences": torch.cat([level.confidences[image_idx] for level in levels], dim=0),
                    "classprobs": torch.cat([level.classprobs[image_idx] for level in levels], dim=0),
                }
                for image_idx in range(levels[0].boxes.shape[0])
            ]
            anchor_points = torch.cat([level.anchor_points for level in levels], dim=0)
            matching_result = self.matching_func(
                preds=predictions,
                targets=targets,
                anchor_points=anchor_points,
                input_is_normalized=first.input_is_normalized,
            )
            return detections, [first.loss_func(matching_result, predictions, first.input_is_normalized, image_size)]

        losses = []
        for layer, level in zip(self.layers, levels, strict=True):
            assert layer.matching_func is not None, (
                "A per-level matcher is required when the head has no head-level matcher."
            )
            preds = level.as_grid()
            matching_result = layer.matching_func(preds, targets, image_size, layer.input_is_normalized)
            losses.append(layer.loss_func(matching_result, preds, layer.input_is_normalized, image_size))
        return detections, losses


class PriorShapeDetectionHeadWithAux(nn.Module):
    """Multi-level detection head combining lead and auxiliary layers for deep supervision.

    Each lead layer is matched to targets to determine assignment; the corresponding auxiliary layer uses the same
    lead predictions for matching but computes its own prediction losses, scaled down by ``aux_weight``. This pattern
    is used by YOLOv7.

    Args:
        layers: The per-level lead detection layers.
        aux_layers: The per-level auxiliary detection layers, ordered like ``layers``.
        aux_weight: Weight for the loss from the auxiliary head.

    """

    def __init__(
        self,
        layers: Sequence[PriorShapeDetectionLayer],
        aux_layers: Sequence[PriorShapeDetectionLayer],
        aux_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("At least one detection layer is required.")
        if len(layers) != len(aux_layers):
            raise ValueError("Lead and auxiliary heads require the same number of detection layers.")
        self.layers = cast(Sequence[PriorShapeDetectionLayer], nn.ModuleList(layers))
        self.aux_layers = cast(Sequence[PriorShapeDetectionLayer], nn.ModuleList(aux_layers))
        self.aux_weight = aux_weight

    def forward(
        self,
        features: Sequence[Tensor],
        aux_features: Sequence[Tensor],
        image_size: Tensor,
        targets: PackedTargetDict | None,
    ) -> tuple[list[Tensor], list[DetectionLossContribution]]:
        """Decodes all lead feature levels and computes lead and auxiliary losses if targets are provided.

        If ``targets`` is given, each lead layer determines which predictions are matched to targets and the
        corresponding auxiliary layer uses the same lead predictions for its assignment.

        Args:
            features: Feature tensors for the lead detection layers.
            aux_features: Feature tensors for the auxiliary detection layers, ordered like ``features``.
            image_size: Image width and height.
            targets: Training targets, or ``None`` during inference.

        Returns:
            Decoded lead detections and lead/auxiliary loss contributions (empty list during inference).

        """
        levels = [layer(feat, image_size) for layer, feat in zip(self.layers, features, strict=True)]
        detections = [level.detections for level in levels]

        if targets is None:
            return detections, []

        losses = []
        for layer, aux_layer, level, aux_feature in zip(
            self.layers, self.aux_layers, levels, aux_features, strict=True
        ):
            lead_matching_func = layer.matching_func
            aux_matching_func = aux_layer.matching_func
            assert lead_matching_func is not None, "Auxiliary detection heads require per-level matchers."
            assert aux_matching_func is not None, "Auxiliary detection heads require per-level matchers."
            preds = level.as_grid()

            # Match lead head predictions to targets and calculate losses from lead head outputs.
            matching_result = lead_matching_func(preds, targets, image_size, layer.input_is_normalized)
            losses.append(layer.loss_func(matching_result, preds, layer.input_is_normalized, image_size))

            # Match lead head predictions to targets and calculate losses from auxiliary head outputs.
            aux_level = aux_layer(aux_feature, image_size)
            aux_preds = aux_level.as_grid()
            aux_matching_result = aux_matching_func(preds, targets, image_size, aux_layer.input_is_normalized)
            aux_loss = aux_layer.loss_func(aux_matching_result, aux_preds, aux_layer.input_is_normalized, image_size)
            losses.append(aux_loss.scaled(self.aux_weight))
        return detections, losses


class DFLExpectation(nn.Module):
    """Decode distance-bin logits into expected point-to-box distances."""

    def __init__(self, num_dfl_bins: int = 16) -> None:
        super().__init__()
        if num_dfl_bins < 2:
            raise ValueError("DFL expectation requires at least two distance bins.")
        self.num_dfl_bins = num_dfl_bins
        self.register_buffer("projection", torch.arange(num_dfl_bins, dtype=torch.float32), persistent=False)

    def forward(self, logits: Tensor) -> Tensor:
        """Decode logits shaped ``[..., 4 * B]`` to distances shaped ``[..., 4]``."""
        if logits.shape[-1] != 4 * self.num_dfl_bins:
            raise ValueError(f"Expected {4 * self.num_dfl_bins} DFL logits, got {logits.shape[-1]}.")
        probabilities = logits.reshape(*logits.shape[:-1], 4, self.num_dfl_bins).softmax(-1)
        projection = self.get_buffer("projection").to(device=probabilities.device, dtype=probabilities.dtype)
        return (probabilities * projection).sum(-1)


class DistributionalDistanceDetectionHead(nn.Module):
    """Decoupled detection head with distributional distance regression.

    The head predicts one anchor per feature-map cell. Box coordinates are represented as distributions for the left,
    top, right, and bottom distances from each anchor point. A separate branch predicts classes without an objectness
    channel.

    Args:
        input_channels: Number of channels in each input feature level, ordered from highest to lowest resolution.
        num_classes: Number of classes predicted by the classification branches.
        num_dfl_bins: Number of bins in each side's distance distribution.
        strides: Pixel-space stride of each feature level, ordered like ``input_channels``. These are fixed by the
            network downsampling ratios and are used to initialize the output biases.
        matching_func: Task-aligned matcher that assigns targets to anchor points. Defaults to a :class:`TALMatching`
            with default hyperparameters.
        loss_func: Criterion that computes the overlap, classification, and DFL losses. Defaults to a
            :class:`DistributionalDistanceLoss` with ``num_dfl_bins`` bins.

    """

    def __init__(
        self,
        input_channels: Sequence[int],
        num_classes: int,
        num_dfl_bins: int = 16,
        strides: Sequence[float] = (8.0, 16.0, 32.0),
        matching_func: TALMatching | None = None,
        loss_func: DistributionalDistanceLoss | None = None,
    ) -> None:
        super().__init__()
        if len(input_channels) != 3:
            raise ValueError(
                f"Distributional-distance detection requires exactly three feature levels, got {len(input_channels)}."
            )
        if num_classes < 1:
            raise ValueError("The detection head must predict at least one class.")
        if num_dfl_bins < 2:
            raise ValueError("The detection head requires at least two DFL bins.")

        self.num_classes = num_classes
        self.num_dfl_bins = num_dfl_bins
        self.matching_func = matching_func or TALMatching()
        self.loss_func = loss_func or DistributionalDistanceLoss(num_dfl_bins=num_dfl_bins)
        first_level_channels = input_channels[0]
        box_hidden_channels = max(16, first_level_channels // 4, 4 * num_dfl_bins)
        class_hidden_channels = max(first_level_channels, min(num_classes, 100))

        self.box_branches = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(channels, box_hidden_channels, kernel_size=3),
                    Conv(box_hidden_channels, box_hidden_channels, kernel_size=3),
                    nn.Conv2d(box_hidden_channels, 4 * num_dfl_bins, kernel_size=1),
                )
                for channels in input_channels
            ]
        )
        self.class_branches = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(channels, class_hidden_channels, kernel_size=3),
                    Conv(class_hidden_channels, class_hidden_channels, kernel_size=3),
                    nn.Conv2d(class_hidden_channels, num_classes, kernel_size=1),
                )
                for channels in input_channels
            ]
        )
        self.dfl_expectation = DFLExpectation(num_dfl_bins)
        self._geometry_cache_key: tuple | None = None
        self._geometry_cache: tuple[Tensor, Tensor] | None = None

        self.initialize_output_biases(strides)

    def initialize_output_biases(self, strides: Sequence[float]) -> None:
        """Initialize final prediction biases.

        Args:
            strides: One pixel-space stride value for each feature level, ordered like ``input_channels``.

        """
        if len(strides) != len(self.box_branches):
            raise ValueError(f"Expected {len(self.box_branches)} stride values, got {len(strides)}.")
        for stride, box_branch, class_branch in zip(strides, self.box_branches, self.class_branches, strict=True):
            box_output = cast(nn.Conv2d, cast(nn.Sequential, box_branch)[-1])
            class_output = cast(nn.Conv2d, cast(nn.Sequential, class_branch)[-1])
            assert box_output.bias is not None
            assert class_output.bias is not None
            nn.init.constant_(box_output.bias, 1.0)
            nn.init.constant_(class_output.bias, log(5 / self.num_classes / (640 / stride) ** 2))

    def forward(
        self,
        features: Sequence[Tensor],
        image_size: Tensor,
        targets: PackedTargetDict | None = None,
    ) -> tuple[list[Tensor], list[DetectionLossContribution]]:
        """Decode feature levels and compute detection loss if targets are provided.

        Args:
            features: Feature tensors for each detection level, ordered from highest to lowest resolution.
            image_size: Image width and height in pixels as an `(x, y)` tensor.
            targets: Training targets, or ``None`` during inference.

        Returns:
            Decoded detections and loss contributions (empty list during inference).

        """
        predictions = self.predict(features, image_size)
        if targets is None:
            return [predictions.detections], []

        batch_size, num_preds, _ = predictions.class_logits.shape
        matcher_predictions: PREDICTIONS = [
            {
                "boxes": predictions.pixel_boxes[image_idx],
                "confidences": predictions.pixel_boxes.new_ones(num_preds),
                "classprobs": predictions.class_logits[image_idx],
            }
            for image_idx in range(batch_size)
        ]
        matching_result = self.matching_func(
            matcher_predictions,
            targets,
            anchor_points=predictions.anchor_points * predictions.strides,
            input_is_normalized=False,
        )
        assert isinstance(matching_result, DenseMatchingResult)
        return [predictions.detections], [self.loss_func(predictions, matching_result)]

    def predict(self, features: Sequence[Tensor], image_size: Tensor) -> DistributionalDistancePredictions:
        """Decode feature levels into public detections and raw training logits.

        Args:
            features: Feature tensors for each detection level, ordered from highest to lowest resolution.
            image_size: Image width and height in pixels as an `(x, y)` tensor.

        Returns:
            Public detections, raw logits, decoded boxes, anchor points, and strides.

        """
        if len(features) != len(self.box_branches):
            raise ValueError(f"Expected {len(self.box_branches)} feature levels, got {len(features)}.")

        prediction_levels = [
            torch.cat((box_branch(feature), class_branch(feature)), dim=1)
            for feature, box_branch, class_branch in zip(features, self.box_branches, self.class_branches, strict=True)
        ]
        flattened_levels = [
            level.permute(0, 2, 3, 1).reshape(level.shape[0], -1, level.shape[1]) for level in prediction_levels
        ]
        flattened = torch.cat(flattened_levels, dim=1)
        dfl_logits = flattened[..., : 4 * self.num_dfl_bins]
        class_logits = flattened[..., 4 * self.num_dfl_bins :]

        anchor_points, strides = self._geometry(features, image_size)
        distances = self.dfl_expectation(dfl_logits)
        grid_boxes = distance_offsets_to_boxes(distances, anchor_points)
        pixel_boxes = grid_boxes * strides
        confidence = torch.ones((*pixel_boxes.shape[:2], 1), dtype=pixel_boxes.dtype, device=pixel_boxes.device)
        detections = torch.cat((pixel_boxes, confidence, class_logits.sigmoid()), dim=-1)

        return DistributionalDistancePredictions(
            detections=detections,
            pixel_boxes=pixel_boxes,
            grid_boxes=grid_boxes,
            dfl_logits=dfl_logits,
            class_logits=class_logits,
            anchor_points=anchor_points,
            strides=strides,
        )

    def _geometry(self, features: Sequence[Tensor], image_size: Tensor) -> tuple[Tensor, Tensor]:
        """Return anchor points and strides for the current feature geometry.

        Args:
            features: Feature tensors for each detection level.
            image_size: Image width and height in pixels as an `(x, y)` tensor.

        Returns:
            Anchor points in grid coordinates and one stride value per point.

        """
        if torch.compiler.is_compiling():
            return anchor_points_and_strides(features, image_size)

        # The strides depend on ``image_size`` (stride = image_size / grid), so the image size is part of the key.
        # Otherwise multi-resolution training that reuses identical feature-map shapes for different input sizes would
        # return stale strides.
        key = (
            tuple((feature.shape[-2], feature.shape[-1]) for feature in features),
            tuple(image_size.flatten().tolist()),
            features[0].dtype,
            features[0].device,
        )
        if key != self._geometry_cache_key:
            self._geometry_cache = anchor_points_and_strides(features, image_size)
            self._geometry_cache_key = key
        assert self._geometry_cache is not None
        return self._geometry_cache


def create_prior_shape_detection_layer(
    prior_shapes: PRIOR_SHAPES,
    prior_shape_idxs: Sequence[int],
    num_classes: int,
    matching: MatchingConfig | None = None,
    loss: LossConfig | None = None,
    xy_scale: float = 1.0,
    input_is_normalized: bool = False,
) -> PriorShapeDetectionLayer:
    """Creates a detection layer module and the required loss function and target matching objects.

    Task-aligned matching ("tal") is applied by the :class:`PriorShapeDetectionHead` across all levels, so the returned
    layer has ``matching_func`` set to ``None`` in that case; :func:`create_prior_shape_detection_head` builds the
    head-level matcher.

    Args:
        prior_shapes: A list of all the prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        num_classes: Number of different classes that this layer predicts.
        matching: Configuration that controls how targets are assigned to anchors.
        loss: Configuration that controls how the detection losses are computed.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
        input_is_normalized: The input is normalized by logistic activation in the previous layer. In this case the
            detection layer will not take the sigmoid of the coordinate and probability predictions, and the width and
            height are scaled up so that the maximum value is four times the anchor dimension. This is used by the
            Darknet configurations of Scaled-YOLOv4.

    """
    matching = (matching or MatchingConfig()).with_defaults()
    loss = (loss or LossConfig()).with_defaults()
    assert matching.ignore_bg_threshold is not None
    assert loss.overlap_func is not None
    assert loss.overlap_multiplier is not None
    assert loss.confidence_multiplier is not None
    assert loss.class_multiplier is not None

    matching_func: ShapeMatching | SimOTAMatching | None
    if matching.algorithm == "tal":
        # Task-aligned matching assigns targets across all levels, so it is created by the head, not the layer.
        matching_func = None
    elif matching.algorithm == "simota":
        cost_func = YOLOLoss(
            loss.overlap_func,
            None,
            None,
            loss.overlap_multiplier,
            loss.confidence_multiplier,
            loss.class_multiplier,
        )
        matching_func = SimOTAMatching(
            prior_shapes, prior_shape_idxs, cost_func, matching.spatial_range, matching.size_range
        )
    elif matching.algorithm == "size":
        if matching.threshold is None:
            raise ValueError("A matching threshold is required with size ratio matching.")
        matching_func = SizeRatioMatching(
            prior_shapes, prior_shape_idxs, matching.threshold, matching.ignore_bg_threshold
        )
    elif matching.algorithm == "iou":
        if matching.threshold is None:
            raise ValueError("A matching threshold is required with IoU threshold matching.")
        matching_func = IoUThresholdMatching(
            prior_shapes, prior_shape_idxs, matching.threshold, matching.ignore_bg_threshold
        )
    elif matching.algorithm == "maxiou" or matching.algorithm is None:
        matching_func = HighestIoUMatching(prior_shapes, prior_shape_idxs, matching.ignore_bg_threshold)
    else:
        raise ValueError(f"Matching algorithm `{matching.algorithm}´ is unknown.")

    loss_func = YOLOLoss(
        loss.overlap_func,
        loss.predict_overlap,
        loss.label_smoothing,
        loss.overlap_multiplier,
        loss.confidence_multiplier,
        loss.class_multiplier,
    )
    layer_shapes = [prior_shapes[i] for i in prior_shape_idxs]
    return PriorShapeDetectionLayer(
        num_classes=num_classes,
        prior_shapes=layer_shapes,
        matching_func=matching_func,
        loss_func=loss_func,
        xy_scale=xy_scale,
        input_is_normalized=input_is_normalized,
    )


def create_prior_shape_detection_head(
    prior_shapes: PRIOR_SHAPES,
    prior_shape_idxs_per_level: Sequence[Sequence[int]],
    num_classes: int,
    matching: MatchingConfig | None = None,
    loss: LossConfig | None = None,
    xy_scale: float = 1.0,
    input_is_normalized: bool = False,
) -> PriorShapeDetectionHead:
    """Creates a multi-level detection head with the appropriate matching strategy.

    Each feature level gets a :class:`PriorShapeDetectionLayer`. For task-aligned matching, a single
    :class:`TALMatching` is created and given to the head, which assigns targets across all levels at once; the
    per-level layers then carry no matcher. For all other algorithms, each layer carries its own per-level matcher.

    Args:
        prior_shapes: A list of all the prior box dimensions in the network input resolution.
        prior_shape_idxs_per_level: For each feature level, the indices into ``prior_shapes`` that the level uses.
        num_classes: Number of different classes that the head predicts.
        matching: Configuration that controls how targets are assigned to anchors.
        loss: Configuration that controls how the detection losses are computed.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor.
        input_is_normalized: Whether the previous layer normalizes its output by a logistic activation.

    Returns:
        A detection head that owns the per-level layers and the matching strategy.

    """
    matching = matching or MatchingConfig()
    layers = [
        create_prior_shape_detection_layer(
            prior_shapes=prior_shapes,
            prior_shape_idxs=list(prior_shape_idxs),
            num_classes=num_classes,
            matching=matching,
            loss=loss,
            xy_scale=xy_scale,
            input_is_normalized=input_is_normalized,
        )
        for prior_shape_idxs in prior_shape_idxs_per_level
    ]

    matching_func = None
    if matching.algorithm == "tal":
        resolved = matching.with_defaults()
        assert resolved.ignore_bg_threshold is not None
        matching_func = TALMatching(
            topk=matching.tal_topk,
            alpha=matching.tal_alpha,
            beta=matching.tal_beta,
            ignore_bg_threshold=resolved.ignore_bg_threshold,
        )
    return PriorShapeDetectionHead(layers, matching_func)


def create_prior_shape_detection_head_with_aux(
    prior_shapes: PRIOR_SHAPES,
    prior_shape_idxs_per_level: Sequence[Sequence[int]],
    num_classes: int,
    matching: MatchingConfig | None = None,
    loss: LossConfig | None = None,
    aux_spatial_range: float = 3.0,
    aux_weight: float = 0.25,
    xy_scale: float = 1.0,
    input_is_normalized: bool = False,
) -> PriorShapeDetectionHeadWithAux:
    """Creates a multi-level detection head with lead and auxiliary layers.

    Args:
        prior_shapes: A list of all prior box dimensions in the network input resolution.
        prior_shape_idxs_per_level: For each feature level, the indices into ``prior_shapes`` used by that level.
        num_classes: Number of classes predicted by the head.
        matching: Configuration that controls target assignment for the lead layers.
        loss: Configuration that controls detection losses.
        aux_spatial_range: Spatial candidate range used by the auxiliary matcher.
        aux_weight: Weight for auxiliary loss contributions.
        xy_scale: Scale factor used to reduce grid sensitivity.
        input_is_normalized: Whether previous layers normalize their outputs with logistic activation.

    Returns:
        A multi-level detection head containing parallel lead and auxiliary layers.

    """
    matching = matching or MatchingConfig()

    def create_layers(layer_matching: MatchingConfig) -> list[PriorShapeDetectionLayer]:
        return [
            create_prior_shape_detection_layer(
                prior_shapes=prior_shapes,
                prior_shape_idxs=prior_shape_idxs,
                num_classes=num_classes,
                matching=layer_matching,
                loss=loss,
                xy_scale=xy_scale,
                input_is_normalized=input_is_normalized,
            )
            for prior_shape_idxs in prior_shape_idxs_per_level
        ]

    layers = create_layers(matching)
    aux_layers = create_layers(replace(matching, spatial_range=aux_spatial_range))
    return PriorShapeDetectionHeadWithAux(layers, aux_layers, aux_weight)


def create_distributional_distance_detection_head(
    input_channels: Sequence[int],
    num_classes: int,
    loss: LossConfig | None = None,
) -> DistributionalDistanceDetectionHead:
    """Creates a distributional-distance detection head with its matcher and loss function.

    Args:
        input_channels: Number of channels in each input feature level, ordered from highest to lowest resolution.
        num_classes: Number of classes predicted by the classification branches.
        loss: Configuration that controls the distributional-distance detection losses.

    Returns:
        A detection head that owns task-aligned matching and distributional-distance loss calculation.

    """
    loss = (loss or LossConfig()).with_defaults()
    assert loss.overlap_multiplier is not None
    assert loss.class_multiplier is not None
    assert loss.dfl_multiplier is not None
    assert loss.num_dfl_bins is not None
    loss_func = DistributionalDistanceLoss(
        overlap_func=loss.overlap_func or "ciou",
        label_smoothing=loss.label_smoothing,
        overlap_multiplier=loss.overlap_multiplier,
        class_multiplier=loss.class_multiplier,
        dfl_multiplier=loss.dfl_multiplier,
        num_dfl_bins=loss.num_dfl_bins,
    )
    return DistributionalDistanceDetectionHead(
        input_channels,
        num_classes=num_classes,
        num_dfl_bins=loss.num_dfl_bins,
        loss_func=loss_func,
    )
