from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import torch
from torch import Tensor, nn
from torchvision.ops import box_convert

from .config import LossConfig, MatchingConfig
from .loss import YOLOLoss
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
    DetectionLossRecord,
    LevelPredictions,
    PackedTargetDict,
)
from .utils import global_xy, grid_centers


def _get_padding(kernel_size: int, stride: int) -> tuple[int, nn.Module]:
    """Returns the amount of padding needed by convolutional and max pooling layers.

    Determines the amount of padding needed to make the output size of the layer the input size divided by the stride.
    The first value that the function returns is the amount of padding to be added to all sides of the input matrix
    (``padding`` argument of the operation). If an uneven amount of padding is needed in different sides of the input,
    the second variable that is returned is an ``nn.ZeroPad2d`` operation that adds an additional column and row of
    padding. If the input size is not divisible by the stride, the output size will be rounded upwards.

    Args:
        kernel_size: Size of the kernel.
        stride: Stride of the operation.

    Returns:
        padding, pad_op: The amount of padding to be added to all sides of the input and an ``nn.Identity`` or
        ``nn.ZeroPad2d`` operation to add one more column and row of padding if necessary.

    """
    # The output size is generally (input_size + padding - max(kernel_size, stride)) / stride + 1 and we want to
    # make it equal to input_size / stride.
    padding, remainder = divmod(max(kernel_size, stride) - stride, 2)

    # If the kernel size is an even number, we need one cell of extra padding, on top of the padding added by MaxPool2d
    # on both sides.
    pad_op: nn.Module = nn.Identity() if remainder == 0 else nn.ZeroPad2d((0, 1, 0, 1))

    return padding, pad_op


class DetectionLayer(nn.Module):
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
        predict_confidence: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.prior_shapes = prior_shapes
        self.matching_func = matching_func
        self.loss_func = loss_func
        self.xy_scale = xy_scale
        self.input_is_normalized = input_is_normalized
        self.predict_confidence = predict_confidence

    def forward(self, x: Tensor, image_size: Tensor) -> LevelPredictions:
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
        box_attrs = 5 if self.predict_confidence else 4
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
        if self.predict_confidence:
            confidence = x[..., 4]
            norm_confidence = norm_x[..., 4]
            classprob = x[..., 5:]
            norm_classprob = norm_x[..., 5:]
        else:
            # Confidence-free heads predict no confidence channel. A constant confidence of one keeps the public output
            # layout and detection post-processing unchanged, so the detection score equals the class probability.
            classprob = x[..., 4:]
            norm_classprob = norm_x[..., 4:]
            confidence = x.new_ones(x.shape[:-1])
            norm_confidence = confidence

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

        return LevelPredictions(
            detections=output,
            boxes=box.reshape(batch_size, -1, 4),
            confidences=confidence.reshape(batch_size, -1),
            classprobs=classprob.reshape(batch_size, -1, self.num_classes),
            anchor_points=anchor_points,
            spatial_shape=(height, width, anchors_per_cell),
        )


class DetectionHead(nn.Module):
    """Detection head for native YOLO architectures with multiple feature levels.

    Owns all detection layers and applies the appropriate assignment strategy. Task-aligned matching operates on the
    whole head: a single matcher is passed as ``matching_func`` and assigns targets across the concatenated candidates
    of every level. All other matchers are per-level and live in the individual detection layers, whose
    ``matching_func`` is used to match each level independently with its own prior shapes and grid geometry.

    This class bundles what would otherwise be separate per-level ``DetectionLayer`` modules plus an external criterion,
    removing the need for a separate ``DetectionCriterion`` object in the network.

    Args:
        layers: The per-level detection layers.
        matching_func: A head-level matcher that assigns targets across all levels at once. When ``None``, each layer
            uses its own ``matching_func`` instead.

    """

    def __init__(self, layers: Sequence[DetectionLayer], matching_func: TALMatching | None = None) -> None:
        super().__init__()
        if not layers:
            raise ValueError("At least one detection layer is required.")
        self.layers = cast(Sequence[DetectionLayer], nn.ModuleList(layers))
        self.matching_func = matching_func

    def forward(
        self,
        features: Sequence[Tensor],
        image_size: Tensor,
        targets: PackedTargetDict | None,
    ) -> tuple[list[Tensor], list[DetectionLossRecord]]:
        """Decodes all feature levels and computes detection losses if targets are provided.

        Args:
            features: Feature tensors for each detection level, one per layer.
            image_size: Image width and height.
            targets: Training targets, or ``None`` during inference.

        Returns:
            Decoded detections and loss records (empty list during inference).

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
            return detections, [
                first.loss_func.matched_losses(matching_result, predictions, first.input_is_normalized, image_size)
            ]

        losses = []
        for layer, level in zip(self.layers, levels, strict=True):
            assert layer.matching_func is not None, (
                "A per-level matcher is required when the head has no head-level matcher."
            )
            preds = level.as_grid()
            matching_result = layer.matching_func(preds, targets, image_size, layer.input_is_normalized)
            losses.append(layer.loss_func.matched_losses(matching_result, preds, layer.input_is_normalized, image_size))
        return detections, losses


class DetectionHeadWithAux(nn.Module):
    """Detection head combining a lead and an auxiliary detection layer for deep supervision.

    The lead head is matched to targets to determine assignment; the auxiliary head uses the same assignments but
    computes its own prediction losses, scaled down by ``aux_weight``. This pattern is used by YOLOv7.

    Args:
        prior_shapes: A list of all the prior box dimensions in the network input resolution.
        prior_shape_idxs: Indices into ``prior_shapes`` selecting the prior shapes used by both heads.
        num_classes: Number of different classes that the heads predict.
        matching: Configuration that controls how targets are assigned to anchors for the lead head. The auxiliary head
            reuses it with ``spatial_range`` replaced by ``aux_spatial_range``.
        loss: Configuration that controls how the detection losses are computed.
        aux_spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target. This parameter specifies `N` for the auxiliary head.
        aux_weight: Weight for the loss from the auxiliary head.
        predict_confidence: Whether the heads predict a confidence (objectness) channel.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor.
        input_is_normalized: Whether the previous layer normalizes its output by a logistic activation.

    """

    def __init__(
        self,
        prior_shapes: PRIOR_SHAPES,
        prior_shape_idxs: Sequence[int],
        num_classes: int,
        matching: MatchingConfig | None = None,
        loss: LossConfig | None = None,
        aux_spatial_range: float = 3.0,
        aux_weight: float = 0.25,
        predict_confidence: bool = True,
        xy_scale: float = 1.0,
        input_is_normalized: bool = False,
    ) -> None:
        super().__init__()
        matching = matching or MatchingConfig()
        self.detection_layer = create_detection_layer(
            prior_shapes=prior_shapes,
            prior_shape_idxs=prior_shape_idxs,
            num_classes=num_classes,
            matching=matching,
            loss=loss,
            predict_confidence=predict_confidence,
            xy_scale=xy_scale,
            input_is_normalized=input_is_normalized,
        )
        self.aux_detection_layer = create_detection_layer(
            prior_shapes=prior_shapes,
            prior_shape_idxs=prior_shape_idxs,
            num_classes=num_classes,
            matching=replace(matching, spatial_range=aux_spatial_range),
            loss=loss,
            predict_confidence=predict_confidence,
            xy_scale=xy_scale,
            input_is_normalized=input_is_normalized,
        )
        self.aux_weight = aux_weight

    def forward(
        self,
        layer_input: Tensor,
        aux_input: Tensor,
        targets: PackedTargetDict | None,
        image_size: Tensor,
        detections: list[Tensor],
        losses: list[DetectionLossRecord],
    ) -> None:
        """Runs the lead and auxiliary detection layers and appends their outputs.

        If ``targets`` is given, computes losses from both heads and appends them to ``losses``. The lead head
        determines which predictions are matched to targets; the auxiliary head uses the same assignment.

        Args:
            layer_input: Input to the lead detection layer.
            aux_input: Input to the auxiliary detection layer.
            targets: List of training targets for each image.
            image_size: Width and height in a vector that defines the scale of the target coordinates.
            detections: A list where a tensor containing the detections will be appended to.
            losses: A list where a tensor containing the losses will be appended to, if ``targets`` is given.

        """
        level = self.detection_layer(layer_input, image_size)
        detections.append(level.detections)

        if targets is not None:
            lead_matching_func = self.detection_layer.matching_func
            aux_matching_func = self.aux_detection_layer.matching_func
            assert lead_matching_func is not None, "Auxiliary detection heads require per-level matchers."
            assert aux_matching_func is not None, "Auxiliary detection heads require per-level matchers."
            preds = level.as_grid()

            # Match lead head predictions to targets and calculate losses from lead head outputs.
            matching_result = lead_matching_func(preds, targets, image_size, self.detection_layer.input_is_normalized)
            losses.append(
                self.detection_layer.loss_func.matched_losses(
                    matching_result, preds, self.detection_layer.input_is_normalized, image_size
                )
            )

            # Match lead head predictions to targets and calculate losses from auxiliary head outputs.
            aux_level = self.aux_detection_layer(aux_input, image_size)
            aux_preds = aux_level.as_grid()
            aux_matching_result = aux_matching_func(
                preds, targets, image_size, self.aux_detection_layer.input_is_normalized
            )
            aux_loss = self.aux_detection_layer.loss_func.matched_losses(
                aux_matching_result, aux_preds, self.aux_detection_layer.input_is_normalized, image_size
            )
            losses.append(aux_loss.scaled(self.aux_weight))


class Conv(nn.Module):
    """A convolutional layer with optional layer normalization and activation.

    If ``padding`` is ``None``, the module tries to add padding so much that the output size will be the input size
    divided by the stride. If the input size is not divisible by the stride, the output size will be rounded upwards.

    Args:
        in_channels: Number of input channels that the layer expects.
        out_channels: Number of output channels that the convolution produces.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to all four sides of the input.
        bias: If ``True``, adds a learnable bias to the output.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
        activation: str | None = "silu",
        norm: str | None = "batchnorm",
    ):
        super().__init__()

        if padding is None:
            padding, self.pad = _get_padding(kernel_size, stride)
        else:
            self.pad = nn.Identity()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm = _create_normalization_module(norm, out_channels)
        self.act = _create_activation_module(activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class MaxPool(nn.Module):
    """A max pooling layer with padding.

    The module tries to add padding so much that the output size will be the input size divided by the stride. If the
    input size is not divisible by the stride, the output size will be rounded upwards.

    """

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        padding, self.pad = _get_padding(kernel_size, stride)
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        return self.maxpool(x)


class RouteLayer(nn.Module):
    """A routing layer concatenates the output (or part of it) from given layers.

    Args:
        source_layers: Indices of the layers whose output will be concatenated.
        num_chunks: Layer outputs will be split into this number of chunks.
        chunk_idx: Only the chunks with this index will be concatenated.

    """

    def __init__(self, source_layers: list[int], num_chunks: int, chunk_idx: int) -> None:
        super().__init__()
        self.source_layers = source_layers
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx

    def forward(self, outputs: list[Tensor]) -> Tensor:
        chunks = [torch.chunk(outputs[layer], self.num_chunks, dim=1)[self.chunk_idx] for layer in self.source_layers]
        return torch.cat(chunks, dim=1)


class ShortcutLayer(nn.Module):
    """A shortcut layer adds a residual connection from the source layer.

    Args:
        source_layer: Index of the layer whose output will be added to the output of the previous layer.

    """

    def __init__(self, source_layer: int) -> None:
        super().__init__()
        self.source_layer = source_layer

    def forward(self, outputs: list[Tensor]) -> Tensor:
        return outputs[-1] + outputs[self.source_layer]


class Mish(nn.Module):
    """Mish activation."""

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(nn.functional.softplus(x))


class ReOrg(nn.Module):
    """Re-organizes the tensor so that every square region of four cells is placed into four different channels.

    The result is a tensor with half the width and height, and four times as many channels.

    """

    def forward(self, x: Tensor) -> Tensor:
        tl = x[..., ::2, ::2]
        bl = x[..., 1::2, ::2]
        tr = x[..., ::2, 1::2]
        br = x[..., 1::2, 1::2]
        return torch.cat((tl, bl, tr, br), dim=1)


def _create_activation_module(name: str | None) -> nn.Module:
    """Creates a layer activation module given its type as a string.

    Args:
        name: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic", "linear",
            or "none".

    """
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "mish":
        return Mish()
    if name == "silu" or name == "swish":
        return nn.SiLU(inplace=True)
    if name == "logistic":
        return nn.Sigmoid()
    if name == "linear" or name == "none" or name is None:
        return nn.Identity()
    raise ValueError(f"Activation type `{name}´ is unknown.")


def _create_normalization_module(name: str | None, num_channels: int) -> nn.Module:
    """Creates a layer normalization module given its type as a string.

    Group normalization uses always 8 channels. The most common network widths are divisible by this number.

    Args:
        name: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        num_channels: The number of input channels that the module expects.

    """
    if name == "batchnorm":
        return nn.BatchNorm2d(num_channels, eps=0.001)
    if name == "groupnorm":
        return nn.GroupNorm(8, num_channels, eps=0.001)
    if name == "none" or name is None:
        return nn.Identity()
    raise ValueError(f"Normalization layer type `{name}´ is unknown.")


def create_detection_layer(
    prior_shapes: PRIOR_SHAPES,
    prior_shape_idxs: Sequence[int],
    num_classes: int,
    matching: MatchingConfig | None = None,
    loss: LossConfig | None = None,
    predict_confidence: bool = True,
    xy_scale: float = 1.0,
    input_is_normalized: bool = False,
) -> DetectionLayer:
    """Creates a detection layer module and the required loss function and target matching objects.

    Task-aligned matching ("tal") is applied by the :class:`DetectionHead` across all levels, so the returned layer has
    ``matching_func`` set to ``None`` in that case; :func:`create_detection_head` builds the head-level matcher.

    Args:
        prior_shapes: A list of all the prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        num_classes: Number of different classes that this layer predicts.
        matching: Configuration that controls how targets are assigned to anchors.
        loss: Configuration that controls how the detection losses are computed.
        predict_confidence: Whether the head predicts a confidence (objectness) channel. Set to ``False`` to drop
            objectness supervision and supervise all anchors via classification loss only.
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
        if not predict_confidence:
            raise ValueError(
                "predict_confidence=False is incompatible with SimOTA matching, which relies on the confidence cost."
            )
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
        predict_confidence=predict_confidence,
    )
    layer_shapes = [prior_shapes[i] for i in prior_shape_idxs]
    return DetectionLayer(
        num_classes=num_classes,
        prior_shapes=layer_shapes,
        matching_func=matching_func,
        loss_func=loss_func,
        xy_scale=xy_scale,
        input_is_normalized=input_is_normalized,
        predict_confidence=predict_confidence,
    )


def create_detection_head(
    prior_shapes: PRIOR_SHAPES,
    prior_shape_idxs_per_level: Sequence[Sequence[int]],
    num_classes: int,
    matching: MatchingConfig | None = None,
    loss: LossConfig | None = None,
    predict_confidence: bool = True,
    xy_scale: float = 1.0,
    input_is_normalized: bool = False,
) -> DetectionHead:
    """Creates a multi-level detection head with the appropriate matching strategy.

    Each feature level gets a :class:`DetectionLayer`. For task-aligned matching, a single :class:`TALMatching` is
    created and given to the head, which assigns targets across all levels at once; the per-level layers then carry no
    matcher. For all other algorithms, each layer carries its own per-level matcher.

    Args:
        prior_shapes: A list of all the prior box dimensions in the network input resolution.
        prior_shape_idxs_per_level: For each feature level, the indices into ``prior_shapes`` that the level uses.
        num_classes: Number of different classes that the head predicts.
        matching: Configuration that controls how targets are assigned to anchors.
        loss: Configuration that controls how the detection losses are computed.
        predict_confidence: Whether the head predicts a confidence (objectness) channel.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor.
        input_is_normalized: Whether the previous layer normalizes its output by a logistic activation.

    Returns:
        A detection head that owns the per-level layers and the matching strategy.

    """
    matching = matching or MatchingConfig()
    layers = [
        create_detection_layer(
            prior_shapes=prior_shapes,
            prior_shape_idxs=list(prior_shape_idxs),
            num_classes=num_classes,
            matching=matching,
            loss=loss,
            predict_confidence=predict_confidence,
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
            prior_shapes,
            [],
            topk=matching.tal_topk,
            alpha=matching.tal_alpha,
            beta=matching.tal_beta,
            ignore_bg_threshold=resolved.ignore_bg_threshold,
        )
    return DetectionHead(layers, matching_func)
