from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import Tensor
from torchvision.ops import box_iou


def grid_offsets(grid_size: Tensor) -> Tensor:
    """Given a grid size, returns a tensor containing offsets to the grid cells.

    Args:
        The width and height of the grid in a tensor.

    Returns:
        A ``[height, width, 2]`` tensor containing the grid cell `(x, y)` offsets.

    """
    x_range = torch.arange(cast(Any, grid_size[0]), device=grid_size.device)
    y_range = torch.arange(cast(Any, grid_size[1]), device=grid_size.device)
    grid_y, grid_x = torch.meshgrid([y_range, x_range], indexing="ij")
    return torch.stack((grid_x, grid_y), -1)


def grid_centers(grid_size: Tensor) -> Tensor:
    """Given a grid size, returns a tensor containing coordinates to the centers of the grid cells.

    Returns:
        A ``[height, width, 2]`` tensor containing coordinates to the centers of the grid cells.

    """
    return grid_offsets(grid_size) + 0.5


def global_xy(xy: Tensor, image_size: Tensor) -> Tensor:
    """Adds offsets to the predicted box center coordinates to obtain global coordinates to the image.

    The predicted coordinates are interpreted as coordinates inside a grid cell whose width and height is 1. Adding
    offset to the cell, dividing by the grid size, and multiplying by the image size, we get global coordinates in the
    image scale.

    Args:
        xy: The predicted center coordinates before scaling. Values from zero to one in a tensor sized
            ``[batch_size, height, width, boxes_per_cell, 2]``.
        image_size: Width and height in a vector that will be used to scale the coordinates.

    Returns:
        Global coordinates scaled to the size of the network input image, in a tensor with the same shape as the input
        tensor.

    """
    height = xy.shape[1]
    width = xy.shape[2]
    grid_size = torch.tensor([width, height], device=xy.device)
    offset = grid_offsets(grid_size).unsqueeze(2)  # [height, width, 1, 2]
    scale = torch.div(image_size, grid_size)
    return (xy + offset) * scale


def anchor_points_and_strides(feature_maps: Sequence[Tensor], image_size: Tensor) -> tuple[Tensor, Tensor]:
    """Generate one anchor point per feature-map location.

    Args:
        feature_maps: Feature tensors shaped ``[B, C, H, W]``.
        image_size: Input image width and height.

    Returns:
        Anchor points in grid coordinates shaped ``[N, 2]`` and matching stride values shaped ``[N, 1]``.

    """
    if len(feature_maps) == 0:
        raise ValueError("At least one feature map is required.")

    anchor_points = []
    strides = []
    for feature_map in feature_maps:
        if feature_map.ndim != 4:
            raise ValueError(f"Feature maps must be shaped [B, C, H, W], got {tuple(feature_map.shape)}.")
        height, width = feature_map.shape[-2:]
        feature_image_size = image_size.to(device=feature_map.device, dtype=feature_map.dtype)
        stride_xy = feature_image_size / torch.tensor(
            [width, height], device=feature_map.device, dtype=feature_map.dtype
        )
        if not torch.compiler.is_compiling() and not torch.isclose(stride_xy[0], stride_xy[1]):
            raise ValueError(
                f"Feature map stride must match in x and y, got {stride_xy[0].item()} and {stride_xy[1].item()}."
            )

        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=feature_map.device, dtype=feature_map.dtype) + 0.5,
            torch.arange(width, device=feature_map.device, dtype=feature_map.dtype) + 0.5,
            indexing="ij",
        )
        level_points = torch.stack((x_coords, y_coords), dim=-1).reshape(-1, 2)
        anchor_points.append(level_points)
        strides.append(stride_xy[:1].expand(level_points.shape[0], 1))

    return torch.cat(anchor_points), torch.cat(strides)


def distance_offsets_to_boxes(distance_offsets: Tensor, anchor_points: Tensor) -> Tensor:
    """Convert left/top/right/bottom distances from anchor points to corner-format boxes.

    Args:
        distance_offsets: Distances shaped ``[..., N, 4]`` or ``[N, 4]``.
        anchor_points: Point coordinates shaped ``[N, 2]``.

    Returns:
        Corner-format boxes shaped like ``distance_offsets``.

    """
    left_top = distance_offsets[..., :2]
    right_bottom = distance_offsets[..., 2:]
    return torch.cat((anchor_points - left_top, anchor_points + right_bottom), dim=-1)


def boxes_to_distance_offsets(boxes: Tensor, anchor_points: Tensor, num_dfl_bins: int) -> Tensor:
    """Convert corner-format boxes to point-relative left/top/right/bottom distances.

    Args:
        boxes: Corner-format boxes shaped ``[..., N, 4]`` or ``[N, 4]``.
        anchor_points: Point coordinates shaped ``[N, 2]``.
        num_dfl_bins: Number of distance bins used by the regression distribution.

    Returns:
        Clipped distance offsets shaped like ``boxes``.

    """
    if num_dfl_bins < 2:
        raise ValueError("Distance targets require at least two DFL bins.")
    left_top = anchor_points - boxes[..., :2]
    right_bottom = boxes[..., 2:] - anchor_points
    return torch.cat((left_top, right_bottom), dim=-1).clamp(0, num_dfl_bins - 1 - 0.01)


def aligned_iou(wh1: Tensor, wh2: Tensor) -> Tensor:
    """Calculates a matrix of intersections over union from box dimensions, assuming that the boxes are located at the
    same coordinates.

    Args:
        wh1: An ``[N, 2]`` matrix of box shapes (width and height).
        wh2: An ``[M, 2]`` matrix of box shapes (width and height).

    Returns:
        An ``[N, M]`` matrix of pairwise IoU values for every element in ``wh1`` and ``wh2``

    """
    area1 = wh1[:, 0] * wh1[:, 1]  # [N]
    area2 = wh2[:, 0] * wh2[:, 1]  # [M]

    inter_wh = torch.min(wh1[:, None, :], wh2)  # [N, M, 2]
    inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2 - inter  # [N, M]

    return inter / union


def iou_below(pred_boxes: Tensor, target_boxes: Tensor, threshold: float) -> Tensor:
    """Creates a binary mask whose value will be ``True``, unless the predicted box overlaps any target significantly
    (IoU greater than ``threshold``).

    Args:
        pred_boxes: The predicted corner coordinates. Tensor of size ``[height, width, boxes_per_cell, 4]``.
        target_boxes: Corner coordinates of the target boxes. Tensor of size ``[num_targets, 4]``.

    Returns:
        A boolean tensor sized ``[height, width, boxes_per_cell]``, with ``False`` where the predicted box overlaps a
        target significantly and ``True`` elsewhere.

    """
    shape = pred_boxes.shape[:-1]
    if target_boxes.shape[0] == 0:
        return torch.ones(shape, dtype=torch.bool, device=pred_boxes.device)

    pred_boxes = pred_boxes.view(-1, 4)
    ious = box_iou(pred_boxes, target_boxes)
    best_iou = ious.max(-1).values
    below_threshold = best_iou <= threshold
    return below_threshold.view(shape)


def is_inside_box(points: Tensor, boxes: Tensor) -> Tensor:
    """Get pairwise truth values of whether the point is inside the box.

    Args:
        points: Point (x, y) coordinates, a tensor shaped ``[points, 2]``.
        boxes: Box (x1, y1, x2, y2) coordinates, a tensor shaped ``[boxes, 4]``.

    Returns:
        A tensor shaped ``[points, boxes]`` containing pairwise truth values of whether the points are inside the boxes.

    """
    lt = points[:, None, :] - boxes[None, :, :2]  # [boxes, points, 2]
    rb = boxes[None, :, 2:] - points[:, None, :]  # [boxes, points, 2]
    deltas = torch.cat((lt, rb), -1)  # [points, boxes, 4]
    return deltas.min(-1).values > 0.0  # [points, boxes]


def box_size_ratio(wh1: Tensor, wh2: Tensor) -> Tensor:
    """Compares the dimensions of the boxes pairwise.

    For each pair of boxes, calculates the largest ratio that can be obtained by dividing the widths with each other or
    dividing the heights with each other.

    Args:
        wh1: An ``[N, 2]`` matrix of box shapes (width and height).
        wh2: An ``[M, 2]`` matrix of box shapes (width and height).

    Returns:
        An ``[N, M]`` matrix of ratios of width or height dimensions, whichever is larger.

    """
    wh_ratio = wh1[:, None, :] / wh2[None, :, :]  # [N, M, 2]
    wh_ratio = torch.max(wh_ratio, 1.0 / wh_ratio)
    return wh_ratio.max(2).values  # [N, M]


def get_image_size(images: Tensor) -> Tensor:
    """Get the image size from an input tensor.

    Args:
        images: An image batch to take the width and height from.

    Returns:
        A tensor that contains the image width and height.

    """
    height = images.shape[2]
    width = images.shape[3]
    return torch.tensor([width, height], device=images.device)
