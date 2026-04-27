import math

import torch
from torch import nn


def detection_confidence_bias(image_size: int = 640, confidence_prior: float = 8.0) -> float:
    """Computes the initial confidence logit bias for detection heads.

    Args:
        image_size: Reference input image size used for the prior estimate.
        confidence_prior: Expected number of positive anchors per image at initialization.

    Returns:
        A logit bias that initializes confidence predictions to a low prior probability.

    """
    return math.log(confidence_prior / ((image_size / 32) ** 2))


def detection_classprob_bias(num_classes: int, image_size: int = 640, objects_per_image: float = 5.0) -> float:
    """Computes the initial class probability logit bias for detection heads.

    Args:
        num_classes: Number of predicted classes.
        image_size: Reference input image size used for the prior estimate.
        objects_per_image: Expected number of objects per image at initialization.

    Returns:
        A logit bias for class predictions.

    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    return math.log(objects_per_image / num_classes / ((image_size / 32) ** 2))


def initialize_constant_bias(conv: nn.Conv2d, bias: float, weight_std: float = 0.01) -> None:
    """Initializes a logit output convolution with a constant bias.

    The weights are initialized with a small normal distribution and the bias is set to the provided prior logit.

    Args:
        conv: Convolution that predicts logits.
        bias: Initial logit bias for predictions.
        weight_std: Standard deviation for normal weight initialization.

    """
    nn.init.normal_(conv.weight, mean=0.0, std=weight_std)
    if conv.bias is not None:
        nn.init.constant_(conv.bias, bias)


def initialize_zero_bias(conv: nn.Conv2d, weight_std: float = 0.01) -> None:
    """Initializes a box-regression output convolution.

    The weights are initialized with a small normal distribution and the bias is set to zero.

    Args:
        conv: Convolution that predicts box coordinates.
        weight_std: Standard deviation for normal weight initialization.

    """
    nn.init.normal_(conv.weight, mean=0.0, std=weight_std)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def initialize_yolo_logits(
    conv: nn.Conv2d,
    num_classes: int,
    confidence_bias: float,
    classprob_bias: float,
    weight_std: float = 0.01,
) -> None:
    """Initializes a coupled YOLO output convolution.

    Coupled heads predict box, confidence, and class values from a single convolution where output channels are grouped
    as ``(x, y, width, height, confidence, class_probs...)`` per anchor.

    Args:
        conv: Coupled output convolution.
        num_classes: Number of predicted classes.
        confidence_bias: Initial logit bias for confidence predictions.
        classprob_bias: Initial logit bias for class probability predictions.
        weight_std: Standard deviation for normal weight initialization.

    Raises:
        ValueError: If the number of convolution output channels is not divisible by ``num_classes + 5``.

    """
    num_attrs = num_classes + 5
    anchors_per_cell, remainder = divmod(conv.out_channels, num_attrs)
    if remainder != 0:
        raise ValueError(
            f"Detection layer receives {conv.out_channels} features, but expects a multiple of {num_attrs} features, "
            f"given {num_classes} classes."
        )

    nn.init.normal_(conv.weight, mean=0.0, std=weight_std)
    if conv.bias is not None:
        with torch.no_grad():
            bias = conv.bias.view(anchors_per_cell, num_attrs)
            bias[:, :4].zero_()
            bias[:, 4].fill_(confidence_bias)
            bias[:, 5:].fill_(classprob_bias)
