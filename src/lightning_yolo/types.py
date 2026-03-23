from typing import TypedDict

from torch import Tensor


class PredictionDict(TypedDict):
    boxes: Tensor
    confidences: Tensor
    classprobs: Tensor


class TargetDict(TypedDict):
    boxes: Tensor
    labels: Tensor


class MatchedPredictionDict(TypedDict):
    boxes: Tensor
    confidences: Tensor
    bg_confidences: Tensor
    classprobs: Tensor


class MatchedTargetDict(TypedDict):
    boxes: Tensor
    labels: Tensor


IMAGES = tuple[Tensor, ...] | list[Tensor]
PREDICTIONS = tuple[PredictionDict, ...] | list[PredictionDict]
PRIOR_SHAPES = list[tuple[int, int]]
TARGETS = tuple[TargetDict, ...] | list[TargetDict]
BATCH = tuple[IMAGES, TARGETS]
NETWORK_OUTPUT = tuple[list[Tensor], list[Tensor], list[int]]  # detections, losses, hits
