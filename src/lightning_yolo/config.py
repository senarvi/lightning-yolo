"""Typed configuration objects for the detection matching and loss parameters.

These dataclasses replace the long list of loose keyword arguments that used to be threaded through the model, the
networks, and the detection head and layer factories. Grouping the parameters makes the call sites explicit and lets
type checking catch misplaced or misspelled arguments.

Fields default to ``None`` where the effective default depends on the context: native architectures fall back to the
values baked into :meth:`MatchingConfig.with_defaults` and :meth:`LossConfig.with_defaults`, while Darknet
configurations read them from the ``.cfg`` file. Call ``with_defaults()`` to obtain a copy with every context-dependent
field resolved to the native default.

"""

from collections.abc import Callable
from dataclasses import dataclass, replace

_DEFAULT_IGNORE_BG_THRESHOLD = 0.7
_DEFAULT_OVERLAP_FUNC = "ciou"
_DEFAULT_OVERLAP_MULTIPLIER = 7.5
_DEFAULT_CONFIDENCE_MULTIPLIER = 1.0
_DEFAULT_CLASS_MULTIPLIER = 0.5
_DEFAULT_DFL_MULTIPLIER = 1.5
_DEFAULT_NUM_DFL_BINS = 16


@dataclass
class MatchingConfig:
    """Parameters that control how targets are assigned to anchors.

    Args:
        algorithm: Which algorithm to use for matching targets to anchors or points. "simota" (the SimOTA matching
            rule from YOLOX), "tal" (task-aligned top-k matching as used in Ultralytics YOLOv8), "size" (match those
            prior shapes whose width and height relative to the target is below a given ratio), "iou" (match all prior
            shapes that give a high enough IoU), or "maxiou" (match the prior shape that gives the highest IoU, the
            default).
        threshold: Threshold for the "size" and "iou" matching algorithms.
        spatial_range: The "simota" algorithm restricts to anchors within an `N × N` grid cell area centered at the
            target, where `N` is this value.
        size_range: The "simota" algorithm restricts to anchors whose dimensions are between `1/N` and `N` times the
            target dimensions, where `N` is this value.
        tal_topk: The "tal" algorithm selects up to this many top candidates per target.
        tal_alpha: Exponent for the class confidence in the TAL alignment metric.
        tal_beta: Exponent for the IoU in the TAL alignment metric.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor is not taken into account when
            calculating the confidence loss. When ``None``, native architectures use 0.7 and Darknet configurations
            read ``ignore_thresh`` from the configuration file.

    """

    algorithm: str | None = None
    threshold: float | None = None
    spatial_range: float = 5.0
    size_range: float = 4.0
    tal_topk: int = 10
    tal_alpha: float = 0.5
    tal_beta: float = 6.0
    ignore_bg_threshold: float | None = None

    def with_defaults(self) -> MatchingConfig:
        """Returns a copy with context-dependent fields resolved to their native defaults."""
        return replace(
            self,
            ignore_bg_threshold=(
                self.ignore_bg_threshold if self.ignore_bg_threshold is not None else _DEFAULT_IGNORE_BG_THRESHOLD
            ),
        )


@dataclass
class LossConfig:
    """Parameters that control how the detection losses are computed.

    Args:
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Valid values are
            "iou", "giou", "diou", and "ciou". When ``None``, native architectures use "ciou" and Darknet
            configurations read ``iou_loss`` from the configuration file.
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means the target
            confidence is one if there's an object, and 1.0 means the target confidence is the output of
            ``overlap_func``. Distributional-distance architectures do not use confidence targets and reject this
            option.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means the target probabilities are always 0.5. Distributional-distance architectures
            apply this to foreground classification targets before task-aligned assignment weighting.
        overlap_multiplier: Overlap loss is scaled by this value. Native default 7.5.
        confidence_multiplier: Confidence loss is scaled by this value. Native default 1.0. Distributional-distance
            architectures do not have a confidence loss and reject this option.
        class_multiplier: Classification loss is scaled by this value. Native default 0.5.
        dfl_multiplier: Distribution focal loss is scaled by this value. Native default 1.5.
        num_dfl_bins: Number of distance bins used by DFL regression when enabled. Native default 16.

    """

    overlap_func: str | Callable | None = None
    predict_overlap: float | None = None
    label_smoothing: float | None = None
    overlap_multiplier: float | None = None
    confidence_multiplier: float | None = None
    class_multiplier: float | None = None
    dfl_multiplier: float | None = None
    num_dfl_bins: int | None = None

    def with_defaults(self) -> LossConfig:
        """Returns a copy with context-dependent fields resolved to their native defaults."""
        return replace(
            self,
            overlap_func=self.overlap_func if self.overlap_func is not None else _DEFAULT_OVERLAP_FUNC,
            overlap_multiplier=(
                self.overlap_multiplier if self.overlap_multiplier is not None else _DEFAULT_OVERLAP_MULTIPLIER
            ),
            confidence_multiplier=(
                self.confidence_multiplier if self.confidence_multiplier is not None else _DEFAULT_CONFIDENCE_MULTIPLIER
            ),
            class_multiplier=(
                self.class_multiplier if self.class_multiplier is not None else _DEFAULT_CLASS_MULTIPLIER
            ),
            dfl_multiplier=(self.dfl_multiplier if self.dfl_multiplier is not None else _DEFAULT_DFL_MULTIPLIER),
            num_dfl_bins=(self.num_dfl_bins if self.num_dfl_bins is not None else _DEFAULT_NUM_DFL_BINS),
        )
