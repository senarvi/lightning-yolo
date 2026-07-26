import io
import re
from collections.abc import Callable, Iterable
from typing import Any
from warnings import warn

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch import Tensor, nn

from .config import LossConfig, MatchingConfig
from .initialization import detection_classprob_bias, detection_confidence_bias, initialize_yolo_logits
from .layers import (
    Conv,
    DetectionLayer,
    MaxPool,
    RouteLayer,
    ShortcutLayer,
    create_detection_layer,
)
from .types import NETWORK_OUTPUT, PRIOR_SHAPES, DetectionLossRecord, PackedTargetDict
from .utils import get_image_size

DARKNET_CONFIG = dict[str, Any]
CREATE_LAYER_OUTPUT = tuple[nn.Module, int]  # layer, num_outputs


class DarknetNetwork(nn.Module):
    """This class can be used to parse the configuration files of the Darknet YOLOv4 implementation.

    Iterates through the layers from the configuration and creates corresponding PyTorch modules. If ``weights_path`` is
    given and points to a Darknet model file, loads the convolutional layer weights from the file.

    Args:
        config_path: Path to a Darknet configuration file that defines the network architecture.
        weights_path: Path to a Darknet model file. If given, the model weights will be read from this file.
        in_channels: Number of channels in the input image.
        num_classes: Number of object classes. If not given, the value will be read from the configuration file.
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        matching: Configuration that controls how targets are assigned to anchors. Fields left as ``None`` will fall
            back to the corresponding values from the Darknet ``.cfg`` file where applicable.
        loss: Configuration that controls how detection losses are computed. Fields left as ``None`` will fall back to
            the corresponding values from the Darknet ``.cfg`` file where applicable.

    """

    def __init__(
        self, config_path: str, weights_path: str | None = None, in_channels: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__()

        with open(config_path) as config_file:
            sections = self._read_config(config_file)

        if len(sections) < 2:
            raise MisconfigurationException("The model configuration file should include at least two sections.")

        self.__dict__.update(sections[0])
        global_config = sections[0]
        layer_configs = sections[1:]

        if in_channels is None:
            in_channels = global_config.get("channels", 3)
            assert isinstance(in_channels, int)

        self.layers = nn.ModuleList()
        # num_inputs will contain the number of channels in the input of every layer up to the current layer. It is
        # initialized with the number of channels in the input image.
        num_inputs = [in_channels]
        for layer_config in layer_configs:
            config = {**global_config, **layer_config}
            layer, num_outputs = _create_layer(config, num_inputs, **kwargs)
            self.layers.append(layer)
            num_inputs.append(num_outputs)

        # Always initialize the output convolutions. Darknet weight files typically contain weights only for the layers
        # up to the first detection layer and load_weights() leaves the rest of the layers unchanged.
        _initialize_detection_logits(self)
        if weights_path is not None:
            with open(weights_path, "rb") as weight_file:
                self.load_weights(weight_file)

    def forward(self, x: Tensor, targets: PackedTargetDict | None = None) -> NETWORK_OUTPUT:
        outputs: list[Tensor] = []  # Outputs from all layers
        detections: list[Tensor] = []  # Outputs from detection layers
        losses: list[DetectionLossRecord] = []  # Loss records from detection layers

        image_size = get_image_size(x)

        for layer in self.layers:
            if isinstance(layer, (RouteLayer, ShortcutLayer)):
                x = layer(outputs)
            elif isinstance(layer, DetectionLayer):
                x, preds = layer(x, image_size)
                detections.append(x)
                if targets is not None:
                    # Darknet configurations always use per-level matchers (TAL is rejected during construction).
                    assert layer.matching_func is not None
                    matching_result = layer.matching_func(preds, targets, image_size, layer.input_is_normalized)
                    losses.append(
                        layer.loss_func.matched_losses(matching_result, preds, layer.input_is_normalized, image_size)
                    )
            else:
                x = layer(x)

            outputs.append(x)

        return detections, losses

    def load_weights(self, weight_file: io.IOBase) -> None:
        """Loads weights to layer modules from a pretrained Darknet model.

        One may want to continue training from pretrained weights, on a dataset with a different number of object
        categories. The number of kernels in the convolutional layers just before each detection layer depends on the
        number of output classes. The Darknet solution is to truncate the weight file and stop reading weights at the
        first incompatible layer. For this reason the function silently leaves the rest of the layers unchanged, when
        the weight file ends.

        Args:
            weight_file: A file-like object containing model weights in the Darknet binary format.

        """
        if not isinstance(weight_file, io.IOBase):
            raise ValueError("weight_file must be a file-like object.")

        version = np.fromfile(weight_file, count=3, dtype=np.int32)
        images_seen = np.fromfile(weight_file, count=1, dtype=np.int64)
        rank_zero_info(
            f"Loading weights from Darknet model version {version[0]}.{version[1]}.{version[2]} "
            f"that has been trained on {images_seen[0]} images."
        )

        def read(tensor: Tensor) -> int:
            """Reads the contents of ``tensor`` from the current position of ``weight_file``.

            Returns the number of elements read. If there's no more data in ``weight_file``, returns 0.

            """
            np_array = np.fromfile(weight_file, count=tensor.numel(), dtype=np.float32)
            num_elements = np_array.size
            if num_elements > 0:
                if num_elements < tensor.numel():
                    raise EOFError("Darknet weight file ended in the middle of a tensor.")
                source = torch.from_numpy(np_array).view_as(tensor)
                with torch.no_grad():
                    tensor.copy_(source)
            return num_elements

        for layer in self.layers:
            # Weights are loaded only to convolutional layers
            if not isinstance(layer, Conv):
                continue

            # If convolution is followed by batch normalization, read the batch normalization parameters. Otherwise we
            # read the convolution bias.
            if isinstance(layer.norm, nn.Identity):
                assert layer.conv.bias is not None
                read(layer.conv.bias)
            else:
                assert isinstance(layer.norm, nn.BatchNorm2d)
                assert layer.norm.running_mean is not None
                assert layer.norm.running_var is not None
                read(layer.norm.bias)
                read(layer.norm.weight)
                read(layer.norm.running_mean)
                read(layer.norm.running_var)

            read_count = read(layer.conv.weight)
            if read_count == 0:
                return

    def _read_config(self, config_file: Iterable[str]) -> list[dict[str, Any]]:
        """Reads a Darnet network configuration file and returns a list of configuration sections.

        Args:
            config_file: The configuration file to read.

        Returns:
            A list of configuration sections.

        """
        section_re = re.compile(r"\[([^]]+)\]")
        list_variables = ("layers", "anchors", "mask", "scales")
        variable_types = {
            "activation": str,
            "anchors": int,
            "angle": float,
            "batch": int,
            "batch_normalize": bool,
            "beta_nms": float,
            "burn_in": int,
            "channels": int,
            "classes": int,
            "cls_normalizer": float,
            "decay": float,
            "exposure": float,
            "filters": int,
            "from": int,
            "groups": int,
            "group_id": int,
            "height": int,
            "hue": float,
            "ignore_thresh": float,
            "iou_loss": str,
            "iou_normalizer": float,
            "iou_thresh": float,
            "jitter": float,
            "layers": int,
            "learning_rate": float,
            "mask": int,
            "max_batches": int,
            "max_delta": float,
            "momentum": float,
            "mosaic": bool,
            "new_coords": int,
            "nms_kind": str,
            "num": int,
            "obj_normalizer": float,
            "pad": bool,
            "policy": str,
            "random": bool,
            "resize": float,
            "saturation": float,
            "scales": float,
            "scale_x_y": float,
            "size": int,
            "steps": str,
            "stopbackward": int,
            "stride": int,
            "subdivisions": int,
            "truth_thresh": float,
            "width": int,
        }

        section = None
        sections = []

        def parse_bool(value: str) -> bool:
            if value == "1":
                return True
            if value == "0":
                return False
            raise ValueError(f"Invalid boolean value in Darknet configuration: {value}")

        def convert(key: str, value: str) -> str | int | float | bool | list[str | int | float]:
            """Converts a value to the correct type based on key."""
            if key not in variable_types:
                warn(f"Unknown YOLO configuration variable: {key}", stacklevel=2)
                return value
            if key in list_variables:
                return [variable_types[key](v) for v in value.split(",")]
            if variable_types[key] is bool:
                return parse_bool(value)
            return variable_types[key](value)

        for line in config_file:
            line = line.strip()
            if (not line) or (line[0] == "#"):
                continue

            section_match = section_re.match(line)
            if section_match:
                if section is not None:
                    sections.append(section)
                section = {"type": section_match.group(1)}
            else:
                if section is None:
                    raise RuntimeError("Darknet network configuration file does not start with a section header.")
                key, value = line.split("=")
                key = key.rstrip()
                value = value.lstrip()
                section[key] = convert(key, value)
        if section is not None:
            sections.append(section)

        return sections


def _create_layer(config: DARKNET_CONFIG, num_inputs: list[int], **kwargs: Any) -> CREATE_LAYER_OUTPUT:
    """Calls one of the ``_create_<layertype>(config, num_inputs)`` functions to create a PyTorch module from the layer
    config.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    create_func: dict[str, Callable[..., CREATE_LAYER_OUTPUT]] = {
        "convolutional": _create_convolutional,
        "maxpool": _create_maxpool,
        "route": _create_route,
        "shortcut": _create_shortcut,
        "upsample": _create_upsample,
        "yolo": _create_yolo,
    }
    return create_func[config["type"]](config, num_inputs, **kwargs)


def _create_convolutional(config: DARKNET_CONFIG, num_inputs: list[int], **_: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a convolutional layer.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    batch_normalize = config.get("batch_normalize", False)
    padding = (config["size"] - 1) // 2 if config["pad"] else 0

    layer = Conv(
        num_inputs[-1],
        config["filters"],
        kernel_size=config["size"],
        stride=config["stride"],
        padding=padding,
        bias=not batch_normalize,
        activation=config["activation"],
        norm="batchnorm" if batch_normalize else None,
    )
    return layer, config["filters"]


def _create_maxpool(config: DARKNET_CONFIG, num_inputs: list[int], **_: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a max pooling layer.

    Padding is added so that the output resolution will be the input resolution divided by stride, rounded upwards.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    layer = MaxPool(config["size"], config["stride"])
    return layer, num_inputs[-1]


def _create_route(config: DARKNET_CONFIG, num_inputs: list[int], **_: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a routing layer.

    A routing layer concatenates the output (or part of it) from the layers specified by the "layers" configuration
    option.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    num_chunks = config.get("groups", 1)
    chunk_idx = config.get("group_id", 0)

    # 0 is the first layer, -1 is the previous layer
    last = len(num_inputs) - 1
    source_layers = [layer if layer >= 0 else last + layer for layer in config["layers"]]

    layer = RouteLayer(source_layers, num_chunks, chunk_idx)

    # The number of outputs of a source layer is the number of inputs of the next layer.
    num_outputs = sum(num_inputs[layer + 1] // num_chunks for layer in source_layers)

    return layer, num_outputs


def _create_shortcut(config: DARKNET_CONFIG, num_inputs: list[int], **_: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a shortcut layer.

    A shortcut layer adds a residual connection from the layer specified by the "from" configuration option.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    layer = ShortcutLayer(config["from"])
    return layer, num_inputs[-1]


def _create_upsample(config: DARKNET_CONFIG, num_inputs: list[int], **_: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a layer that upsamples the data.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    layer = nn.Upsample(scale_factor=config["stride"], mode="nearest")
    return layer, num_inputs[-1]


def _create_yolo(
    config: DARKNET_CONFIG,
    num_inputs: list[int],  # noqa: ARG001
    num_classes: int | None = None,
    prior_shapes: PRIOR_SHAPES | None = None,
    matching: MatchingConfig | None = None,
    loss: LossConfig | None = None,
    xy_scale: float | None = None,
    **_: Any,
) -> CREATE_LAYER_OUTPUT:
    """Creates a YOLO detection layer.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer. Not used by the detection layer.
        num_classes: Number of object classes. If not given, the value will be read from the configuration file.
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        matching: Configuration that controls how targets are assigned to anchors. Fields left as ``None`` will fall
            back to the corresponding values from the Darknet ``.cfg`` file where applicable.
        loss: Configuration that controls how detection losses are computed. Fields left as ``None`` will fall back to
            the corresponding values from the Darknet ``.cfg`` file where applicable.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output (always 0 for a detection layer).

    """
    matching = matching or MatchingConfig()
    loss = loss or LossConfig()

    if num_classes is None:
        num_classes = config["classes"]
        assert isinstance(num_classes, int)
    if matching.algorithm == "tal":
        raise ValueError(
            'Task-aligned matching ("tal") is not supported in Darknet configurations. It assigns targets across all '
            "feature levels at once, but a Darknet model is a sequential graph that processes one detection layer at a "
            "time. Use a native architecture for task-aligned matching."
        )
    if prior_shapes is None:
        # The "anchors" list alternates width and height.
        dims = config["anchors"]
        prior_shapes = [(dims[i], dims[i + 1]) for i in range(0, len(dims), 2)]
    ignore_bg_threshold = matching.ignore_bg_threshold
    if ignore_bg_threshold is None:
        ignore_bg_threshold = config.get("ignore_thresh", 1.0)
        assert isinstance(ignore_bg_threshold, float)

    overlap_func = loss.overlap_func
    if overlap_func is None:
        overlap_func = config.get("iou_loss", "iou")
        assert isinstance(overlap_func, str | Callable)

    overlap_multiplier = loss.overlap_multiplier
    if overlap_multiplier is None:
        overlap_multiplier = config.get("iou_normalizer", 1.0)
        assert isinstance(overlap_multiplier, float)

    confidence_multiplier = loss.confidence_multiplier
    if confidence_multiplier is None:
        confidence_multiplier = config.get("obj_normalizer", 1.0)
        assert isinstance(confidence_multiplier, float)

    class_multiplier = loss.class_multiplier
    if class_multiplier is None:
        class_multiplier = config.get("cls_normalizer", 1.0)
        assert isinstance(class_multiplier, float)

    if xy_scale is None:
        xy_scale = config.get("scale_x_y", 1.0)
        assert isinstance(xy_scale, float)

    layer = create_detection_layer(
        num_classes=num_classes,
        prior_shapes=prior_shapes,
        prior_shape_idxs=config["mask"],
        matching=MatchingConfig(
            algorithm=matching.algorithm,
            threshold=matching.threshold,
            spatial_range=matching.spatial_range,
            size_range=matching.size_range,
            ignore_bg_threshold=ignore_bg_threshold,
        ),
        loss=LossConfig(
            overlap_func=overlap_func,
            predict_overlap=loss.predict_overlap,
            label_smoothing=loss.label_smoothing,
            overlap_multiplier=overlap_multiplier,
            confidence_multiplier=confidence_multiplier,
            class_multiplier=class_multiplier,
        ),
        xy_scale=xy_scale,
        input_is_normalized=config.get("new_coords", 0) > 0,
    )
    return layer, 0


def _initialize_detection_logits(network: DarknetNetwork) -> None:
    """Initializes output convolutions that directly precede Darknet detection layers.

    Args:
        network: Parsed Darknet network to initialize.

    """
    confidence_bias = detection_confidence_bias()
    for idx, layer in enumerate(network.layers):
        if isinstance(layer, DetectionLayer) and idx > 0:
            previous_layer = network.layers[idx - 1]
            if isinstance(previous_layer, Conv):
                classprob_bias = detection_classprob_bias(layer.num_classes)
                initialize_yolo_logits(previous_layer.conv, layer.num_classes, confidence_bias, classprob_bias)
