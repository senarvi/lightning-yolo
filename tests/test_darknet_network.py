import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
from lightning.pytorch.utilities.warnings import PossibleUserWarning

from lightning_yolo.darknet_network import (
    DarknetNetwork,
    _create_convolutional,
    _create_maxpool,
    _create_shortcut,
    _create_upsample,
)
from lightning_yolo.initialization import detection_classprob_bias, detection_confidence_bias


def test_darknet_network(tmp_path) -> None:
    config_path = tmp_path / "tiny.cfg"
    config_path.write_text(
        """
[net]
width=32
height=32
channels=3

[convolutional]
batch_normalize=0
filters=7
size=1
stride=1
pad=1
activation=linear

[yolo]
mask=0
anchors=10,10
classes=2
ignore_thresh=.7
""".lstrip(),
        encoding="utf-8",
    )

    weights_path = tmp_path / "tiny.conv"
    with weights_path.open("wb") as weights_file:
        np.array([0, 2, 5], dtype=np.int32).tofile(weights_file)
        np.array([0], dtype=np.int64).tofile(weights_file)

    network = DarknetNetwork(str(config_path), str(weights_path))
    conv = network.layers[0].conv
    assert conv.bias is not None
    bias = conv.bias.view(1, 7)

    torch.testing.assert_close(bias[:, :4], torch.zeros_like(bias[:, :4]))
    torch.testing.assert_close(bias[:, 4], torch.full_like(bias[:, 4], detection_confidence_bias()))
    torch.testing.assert_close(bias[:, 5:], torch.full_like(bias[:, 5:], detection_classprob_bias(2)))


@pytest.mark.parametrize(
    "config",
    [
        (
            {
                "batch_normalize": 1,
                "filters": 8,
                "size": 3,
                "stride": 1,
                "pad": 1,
                "activation": "leaky",
            }
        ),
        (
            {
                "batch_normalize": 0,
                "filters": 2,
                "size": 1,
                "stride": 1,
                "pad": 1,
                "activation": "mish",
            }
        ),
        (
            {
                "batch_normalize": 1,
                "filters": 6,
                "size": 3,
                "stride": 2,
                "pad": 1,
                "activation": "logistic",
            }
        ),
        (
            {
                "batch_normalize": 0,
                "filters": 4,
                "size": 3,
                "stride": 2,
                "pad": 0,
                "activation": "linear",
            }
        ),
    ],
)
def test_create_convolutional(config):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    conv, _ = _create_convolutional(config, [3])

    assert conv.conv.out_channels == config["filters"]
    assert conv.conv.kernel_size == (config["size"], config["size"])
    assert conv.conv.stride == (config["stride"], config["stride"])

    pad_size = (config["size"] - 1) // 2 if config["pad"] else 0
    if config["pad"]:
        assert conv.conv.padding == (pad_size, pad_size)

    if config["batch_normalize"]:
        assert isinstance(conv.norm, nn.BatchNorm2d)

    if config["activation"] == "linear":
        assert isinstance(conv.act, nn.Identity)
    elif config["activation"] == "logistic":
        assert isinstance(conv.act, nn.Sigmoid)
    else:
        assert conv.act.__class__.__name__.lower().startswith(config["activation"])


@pytest.mark.parametrize(
    "config",
    [
        (
            {
                "size": 2,
                "stride": 2,
            }
        ),
        (
            {
                "size": 6,
                "stride": 3,
            }
        ),
    ],
)
def test_create_maxpool(config):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    pad_size, remainder = divmod(max(config["size"], config["stride"]) - config["stride"], 2)
    maxpool, _ = _create_maxpool(config, [3])

    assert maxpool.maxpool.kernel_size == config["size"]
    assert maxpool.maxpool.stride == config["stride"]
    assert maxpool.maxpool.padding == pad_size
    if remainder != 0:
        assert isinstance(maxpool.pad, nn.ZeroPad2d)


@pytest.mark.parametrize(
    "config",
    [
        ({"from": 1, "activation": "linear"}),
        ({"from": 3, "activation": "linear"}),
    ],
)
def test_create_shortcut(config):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    shortcut, _ = _create_shortcut(config, [3])

    assert shortcut.source_layer == config["from"]


@pytest.mark.parametrize(
    "config",
    [
        ({"stride": 2}),
        ({"stride": 4}),
    ],
)
def test_create_upsample(config):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    upsample, _ = _create_upsample(config, [3])

    assert upsample.scale_factor == float(config["stride"])
