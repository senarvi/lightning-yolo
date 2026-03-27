import torch

from lightning_yolo.layers import (
    Conv,
    DetectionLayer,
    MaxPool,
    Mish,
    ReOrg,
    RouteLayer,
    ShortcutLayer,
)
from lightning_yolo.loss import YOLOLoss
from lightning_yolo.target_matching import HighestIoUMatching


def test_detection_layer_forward_shapes():
    prior_shapes = [(10, 12), (20, 24)]
    layer = DetectionLayer(
        num_classes=2,
        prior_shapes=prior_shapes,
        matching_func=HighestIoUMatching(prior_shapes, [0, 1]),
        loss_func=YOLOLoss("ciou"),
    )
    x = torch.randn(1, 14, 2, 2)
    image_size = torch.tensor([64, 64])
    output, preds = layer(x, image_size)

    assert output.shape == (1, 8, 7)
    assert len(preds) == 1
    assert preds[0]["boxes"].shape == (2, 2, 2, 4)
    assert preds[0]["confidences"].shape == (2, 2, 2)
    assert preds[0]["classprobs"].shape == (2, 2, 2, 2)
    assert torch.isfinite(output).all()


def test_conv_layer_output_shape():
    layer = Conv(3, 8, kernel_size=3, stride=2, activation="silu", norm="batchnorm")
    x = torch.randn(2, 3, 11, 11)
    y = layer(x)

    assert y.shape == (2, 8, 5, 5)


def test_maxpool_layer_output_shape():
    layer = MaxPool(kernel_size=2, stride=2)
    x = torch.randn(1, 3, 5, 5)
    y = layer(x)

    assert y.shape == (1, 3, 2, 2)


def test_route_layer_concatenates_selected_chunks():
    outputs = [
        torch.arange(1 * 4 * 2 * 2, dtype=torch.float32).view(1, 4, 2, 2),
        torch.arange(100, 100 + (1 * 4 * 2 * 2), dtype=torch.float32).view(1, 4, 2, 2),
    ]
    layer = RouteLayer(source_layers=[0, 1], num_chunks=2, chunk_idx=1)
    y = layer(outputs)
    expected = torch.cat((outputs[0][:, 2:], outputs[1][:, 2:]), dim=1)

    assert y.shape == (1, 4, 2, 2)
    assert torch.equal(y, expected)


def test_shortcut_layer_adds_residual_source():
    outputs = [
        torch.ones(1, 3, 2, 2),
        torch.full((1, 3, 2, 2), 2.0),
        torch.full((1, 3, 2, 2), 3.0),
    ]
    layer = ShortcutLayer(source_layer=0)
    y = layer(outputs)

    assert torch.equal(y, outputs[-1] + outputs[0])


def test_mish_activation_matches_definition():
    x = torch.tensor([-2.0, 0.0, 3.0])
    layer = Mish()
    y = layer(x)
    expected = x * torch.tanh(torch.nn.functional.softplus(x))

    torch.testing.assert_close(y, expected)


def test_reorg_rearranges_spatial_to_channels():
    x = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    layer = ReOrg()
    y = layer(x)

    assert y.shape == (1, 4, 2, 2)
    assert torch.equal(y[:, 0], x[:, 0, ::2, ::2])
    assert torch.equal(y[:, 1], x[:, 0, 1::2, ::2])
    assert torch.equal(y[:, 2], x[:, 0, ::2, 1::2])
    assert torch.equal(y[:, 3], x[:, 0, 1::2, 1::2])
