import torch

from lightning_yolo.layers import Conv, MaxPool, Mish, ReOrg, RouteLayer, ShortcutLayer


def test_conv() -> None:
    layer = Conv(3, 8, kernel_size=3, stride=2, activation="silu", norm="batchnorm")
    x = torch.randn(2, 3, 11, 11)
    y = layer(x)

    assert y.shape == (2, 8, 5, 5)


def test_max_pool() -> None:
    layer = MaxPool(kernel_size=2, stride=2)
    x = torch.randn(1, 3, 5, 5)
    y = layer(x)

    assert y.shape == (1, 3, 2, 2)


def test_route_layer() -> None:
    outputs = [
        torch.arange(1 * 4 * 2 * 2, dtype=torch.float32).view(1, 4, 2, 2),
        torch.arange(100, 100 + (1 * 4 * 2 * 2), dtype=torch.float32).view(1, 4, 2, 2),
    ]
    layer = RouteLayer(source_layers=[0, 1], num_chunks=2, chunk_idx=1)
    y = layer(outputs)
    expected = torch.cat((outputs[0][:, 2:], outputs[1][:, 2:]), dim=1)

    assert y.shape == (1, 4, 2, 2)
    assert torch.equal(y, expected)


def test_shortcut_layer() -> None:
    outputs = [
        torch.ones(1, 3, 2, 2),
        torch.full((1, 3, 2, 2), 2.0),
        torch.full((1, 3, 2, 2), 3.0),
    ]
    layer = ShortcutLayer(source_layer=0)
    y = layer(outputs)

    assert torch.equal(y, outputs[-1] + outputs[0])


def test_mish() -> None:
    x = torch.tensor([-2.0, 0.0, 3.0])
    layer = Mish()
    y = layer(x)
    expected = x * torch.tanh(torch.nn.functional.softplus(x))

    torch.testing.assert_close(y, expected)


def test_reorg() -> None:
    x = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    layer = ReOrg()
    y = layer(x)

    assert y.shape == (1, 4, 2, 2)
    assert torch.equal(y[:, 0], x[:, 0, ::2, ::2])
    assert torch.equal(y[:, 1], x[:, 0, 1::2, ::2])
    assert torch.equal(y[:, 2], x[:, 0, ::2, 1::2])
    assert torch.equal(y[:, 3], x[:, 0, 1::2, 1::2])
