import torch
from torch import Tensor, nn


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
