import onnx
import torch
from torch import nn

from lightning_yolo.initialization import detection_classprob_bias
from lightning_yolo.torch_networks import YOLOXHead
from lightning_yolo.yolo_module import YOLO


def test_yolo_to_onnx(tmp_path):
    output_path = tmp_path / "yolov4-tiny.onnx"
    model = YOLO(architecture="yolov4-tiny", num_classes=2)
    model.eval()
    model.to_onnx(
        output_path,
        torch.rand(1, 3, 64, 64),
        input_names=["images"],
        output_names=["detections"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        fallback=False,
        verify=False,
        dynamic_shapes={
            "images": {
                2: torch.export.Dim("height", min=32),
                3: torch.export.Dim("width", min=32),
            }
        },
    )

    assert output_path.is_file()
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    assert onnx_model.graph.input[0].name == "images"
    image_shape = onnx_model.graph.input[0].type.tensor_type.shape.dim
    assert image_shape[2].dim_param == "height"
    assert image_shape[3].dim_param == "width"


def test_yolov4_bias_init() -> None:
    module = YOLO(architecture="yolov4", num_classes=2)
    classprob_bias = detection_classprob_bias(2)
    output_convs = [
        conv
        for name, conv in module.network.named_modules()
        if isinstance(conv, nn.Conv2d) and any(part.startswith("outputs_") for part in name.split("."))
    ]

    assert output_convs
    for conv in output_convs:
        assert conv.bias is not None
        bias = conv.bias.view(-1, 7)
        assert torch.allclose(bias[:, :4], torch.zeros_like(bias[:, :4]))
        assert torch.all(bias[:, 4] < 0)
        assert torch.allclose(bias[:, 5:], torch.full_like(bias[:, 5:], classprob_bias))


def test_yolox_bias_init() -> None:
    module = YOLO(architecture="yolox-tiny", num_classes=2)
    classprob_bias = detection_classprob_bias(2)
    heads = [head for head in module.network.modules() if isinstance(head, YOLOXHead)]

    assert heads
    for head in heads:
        assert head.box.bias is not None
        assert head.confidence.bias is not None
        classprob_output = head.classprob[-1]
        assert isinstance(classprob_output, nn.Conv2d)
        assert classprob_output.bias is not None
        assert torch.allclose(head.box.bias, torch.zeros_like(head.box.bias))
        assert torch.all(head.confidence.bias < 0)
        assert torch.allclose(classprob_output.bias, torch.full_like(classprob_output.bias, classprob_bias))
