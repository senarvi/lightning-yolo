import onnx
import torch

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
