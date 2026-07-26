import onnx
import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from lightning_yolo.batching import pack_targets
from lightning_yolo.config import MatchingConfig
from lightning_yolo.initialization import detection_classprob_bias
from lightning_yolo.torch_networks import YOLOXHead
from lightning_yolo.yolo import YOLO


@pytest.mark.parametrize("predict_confidence", [False, True])
def test_yolo_forward(predict_confidence: bool) -> None:
    module = YOLO(
        architecture="yolov8n",
        num_classes=3,
        matching=MatchingConfig(algorithm="tal"),
        predict_confidence=predict_confidence,
    )
    images = torch.rand(1, 3, 64, 64)
    targets = pack_targets([{"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.int64)}])

    _, losses = module(images, targets)

    # Finite losses with empty targets.
    assert torch.isfinite(losses).all()
    assert losses.max() < 100

    module.eval()
    uint8_images = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    float_images = uint8_images.to(torch.float32).div(255.0)
    targets = pack_targets(
        [
            {"boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0]]), "labels": torch.tensor([0])},
            {"boxes": torch.tensor([[8.0, 8.0, 24.0, 24.0]]), "labels": torch.tensor([1])},
        ]
    )

    torch.manual_seed(0)
    uint8_detections, uint8_losses = module(uint8_images, targets)
    torch.manual_seed(0)
    float_detections, float_losses = module(float_images, targets)

    torch.testing.assert_close(uint8_detections, float_detections)
    torch.testing.assert_close(uint8_losses, float_losses)


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


def test_yolo_init_yolov4_bias() -> None:
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


def test_yolo_init_yolox_bias() -> None:
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


def test_yolo_get_optimizer() -> None:
    module = YOLO(architecture="yolov8n", num_classes=2)

    optimizer = module._get_optimizer()

    assert isinstance(optimizer, SGD)
    assert optimizer.defaults["momentum"] == 0.9
    assert optimizer.defaults["nesterov"] is True
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.0, 0.0005]


def test_yolo_get_lr_scheduler() -> None:
    module = YOLO(architecture="yolov8n", num_classes=2)
    optimizer = module._get_optimizer()
    total_steps = 2000
    total_epochs = 10

    scheduler = module._get_lr_scheduler(optimizer, total_steps, total_epochs)

    assert isinstance(scheduler, SequentialLR)
    assert all(isinstance(stage, LinearLR) for stage in scheduler._schedulers)

    base_lr = module.hparams["lr"]
    warmup_steps = round(module.hparams["warmup_epochs"] * total_steps / total_epochs)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr / warmup_steps)

    optimizer.step()
    for _ in range(total_steps - 1):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * module.hparams["final_lr_multiplier"])
