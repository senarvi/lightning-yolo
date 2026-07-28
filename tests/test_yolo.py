from pathlib import Path

import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, SequentialLR

import lightning_yolo.yolo as yolo_module
from lightning_yolo.batching import pack_targets
from lightning_yolo.initialization import detection_classprob_bias
from lightning_yolo.torch_networks import YOLOXHead
from lightning_yolo.types import DetectionLoss
from lightning_yolo.yolo import YOLO


def test_yolo_forward() -> None:
    module = YOLO(
        architecture="yolov8n",
        num_classes=3,
    )
    module.eval()
    uint8_images = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    float_images = uint8_images.to(torch.float32).div(255.0)

    torch.manual_seed(0)
    uint8_detections = module(uint8_images)
    torch.manual_seed(0)
    float_detections = module(float_images)

    assert uint8_detections.shape == (2, 84, 8)
    torch.testing.assert_close(uint8_detections, float_detections)


def test_yolo_forward_with_losses() -> None:
    module = YOLO(
        architecture="yolov8n",
        num_classes=3,
    )
    images = torch.rand(1, 3, 64, 64)
    targets = pack_targets([{"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.int64)}])

    detections, losses = module._forward_with_losses(images, targets)

    assert detections.shape == (1, 84, 8)
    assert losses.names == ("overlap", "classification", "dfl")
    assert torch.isfinite(losses.values).all()
    assert losses.values.max() < 100


def test_yolo_forward_with_losses_multilabel() -> None:
    module = YOLO(architecture="yolov8n", num_classes=3)
    images = torch.rand(1, 3, 64, 64)
    # A boolean class mask assigns multiple classes to one box, exercising the multi-label training path.
    targets = pack_targets(
        [{"boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]]), "labels": torch.tensor([[True, False, True]])}]
    )

    detections, losses = module._forward_with_losses(images, targets)

    assert detections.shape == (1, 84, 8)
    assert losses.names == ("overlap", "classification", "dfl")
    assert torch.isfinite(losses.values).all()


def test_yolo_log_losses() -> None:
    module = YOLO(architecture="yolov8n", num_classes=3)
    logged: dict[str, tuple[torch.Tensor, dict[str, object]]] = {}

    def log(name: str, value: torch.Tensor, **kwargs) -> None:
        logged[name] = (value, kwargs)

    module.log = log  # type: ignore[assignment]
    losses = DetectionLoss(values=torch.tensor([2.0, 3.0, 5.0]), names=("overlap", "classification", "dfl"))

    module._log_losses("val", losses, sync_dist=True, batch_size=4)

    torch.testing.assert_close(logged["val/overlap_loss"][0], torch.tensor(2.0))
    torch.testing.assert_close(logged["val/classification_loss"][0], torch.tensor(3.0))
    torch.testing.assert_close(logged["val/dfl_loss"][0], torch.tensor(5.0))
    torch.testing.assert_close(logged["val/total_loss"][0], torch.tensor(10.0))
    assert all(kwargs["sync_dist"] is True for _, kwargs in logged.values())
    assert all(kwargs["batch_size"] == 4 for _, kwargs in logged.values())


def test_yolo_process_detections(monkeypatch: pytest.MonkeyPatch) -> None:
    module = YOLO(architecture="yolov8n", num_classes=2, detections_per_image=2)
    captured: dict[str, torch.Tensor | float] = {}

    def fake_batched_nms(
        boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, nms_threshold: float
    ) -> torch.Tensor:
        captured["boxes"] = boxes
        captured["scores"] = scores
        captured["labels"] = labels
        captured["nms_threshold"] = nms_threshold
        return torch.arange(scores.numel(), device=scores.device)

    monkeypatch.setattr(yolo_module, "batched_nms", fake_batched_nms)
    detections = torch.tensor(
        [
            [
                [10.0, 20.0, 30.0, 40.0, 1.0, 0.2, 0.7],
                [50.0, 60.0, 70.0, 80.0, 0.75, 0.8, 0.1],
                [90.0, 100.0, 110.0, 120.0, 1.0, 0.6, 0.6],
            ]
        ]
    )

    processed = module.process_detections(detections, confidence_threshold=0.25)

    expected_nms_boxes = torch.tensor(
        [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0], [90.0, 100.0, 110.0, 120.0], [90.0, 100.0, 110.0, 120.0]]
    )
    torch.testing.assert_close(captured["boxes"], expected_nms_boxes)
    torch.testing.assert_close(captured["scores"], torch.tensor([0.7, 0.6, 0.6, 0.6]))
    torch.testing.assert_close(captured["labels"], torch.tensor([1, 0, 0, 1]))
    assert captured["nms_threshold"] == module.nms_threshold
    torch.testing.assert_close(processed[0]["boxes"], expected_nms_boxes[:2])
    torch.testing.assert_close(processed[0]["scores"], torch.tensor([0.7, 0.6]))
    torch.testing.assert_close(processed[0]["labels"], torch.tensor([1, 0]))


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


@pytest.mark.parametrize(
    ("architecture", "expected_num_detections"),
    [
        ("yolov4-tiny", 252),
        ("yolov8n", 84),
    ],
)
def test_yolo_to_onnx(
    tmp_path: Path,
    architecture: str,
    expected_num_detections: int,
) -> None:
    output_path = tmp_path / f"{architecture}.onnx"
    model = YOLO(architecture=architecture, num_classes=2)
    model.eval()
    images = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        eager_detections = model(images)
    assert eager_detections.shape == (1, expected_num_detections, 7)

    model.to_onnx(
        output_path,
        images,
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
    assert [output.name for output in onnx_model.graph.output] == ["detections"]
    output_shape = onnx_model.graph.output[0].type.tensor_type.shape.dim
    assert output_shape[2].dim_value == 7

    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    (exported_detections,) = session.run(["detections"], {"images": images.numpy()})

    torch.testing.assert_close(torch.from_numpy(exported_detections), eager_detections, rtol=1e-4, atol=1e-4)
