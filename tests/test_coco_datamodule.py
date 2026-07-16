import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch import Trainer
from PIL import Image
from torchvision import tv_tensors

from lightning_yolo.coco_datamodule import (
    CloseMosaic,
    COCODetectionDataModule,
    COCODetectionDataset,
    collate_fn,
    convert_annotations,
)
from lightning_yolo.transforms import LetterBox, MixUp, Mosaic


def create_coco_data(data_dir: Path) -> tuple[Path, Path]:
    image_dir = data_dir / "train2017"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "000000000001.jpg"
    Image.new("RGB", (10, 6)).save(image_path)

    annotation_path = data_dir / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": image_path.name, "width": 10, "height": 6}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 67,
                        "bbox": [1.0, 1.0, 2.0, 2.0],
                        "area": 4.0,
                        "iscrowd": 0,
                    }
                ],
                "categories": [
                    {"id": 65, "name": "bed"},
                    {"id": 67, "name": "dining table"},
                    {"id": 70, "name": "toilet"},
                ],
            }
        ),
        encoding="utf-8",
    )

    return image_dir, annotation_path


def test_convert_annotations() -> None:
    annotations = [
        {"bbox": [10, 10, 20, 5], "category_id": 65, "iscrowd": 0},
        {"bbox": [-5, -5, 8, 8], "category_id": 67, "iscrowd": 0},
        {"bbox": [0, 0, 1, 1], "category_id": 70, "iscrowd": 1},
        {"bbox": [5, 5, 0, 10], "category_id": 8, "iscrowd": 0},
    ]
    category_id_to_label = {8: 0, 65: 1, 67: 2, 70: 3}
    target = convert_annotations(
        annotations=annotations,
        width=20,
        height=20,
        category_id_to_label=category_id_to_label,
        include_crowd=False,
    )
    expected_boxes = torch.tensor(
        [
            [10.0, 10.0, 20.0, 15.0],
            [0.0, 0.0, 3.0, 3.0],
        ]
    )
    expected_labels = torch.tensor([1, 2])

    torch.testing.assert_close(target["boxes"], expected_boxes)
    assert torch.equal(target["labels"], expected_labels)


def test_collate_fn() -> None:
    image1 = torch.zeros((3, 10, 12))
    image2 = torch.zeros((3, 20, 18))
    target1 = {"boxes": torch.zeros((1, 4)), "labels": torch.tensor([1])}
    target2 = {
        "boxes": torch.zeros((0, 4)),
        "labels": torch.empty((0,), dtype=torch.int64),
    }
    images, targets = collate_fn([(image1, target1), (image2, target2)])

    assert isinstance(images, list)
    assert isinstance(targets, list)
    assert images[0].shape == (3, 10, 12)
    assert images[1].shape == (3, 20, 18)
    assert targets[0]["boxes"].shape == (1, 4)
    assert targets[1]["boxes"].shape == (0, 4)


def test_coco_detection_dataset(tmp_path: Path) -> None:
    image_dir, annotation_path = create_coco_data(tmp_path)
    dataset = COCODetectionDataset(
        image_dir=image_dir,
        ann_file=annotation_path,
        image_size=(12, 20),
        training=False,
    )
    image, target = dataset[0]

    assert image.shape == (3, 12, 20)
    torch.testing.assert_close(target["boxes"], torch.tensor([[2.0, 2.0, 6.0, 6.0]]))
    assert torch.equal(target["labels"], torch.tensor([1]))


def test_coco_detection_datamodule(tmp_path: Path) -> None:
    image_dir, annotation_path = create_coco_data(tmp_path)
    COCODetectionDataModule(tmp_path, image_size=(12, 20))
    dataset = COCODetectionDataset(
        image_dir=image_dir,
        ann_file=annotation_path,
        image_size=(12, 20),
        training=False,
    )
    image, target = dataset[0]

    assert image.shape == (3, 12, 20)
    torch.testing.assert_close(target["boxes"], torch.tensor([[2.0, 2.0, 6.0, 6.0]]))


def test_close_mosaic(tmp_path: Path) -> None:
    image_dir, annotation_path = create_coco_data(tmp_path)
    dataset = COCODetectionDataset(
        image_dir=image_dir,
        ann_file=annotation_path,
        image_size=(12, 20),
        training=True,
    )
    datamodule = COCODetectionDataModule(tmp_path)
    datamodule.train_dataset = dataset
    trainer = Trainer(
        max_epochs=100,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.datamodule = datamodule
    callback = CloseMosaic(epochs=10)

    assert dataset.mosaic_enabled.is_shared()
    trainer.fit_loop.epoch_progress.current.completed = 89
    callback.on_train_epoch_start(trainer, Mock())
    assert bool(dataset.mosaic_enabled)
    trainer.fit_loop.epoch_progress.current.completed = 90
    callback.on_train_epoch_start(trainer, Mock())
    assert not bool(dataset.mosaic_enabled)

    image, target = dataset[0]
    assert image.shape == (3, 12, 20)
    assert target["boxes"].shape[1:] == (4,)


def test_mosaic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "randint", lambda *_args, **_kwargs: torch.tensor(8))
    boxes = ([5.0, 1.0, 7.0, 3.0], [1.0, 1.0, 3.0, 3.0]) * 2
    samples = [
        (
            tv_tensors.Image(torch.zeros((3, 4, 8), dtype=torch.uint8)),
            {
                "boxes": tv_tensors.BoundingBoxes([box], format="XYXY", canvas_size=(4, 8)),
                "labels": torch.tensor([label]),
            },
        )
        for label, box in enumerate(boxes)
    ]

    image, target = Mosaic((8, 8))(samples)

    assert image.shape == (3, 8, 8)
    torch.testing.assert_close(
        target["boxes"],
        torch.tensor(
            [
                [1.0, 1.0, 3.0, 3.0],
                [5.0, 1.0, 7.0, 3.0],
                [1.0, 5.0, 3.0, 7.0],
                [5.0, 5.0, 7.0, 7.0],
            ]
        ),
    )
    assert torch.equal(target["labels"], torch.arange(4))


def test_letterbox() -> None:
    image = tv_tensors.Image(torch.zeros((3, 10, 10), dtype=torch.uint8))
    target = {
        "boxes": tv_tensors.BoundingBoxes([[1.0, 1.0, 3.0, 3.0]], format="XYXY", canvas_size=(10, 10)),
        "labels": torch.tensor([1]),
    }

    image, target = LetterBox((13, 20))(image, target)

    assert image.shape == (3, 13, 20)
    assert torch.all(image[:, :, :3] == 114)
    assert torch.all(image[:, :, -4:] == 114)
    torch.testing.assert_close(target["boxes"], torch.tensor([[4.3, 1.3, 6.9, 3.9]]))
    assert target["boxes"].canvas_size == (13, 20)


def test_mixup(monkeypatch: pytest.MonkeyPatch) -> None:
    beta = Mock()
    beta.return_value.sample.return_value = torch.tensor(0.25)
    monkeypatch.setattr(torch.distributions, "Beta", beta)
    samples = []
    for label, value in enumerate((0, 200)):
        image = tv_tensors.Image(torch.full((3, 4, 4), value, dtype=torch.uint8))
        target = {
            "boxes": tv_tensors.BoundingBoxes([[1.0, 1.0, 3.0, 3.0]], format="XYXY", canvas_size=(4, 4)),
            "labels": torch.tensor([label]),
        }
        samples.append((image, target))

    image, target = MixUp(alpha=2.0)(samples)

    assert image.shape == (3, 4, 4)
    assert image.dtype == torch.uint8
    assert image[0, 0, 0] == 150
    assert target["boxes"].shape == (2, 4)
    assert torch.equal(target["labels"], torch.tensor([0, 1]))
    beta.assert_called_once_with(2.0, 2.0)
