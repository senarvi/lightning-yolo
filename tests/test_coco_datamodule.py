import json
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch
from lightning.pytorch import Trainer

from lightning_yolo.batching import collate_packed_batch
from lightning_yolo.coco_datamodule import (
    CloseMosaic,
    COCODetectionDataModule,
    COCODetectionDataset,
    convert_annotations,
    read_coco_index,
)


def create_coco_data(data_dir: Path) -> tuple[Path, Path]:
    image_dir = data_dir / "train2017"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "000000000001.jpg"
    cv2.imwrite(str(image_path), np.zeros((6, 10, 3), dtype=np.uint8))

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
        {
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "segmentation": [[2.0, 1.0, 9.0, 1.0, 9.0, 7.0, 2.0, 7.0]],
            "category_id": 67,
            "iscrowd": 0,
        },
        {
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "segmentation": [[2.0, 1.0, 9.0, 1.0, 9.0, 7.0, 2.0, 7.0]],
            "category_id": 67,
            "iscrowd": 0,
        },
        {
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "segmentation": [
                [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.0, 4.0],
                [6.0, 5.0, 8.0, 5.0, 8.0, 9.0, 6.0, 9.0],
            ],
            "category_id": 65,
            "iscrowd": 0,
        },
        {
            "bbox": [4.0, 6.0, 3.0, 5.0],
            "segmentation": [],
            "category_id": 65,
            "iscrowd": 0,
        },
    ]
    category_id_to_label = {8: 0, 65: 1, 67: 2, 70: 3}
    boxes, labels = convert_annotations(
        annotations=annotations,
        width=20,
        height=20,
        category_id_to_label=category_id_to_label,
        include_crowd=False,
    )
    expected_boxes = np.array(
        [
            [10.0, 10.0, 20.0, 15.0],
            [0.0, 0.0, 3.0, 3.0],
            [2.0, 1.0, 9.0, 7.0],
            [1.0, 2.0, 8.0, 9.0],
            [4.0, 6.0, 7.0, 11.0],
        ],
        dtype=np.float32,
    )
    expected_labels = np.array([1, 2, 2, 1, 1], dtype=np.int64)

    np.testing.assert_allclose(boxes, expected_boxes)
    np.testing.assert_array_equal(labels, expected_labels)


def test_read_coco_index(tmp_path: Path) -> None:
    image_dir, annotation_path = create_coco_data(tmp_path)

    index = read_coco_index(image_dir, annotation_path, include_crowd=False)

    assert len(index) == 1
    assert index.image_paths[0] == image_dir / "000000000001.jpg"
    assert index.image_sizes[0] == (10, 6)
    np.testing.assert_allclose(index.boxes[0], np.array([[1.0, 1.0, 3.0, 3.0]], dtype=np.float32))
    np.testing.assert_array_equal(index.labels[0], np.array([1], dtype=np.int64))


def test_collate_packed_batch() -> None:
    image1 = torch.zeros((3, 10, 12), dtype=torch.uint8)
    image2 = torch.zeros((3, 10, 12), dtype=torch.uint8)
    target1 = {"boxes": torch.zeros((1, 4)), "labels": torch.tensor([1])}
    target2 = {
        "boxes": torch.zeros((0, 4)),
        "labels": torch.empty((0,), dtype=torch.int64),
    }
    images, targets = collate_packed_batch([(image1, target1), (image2, target2)])

    assert isinstance(images, torch.Tensor)
    assert images.shape == (2, 3, 10, 12)
    assert images.dtype == torch.uint8
    assert targets["boxes"].shape == (1, 4)
    assert torch.equal(targets["labels"], torch.tensor([1]))
    assert torch.equal(targets["batch_indices"], torch.tensor([0]))
    assert targets["counts"] == [1, 0]


def test_coco_detection_dataset(tmp_path: Path) -> None:
    image_dir, annotation_path = create_coco_data(tmp_path)
    dataset = COCODetectionDataset(
        image_dir=image_dir,
        ann_file=annotation_path,
        image_size=(20, 12),
        training=False,
    )
    image, target = dataset[0]

    assert image.shape == (3, 12, 20)
    assert image.dtype == torch.uint8
    torch.testing.assert_close(target["boxes"], torch.tensor([[2.0, 2.0, 6.0, 6.0]]))
    assert torch.equal(target["labels"], torch.tensor([1]))

    resized_image, resized_boxes = dataset._fit_within_output(
        np.zeros((5, 7, 3), dtype=np.uint8),
        np.array([[0.0, 0.0, 7.0, 5.0]], dtype=np.float32),
    )
    assert resized_image.shape == (12, 17, 3)
    np.testing.assert_array_equal(resized_boxes, np.array([[0.0, 0.0, 17.0, 12.0]], dtype=np.float32))


def test_coco_detection_dataset_training(tmp_path: Path) -> None:
    image_dir, annotation_path = create_coco_data(tmp_path)
    dataset = COCODetectionDataset(
        image_dir=image_dir,
        ann_file=annotation_path,
        image_size=(16, 16),
        training=True,
        cache_size=4,
    )

    image, target = dataset[0]

    assert image.shape == (3, 16, 16)
    assert image.dtype == torch.uint8
    assert target["boxes"].shape[1:] == (4,)
    assert target["labels"].ndim == 1


def test_coco_detection_dataset_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_dir, annotation_path = create_coco_data(tmp_path)
    dataset = COCODetectionDataset(
        image_dir=image_dir,
        ann_file=annotation_path,
        image_size=(20, 12),
        training=False,
        cache_size=1,
    )
    first = dataset.load_sample(0)
    first.labels[0] = 99
    imread = Mock(side_effect=AssertionError("cached sample was decoded again"))
    monkeypatch.setattr(cv2, "imread", imread)

    cached = dataset.load_sample(0)

    assert cached.image.shape == (12, 20, 3)
    np.testing.assert_array_equal(cached.labels, np.array([1]))
    imread.assert_not_called()


def test_coco_detection_datamodule_setup(tmp_path: Path) -> None:
    image_dir, annotation_path = create_coco_data(tmp_path)
    (tmp_path / "val2017").mkdir(exist_ok=True)
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    for name in ("instances_train2017.json", "instances_val2017.json"):
        (annotations_dir / name).write_text(annotation_path.read_text(encoding="utf-8"), encoding="utf-8")
    # The images live under train2017; reuse them for both splits in this fixture.
    (tmp_path / "val2017" / "000000000001.jpg").write_bytes((image_dir / "000000000001.jpg").read_bytes())

    datamodule = COCODetectionDataModule(tmp_path, batch_size=1, num_workers=0, image_size=(16, 16))
    datamodule.setup("fit")

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    image, target = datamodule.train_dataset[0]
    assert image.shape[0] == 3
    assert target["boxes"].shape[1:] == (4,)


@pytest.mark.parametrize(("num_workers", "expected_prefetch_factor"), [(0, None), (1, 4)])
def test_coco_detection_dataloader_prefetch_factor(
    tmp_path: Path, num_workers: int, expected_prefetch_factor: int | None
) -> None:
    image_dir, annotation_path = create_coco_data(tmp_path)
    dataset = COCODetectionDataset(
        image_dir=image_dir,
        ann_file=annotation_path,
        image_size=(20, 12),
        training=False,
    )
    datamodule = COCODetectionDataModule(tmp_path, num_workers=num_workers, persistent_workers=False)
    datamodule.train_dataset = dataset

    dataloader = datamodule.train_dataloader()

    assert dataloader.prefetch_factor == expected_prefetch_factor


def test_close_mosaic(tmp_path: Path) -> None:
    image_dir, annotation_path = create_coco_data(tmp_path)
    dataset = COCODetectionDataset(
        image_dir=image_dir,
        ann_file=annotation_path,
        image_size=(16, 16),
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

    assert dataset._mosaic_enabled.is_shared()
    trainer.fit_loop.epoch_progress.current.completed = 89
    callback.on_train_epoch_start(trainer, Mock())
    assert dataset.mosaic_enabled
    trainer.fit_loop.epoch_progress.current.completed = 90
    callback.on_train_epoch_start(trainer, Mock())
    assert not dataset.mosaic_enabled

    image, target = dataset[0]
    assert image.shape == (3, 16, 16)
    assert target["boxes"].shape[1:] == (4,)
