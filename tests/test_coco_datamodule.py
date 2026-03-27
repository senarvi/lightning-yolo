import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from lightning_yolo.coco_datamodule import (
    COCODetectionDataModule,
    COCODetectionDataset,
    collate_fn,
    convert_annotations,
)


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
                        "category_id": 3,
                        "bbox": [1.0, 1.0, 2.0, 2.0],
                        "area": 4.0,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 3, "name": "category-3"}],
            }
        ),
        encoding="utf-8",
    )

    return image_dir, annotation_path


def test_convert_annotations() -> None:
    annotations = [
        {"bbox": [10, 10, 20, 5], "category_id": 3, "iscrowd": 0},
        {"bbox": [-5, -5, 8, 8], "category_id": 7, "iscrowd": 0},
        {"bbox": [0, 0, 1, 1], "category_id": 99, "iscrowd": 1},
        {"bbox": [5, 5, 0, 10], "category_id": 8, "iscrowd": 0},
    ]
    target = convert_annotations(annotations=annotations, width=20, height=20, include_crowd=False)
    expected_boxes = torch.tensor(
        [
            [10.0, 10.0, 20.0, 15.0],
            [0.0, 0.0, 3.0, 3.0],
        ]
    )
    expected_labels = torch.tensor([3, 7])

    torch.testing.assert_close(target["boxes"], expected_boxes)
    torch.testing.assert_close(target["labels"], expected_labels)


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
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=1.0),
        ]
    )
    dataset = COCODetectionDataset(
        image_dir=image_dir,
        ann_file=annotation_path,
        transforms=transforms,
    )
    image, target = dataset[0]

    assert image.shape == (3, 6, 10)
    torch.testing.assert_close(target["boxes"], torch.tensor([[7.0, 1.0, 9.0, 3.0]]))
    torch.testing.assert_close(target["labels"], torch.tensor([3]))


def test_coco_detection_datamodule(tmp_path: Path) -> None:
    image_dir, annotation_path = create_coco_data(tmp_path)
    datamodule = COCODetectionDataModule(tmp_path, image_size=(12, 20))
    dataset = COCODetectionDataset(
        image_dir=image_dir,
        ann_file=annotation_path,
        transforms=datamodule.test_transforms,
    )
    image, target = dataset[0]

    assert image.shape == (3, 12, 20)
    torch.testing.assert_close(target["boxes"], torch.tensor([[2.0, 2.0, 6.0, 6.0]]))
