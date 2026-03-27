import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import torch
from lightning.pytorch import LightningDataModule
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2

from .types import BATCH, IMAGES, TARGETS, TargetDict

TransformFn = Callable[
    [Image.Image, dict[str, Tensor | tv_tensors.BoundingBoxes]],
    tuple[Image.Image, dict[str, Tensor | tv_tensors.BoundingBoxes]],
]


def collate_fn(batch: list[tuple[Tensor, TargetDict]]) -> BATCH:
    """Collate a batch of image-target pairs for detection training.

    Args:
        batch: List of ``(image, target)`` samples, where each target may contain a variable number of bounding boxes.

    Returns:
        A tuple ``(images, targets)`` where both values are lists aligned by sample index.

    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def convert_annotations(
    annotations: list[dict],
    width: int,
    height: int,
    include_crowd: bool = False,
) -> dict[str, Any]:
    """Convert COCO annotations into a target dictionary in Torchvision v2 transforms format.

    Args:
        annotations: Raw COCO annotations for one image.
        width: Image width in pixels.
        height: Image height in pixels.
        include_crowd: Whether annotations marked as crowd should be included.

    Returns:
        A dictionary containing:
        - ``boxes``: Tensor of shape ``(N, 4)`` in ``XYXY`` pixel coordinates.
        - ``labels``: Tensor of shape ``(N,)`` with class ids.

    """
    boxes: list[list[float]] = []
    labels: list[int] = []

    for annotation in annotations:
        if not include_crowd and annotation.get("iscrowd", 0):
            continue

        x, y, w, h = annotation["bbox"]
        if w <= 0 or h <= 0:
            continue

        x1 = max(0.0, min(float(width), float(x)))
        y1 = max(0.0, min(float(height), float(y)))
        x2 = max(0.0, min(float(width), float(x + w)))
        y2 = max(0.0, min(float(height), float(y + h)))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])
        labels.append(int(annotation["category_id"]))

    if boxes:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
    else:
        boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
        labels_tensor = torch.empty((0,), dtype=torch.int64)

    return {
        "boxes": tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(height, width)),
        "labels": labels_tensor,
    }


class COCODetectionDataset(CocoDetection):
    """Torchvision COCO wrapper that returns tensors and YOLO targets.

    Args:
        image_dir: Directory containing image files.
        ann_file: Path to the COCO annotation JSON file.
        transforms: Optional transform pipeline receiving ``(image, target)`` and returning transformed
            ``(image, target)``.
        include_crowd: Whether to keep annotations marked as "crowd". These are boxes the contain multiple objects.

    """

    def __init__(
        self,
        image_dir: Path,
        ann_file: Path,
        transforms: TransformFn,
        include_crowd: bool = False,
    ) -> None:
        # Keep VisionDataset transforms disabled. The transforms will be applied after converting raw COCO annotations
        # into the correct format.
        super().__init__(root=str(image_dir), annFile=str(ann_file), transforms=None)
        self.sample_transforms = transforms
        self.include_crowd = include_crowd

    def __getitem__(self, index: int) -> tuple[Tensor, TargetDict]:
        """Loads one sample and converts it to tensor-based detection targets.

        Args:
            index: Zero-based sample index.

        Returns:
            A tuple ``(image, target)`` where ``target`` contains ``boxes`` and ``labels`` tensors.

        """
        image = self._load_image(self.ids[index])
        annotations = self._load_target(self.ids[index])
        width, height = image.size
        target = convert_annotations(annotations, width=width, height=height, include_crowd=self.include_crowd)
        image, target = self.sample_transforms(image, target)

        image_tensor = torch.as_tensor(image, dtype=torch.float32)
        boxes_tensor = torch.as_tensor(target["boxes"], dtype=torch.float32)
        labels_tensor = torch.as_tensor(target["labels"], dtype=torch.int64)
        return image_tensor, {"boxes": boxes_tensor, "labels": labels_tensor}


class COCODetectionDataModule(LightningDataModule):
    """Lightning DataModule for COCO 2017 object detection.

    Args:
        data_dir: Root directory where COCO data is stored or downloaded.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for each DataLoader.
        image_size: Target image size as ``(height, width)``.
        pin_memory: Whether DataLoaders should pin memory.
        persistent_workers: Whether worker processes persist across epochs.
        include_crowd: Whether to include crowd annotations.
        train_transforms: Optional custom transform pipeline for training.
        val_transforms: Optional custom transform pipeline for validation/test.

    """

    _DOWNLOADS = (
        ("train2017", "http://images.cocodataset.org/zips/train2017.zip"),
        ("val2017", "http://images.cocodataset.org/zips/val2017.zip"),
        ("annotations", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"),
    )

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 16,
        num_workers: int = 8,
        image_size: tuple[int, int] = (640, 640),
        pin_memory: bool = True,
        persistent_workers: bool = True,
        include_crowd: bool = False,
        train_transforms: TransformFn | None = None,
        val_transforms: TransformFn | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_transforms = train_transforms or v2.Compose(
            [
                v2.ToImage(),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomPhotometricDistort(p=0.8),
                v2.Resize(size=image_size),
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.test_transforms = val_transforms or v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=image_size),
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        self.train_dataset: COCODetectionDataset | None = None
        self.val_dataset: COCODetectionDataset | None = None
        self.test_dataset: COCODetectionDataset | None = None

    def prepare_data(self) -> None:
        """Downloads and extracts the COCO files if they are missing.

        Args:
            self: Data module instance.

        """
        data_dir = Path(self.hparams.data_dir)  # type: ignore[attr-defined]
        data_dir.mkdir(parents=True, exist_ok=True)

        markers = {
            "train2017": data_dir / "train2017",
            "val2017": data_dir / "val2017",
            "annotations": data_dir / "annotations" / "instances_train2017.json",
        }

        for name, url in self._DOWNLOADS:
            if markers[name].exists():
                continue

            archive_path = data_dir / f"{name}.zip"
            if not archive_path.exists():
                urlretrieve(url, archive_path)  # noqa: S310
            with zipfile.ZipFile(archive_path, "r") as archive_file:
                archive_file.extractall(data_dir)

    def setup(self, stage: str | None = None) -> None:
        """Creates datasets for the requested Lightning stage.

        Args:
            stage: Stage hint from Lightning (``fit``, ``validate``, ``test``),
                or ``None`` to prepare all relevant datasets.

        """
        data_dir = Path(self.hparams.data_dir)  # type: ignore[attr-defined]
        include_crowd = self.hparams.include_crowd  # type: ignore[attr-defined]

        train_images = data_dir / "train2017"
        val_images = data_dir / "val2017"
        train_ann = data_dir / "annotations" / "instances_train2017.json"
        val_ann = data_dir / "annotations" / "instances_val2017.json"

        if stage in (None, "fit"):
            self.train_dataset = COCODetectionDataset(
                image_dir=train_images,
                ann_file=train_ann,
                transforms=self.train_transforms,
                include_crowd=include_crowd,
            )
            self.val_dataset = COCODetectionDataset(
                image_dir=val_images,
                ann_file=val_ann,
                transforms=self.test_transforms,
                include_crowd=include_crowd,
            )

        if stage in (None, "validate"):
            self.val_dataset = COCODetectionDataset(
                image_dir=val_images,
                ann_file=val_ann,
                transforms=self.test_transforms,
                include_crowd=include_crowd,
            )

        if stage in (None, "test"):
            self.test_dataset = COCODetectionDataset(
                image_dir=val_images,
                ann_file=val_ann,
                transforms=self.test_transforms,
                include_crowd=include_crowd,
            )

    def train_dataloader(self) -> DataLoader[tuple[IMAGES, TARGETS]]:
        """Builds the training DataLoader.

        Returns:
            A DataLoader yielding batches of training images and targets.

        """
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting train_dataloader().")

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore[attr-defined]
            shuffle=True,
            num_workers=self.hparams.num_workers,  # type: ignore[attr-defined]
            pin_memory=self.hparams.pin_memory,  # type: ignore[attr-defined]
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,  # type: ignore[attr-defined]
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader[tuple[IMAGES, TARGETS]]:
        """Builds the validation DataLoader.

        Returns:
            A DataLoader yielding batches of validation images and targets.

        """
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit' or 'validate') before requesting val_dataloader().")

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore[attr-defined]
            shuffle=False,
            num_workers=self.hparams.num_workers,  # type: ignore[attr-defined]
            pin_memory=self.hparams.pin_memory,  # type: ignore[attr-defined]
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,  # type: ignore[attr-defined]
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader[tuple[IMAGES, TARGETS]]:
        """Builds the test DataLoader.

        Returns:
            A DataLoader yielding batches of test images and targets.

        """
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before requesting test_dataloader().")

        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore[attr-defined]
            shuffle=False,
            num_workers=self.hparams.num_workers,  # type: ignore[attr-defined]
            pin_memory=self.hparams.pin_memory,  # type: ignore[attr-defined]
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,  # type: ignore[attr-defined]
            collate_fn=collate_fn,
        )
