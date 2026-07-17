import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast
from urllib.request import urlretrieve

import numpy as np
import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2

from .transforms import DetectionSample, LetterBox, MixUp, Mosaic
from .types import BATCH, IMAGES, TARGETS, TargetDict


def collate_fn(batch: list[tuple[Tensor, TargetDict]]) -> BATCH:
    """Collate a batch of image-target pairs for detection training.

    Args:
        batch: List of ``(image, target)`` samples, where each target may contain a variable number of bounding boxes.

    Returns:
        A tuple ``(images, targets)`` where both values are lists aligned by sample index.

    """
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def convert_annotations(
    annotations: list[dict],
    width: int,
    height: int,
    category_id_to_label: Mapping[int, int],
    include_crowd: bool = False,
) -> TargetDict:
    """Convert COCO annotations into a target dictionary in Torchvision v2 transforms format.

    If a polygon segmentation is available, the box is derived from polygon extents, otherwise COCO ``bbox`` is used.
    Duplicate ``(class, box)`` rows are removed.

    Args:
        annotations: Raw COCO annotations for one image.
        width: Image width in pixels.
        height: Image height in pixels.
        category_id_to_label: Mapping from COCO category IDs to zero-based class labels.
        include_crowd: Whether annotations marked as crowd should be included.

    Returns:
        A dictionary containing:
        - ``boxes``: Tensor of shape ``(N, 4)`` in ``XYXY`` pixel coordinates.
        - ``labels``: Tensor of shape ``(N,)`` with zero-based class labels.

    """
    boxes: list[list[float]] = []
    labels: list[int] = []
    seen_rows: set[tuple[int, float, float, float, float]] = set()

    for annotation in annotations:
        if not include_crowd and annotation.get("iscrowd", 0):
            continue

        x, y, w, h = annotation["bbox"]
        if w <= 0 or h <= 0:
            continue

        x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, list) and len(segmentation) > 0:
            polygon_points: list[np.ndarray] = []
            for segment in segmentation:
                if not isinstance(segment, list) or len(segment) < 6:
                    continue
                points = np.asarray(segment, dtype=np.float32).reshape(-1, 2)
                polygon_points.append(points)
            if polygon_points:
                merged_points = np.concatenate(polygon_points, axis=0)
                x_values, y_values = merged_points.T
                x1, y1 = float(x_values.min()), float(y_values.min())
                x2, y2 = float(x_values.max()), float(y_values.max())

        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))

        if x2 <= x1 or y2 <= y1:
            continue

        label = category_id_to_label[int(annotation["category_id"])]
        row = (label, x1, y1, x2, y2)
        if row in seen_rows:
            continue
        seen_rows.add(row)

        boxes.append([x1, y1, x2, y2])
        labels.append(label)

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
        image_size: Output image size as ``(height, width)``.
        training: Whether to apply the fixed training augmentation pipeline.
        include_crowd: Whether to keep annotations marked as "crowd". These are boxes the contain multiple objects.
        translate: Maximum affine translation as a fraction of image width and height.
        scale: Maximum affine scale variation around one.
        mixup: Probability of applying MixUp to a training sample.

    """

    def __init__(
        self,
        image_dir: Path,
        ann_file: Path,
        image_size: tuple[int, int],
        training: bool,
        include_crowd: bool = False,
        translate: float = 0.1,
        scale: float = 0.9,
        mixup: float = 0.1,
    ) -> None:
        # Keep VisionDataset transforms disabled. The transforms will be applied after converting raw COCO annotations
        # into the correct format.
        super().__init__(root=str(image_dir), annFile=str(ann_file), transforms=None)
        if not 0.0 <= translate <= 1.0:
            raise ValueError("translate must be between zero and one.")
        if not 0.0 <= scale < 1.0:
            raise ValueError("scale must be at least zero and less than one.")
        if not 0.0 <= mixup <= 1.0:
            raise ValueError("mixup must be between zero and one.")
        self.image_size = image_size
        self.training = training
        self.include_crowd = include_crowd
        self.mixup_p = mixup
        self.mosaic_enabled = torch.tensor(True).share_memory_()
        self.category_id_to_label = {category_id: label for label, category_id in enumerate(sorted(self.coco.cats))}
        self.mosaic = Mosaic(image_size=image_size, fill=114)
        self.mixup = MixUp(alpha=32.0)
        self.letterbox = LetterBox(image_size=image_size, fill=114)
        self.random_affine = v2.RandomAffine(
            degrees=0.0,
            translate=(translate, translate),
            scale=(1.0 - scale, 1.0 + scale),
            shear=0.0,
            interpolation=v2.InterpolationMode.BILINEAR,
            fill=114,
        )
        self.train_transforms = v2.Compose(
            [
                v2.ColorJitter(brightness=0.4, saturation=0.7, hue=0.015),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ClampBoundingBoxes(),
                v2.SanitizeBoundingBoxes(min_size=2.0, min_area=4.0),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.test_transforms = v2.Compose(
            [
                self.letterbox,
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __getitem__(self, index: int) -> tuple[Tensor, TargetDict]:
        """Loads one sample and converts it to tensor-based detection targets.

        Args:
            index: Zero-based sample index.

        Returns:
            A tuple ``(image, target)`` where ``target`` contains ``boxes`` and ``labels`` tensors.

        """
        if self.training:
            image, target = self._load_augmented_sample(index)
            if bool(self.mosaic_enabled) and torch.rand(()) < self.mixup_p:
                mixup_index = int(torch.randint(len(self), ()).item())
                image, target = self.mixup([(image, target), self._load_augmented_sample(mixup_index)])
            image, target = self.train_transforms(image, target)
        else:
            image, target = self._load_sample(index)
            image, target = self.test_transforms(image, target)

        image_tensor = torch.as_tensor(image, dtype=torch.float32)
        boxes_tensor = torch.as_tensor(target["boxes"], dtype=torch.float32)
        labels_tensor = torch.as_tensor(target["labels"], dtype=torch.int64)
        return image_tensor, {"boxes": boxes_tensor, "labels": labels_tensor}

    def _load_sample(self, index: int) -> DetectionSample:
        """Load and pre-transform one sample without dataset-aware augmentation.

        Args:
            index: Zero-based sample index.

        Returns:
            Tensor image and converted detection target.

        """
        image = self._load_image(self.ids[index])
        annotations = self._load_target(self.ids[index])
        width, height = image.size
        target = convert_annotations(
            annotations,
            width=width,
            height=height,
            category_id_to_label=self.category_id_to_label,
            include_crowd=self.include_crowd,
        )
        return cast(Tensor, v2.functional.to_image(image)), target

    def set_mosaic_enabled(self, enabled: bool) -> None:
        """Enable or disable Mosaic and MixUp augmentation.

        Args:
            enabled: Whether dataset-aware mixing augmentations should be applied.

        """
        self.mosaic_enabled.fill_(enabled)

    def _load_augmented_sample(self, index: int) -> DetectionSample:
        """Load a four-image mosaic and apply random affine augmentation.

        Args:
            index: Index of the primary sample.

        Returns:
            Augmented image-target pair at the configured image size.

        """
        if bool(self.mosaic_enabled):
            auxiliary_indices = torch.randint(len(self), (3,)).tolist()
            samples = [self._load_sample(sample_index) for sample_index in [index, *auxiliary_indices]]
            image, target = self.mosaic(samples)
        else:
            image, target = self.letterbox(*self._load_sample(index))
        return self.random_affine(image, target)


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
        translate: Maximum affine translation as a fraction of image width and height.
        scale: Maximum affine scale variation around one.
        mixup: Probability of applying MixUp to a training sample.

    """

    _DOWNLOADS = (
        ("train2017", "http://images.cocodataset.org/zips/train2017.zip"),
        ("val2017", "http://images.cocodataset.org/zips/val2017.zip"),
        ("annotations", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"),
    )

    def __init__(
        self,
        data_dir: str | Path,  # noqa: ARG002
        batch_size: int = 16,  # noqa: ARG002
        num_workers: int = 8,  # noqa: ARG002
        image_size: tuple[int, int] = (640, 640),  # noqa: ARG002
        pin_memory: bool = True,  # noqa: ARG002
        persistent_workers: bool = True,  # noqa: ARG002
        include_crowd: bool = False,  # noqa: ARG002
        translate: float = 0.1,  # noqa: ARG002
        scale: float = 0.9,  # noqa: ARG002
        mixup: float = 0.1,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

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
                urlretrieve(url, archive_path)
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
        image_size = tuple(self.hparams.image_size)  # type: ignore[attr-defined]

        train_images = data_dir / "train2017"
        val_images = data_dir / "val2017"
        train_ann = data_dir / "annotations" / "instances_train2017.json"
        val_ann = data_dir / "annotations" / "instances_val2017.json"

        if stage in (None, "fit"):
            self.train_dataset = COCODetectionDataset(
                image_dir=train_images,
                ann_file=train_ann,
                image_size=image_size,
                training=True,
                include_crowd=include_crowd,
                translate=self.hparams.translate,  # type: ignore[attr-defined]
                scale=self.hparams.scale,  # type: ignore[attr-defined]
                mixup=self.hparams.mixup,  # type: ignore[attr-defined]
            )
            self.val_dataset = COCODetectionDataset(
                image_dir=val_images,
                ann_file=val_ann,
                image_size=image_size,
                training=False,
                include_crowd=include_crowd,
            )

        if stage in (None, "validate"):
            self.val_dataset = COCODetectionDataset(
                image_dir=val_images,
                ann_file=val_ann,
                image_size=image_size,
                training=False,
                include_crowd=include_crowd,
            )

        if stage in (None, "test"):
            self.test_dataset = COCODetectionDataset(
                image_dir=val_images,
                ann_file=val_ann,
                image_size=image_size,
                training=False,
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


class CloseMosaic(Callback):
    """Disable Mosaic and MixUp for the final training epochs.

    Args:
        epochs: Number of final epochs without Mosaic and MixUp. Zero keeps them enabled.

    """

    def __init__(self, epochs: int = 10) -> None:
        if epochs < 0:
            raise ValueError("epochs must be non-negative.")
        self.epochs = epochs

    def on_train_epoch_start(self, trainer: Trainer, _pl_module: LightningModule) -> None:
        """Disable dataset-aware mixing augmentations for the configured final epochs.

        Args:
            trainer: Trainer running the current epoch.
            _pl_module: Model being trained.

        """
        datamodule = getattr(trainer, "datamodule", None)
        if (
            isinstance(datamodule, COCODetectionDataModule)
            and datamodule.train_dataset is not None
            and trainer.max_epochs is not None
        ):
            if trainer.max_epochs == -1:
                rank_zero_warn("CloseMosaic cannot disable Mosaic and MixUp when max_epochs=-1.", stacklevel=2)
                return
            mosaic_enabled = self.epochs == 0 or trainer.current_epoch < trainer.max_epochs - self.epochs
            datamodule.train_dataset.set_mosaic_enabled(mosaic_enabled)
