"""COCO detection data pipeline for Lightning.

A COCO-format detection split is read once into a compact in-memory index of image paths, boxes, and labels. Images are
decoded and resized on demand inside each dataloader worker, augmented with the NumPy/OpenCV pipeline, and collated
into batches of variable-length targets.

Conventions: images are ``(H, W, 3)`` uint8 RGB arrays; boxes are ``(N, 4)`` float32 `(x1, y1, x2, y2)` pixel
coordinates; image sizes passed as arguments use `(width, height)` order.

"""

import json
import os
import random
import zipfile
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .batching import collate_packed_batch
from .transforms import EvalAugmentation, Sample, TrainAugmentation, copy_sample, to_tensor_sample
from .types import BATCH, TargetDict

# OpenCV's internal thread pool oversubscribes CPU cores when several DataLoader workers decode and augment images in
# parallel. Keep each worker single-threaded so throughput scales with the worker count instead of fighting it.
cv2.setNumThreads(0)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"


def _seed_worker(worker_id: int) -> None:
    """Seeds worker-local random number generators and disable OpenCV threading.

    Deriving both seeds from the per-worker torch seed keeps augmentation streams independent across workers and
    reproducible across runs.

    Args:
        worker_id: Worker index provided by the DataLoader; unused because the seed derives from the torch seed.

    """
    del worker_id
    cv2.setNumThreads(0)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def convert_annotations(
    annotations: list[dict],
    width: int,
    height: int,
    category_id_to_label: Mapping[int, int],
    include_crowd: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Converts one image's COCO annotations into boxes and labels in original-image pixels.

    When a polygon segmentation is present the box is taken from the polygon extents, otherwise from the stored
    bounding box. Boxes are clamped to the image and duplicate ``(label, box)`` rows are dropped.

    Args:
        annotations: Raw COCO annotations for one image.
        width: Image width in pixels.
        height: Image height in pixels.
        category_id_to_label: Mapping from COCO category IDs to zero-based class labels.
        include_crowd: Whether to keep annotations flagged as crowd.

    Returns:
        A tuple ``(boxes, labels)`` where ``boxes`` is ``(N, 4)`` float32 `(x1, y1, x2, y2)` pixels and
        ``labels`` is ``(N,)`` int64.

    """
    boxes: list[list[float]] = []
    labels: list[int] = []
    seen: set[tuple[int, float, float, float, float]] = set()

    for annotation in annotations:
        if not include_crowd and annotation.get("iscrowd", 0):
            continue

        x, y, w, h = annotation["bbox"]
        if w <= 0 or h <= 0:
            continue

        x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, list) and segmentation:
            polygons = [
                np.asarray(segment, dtype=np.float32).reshape(-1, 2)
                for segment in segmentation
                if isinstance(segment, list) and len(segment) >= 6
            ]
            if polygons:
                points = np.concatenate(polygons, axis=0)
                x1, y1 = float(points[:, 0].min()), float(points[:, 1].min())
                x2, y2 = float(points[:, 0].max()), float(points[:, 1].max())

        x1 = min(max(0.0, x1), float(width))
        y1 = min(max(0.0, y1), float(height))
        x2 = min(max(0.0, x2), float(width))
        y2 = min(max(0.0, y2), float(height))
        if x2 <= x1 or y2 <= y1:
            continue

        label = category_id_to_label[int(annotation["category_id"])]
        row = (label, x1, y1, x2, y2)
        if row in seen:
            continue
        seen.add(row)
        boxes.append([x1, y1, x2, y2])
        labels.append(label)

    if boxes:
        return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)
    return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)


@dataclass(frozen=True)
class COCOIndex:
    """Compact in-memory index of one COCO detection split.

    Attributes:
        image_paths: Path to each image file.
        image_sizes: Per-image original size as `(width, height)`.
        boxes: Per-image ``(N, 4)`` float32 `(x1, y1, x2, y2)` boxes in original-image pixels.
        labels: Per-image ``(N,)`` int64 class labels.

    """

    image_paths: list[Path]
    image_sizes: list[tuple[int, int]]
    boxes: list[np.ndarray]
    labels: list[np.ndarray]

    def __len__(self) -> int:
        """Return the number of images in the split."""
        return len(self.image_paths)


def read_coco_index(image_dir: Path, ann_file: Path, include_crowd: bool) -> COCOIndex:
    """Reads a COCO annotation file into a compact index of paths, boxes, and labels.

    Category IDs are remapped to contiguous zero-based labels ordered by category ID, and images are ordered by image
    ID so the dataset is reproducible.

    Args:
        image_dir: Directory containing the image files.
        ann_file: Path to the COCO annotation JSON file.
        include_crowd: Whether to keep annotations flagged as crowd.

    Returns:
        A :class:`COCOIndex` describing the split.

    """
    content = json.loads(ann_file.read_text(encoding="utf-8"))
    sorted_category_ids = sorted(category["id"] for category in content["categories"])
    category_id_to_label = {category_id: label for label, category_id in enumerate(sorted_category_ids)}

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in content["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    image_paths: list[Path] = []
    image_sizes: list[tuple[int, int]] = []
    boxes: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for image in sorted(content["images"], key=lambda item: item["id"]):
        image_boxes, image_labels = convert_annotations(
            annotations_by_image.get(image["id"], []),
            width=image["width"],
            height=image["height"],
            category_id_to_label=category_id_to_label,
            include_crowd=include_crowd,
        )
        image_paths.append(image_dir / image["file_name"])
        image_sizes.append((int(image["width"]), int(image["height"])))
        boxes.append(image_boxes)
        labels.append(image_labels)
    return COCOIndex(image_paths, image_sizes, boxes, labels)


class COCODetectionDataset(Dataset):
    """COCO detection dataset that decodes with OpenCV and augments with the NumPy pipeline.

    Each worker keeps a bounded cache of decoded, aspect-fit-resized samples. The cache also serves as the pool of
    recent images that the mosaic augmentation draws its extra tiles from. Because transforms never mutate the image
    buffer, cached images are shared by reference while boxes and labels are copied per access.

    Args:
        image_dir: Directory containing the image files.
        ann_file: Path to the COCO annotation JSON file.
        image_size: Output image size as `(width, height)`.
        training: Whether to apply the training augmentation pipeline instead of the evaluation pipeline.
        include_crowd: Whether to keep annotations flagged as crowd.
        translate: Maximum affine translation as a fraction of the output size.
        scale: Maximum affine scale variation around one.
        mixup: Probability of applying MixUp to a training sample.
        hsv: HSV gains as `(hue, saturation, value)` for the training augmentation.
        flip: Probability of horizontal flipping in the training augmentation.
        cache_size: Maximum number of resized samples retained per worker; zero disables caching.

    """

    def __init__(
        self,
        image_dir: Path,
        ann_file: Path,
        image_size: tuple[int, int],
        training: bool,
        include_crowd: bool = False,
        translate: float = 0.1,
        scale: float = 0.5,
        mixup: float = 0.0,
        hsv: tuple[float, float, float] = (0.015, 0.7, 0.4),
        flip: float = 0.5,
        cache_size: int = 512,
    ) -> None:
        if not 0.0 <= translate <= 1.0:
            raise ValueError("translate must be between zero and one.")
        if not 0.0 <= scale < 1.0:
            raise ValueError("scale must be at least zero and less than one.")
        if not 0.0 <= mixup <= 1.0:
            raise ValueError("mixup must be between zero and one.")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative.")

        index = read_coco_index(Path(image_dir), Path(ann_file), include_crowd)
        self.image_paths = index.image_paths
        self.image_sizes = index.image_sizes
        self._boxes = index.boxes
        self._labels = index.labels
        self.image_size = image_size
        self.cache_size = min(cache_size, len(index))
        self._mosaic_enabled = torch.tensor(True).share_memory_()
        self._cache: list[Sample | None] = [None] * len(index)
        self._recent: deque[int] = deque()
        self._augmentation: TrainAugmentation | EvalAugmentation = (
            TrainAugmentation(image_size, translate=translate, scale=scale, mixup=mixup, hsv=hsv, flip=flip, fill=114)
            if training
            else EvalAugmentation(image_size, fill=114)
        )

    def __len__(self) -> int:
        """Return the number of images in the split."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, TargetDict]:
        """Augments the sample at ``index`` and convert it to tensors.

        Args:
            index: Zero-based sample index.

        Returns:
            A tuple ``(image, target)`` where ``target`` holds ``boxes`` and ``labels`` tensors.

        """
        return to_tensor_sample(self._augmentation(index, self))

    @property
    def mosaic_enabled(self) -> bool:
        """Whether mosaic and MixUp augmentation are currently enabled."""
        return bool(self._mosaic_enabled)

    def set_mosaic_enabled(self, enabled: bool) -> None:
        """Enables or disables mosaic and MixUp augmentation across worker processes.

        Args:
            enabled: Whether image-mixing augmentations should be applied.

        """
        self._mosaic_enabled.fill_(enabled)

    def mosaic_indices(self) -> list[int]:
        """Returns three extra sample indices for a four-image mosaic.

        Indices come from the recent-image pool when it is populated, otherwise from uniform random sampling.

        Returns:
            Three sample indices.

        """
        if self._recent:
            return random.choices(self._recent, k=3)
        return [random.randrange(len(self.image_paths)) for _ in range(3)]

    def mixup_index(self) -> int:
        """Returns one uniformly random sample index for MixUp."""
        return random.randrange(len(self.image_paths))

    def load_sample(self, index: int) -> Sample:
        """Returns the sample at ``index``, decoding and resizing it on a cache miss.

        Args:
            index: Zero-based sample index.

        Returns:
            A :class:`~lightning_yolo.transforms.Sample` with independent box and label arrays; its image buffer may be
            shared with the cache and must not be modified in place.

        """
        cached = self._cache[index]
        if cached is not None:
            return copy_sample(cached)
        image_path = self.image_paths[index]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        image, boxes = self._fit_within_output(image, self._boxes[index])
        sample = Sample(image, boxes, self._labels[index])
        if self.cache_size:
            self._cache_sample(index, sample)
        return copy_sample(sample)

    def _fit_within_output(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resizes an image to fit the output size while preserving aspect ratio, and scale its boxes to match.

        Args:
            image: Decoded image in ``(H, W, 3)`` RGB.
            boxes: Boxes in original-image pixels.

        Returns:
            A tuple ``(image, boxes)`` with the resized image and correspondingly scaled boxes.

        """
        input_height, input_width = image.shape[:2]
        output_width, output_height = self.image_size
        scale = min(output_height / input_height, output_width / input_width)
        target_width = min(round(input_width * scale), output_width)
        target_height = min(round(input_height * scale), output_height)
        if (target_height, target_width) == (input_height, input_width):
            return image, boxes.astype(np.float32, copy=False)
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        scaled = boxes * np.array([target_width / input_width, target_height / input_height] * 2, dtype=np.float32)
        return image, scaled.astype(np.float32, copy=False)

    def _cache_sample(self, index: int, sample: Sample) -> None:
        """Inserts a sample into the bounded cache, evicting the oldest entry when full.

        Args:
            index: Sample index used as the cache key.
            sample: Resized sample to retain.

        """
        if len(self._recent) >= self.cache_size:
            self._cache[self._recent.popleft()] = None
        self._cache[index] = sample
        self._recent.append(index)


class COCODetectionDataModule(LightningDataModule):
    """Lightning DataModule for COCO detection.

    Args:
        data_dir: Root directory holding (or receiving) the COCO files.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes per DataLoader.
        image_size: Output image size as `(width, height)`.
        pin_memory: Whether DataLoaders pin host memory.
        persistent_workers: Whether workers persist between epochs.
        prefetch_factor: Number of batches prefetched per worker.
        include_crowd: Whether to keep annotations flagged as crowd.
        translate: Maximum affine translation as a fraction of the output size.
        scale: Maximum affine scale variation around one.
        mixup: Probability of applying MixUp to a training sample.
        hsv: HSV gains as `(hue, saturation, value)` for the training augmentation.
        flip: Probability of horizontal flipping in the training augmentation.
        cache_size: Resized samples cached per training worker; ``None`` derives a value from the batch size.

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
        prefetch_factor: int = 2,
        include_crowd: bool = False,
        translate: float = 0.1,
        scale: float = 0.5,
        mixup: float = 0.0,
        cache_size: int | None = None,
        hsv: tuple[float, float, float] = (0.015, 0.7, 0.4),
        flip: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset: COCODetectionDataset | None = None
        self.val_dataset: COCODetectionDataset | None = None
        self.test_dataset: COCODetectionDataset | None = None

    def prepare_data(self) -> None:
        """Downloads and extracts the COCO files when they are missing."""
        data_dir = Path(self.hparams["data_dir"])
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
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(data_dir)

    def setup(self, stage: str | None = None) -> None:
        """Creates the datasets needed for the requested stage.

        Args:
            stage: Lightning stage hint (``fit``, ``validate``, ``test``), or ``None`` to prepare all datasets.

        """
        data_dir = Path(self.hparams["data_dir"])
        train_images = data_dir / "train2017"
        val_images = data_dir / "val2017"
        train_annotations = data_dir / "annotations" / "instances_train2017.json"
        val_annotations = data_dir / "annotations" / "instances_val2017.json"
        cache_size = self.hparams["cache_size"]
        if cache_size is None:
            cache_size = min(self.hparams["batch_size"] * 8, 1000)

        if stage in (None, "fit"):
            self.train_dataset = self._create_dataset(
                image_dir=train_images, ann_file=train_annotations, training=True, cache_size=cache_size
            )
        if stage in (None, "fit", "validate"):
            self.val_dataset = self._create_dataset(
                image_dir=val_images, ann_file=val_annotations, training=False, cache_size=0
            )
        if stage in (None, "test"):
            self.test_dataset = self._create_dataset(
                image_dir=val_images, ann_file=val_annotations, training=False, cache_size=0
            )

    def _create_dataset(self, image_dir: Path, ann_file: Path, training: bool, cache_size: int) -> COCODetectionDataset:
        """Constructs one detection dataset for a split.

        Args:
            image_dir: Directory containing the split images.
            ann_file: COCO instances annotation file for the split.
            training: Whether to apply the training augmentation pipeline.
            cache_size: Resized-sample cache size; only used for the training split.

        Returns:
            A configured :class:`COCODetectionDataset`.

        """
        if training:
            return COCODetectionDataset(
                image_dir=image_dir,
                ann_file=ann_file,
                image_size=self.hparams["image_size"],
                training=True,
                include_crowd=self.hparams["include_crowd"],
                translate=self.hparams["translate"],
                scale=self.hparams["scale"],
                mixup=self.hparams["mixup"],
                cache_size=cache_size,
                hsv=self.hparams["hsv"],
                flip=self.hparams["flip"],
            )
        return COCODetectionDataset(
            image_dir=image_dir,
            ann_file=ann_file,
            image_size=self.hparams["image_size"],
            training=False,
            include_crowd=self.hparams["include_crowd"],
        )

    def _build_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader[BATCH]:
        """Builds a DataLoader with the shared worker, memory, and collation settings.

        Args:
            dataset: Dataset to load from.
            shuffle: Whether to shuffle sample order.

        Returns:
            A configured DataLoader yielding ``(images, targets)`` batches.

        """
        num_workers = self.hparams["num_workers"]
        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.hparams["pin_memory"],
            persistent_workers=self.hparams["persistent_workers"] and num_workers > 0,
            prefetch_factor=self.hparams["prefetch_factor"] if num_workers > 0 else None,
            multiprocessing_context="fork" if num_workers > 0 else None,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            collate_fn=collate_packed_batch,
        )

    def train_dataloader(self) -> DataLoader[BATCH]:
        """Builds the training DataLoader.

        Returns:
            A DataLoader yielding batches of training images and targets.

        """
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting the training dataloader.")
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[BATCH]:
        """Builds the validation DataLoader.

        Returns:
            A DataLoader yielding batches of validation images and targets.

        """
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit' or 'validate') before requesting the validation dataloader.")
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[BATCH]:
        """Builds the test DataLoader.

        Returns:
            A DataLoader yielding batches of test images and targets.

        """
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before requesting the test dataloader.")
        return self._build_dataloader(self.test_dataset, shuffle=False)


class CloseMosaic(Callback):
    """Disables mosaic and MixUp augmentation for the final training epochs.

    Args:
        epochs: Number of final epochs without mosaic and MixUp. Zero keeps them enabled throughout.

    """

    def __init__(self, epochs: int = 10) -> None:
        if epochs < 0:
            raise ValueError("epochs must be non-negative.")
        self.epochs = epochs

    def on_train_epoch_start(self, trainer: Trainer, _pl_module: LightningModule) -> None:
        """Toggles image-mixing augmentations at the start of each training epoch.

        Args:
            trainer: The running trainer.
            _pl_module: The model being trained.

        """
        datamodule = getattr(trainer, "datamodule", None)
        dataset = getattr(datamodule, "train_dataset", None)
        if not isinstance(dataset, COCODetectionDataset) or trainer.max_epochs is None:
            return
        if trainer.max_epochs == -1:
            rank_zero_warn("CloseMosaic cannot disable mosaic and MixUp when max_epochs is -1.", stacklevel=2)
            return
        keep_enabled = self.epochs == 0 or trainer.current_epoch < trainer.max_epochs - self.epochs
        dataset.set_mosaic_enabled(keep_enabled)
