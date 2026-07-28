"""Detection augmentation pipeline built on NumPy and OpenCV.

Every transform operates on a single :class:`Sample` (a NumPy image with NumPy boxes and labels), and the augmentation
sequence keeps that representation until a single conversion to PyTorch tensors in :func:`to_tensor_sample`.

Architecture:

- :class:`Sample` is the data structure passed between transforms.
- :class:`SampleSource` is the protocol a dataset implements so the mixing transforms (:class:`Mosaic`, :class:`MixUp`)
  can pull auxiliary samples without depending on a concrete dataset class.
- The individual transforms each encapsulate one operation.
- :class:`TrainAugmentation` and :class:`EvalAugmentation` compose fixed sequences of those transforms.

Conventions: images are ``(H, W, 3)`` uint8 arrays in RGB order; boxes are ``(N, 4)`` float32 arrays in ``xyxy`` pixel
coordinates; sizes passed as arguments use ``(width, height)`` order.

"""

import random
from dataclasses import dataclass, replace
from typing import Protocol

import cv2
import numpy as np
import torch
from torch import Tensor

from .types import TargetDict


@dataclass
class Sample:
    """A single detection sample in NumPy form.

    Attributes:
        image: Image array of shape ``(H, W, 3)`` with dtype ``uint8`` in RGB order.
        boxes: Bounding boxes of shape ``(N, 4)`` with dtype ``float32`` in ``xyxy`` pixel coordinates.
        labels: Class labels of shape ``(N,)`` with dtype ``int64``.

    """

    image: np.ndarray
    boxes: np.ndarray
    labels: np.ndarray


class SampleSource(Protocol):
    """Protocol for a dataset that provides samples to the mixing transforms.

    :class:`Mosaic` and :class:`MixUp` need samples beyond the primary one. Implementing this protocol lets those
    transforms stay decoupled from any concrete dataset class.

    """

    def __len__(self) -> int:
        """Return the number of samples in the source."""
        ...

    def load_sample(self, index: int) -> Sample:
        """Load and resize the sample at ``index``."""
        ...

    def mosaic_indices(self) -> list[int]:
        """Return three auxiliary sample indices for a four-image mosaic."""
        ...

    def mixup_index(self) -> int:
        """Return one auxiliary sample index for MixUp."""
        ...

    @property
    def mosaic_enabled(self) -> bool:
        """Whether Mosaic and MixUp are currently enabled."""
        ...


def _empty_boxes() -> np.ndarray:
    """Return an empty ``(0, 4)`` float32 box array."""
    return np.zeros((0, 4), dtype=np.float32)


def _empty_labels() -> np.ndarray:
    """Return an empty ``(0,)`` int64 label array."""
    return np.zeros((0,), dtype=np.int64)


class LetterBox:
    """Resizes a sample to fit the target size preserving aspect ratio, then center-pad to the exact size.

    Args:
        image_size: Output size as ``(width, height)``.
        fill: Constant pad value.

    """

    def __init__(self, image_size: tuple[int, int], fill: int = 114) -> None:
        self.width, self.height = image_size
        self.fill = fill

    def __call__(self, sample: Sample) -> Sample:
        """Letterboxes one sample.

        Args:
            sample: Input sample.

        Returns:
            A new sample resized and padded to the configured ``(width, height)`` size.

        """
        image = sample.image
        boxes = sample.boxes
        input_height, input_width = image.shape[:2]
        scale = min(self.height / input_height, self.width / input_width)
        resized_height = round(input_height * scale)
        resized_width = round(input_width * scale)
        if (resized_height, resized_width) != (input_height, input_width):
            image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
            if boxes.size:
                boxes = boxes * scale

        canvas = np.full((self.height, self.width, image.shape[2]), self.fill, dtype=np.uint8)
        left = round((self.width - resized_width) / 2 - 0.1)
        top = round((self.height - resized_height) / 2 - 0.1)
        canvas[top : top + resized_height, left : left + resized_width] = image
        if boxes.size:
            boxes = boxes.astype(np.float32, copy=True)
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
        else:
            boxes = _empty_boxes()
        return Sample(canvas, boxes, sample.labels)


class Mosaic:
    """Combines four samples into a single square canvas of twice the tile size.

    The oversized canvas leaves room for a following :class:`RandomPerspective` to sample geometry and crop back to the
    tile size.

    Args:
        image_size: Tile size as ``(width, height)``; must be square.
        fill: Constant fill value for uncovered canvas areas.

    """

    def __init__(self, image_size: tuple[int, int], fill: int = 114) -> None:
        if image_size[0] != image_size[1]:
            raise ValueError("Mosaic requires a square image size.")
        self.size = image_size[0]
        self.fill = fill

    def __call__(self, primary: Sample, source: SampleSource) -> Sample:
        """Build a four-image mosaic.

        Args:
            primary: The main sample, placed first.
            source: Sample source used to draw the three auxiliary tiles.

        Returns:
            The mosaic sample, a canvas of ``2 * size`` in each dimension.

        """
        tiles = [primary, *(source.load_sample(index) for index in source.mosaic_indices())]
        canvas_size = self.size * 2
        center_x = random.randint(self.size // 2, self.size + self.size // 2)
        center_y = random.randint(self.size // 2, self.size + self.size // 2)
        canvas = np.full((canvas_size, canvas_size, primary.image.shape[2]), self.fill, dtype=np.uint8)
        boxes: list[np.ndarray] = []
        labels: list[np.ndarray] = []

        for position, tile in enumerate(tiles):
            tile_height, tile_width = tile.image.shape[:2]
            if position == 0:  # top-left
                dst_x1, dst_y1 = max(center_x - tile_width, 0), max(center_y - tile_height, 0)
                dst_x2, dst_y2 = center_x, center_y
                src_x1, src_y1 = tile_width - (dst_x2 - dst_x1), tile_height - (dst_y2 - dst_y1)
            elif position == 1:  # top-right
                dst_x1, dst_y1 = center_x, max(center_y - tile_height, 0)
                dst_x2, dst_y2 = min(center_x + tile_width, canvas_size), center_y
                src_x1, src_y1 = 0, tile_height - (dst_y2 - dst_y1)
            elif position == 2:  # bottom-left
                dst_x1, dst_y1 = max(center_x - tile_width, 0), center_y
                dst_x2, dst_y2 = center_x, min(center_y + tile_height, canvas_size)
                src_x1, src_y1 = tile_width - (dst_x2 - dst_x1), 0
            else:  # bottom-right
                dst_x1, dst_y1 = center_x, center_y
                dst_x2, dst_y2 = min(center_x + tile_width, canvas_size), min(center_y + tile_height, canvas_size)
                src_x1, src_y1 = 0, 0
            src_x2 = src_x1 + (dst_x2 - dst_x1)
            src_y2 = src_y1 + (dst_y2 - dst_y1)

            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = tile.image[src_y1:src_y2, src_x1:src_x2]
            if not tile.boxes.size:
                continue
            shifted = tile.boxes.astype(np.float32, copy=True)
            shifted[:, [0, 2]] += dst_x1 - src_x1
            shifted[:, [1, 3]] += dst_y1 - src_y1
            shifted[:, [0, 2]] = shifted[:, [0, 2]].clip(0, canvas_size)
            shifted[:, [1, 3]] = shifted[:, [1, 3]].clip(0, canvas_size)
            keep = (shifted[:, 2] > shifted[:, 0]) & (shifted[:, 3] > shifted[:, 1])
            boxes.append(shifted[keep])
            labels.append(tile.labels[keep])

        merged_boxes = np.concatenate(boxes, axis=0) if boxes else _empty_boxes()
        merged_labels = np.concatenate(labels, axis=0) if labels else _empty_labels()
        return Sample(canvas, merged_boxes, merged_labels)


class RandomPerspective:
    """Applies a random centered scale-and-translate affine warp to the image and its boxes.

    Args:
        translate: Maximum translation as a fraction of the output size.
        scale: Maximum scale variation around one.
        fill: Constant border fill value.

    """

    def __init__(self, translate: float = 0.1, scale: float = 0.5, fill: int = 114) -> None:
        if not 0.0 <= translate <= 1.0:
            raise ValueError("translate must be between zero and one.")
        if not 0.0 <= scale < 1.0:
            raise ValueError("scale must be at least zero and less than one.")
        self.translate = translate
        self.scale = scale
        self.fill = fill

    def __call__(self, sample: Sample, output_size: tuple[int, int]) -> Sample:
        """Warps a sample and its boxes into an output canvas.

        The transform centers the input image, scales it about the center, and translates it into an output canvas of
        ``output_size``. When the input is larger than the output (as with a mosaic canvas), this also crops it.

        Args:
            sample: Input sample.
            output_size: Output canvas size as ``(width, height)``.

        Returns:
            A new sample warped into ``output_size``.

        """
        image = sample.image
        output_width, output_height = output_size
        input_height, input_width = image.shape[:2]

        recenter = np.eye(3, dtype=np.float32)
        recenter[0, 2] = -input_width / 2
        recenter[1, 2] = -input_height / 2

        scale_about_center = np.eye(3, dtype=np.float32)
        sampled_scale = random.uniform(1 - self.scale, 1 + self.scale)
        scale_about_center[:2] = cv2.getRotationMatrix2D(angle=0.0, center=(0, 0), scale=sampled_scale)

        translate_to_output = np.eye(3, dtype=np.float32)
        translate_to_output[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * output_width
        translate_to_output[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * output_height

        matrix = translate_to_output @ scale_about_center @ recenter
        warped = cv2.warpAffine(
            image,
            matrix[:2],
            dsize=(output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderValue=(self.fill, self.fill, self.fill),
        )
        if not sample.boxes.size:
            return Sample(warped, _empty_boxes(), _empty_labels())

        warped_boxes = self._transform_boxes(sample.boxes, matrix)
        warped_boxes[:, [0, 2]] = warped_boxes[:, [0, 2]].clip(0, output_width)
        warped_boxes[:, [1, 3]] = warped_boxes[:, [1, 3]].clip(0, output_height)
        keep = self._surviving_boxes(sample.boxes * sampled_scale, warped_boxes)
        return Sample(warped, warped_boxes[keep], sample.labels[keep])

    @staticmethod
    def _transform_boxes(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Warps the four corners of each box and return the enclosing axis-aligned boxes."""
        count = len(boxes)
        corners = np.ones((count * 4, 3), dtype=np.float32)
        corners[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(count * 4, 2)
        corners = (corners @ matrix.T)[:, :2].reshape(count, 8)
        xs = corners[:, [0, 2, 4, 6]]
        ys = corners[:, [1, 3, 5, 7]]
        return np.stack((xs.min(1), ys.min(1), xs.max(1), ys.max(1)), axis=1).astype(np.float32)

    @staticmethod
    def _surviving_boxes(
        original: np.ndarray,
        warped: np.ndarray,
        min_side: float = 2.0,
        max_aspect_ratio: float = 100.0,
        min_area_ratio: float = 0.1,
        epsilon: float = 1e-16,
    ) -> np.ndarray:
        """Returns a boolean mask of boxes still large and well-shaped enough to keep after warping.

        Args:
            original: Boxes before warping, scaled by the applied scale factor, shape ``(N, 4)``.
            warped: Boxes after warping, shape ``(N, 4)``.
            min_side: Minimum width and height in pixels.
            max_aspect_ratio: Maximum allowed width-to-height (or height-to-width) ratio.
            min_area_ratio: Minimum warped-to-original area ratio.
            epsilon: Small constant guarding against division by zero.

        Returns:
            A boolean mask of shape ``(N,)``.

        """
        original_width = original[:, 2] - original[:, 0]
        original_height = original[:, 3] - original[:, 1]
        warped_width = warped[:, 2] - warped[:, 0]
        warped_height = warped[:, 3] - warped[:, 1]
        aspect_ratio = np.maximum(warped_width / (warped_height + epsilon), warped_height / (warped_width + epsilon))
        area_ratio = warped_width * warped_height / (original_width * original_height + epsilon)
        return (
            (warped_width > min_side)
            & (warped_height > min_side)
            & (area_ratio > min_area_ratio)
            & (aspect_ratio < max_aspect_ratio)
        )


class MixUp:
    """Blends two samples by a beta-distributed ratio and concatenate their annotations.

    Args:
        alpha: Symmetric beta distribution parameter.

    """

    def __init__(self, alpha: float = 32.0) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        self.alpha = alpha

    def __call__(self, primary: Sample, other: Sample) -> Sample:
        """Blend two equally sized samples.

        Args:
            primary: First sample.
            other: Second sample; must have the same image shape as ``primary``.

        Returns:
            A new blended sample.

        """
        if primary.image.shape != other.image.shape:
            raise ValueError(f"MixUp images must match, got {primary.image.shape} and {other.image.shape}.")
        ratio = float(np.random.beta(self.alpha, self.alpha))
        image = (primary.image.astype(np.float32) * ratio + other.image.astype(np.float32) * (1.0 - ratio)).astype(
            np.uint8
        )
        boxes = np.concatenate((primary.boxes, other.boxes), axis=0)
        labels = np.concatenate((primary.labels, other.labels), axis=0)
        return Sample(image, boxes, labels)


class RandomHSV:
    """Randomly shifts hue, saturation, and value with per-channel lookup tables.

    Args:
        hue: Maximum hue gain.
        saturation: Maximum saturation gain.
        value: Maximum value gain.

    """

    def __init__(self, hue: float = 0.015, saturation: float = 0.7, value: float = 0.4) -> None:
        self.hue = hue
        self.saturation = saturation
        self.value = value

    def __call__(self, sample: Sample) -> Sample:
        """Apply a random HSV shift to the image, leaving annotations unchanged.

        Args:
            sample: Input sample.

        Returns:
            A new sample with an HSV-adjusted image.

        """
        if self.hue == 0.0 and self.saturation == 0.0 and self.value == 0.0:
            return sample
        gains = np.random.uniform(-1, 1, 3) * (self.hue, self.saturation, self.value)
        table = np.arange(256, dtype=gains.dtype)
        hue_lut = ((table + gains[0] * 180) % 180).astype(np.uint8)
        saturation_lut = np.clip(table * (gains[1] + 1), 0, 255).astype(np.uint8)
        value_lut = np.clip(table * (gains[2] + 1), 0, 255).astype(np.uint8)
        saturation_lut[0] = 0  # Keep fully desaturated pixels from gaining color.

        hue, saturation, value = cv2.split(cv2.cvtColor(sample.image, cv2.COLOR_RGB2HSV))
        adjusted = cv2.merge((cv2.LUT(hue, hue_lut), cv2.LUT(saturation, saturation_lut), cv2.LUT(value, value_lut)))
        image = cv2.cvtColor(adjusted, cv2.COLOR_HSV2RGB)
        return Sample(image, sample.boxes, sample.labels)


class RandomHorizontalFlip:
    """Flips the image horizontally with a given probability and mirror the boxes.

    Args:
        probability: Probability of applying the flip.

    """

    def __init__(self, probability: float = 0.5) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between zero and one.")
        self.probability = probability

    def __call__(self, sample: Sample) -> Sample:
        """Flip a sample horizontally with the configured probability.

        Args:
            sample: Input sample.

        Returns:
            The flipped sample, or the input sample unchanged.

        """
        if random.random() >= self.probability:
            return sample
        image = np.ascontiguousarray(sample.image[:, ::-1])
        width = image.shape[1]
        boxes = sample.boxes
        if boxes.size:
            boxes = boxes.astype(np.float32, copy=True)
            left = boxes[:, 0].copy()
            boxes[:, 0] = width - boxes[:, 2]
            boxes[:, 2] = width - left
        return Sample(image, boxes, sample.labels)


class SanitizeBoxes:
    """Clips boxes to the image bounds and drop any that a clip collapses to zero area.

    Size and aspect-ratio filtering of augmented boxes is applied in :class:`RandomPerspective`; this final step only
    clamps boxes to the image and removes ones left with no width or height.

    """

    def __call__(self, sample: Sample) -> Sample:
        """Clip boxes to the image and drop those with non-positive area.

        Args:
            sample: Input sample.

        Returns:
            A new sample with clipped boxes and their labels.

        """
        if not sample.boxes.size:
            return sample
        height, width = sample.image.shape[:2]
        boxes = sample.boxes.astype(np.float32, copy=True)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return Sample(sample.image, boxes[keep], sample.labels[keep])


class TrainAugmentation:
    """Training augmentation: mosaic or letterbox, an affine warp, optional MixUp, HSV, flip, and box cleanup.

    Args:
        image_size: Output size as ``(width, height)``; must be square because mosaic is used.
        translate: Maximum affine translation as a fraction of the output size.
        scale: Maximum affine scale variation around one.
        mixup: Probability of applying MixUp; zero disables it.
        hsv: HSV gains as ``(hue, saturation, value)``.
        flip: Probability of a horizontal flip.
        fill: Constant fill value for padding and warp borders.

    """

    def __init__(
        self,
        image_size: tuple[int, int],
        translate: float = 0.1,
        scale: float = 0.5,
        mixup: float = 0.0,
        hsv: tuple[float, float, float] = (0.015, 0.7, 0.4),
        flip: float = 0.5,
        fill: int = 114,
    ) -> None:
        self.image_size = image_size
        self._mosaic = Mosaic(image_size, fill=fill)
        self._letterbox = LetterBox(image_size, fill=fill)
        self._perspective = RandomPerspective(translate=translate, scale=scale, fill=fill)
        self._mixup = MixUp(alpha=32.0)
        self._mixup_probability = mixup
        self._hsv = RandomHSV(*hsv)
        self._flip = RandomHorizontalFlip(flip)
        self._sanitize = SanitizeBoxes()

    def __call__(self, index: int, source: SampleSource) -> Sample:
        """Produce one fully augmented training sample.

        Args:
            index: Primary sample index.
            source: Sample source used for mosaic tiles and MixUp partners.
        Returns:
            A fully augmented sample in NumPy form.

        """
        sample = self._geometry(index, source)
        if self._mixup_probability > 0.0 and source.mosaic_enabled and random.random() < self._mixup_probability:
            other = self._geometry(source.mixup_index(), source)
            sample = self._mixup(sample, other)
        sample = self._hsv(sample)
        sample = self._flip(sample)
        return self._sanitize(sample)

    def _geometry(self, index: int, source: SampleSource) -> Sample:
        """Build one geometrically augmented sample: mosaic or letterbox, then an affine warp to the output size."""
        primary = source.load_sample(index)
        composed = self._mosaic(primary, source) if source.mosaic_enabled else self._letterbox(primary)
        return self._perspective(composed, self.image_size)


class EvalAugmentation:
    """Evaluation augmentation: letterbox to the target size and clip boxes.

    Args:
        image_size: Output size as ``(width, height)``.
        fill: Constant pad value.

    """

    def __init__(self, image_size: tuple[int, int], fill: int = 114) -> None:
        self._letterbox = LetterBox(image_size, fill=fill)
        self._sanitize = SanitizeBoxes()

    def __call__(self, index: int, source: SampleSource) -> Sample:
        """Produce one letterboxed evaluation sample.

        Args:
            index: Sample index.
            source: Sample source.
        Returns:
            A letterboxed sample with clipped boxes.

        """
        sample = self._letterbox(source.load_sample(index))
        return self._sanitize(sample)


def to_tensor_sample(sample: Sample) -> tuple[Tensor, TargetDict]:
    """Converts a :class:`Sample` to the tensor batch element expected by the model.

    Args:
        sample: Augmented sample in NumPy form.

    Returns:
        A tuple ``(image, target)`` where ``image`` is a ``(3, H, W)`` uint8 tensor and ``target`` contains ``boxes``
        (``(N, 4)`` float32, ``xyxy`` pixels) and ``labels`` (``(N,)`` int64).

    """
    image = torch.from_numpy(np.ascontiguousarray(sample.image)).permute(2, 0, 1)
    boxes = torch.from_numpy(np.ascontiguousarray(sample.boxes, dtype=np.float32)).reshape(-1, 4)
    labels = torch.from_numpy(np.ascontiguousarray(sample.labels, dtype=np.int64)).reshape(-1)
    return image, {"boxes": boxes, "labels": labels}


def copy_sample(sample: Sample) -> Sample:
    """Returns a copy of ``sample`` that shares the image but owns independent box and label arrays.

    The image buffer is shared because transforms never write to their input image; only boxes and labels are copied,
    so a cached sample can be reused safely by later augmentations.

    Args:
        sample: Sample to copy.

    Returns:
        A new :class:`Sample` sharing the image with fresh box and label arrays.

    """
    return replace(sample, boxes=sample.boxes.copy(), labels=sample.labels.copy())
