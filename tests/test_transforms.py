import numpy as np
import torch

from lightning_yolo.transforms import (
    EvalAugmentation,
    LetterBox,
    MixUp,
    Mosaic,
    RandomHorizontalFlip,
    RandomHSV,
    RandomPerspective,
    Sample,
    SanitizeBoxes,
    TrainAugmentation,
    to_tensor_sample,
)


def _sample(height: int = 8, width: int = 8, box: list[float] | None = None, label: int = 3) -> Sample:
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    boxes = np.array([box if box is not None else [1.0, 1.0, 5.0, 5.0]], dtype=np.float32)
    labels = np.array([label], dtype=np.int64)
    return Sample(image, boxes, labels)


class _StubSource:
    """Minimal SampleSource for exercising mixing transforms deterministically."""

    def __init__(self, sample: Sample, mosaic_enabled: bool = True) -> None:
        self._sample = sample
        self._mosaic_enabled = mosaic_enabled

    def __len__(self) -> int:
        return 10

    def load_sample(self, index: int) -> Sample:
        del index
        return Sample(self._sample.image.copy(), self._sample.boxes.copy(), self._sample.labels.copy())

    def mosaic_indices(self) -> list[int]:
        return [0, 0, 0]

    def mixup_index(self) -> int:
        return 0

    @property
    def mosaic_enabled(self) -> bool:
        return self._mosaic_enabled


def test_letterbox() -> None:
    sample = Sample(np.zeros((10, 10, 3), dtype=np.uint8), np.array([[1.0, 1.0, 3.0, 3.0]], np.float32), np.array([1]))

    result = LetterBox((20, 13))(sample)

    assert result.image.shape == (13, 20, 3)
    assert np.all(result.image[:, :3] == 114)
    assert np.all(result.image[:, -4:] == 114)
    np.testing.assert_allclose(result.boxes, np.array([[4.3, 1.3, 6.9, 3.9]], np.float32))


def test_mosaic() -> None:
    source = _StubSource(_sample())

    result = Mosaic((8, 8))(source.load_sample(0), source)

    assert result.image.shape == (16, 16, 3)
    assert result.boxes.shape[1] == 4
    assert len(result.boxes) == len(result.labels)


def test_random_perspective() -> None:
    sample = _sample(8, 8, box=[1.0, 1.0, 6.0, 6.0])

    result = RandomPerspective(translate=0.0, scale=0.0)(sample, output_size=(8, 8))

    assert result.image.shape == (8, 8, 3)
    np.testing.assert_allclose(result.boxes, sample.boxes, atol=1e-4)
    np.testing.assert_array_equal(result.labels, sample.labels)


def test_random_hsv() -> None:
    sample = _sample()

    result = RandomHSV(0.0, 0.0, 0.0)(sample)

    assert result is sample


def test_random_horizontal_flip() -> None:
    sample = Sample(np.zeros((4, 8, 3), dtype=np.uint8), np.array([[1.0, 1.0, 3.0, 3.0]], np.float32), np.array([2]))

    result = RandomHorizontalFlip(probability=1.0)(sample)

    np.testing.assert_allclose(result.boxes, np.array([[5.0, 1.0, 7.0, 3.0]], np.float32))
    np.testing.assert_array_equal(result.labels, sample.labels)


def test_sanitize_boxes() -> None:
    sample = Sample(
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.array([[-2.0, -2.0, 4.0, 4.0], [12.0, 5.0, 14.0, 8.0]], np.float32),
        np.array([1, 2], np.int64),
    )

    result = SanitizeBoxes()(sample)

    np.testing.assert_allclose(result.boxes, np.array([[0.0, 0.0, 4.0, 4.0]], np.float32))
    np.testing.assert_array_equal(result.labels, np.array([1], np.int64))


def test_mixup() -> None:
    a = Sample(np.zeros((4, 4, 3), dtype=np.uint8), np.array([[1.0, 1.0, 3.0, 3.0]], np.float32), np.array([0]))
    b = Sample(np.full((4, 4, 3), 200, dtype=np.uint8), np.array([[0.0, 0.0, 2.0, 2.0]], np.float32), np.array([1]))

    result = MixUp(alpha=2.0)(a, b)

    assert result.image.shape == (4, 4, 3)
    assert result.image.dtype == np.uint8
    assert result.boxes.shape == (2, 4)
    np.testing.assert_array_equal(result.labels, np.array([0, 1], np.int64))


def test_train_augmentation() -> None:
    source = _StubSource(_sample(8, 8, box=[1.0, 1.0, 7.0, 7.0]))

    result = TrainAugmentation((8, 8), translate=0.0, scale=0.0, mixup=0.0)(0, source)

    assert result.image.shape == (8, 8, 3)
    assert result.boxes.shape[1] == 4
    assert len(result.boxes) == len(result.labels)

    # Mosaic disabled: source image is smaller and gets letterboxed to the target size.
    source_no_mosaic = _StubSource(_sample(4, 8, box=[1.0, 1.0, 3.0, 3.0]), mosaic_enabled=False)

    result_no_mosaic = TrainAugmentation((8, 8), translate=0.0, scale=0.0, mixup=0.0)(0, source_no_mosaic)

    assert result_no_mosaic.image.shape == (8, 8, 3)


def test_eval_augmentation() -> None:
    source = _StubSource(_sample(4, 8, box=[1.0, 1.0, 3.0, 3.0]))

    result = EvalAugmentation((8, 8))(0, source)

    assert result.image.shape == (8, 8, 3)


def test_to_tensor_sample() -> None:
    sample = Sample(
        np.full((6, 8, 3), 255, dtype=np.uint8),
        np.array([[1.0, 2.0, 3.0, 4.0]], np.float32),
        np.array([5], np.int64),
    )

    image, target = to_tensor_sample(sample)

    assert image.shape == (3, 6, 8)
    assert image.dtype == torch.uint8
    assert torch.equal(image, torch.full_like(image, 255))
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].dtype == torch.int64
    torch.testing.assert_close(target["boxes"], torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    # Empty boxes.
    empty_sample = Sample(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((0, 4), np.float32), np.zeros((0,), np.int64))

    empty_image, empty_target = to_tensor_sample(empty_sample)

    assert empty_image.shape == (3, 4, 4)
    assert empty_target["boxes"].shape == (0, 4)
    assert empty_target["labels"].shape == (0,)
