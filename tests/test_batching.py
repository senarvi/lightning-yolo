import torch

from lightning_yolo.batching import (
    collate_packed_batch,
    pack_targets,
    split_targets,
)


def _targets() -> list[dict[str, torch.Tensor]]:
    return [
        {"boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]), "labels": torch.tensor([3, 5])},
        {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)},
        {"boxes": torch.tensor([[2.0, 2.0, 4.0, 4.0]]), "labels": torch.tensor([7])},
    ]


def test_pack_targets() -> None:
    packed = pack_targets(_targets())

    assert packed["boxes"].shape == (3, 4)
    assert torch.equal(packed["labels"], torch.tensor([3, 5, 7]))
    assert torch.equal(packed["sample_idxs"], torch.tensor([0, 0, 2]))
    assert packed["counts"] == [2, 0, 1]

    # Empty batch.
    empty_targets = [
        {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)},
        {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)},
    ]
    empty_packed = pack_targets(empty_targets)

    assert empty_packed["boxes"].shape == (0, 4)
    assert empty_packed["labels"].shape == (0,)
    assert empty_packed["sample_idxs"].shape == (0,)
    assert empty_packed["counts"] == [0, 0]

    # Multi-label targets.
    multilabel_targets = [
        {"boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]), "labels": torch.tensor([[True, False]])},
        {"boxes": torch.empty((0, 4)), "labels": torch.empty((0, 2), dtype=torch.bool)},
    ]
    multilabel_packed = pack_targets(multilabel_targets)

    assert multilabel_packed["labels"].shape == (1, 2)
    assert multilabel_packed["labels"].dtype == torch.bool


def test_split_targets_round_trip() -> None:
    targets = _targets()

    unpacked = split_targets(pack_targets(targets))

    assert len(unpacked) == len(targets)
    for original, restored in zip(unpacked, targets, strict=True):
        torch.testing.assert_close(original["boxes"], restored["boxes"])
        torch.testing.assert_close(original["labels"], restored["labels"])


def test_collate_packed_batch_stacks_images() -> None:
    batch = [(torch.zeros((3, 8, 8), dtype=torch.uint8), target) for target in _targets()]

    images, packed = collate_packed_batch(batch)

    assert isinstance(images, torch.Tensor)
    assert images.shape == (3, 3, 8, 8)
    assert images.dtype == torch.uint8
    assert packed["counts"] == [2, 0, 1]
