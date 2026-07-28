"""Batching helpers for detection dataloaders.

Images are stacked as one uint8 tensor so the batch moves to the GPU in one contiguous transfer and is normalized on the
target device. Target tensors are packed separately: per-image boxes and labels are concatenated into single tensors and
tagged with the image each row belongs to. This replaces dozens of tiny per-image host-to-device copies with a few
contiguous transfers and lets the task-aligned matcher build its padded layout without a Python loop.

"""

import torch
from torch import Tensor

from .types import BATCH, PackedTargetDict, TargetDict


def pack_targets(targets: list[TargetDict]) -> PackedTargetDict:
    """Concatenate per-image target dictionaries into a single packed target dictionary.

    Args:
        targets: Per-image target dictionaries, each with ``boxes`` shaped ``[N_i, 4]`` and ``labels`` shaped
            ``[N_i]`` or ``[N_i, C]``.

    Returns:
        A packed target dictionary with ``boxes`` shaped ``[T, 4]``, ``labels`` shaped ``[T]`` or ``[T, C]``,
        ``sample_idxs`` shaped ``[T]`` mapping each row to its image, and ``counts`` giving the per-image target
        count. ``T`` is the total number of targets in the batch.

    """
    device = targets[0]["boxes"].device
    counts = [int(target["boxes"].shape[0]) for target in targets]
    total = sum(counts)
    if total:
        boxes = torch.cat([target["boxes"] for target in targets], dim=0)
        labels = torch.cat([target["labels"] for target in targets], dim=0)
        sample_idxs = torch.cat(
            [
                torch.full((count,), image_idx, dtype=torch.int64, device=device)
                for image_idx, count in enumerate(counts)
                if count
            ],
            dim=0,
        )
    else:
        first = targets[0]
        boxes = first["boxes"].new_zeros((0, 4))
        labels = first["labels"].new_zeros((0, *first["labels"].shape[1:]))
        sample_idxs = torch.zeros((0,), dtype=torch.int64, device=device)
    return {"boxes": boxes, "labels": labels, "sample_idxs": sample_idxs, "counts": counts}


def collate_packed_batch(batch: list[tuple[Tensor, TargetDict]]) -> BATCH:
    """Collate image-target pairs into a stacked image tensor and packed targets.

    Args:
        batch: List of ``(image, target)`` pairs.

    Returns:
        A tuple ``(images, packed_targets)`` where ``images`` is a ``[B, C, H, W]`` tensor and ``packed_targets`` is the
        concatenated target dictionary produced by :func:`pack_targets`.

    """
    images, targets = zip(*batch, strict=True)
    return torch.stack(images, dim=0), pack_targets(list(targets))


def split_targets(targets: PackedTargetDict) -> list[TargetDict]:
    """Split packed targets back into per-image target dictionaries.

    Args:
        targets: Packed targets.

    Returns:
        A list of per-image target dictionaries.

    """
    counts = targets["counts"]
    boxes = targets["boxes"].split(counts)
    labels = targets["labels"].split(counts)
    return [
        {"boxes": image_boxes, "labels": image_labels} for image_boxes, image_labels in zip(boxes, labels, strict=True)
    ]
