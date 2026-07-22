import sys
from pathlib import Path
from typing import Any, override

import torch
from lightning.pytorch.cli import LightningCLI

from .coco_datamodule import COCODetectionDataModule
from .yolo_module import YOLO

__all__ = [
    "COCODetectionDataModule",
    "YOLO",
]


def _migrate_legacy_checkpoint_hparams(hparams: dict[str, Any]) -> dict[str, Any]:
    """Move pre-config-dataclass checkpoint hyperparameters to the current nested schema."""
    migrated = dict(hparams)
    matching = dict(migrated["matching"]) if isinstance(migrated.get("matching"), dict) else {}
    loss = dict(migrated["loss"]) if isinstance(migrated.get("loss"), dict) else {}

    for old_key, new_key in {
        "matching_algorithm": "algorithm",
        "matching_threshold": "threshold",
        "spatial_range": "spatial_range",
        "size_range": "size_range",
        "ignore_bg_threshold": "ignore_bg_threshold",
    }.items():
        value = migrated.pop(old_key, None)
        if value is not None and new_key not in matching:
            matching[new_key] = value

    for old_key, new_key in {
        "overlap_func": "overlap_func",
        "predict_overlap": "predict_overlap",
        "label_smoothing": "label_smoothing",
        "overlap_loss_multiplier": "overlap_multiplier",
        "confidence_loss_multiplier": "confidence_multiplier",
        "class_loss_multiplier": "class_multiplier",
    }.items():
        value = migrated.pop(old_key, None)
        if value is not None and new_key not in loss:
            loss[new_key] = value

    if matching:
        migrated["matching"] = matching
    if loss:
        migrated["loss"] = loss
    return migrated


class YOLOCLI(LightningCLI):
    """LightningCLI with checkpoint migration for historical YOLO hyperparameter names."""

    @override
    def _parse_ckpt_path(self) -> None:
        """Parse checkpoint hyperparameters after migrating legacy matching/loss keys."""
        if not self.config.get("subcommand"):
            return
        ckpt_path = self.config[self.config.subcommand].get("ckpt_path")
        if ckpt_path and Path(ckpt_path).is_file():
            ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
            hparams = _migrate_legacy_checkpoint_hparams(ckpt.get("hyper_parameters", {}))
            hparams.pop("_instantiator", None)
            if not hparams:
                return
            if "_class_path" in hparams:
                hparams = {
                    "class_path": hparams.pop("_class_path"),
                    "dict_kwargs": hparams,
                }
            hparams = {self.config.subcommand: {"model": hparams}}
            try:
                self.config = self.parser.parse_object(hparams, self.config)
            except SystemExit:
                sys.stderr.write("Parsing of ckpt_path hyperparameters failed!\n")
                raise


def main() -> None:
    YOLOCLI(
        YOLO,
        COCODetectionDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )
