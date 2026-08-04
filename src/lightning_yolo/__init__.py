from lightning.pytorch.cli import LightningCLI

from .coco_datamodule import COCODetectionDataModule
from .yolo import YOLO

__all__ = ["COCODetectionDataModule", "YOLO"]


def main() -> None:
    """Entry point for the ``lightning-yolo`` CLI."""
    LightningCLI(YOLO, COCODetectionDataModule, seed_everything_default=42, save_config_kwargs={"overwrite": True})
