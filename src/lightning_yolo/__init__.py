from lightning.pytorch.cli import LightningCLI

from .coco_datamodule import COCODetectionDataModule
from .yolo_module import YOLO

__all__ = [
    "COCODetectionDataModule",
    "YOLO",
]


def main() -> None:
    LightningCLI(YOLO, COCODetectionDataModule, seed_everything_default=42)
