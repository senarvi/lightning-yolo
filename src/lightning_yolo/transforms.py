import math
from collections.abc import Sequence

import torch
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms import v2

from .types import TargetDict

DetectionSample = tuple[Tensor, TargetDict]


class LetterBox(v2.Transform):
    """Resize a detection sample without changing its aspect ratio and pad it to a fixed size.

    Args:
        image_size: Output image size as ``(height, width)``.
        fill: Pixel value used for padding.

    """

    def __init__(self, image_size: tuple[int, int], fill: int = 114) -> None:
        super().__init__()
        self.image_size = image_size
        self.fill = fill

    def forward(self, image: Tensor, target: TargetDict) -> DetectionSample:
        """Resize and center-pad an image-target pair.

        Args:
            image: Image tensor with shape ``(channels, height, width)``.
            target: Detection target containing bounding boxes and labels.

        Returns:
            Letterboxed image and transformed target.

        """
        input_height, input_width = image.shape[-2:]
        output_height, output_width = self.image_size
        scale = min(output_height / input_height, output_width / input_width)
        resized_height = round(input_height * scale)
        resized_width = round(input_width * scale)

        image, target = v2.Resize(
            (resized_height, resized_width),
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        )(image, target)

        horizontal_padding = output_width - resized_width
        vertical_padding = output_height - resized_height
        left = round(horizontal_padding / 2 - 0.1)
        right = round(horizontal_padding / 2 + 0.1)
        top = round(vertical_padding / 2 - 0.1)
        bottom = round(vertical_padding / 2 + 0.1)
        return v2.Pad((left, top, right, bottom), fill=self.fill)(image, target)


class Mosaic(v2.Transform):
    """Combine four detection samples into a 2x2 mosaic.

    Args:
        image_size: Output image size as ``(height, width)``.
        fill: Pixel value used for an empty tile area.

    """

    def __init__(self, image_size: tuple[int, int], fill: int = 114) -> None:
        super().__init__()
        self.image_size = image_size
        self.fill = fill

    def forward(self, samples: Sequence[DetectionSample]) -> DetectionSample:
        """Compose four image-target pairs into one sample.

        Args:
            samples: Exactly four image-target pairs.

        Returns:
            Mosaic image and concatenated targets.

        """
        if len(samples) != 4:
            raise ValueError(f"Mosaic requires exactly four samples, got {len(samples)}.")

        output_height, output_width = self.image_size
        canvas_height = output_height * 2
        canvas_width = output_width * 2
        center_y = int(torch.randint(output_height // 2, output_height * 3 // 2, ()).item())
        center_x = int(torch.randint(output_width // 2, output_width * 3 // 2, ()).item())
        canvas = torch.full(
            (samples[0][0].shape[-3], canvas_height, canvas_width),
            self.fill,
            dtype=samples[0][0].dtype,
            device=samples[0][0].device,
        )
        boxes: list[Tensor] = []
        labels: list[Tensor] = []

        for sample_idx, (image, target) in enumerate(samples):
            input_height, input_width = image.shape[-2:]
            scale = min(output_height / input_height, output_width / input_width)
            resized_height = min(math.ceil(input_height * scale), output_height)
            resized_width = min(math.ceil(input_width * scale), output_width)
            resized_image, resized_target = v2.Resize(
                (resized_height, resized_width),
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
            )(image, target)

            if sample_idx == 0:
                left, top, right, bottom = (
                    max(center_x - resized_width, 0),
                    max(center_y - resized_height, 0),
                    center_x,
                    center_y,
                )
                source_left, source_top = resized_width - (right - left), resized_height - (bottom - top)
            elif sample_idx == 1:
                left, top, right, bottom = (
                    center_x,
                    max(center_y - resized_height, 0),
                    min(center_x + resized_width, canvas_width),
                    center_y,
                )
                source_left, source_top = 0, resized_height - (bottom - top)
            elif sample_idx == 2:
                left, top, right, bottom = (
                    max(center_x - resized_width, 0),
                    center_y,
                    center_x,
                    min(center_y + resized_height, canvas_height),
                )
                source_left, source_top = resized_width - (right - left), 0
            else:
                left, top, right, bottom = (
                    center_x,
                    center_y,
                    min(center_x + resized_width, canvas_width),
                    min(center_y + resized_height, canvas_height),
                )
                source_left, source_top = 0, 0

            source_right = source_left + right - left
            source_bottom = source_top + bottom - top
            canvas[:, top:bottom, left:right] = resized_image[:, source_top:source_bottom, source_left:source_right]
            sample_boxes = torch.as_tensor(resized_target["boxes"]).clone()
            sample_boxes[:, 0::2] += left - source_left
            sample_boxes[:, 1::2] += top - source_top
            boxes.append(sample_boxes)
            labels.append(resized_target["labels"])

        boxes_tensor = torch.cat(boxes) if boxes else torch.empty((0, 4), device=canvas.device)
        labels_tensor = torch.cat(labels) if labels else torch.empty((0,), dtype=torch.int64, device=canvas.device)
        crop_top = (canvas_height - output_height) // 2
        crop_left = (canvas_width - output_width) // 2
        canvas = canvas[:, crop_top : crop_top + output_height, crop_left : crop_left + output_width]
        boxes_tensor[:, 0::2].sub_(crop_left).clamp_(0, output_width)
        boxes_tensor[:, 1::2].sub_(crop_top).clamp_(0, output_height)
        valid = (boxes_tensor[:, 2] > boxes_tensor[:, 0]) & (boxes_tensor[:, 3] > boxes_tensor[:, 1])
        target: TargetDict = {
            "boxes": tv_tensors.BoundingBoxes(boxes_tensor[valid], format="XYXY", canvas_size=self.image_size),
            "labels": labels_tensor[valid],
        }
        return tv_tensors.Image(canvas), target


class MixUp(v2.Transform):
    """Blend two detection samples and concatenate their targets.

    Args:
        alpha: Concentration parameter for the symmetric beta distribution.

    """

    def __init__(self, alpha: float = 32.0) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.alpha = alpha

    def forward(self, samples: Sequence[DetectionSample]) -> DetectionSample:
        """Blend two image-target pairs using a beta distribution.

        Args:
            samples: Exactly two image-target pairs with equal image shapes.

        Returns:
            Blended image and concatenated targets.

        """
        if len(samples) != 2:
            raise ValueError(f"MixUp requires exactly two samples, got {len(samples)}.")
        (image1, target1), (image2, target2) = samples
        if image1.shape != image2.shape:
            raise ValueError(f"MixUp images must have equal shapes, got {image1.shape} and {image2.shape}.")

        ratio = torch.distributions.Beta(self.alpha, self.alpha).sample().to(image1.device)
        mixed = image1.to(torch.float32).mul(ratio).add_(image2.to(torch.float32).mul(1.0 - ratio))
        if not image1.dtype.is_floating_point:
            mixed = mixed.round().clamp_(0, torch.iinfo(image1.dtype).max).to(image1.dtype)

        target: TargetDict = {
            "boxes": tv_tensors.BoundingBoxes(
                torch.cat((target1["boxes"], target2["boxes"])),
                format="XYXY",
                canvas_size=image1.shape[-2:],
            ),
            "labels": torch.cat((target1["labels"], target2["labels"])),
        }
        return tv_tensors.Image(mixed), target
