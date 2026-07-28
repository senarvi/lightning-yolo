from typing import Any, override

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor, optim
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import batched_nms
from torchvision.transforms import functional as T

from .batching import split_targets
from .config import LossConfig, MatchingConfig
from .darknet_network import DarknetNetwork
from .torch_networks import create_network
from .types import BATCH, PRIOR_SHAPES, DetectionLoss, PackedTargetDict


class YOLO(LightningModule):
    """PyTorch Lightning implementation of YOLO that supports the most important features of YOLOv3, YOLOv4, YOLOv5,
    YOLOv7, YOLOv8, Scaled-YOLOv4, and YOLOX.

    *YOLOv3 paper*: `Joseph Redmon and Ali Farhadi <https://arxiv.org/abs/1804.02767>`__

    *YOLOv4 paper*: `Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao <https://arxiv.org/abs/2004.10934>`__

    *YOLOv7 paper*: `Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao <https://arxiv.org/abs/2207.02696>`__

    *Scaled-YOLOv4 paper*: `Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao
    <https://arxiv.org/abs/2011.08036>`__

    *YOLOX paper*: `Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun <https://arxiv.org/abs/2107.08430>`__

    *Implementation*: `Seppo Enarvi <https://github.com/senarvi>`__

    Either loads a Darknet configuration file, or constructs a built-in network. Parameters that are provided to the
    constructor will override parameters defined in a configuration file. It's also possible to read weights that have
    been saved by Darknet when using a Darknet configuration.

    The input is expected to be a list of images. Each image is a tensor with shape ``[channels, height, width]``. The
    images from a single batch will be stacked into a single tensor, so the sizes have to match. Different batches can
    have different image sizes, as long as the size is divisible by the ratio in which the network downsamples the
    input. For architectures that use prior shapes, the shapes should be scaled when the input image size changes.
    Distributional-distance architectures predict one anchor per feature-map location and do not use prior shapes or
    confidence logits.

    During training, the model expects both the image tensors and the packed targets. It's possible to train a model
    using one integer class label per box, but the YOLO model supports also multiple labels per box. For multi-label
    training, simply use a boolean matrix that indicates which classes are assigned to which boxes, in place of the
    class labels. *The targets are packed for the whole batch into a dictionary containing the following tensors*:

    - boxes (``FloatTensor[targets, 4]``): the ground-truth boxes in `(x1, y1, x2, y2)` format
    - labels (``Int64Tensor[targets]`` or ``BoolTensor[targets, classes]``): the class label or a boolean class mask
      for each ground-truth box
    - sample_idxs (``Int64Tensor[targets]``): the index of the image each target belongs to
    - counts (``list[int]``): the number of targets in each image

    :func:`~.yolo.YOLO.forward` method returns all predictions from all detection heads in one tensor with shape
    ``[N, detections, classes + 5]``, where ``detections`` is the total number of anchors across the heads. The
    coordinates are scaled to the input image size. Training and evaluation compute a 1-D tensor of named loss values
    instead. Prior-shape models use overlap, confidence, and class components. Distributional-distance models use
    overlap, classification, and DFL components.

    During inference, the model requires only the image tensor. :func:`~.yolo.YOLO.infer` method filters and
    processes the predictions. If a prediction has a high score for more than one class, it will be duplicated. *The
    processed output is returned in a dictionary containing the following tensors*:

    - boxes (``FloatTensor[N, 4]``): predicted bounding box `(x1, y1, x2, y2)` coordinates in image space
    - scores (``FloatTensor[N]``): detection confidences
    - labels (``Int64Tensor[N]``): the predicted labels for each object

    CLI command::

        # Darknet network configuration
        wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny-3l.cfg
        uv run lightning-yolo fit \
            --model.darknet_config yolov4-tiny-3l.cfg \
            --model.num_classes 80 \
            --data.batch_size 8 \
            --data.num_workers 4 \
            --trainer.accelerator gpu \
            --trainer.devices 8 \
            --trainer.accumulate_grad_batches 2 \
            --trainer.gradient_clip_val 5.0 \
            --trainer.max_epochs=100

        # YOLOv4
        uv run lightning-yolo fit \
            --model.architecture yolov4 \
            --model.num_classes 80 \
            --data.batch_size 8 \
            --data.num_workers 4 \
            --trainer.accelerator gpu \
            --trainer.devices 8 \
            --trainer.accumulate_grad_batches 2 \
            --trainer.gradient_clip_val 5.0 \
            --trainer.max_epochs=100

    Args:
        darknet_config: Path to a Darknet configuration file that defines the network architecture. If not given, a
            YOLOv4 network will be constructed.
        darknet_weights: Path to a Darknet weights file. If both ``darknet_config`` and ``darknet_weights`` are given,
            the network will be initialized by these weights.
        architecture: Name of the built-in architecture to construct when ``darknet_config`` is not given. Supported
            values are "yolov4", "yolov4-tiny", "yolov4-p6", "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x",
            "yolov7-w6", "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x", "yolox-tiny", "yolox-s",
            "yolox-m", and "yolox-l".
        num_channels: Number of input image channels.
        num_classes: Number of object classes.
        prior_shapes: A list of prior box dimensions for prior-shape architectures, used for scaling predicted
            dimensions and possibly for matching targets to anchors. The list should contain (width, height) tuples in
            the network input resolution. There should be `3N` tuples, where `N` defines the number of anchors per
            spatial location. They are assigned to the layers from the lowest (high-resolution) to the highest
            (low-resolution) layer, meaning that you typically want to sort the shapes from the smallest to the largest.
            Distributional-distance architectures reject prior shapes.
        matching: Configuration that controls how targets are assigned to anchors (matching algorithm,
            thresholds, and the task-aligned matching hyperparameters). Distributional-distance architectures use
            task-aligned matching and reject prior-shape matching algorithms. See
            :class:`~lightning_yolo.config.MatchingConfig`.
        loss: Configuration that controls how the detection losses are computed (overlap function, label smoothing, and
            the loss multipliers). Distributional-distance architectures use ``overlap_func`` for the foreground box
            loss and reject confidence-specific options. See :class:`~lightning_yolo.config.LossConfig`.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one. Distributional-distance architectures decode distances from
            anchor points and reject non-default ``xy_scale`` values.
        lr: Learning rate after warmup.
        warmup_epochs: Number of epochs for linear learning rate warmup.
        final_lr_multiplier: Learning rate at the end of training as a fraction of ``lr``.
        momentum: SGD momentum.
        weight_decay: Weight decay for convolution weights.
        confidence_threshold: Postprocessing will remove bounding boxes whose confidence score is not higher than this
            threshold. Only applies during inference, not when calculating test metrics.
        nms_threshold: Non-maximum suppression will remove bounding boxes whose IoU with a higher confidence box is
            higher than this threshold, if the predicted categories are equal.
        detections_per_image: Keep at most this number of highest-confidence detections per image.

    """

    def __init__(
        self,
        darknet_config: str | None = None,
        darknet_weights: str | None = None,
        architecture: str | None = None,
        num_channels: int = 3,
        num_classes: int | None = None,
        prior_shapes: PRIOR_SHAPES | None = None,
        matching: MatchingConfig | None = None,
        loss: LossConfig | None = None,
        xy_scale: float | None = None,
        lr: float = 0.01,  # noqa: ARG002
        warmup_epochs: float = 3.0,  # noqa: ARG002
        final_lr_multiplier: float = 0.01,  # noqa: ARG002
        momentum: float = 0.9,  # noqa: ARG002
        weight_decay: float = 0.0005,  # noqa: ARG002
        confidence_threshold: float = 0.2,
        nms_threshold: float = 0.45,
        detections_per_image: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        loss = loss if loss is not None else LossConfig()

        if darknet_config is not None:
            if architecture is not None:
                raise ValueError("Cannot specify both a Darknet configuration and a built-in architecture.")

            self.network: nn.Module = DarknetNetwork(
                darknet_config,
                darknet_weights,
                in_channels=num_channels,
                num_classes=num_classes,
                prior_shapes=prior_shapes,
                matching=matching if matching is not None else MatchingConfig(),
                loss=loss,
                xy_scale=xy_scale,
            )
        else:
            if architecture is None:
                raise ValueError("Either a Darknet configuration or a built-in architecture must be specified.")
            if num_classes is None:
                raise ValueError("Number of classes must be specified when not using a Darknet configuration.")

            self.network = create_network(
                architecture=architecture,
                num_classes=num_classes,
                in_channels=num_channels,
                prior_shapes=prior_shapes,
                matching=matching,
                loss=loss,
                xy_scale=xy_scale if xy_scale is not None else 1.0,
            )

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.detections_per_image = detections_per_image

        # TorchMetrics needs 100 in the threshold list for the COCO mAP summary, while the last value controls how
        # many post-NMS detections are retained for metric computation.
        thresholds = [1, 100, detections_per_image] if detections_per_image > 100 else [1, 10, 100]
        self._val_map = MeanAveragePrecision(max_detection_thresholds=thresholds)
        self._test_map = MeanAveragePrecision(max_detection_thresholds=thresholds)

    @override
    def forward(self, images: Tensor | list[Tensor]) -> Tensor:
        """Runs an inference forward pass through the network and returns the detections.

        Detections are concatenated from all detection heads. Each head produces a number of detections that depends on
        the size of the feature map and the number of anchors per feature-map cell. Training and evaluation steps
        compute losses through :meth:`_forward_with_losses` instead.

        Args:
            images: A tensor of size ``[batch_size, channels, height, width]`` containing a batch of images or a list
                of image tensors.

        Returns:
            The detections tensor shaped ``[batch_size, detections, classes + 5]``, where ``detections`` is the total
            number of anchors across the heads. Box coordinates are in `(x1, y1, x2, y2)` format and scaled to the
            input image size.

        """
        self.validate_batch(images, None)
        detections, _ = self.network(self._preprocess_images(images), None)
        return torch.cat(detections, 1)

    def _forward_with_losses(
        self, images: Tensor | list[Tensor], targets: PackedTargetDict
    ) -> tuple[Tensor, DetectionLoss]:
        """Runs a training or evaluation forward pass and returns detections with normalized losses.

        Args:
            images: A batch of images as a tensor or a list of image tensors.
            targets: Packed ground-truth targets for the whole batch.

        Returns:
            The concatenated detections and the normalized detection loss aggregated over all heads.

        """
        self.validate_batch(images, targets)
        detections, contributions = self.network(self._preprocess_images(images), targets)
        return torch.cat(detections, 1), DetectionLoss.from_contributions(contributions)

    @staticmethod
    def _preprocess_images(images: Tensor | list[Tensor]) -> Tensor:
        """Stacks images into a batch tensor and converts uint8 pixels to normalized floats.

        Args:
            images: A batch of images as a tensor or a list of equally shaped image tensors.

        Returns:
            A float image batch tensor with pixel values scaled to ``[0, 1]`` when the input is uint8.

        """
        images_tensor = images if isinstance(images, Tensor) else torch.stack(images)
        if images_tensor.dtype == torch.uint8:
            images_tensor = images_tensor.to(torch.float32).div(255.0)
        return images_tensor

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Construct SGD parameter groups and a step-based warmup plus linear decay schedule.

        Returns:
            Optimizer and scheduler configuration accepted by Lightning.

        """
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(
            optimizer, int(self.trainer.estimated_stepping_batches), self.trainer.max_epochs
        )
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _get_optimizer(self) -> optim.Optimizer:
        """Construct the SGD optimizer with weight decay applied only to convolution weights.

        Returns:
            SGD optimizer with Nesterov momentum whose convolution weights use weight decay.

        """
        convolution_weights, no_decay = self._get_parameter_groups()
        momentum = float(self.hparams["momentum"])
        optimizer = optim.SGD(
            no_decay,
            lr=float(self.hparams["lr"]),
            momentum=momentum,
            nesterov=momentum > 0.0,
            weight_decay=0.0,
        )
        optimizer.add_param_group({"params": convolution_weights, "weight_decay": float(self.hparams["weight_decay"])})
        return optimizer

    def _get_lr_scheduler(
        self, optimizer: optim.Optimizer, total_steps: int, total_epochs: int | None
    ) -> optim.lr_scheduler.LRScheduler | None:
        """Construct a step-based linear warmup plus linear decay scheduler.

        Args:
            optimizer: Optimizer whose learning rate is scheduled.
            total_steps: Total number of optimizer steps over the whole training run.
            total_epochs: Total number of training epochs, or ``None`` if unknown.

        Returns:
            Learning-rate scheduler stepped once per optimizer step, or ``None`` if the given
            step and epoch counts are insufficient to construct one.

        """
        if total_steps <= 0 or total_epochs is None or total_epochs <= 0:
            return None

        warmup_epochs = float(self.hparams["warmup_epochs"])
        warmup_steps = round(warmup_epochs * total_steps / total_epochs) if warmup_epochs > 0.0 else 0
        warmup_steps = min(warmup_steps, total_steps)
        decay_steps = total_steps - warmup_steps
        final_lr_multiplier = float(self.hparams["final_lr_multiplier"])

        if warmup_steps < 2:
            return optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=final_lr_multiplier,
                total_iters=max(total_steps - 1, 1),
            )

        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps - 1,
        )
        if decay_steps <= 0:
            return warmup_scheduler

        decay_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=final_lr_multiplier,
            total_iters=decay_steps,
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps - 1],
        )

    def _get_parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Group trainable parameters by weight-decay behavior.

        Returns:
            Convolution weights and all remaining parameters.

        """
        convolution_weights: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []

        for module in self.modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if not parameter.requires_grad:
                    continue
                if isinstance(module, nn.Conv2d) and parameter_name == "weight":
                    convolution_weights.append(parameter)
                else:
                    no_decay.append(parameter)

        return convolution_weights, no_decay

    @override
    def training_step(self, batch: BATCH, batch_idx: int) -> STEP_OUTPUT:
        """Computes the training loss.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors. Targets is a list of target
                dictionaries.
            batch_idx: Index of the current batch.

        Returns:
            A dictionary that includes the training loss in 'loss'.

        """
        images, targets = batch
        _, losses = self._forward_with_losses(images, targets)

        self._log_losses("train", losses, prog_bar=True, sync_dist=False)

        return {"loss": losses.total}

    @override
    def validation_step(self, batch: BATCH, batch_idx: int) -> STEP_OUTPUT | None:
        """Evaluates a batch of data from the validation set.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors. Targets is a list of target
                dictionaries.
            batch_idx: Index of the current batch.

        """
        images, targets = batch
        detections, losses = self._forward_with_losses(images, targets)

        self._log_losses("val", losses, sync_dist=True, batch_size=len(images))

        # For computing the MAP metrics, we want to use a very low confidence threshold.
        detections = self.process_detections(detections, 0.001)
        targets = self.process_targets(targets)
        self._val_map.update(detections, targets)
        return None

    @override
    def on_validation_epoch_end(self) -> None:
        # When continuing training from a checkpoint, it may happen that epoch_end is called without detections. In this
        # case the metrics cannot be computed.
        if not self._val_map.detection_labels:
            return

        map_scores = self._val_map.compute()
        map_scores = {
            "val/" + k: v
            for k, v in map_scores.items()
            if isinstance(v, (int, float)) or (isinstance(v, Tensor) and v.numel() == 1)
        }
        self.log_dict(map_scores, sync_dist=True)
        self._val_map.reset()

    @override
    def test_step(self, batch: BATCH, batch_idx: int) -> STEP_OUTPUT | None:
        """Evaluates a batch of data from the test set.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors. Targets is a list of target
                dictionaries.
            batch_idx: Index of the current batch.

        """
        images, targets = batch
        detections, losses = self._forward_with_losses(images, targets)

        self._log_losses("test", losses, sync_dist=True)

        # For computing the MAP metrics, we want to use a very low confidence threshold.
        detections = self.process_detections(detections, 0.001)
        targets = self.process_targets(targets)
        self._test_map.update(detections, targets)
        return None

    @override
    def on_test_epoch_end(self) -> None:
        # When continuing training from a checkpoint, it may happen that epoch_end is called without detections. In this
        # case the metrics cannot be computed.
        if not self._test_map.detection_labels:
            return

        map_scores = self._test_map.compute()
        map_scores = {
            "test/" + k: v
            for k, v in map_scores.items()
            if isinstance(v, (int, float)) or (isinstance(v, Tensor) and v.numel() == 1)
        }
        self.log_dict(map_scores, sync_dist=True)
        self._test_map.reset()

    @override
    def predict_step(self, batch: BATCH, batch_idx: int, dataloader_idx: int = 0) -> list[dict[str, Tensor]]:
        """Feeds a batch of images to the network and returns the detected bounding boxes, confidence scores, and class
        labels.

        If a prediction has a high score for more than one class, it will be duplicated.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors. Targets is a list of target
                dictionaries.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            A list of dictionaries containing tensors "boxes", "scores", and "labels". "boxes" is a matrix of detected
            bounding box `(x1, y1, x2, y2)` coordinates. "scores" is a vector of confidence scores for the bounding box
            detections. "labels" is a vector of predicted class labels.

        """
        images, _ = batch
        detections = self(images)
        return self.process_detections(detections, self.hparams.confidence_threshold)  # type: ignore

    def infer(self, image: Tensor) -> dict[str, Tensor]:
        """Feeds an image to the network and returns the detected bounding boxes, confidence scores, and class labels.

        If a prediction has a high score for more than one class, it will be duplicated.

        Args:
            image: An input image, a tensor of uint8 values sized ``[channels, height, width]``.

        Returns:
            A dictionary containing tensors "boxes", "scores", and "labels". "boxes" is a matrix of detected bounding
            box `(x1, y1, x2, y2)` coordinates. "scores" is a vector of confidence scores for the bounding box
            detections. "labels" is a vector of predicted class labels.

        """
        if not isinstance(image, Tensor):
            image = T.to_tensor(image)

        was_training = self.training
        self.eval()

        detections = self([image])
        detections = self.process_detections(detections, self.hparams.confidence_threshold)  # type: ignore
        detections = detections[0]

        if was_training:
            self.train()
        return detections

    def process_detections(self, preds: Tensor, confidence_threshold: float) -> list[dict[str, Tensor]]:
        """Splits the detection tensor returned by a forward pass into a list of prediction dictionaries, and filters
        them based on confidence threshold, non-maximum suppression (NMS), and maximum number of predictions.

        If for any single detection there are multiple categories whose score is above the confidence threshold, the
        detection will be duplicated to create one detection for each category. NMS processes one category at a time,
        iterating over the bounding boxes in descending order of confidence score, and removes lower scoring boxes that
        have an IoU greater than the NMS threshold with a higher scoring box.

        The returned detections are sorted by descending confidence. The items of the dictionaries are as follows:
        - boxes (``Tensor[batch_size, N, 4]``): detected bounding box `(x1, y1, x2, y2)` coordinates
        - scores (``Tensor[batch_size, N]``): detection confidences
        - labels (``Int64Tensor[batch_size, N]``): the predicted class IDs

        Args:
            preds: A tensor of detected bounding boxes and their attributes.
            confidence_threshold: Remove bounding boxes whose confidence score is not higher than this threshold.

        Returns:
            Filtered detections. A list of prediction dictionaries, one for each image.

        """

        def process(boxes: Tensor, confidences: Tensor, classprobs: Tensor) -> dict[str, Tensor]:
            scores = classprobs * confidences[..., None]

            # Select predictions with high scores. If a prediction has a high score for more than one class, it will be
            # duplicated.
            idxs, labels = (scores > confidence_threshold).nonzero().T
            boxes = boxes[idxs]
            scores = scores[idxs, labels]

            keep = batched_nms(boxes, scores, labels, self.nms_threshold)
            keep = keep[: self.detections_per_image]
            return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}

        return [process(p[..., :4], p[..., 4], p[..., 5:]) for p in preds]

    def process_targets(self, targets: PackedTargetDict) -> list[dict[str, Tensor]]:
        """Duplicates multi-label targets to create one target for each label.

        Args:
            targets: List of target dictionaries. Each dictionary must contain "boxes" and "labels". "labels" is either
                a one-dimensional list of class IDs, or a two-dimensional boolean class map.

        Returns:
            Single-label targets. A list of target dictionaries, one for each image.

        """

        def process(boxes: Tensor, labels: Tensor, **other: Any) -> dict[str, Any]:
            if labels.ndim == 2:
                idxs, labels = labels.nonzero().T
                boxes = boxes[idxs]
            return {"boxes": boxes, "labels": labels, **other}

        return [process(**target) for target in split_targets(targets)]

    def validate_batch(self, images: Tensor | list[Tensor], targets: PackedTargetDict | None) -> None:
        """Validates the format of a batch of data.

        Args:
            images: A tensor containing a batch of images or a list of image tensors.
            targets: Packed ground-truth targets for the batch, or ``None`` for inference. Must be given in training
                mode.

        """
        if isinstance(images, Tensor):
            batch_size = int(images.shape[0])
        else:
            if not isinstance(images, list):
                raise TypeError(f"Expected images to be a Tensor or a list, got {type(images).__name__}.")
            if not images:
                raise ValueError("No images in batch.")
            batch_size = len(images)
            shape = images[0].shape
            for image in images:
                if not isinstance(image, Tensor):
                    raise ValueError(f"Expected image to be of type Tensor, got {type(image).__name__}.")
                if image.shape != shape:
                    raise ValueError(f"Images with different shapes in one batch: {shape} and {image.shape}")

        if targets is None:
            if self.training:
                raise ValueError("Targets should be given in training mode.")
            return

        if not isinstance(targets, dict):
            raise TypeError(f"Expected packed targets to be a dict, got {type(targets).__name__}.")
        for key in ("boxes", "labels", "sample_idxs", "counts"):
            if key not in targets:
                raise ValueError(f"Packed target dictionary doesn't contain {key}.")
        boxes = targets["boxes"]
        labels = targets["labels"]
        sample_idxs = targets["sample_idxs"]
        counts = targets["counts"]
        if not isinstance(boxes, Tensor):
            raise TypeError(f"Expected target boxes to be of type Tensor, got {type(boxes).__name__}.")
        if not isinstance(labels, Tensor):
            raise TypeError(f"Expected target labels to be of type Tensor, got {type(labels).__name__}.")
        if not isinstance(sample_idxs, Tensor):
            raise TypeError(f"Expected target sample_idxs to be of type Tensor, got {type(sample_idxs).__name__}.")
        if not isinstance(counts, list):
            raise TypeError(f"Expected target counts to be a list, got {type(counts).__name__}.")
        if (boxes.ndim != 2) or (boxes.shape[-1] != 4):
            raise ValueError(f"Expected target boxes to be tensors of shape [N, 4], got {list(boxes.shape)}.")
        if (labels.ndim < 1) or (labels.ndim > 2) or (len(labels) != len(boxes)):
            raise ValueError(
                f"Expected target labels to be tensors of shape [N] or [N, num_classes], got {list(labels.shape)}."
            )
        if sample_idxs.ndim != 1 or len(sample_idxs) != len(boxes):
            raise ValueError(f"Expected target sample_idxs to be tensors of shape [N], got {list(sample_idxs.shape)}.")
        if len(counts) != batch_size:
            raise ValueError(f"Got {batch_size} images, but target counts for {len(counts)} images.")
        if sum(counts) != len(boxes):
            raise ValueError(f"Target counts sum to {sum(counts)}, but there are {len(boxes)} boxes.")

    def _log_losses(
        self,
        prefix: str,
        losses: DetectionLoss,
        *,
        prog_bar: bool = False,
        sync_dist: bool = False,
        batch_size: int | None = None,
    ) -> None:
        """Log named loss components and their total.

        Args:
            prefix: Prefix prepended to each logged metric name, such as ``"train"`` or ``"val"``.
            losses: Normalized detection loss whose components and total are logged.
            prog_bar: Whether to show the individual loss components in the progress bar.
            sync_dist: Whether to synchronize the logged values across distributed processes.
            batch_size: Batch size used to weight the logged values, or ``None`` to let Lightning infer it.

        """
        if batch_size is None:
            for loss_name, loss in zip(losses.names, losses.values, strict=True):
                self.log(f"{prefix}/{loss_name}_loss", loss, prog_bar=prog_bar, sync_dist=sync_dist)
            self.log(f"{prefix}/total_loss", losses.total, sync_dist=sync_dist)
        else:
            for loss_name, loss in zip(losses.names, losses.values, strict=True):
                self.log(
                    f"{prefix}/{loss_name}_loss", loss, prog_bar=prog_bar, sync_dist=sync_dist, batch_size=batch_size
                )
            self.log(f"{prefix}/total_loss", losses.total, sync_dist=sync_dist, batch_size=batch_size)
