from typing import Any, override

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback, WeightAveraging
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor, optim
from torch.optim.swa_utils import get_ema_multi_avg_fn
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import batched_nms
from torchvision.transforms import functional as T

from .config import LossConfig, MatchingConfig
from .darknet_network import DarknetNetwork
from .torch_networks import create_network
from .types import BATCH, IMAGES, PRIOR_SHAPES, TARGETS


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
    input. Note that if you change the size of the input images, you should also scale the prior shapes (a.k.a.
    anchors).

    During training, the model expects both the image tensors and a list of targets. It's possible to train a model
    using one integer class label per box, but the YOLO model supports also multiple labels per box. For multi-label
    training, simply use a boolean matrix that indicates which classes are assigned to which boxes, in place of the
    class labels. *Each target is a dictionary containing the following tensors*:

    - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in `(x1, y1, x2, y2)` format
    - labels (``Int64Tensor[N]`` or ``BoolTensor[N, classes]``): the class label or a boolean class mask for each
      ground-truth box

    :func:`~.yolo_module.YOLO.forward` method returns all predictions from all detection layers in one tensor with shape
    ``[N, anchors, classes + 5]``, where ``anchors`` is the total number of anchors in all detection layers. The
    coordinates are scaled to the input image size. During training it also returns a dictionary containing the
    classification, box overlap, and confidence losses.

    During inference, the model requires only the image tensor. :func:`~.yolo_module.YOLO.infer` method filters and
    processes the predictions. If a prediction has a high score for more than one class, it will be duplicated. *The
    processed output is returned in a dictionary containing the following tensors*:

    - boxes (``FloatTensor[N, 4]``): predicted bounding box `(x1, y1, x2, y2)` coordinates in image space
    - scores (``FloatTensor[N]``): detection confidences
    - labels (``Int64Tensor[N]``): the predicted labels for each object

    CLI command::

        # Darknet network configuration
        wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny-3l.cfg
        python yolo_module.py fit \
            --model.network_config yolov4-tiny-3l.cfg \
            --data.batch_size 8 \
            --data.num_workers 4 \
            --trainer.accelerator gpu \
            --trainer.devices 8 \
            --trainer.accumulate_grad_batches 2 \
            --trainer.gradient_clip_val 5.0 \
            --trainer.max_epochs=100

        # YOLOv4
        python yolo_module.py fit \
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
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        predict_confidence: Whether the head predicts a confidence (objectness) channel. Set to ``False`` to drop
            confidence supervision and supervise all anchors via classification loss only.
        matching: Configuration that controls how targets are assigned to anchors (matching algorithm, thresholds, and
            the task-aligned matching hyperparameters). See :class:`~lightning_yolo.config.MatchingConfig`.
        loss: Configuration that controls how the detection losses are computed (overlap function, label smoothing, and
            the loss multipliers). See :class:`~lightning_yolo.config.LossConfig`.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
        lr: Learning rate after warmup.
        warmup_epochs: Number of epochs for linear learning rate warmup.
        final_lr_multiplier: Learning rate at the end of training as a fraction of ``lr``.
        momentum: SGD momentum.
        weight_decay: Weight decay for convolution weights.
        ema_decay: Exponential moving average decay.
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
        ema_decay: float = 0.9999,  # noqa: ARG002
        predict_confidence: bool = True,
        confidence_threshold: float = 0.2,
        nms_threshold: float = 0.45,
        detections_per_image: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        matching = matching if matching is not None else MatchingConfig()
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
                matching=matching,
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
                predict_confidence=predict_confidence,
                xy_scale=xy_scale if xy_scale is not None else 1.0,
            )

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.detections_per_image = detections_per_image

        map_thresholds = [1, min(10, self.detections_per_image), self.detections_per_image]
        self._val_map = MeanAveragePrecision(max_detection_thresholds=map_thresholds)
        self._test_map = MeanAveragePrecision(max_detection_thresholds=map_thresholds)

    @override
    def forward(self, images: Tensor | IMAGES, targets: TARGETS | None = None) -> Tensor | tuple[Tensor, Tensor]:
        """Runs a forward pass through the network (all layers listed in ``self.network``), and if training targets are
        provided, computes the losses from the detection layers.

        Detections are concatenated from the detection layers. Each detection layer will produce a number of detections
        that depends on the size of the feature map and the number of anchors per feature map cell.

        Args:
            images: A tensor of size ``[batch_size, channels, height, width]`` containing a batch of images or a list of
                image tensors.
            targets: If given, computes losses from detection layers against these targets. A list of target
                dictionaries, one for each image.

        Returns:
            detections (:class:`~torch.Tensor`), losses (:class:`~torch.Tensor`): Detections, and if targets were
            provided, a dictionary of losses. Detections are shaped ``[batch_size, anchors, classes + 5]``, where
            ``anchors`` is the feature map size (width * height) times the number of anchors per cell. The predicted box
            coordinates are in `(x1, y1, x2, y2)` format and scaled to the input image size.

        """
        self.validate_batch(images, targets)
        images_tensor = images if isinstance(images, Tensor) else torch.stack(images)
        detections, loss_records = self.network(images_tensor, targets)

        detections = torch.cat(detections, 1)
        if targets is None:
            return detections

        loss_sums = torch.stack([record.loss_sums for record in loss_records]).sum(0)
        normalizers = torch.stack([record.normalizers for record in loss_records]).sum(0)
        # Empty batches still supervise background predictions, so a minimum denominator of one keeps those losses
        # finite without changing non-empty batches.
        losses = loss_sums / normalizers.clamp_min(1)
        return detections, losses

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
    def configure_callbacks(self) -> list[Callback]:
        """Configure exponential moving average weights for training and evaluation.

        Returns:
            The EMA callback, or an empty list when ``ema_decay`` is not positive (EMA disabled).

        """
        ema_decay = float(self.hparams["ema_decay"])
        if ema_decay <= 0.0:
            return []
        return [
            WeightAveraging(
                use_buffers=True,
                multi_avg_fn=get_ema_multi_avg_fn(decay=ema_decay),
            )
        ]

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
        _, losses = self(images, targets)
        total_loss = losses.sum()

        self.log("train/overlap_loss", losses[0], prog_bar=True, sync_dist=False)
        self.log("train/confidence_loss", losses[1], prog_bar=True, sync_dist=False)
        self.log("train/class_loss", losses[2], prog_bar=True, sync_dist=False)
        self.log("train/total_loss", total_loss, sync_dist=False)

        return {"loss": total_loss}

    @override
    def validation_step(self, batch: BATCH, batch_idx: int) -> STEP_OUTPUT | None:
        """Evaluates a batch of data from the validation set.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors. Targets is a list of target
                dictionaries.
            batch_idx: Index of the current batch.

        """
        images, targets = batch
        detections, losses = self(images, targets)

        self.log("val/overlap_loss", losses[0], sync_dist=True, batch_size=len(images))
        self.log("val/confidence_loss", losses[1], sync_dist=True, batch_size=len(images))
        self.log("val/class_loss", losses[2], sync_dist=True, batch_size=len(images))
        self.log("val/total_loss", losses.sum(), sync_dist=True, batch_size=len(images))

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
        detections, losses = self(images, targets)

        self.log("test/overlap_loss", losses[0], sync_dist=True)
        self.log("test/confidence_loss", losses[1], sync_dist=True)
        self.log("test/class_loss", losses[2], sync_dist=True)
        self.log("test/total_loss", losses.sum(), sync_dist=True)

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

    def process_targets(self, targets: TARGETS) -> list[dict[str, Tensor]]:
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

        return [process(**t) for t in targets]

    def validate_batch(self, images: Tensor | IMAGES, targets: TARGETS | None) -> None:
        """Validates the format of a batch of data.

        Args:
            images: A tensor containing a batch of images or a list of image tensors.
            targets: A list of target dictionaries or ``None``. If a list is provided, there should be as many target
                dictionaries as there are images.

        """
        if not isinstance(images, Tensor):
            if not isinstance(images, (tuple, list)):
                raise TypeError(f"Expected images to be a Tensor, tuple, or a list, got {type(images).__name__}.")
            if not images:
                raise ValueError("No images in batch.")
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

        if not isinstance(targets, (tuple, list)):
            raise TypeError(f"Expected targets to be a tuple or a list, got {type(images).__name__}.")
        if len(images) != len(targets):
            raise ValueError(f"Got {len(images)} images, but targets for {len(targets)} images.")

        for target in targets:
            if "boxes" not in target:
                raise ValueError("Target dictionary doesn't contain boxes.")
            boxes = target["boxes"]
            if not isinstance(boxes, Tensor):
                raise TypeError(f"Expected target boxes to be of type Tensor, got {type(boxes).__name__}.")
            if (boxes.ndim != 2) or (boxes.shape[-1] != 4):
                raise ValueError(f"Expected target boxes to be tensors of shape [N, 4], got {list(boxes.shape)}.")
            if "labels" not in target:
                raise ValueError("Target dictionary doesn't contain labels.")
            labels = target["labels"]
            if not isinstance(labels, Tensor):
                raise ValueError(f"Expected target labels to be of type Tensor, got {type(labels).__name__}.")
            if (labels.ndim < 1) or (labels.ndim > 2) or (len(labels) != len(boxes)):
                raise ValueError(
                    f"Expected target labels to be tensors of shape [N] or [N, num_classes], got {list(labels.shape)}."
                )
