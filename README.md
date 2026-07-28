# YOLO

The YOLO model has evolved quite a bit, since the original publication in 2016. The original source code was
written in C, using a framework called [Darknet](https://github.com/pjreddie/darknet). The final revision by the
original author was called YOLOv3 and described in an [arXiv paper](https://arxiv.org/abs/1804.02767). Later
various other authors have written implementations that improve different aspects of the model or the training
procedure. [YOLOv4 implementation](https://github.com/AlexeyAB/darknet) was still based on Darknet and
[YOLOv5](https://github.com/ultralytics/yolov5) was written using PyTorch. Most other implementations are based on
these.

This PyTorch Lightning implementation combines features from some of the notable YOLO implementations. The most
important papers are:

- *YOLOv3*: [Joseph Redmon and Ali Farhadi](https://arxiv.org/abs/1804.02767)
- *YOLOv4*: [Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao](https://arxiv.org/abs/2004.10934)
- *YOLOv7*: [Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao](https://arxiv.org/abs/2207.02696)
- *Scaled-YOLOv4*: [Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao](https://arxiv.org/abs/2011.08036)
- *YOLOX*: [Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun](https://arxiv.org/abs/2107.08430)

## Network Architecture

Any network can be used with YOLO detection heads as long as it produces feature maps with the correct number of
features. Typically the network consists of a CNN backbone combined with a
[Feature Pyramid Network](https://arxiv.org/abs/1612.03144) or a
[Path Aggregation Network](https://arxiv.org/abs/1803.01534). Backbone layers reduce the size of the feature map and
the network may contain multiple detection heads that operate at different resolutions.

The user can write the network architecture in PyTorch, or construct a computational graph based on a Darknet
configuration file using the
[`DarknetNetwork`](https://github.com/Lightning-AI/lightning-bolts/tree/master/pl_bolts/models/detection/yolo/darknet_network.py)
class. The network object is passed to the YOLO constructor in the `network` argument. `DarknetNetwork` is also able
to read weights from a Darknet model file.

There are several network architectures included in the
[`torch_networks`](https://github.com/Lightning-AI/lightning-bolts/tree/master/pl_bolts/models/detection/yolo/torch_networks.py)
module (YOLOv4, YOLOv5, YOLOv7, YOLOv8, YOLOX). Larger and smaller variants of these models can be created by
varying the `width` and `depth` arguments.

## Detection Heads

This project uses the following terminology:

- An **anchor** is one prediction candidate. Every detection head predicts a box and class scores for each anchor.
- An **anchor point** is the anchor's `(x, y)` reference location. Feature-map geometry determines the anchor points.
- A **prior shape** is an optional `(width, height)` template attached to an anchor. It provides a reference size for
  box regression and is separate from the anchor point.
- A **prior-shape head** predicts one or more anchors per feature-map cell and regresses each box relative to its anchor
  point and prior shape.
- A **distributional-distance head** predicts one anchor per feature-map cell and regresses distributions for the
  left, top, right, and bottom distances from the anchor point. It uses no prior shapes.

Thus, all supported detection architectures use anchors and anchor points. The distinction is whether box regression
uses prior shapes or distributional distances.

Prior-shape heads typically predict several anchors per feature-map cell. The number of features predicted per cell is
`(5 + num_classes) * anchors_per_cell` when the head predicts box coordinates, confidence, and class probabilities.
The configured prior shapes are normally obtained by clustering bounding-box dimensions in the training data. They
should be scaled when the network input size changes.

The prior shapes are also used for matching the ground-truth targets to anchors during training. With the exception of
the SimOTA and TAL matching algorithms, targets are matched only to anchors from the closest grid cell. The prior shapes
are used to determine, to which anchors from that cell the target is matched. The losses are computed between the target
boxes and the predictions that correspond to their matched anchors. Different matching rules have been implemented:

- *maxiou*: The original matching rule that matches a target to the prior shape that gives the highest IoU.
- *iou*: Matches a target to an anchor, if the IoU between the target and the prior shape is above a threshold.
  Multiple anchors may be matched to the same target, and the loss will be computed from a number of pairs that is
  generally not the same as the number of ground-truth boxes.
- *size*: Calculates the ratio between the width and height of the target box to the prior width and height. If both
  the width and the height are close enough to the prior shape, matches the target to the anchor.
- *simota*: The SimOTA matching algorithm from YOLOX. Targets can be matched not only to anchors from the closest
  grid cell, but to any anchors that are inside the target bounding box and whose prior shape is close enough to the
  target shape. The matching algorithm is based on Optimal Transport and uses the training loss between the target and
  the predictions as the cost. That is, the prior shapes are not used for matching, but the predictions corresponding
  to the anchors.
- *tal*: Task-aligned matching. For each target, anchors whose anchor point is inside the target box are ranked using an
  alignment score based on class confidence and box overlap, and the top-k anchors are selected. Like SimOTA, this rule
  uses predictions for matching instead of prior shapes.

Distributional-distance heads use decoupled box and classification branches. Each feature-map location predicts one
anchor with `4 * num_dfl_bins` distance logits and `num_classes` class logits. The default `num_dfl_bins` is 16, so
every anchor predicts 64 regression logits. The expected left, top, right, and bottom distances are decoded from the
distributions and converted to pixel-space `(x1, y1, x2, y2)` boxes relative to the anchor point.

Currently YOLOv8 is the only distributional-distance architecture. It has no confidence/objectness logit. Its training
losses are overlap, classification, and DFL. Public detections still use the common `[batch, detections, 5 + classes]`
shape for post-processing and export compatibility. In those public tensors, the confidence column is filled with ones,
so the post-processed score is the class probability.

Distributional-distance architectures reject options specific to the prior-shape or confidence path: `prior_shapes`,
matching configuration, non-default `xy_scale`, confidence loss settings, and overlap-prediction settings. Their
foreground overlap loss respects `overlap_func`, label smoothing applies to foreground classification targets, and TAL
matching uses complete overlap for assignment consistency.

## Input Data

The model input is a tensor with shape `[batch, channels, height, width]`. The provided data modules emit uint8
tensors, which the model converts to floating point and normalizes on the target device. Floating-point image tensors
are also accepted. Different batches can have different image sizes. The feature pyramid network introduces another
constraint on the image size: the width and the height have to be divisible by the ratio in which the network
downsamples the input.

During training, targets are packed into one dictionary for the whole batch. It is possible to train a model using one
integer class label per box, or a boolean class mask for multi-label training. The packed dictionary contains:

- *boxes*: `(x1, y1, x2, y2)` coordinates of all ground-truth boxes in a tensor with shape `[T, 4]`.
- *labels*: Integer class labels with shape `[T]` or boolean class masks with shape `[T, classes]`.
- *batch_indices*: Image index for each target, with shape `[T]`.
- *counts*: Number of targets in each image, as a list of length `batch`.

Here `T` is the total number of targets in the batch. Data loaders should use `collate_packed_batch` to stack images
and construct this representation.

## Training

The `YOLO` class defined in [yolo.py](src/lightning_yolo/yolo.py) is a `LightningModule` that can be used with
PyTorch Lightning Trainer. First the module creates a network, either from a Darknet configuration file, or using one
of the built-in PyTorch networks.

A data module for the COCO object detection dataset is provided for demonstration purposes. The data module needs to
resize the data to a suitable size, in addition to any augmenting transforms. For example, YOLOv4 network requires
that the width and the height are multiples of 32.

There's also a command line tool `lightning-yolo` that demonstrates training using Lightning CLI. It downloads the
COCO dataset automatically.

### Darknet fine-tuning example

This example fine-tunes a YOLOv4-tiny model, loading the architecture and the pretrained weights from Darknet files.

```bash
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny-3l.cfg
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.conv.29
sed -i 's/iou_normalizer=.*/iou_normalizer=5.0/' yolov4-tiny-3l.cfg

uv run lightning-yolo fit \
  --model.darknet_config yolov4-tiny-3l.cfg \
  --model.darknet_weights yolov4-tiny.conv.29 \
  --model.num_classes 80 \
  --data.data_dir data \
  --data.batch_size 64 \
  --trainer.precision 16-mixed \
  --trainer.accumulate_grad_batches 2 \
  --trainer.gradient_clip_val 10.0 \
  --trainer.max_epochs 20
```

### Configuration files

Training hyperparameters can be provided through Lightning YAML config files.

You can also print the full default config and customize it:

```bash
uv run lightning-yolo fit --print_config >config.yaml
uv run lightning-yolo fit --config config.yaml
```

To list all available CLI options:

```bash
uv run lightning-yolo fit -h
```

### Example configurations

Ready-to-run YOLOv8 examples are available in the [cfg directory](cfg). For example:

```bash
uv run lightning-yolo fit --config cfg/yolov8n.yaml
```

Metrics are written to `runs/<recipe>/csv/version_0/metrics.csv`, and checkpoints are written under
`runs/<recipe>/checkpoints/`.

## Inference

During inference, the model requires only the input images. `forward()` method receives a mini-batch of images in a
tensor with shape `[N, channels, height, width]`.

`forward()` returns the predictions from all detection heads in a tensor with shape `[N, detections, classes + 5]`,
where `detections` is the total number of anchors in all detection heads. The predictions are `x1`,
`y1`, `x2`, `y2`, confidence, and the probability for each class. The coordinates are scaled to the input image size.

`infer()` method filters and processes the predictions. A class-specific score is obtained by multiplying the class
probability with the detection confidence. Only detections with a high enough score are kept. YOLO does not use
`softmax` to normalize the class probabilities, but each probability is normalized individually using `sigmoid`.
Consequently, one object can be assigned to multiple categories. If more than one class has a score that is above the
confidence threshold, these will be split into multiple detections during postprocessing. Then the detections are
filtered using non-maximum suppression. The processed output is returned in a dictionary containing the following
tensors:

- *boxes*: a matrix of predicted bounding box `(x1, y1, x2, y2)` coordinates in image space
- *scores*: a vector of detection confidences
- *labels*: a vector of predicted class labels

## Development

This repository uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

Before every commit, run the pre-commit hooks.

```bash
uv run pre-commit run -a
```

Optionally, install the git hook so that pre-commit runs automatically before each commit.

```bash
uv run pre-commit install
```

The pre-commit hooks cannot run Mypy. You have to run it separately.

```bash
uv run mypy
```

The pre-commit hooks do run the unit tests, but if you want, you can also run them separately.

```bash
uv run pytest -v
```
