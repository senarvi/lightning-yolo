import torch

from lightning_yolo.target_matching import TALMatching, _sim_ota_match


def test_sim_ota_match() -> None:
    # For each of the two targets, k will be the sum of the IoUs. 2 and 1 predictions will be selected for the first and
    # the second target respectively.
    ious = torch.tensor([[0.1, 0.2], [0.1, 0.3], [0.9, 0.4], [0.9, 0.1]])
    # Costs will determine that the first and the last prediction will be selected for the first target, and the first
    # prediction will be selected for the second target. The first prediction was selected for two targets, but it will
    # be matched to the best target only (the second one).
    costs = torch.tensor([[0.3, 0.1], [0.5, 0.2], [0.4, 0.5], [0.3, 0.3]])
    matched_preds, matched_targets = _sim_ota_match(costs, ious)

    # The first and the last prediction were matched.
    assert len(matched_preds) == 4
    assert matched_preds[0]
    assert not matched_preds[1]
    assert not matched_preds[2]
    assert matched_preds[3]

    # The first prediction was matched to the target 1 and the last prediction was matched to target 0.
    assert len(matched_targets) == 2
    assert matched_targets[0] == 1
    assert matched_targets[1] == 0


def test_tal_matching() -> None:
    matcher = TALMatching(prior_shapes=[[2, 2]], prior_shape_idxs=[0], topk=1, alpha=0.5, beta=6.0)

    preds = {
        "boxes": torch.tensor([[[[0.0, 0.0, 1.0, 1.0]], [[1.0, 0.0, 2.0, 1.0]]]]),
        "confidences": torch.zeros((1, 2, 1)),
        "classprobs": torch.tensor([[[[8.0, -8.0]], [[-8.0, 8.0]]]]),
    }
    targets = {
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 2.0, 1.0]]),
        "labels": torch.tensor([0, 1]),
        "polygons": torch.empty((0, 8)),
    }
    image_size = torch.tensor([2.0, 1.0])

    pred_selector, background_mask, target_selector = matcher(preds, targets, image_size)

    anchor_y, anchor_x, anchor_idx = pred_selector
    assert torch.equal(anchor_y, torch.tensor([0, 0]))
    assert torch.equal(anchor_x, torch.tensor([0, 1]))
    assert torch.equal(anchor_idx, torch.tensor([0, 0]))
    assert torch.equal(target_selector, torch.tensor([0, 1]))

    assert background_mask.shape == torch.Size([1, 2, 1])
    assert not background_mask[0, 0, 0]
    assert not background_mask[0, 1, 0]
