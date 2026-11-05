"""Exact mean causal loss on supervised positions (batch size one)."""
import torch
import torch.nn.functional as F


def selected_targets(labels):
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError('sparse supervised logits require batch size one')
    positions = torch.nonzero(labels[0, 1:].ne(-100), as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError('no causally predictable supervised tokens')
    return positions, labels[:, positions + 1]


def selected_loss(logits, targets):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))


def selected_subset_loss(logits, targets, subset):
    """Mean CE for a nonempty boolean subset of selected causal positions."""
    if subset.shape != targets.shape:
        raise ValueError('subset shape must equal selected target shape')
    if not bool(subset.any()):
        raise ValueError('selected subset is empty')
    return F.cross_entropy(logits[subset].float(), targets[subset])
