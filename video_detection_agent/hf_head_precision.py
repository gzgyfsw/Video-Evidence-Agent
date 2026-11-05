"""Explicit computation contract; never changes saved weights or behavior logs."""
import torch
import torch.nn.functional as F


def float_head_hook(module, inputs, output):
    with torch.autocast(device_type=inputs[0].device.type, enabled=False):
        bias=None if module.bias is None else module.bias.float()
        return F.linear(inputs[0].float(),module.weight.float(),bias)


def configure_head(model, precision='bfloat16'):
    if precision not in {'bfloat16','float32'}:raise ValueError('Unknown output head precision')
    if hasattr(model,'_hf_head_precision'):
        if model._hf_head_precision!=precision:raise ValueError('Cannot switch head computation mid-run')
        return
    model._hf_head_precision=precision
    if precision=='float32':
        model.get_output_embeddings().register_forward_hook(float_head_hook)
