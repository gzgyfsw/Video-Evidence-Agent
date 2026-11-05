"""Drop only a fully masked textual suffix from a single-state causal batch."""
import torch


def trim_masked_suffix(batch, route_mask):
    """Caller guarantees all multimodal inputs precede the assistant target.

    Never use this on full trajectories: their later images have separate grids.
    Images, grids and all prefix tokens are retained. Labels/targets are unchanged.
    """
    labels=batch['labels']
    if labels.ndim!=2 or labels.shape[0]!=1:raise ValueError('single-state batch size must be one')
    positions=torch.nonzero(labels[0].ne(-100),as_tuple=False).flatten()
    if not positions.numel():raise ValueError('no supervised tokens')
    end=int(positions[-1])+1
    length=labels.shape[1]
    if route_mask.shape!=labels.shape:raise ValueError('route mask shape mismatch')
    result=dict(batch)
    for key in ['input_ids','attention_mask','labels','token_type_ids','mm_token_type_ids','position_ids']:
        if key in result:
            value=result[key]
            if value.shape[-1]!=length:raise ValueError(f'{key}: sequence axis mismatch')
            result[key]=value[..., :end].contiguous()
    return result,route_mask[..., :end].contiguous()
