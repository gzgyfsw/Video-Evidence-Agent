"""Single-request HF sampling/replay primitives. No rewards or optimizer.

This is a new HF behavior policy, not a correction to recorded vLLM scores.
Sampling and replay consume actual token IDs and the same frozen pixels.
"""
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from actor_request_credit import request_batch
from actor_policy_probabilities import allowed_from_bitmask, constrained_token_logprobs
from request_policy_replay import sha

VERSION = 'hf-request-policy-v1'


def prompt_batch(request, processor, device):
    if request['chat_template_kwargs'] != {'enable_thinking': False}:
        raise ValueError('Only explicit thinking=False is supported')
    batch = request_batch(request, processor)
    if batch['input_ids'].shape[0] != 1 or not bool((batch['attention_mask'] == 1).all()):
        raise ValueError('Exactly one unpadded request is required')
    return {k: v.to(device) for k, v in batch.items()}


def extend_batch(prompt, completion):
    """Append original output IDs inside ONE request, never other calls."""
    result = dict(prompt)
    device = prompt['input_ids'].device
    suffix = torch.tensor([completion], dtype=torch.long, device=device)
    result['input_ids'] = torch.cat([prompt['input_ids'], suffix], dim=1)
    for key in ['attention_mask', 'mm_token_type_ids']:
        if key in prompt:
            extension = torch.full_like(suffix, 1 if key == 'attention_mask' else 0)
            result[key] = torch.cat([prompt[key], extension], dim=1)
    return result


def forward_logits(model, batch, positions, use_cache=False, past_key_values=None):
    batch = dict(batch)
    if 'pixel_values' in batch:
        batch['pixel_values'] = batch['pixel_values'].to(next(model.parameters()).dtype)
    scope=(torch.autocast(batch['input_ids'].device.type,enabled=False)
           if getattr(model,'_hf_compute_dtype','bfloat16')=='float32' else nullcontext())
    with scope:
        output = model(**batch, use_cache=use_cache, past_key_values=past_key_values,
                       logits_to_keep=positions)
    return output.logits[0].float(), output.past_key_values


def compile_matcher(compiler, request):
    import xgrammar as xgr
    output = request.get('response_format')
    if output is None:
        return None
    if output['type'] != 'json_schema':
        raise ValueError('Unsupported response grammar')
    grammar = compiler.compile_json_schema(json.dumps(output['json_schema']['schema']),
                                            any_whitespace=False)
    return xgr.GrammarMatcher(grammar, max_rollback_tokens=0)


def sample_request(model, prompt, matcher, vocab_size, temperature, limit, eos_ids,
                   generator, decode_mode='cached'):
    """Record behavior logprobs at sampling time, before any replay exists."""
    import xgrammar as xgr
    if not 0 < temperature <= 2 or limit < 1 or decode_mode not in {'cached', 'full_causal'}:
        raise ValueError('Invalid explicit sampling configuration')
    device = prompt['input_ids'].device
    workspace = xgr.allocate_token_bitmask(1, vocab_size)
    ids, scores, masks, raw_scores = [], [], [], []
    cache = None
    for i in range(limit):
        if decode_mode == 'cached' and i:
            batch = {'input_ids': torch.tensor([[ids[-1]]], device=device),
                     'mm_token_type_ids': torch.zeros((1, 1), dtype=torch.long, device=device)}
        else:
            batch = extend_batch(prompt, ids) if ids else prompt
        logits, cache = forward_logits(model, batch, 1, use_cache=decode_mode == 'cached',
                                       past_key_values=cache if decode_mode == 'cached' else None)
        workspace.fill_(-1)
        if matcher is not None:
            matcher.fill_next_token_bitmask(workspace)
        mask = workspace[0].numpy().copy()
        allowed = allowed_from_bitmask(mask[None], vocab_size, device)
        if not bool(allowed.any()):
            raise ValueError('Grammar permits no output token')
        distribution = (logits.masked_fill(~allowed, -torch.inf) / temperature).log_softmax(-1)
        token = int(torch.multinomial(distribution.exp(), 1, generator=generator)[0, 0])
        scores.append(float(distribution[0, token]));ids.append(token);masks.append(mask)
        raw_scores.append(float(logits.log_softmax(-1)[0, token]))
        if matcher is not None and not matcher.accept_token(token):
            raise ValueError('Sampled token rejected by its actual grammar')
        if token in eos_ids or (matcher is not None and matcher.is_terminated()):
            return ids, scores, np.stack(masks), raw_scores, 'stop'
    return ids, scores, np.stack(masks), raw_scores, 'length'


def read_bound_item(path):
    path = Path(path)
    item = json.loads(path.read_text())
    if item['version'] != VERSION or item['source_partition'] != 'train':
        raise ValueError('Only explicitly bound Train HF requests are allowed')
    for name in ['request', 'response', 'masks']:
        if sha(item[f'{name}_path']) != item[f'{name}_sha256']:
            raise ValueError(f'Frozen {name} changed')
    request = json.loads(Path(item['request_path']).read_text())
    response = json.loads(Path(item['response_path']).read_text())
    if (request['sampling_contract'] != item['sampling_contract']
            or response['sampling_contract'] != item['sampling_contract']
            or response['model_binding'] != item['model_binding']):
        raise ValueError('Behavior contract or checkpoint mismatch')
    contract = item['sampling_contract']
    if (contract['backend'] != 'hf' or contract['temperature'] != .7
            or contract['top_p'] != 1 or contract['top_k'] != -1
            or contract['min_p'] != 0 or contract['thinking'] is not False
            or contract['penalties'] != 'none' or contract['grammar_any_whitespace'] is not False
            or contract['attention_implementation'] != 'sdpa' or contract['dtype'] not in {'bfloat16','float32'}
            or contract['dropout'] != 0):
        raise ValueError('Unsupported HF behavior distribution')
    if contract.get('lm_head_compute','bfloat16') not in {'bfloat16','float32'}:
        raise ValueError('Unsupported output-head computation contract')
    return item, request, response


def score_saved_request(model, processor, path, device='cuda', decode_mode='full_causal'):
    item, request, response = read_bound_item(path)
    if getattr(model,'_hf_head_precision','bfloat16') != item['sampling_contract'].get('lm_head_compute','bfloat16'):
        raise ValueError('Sampling/replay output-head computation differs')
    if getattr(model,'_hf_compute_dtype','bfloat16') != item['sampling_contract']['dtype']:
        raise ValueError('Sampling/replay forward computation dtype differs')
    return score_token_probe(model,processor,item,request,response,device,decode_mode)


def score_token_probe(model,processor,item,request,response,device='cuda',decode_mode='full_causal'):
    """Numeric primitive on fixed tokens. Alone it is NOT behavior verification.

    Training must use score_saved_request, which checks the actual behavior
    computation contract first. Precision diagnostics use this primitive only
    to compare two forward paths; they must not replace sampled probabilities.
    """
    prompt = prompt_batch(request, processor, device)
    if prompt['input_ids'][0].tolist() != response['prompt_token_ids']:
        raise ValueError('Replay input differs from actual sampling input')
    ids = response['completion_token_ids']
    if not ids or len(ids) != len(response['sampled_processed_logprobs']):
        raise ValueError('Actual output IDs/probabilities missing')
    targets = torch.tensor(ids, dtype=torch.long, device=device)
    if decode_mode == 'full_causal':
        batch = extend_batch(prompt, ids[:-1])
        positions = torch.arange(len(ids), device=device) + len(response['prompt_token_ids']) - 1
        logits, _ = forward_logits(model, batch, positions)
    elif decode_mode == 'cached':
        logits, cache = forward_logits(model, prompt, 1, use_cache=True)
        rows = [logits[-1]]
        for token in ids[:-1]:
            batch = {'input_ids': torch.tensor([[token]], device=device),
                     'mm_token_type_ids': torch.zeros((1, 1), dtype=torch.long, device=device)}
            logits, cache = forward_logits(model, batch, 1, use_cache=True, past_key_values=cache)
            rows.append(logits[-1])
        logits = torch.stack(rows)
    else:
        raise ValueError('Unknown replay mode')
    with np.load(item['masks_path']) as arrays:
        if arrays['token_ids'].tolist() != ids:
            raise ValueError('Grammar/actual-token alignment mismatch')
        allowed = allowed_from_bitmask(arrays['bitmask'], logits.shape[-1], device)
    scores = constrained_token_logprobs(logits, targets, allowed, item['sampling_contract']['temperature'])
    return scores, allowed.sum(-1), targets
