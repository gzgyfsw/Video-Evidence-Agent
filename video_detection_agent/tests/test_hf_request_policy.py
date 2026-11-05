"""Independent behavior-score replay tests, not video semantic validation."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hf_request_policy import (VERSION, extend_batch, prompt_batch, sample_request,
                               read_bound_item, score_saved_request)
from request_policy_replay import sha


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{'role': 'user', 'content': [{'type': 'text', 'text': 'current-only'}]}]
        assert kwargs['enable_thinking'] is False
        return 'current-only'

    def __call__(self, **kwargs):
        return {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.ones((1, 2), dtype=torch.long)}


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.arange(32, dtype=torch.float32)/32)

    def forward(self, input_ids, logits_to_keep, use_cache=False, past_key_values=None, **kwargs):
        state = input_ids.cumsum(-1) + (0 if past_key_values is None else past_key_values.total)
        logits = (state.float()/100)[..., None]*self.weights
        logits = logits[:, -logits_to_keep:] if isinstance(logits_to_keep, int) else logits[:, logits_to_keep]
        return SimpleNamespace(logits=logits, past_key_values=SimpleNamespace(total=state[:, -1:]))


def saved(tmp_path, mode='cached'):
    contract = {'backend': 'hf', 'temperature': .7, 'top_p': 1., 'top_k': -1, 'min_p': 0.,
                'thinking': False, 'penalties': 'none', 'dropout': 0, 'dtype': 'bfloat16',
                'attention_implementation': 'sdpa', 'grammar_any_whitespace': False,
                'collection_decode_mode': mode}
    request = {'chat_template_kwargs': {'enable_thinking': False},
               'content_order': [{'type': 'text', 'text': 'current-only'}], 'sampling_contract': contract}
    model = Toy();processor = Processor();prompt = prompt_batch(request, processor, 'cpu')
    with torch.no_grad():
        ids, scores, masks, _, finish = sample_request(model, prompt, None, 32, .7, 4, set(),
                                            torch.Generator().manual_seed(19), mode)
    response = {'model_binding': {'test_checkpoint': 'frozen'}, 'sampling_contract': contract,
                'prompt_token_ids': [1, 2], 'completion_token_ids': ids, 'sampled_processed_logprobs': scores}
    (tmp_path/'request.json').write_text(json.dumps(request))
    (tmp_path/'response.json').write_text(json.dumps(response))
    np.savez(tmp_path/'masks.npz', bitmask=masks, token_ids=np.array(ids))
    item = {'version': VERSION, 'source_partition': 'train', 'sampling_contract': contract,
            'model_binding': response['model_binding']}
    for name, filename in [('request', 'request.json'), ('response', 'response.json'), ('masks', 'masks.npz')]:
        item[name+'_path'] = str(tmp_path/filename);item[name+'_sha256'] = sha(tmp_path/filename)
    path = tmp_path/'item.json';path.write_text(json.dumps(item))
    return model, processor, path, response


@pytest.mark.parametrize('generation', ['cached', 'full_causal'])
@pytest.mark.parametrize('replay', ['cached', 'full_causal'])
def test_actual_sampled_scores_equal_independent_replay_and_gradients_exist(tmp_path, generation, replay):
    model, processor, path, response = saved(tmp_path, generation)
    scores, counts, tokens = score_saved_request(model, processor, path, 'cpu', replay)
    torch.testing.assert_close(scores, torch.tensor(response['sampled_processed_logprobs']))
    assert tokens.tolist() == response['completion_token_ids']
    assert counts.tolist() == [32]*4
    (-scores.sum()).backward()
    assert torch.isfinite(model.weights.grad).all() and model.weights.grad.norm() > 0


def test_actual_response_and_masks_cannot_change(tmp_path):
    _, _, path, _ = saved(tmp_path)
    (tmp_path/'response.json').write_text('{}')
    with pytest.raises(ValueError, match='Frozen response'):read_bound_item(path)


def test_head_precision_cannot_silently_change_replay_distribution(tmp_path):
    model,processor,path,_=saved(tmp_path)
    model._hf_head_precision='float32'
    with pytest.raises(ValueError,match='head computation differs'):
        score_saved_request(model,processor,path,'cpu')


def test_forward_dtype_cannot_silently_change_replay_distribution(tmp_path):
    model,processor,path,_=saved(tmp_path)
    model._hf_compute_dtype='float32'
    with pytest.raises(ValueError,match='forward computation dtype differs'):
        score_saved_request(model,processor,path,'cpu')


def test_test_partition_and_temperature_override_are_rejected(tmp_path):
    _, _, path, _ = saved(tmp_path)
    original = json.loads(path.read_text());changed = copy.deepcopy(original)
    changed['source_partition'] = 'test';path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='Train'):read_bound_item(path)
    changed = copy.deepcopy(original);changed['sampling_contract']['temperature'] = 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='contract'):read_bound_item(path)


def test_extra_history_cannot_replace_original_prefix(tmp_path):
    model, processor, path, _ = saved(tmp_path)
    class Wrong(Processor):
        def __call__(self, **kwargs):
            return {'input_ids': torch.tensor([[9, 1, 2]]), 'attention_mask': torch.ones((1, 3), dtype=torch.long)}
    with pytest.raises(ValueError, match='Replay input differs'):
        score_saved_request(model, Wrong(), path, 'cpu')


def test_extending_request_preserves_original_images_and_order():
    prompt = {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.ones((1, 2), dtype=torch.long),
              'mm_token_type_ids': torch.tensor([[1, 0]]), 'pixel_values': torch.arange(4)}
    result = extend_batch(prompt, [5, 7])
    assert result['input_ids'].tolist() == [[1, 2, 5, 7]]
    assert result['mm_token_type_ids'].tolist() == [[1, 0, 0, 0]]
    assert result['pixel_values'] is prompt['pixel_values']
    assert prompt['input_ids'].tolist() == [[1, 2]]
