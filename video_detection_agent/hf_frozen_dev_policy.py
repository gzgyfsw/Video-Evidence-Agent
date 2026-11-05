"""Evaluation-only Dev receipts; generation body matches the Train HF actor.

Only emitted provenance version/partition differ. The Train reader rejects
these items, so Dev output cannot silently become a GRPO/SFT source.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch
from hf_unified_rollout import HFUnifiedStudent, next_step_schema
from check_hf_policy_alignment import fingerprint, write
from hf_request_policy import compile_matcher, prompt_batch, sample_request
from hf_request_rng import request_seed
from request_policy_replay import sha
from unified_student_contract import content, public_context, PUBLIC_CONTEXT_VERSION, POLICY_SHA

VERSION='hf-dev-request-policy-v1'


class FrozenDevStudent(HFUnifiedStudent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.source.get('frozen_split')!='dev':
            raise ValueError('Only explicit original Dev sources are allowed')

    def next_step(self, context, images, out):
        out = Path(out);out.mkdir(parents=True, exist_ok=False)
        context = public_context(context)
        blocks = content(context, self.policy, images)
        order = [];n = 0
        for block in blocks:
            if block['type'] == 'text':order.append(block)
            else:
                order.append({'type': 'image_reference', **images[n]});n += 1
        request = {'model': self.model, 'stage': 'unified_next_step', 'content_order': order,
            'images': images, 'temperature': .7, 'max_tokens': 512,
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'next_step', 'strict': True,
                                    'schema': next_step_schema(context, images)}},
            'chat_template_kwargs': {'enable_thinking': False}, 'policy_sha256': POLICY_SHA,
            'sampling_contract': self.contract,
            'seed': request_seed(self.seed,len(self.calls),self.request_seed_schedule),
            'request_seed_schedule':self.request_seed_schedule,
            'context': context, 'public_context_version': PUBLIC_CONTEXT_VERSION}
        write(out/'request.json', request)
        batch = prompt_batch(request, self.processor, 'cuda')
        receipt = fingerprint(batch);matcher = compile_matcher(self.compiler, request)
        generator = torch.Generator(device='cuda').manual_seed(request['seed'])
        eos = self.network.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos]);start = time.time()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            ids, scores, masks, raw, finish = sample_request(self.network, batch, matcher,
                self.network.config.text_config.vocab_size, .7, 512, eos, generator)
        text = self.processor.tokenizer.decode(ids, skip_special_tokens=True)
        response = {'backend': 'hf', 'model_binding': self.binding, 'sampling_contract': self.contract,
            'prompt_token_ids': batch['input_ids'][0].tolist(), 'completion_token_ids': ids,
            'sampled_processed_logprobs': scores, 'sampled_raw_T1_logprobs': raw,
            'finish_reason': finish, 'decoded_text_for_inspection_only': text,
            'prompt_tensor_receipt': receipt, 'semantic_support': 'NOT_AUDITED'}
        write(out/'response.json', response)
        np.savez_compressed(out/'masks.npz', bitmask=masks, token_ids=np.array(ids, dtype=np.int64))
        item = {'version': VERSION, 'source_partition': 'dev',
            'source_trajectory_id': self.source['merged_id'], 'source_group': self.source['source_group'],
            'model_binding': self.binding, 'sampling_contract': self.contract, 'kind': 'unified_next_step'}
        for name, filename in [('request','request.json'), ('response','response.json'), ('masks','masks.npz')]:
            item[name+'_path'] = str((out/filename).resolve());item[name+'_sha256'] = sha(out/filename)
        write(out/'item.json', item)
        self.calls.append({'item_path': str((out/'item.json').resolve()), 'tokens': len(ids),
                           'elapsed_seconds': time.time()-start})
        print(json.dumps({'stage': 'HF_NEXT_STEP', 'source': self.source['merged_id'],
            'call': len(self.calls), 'tokens': len(ids), 'finish': finish}), flush=True)
        if finish != 'stop':raise ValueError('Incomplete model JSON; saved output is not repaired')
        return json.loads(text)


def read_dev_item(path):
    path = Path(path)
    item = json.loads(path.read_text())
    if item['version'] != VERSION or item['source_partition'] != 'dev':
        raise ValueError('Only explicitly bound Dev HF requests are allowed')
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
