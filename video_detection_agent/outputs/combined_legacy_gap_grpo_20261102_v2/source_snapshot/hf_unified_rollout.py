"""One HF Student, actual tools, and replayable independent request receipts."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from agent.tools import TemporalEvidenceExpander
from agent.video import extract_frames
from check_hf_policy_alignment import fingerprint, write
from hf_request_policy import VERSION, compile_matcher, prompt_batch, sample_request
from request_policy_replay import sha
from hf_request_rng import request_seed,LEGACY
from unified_student_contract import (content, freeze_runtime_image, POLICY_SHA,
                                      public_context, PUBLIC_CONTEXT_VERSION,
                                      remaining_frames, budget_exhausted)


def sampling_contract(head_precision='bfloat16',compute_dtype='bfloat16'):
    if head_precision not in {'bfloat16','float32'}:raise ValueError('Unknown head precision')
    if compute_dtype not in {'bfloat16','float32'}:raise ValueError('Unknown forward dtype')
    result = {'backend': 'hf', 'temperature': .7, 'top_p': 1., 'top_k': -1,
        'min_p': 0., 'thinking': False, 'penalties': 'none', 'dropout': 0,
        'attention_implementation': 'sdpa', 'dtype': compute_dtype,
        'grammar_any_whitespace': False, 'collection_decode_mode': 'cached',
        'logprobs': 'masked_temperature_log_softmax_before_sampling'}
    if head_precision=='float32':result['lm_head_compute']='float32'
    return result


def next_step_schema(context, images):
    """Constrain executability, never select the next action or a risk label."""
    from check_selected_sft_live import tool_payload_schema
    anchors = {im['timestamp'] for im in images}
    for event in context['events']:
        if event['type'] == 'tool_result':
            anchors.update(c['timestamp'] for c in event['result'].get('candidates', []))
    remaining = remaining_frames(context['budget'], len(images))
    used = sum(e['type'] == 'decision' and e['action'] != 'ANSWER' for e in context['events'])
    branches = []
    if not budget_exhausted(context['budget'], used, len(images)):
        branches.append(tool_payload_schema('JOINT_SEMANTIC_SCAN'))
        expansion = tool_payload_schema('TEMPORAL_EXPAND', sorted(anchors), remaining)
        if remaining is None:
            expansion['properties']['num_frames'].pop('maximum', None)
        branches.append(expansion)
    times = sorted({im['timestamp'] for im in images})
    evidence = {'type': 'array', 'minItems': 1, 'maxItems': 3, 'items': {
        'type': 'object', 'additionalProperties': False, 'required': ['timestamp', 'description'],
        'properties': {'timestamp': {'type': 'number', 'enum': times},
                       'description': {'type': 'string', 'minLength': 1, 'maxLength': 512}}}}
    for category, level, disposition, reason in [
            (['Harmless'], ['L0'], 'ALLOW', None),
            (['Sexual Content','Violence & Dangerous','Sensitive Figures & Ideologies'], ['L1'], 'REVIEW', 'CONFIRMED_L1'),
            (['Sexual Content','Violence & Dangerous','Sensitive Figures & Ideologies'], ['L2'], 'BLOCK', None),
            (['Harmless','Sexual Content','Violence & Dangerous','Sensitive Figures & Ideologies', None], [None], 'REVIEW', 'EVIDENCE_INSUFFICIENT')]:
        branches.append({'type': 'object', 'additionalProperties': False,
            'required': ['action','category','severity','moderation_action','review_reason','evidence'],
            'properties': {'action': {'const': 'ANSWER'}, 'category': {'enum': category},
                'severity': {'enum': level}, 'moderation_action': {'const': disposition},
                'review_reason': {'const': reason}, 'evidence': evidence}})
    return {'anyOf': branches}


class HFUnifiedStudent:
    def __init__(self, network, processor, compiler, binding, policy, source, seed,request_seed_schedule=LEGACY):
        self.network, self.processor, self.compiler = network, processor, compiler
        self.binding, self.policy, self.source, self.seed = binding, policy, source, seed
        self.request_seed_schedule=request_seed_schedule
        self.model = 'hf-unified-fixed-student'
        self.calls = []
        self.contract = sampling_contract(getattr(network,'_hf_head_precision','bfloat16'),
                                          getattr(network,'_hf_compute_dtype','bfloat16'))

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
        item = {'version': VERSION, 'source_partition': 'train',
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


class RealVideoTools:
    def __init__(self, source, locator, out, max_search_new_frames=None):
        self.source, self.locator, self.out = source, locator, Path(out)
        if max_search_new_frames is not None and (type(max_search_new_frames) is not int or max_search_new_frames < 1):
            raise ValueError('Search delivery cap must be a positive integer')
        self.max_search_new_frames = max_search_new_frames
        self.calls = []

    def __call__(self, value, context, images):
        path = self.source['video_path'];start = time.time()
        remaining = remaining_frames(context['budget'], len(images))
        known = [im['timestamp'] for im in images]
        if value['action'] == 'JOINT_SEMANTIC_SCAN':
            hits = self.locator(path, value['queries_by_risk'], top_k_per_risk=2)
            candidates = [dict(h.__dict__) for h in hits]
            timestamps = []
            for hit in hits:
                if not any(abs(hit.timestamp-t) <= .001 for t in known+timestamps):
                    timestamps.append(hit.timestamp)
            delivery_limit = remaining
            if self.max_search_new_frames is not None:
                delivery_limit = (self.max_search_new_frames if remaining is None
                                  else min(remaining, self.max_search_new_frames))
            fresh = extract_frames(path, timestamps[:delivery_limit])
        elif value['action'] == 'TEMPORAL_EXPAND':
            candidates = []
            fresh = TemporalEvidenceExpander()(path, value['timestamps'],
                window=value['window'], num_frames=value['num_frames'])
        else:raise ValueError('Unknown executable tool')
        new = []
        for frame in fresh:
            if not any(abs(frame.timestamp-t) <= .001 for t in known+[m['timestamp'] for m in new]):
                new.append(freeze_runtime_image(frame.image, frame.timestamp,
                    f"frame-{len(images)+len(new):04d}", self.out/'images'))
        original = {'executed_action': value['action'], 'parameters': value,
            'video_path': path, 'video_sha256': self.source['sha256'],
            'returned_timestamps': [im['timestamp'] for im in new],
            'elapsed_seconds': time.time()-start, 'actual_execution': True}
        self.calls.append(original)
        print(json.dumps({'stage': 'REAL_TOOL', **original}), flush=True)
        return {'frame_ids': [im['id'] for im in new], 'candidates': candidates,
                'original_tool_record': original}, new
