#!/usr/bin/env python3
"""Bounded same-backend probability probe; separate collection and replay.

No optimizer, reward, tools, dataset edits or Test inputs. Output must be new.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from hf_request_policy import (VERSION, compile_matcher, prompt_batch,
                               sample_request, score_saved_request, read_bound_item)
from request_policy_replay import sha


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2);f.write('\n')


def fingerprint(batch):
    return {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                'sha256': hashlib.sha256(v.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()}
            for k, v in batch.items()}


def binding(model_path, checkpoint):
    if any((folder/'QUARANTINED.json').exists() for folder in [checkpoint, checkpoint.parent]):
        raise ValueError('Quarantined checkpoint must not initialize a training/rollout candidate')
    return {'backend': 'hf', 'base_path': str(model_path.resolve()),
            'base_config_sha256': sha(model_path/'config.json'),
            'checkpoint_path': str(checkpoint.resolve()),
            'adapter_weight_sha256': sha(checkpoint/'adapter_model.safetensors'),
            'adapter_config_sha256': sha(checkpoint/'adapter_config.json')}


def load_model(args):
    if torch.cuda.mem_get_info()[0] < 35*1024**3:
        raise RuntimeError('Insufficient GPU headroom; do not stop peers')
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    compute_dtype=getattr(args,'compute_dtype','bfloat16')
    if compute_dtype not in {'bfloat16','float32'}:raise ValueError('Unknown forward dtype')
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=getattr(torch,compute_dtype),
            local_files_only=True, trust_remote_code=True, attn_implementation='sdpa', low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.checkpoint, is_trainable=False).to('cuda')
    model._hf_compute_dtype=compute_dtype
    from hf_head_precision import configure_head
    configure_head(model, getattr(args,'head_precision','bfloat16'))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    model.eval()
    if getattr(args, 'train_mode', False):
        model.train()
    return processor, model


def collect(args, started):
    import xgrammar as xgr
    old_plan = json.loads((args.contexts/'PLAN.json').read_text())
    checkpoint_binding = binding(args.model, args.checkpoint)
    if checkpoint_binding['adapter_weight_sha256'] != old_plan['fixed_single_adapter']['weight_sha256']:
        raise ValueError('Probe checkpoint differs from the fixed previous probe')
    contract = {'backend': 'hf', 'temperature': .7, 'top_p': 1., 'top_k': -1,
                'min_p': 0., 'thinking': False, 'penalties': 'none', 'dropout': 0,
                'attention_implementation': 'sdpa', 'dtype': 'bfloat16',
                'grammar_any_whitespace': False, 'collection_decode_mode': args.decode_mode,
                'logprobs': 'masked_temperature_log_softmax_before_sampling'}
    if args.context_index:
        index = json.loads(args.context_index.read_text())
        sources = index['requests']
        if not sources or len(sources) > 20:
            raise ValueError('Numerical probe must contain 1..20 archived requests')
    else:
        sources = []
        for i in range(4):
            old_path = args.contexts/f'{i:02d}/replay_item.json'
            old = json.loads(old_path.read_text())
            sources.append({'archive_path': old['request_path'], 'archive_sha256': old['request_sha256'],
                'source_partition': old['source_partition'], 'source_trajectory_id': old['source_trajectory_id'],
                'prompt_token_ids': old['prompt_token_ids'], 'kind': ['route', 'scan', 'expand', 'answer'][i],
                'provenance_path': str(old_path.resolve()), 'provenance_sha256': sha(old_path)})
    plan = {'version': VERSION, 'thresholds': old_plan['thresholds'], 'sampling_contract': contract,
            'model_binding': checkpoint_binding, 'fixed_original_contexts': str(args.contexts.resolve()),
            'old_plan_sha256': sha(args.contexts/'PLAN.json'), 'requests': len(sources),
            'source_train_videos': len({s['source_trajectory_id'] for s in sources}),
            'context_index_sha256': sha(args.context_index) if args.context_index else None,
            'new_hf_behavior_not_vllm_relabel': True,
            'optimizer_steps': 0, 'test_used': False, 'semantic_audit': False}
    write(args.out/'PLAN.json', plan)
    processor, model = load_model(args)
    vocab = model.config.text_config.vocab_size
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(processor.tokenizer, vocab_size=vocab), max_threads=4)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    rows = []
    for i, source in enumerate(sources):
        if (source['source_partition'] != 'train' or sha(source['archive_path']) != source['archive_sha256']
                or sha(source['provenance_path']) != source['provenance_sha256']):
            raise ValueError('Invalid archived Train provenance')
        request = json.loads(Path(source['archive_path']).read_text())
        folder = args.out/f'{i:02d}';folder.mkdir()
        request.pop('effective_sampling_contract', None)
        request['model'] = 'hf_fixed_student_probe';request['sampling_contract'] = contract
        request['temperature'] = .7
        if args.context_index:
            request['max_tokens'] = {'route': 32, 'scan': 256, 'expand': 256, 'answer': 384}[source['kind']]
            request['seed'] = 2026101950+i
        write(folder/'request.json', request)
        batch = prompt_batch(request, processor, 'cuda')
        if batch['input_ids'][0].tolist() != source['prompt_token_ids']:
            raise ValueError('HF probe changed the archived input tokens')
        tensor_receipt = fingerprint(batch)
        matcher = compile_matcher(compiler, request)
        generator = torch.Generator(device='cuda').manual_seed(request['seed'])
        start = time.time()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            ids, scores, masks, raw_scores, finish = sample_request(model, batch, matcher, vocab,
                .7, request['max_tokens'], eos, generator, args.decode_mode)
        response = {'backend': 'hf', 'model_binding': checkpoint_binding,
                    'sampling_contract': contract, 'prompt_token_ids': batch['input_ids'][0].tolist(),
                    'completion_token_ids': ids, 'sampled_processed_logprobs': scores,
                    'sampled_raw_T1_logprobs': raw_scores, 'finish_reason': finish,
                    'decoded_text_for_inspection_only': processor.tokenizer.decode(ids, skip_special_tokens=True),
                    'prompt_tensor_receipt': tensor_receipt, 'semantic_support': 'NOT_AUDITED'}
        write(folder/'response.json', response)
        np.savez_compressed(folder/'masks.npz', bitmask=masks, token_ids=np.array(ids, dtype=np.int64))
        item = {'version': VERSION, 'source_partition': 'train',
                'source_trajectory_id': source['source_trajectory_id'], 'kind': source['kind'],
                'model_binding': checkpoint_binding, 'sampling_contract': contract,
                'archive_provenance': source}
        for name, file in [('request', 'request.json'), ('response', 'response.json'), ('masks', 'masks.npz')]:
            item[name+'_path'] = str((folder/file).resolve());item[name+'_sha256'] = sha(folder/file)
        write(folder/'item.json', item)
        row = {'item_path': str((folder/'item.json').resolve()), 'kind': item['kind'],
               'tokens': len(ids), 'finish_reason': finish, 'elapsed_seconds': time.time()-start}
        rows.append(row);print(json.dumps(row), flush=True)
        del batch;torch.cuda.empty_cache()
    if binding(args.model, args.checkpoint) != checkpoint_binding:
        raise ValueError('Sampling weights/config changed')
    write(args.out/'summary.json', {'status': 'COLLECTED_PENDING_INDEPENDENT_REPLAY', 'rows': rows,
        'actual_tokens': sum(r['tokens'] for r in rows), 'model_binding': checkpoint_binding,
        'optimizer_steps': 0, 'weight_updates': 0, 'test_used': False,
        'elapsed_seconds': time.time()-started, 'environment': versions()})


def versions():
    return {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'peft', 'xgrammar']}


def replay(args, started):
    plan = json.loads((args.collection/'PLAN.json').read_text())
    expected_binding = plan['model_binding']
    if binding(args.model, args.checkpoint) != expected_binding:
        raise ValueError('Replay uses different actual weights/config')
    collection = json.loads((args.collection/'summary.json').read_text())
    if versions() != collection['environment']:
        raise ValueError('HF sampling/replay software versions differ')
    processor, model = load_model(args)
    results = []
    for row in collection['rows']:
        path = Path(row['item_path']);item, request, response = read_bound_item(path)
        if item['model_binding'] != expected_binding or item['sampling_contract'] != plan['sampling_contract']:
            raise ValueError('Request differs from the frozen collection plan')
        batch = prompt_batch(request, processor, 'cuda')
        if fingerprint(batch) != response['prompt_tensor_receipt']:
            raise ValueError('Actual collection/replay tensors differ')
        del batch
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            scores, allowed_counts, ids = score_saved_request(model, processor, path, decode_mode=args.decode_mode)
        recorded = torch.tensor(response['sampled_processed_logprobs'], device=scores.device)
        error = (scores-recorded).abs();free = allowed_counts > 1
        if not bool(torch.isfinite(recorded).all()):
            raise ValueError('Nonfinite collected logprobs')
        mean = float(error[free].mean()) if bool(free.any()) else 0.
        limits = plan['thresholds']
        passed = mean <= limits['mean_free_token_absolute_logprob_difference'] and float(error.max()) <= limits['maximum_absolute_logprob_difference']
        result = {'kind': item['kind'], 'tokens': len(ids), 'free_tokens': int(free.sum()),
                  'mean_free_logprob_difference': mean, 'maximum_logprob_difference': float(error.max()),
                  'passed': passed, 'request_sha256': item['request_sha256'],
                  'actual_sampling_and_replay_tensor_hashes_equal': True,
                  'per_token': [{'index': i, 'token_id': int(ids[i]), 'sampled_logprob': float(recorded[i]),
                                 'replayed_logprob': float(scores[i]), 'error': float(error[i]),
                                 'allowed_count': int(allowed_counts[i])} for i in range(len(ids))]}
        write(args.out/f'{len(results):02d}_tokens.json', result)
        results.append({k: v for k, v in result.items() if k != 'per_token'})
        print(json.dumps(results[-1]), flush=True)
        del scores, allowed_counts, ids, recorded, error, free;torch.cuda.empty_cache()
    if binding(args.model, args.checkpoint) != expected_binding:
        raise ValueError('Replay weights/config changed')
    write(args.out/'summary.json', {'status': 'PASS' if all(r['passed'] for r in results) else 'FAIL_NUMERICAL_ALIGNMENT',
        'rows': results, 'thresholds': plan['thresholds'], 'model_binding': expected_binding,
        'collection_plan_sha256': sha(args.collection/'PLAN.json'), 'decode_mode': args.decode_mode,
        'train_mode': args.train_mode, 'actual_tokens': sum(r['tokens'] for r in results),
        'separate_collection_process': True, 'sampling_scores_reused_as_replay_scores': False,
        'environment': versions(), 'optimizer_steps': 0, 'weight_updates': 0, 'test_used': False,
        'semantic_audit': False, 'real_model_backward_verified': False,
        'elapsed_seconds': time.time()-started, 'maximum_gpu_allocated_bytes': torch.cuda.max_memory_allocated()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=['collect', 'replay'], required=True)
    ap.add_argument('--contexts', type=Path);ap.add_argument('--collection', type=Path)
    ap.add_argument('--context-index', type=Path, help='Optional bounded Train request index; at most 20')
    ap.add_argument('--model', type=Path, required=True);ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--decode-mode', choices=['cached', 'full_causal'], required=True)
    ap.add_argument('--train-mode', action='store_true')
    args = ap.parse_args();args.out.mkdir(parents=True, exist_ok=False);started = time.time()
    try:
        if args.phase == 'collect':
            if args.train_mode:raise ValueError('Collect in eval mode with explicitly zero dropout')
            collect(args, started)
        else:
            replay(args, started)
    except Exception as exc:
        write(args.out/'FAILURE.json', {'error': repr(exc), 'automatic_retry': False, 'weight_updates': 0})
        raise


if __name__ == '__main__':
    main()
