#!/usr/bin/env python3
"""Real-model gradient plumbing check. No optimizer or parameter updates."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from check_hf_policy_alignment import binding, fingerprint, write, versions
from hf_request_policy import read_bound_item, prompt_batch, score_saved_request
from request_policy_replay import sha


def parameter_digest(model):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            h.update(name.encode());h.update(parameter.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collection', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True);ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--plan-extension', type=Path, help='Explicit hash-bound metadata completion, never inferred defaults')
    ap.add_argument('--head-precision',choices=['bfloat16','float32'],default='bfloat16')
    ap.add_argument('--compute-dtype',choices=['bfloat16','float32'],default='bfloat16')
    args = ap.parse_args();args.out.mkdir(parents=True, exist_ok=False);started = time.time()
    def stage(name,**details):
        receipt={'stage':name,'elapsed_seconds':time.time()-started,**details}
        progress=args.out/'progress.tmp'
        progress.write_text(json.dumps(receipt,indent=2)+'\n')
        progress.replace(args.out/'progress.json')
        print(json.dumps(receipt),flush=True)
    try:
        stage('VALIDATING_INPUTS')
        plan = json.loads((args.collection/'PLAN.json').read_text())
        if args.plan_extension:
            extension = json.loads(args.plan_extension.read_text())
            if extension['source_plan_sha256'] != sha(args.collection/'PLAN.json'):
                raise ValueError('Plan extension binds a different original plan')
            if 'sampling_contract' in plan and plan['sampling_contract'] != extension['sampling_contract']:
                raise ValueError('Plan extension cannot override an existing sampling contract')
            plan['sampling_contract'] = extension['sampling_contract']
        collection = json.loads((args.collection/'summary.json').read_text())
        if versions() != collection['environment']:
            raise ValueError('HF sampling/backward software versions differ')
        if binding(args.model, args.checkpoint) != plan['model_binding']:
            raise ValueError('Different checkpoint/config')
        if torch.cuda.mem_get_info()[0] < 40*1024**3:
            raise RuntimeError('Insufficient shared GPU headroom; do not interrupt peers')
        torch.set_num_threads(4)
        stage('LOADING_PROCESSOR_AND_BASE')
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
        base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=getattr(torch,args.compute_dtype),
            local_files_only=True, trust_remote_code=True, attn_implementation='sdpa', low_cpu_mem_usage=True)
        stage('LOADING_ADAPTER')
        model = PeftModel.from_pretrained(base, args.checkpoint, is_trainable=True)
        stage('MOVING_MODEL_TO_CUDA')
        model=model.to('cuda')
        stage('CONFIGURING_TRAIN_MODE')
        model._hf_compute_dtype=args.compute_dtype
        from hf_head_precision import configure_head
        configure_head(model,args.head_precision)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_('lora_' in name)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):module.p = 0.
        model.train()
        before = parameter_digest(model)
        results = []
        for source in collection['rows']:
            stage('REQUEST_START',request_index=len(results),item_path=source['item_path'])
            path = Path(source['item_path']);item, request, response = read_bound_item(path)
            if item['model_binding'] != plan['model_binding'] or item['sampling_contract'] != plan['sampling_contract']:
                raise ValueError('Unbound per-request checkpoint/sampling contract')
            batch = prompt_batch(request, processor, 'cuda')
            if fingerprint(batch) != response['prompt_tensor_receipt']:
                raise ValueError('Actual image/token tensors differ')
            del batch
            model.zero_grad(set_to_none=True);start = time.time()
            with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                scores, counts, ids = score_saved_request(model, processor, path, decode_mode='full_causal')
                behavior = torch.tensor(response['sampled_processed_logprobs'], device=scores.device)
                errors = (scores.detach()-behavior).abs();free = counts > 1
                mean = float(errors[free].mean())
                aligned = mean <= plan['thresholds']['mean_free_token_absolute_logprob_difference'] and float(errors.max()) <= plan['thresholds']['maximum_absolute_logprob_difference']
                if not bool(free.any()) or not scores.requires_grad:
                    raise ValueError('No differentiable free-token policy signal')
                # A plumbing diagnostic only, not GRPO or a proposed reward.
                diagnostic = -scores[free].mean()
            stage('REQUEST_BACKWARD',request_index=len(results))
            diagnostic.backward()
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
            norm = sum(float(g.detach().float().square().sum()) for g in gradients)**.5
            base_gradients = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
            if not finite or not norm > 0 or base_gradients:
                raise ValueError('Invalid/missing policy gradient or unexpected base gradients')
            row = {'kind': item['kind'], 'tokens': len(ids), 'free_tokens': int(free.sum()),
                'mean_free_logprob_difference': mean, 'maximum_logprob_difference': float(errors.max()),
                'train_mode_differentiable_scores_aligned': aligned,
                'finite_nonzero_lora_gradients': True, 'gradient_tensors': len(gradients),
                'gradient_norm': norm, 'base_parameter_gradients': len(base_gradients),
                'diagnostic_loss': float(diagnostic.detach()), 'parameter_updates': 0,
                'elapsed_seconds': time.time()-start,
                'per_token': [{'index': i, 'id': int(ids[i]), 'behavior': float(behavior[i]),
                    'differentiable_score': float(scores[i].detach()), 'error': float(errors[i])} for i in range(len(ids))]}
            write(args.out/f'{len(results):02d}_gradient.json', row)
            results.append({k: v for k, v in row.items() if k != 'per_token'})
            print(json.dumps(results[-1]), flush=True)
            model.zero_grad(set_to_none=True)
            del scores, counts, ids, errors, behavior, free, diagnostic, gradients
            torch.cuda.empty_cache()
        after = parameter_digest(model)
        if before != after or binding(args.model, args.checkpoint) != plan['model_binding']:
            raise ValueError('Parameters changed without authorization')
        write(args.out/'summary.json', {'status': 'PASS' if all(r['train_mode_differentiable_scores_aligned'] for r in results) else 'FAIL_NUMERICAL_ALIGNMENT',
            'rows': results, 'same_trainable_parameter_digest_before_after': before == after,
            'parameter_sha256_before': before, 'parameter_sha256_after': after,
            'model_binding': plan['model_binding'], 'thresholds': plan['thresholds'],
            'optimizer_created': False, 'optimizer_steps': 0, 'weight_updates': 0,
            'reward_used': False, 'grpo_started': False, 'sft_started': False,
            'test_used': False, 'semantic_audit': False, 'environment': versions(),
            'dropout': 0, 'gradient_checkpointing': 'nonreentrant', 'model_mode': 'train',
            'elapsed_seconds': time.time()-started, 'maximum_gpu_allocated_bytes': torch.cuda.max_memory_allocated()})
    except Exception as exc:
        write(args.out/'FAILURE.json', {'error': repr(exc), 'automatic_retry': False,
            'optimizer_created': False, 'weight_updates': 0})
        raise


if __name__ == '__main__':main()
