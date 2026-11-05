"""Collect frozen groups with one loaded SFT Student, real tools and no optimizer."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from prepare_grpo_group_selection import ROOT, digest, save


def state(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def gpu():
    import subprocess
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    used, free, util = map(int, raw.strip().split(','))
    return {'used_mib': used, 'free_mib': free, 'utilization_percent': util}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort', type=Path, required=True)
    a = ap.parse_args()
    cohort = a.cohort.resolve()
    lock = (cohort/'.collection.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((cohort/'PLAN.json').read_text())
    selection = json.loads((cohort/'SELECTION.json').read_text())
    if digest(plan['source_manifest']) != plan['train_manifest_sha256']:
        raise ValueError('Train source manifest changed')
    pending = [s for s in selection if not (Path(s['rollout_dir'])/'summary.json').exists()]
    for s in pending:
        if Path(s['rollout_dir']).exists():
            raise ValueError('Partial output preserved, requires explicit recovery: '+s['rollout_dir'])
    # Preserve the protocol of completed cohorts; never append uncapped
    # trajectories to a historical capped experiment under the same identity.
    for s in selection:
        recorded_plan = Path(s['rollout_dir'])/'PLAN.json'
        if pending and recorded_plan.exists():
            previous = json.loads(recorded_plan.read_text())
            if (previous['budget'] != {'max_tool_calls': None, 'max_frames': None}
                    or previous['max_decisions'] is not None
                    or previous.get('max_search_new_frames') is not None):
                raise ValueError('Historical capped cohort preserved; use a new cohort for uncapped collection')
    started = time.time()
    status = {'stage': 'PREFLIGHT', 'pid': os.getpid(), 'total_groups': len(selection),
              'optimizer_updates': 0, 'test_used': False, 'started_unix': started,
              'completed_groups': [s['group'] for s in selection if s not in pending]}
    state(cohort/'COLLECTION_PROGRESS.json', status)
    if not pending:
        status.update(stage='COMPLETE', updated_unix=time.time())
        state(cohort/'COLLECTION_PROGRESS.json', status)
        return
    initial_gpu = gpu()
    if initial_gpu['free_mib'] < 43000 or initial_gpu['used_mib'] > 16000:
        status.update(stage='WAITING_FOR_GPU', gpu=initial_gpu)
        state(cohort/'COLLECTION_PROGRESS.json', status)
        raise RuntimeError('Insufficient safe headroom; peers left untouched')
    import torch
    import xgrammar as xgr
    from agent.tools import JointRiskSemanticLocator
    from agent.video import uniform_frames, video_duration
    from check_hf_policy_alignment import binding, load_model, versions
    from hf_unified_rollout import HFUnifiedStudent, RealVideoTools, sampling_contract
    from hf_request_policy import read_bound_item
    from hf_request_rng import validate_seeds
    from student_action_outcomes import run_accounted
    from unified_student_contract import freeze_runtime_image, PUBLIC_CONTEXT_VERSION, INSTRUCTION, UNLIMITED_BUDGET
    torch.cuda.set_per_process_memory_fraction(0.65)
    args = SimpleNamespace(model=Path(plan['model_path']), checkpoint=Path(plan['checkpoint_path']),
                           compute_dtype='float32', head_precision='float32')
    if binding(args.model, args.checkpoint) != plan['model_binding']:
        raise ValueError('SFT binding mismatch')
    processor, model = load_model(args)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(processor.tokenizer,
        vocab_size=model.config.text_config.vocab_size), max_threads=4)
    locator = JointRiskSemanticLocator(model_name=plan['locator_model_path'], device='cuda',
        embedding_cache_dir=str(cohort/'locator_cache'))
    policy = Path(plan['policy_path']).read_text()
    if digest(plan['policy_path']) != plan['policy_sha256']:
        raise ValueError('Policy changed')
    for s in pending:
        gi, folder = s['group'], Path(s['rollout_dir'])
        folder.mkdir(parents=True, exist_ok=False)
        begin = time.time()
        group_plan = {'model_binding': plan['model_binding'], 'policy_sha256': plan['policy_sha256'],
            'sampling_contract': sampling_contract('float32', 'float32'),
            'request_seed_schedule': 'disjoint-request-v2',
            'public_context_version': PUBLIC_CONTEXT_VERSION, 'instruction_sha256': digest(INSTRUCTION),
            'train_manifest_sha256': plan['train_manifest_sha256'],
            'budget': dict(UNLIMITED_BUDGET), 'max_decisions': None, 'initial_frames': 4,
            'max_search_new_frames': None, 'all_stages_same_adapter': True,
            'source_labels_in_model_input': False, 'reward_not_enabled': True,
            'optimizer_steps': 0, 'test_used': False, 'semantic_audit': False,
            'batch_runner_sha256': digest(__file__)}
        save(folder/'PLAN.json', group_plan)
        save(folder/'MANIFEST.json', s['samples'])
        status.update(stage='REAL_STUDENT_ROLLOUT', group=gi, updated_unix=time.time(), gpu=gpu())
        state(cohort/'COLLECTION_PROGRESS.json', status)
        results, requests = [], []
        try:
            for row in s['samples']:
                meta = row['source']
                if digest(meta['video_path']) != meta['sha256']:
                    raise ValueError('Video changed')
                case = folder/'rollouts'/(meta['merged_id']+f"-sample-{row['sample']}")
                case.mkdir(parents=True)
                initial = [freeze_runtime_image(frame.image, frame.timestamp, f'frame-{i:04d}', case/'images')
                           for i, frame in enumerate(uniform_frames(meta['video_path'], 4))]
                student = HFUnifiedStudent(model, processor, compiler, plan['model_binding'], policy,
                    meta, row['seed'], request_seed_schedule='disjoint-request-v2')
                executor = RealVideoTools(meta, locator, case)
                result = run_accounted(student, video_duration(meta['video_path']), group_plan['budget'],
                    initial, executor, case/'loop', max_decisions=group_plan['max_decisions'], reader=read_bound_item)
                result.update(source_id=meta['merged_id'], sample=row['sample'], source_partition='train',
                    actual_tools=executor.calls, requests=student.calls)
                save(case/'receipt.json', result)
                results.append(result)
                requests.extend(student.calls)
                status.update(sample=row['sample'], updated_unix=time.time(), gpu=gpu())
                state(cohort/'COLLECTION_PROGRESS.json', status)
                print(json.dumps({'group': gi, 'sample': row['sample'], 'status': result['status'],
                    'tools': len(executor.calls)}, ensure_ascii=False), flush=True)
            validate_seeds(results, s['samples'], read_bound_item)
            save(folder/'summary.json', {'status': 'ROLLOUT_COMPLETE_PENDING_REPLAY_AND_REWARD_GATE',
                'results': results, 'rows': requests, 'source_train_videos': 1, 'rollouts': 4,
                'model_requests': len(requests), 'actual_tokens': sum(r['tokens'] for r in requests),
                'actual_tool_calls': sum(len(r['actual_tools']) for r in results),
                'optimizer_steps': 0, 'weight_updates': 0, 'grpo_started': False,
                'test_used': False, 'environment': versions(), 'elapsed_seconds': time.time()-begin,
                'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
                'cuda_peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2})
        except Exception as exc:
            save(folder/'FAILURE.json', {'error': repr(exc), 'optimizer_steps': 0, 'test_used': False})
            status.update(stage='STOPPED_FAILURE', error=repr(exc), updated_unix=time.time())
            state(cohort/'COLLECTION_PROGRESS.json', status)
            raise
        status['completed_groups'].append(gi)
        status.update(updated_unix=time.time(), elapsed_seconds=time.time()-started)
        state(cohort/'COLLECTION_PROGRESS.json', status)
    if binding(args.model, args.checkpoint) != plan['model_binding']:
        raise ValueError('SFT weights changed during collection')
    status.update(stage='COMPLETE', updated_unix=time.time(), elapsed_seconds=time.time()-started,
        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    state(cohort/'COLLECTION_PROGRESS.json', status)


if __name__ == '__main__':
    main()
