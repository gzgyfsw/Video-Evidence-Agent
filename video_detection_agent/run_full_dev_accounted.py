"""Resumable chunks of the complete frozen Dev, one unified Student per role.

No optimizer or model reviewer. Missing/unfinished cases remain in denominator;
only exact immutable receipts can be reused. Full metrics require all1679.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time

from check_hf_policy_alignment import binding, write, versions
from request_policy_replay import sha


def load(p):return json.loads(Path(p).read_text())


def progress_write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def training_inactive(trial):
    if trial is None:return True
    state=load(trial/'PIPELINE_STATE.json')
    return state['stage'] in {'SHARED_WEAK_SEMANTIC_REVIEW','EXPLICIT_FAILED_GROUP_RECOVERY',
                             'COMPLETED','STOPPED_ON_FAILURE','COMPLETED_NO_UPDATE_INSUFFICIENT_EVIDENCE_SIGNAL'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['frozen-dev','model','checkpoint','locator-model','policy','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--role',choices=['sft_v4','candidate'],required=True)
    p.add_argument('--max-new-rollouts',type=int,required=True)
    p.add_argument('--yield-to-trial',type=Path)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    if not 1<=a.max_new_rollouts<=1679:raise ValueError('Explicit finite chunk limit required')
    lock=(a.out/'.writer.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=load(a.frozen_dev/'PLAN.json');manifest=load(a.frozen_dev/'MANIFEST1679.json')
    if len(manifest)!=1679 or sha(a.frozen_dev/'MANIFEST1679.json')!=protocol['manifest_sha256']:
        raise ValueError('Do not replace full Dev with a subset')
    if any(r['source']['frozen_split']!='dev' for r in manifest):raise ValueError('Only Dev, never Train/Test')
    if sha(a.policy)!=protocol['policy_sha256']:raise ValueError('Policy changed')
    bind=binding(a.model,a.checkpoint)
    runtime_path=a.frozen_dev/'RUNTIME_BINDINGS_v3.json'
    runtime=load(runtime_path)
    def verify_runtime():
        if any(sha(path)!=expected for path,expected in runtime.items()):
            raise ValueError('Frozen full-Dev inference implementation changed; explicit paired recovery required')
    verify_runtime()
    role_runtime=a.out/'RUNTIME_BINDING.json'
    if role_runtime.exists():
        if load(role_runtime)['source_binding_sha256']!=sha(runtime_path):raise ValueError('Role runtime binding changed')
    else:
        write(role_runtime,{'source_binding_sha256':sha(runtime_path),
            'preexisting_smoke_receipts':len(list(a.out.glob('rollouts/*/receipt.json'))),
            'predecessor_manifest_sha256':sha(a.out/'REUSED_NATIVE_RECEIPTS.json') if (a.out/'REUSED_NATIVE_RECEIPTS.json').exists() else None,
            'actual_prompt_tensor_pairing_still_required':True})
    contract={'role':a.role,'model_binding':bind,'manifest_sha256':protocol['manifest_sha256'],
        'policy_sha256':protocol['policy_sha256'],'budget':protocol['budget'],'sampling':protocol['sampling'],
        'version':'full-original-dev-fp32-accounted-v2','test_used':False,'optimizer_steps':0,'environment':versions()}
    if (a.out/'PLAN.json').exists() and load(a.out/'PLAN.json')!=contract:raise ValueError('Frozen evaluation contract changed')
    if not (a.out/'PLAN.json').exists():write(a.out/'PLAN.json',contract)
    results=[];pending=[]
    for row in manifest:
        path=a.out/'rollouts'/row['source']['merged_id']/'receipt.json'
        if not path.exists():pending.append(row);continue
        result=load(path)
        if (result['source_id']!=row['source']['merged_id'] or result['seed']!=row['seed']
                or result['model_binding']!=bind or result['source_partition']!='dev'):
            raise ValueError('Receipt belongs to another model/source/seed')
        results.append(result)
    def status(stage,**kw):
        progress_write(a.out/'progress.json',{'stage':stage,'completed':len(results),'total':1679,'role':a.role,
            'optimizer_steps':0,'test_used':False,'updated_unix':time.time(),**kw})
    status('PREPARED')
    if not pending:
        if (a.out/'summary.json').exists():return
        write(a.out/'summary.json',{'status':'ALL_DEV_ROLLOUTS_COMPLETE_PENDING_SEMANTIC_REVIEW',
              'results':results,'count':1679,'optimizer_steps':0,'test_used':False});return
    if not training_inactive(a.yield_to_trial):status('YIELDED_TO_TRAINING');return
    import torch
    import xgrammar as xgr
    from agent.tools import JointRiskSemanticLocator
    from agent.video import uniform_frames, video_duration
    from check_hf_policy_alignment import load_model
    from hf_frozen_dev_policy import FrozenDevStudent, read_dev_item
    from hf_unified_rollout import RealVideoTools
    from unified_student_contract import freeze_runtime_image
    from student_action_outcomes import run_accounted
    if torch.cuda.mem_get_info()[0]<60*1024**3:status('YIELDED_TO_GPU_OWNER');return
    # Preserve exact full-FP32 inference settings used by the GRPO behavior.
    a.head_precision='float32';a.compute_dtype='float32'
    processor,network=load_model(a)
    compiler=xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(processor.tokenizer,
                vocab_size=network.config.text_config.vocab_size),max_threads=4)
    locator=JointRiskSemanticLocator(model_name=str(a.locator_model.resolve()),device='cuda',
                                    embedding_cache_dir=str(a.out/'locator_cache'))
    try:
        for row in pending[:a.max_new_rollouts]:
            verify_runtime()
            if not training_inactive(a.yield_to_trial):status('YIELDED_TO_TRAINING');return
            meta=row['source'];case=a.out/'rollouts'/meta['merged_id']
            if case.exists():raise ValueError('Incomplete prior case needs explicit recovery, never silently regenerate')
            case.mkdir(parents=True)
            if sha(meta['video_path'])!=meta['sha256']:raise ValueError('Dev source media changed')
            status('REAL_DEV_ROLLOUT',source_id=meta['merged_id'])
            duration=video_duration(meta['video_path'])
            initial=[freeze_runtime_image(f.image,f.timestamp,f'frame-{i:04d}',case/'images')
                     for i,f in enumerate(uniform_frames(meta['video_path'],4))]
            student=FrozenDevStudent(network,processor,compiler,bind,a.policy.read_text(),meta,row['seed'],
                                     request_seed_schedule='disjoint-request-v2')
            executor=RealVideoTools(meta,locator,case,max_search_new_frames=4)
            result=run_accounted(student,duration,{'max_tool_calls':3,'max_frames':12},initial,executor,case/'loop',max_decisions=4,reader=read_dev_item)
            for request in student.calls:read_dev_item(request['item_path'])
            result.update(source_id=meta['merged_id'],seed=row['seed'],sample=0,source_partition='dev',
                          model_binding=bind,actual_tools=executor.calls,requests=student.calls,
                          semantic_support='NOT_AUDITED',reference_labels_hidden=True,
                          execution_runtime_binding_sha256=sha(runtime_path),execution_accounting_version='native-request-proven-v1')
            write(case/'receipt.json',result);results.append(result);status('DEV_RECEIPT_COMMITTED')
        if binding(a.model,a.checkpoint)!=bind:raise ValueError('Evaluation checkpoint changed')
        status('CHUNK_COMPLETE_FULL_DEV_INCOMPLETE' if len(results)<1679 else 'ALL_DEV_ROLLOUTS_COMPLETE')
        if len(results)==1679:
            write(a.out/'summary.json',{'status':'ALL_DEV_ROLLOUTS_COMPLETE_PENDING_SEMANTIC_REVIEW',
                  'results':results,'count':1679,'optimizer_steps':0,'test_used':False})
    except Exception as exc:
        status('STOPPED_ON_FAILURE',error=repr(exc));write(a.out/'FAILURE.json',{'error':repr(exc),
            'completed':len(results),'automatic_retry':False,'optimizer_steps':0});raise


if __name__=='__main__':main()
