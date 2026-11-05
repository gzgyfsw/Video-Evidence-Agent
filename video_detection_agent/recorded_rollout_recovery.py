"""Recover complete native evaluation cases, retaining recorded Student failures.

No model calls. Missing cases remain missing. All image, seed and model bindings
are checked; elapsed request timing unavailable in native items stays unknown.
"""
import argparse
import copy
import json
from pathlib import Path

from hf_request_rng import request_seed
from hf_request_policy import read_bound_item
from request_policy_replay import sha
from student_action_outcomes import classify_recorded_failure
from unified_student_contract import POLICY_SHA,public_context


def load(path):return json.loads(path.read_text())


def recover_case(case, entry, model_binding, reader, partition):
    meta=entry['source'];loop=case/'loop';paths=sorted(loop.glob('turn-*/item.json'),key=lambda p:int(p.parent.name.split('-')[1]))
    if not paths:raise ValueError('Case has no complete native request; explicit technical recovery needed')
    if [p.parent.name for p in paths]!=[f'turn-{i+1}' for i in range(len(paths))]:
        raise ValueError('Native request sequence has gaps')
    calls=[];last=None
    for index,path in enumerate(paths):
        item,request,response=reader(path)
        if (item['source_partition']!=partition or meta['frozen_split']!=partition
                or item['model_binding']!=model_binding or item['source_trajectory_id']!=meta['merged_id']
                or request['seed']!=request_seed(entry['seed'],index)
                or request['policy_sha256']!=POLICY_SHA or request['context']!=public_context(request['context'])):
            raise ValueError('Recorded source/partition/model/seed differs from frozen sample')
        for image in request['images']:
            if sha(image['path'])!=image['sha256']:raise ValueError('Recorded image bytes changed')
        calls.append({'item_path':str(path.resolve()),'tokens':len(response['completion_token_ids']),
                      'elapsed_seconds':None,'recovered_from_native_item':True})
        last=response
    tools=[]
    for path in sorted(loop.glob('turn-*-tool-return.json'),key=lambda p:int(p.name.split('-')[1])):
        returned=load(path);record=returned['record'];original=record.get('original_tool_record')
        if (not original or original.get('actual_execution') is not True or record.get('error')
                or original['video_sha256']!=meta['sha256']
                or record['frame_ids']!=[im['id'] for im in returned['images']]
                or original['returned_timestamps']!=[im['timestamp'] for im in returned['images']]):
            raise ValueError('Tool ledger cannot certify the actual successful calls')
        tools.append(original)
    if (case/'receipt.json').exists():
        result=load(case/'receipt.json')
        if (result['source_id']!=meta['merged_id'] or result['sample']!=entry['sample']
                or result['source_partition']!=partition or result['actual_tools']!=tools
                or [r['item_path'] for r in result['requests']]!=[r['item_path'] for r in calls]):
            raise ValueError('Existing case receipt differs from native records')
        calls=result['requests']
    elif (loop/'result.json').exists():
        result=load(loop/'result.json')
    else:
        result=classify_recorded_failure(loop,calls,reader)
        if result is None:raise ValueError('Unresolved technical failure cannot be relabelled as a Student result')
    if result.get('tool_calls')!=len(tools):raise ValueError('Executed tool count differs')
    if result['status']=='INVALID_STUDENT_ACTION':
        proof=classify_recorded_failure(loop,calls,reader)
        if proof is None or proof['failure_reason']!=result['failure_reason']:
            raise ValueError('Student failure label cannot be re-proven from native records')
    if result['status']=='MODEL_ANSWER_UNREVIEWED' and result['answer']!=json.loads(last['decoded_text_for_inspection_only']):
        raise ValueError('Saved answer differs from actual completion')
    recovered=copy.deepcopy(result)
    recovered.update(source_id=meta['merged_id'],sample=entry['sample'],source_partition=partition,
                     actual_tools=tools,requests=calls)
    if partition=='dev':recovered.update(seed=entry['seed'],model_binding=model_binding)
    recovered['recovery_provenance']={'recorded_case':str(case.resolve()),
        'original_case_receipt_sha256':sha(case/'receipt.json') if (case/'receipt.json').exists() else None,
        'native_request_item_sha256':{str(p.resolve()):sha(p) for p in paths},
        'no_resampling':True,'request_timing_not_invented':True}
    return recovered


def complete_collection(partial, missing_collection):
    """Merge only never-sampled IDs; keep the earlier failed outcome unchanged."""
    from check_hf_policy_alignment import versions
    plan=load(partial/'PLAN.json');other_plan=load(missing_collection/'PLAN.json')
    manifest=load(partial/'MANIFEST.json');missing=load(partial/'missing_manifest.json')
    if load(missing_collection/'MANIFEST.json')!=missing:
        raise ValueError('Missing-only sampling manifest changed; failed cases must not be resampled')
    for field in ['model_binding','policy_sha256','sampling_contract','train_manifest_sha256',
                  'budget','max_decisions','initial_frames','max_search_new_frames']:
        if plan[field]!=other_plan[field]:raise ValueError('Recovery sampling protocol changed: '+field)
    results=load(partial/'recovered_results.json');by_id={(r['source_id'],r['sample']):r for r in results}
    for entry in manifest:
        key=(entry['source']['merged_id'],entry['sample'])
        if key in by_id:
            original=Path(by_id[key]['recovery_provenance']['recorded_case'])
            actual=recover_case(original,entry,plan['model_binding'],read_bound_item,'train')
            if actual!=by_id[key]:raise ValueError('Recovered native case changed before assembly')
    for entry in missing:
        ident=entry['source']['merged_id'];case=missing_collection/'rollouts'/f"{ident}-sample-{entry['sample']}"
        result=recover_case(case,entry,plan['model_binding'],read_bound_item,'train')
        key=(ident,entry['sample'])
        if key in by_id:raise ValueError('Recovery would replace an already recorded outcome')
        by_id[key]=result;destination=partial/'rollouts'/case.name;destination.mkdir(parents=True)
        (destination/'recorded_case').symlink_to(case.resolve(),target_is_directory=True)
        (destination/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    expected=[(row['source']['merged_id'],row['sample']) for row in manifest]
    if set(by_id)!=set(expected):raise ValueError('Recovery did not retain the full original manifest')
    ordered=[by_id[key] for key in expected];calls=[c for r in ordered for c in r['requests']]
    other=load(missing_collection/'summary.json')
    if other['environment']!=versions():raise ValueError('Sampling/recovery environment differs')
    summary={'status':'ROLLOUT_COMPLETE_PENDING_REPLAY_AND_REWARD_GATE','results':ordered,'rows':calls,
        'source_train_videos':len({r['source_id'] for r in ordered}),'rollouts':len(ordered),
        'model_requests':len(calls),'actual_tokens':sum(c['tokens'] for c in calls),
        'actual_tool_calls':sum(len(r['actual_tools']) for r in ordered),'environment':versions(),
        'invalid_student_actions':sum(r['status']=='INVALID_STUDENT_ACTION' for r in ordered),
        'failed_outcomes_not_resampled':True,'semantic_support':'NOT_AUDITED',
        'optimizer_steps':0,'weight_updates':0,'test_used':False}
    with (partial/'summary.json').open('x') as stream:json.dump(summary,stream,ensure_ascii=False,indent=2)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed-collection',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();source=args.failed_collection.resolve()
    if not (source/'FAILURE.json').exists():raise ValueError('Explicit failed collection required')
    plan=load(source/'PLAN.json');manifest=load(source/'MANIFEST.json')
    if not 1<=len(manifest)<=4:raise ValueError('Only the bounded internal Train evaluation pilot is supported here')
    args.out.mkdir(parents=True,exist_ok=False);results=[];missing=[]
    for entry in manifest:
        meta=entry['source'];case=source/'rollouts'/f"{meta['merged_id']}-sample-{entry['sample']}"
        if not case.exists():missing.append(entry);continue
        result=recover_case(case,entry,plan['model_binding'],read_bound_item,'train');results.append(result)
        destination=args.out/'rollouts'/case.name;destination.mkdir(parents=True)
        (destination/'recorded_case').symlink_to(case,target_is_directory=True)
        (destination/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for name in ['PLAN.json','MANIFEST.json']:(args.out/name).write_bytes((source/name).read_bytes())
    for name,value in [('recovered_results.json',results),('missing_manifest.json',missing),
        ('RECOVERY_STATE.json',{'status':'WAITING_FOR_NEVER_SAMPLED_CASES','expected':len(manifest),
            'recovered':len(results),'missing':len(missing),'student_failures':sum(r['status']=='INVALID_STUDENT_ACTION' for r in results),
            'source_failure_sha256':sha(source/'FAILURE.json'),'model_calls':0,'optimizer_updates':0})]:
        (args.out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'recovered':len(results),'missing':len(missing),'new_model_calls':0}))


if __name__=='__main__':main()
