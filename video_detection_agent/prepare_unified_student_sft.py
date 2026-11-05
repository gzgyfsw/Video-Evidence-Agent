#!/usr/bin/env python3
"""Read completed Train audits, project approved NEXT outputs; no model calls."""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from unified_student_contract import VERSION, POLICY_SHA, INSTRUCTION, dumps, sha, prompt, content, validate_output


def read(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def write(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


def resolve(value, pointer):
    for key in pointer.lstrip('/').split('/'):
        key=key.replace('~1','/').replace('~0','~')
        value=value[int(key)] if isinstance(value,list) else value[key]
    return value


def covered(markers, path, value):
    selected=[m for m in markers if m['supervise'] and m['target'].get('path')==path]
    if any('span' not in m['target'] for m in selected): return selected
    if not isinstance(value,str): return []
    spans=sorted(m['target']['span'] for m in selected)
    cursor=0
    for start,end in spans:
        if start>cursor or end<start: return []
        cursor=max(cursor,end)
    return selected if cursor==len(value) else []


def prefix_events(steps):
    events=[]
    for step in steps:
        event={'step_id':step['id'],'type':step['type']}
        if step['type']=='observe': event['frame_ids']=step['frame_ids']
        elif step['type']=='decision':
            event.update(action=step['action'],parameters=step.get('parameters',{}),claims=step.get('claims',[]))
        elif step['type']=='tool_result':
            if set(step['result'])-{'frame_ids','candidates','original_tool_record','error'}:
                raise ValueError('Unknown tool result fields')
            event.update(call_step_id=step['call_step_id'],tool=step['tool'],result=step['result'],frame_ids=step['result'].get('frame_ids',[]))
        else: raise ValueError('Unknown prefix event')
        events.append(event)
    return events


def approved_checks(folder, candidate, markers, source, index, visible):
    """Bind approval to saved inputs, prefix, policy and raw accepted review."""
    checks={c['task_id']:c for c in candidate['checks']}; receipts=[]; inputs=[]
    for tid in sorted({m['task_id'] for m in markers}):
        check=checks[tid]
        if check['status']!='COMPLETED' or check['judgment']!='SUPPORTED': raise ValueError('Approval is not effective support')
        task=folder/'tasks'/tid; inp=json.loads((task/'input.json').read_text())
        review=json.loads((task/'review.json').read_text()); prov=check['provenance']
        digest=hashlib.sha256(dumps(inp).encode()).hexdigest()
        if digest!=prov['input_sha256'] or review['input_sha256']!=digest: raise ValueError('Audit input hash mismatch')
        if prov['policy_sha256']!=POLICY_SHA or (inp.get('policy') or {}).get('sha256')!=POLICY_SHA: raise ValueError('Missing/different policy')
        if review.get('response',{}).get('judgment')!='SUPPORTED' or not review['response'].get('prefix_isolated'): raise ValueError('Review not isolated/supported')
        if inp.get('audit_context') or inp.get('unavailable_seen_frame_ids') or inp.get('future_information_included') is not False: raise ValueError('Incomplete/noncausal audit input')
        if [e['step_id'] for e in inp['prefix_events']] != [e['id'] for e in source['steps'][:index]]: raise ValueError('Prefix events differ from original order')
        if inp['prefix_events']!=prefix_events(source['steps'][:index]): raise ValueError('Prefix content differs from original events')
        if [im['id'] for im in inp['images']] != visible: raise ValueError('Audit/recorded image order mismatch')
        if inp['images']!=check['image_manifest']: raise ValueError('Audit manifest changed')
        if check['kind'] in {'query','selection_expand'} and inp['subject']['parameters']!=source['steps'][index]['parameters']: raise ValueError('Tool parameters differ from audit')
        if check['kind']=='claim':
            subject=check['subject']; target=subject['target']; original=resolve(source,target['path'])
            if 'span' in target: original=original[slice(*target['span'])]
            if original!=subject['text']: raise ValueError('Claim text changed')
        receipts.append({'task_id':tid,'kind':check['kind'],'input_path':str(task/'input.json'),
            'input_sha256':digest,'review_path':str(task/'review.json'),'review_file_sha256':sha(task/'review.json'),
            'provenance':prov})
        inputs.append(inp)
    if not inputs: raise ValueError('No supported audit')
    images=inputs[0]['images']
    if any(i['images']!=images or i['prefix_events']!=inputs[0]['prefix_events'] for i in inputs): raise ValueError('Audit prefixes disagree')
    return inputs[0],receipts


def project_answer(source, step, markers):
    stop=step.get('stop_target',{})
    action_marks=covered(markers,stop.get('path',''), 'ANSWER')
    if not action_marks: raise ValueError('Stop not supported')
    root=stop['path'].rsplit('/',1)[0]; original=resolve(source,root)
    target={'action':'ANSWER'}; used=list(action_marks); bindings={'/action':stop}
    for key in ['category','severity','moderation_action']:
        ms=covered(markers,root+'/'+key,original.get(key))
        if not ms: raise ValueError('Answer core field not supported: '+key)
        used+=ms; target[key]=original[key]; bindings['/'+key]={'path':root+'/'+key}
    target['evidence']=[]
    for i,e in enumerate(original.get('evidence',[])):
        path=f'{root}/evidence/{i}/description'; ms=covered(markers,path,e['description'])
        claims=[c for c in step.get('claims',[]) if c.get('target',{}).get('path')==path]
        if not ms or not claims: continue
        if any(c.get('timestamp')!=e['timestamp'] or not c.get('frame_ids') for c in claims): raise ValueError('Evidence citation not bound to reviewed claim')
        j=len(target['evidence']); target['evidence'].append(copy.deepcopy(e)); used+=ms
        bindings[f'/evidence/{j}/description']={'path':path}
        bindings[f'/evidence/{j}/timestamp']={'path':f'{root}/evidence/{i}/timestamp','basis':'Claim citation + mechanical visible timestamp check'}
    for key in ['rationale','review_reason']:
        if key not in original: continue
        ms=covered(markers,root+'/'+key,original[key])
        if ms:
            used+=ms; target[key]=original[key]; bindings['/'+key]={'path':root+'/'+key}
    return target,used,bindings


def export_case(meta, candidate_path, policy):
    folder=candidate_path.parent; source_path=folder/'original.json'
    source=json.loads(source_path.read_text()); candidate=json.loads(candidate_path.read_text())
    if source['id']!=meta['merged_id'] or candidate['trajectory_id']!=source['id'] or source['split']!='train' or meta['frozen_split']!='train' or source['video']['sha256']!=meta['sha256']:
        raise ValueError('Not the original declared Train video')
    if candidate.get('technical_failures') or any(v['severity']!='WARNING' for v in candidate.get('mechanical_issues',[])):
        raise ValueError('Technical failure or blocking mechanical audit issue')
    frames={f['id']:f for f in source['frames']}; seen=set(); blocked=False; terminal=False
    rows=[]; rejected=[]; markers=candidate['supervision_markers']
    for index,step in enumerate(source['steps']):
        if terminal: raise ValueError('Step after terminal ANSWER')
        if step['type']=='observe': seen.update(step['frame_ids']); continue
        if step['type']=='tool_result':
            if step['result'].get('error'): blocked=True
            if source['tools'][step['tool']]['returns_images']: seen.update(step['result'].get('frame_ids',[]))
            continue
        if step['type']!='decision': raise ValueError('Unknown step type')
        sm=[m for m in markers if m['step_id']==step['id']]
        try:
            if blocked: raise ValueError('Unverified critical prefix; original history retained, no splicing')
            visible=step.get('historical_visible_frame_ids')
            if not visible or set(visible)!=seen or any(frames[f]['role']!='recorded_observation' for f in visible): raise ValueError('Visible ledger incomplete/unseen/audit-only image')
            if not step.get('dependencies_complete',False): raise ValueError('Dependencies incomplete')
            if step['action']=='ANSWER':
                target,used,bindings=project_answer(source,step,sm)
            else:
                used=[]
                for key in ['action','parameters']:
                    used+=covered(sm,f'/steps/{index}/{key}',step[key])
                    if not covered(sm,f'/steps/{index}/{key}',step[key]): raise ValueError('Tool action/parameters not supported')
                if not any(s['type']=='tool_result' and s.get('call_step_id')==step['id'] for s in source['steps'][index+1:]): raise ValueError('Tool execution record missing')
                target={'action':step['action'],**step['parameters']}
                bindings={'/action':{'path':f'/steps/{index}/action'},**{
                    '/'+k:{'path':f'/steps/{index}/parameters/{k}'} for k in step['parameters']}}
            inp,receipts=approved_checks(folder,candidate,used,source,index,visible)
            context={'duration_seconds':source['video']['duration_seconds'],'budget':copy.deepcopy(inp['budget']),
                     'events':copy.deepcopy(inp['prefix_events']),'visible_frame_ids':visible}
            images=copy.deepcopy(inp['images'])
            for im in images:
                f=frames[im['id']]
                if f['sha256']!=im['sha256'] or abs(f['timestamp']-im['timestamp'])>1e-8 or f['video_sha256']!=meta['sha256']: raise ValueError('Video/image identity mismatch')
            validate_output(target,context,images)
            content(context,policy,images)  # Hashes, pixels and frozen policy, no model.
            rows.append({'id':source['id']+'::'+step['id']+'::unified-next-step',
                'source_trajectory_id':source['id'],'source_group':meta['source_group'],'source_partition':'train',
                'contract_version':VERSION,'stage':'unified_next_step','action':step['action'],'step_index':index,
                'context':context,'policy_text':policy,'policy_sha256':POLICY_SHA,'prompt':prompt(context,policy),
                'target':json.dumps(target,ensure_ascii=False,separators=(',',':')),
                'audit_images':images,'timestamps':[im['timestamp'] for im in images],
                'supported_output_bindings':bindings,'supervision_mode':'ALL_PROJECTED_TARGET_TOKENS',
                'original_decision':copy.deepcopy(resolve(source,step['stop_target']['path'].rsplit('/',1)[0])) if step.get('stop_target') else {'action':step['action'],**step['parameters']},
                'provenance':{'source_path':str(source_path),'source_sha256':sha(source_path),
                    'candidate_path':str(candidate_path),'candidate_sha256':sha(candidate_path),'audit_receipts':receipts,
                    'historical_pixel_identity':all(frames[f].get('historical_pixel_identity') is True for f in visible),
                    'new_prompt_not_historical_native_request':True,'original_candidate':candidate['candidate']},
                'weak_reference_only':True,'independent_gold':False})
        except (ValueError,KeyError,FileNotFoundError,TypeError) as exc:
            rejected.append({'id':source['id'],'step_id':step['id'],'action':step['action'],'reason':str(exc)})
            blocked=True
        # Never let unsupported critical same-step premises condition later targets.
        if any(c.get('critical',True) and not covered(sm,c.get('target',{}).get('path',''),resolve(source,c['target']['path'])) for c in step.get('claims',[])):
            blocked=True
        terminal=step['action']=='ANSWER'
    return rows,rejected


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--train-manifest',type=Path,required=True)
    ap.add_argument('--policy',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    policy=args.policy.read_text(); rows=[]; rejects=[]; videos=set()
    if sha(args.policy)!=POLICY_SHA: raise ValueError('Wrong frozen policy')
    metas=read(args.train_manifest)
    for meta in metas:
        if meta['frozen_split']!='train': raise ValueError('Only Train manifest may be supplied')
        if meta['merged_id'] in videos: raise ValueError('Duplicate Train video')
        videos.add(meta['merged_id']); path=Path(meta.get('candidate_file',''))
        if not path.is_file():
            rejects.append({'id':meta['merged_id'],'reason':'No existing completed audit candidate'}); continue
        try:
            accepted,failed=export_case(meta,path,policy); rows+=accepted; rejects+=failed
        except (ValueError,KeyError,FileNotFoundError,TypeError) as exc:
            rejects.append({'id':meta['merged_id'],'reason':str(exc)})
    write(args.out/'train.jsonl',rows); write(args.out/'excluded.jsonl',rejects)
    summary={'input_train_videos':len(metas),'rows':len(rows),'unique_videos':len({r['source_trajectory_id'] for r in rows}),
        'actions':dict(Counter(r['action'] for r in rows)),
        'stage_counts':dict(Counter('initial' if not any(e['type']=='decision' for e in r['context']['events']) else 'post_tool' for r in rows)),
        'original_candidate_counts':dict(Counter(r['provenance']['original_candidate'] for r in rows)),
        'historical_pixel_identity_rows':sum(r['provenance']['historical_pixel_identity'] for r in rows),
        'exclusion_reasons':dict(Counter(r['reason'] for r in rejects)),
        'train_sha256':sha(args.out/'train.jsonl'),'train_manifest_sha256':sha(args.train_manifest),
        'policy_sha256':POLICY_SHA,'contract_sha256':sha(Path(__file__).with_name('unified_student_contract.py')),
        'prompt_sha256':sha(INSTRUCTION),'exporter_sha256':sha(Path(__file__)),
        'test_read':False,'model_calls':0,'training_started':False,'weak_reference_only':True}
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    for file in [Path(__file__),Path(__file__).with_name('unified_student_contract.py'),INSTRUCTION]:
        (args.out/file.name).write_bytes(file.read_bytes())
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__': main()
