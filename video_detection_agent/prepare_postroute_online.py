#!/usr/bin/env python3
"""Reuse audited Train prefixes, flattened to the exact online router template."""
import argparse
import copy
import hashlib
import io
import json
import random
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
from check_selected_sft_live import route_prompt, POLICY_SHA
from evaluate_thin_harness_teacher_pilot import _history_summary


TOOLS={'JOINT_SEMANTIC_SCAN','TEMPORAL_EXPAND'}


def read(path):return [json.loads(l) for l in path.open() if l.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))


def flatten(row,meta,base):
    images={};initial=[];acquired=[];history=[];tool_count=0;pending=None;states=[]
    for index,turn in enumerate(row['turns']):
        if turn['role']=='user':
            returned=turn['audit_images']
            fresh=[]
            for image in returned:
                key=round(image['timestamp'],6)
                if key not in images:fresh.append(image['timestamp'])
                images[key]=image
            if not history:
                initial=[im['timestamp'] for im in returned]
                history.append({'observation':{'duration':float(meta['duration_seconds']),
                    'initial_timestamps':initial}})
            else:
                assert pending and pending['action'] in TOOLS
                payload=json.loads(turn['text'][turn['text'].index('{'):])
                assert payload['tool']==pending['action']
                acquired+=fresh;tool_count+=1
                event={'tool':payload['tool'],'executed_action':payload['tool'],
                       'added_timestamps':fresh}
                if payload['tool']=='JOINT_SEMANTIC_SCAN':
                    event.update(queries_by_risk=pending['queries_by_risk'],matches=payload['candidates'])
                else:event.update(anchors=pending['timestamps'])
                history.append(event)
            continue
        assert turn['role']=='assistant'
        proposal=json.loads(turn['content'])
        visible=sorted(images)
        if len(visible)>12:
            visible=sorted(set(round(t,6) for t in initial[:4]+acquired[-8:]))
        stamps=[images[key]['timestamp'] for key in visible]
        if tool_count:
            state={'turn':sum('raw_decision' in h for h in history)+1,
                'duration':round(float(meta['duration_seconds']),3),
                'visible_timestamps':[round(t,3) for t in stamps],
                'all_acquired_timestamps':[round(t,3) for t in initial+acquired],
                'tools_remaining':3-tool_count,'frames_remaining':24-len(images),
                'recent_tool_observations':history[-3:],'history_summary':_history_summary(history)}
            assert state['tools_remaining']>=0 and state['frames_remaining']>=0
            states.append({'id':row['id']+f'::prefix-{index}',
                'source_trajectory_id':row['id'],'source_partition':'train',
                'video_path':meta['video_path'],'timestamps':stamps,
                'audit_images':[copy.deepcopy(images[k]) for k in visible],
                'prompt':route_prompt(base,stamps,state),'target':turn['content'],
                'stage':'post_route','category':meta['mapped_category'],
                'action':proposal['action'],'prefix_turn_index':index,
                'weak_reference_only':True,'independent_gold':False,
                'audit_provenance':row['audit_provenance']})
        # No later tool result/answer is consulted while building this state.
        history.append({'turn':sum('raw_decision' in h for h in history)+1,
                        'raw_decision':copy.deepcopy(proposal),'visible_timestamps':stamps})
        pending=proposal
    return states


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pack',required=True,type=Path)
    ap.add_argument('--pool',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args();p=args.pack.resolve();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    train={r['merged_id']:r for r in read(p/'train_manifest.jsonl')}
    dev=read(p/'dev_manifest.jsonl')
    base=read(p/'train_route_round1.jsonl')[0]['prompt'].rsplit('\n当前真实已见帧时间戳：',1)[0]
    audited=[]
    for row in read(args.pool):
        meta=train.get(row['id'])
        if not meta or meta['route_tier']=='DIRECT' or meta['warnings']:continue
        provenance=row['audit_provenance'];candidate_path=Path(provenance['candidate_path'])
        assert sha(candidate_path)==provenance['candidate_sha256']
        candidate=json.loads(candidate_path.read_text())
        if candidate['candidate']!='FULL_TRAJECTORY_CANDIDATE':continue
        assert provenance['policy_sha256']==POLICY_SHA
        assert candidate['trajectory_id']==row['id']
        audited.extend(flatten(row,meta,base))
    counts=Counter(r['action'] for r in audited)
    assert counts['ANSWER'] and counts['TEMPORAL_EXPAND']
    # The scarce repeat-SCAN examples are not invented or replaced by EXPAND.
    by_action=defaultdict(list)
    for r in audited:by_action[r['action']].append(r)
    balanced=[];rng=random.Random(20261017)
    for action,rows in sorted(by_action.items()):
        for i in range(max(counts.values())):
            copied=copy.deepcopy(rows[i%len(rows)])
            copied['oversampled_from']=copied['id'];copied['id']+=f'::replica-{i}'
            balanced.append(copied)
    rng.shuffle(balanced)
    # Freeze new post-tool dev BEFORE training: six one-tool videos per class
    # plus every multi-tool dev. No selection by candidate predictions.
    selected=[]
    for category in sorted({r['mapped_category'] for r in dev}):
        selected+=sorted([r for r in dev if r['route_tier']=='ONE_TOOL'
                          and r['mapped_category']==category and not r['warnings']],
                         key=lambda r:r['merged_id'])[:6]
    selected+=sorted([r for r in dev if r['route_tier']=='TWO_PLUS_TOOLS'],key=lambda r:r['merged_id'])
    wanted=defaultdict(dict)
    for r in selected:wanted[r['teacher_record_file']][int(r['teacher_record_line'])]=r
    raw={}
    for filename,lines in wanted.items():
        for n,line in enumerate(Path(filename).open('rb'),1):
            if n in lines:
                assert hashlib.sha256(line).hexdigest()==lines[n]['teacher_record_sha256']
                raw[lines[n]['merged_id']]=json.loads(line)
    dev_states=[]
    for meta in selected:
        transcript=raw[meta['merged_id']]['result']['transcript'];tools=0;history=[]
        initial=transcript[0]['observation']['initial_timestamps'];acquired=[]
        for index,event in enumerate(transcript):
            if 'raw_decision' in event and tools:
                stamps=event['visible_timestamps']
                state={'turn':event['turn'],'duration':round(float(meta['duration_seconds']),3),
                    'visible_timestamps':[round(t,3) for t in stamps],
                    'all_acquired_timestamps':[round(t,3) for t in initial+acquired],
                    'tools_remaining':3-tools,'frames_remaining':24-len(initial)-len(acquired),
                    'recent_tool_observations':history[-3:],'history_summary':_history_summary(history)}
                dev_states.append({'id':meta['merged_id']+f'::prefix-{index}',
                    'source_trajectory_id':meta['merged_id'],'source_partition':'dev',
                    'video_path':meta['video_path'],'timestamps':stamps,
                    'prompt':route_prompt(base,stamps,state),
                    'target':json.dumps(event['raw_decision'],ensure_ascii=False,separators=(',',':')),
                    'stage':'post_route','category':meta['mapped_category'],
                    'action':event['raw_decision']['action'],
                    'teacher_record_sha256':meta['teacher_record_sha256'],
                    'weak_reference_only':True,'independent_gold':False})
            history.append(event)
            if event.get('executed_action') in TOOLS:
                tools+=1;acquired+=event['added_timestamps']
    assert not {r['source_trajectory_id'] for r in audited}&{r['source_trajectory_id'] for r in dev_states}
    out.mkdir(parents=True);(out/'images').mkdir()
    transformed={}
    for r in audited+balanced:
        for im in r['audit_images']:
            source=Path(im['path']);assert sha(source)==im['sha256']
            source_sha=im['sha256']
            if source_sha not in transformed:
                image=Image.open(source).convert('RGB')
                if image.width*image.height>409600:
                    scale=(409600/(image.width*image.height))**.5
                    image=image.resize((max(1,int(image.width*scale)),max(1,int(image.height*scale))))
                buf=io.BytesIO();image.save(buf,format='JPEG',quality=95);b=buf.getvalue()
                path=out/'images'/(hashlib.sha256(b).hexdigest()+'.jpg');path.write_bytes(b)
                transformed[source_sha]=(str(path),sha(path))
            im['audit_source_path']=str(source);im['audit_source_sha256']=source_sha
            im['path'],im['sha256']=transformed[source_sha]
            im['transform']='RGB; same runtime 409600-pixel budget; JPEG quality 95; not a new visual audit'
    write(out/'train_unique.jsonl',audited);write(out/'train.jsonl',balanced)
    write(out/'dev.jsonl',dev_states);write(out/'dev_manifest.jsonl',selected)
    summary={'unique_train_videos':len({r['source_trajectory_id'] for r in audited}),
        'unique_train_states':len(audited),'train_action_counts':dict(counts),
        'balanced_train_rows':len(balanced),'dev_videos':len(selected),'dev_states':len(dev_states),
        'dev_action_counts':dict(Counter(r['action'] for r in dev_states)),
        'source_export_sha256':sha(args.pool),'policy_sha256':POLICY_SHA,
        'train_sha256':sha(out/'train.jsonl'),'dev_sha256':sha(out/'dev.jsonl'),
        'new_semantic_audit':False,'weak_reference_only':True,'independent_gold':False,
        'test_used':False,'transform_note':'Image source bindings preserved; runtime resizing/encoding recorded.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'script_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(summary))


if __name__=='__main__':main()
