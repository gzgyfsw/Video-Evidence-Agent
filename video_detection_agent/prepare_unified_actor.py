#!/usr/bin/env python3
"""One trainable Actor for both initial and post-tool decisions; no label changes."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import random
from prepare_postroute_online import read,write,sha
from check_selected_sft_live import route_prompt
from evaluate_thin_harness_teacher_pilot import _history_summary


def online_initial(row,meta):
    copied=copy.deepcopy(row);times=row['timestamps']
    base=row['prompt'].rsplit('\n当前真实已见帧时间戳：',1)[0]
    history=[{'observation':{'duration':float(meta['duration_seconds']),'initial_timestamps':times}}]
    state={'turn':1,'duration':round(float(meta['duration_seconds']),3),
        'visible_timestamps':[round(t,3) for t in times],
        'all_acquired_timestamps':[round(t,3) for t in times],
        'tools_remaining':3,'frames_remaining':24-len(times),
        'recent_tool_observations':history,'history_summary':_history_summary(history)}
    copied['prompt']=route_prompt(base,times,state)
    copied['online_initial_view']=True
    return copied


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pack',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    p=args.pack.resolve();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    trainmeta={r['merged_id']:r for r in read(p/'train_manifest.jsonl')}
    devmeta={r['merged_id']:r for r in read(p/'dev_manifest.jsonl')}
    initial=[online_initial(r,trainmeta[r['source_trajectory_id']]) for r in read(p/'train_route_round1.jsonl')]
    dev=[online_initial(r,devmeta[r['source_trajectory_id']]) for r in read(p/'dev_route.jsonl')]
    post=read(p/'postroute_online_v1/train.jsonl')
    rows=initial[:]
    for i in range(2):
        for r in post:
            c=copy.deepcopy(r);c['unified_replay_source_id']=c['id'];c['id']+=f'::joint-replica-{i}'
            rows.append(c)
    assert all(r['source_partition']=='train' for r in rows)
    assert not {r['source_trajectory_id'] for r in rows}&set(devmeta)
    forbidden={r['merged_id'] for r in read(p/'test_manifest.jsonl')}
    assert not {r['source_trajectory_id'] for r in rows}&forbidden
    random.Random(20261017).shuffle(rows)
    out.mkdir(parents=True);write(out/'train.jsonl',rows);write(out/'dev_initial_online.jsonl',dev)
    summary={'rows':len(rows),'unique_videos':len({r['source_trajectory_id'] for r in rows}),
        'stage_counts':dict(Counter(r['stage'] for r in rows)),
        'actions':dict(Counter(json.loads(r['target'])['action'] for r in rows)),
        'online_dev_rows':len(dev),'train_sha256':sha(out/'train.jsonl'),
        'online_dev_sha256':sha(out/'dev_initial_online.jsonl'),
        'data_changed':False,'test_used':False,'weak_reference_only':True,'independent_gold':False,
        'interpretation':'Initial and post-tool task views only; full target JSON copied, only action prefix supervised.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'script_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(summary))


if __name__=='__main__':main()
