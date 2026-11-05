"""Hash-bound outcome/level/cost GRPO pilot: prepare, check, one update, reload.

Six real result-difference groups, all four attempts. Cost-only groups are held
aside so tiny cost gaps cannot acquire unit-sized normalized advantages.
"""
import argparse
import gc
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from check_hf_policy_alignment import binding, versions, write
from check_hf_policy_backward import parameter_digest
from hf_request_policy import read_bound_item, score_saved_request
from hf_request_rng import validate_seeds
from outcome_cost_reward_v1 import VERSION, reward
from prepare_grpo_group_selection import ROOT, digest
from review_grpo_selection_qwen import trace
from train_hf_request_grpo_smoke import group_advantages, loss_terms
from train_partial_report_grpo import load_actor, bound_row, progress
from train_teacher_category_grpo_pilot import zero_check, teacher_reference
from unified_student_contract import PUBLIC_CONTEXT_VERSION, public_context


def read(p):return json.loads(Path(p).read_text())


def prepare(root, cohort):
    root.mkdir(parents=True, exist_ok=False)
    original=read(cohort/'PLAN.json');selection=read(cohort/'SELECTION.json')
    stats=read(cohort/'SUMMARY.json')
    wanted={g['group'] for g in stats['groups'] if g['kind']=='RESULT_DIFFERENCE'}
    if len(wanted)!=6:raise ValueError('Expected six frozen result-difference groups')
    dependencies={str(cohort/n):digest(cohort/n) for n in ['PLAN.json','SELECTION.json','SUMMARY.json','TRACEABLE_REWARDS.jsonl']}
    for p,h in original['frozen_source_hashes'].items():
        if digest(p)!=h:raise ValueError('Original source changed')
        dependencies[p]=h
    for x in original['exclusion_files']:
        if digest(x['path'])!=x['sha256']:raise ValueError('Exclusion changed')
        dependencies[x['path']]=x['sha256']
    rows=[];groups=[];seeds=set();behavior=None;contract=None
    for selected in selection:
        if selected['group'] not in wanted:continue
        source=selected['source']
        if source['frozen_split']!='train' or digest(source['video_path'])!=source['sha256']:
            raise ValueError('Invalid Train identity/media')
        ref=teacher_reference(source)
        summary_path=Path(selected['rollout_dir'])/'summary.json';summary=read(summary_path)
        dependencies[str(summary_path)]=digest(summary_path)
        current=set(validate_seeds(summary['results'],selected['samples'],read_bound_item))
        if seeds & current:raise ValueError('Colliding request RNG streams')
        seeds.update(current)
        results=sorted(summary['results'],key=lambda x:x['sample'])
        if [r['sample'] for r in results]!=[0,1,2,3]:raise ValueError('Incomplete original group')
        totals=[];tasks=[];group_rows=[]
        for result in results:
            t=trace(selected,result);answer=t['final_report'];occurrences=0
            for turn,call in enumerate(result['requests']):
                path=Path(call['item_path']);item,request,response=read_bound_item(path)
                behavior=behavior or item['model_binding'];contract=contract or item['sampling_contract']
                if item['model_binding']!=behavior or item['sampling_contract']!=contract:
                    raise ValueError('Mixed behavior policies')
                if request.get('public_context_version')!=PUBLIC_CONTEXT_VERSION or request['context']!=public_context(request['context']):
                    raise ValueError('Non-public model input')
                text='\n'.join(x['text'] for x in request['content_order'] if x['type']=='text')
                if any(k in text for k in ['teacher_record','candidate_file','video_path','reference_category']):
                    raise ValueError('Private label/source hint in prompt')
                occurrences+=len(request['images'])
                group_rows.append({'group':selected['group'],'sample':result['sample'],'turn':turn,
                    'trajectory_key':f"{selected['group']}/{result['sample']}",'source_id':source['merged_id'],
                    'item_path':str(path),'item_sha256':digest(path),'sampler_environment':summary['environment'],
                    'actual_tool_calls':len(t['tools'])})
            r=reward(reference_category=source['reference_category'],reference_severity=source['reference_severity'],
                predicted_category=answer['category'],predicted_severity=answer['severity'],
                visual_inputs=occurrences,boundary=source['reference_boundary'])
            if r['total'] is None:raise ValueError('Unresolved reward')
            totals.append(r['total']);tasks.append(r['task'])
        frozen=next(g for g in stats['groups'] if g['group']==selected['group'])
        if frozen['rewards']!=totals or len(set(tasks))<2:raise ValueError('Frozen rewards changed or cost-only group')
        advantages=group_advantages(totals)
        if advantages is None:raise ValueError('No relative signal')
        groups.append({'group':selected['group'],'source':source,'teacher_reference':ref,
                       'rewards':totals,'task_scores':tasks,'advantages':advantages})
        for row in group_rows:
            row.update(reward=totals[row['sample']],advantage=advantages[row['sample']],active=True)
            rows.append(row)
    if binding(Path(behavior['base_path']),Path(behavior['checkpoint_path']))!=behavior:
        raise ValueError('Behavior checkpoint changed')
    helpers=['train_teacher_category_grpo_pilot.py','train_partial_report_grpo.py',
             'hf_request_policy.py','train_hf_request_grpo_smoke.py','outcome_cost_reward_v1.py',
             'hf_request_rng.py','review_grpo_selection_qwen.py']
    dependencies.update({str(Path(__file__).with_name(n)):digest(Path(__file__).with_name(n)) for n in helpers})
    plan={'protocol':'legacy-outcome-level-cost-grpo-one-update-v1','script_sha256':digest(__file__),
        'cohort':str(cohort),'model_binding':behavior,'sampling_contract':contract,'environment':versions(),
        'rows':rows,'selected_groups':sorted(wanted),'selected_trajectories':24,'selected_model_requests':len(rows),
        'reward_version':VERSION,'frozen_dependencies':dependencies,'optimizer_steps_maximum':1,
        'learning_rate':1e-6,'clip_epsilon':.2,'kl_beta':.01,'gradient_clip_norm':1.,
        'thresholds':{'mean_free_logprob_error':.05,'maximum_logprob_error':.2},
        'normalization':'sum free assistant-token losses / each trajectory total free tokens / 24 trajectories',
        'behavior_denominator':'actual recorded masked-temperature sample logprobs, never replacement rescoring',
        'credit_scope':'all actual free assistant decision tokens across all turns; no prompt, image, tool result or grammar-forced tokens',
        'cost_only_groups_excluded':[g['group'] for g in stats['groups'] if g['kind']=='COST_ONLY_DIFFERENCE'],
        'teacher_role':'weak old Train reference, not independent ground truth',
        'dev_test_used':False,'fixed120_used':False,'frozen_unix':time.time()}
    write(root/'PLAN.json',plan);write(root/'GROUP_REWARDS.json',groups)
    (root/'CODE_SNAPSHOT.py').write_bytes(Path(__file__).read_bytes())
    summary={'status':'PREPARED','groups':6,'trajectories':24,'model_requests':len(rows),
             'categories':dict(Counter(g['source']['reference_category'] for g in groups)),
             'all_four_attempts_retained':True,'optimizer_steps':0}
    write(root/'PREPARATION_SUMMARY.json',summary);progress(root,'PREPARED',**{k:v for k,v in summary.items() if k!='status'})


def validate(root):
    plan=read(root/'PLAN.json')
    if digest(__file__)!=plan['script_sha256']:raise ValueError('Training code changed')
    for p,h in plan['frozen_dependencies'].items():
        if digest(p)!=h:raise ValueError('Dependency changed: '+p)
    for group in read(root/'GROUP_REWARDS.json'):
        if teacher_reference(group['source'])!=group['teacher_reference']:raise ValueError('Teacher changed')
    return plan


def update(root,plan):
    check=read(root/'zero_update_check/summary.json')
    if check['status']!='PASS' or len(check['rows'])!=len(plan['rows']):raise ValueError('Zero-update check incomplete')
    out=root/'one_update';out.mkdir(exist_ok=False);start=time.time();steps=0
    try:
        progress(out,'LOADING_ORIGINAL_SFT',optimizer_steps=0)
        processor,actor=load_actor(plan)
        before=parameter_digest(actor)
        if before!=check['parameter_sha256_before']:raise ValueError('Check/update start differs')
        params=[p for p in actor.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=plan['learning_rate'],weight_decay=0.)
        optimizer.zero_grad(set_to_none=True);pre=0.
        for index,row in enumerate(check['rows']):
            progress(out,'ACCUMULATING_ONE_UPDATE',completed=index,total=len(check['rows']),optimizer_steps=0)
            path,_,_,_=bound_row(row,plan)
            if digest(row['reference_path'])!=row['reference_sha256']:raise ValueError('Reference changed')
            with np.load(row['reference_path']) as a:
                reference=torch.from_numpy(a['reference']).cuda();behavior=torch.from_numpy(a['behavior']).cuda();free=torch.from_numpy(a['free']).cuda()
            current,counts,_=score_saved_request(actor,processor,path)
            if not torch.equal(counts>1,free):raise ValueError('Free mask changed')
            error=(current.detach()-behavior).abs()
            if float(error[free].mean())>.05 or float(error.max())>.2:raise ValueError('Pre-update behavior drift')
            terms,_,_=loss_terms(current,behavior,reference,row['advantage'],plan['clip_epsilon'],plan['kl_beta'])
            loss=terms[free].sum()/row['trajectory_total_free_tokens']/24
            loss.backward();pre+=float(loss.detach())
            del reference,behavior,free,current,counts,error,terms,loss;torch.cuda.empty_cache()
        grads=[p.grad for p in params if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):raise ValueError('Invalid accumulated gradient')
        if any(p.grad is not None for p in actor.parameters() if not p.requires_grad):raise ValueError('Base gradient leak')
        norm=float(torch.nn.utils.clip_grad_norm_(params,plan['gradient_clip_norm']))
        if not 0<norm<math.inf:raise ValueError('Invalid gradient norm')
        optimizer.step();steps=1;optimizer.zero_grad(set_to_none=True)
        after=parameter_digest(actor)
        if after==before:raise ValueError('Weights did not change')
        adapter=out/'adapter_after_one_update';actor.save_pretrained(adapter)
        torch.save(optimizer.state_dict(),out/'optimizer_state.pt')
        progress(out,'POST_UPDATE_REPLAY',optimizer_steps=1)
        post=0.;kl_sum=0.;tokens=0;diagnostics=[]
        for index,row in enumerate(check['rows']):
            with np.load(row['reference_path']) as a:
                reference=torch.from_numpy(a['reference']).cuda();behavior=torch.from_numpy(a['behavior']).cuda();free=torch.from_numpy(a['free']).cuda()
            with torch.inference_mode():
                current,_,_=score_saved_request(actor,processor,row['item_path'])
                terms,ratio,kl=loss_terms(current,behavior,reference,row['advantage'],plan['clip_epsilon'],plan['kl_beta'])
            if not bool(torch.isfinite(terms[free]).all()):raise ValueError('Nonfinite post-update objective')
            post+=float(terms[free].sum())/row['trajectory_total_free_tokens']/24
            kl_sum+=float(kl[free].sum());tokens+=int(free.sum())
            if index==0:np.savez_compressed(out/'reload_reference.npz',scores=current.cpu().numpy())
            diagnostics.append({'group':row['group'],'sample':row['sample'],'turn':row['turn'],
                'advantage':row['advantage'],'mean_logprob_delta':float((current-reference)[free].mean()),
                'ratio_min':float(ratio[free].min()),'ratio_max':float(ratio[free].max())})
            del reference,behavior,free,current,terms,ratio,kl;torch.cuda.empty_cache()
        old=plan['model_binding']
        if binding(Path(old['base_path']),Path(old['checkpoint_path']))!=old:raise ValueError('Original SFT changed')
        saved=torch.load(out/'optimizer_state.pt',map_location='cpu',weights_only=True)
        if not saved['state'] or any(int(v['step'])!=1 for v in saved['state'].values()):raise ValueError('Optimizer step accounting failed')
        summary={'status':'ONE_OUTCOME_LEVEL_COST_UPDATE_COMPLETE_NOT_SELECTED',
            'optimizer_steps':1,'adapter_path':str(adapter.resolve()),'learning_rate':plan['learning_rate'],
            'parameter_sha256_before':before,'parameter_sha256_after':after,'pre_surrogate_loss':pre,
            'post_surrogate_loss':post,'mean_reference_kl':kl_sum/tokens,'gradient_norm_before_clip':norm,
            'original_checkpoint_unchanged':True,'base_parameters_frozen':True,'reward_version':VERSION,
            'selected_groups':plan['selected_groups'],'trajectories':24,'model_requests':len(check['rows']),
            'request_diagnostics':diagnostics,'elapsed_seconds':time.time()-start,
            'max_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'accuracy_or_efficiency_improvement_claimed':False,'dev_test_used':False,'fixed120_used':False}
        write(out/'summary.json',summary);progress(out,'COMPLETE',optimizer_steps=1)
    except Exception as exc:
        write(out/'FAILURE.json',{'error':repr(exc),'optimizer_steps_executed':steps,'automatic_retry':False});raise


def reload_check(root,plan):
    out=root/'reload_check';out.mkdir(exist_ok=False)
    update=read(root/'one_update/summary.json')
    progress(out,'LOADING_SAVED_UPDATED_ADAPTER',optimizer_steps=0)
    new_plan={**plan,'model_binding':binding(Path(plan['model_binding']['base_path']),Path(update['adapter_path']))}
    processor,actor=load_actor(new_plan)
    restored=parameter_digest(actor)
    if restored!=update['parameter_sha256_after']:raise ValueError('Reloaded trainable parameters differ')
    path=plan['rows'][0]['item_path']
    with torch.inference_mode():scores,_,_=score_saved_request(actor,processor,path)
    with np.load(root/'one_update/reload_reference.npz') as saved:
        delta=float(np.max(np.abs(scores.cpu().numpy()-saved['scores'])))
    if delta>1e-5:raise ValueError('Restored fixed-token scores differ')
    write(out/'summary.json',{'status':'PASS','parameter_sha256':restored,
        'maximum_reloaded_score_difference':delta,'optimizer_steps':0,'real_model_reload':True})
    progress(out,'PASS',optimizer_steps=0)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=['prepare','check','update','reload'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cohort',type=Path,default=ROOT/'video_detection_agent/outputs/legacy_train_grpo_pilot_20261102_v1')
    a=ap.parse_args();root=a.out.resolve()
    if a.phase=='prepare':prepare(root,a.cohort.resolve())
    else:
        try:
            plan=validate(root)
            if a.phase=='check':zero_check(root,plan)
            elif a.phase=='update':update(root,plan)
            else:reload_check(root,plan)
        except Exception as exc:
            write(root/('FAILURE_'+a.phase+'.json'),{'phase':a.phase,'error':repr(exc),'automatic_retry':False})
            raise
