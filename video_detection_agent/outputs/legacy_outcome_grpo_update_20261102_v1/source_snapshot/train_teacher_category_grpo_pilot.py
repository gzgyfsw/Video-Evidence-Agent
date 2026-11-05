"""Teacher-category-only terminal rewards on genuine four-attempt Train groups.

One bounded full-trajectory GRPO update from the original SFT. No local-Qwen
semantic rewards or new image review. Matching the frozen Teacher is the task;
the reference is not asserted to be independent visual ground truth.
"""
import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from check_hf_policy_alignment import binding, fingerprint, versions, write
from check_hf_policy_backward import parameter_digest
from hf_request_policy import read_bound_item, prompt_batch, score_saved_request
from hf_request_rng import validate_seeds
from prepare_grpo_group_selection import ROOT, digest, identities
from review_grpo_selection_qwen import trace
from train_hf_request_grpo_smoke import group_advantages, loss_terms
from train_partial_report_grpo import load_actor, bound_row, progress
from unified_student_contract import PUBLIC_CONTEXT_VERSION, public_context


LABELS={'Harmless','Sexual Content','Violence & Dangerous','Sensitive Figures & Ideologies'}


def read(path):return json.loads(Path(path).read_text())


def category_reward(predicted,teacher):
    if teacher not in LABELS:raise ValueError('Unknown Teacher category must not become a negative label')
    if predicted is not None and predicted not in LABELS:raise ValueError('Invalid model category')
    return float(predicted==teacher)


def teacher_reference(source):
    path=Path(source['teacher_record_file']);line=int(source['teacher_record_line'])
    raw=None
    with path.open('rb') as handle:
        for i,data in enumerate(handle,1):
            if i==line:raw=data;break
    if raw is None or hashlib.sha256(raw).hexdigest()!=source['teacher_record_sha256']:
        raise ValueError('Original Teacher line/hash mismatch')
    obj=json.loads(raw);final=obj.get('result',{}).get('final_action',{})
    category=final.get('category')
    if obj.get('id')!=source['teacher_id'] or obj.get('model')!=source['teacher_model']:
        raise ValueError('Teacher identity/model mismatch')
    if final.get('action')!='ANSWER' or category not in LABELS:
        raise ValueError('Teacher has no usable final four-class answer')
    if category!=source['teacher_category'] or category!=source['mapped_category']:
        raise ValueError('Teacher/Train label mapping conflict')
    return {'category':category,'model':obj['model'],'id':obj['id'],'path':str(path),
            'line':line,'line_sha256':hashlib.sha256(raw).hexdigest(),'weak_training_reference':True}


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    cohort=ROOT/'video_detection_agent/outputs/grpo_group_selection_20261030_final_v1'
    original=read(cohort/'PLAN.json');selection=read(cohort/'SELECTION.json')
    if digest(original['source_manifest'])!=original['train_manifest_sha256']:raise ValueError('Train manifest changed')
    exclusions=set()
    for row in original['exclusion_files']:
        if digest(row['path'])!=row['sha256']:raise ValueError('Exclusion source changed')
        for sample in read(row['path']):exclusions.update(identities(sample['source']))
    rows=[];groups=[];seen=set();model_binding=None;contract=None;all_request_seeds=set()
    dependencies={str(cohort/'SELECTION.json'):digest(cohort/'SELECTION.json'),
                  str(cohort/'PLAN.json'):digest(cohort/'PLAN.json')}
    for selected in selection:
        source=selected['source'];keys=identities(source)
        if source.get('frozen_split')!='train' or keys&(seen|exclusions):raise ValueError('Excluded, aliased or non-Train video')
        seen.update(keys)
        if digest(source['video_path'])!=source['sha256']:raise ValueError('Source video changed')
        ref=teacher_reference(source);folder=Path(selected['rollout_dir']);summary=read(folder/'summary.json')
        dependencies[str(folder/'summary.json')]=digest(folder/'summary.json')
        seeds=set(validate_seeds(summary['results'],selected['samples'],read_bound_item))
        if all_request_seeds & seeds:raise ValueError('Request RNG seeds collide across groups')
        all_request_seeds.update(seeds)
        results=sorted(summary['results'],key=lambda x:x['sample'])
        if [r['sample'] for r in results]!=[0,1,2,3]:raise ValueError('Exactly four original attempts required')
        rewards=[];group_rows=[]
        for result in results:
            verified=trace(selected,result)
            rewards.append(category_reward(verified['final_report']['category'],ref['category']))
            for turn,call in enumerate(result['requests']):
                path=Path(call['item_path']);item,request,response=read_bound_item(path)
                model_binding=model_binding or item['model_binding'];contract=contract or item['sampling_contract']
                if item['model_binding']!=model_binding or item['sampling_contract']!=contract:raise ValueError('Mixed behavior policy')
                if request.get('public_context_version')!=PUBLIC_CONTEXT_VERSION or request['context']!=public_context(request['context']):
                    raise ValueError('Non-public context in Student request')
                text='\n'.join(x['text'] for x in request['content_order'] if x['type']=='text')
                if any(x in text for x in ['teacher_record','candidate_file','video_path']):raise ValueError('Private label/source hint in Student prompt')
                group_rows.append({'group':selected['group'],'sample':result['sample'],'turn':turn,
                    'trajectory_key':f"{selected['group']}/{result['sample']}",'source_id':source['merged_id'],
                    'item_path':str(path),'item_sha256':digest(path),'sampler_environment':summary['environment'],
                    'actual_tool_calls':len(verified['tools'])})
        advantages=group_advantages(rewards)
        selected_group=advantages is not None
        groups.append({'group':selected['group'],'source':source,'teacher_reference':ref,'rewards':rewards,
            'predicted_categories':[r['answer']['category'] for r in results],
            'advantages':advantages,'selected_for_update':selected_group,
            'reason':'NONZERO_CATEGORY_REWARD_VARIANCE' if selected_group else 'FOUR_ATTEMPTS_SAME_REWARD',
            'rollout_dir':str(folder),'requests':[r['item_path'] for r in group_rows]})
        if selected_group:
            for row in group_rows:
                row.update(reward=rewards[row['sample']],advantage=advantages[row['sample']],active=True)
                rows.append(row)
    selected_groups=[g for g in groups if g['selected_for_update']]
    if not selected_groups:raise ValueError('No category-relative signal; do not fabricate a training update')
    if binding(Path(model_binding['base_path']),Path(model_binding['checkpoint_path']))!=model_binding:
        raise ValueError('Original SFT changed')
    helpers=['train_partial_report_grpo.py','hf_request_policy.py','train_hf_request_grpo_smoke.py']
    dependencies.update({str(Path(__file__).with_name(name)):digest(Path(__file__).with_name(name)) for name in helpers})
    plan={'protocol':'teacher-category-only-grpo-one-update-v1','script_sha256':digest(__file__),
        'cohort':str(cohort),'model_binding':model_binding,'sampling_contract':contract,'environment':versions(),
        'rows':rows,'selected_groups':[g['group'] for g in selected_groups],
        'selected_trajectories':4*len(selected_groups),'selected_model_requests':len(rows),
        'reward':'1 iff final Student category equals frozen Teacher final category, else 0; valid category=null abstention is a non-match; no severity, cost, format bonus or semantic score',
        'teacher_reference_role':'weak fixed Train target, not independent ground truth',
        'credit_scope':'all grammar-free assistant tokens across every actual Student decision request in the selected four-attempt groups; never tool outputs/prompt/image tokens',
        'normalization':'sum request free-token losses / trajectory total free tokens / selected trajectories',
        'behavior_denominator':'original actual sampled masked-temperature probabilities; no replacement with reference rescores',
        'original_sft_initializer':True,'prior_qwen_semantic_adapter_used':False,
        'optimizer_steps_maximum':1,'learning_rate':1e-6,'clip_epsilon':.2,'kl_beta':.01,'gradient_clip_norm':1.,
        'thresholds':{'mean_free_logprob_error':.05,'maximum_logprob_error':.2},
        'frozen_dependencies':dependencies,'exclusion_files':original['exclusion_files'],
        'fixed120_used':False,'qwen_semantic_rewards_used':False,'new_online_review':False,
        'no_semantic_quality_improvement_claimed':True,'frozen_unix':time.time()}
    write(out/'PLAN.json',plan);write(out/'GROUP_REWARDS.json',groups)
    (out/'CODE_SNAPSHOT.py').write_bytes(Path(__file__).read_bytes())
    summary={'groups_audited':len(groups),'attempts_audited':len(groups)*4,
        'teacher_matches':sum(sum(g['rewards']) for g in groups),'selected_groups':plan['selected_groups'],
        'selected_trajectories':plan['selected_trajectories'],'selected_model_requests':len(rows),
        'zero_variance_groups_skipped':len(groups)-len(selected_groups),
        'selected_category_distribution':dict(Counter(g['teacher_reference']['category'] for g in selected_groups)),
        'only_correct_attempts_filter_used':False,'optimizer_steps':0,'fixed120_used':False}
    write(out/'PREPARATION_SUMMARY.json',summary);progress(out,'PREPARED',**summary)


def validate_plan(root):
    plan=read(root/'PLAN.json')
    if digest(__file__)!=plan['script_sha256']:raise ValueError('Frozen training code changed')
    for path,expected in plan['frozen_dependencies'].items():
        if digest(path)!=expected:raise ValueError('Frozen dependency changed: '+path)
    groups=read(root/'GROUP_REWARDS.json')
    for g in groups:
        if teacher_reference(g['source'])!=g['teacher_reference']:raise ValueError('Frozen Teacher reference changed')
        if [category_reward(c,g['teacher_reference']['category']) for c in g['predicted_categories']]!=g['rewards']:
            raise ValueError('Frozen category rewards changed')
    if [g['group'] for g in groups if g['selected_for_update']]!=plan['selected_groups']:raise ValueError('Selection changed')
    return plan


def zero_check(root,plan):
    out=root/'zero_update_check';out.mkdir(exist_ok=False);start=time.time()
    progress(out,'LOADING_ORIGINAL_SFT',optimizer_steps=0)
    processor,actor=load_actor(plan);before=parameter_digest(actor)
    prepared=[];totals=Counter()
    for index,row in enumerate(plan['rows']):
        progress(out,'FREE_TOKEN_REFERENCE_REPLAY',completed=index,total=len(plan['rows']),group=row['group'],sample=row['sample'],turn=row['turn'],optimizer_steps=0)
        path,item,request,response=bound_row(row,plan);batch=prompt_batch(request,processor,'cuda')
        if fingerprint(batch)!=response['prompt_tensor_receipt']:raise ValueError('Actual prompt/image tensors changed')
        del batch
        with torch.inference_mode():reference,counts,ids=score_saved_request(actor,processor,path)
        free=counts>1;behavior=torch.tensor(response['sampled_processed_logprobs'],device='cuda')
        error=(reference-behavior).abs()
        if not bool(free.any()) or float(error[free].mean())>.05 or float(error.max())>.2:raise ValueError('Sampling/reference alignment failed')
        name=f"group-{row['group']:02d}-sample-{row['sample']}-turn-{row['turn']}"
        refpath=out/(name+'.npz')
        np.savez_compressed(refpath,reference=reference.cpu().numpy(),behavior=behavior.cpu().numpy(),free=free.cpu().numpy())
        r={**row,'reference_path':str(refpath.resolve()),'reference_sha256':digest(refpath),
            'free_tokens':int(free.sum()),'mean_free_logprob_error':float(error[free].mean()),'maximum_logprob_error':float(error.max())}
        prepared.append(r);totals[row['trajectory_key']]+=r['free_tokens']
        del reference,counts,ids,free,behavior,error;torch.cuda.empty_cache()
    checked=[]
    for index,row in enumerate(prepared):
        progress(out,'DIFFERENTIABLE_REPLAY_AND_BACKWARD',completed=index,total=len(prepared),group=row['group'],sample=row['sample'],turn=row['turn'],optimizer_steps=0)
        with np.load(row['reference_path']) as arr:
            reference=torch.from_numpy(arr['reference']).to('cuda');behavior=torch.from_numpy(arr['behavior']).to('cuda');free=torch.from_numpy(arr['free']).to('cuda')
        actor.zero_grad(set_to_none=True);current,counts,_=score_saved_request(actor,processor,row['item_path'])
        if not torch.equal(counts>1,free):raise ValueError('Original grammar-free mask changed')
        error=(current.detach()-behavior).abs()
        if float(error[free].mean())>.05 or float(error.max())>.2:raise ValueError('Differentiable replay alignment failed')
        terms,_,_=loss_terms(current,behavior,reference,row['advantage'],plan['clip_epsilon'],plan['kl_beta'])
        value=terms[free].sum()/totals[row['trajectory_key']]/plan['selected_trajectories'];value.backward()
        grads=[p.grad for p in actor.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):raise ValueError('Missing/nonfinite LoRA gradients')
        norm=math.sqrt(sum(float(g.float().square().sum()) for g in grads))
        if not 0<norm<math.inf or any(p.grad is not None for p in actor.parameters() if not p.requires_grad):raise ValueError('Gradient check failed')
        checked.append({**row,'trajectory_total_free_tokens':totals[row['trajectory_key']],
            'differentiable_mean_free_logprob_error':float(error[free].mean()),'differentiable_max_logprob_error':float(error.max()),
            'gradient_norm':norm,'loss':float(value.detach())})
        actor.zero_grad(set_to_none=True)
        del current,counts,error,terms,value,grads,reference,behavior,free;torch.cuda.empty_cache()
    after=parameter_digest(actor)
    if before!=after:raise ValueError('Zero-update check changed parameters')
    b=plan['model_binding']
    if binding(Path(b['base_path']),Path(b['checkpoint_path']))!=b:raise ValueError('Original SFT changed')
    write(out/'summary.json',{'status':'PASS','rows':checked,'optimizer_steps':0,'optimizer_created':False,
        'parameter_sha256_before':before,'parameter_sha256_after':after,'original_checkpoint_unchanged':True,
        'selected_trajectories':plan['selected_trajectories'],'selected_model_requests':len(checked),
        'tool_action_requests':sum(r['turn']==0 and r['actual_tool_calls']>0 for r in checked),
        'elapsed_seconds':time.time()-start,'max_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})
    progress(out,'PASS',optimizer_steps=0,elapsed_seconds=time.time()-start)


def one_update(root,plan):
    checked=read(root/'zero_update_check/summary.json')
    if checked['status']!='PASS' or len(checked['rows'])!=len(plan['rows']):raise ValueError('Complete zero-update check required')
    out=root/'one_update';out.mkdir(exist_ok=False);start=time.time();steps=0
    try:
        progress(out,'LOADING_ORIGINAL_SFT',optimizer_steps=0)
        processor,actor=load_actor(plan);before=parameter_digest(actor)
        if before!=checked['parameter_sha256_before']:raise ValueError('Initial actor differs from checked SFT')
        params=[p for p in actor.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=plan['learning_rate'],weight_decay=0.)
        optimizer.zero_grad(set_to_none=True);pre=0.
        for index,row in enumerate(checked['rows']):
            progress(out,'ACCUMULATING_ONE_UPDATE',completed=index,total=len(checked['rows']),group=row['group'],sample=row['sample'],turn=row['turn'],optimizer_steps=0)
            path,_,_,_=bound_row(row,plan)
            if digest(row['reference_path'])!=row['reference_sha256']:raise ValueError('Frozen reference changed')
            with np.load(row['reference_path']) as arr:
                reference=torch.from_numpy(arr['reference']).to('cuda');behavior=torch.from_numpy(arr['behavior']).to('cuda');free=torch.from_numpy(arr['free']).to('cuda')
            current,counts,_=score_saved_request(actor,processor,path);error=(current.detach()-behavior).abs()
            if float(error[free].mean())>.05 or float(error.max())>.2:raise ValueError('Pre-update behavior drift')
            terms,_,_=loss_terms(current,behavior,reference,row['advantage'],plan['clip_epsilon'],plan['kl_beta'])
            value=terms[free].sum()/row['trajectory_total_free_tokens']/plan['selected_trajectories']
            value.backward();pre+=float(value.detach())
            del reference,behavior,free,current,counts,error,terms,value;torch.cuda.empty_cache()
        grads=[p.grad for p in params if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):raise ValueError('Nonfinite accumulated gradient')
        if any(p.grad is not None for p in actor.parameters() if not p.requires_grad):raise ValueError('Base gradient leak')
        norm=float(torch.nn.utils.clip_grad_norm_(params,plan['gradient_clip_norm']))
        if not 0<norm<math.inf:raise ValueError('Invalid gradient norm')
        optimizer.step();steps=1;optimizer.zero_grad(set_to_none=True)
        progress(out,'POST_UPDATE_REPLAY',optimizer_steps=1)
        after=parameter_digest(actor)
        if before==after:raise ValueError('No trainable tensor changed')
        adapter=out/'adapter_after_one_update';actor.save_pretrained(adapter)
        torch.save(optimizer.state_dict(),out/'optimizer_state.pt')
        post=0.;diagnostics=[];kl_sum=0.;free_total=0
        for row in checked['rows']:
            with np.load(row['reference_path']) as arr:
                reference=torch.from_numpy(arr['reference']).to('cuda');behavior=torch.from_numpy(arr['behavior']).to('cuda');free=torch.from_numpy(arr['free']).to('cuda')
            with torch.inference_mode():
                current,_,_=score_saved_request(actor,processor,row['item_path'])
                terms,ratio,kl=loss_terms(current,behavior,reference,row['advantage'],plan['clip_epsilon'],plan['kl_beta'])
            if not bool(torch.isfinite(terms[free]).all()):raise ValueError('Nonfinite post-update objective')
            post+=float(terms[free].sum())/row['trajectory_total_free_tokens']/plan['selected_trajectories']
            kl_sum+=float(kl[free].sum());free_total+=int(free.sum())
            diagnostics.append({'group':row['group'],'sample':row['sample'],'turn':row['turn'],
                'advantage':row['advantage'],'reward':row['reward'],
                'mean_free_logprob_delta':float((current-reference)[free].mean()),
                'ratio_min':float(ratio[free].min()),'ratio_max':float(ratio[free].max())})
            del reference,behavior,free,current,terms,ratio,kl;torch.cuda.empty_cache()
        b=plan['model_binding']
        if binding(Path(b['base_path']),Path(b['checkpoint_path']))!=b:raise ValueError('Original SFT modified')
        saved=torch.load(out/'optimizer_state.pt',map_location='cpu',weights_only=True)
        if not saved['state'] or any(int(v['step'])!=1 for v in saved['state'].values()):raise ValueError('Saved optimizer is not exactly one update')
        write(out/'summary.json',{'status':'ONE_TEACHER_CATEGORY_GRPO_UPDATE_COMPLETED_NOT_A_SELECTED_MODEL',
            'optimizer_steps':steps,'learning_rate':plan['learning_rate'],'selected_groups':plan['selected_groups'],
            'source_train_videos':len(plan['selected_groups']),'sampled_trajectories':plan['selected_trajectories'],
            'actual_model_requests':len(checked['rows']),'pre_surrogate_loss':pre,'post_surrogate_loss':post,
            'fixed_sample_surrogate_improved':post<pre,'mean_reference_kl':kl_sum/free_total,
            'gradient_norm_before_clip':norm,'parameter_sha256_before':before,'parameter_sha256_after':after,
            'original_checkpoint_unchanged':True,'base_parameters_frozen':True,'qwen_semantic_rewards_used':False,
            'reward_only_final_teacher_category_match':True,'teacher_is_weak_reference':True,
            'accuracy_improvement_verified':False,'semantic_improvement_verified':False,'fixed120_used':False,
            'adapter_path':str(adapter.resolve()),'request_diagnostics':diagnostics,
            'elapsed_seconds':time.time()-start,'max_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})
        progress(out,'COMPLETE',optimizer_steps=steps,elapsed_seconds=time.time()-start)
    except Exception as exc:
        write(out/'FAILURE.json',{'error':repr(exc),'optimizer_steps_executed':steps,'automatic_retry':False});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','check','update']);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.out.resolve()
    if args.phase=='prepare':prepare(root)
    else:
        plan=validate_plan(root)
        phase=root/('zero_update_check' if args.phase=='check' else 'one_update')
        try:
            if args.phase=='check':zero_check(root,plan)
            else:one_update(root,plan)
        except Exception as exc:
            if phase.exists() and not (phase/'FAILURE.json').exists():write(phase/'FAILURE.json',{'error':repr(exc),'optimizer_steps_executed':0,'automatic_retry':False})
            raise
