"""Bounded single-pass clipped GRPO over the frozen v4 real-tool samples.

Default is CPU-only preparation. --execute is required for at most8 updates.
This is one fixed behavior-policy batch, never relabelled as fresh on-policy
data after an update. No Teacher-time matching, synthetic tool output or Test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from check_hf_policy_alignment import binding, versions, write
from check_hf_policy_backward import parameter_digest
from hf_request_policy import read_bound_item, score_saved_request
from hf_request_rng import validate_seeds,SCHEDULE
from hf_group_storage import group_folder, numerical_folder, report_folder
from request_policy_replay import sha
from score_hf_semantic_group import group_signal
from train_hf_request_grpo_smoke import group_advantages, loss_terms
from unified_student_contract import POLICY_SHA, PUBLIC_CONTEXT_VERSION, public_context


def load(path):return json.loads(Path(path).read_text())


def semantic_group_advantages(rewards,standard_deviation_floor=.5):
    """Keep small true cost differences small relative to evidence gains.

    Without a floor, [1,1,1,.99] gives essentially the same unit advantages
    as [1,1,1,.5], erasing the intended small cost coefficient across groups.
    No artificial reward spread: identical groups still return no signal.
    """
    if not standard_deviation_floor>0:raise ValueError('Need a positive fixed advantage scale floor')
    ordinary=group_advantages(rewards)
    if ordinary is None:return None
    values=torch.tensor(rewards,dtype=torch.float64)
    return ((values-values.mean())/max(float(values.std(unbiased=False)),standard_deviation_floor)).tolist()


def prepare(trial, model_path, checkpoint):
    """Require all128 samples; retain all groups, update only informative ones."""
    plan=load(trial/'PLAN.json');manifest=load(trial/'MANIFEST128.json')
    initial=binding(model_path,checkpoint)
    if len(manifest)!=128 or plan['model_binding']!=initial:
        raise ValueError('Need the frozen128 v4 samples and unchanged initializer')
    if plan.get('request_seed_schedule')!=SCHEDULE:raise ValueError('Independent request RNG schedule not certified')
    groups=[];receipts={};parameter_sha=None;all_seeds=set()
    if (trial/'GROUP_OUTPUT_OVERRIDES.json').exists():
        receipts[str(trial/'GROUP_OUTPUT_OVERRIDES.json')]=sha(trial/'GROUP_OUTPUT_OVERRIDES.json')
    if (trial/'NUMERICAL_OUTPUT_OVERRIDES.json').exists():
        receipts[str(trial/'NUMERICAL_OUTPUT_OVERRIDES.json')]=sha(trial/'NUMERICAL_OUTPUT_OVERRIDES.json')
    if (trial/'REPORT_OUTPUT_OVERRIDES.json').exists():
        receipts[str(trial/'REPORT_OUTPUT_OVERRIDES.json')]=sha(trial/'REPORT_OUTPUT_OVERRIDES.json')
    for index in range(32):
        folder=group_folder(trial,index)
        if any((folder/name).exists() for name in ['FAILURE.json','QUARANTINED.json']):
            raise ValueError('Technical failure or quarantined group; no update')
        collection=load(folder/'summary.json');sampling_plan=load(folder/'PLAN.json')
        seeds=validate_seeds(collection['results'],manifest[index*4:index*4+4],read_bound_item)
        if all_seeds.intersection(seeds):raise ValueError('Cross-video RNG seed collision')
        all_seeds.update(seeds)
        if (sampling_plan['policy_sha256']!=POLICY_SHA
                or sampling_plan['train_manifest_sha256']!=plan['train_manifest_sha256']
                or load(folder/'MANIFEST.json')!=manifest[index*4:index*4+4]
                or sampling_plan['budget']!={'max_tool_calls':3,'max_frames':12}
                or sampling_plan['max_search_new_frames']!=4 or sampling_plan['initial_frames']!=4
                or sampling_plan['max_decisions']!=4 or sampling_plan['test_used'] is not False):
            raise ValueError('Frozen source, policy or execution budget differs')
        gate_dir=numerical_folder(folder);gate=load(gate_dir/'summary.json')
        reward_path=trial/(folder.name+'_rewards')/'FROZEN_SEMANTIC_REWARDS.json'
        reward=load(reward_path)
        if (gate['status']!='PASS' or gate['model_binding']!=initial or gate['optimizer_steps']!=0
                or not gate['same_trainable_parameter_digest_before_after']
                or gate['parameter_sha256_before']!=gate['parameter_sha256_after']
                or sampling_plan['model_binding']!=initial or versions()!=collection['environment']):
            raise ValueError('Actual zero-update gate/initializer/environment failed')
        if parameter_sha is None:parameter_sha=gate['parameter_sha256_before']
        if parameter_sha!=gate['parameter_sha256_before']:raise ValueError('Adapter tensors differ between groups')
        contract=sampling_plan['sampling_contract']
        if contract['dtype']!='float32' or contract.get('lm_head_compute')!='float32':
            raise ValueError('Old numeric diagnostic distribution cannot train this revision')
        if (reward['version']!='semantic-evidence-acquisition-report-v1'
                or reward['rollout_summary_sha256']!=sha(folder/'summary.json')
                or 'Actual terminal evidence/claims/stop' not in reward.get('report_reward_scope','')):
            raise ValueError('Reward belongs to a different trace/protocol')
        for part,key in [('report_review','report_summary_sha256'),('acquisition','acquisition_summary_sha256')]:
            path=(report_folder(folder) if part=='report_review' else trial/(folder.name+'_'+part))/'summary.json'
            if sha(path)!=reward[key]:raise ValueError('Review summary changed after reward freeze')
            receipts[str(path)]=sha(path)
        for path in [folder/'PLAN.json',folder/'summary.json',gate_dir/'summary.json',reward_path]:
            receipts[str(path)]=sha(path)
        expected={(r['source']['merged_id'],r['sample']) for r in manifest[index*4:index*4+4]}
        actual={(r['source_id'],r['sample']) for r in collection['results']}
        if actual!=expected or len(collection['results'])!=4 or len(reward['rows'])!=4:
            raise ValueError('Missing/duplicate sampled source or seed')
        if {(r['source_id'],r['sample']) for r in reward['rows']}!=expected:
            raise ValueError('Rewards duplicate or omit an independent sample')
        if len(gate['rows'])!=len(collection['rows']) or len(collection['rows'])!=collection['model_requests']:
            raise ValueError('A real sampled request lacks its numeric/gradient check')
        requests={}
        for i,row in enumerate(collection['rows']):
            path=Path(row['item_path']);item,request,response=read_bound_item(path)
            if (item['model_binding']!=initial or item['sampling_contract']!=contract
                    or request.get('public_context_version')!=PUBLIC_CONTEXT_VERSION
                    or request['context']!=public_context(request['context'])):
                raise ValueError('Adapter/behavior/public-context boundary changed')
            numerical_path=gate_dir/f'{i:02d}_gradient.json';numerical=load(numerical_path)
            pts=numerical['per_token']
            if (not numerical['train_mode_differentiable_scores_aligned']
                    or not numerical['finite_nonzero_lora_gradients'] or numerical['base_parameter_gradients']!=0
                    or [t['id'] for t in pts]!=response['completion_token_ids']
                    or [t['behavior'] for t in pts]!=response['sampled_processed_logprobs']):
                raise ValueError('Reference numeric tokens do not match actual behavior')
            receipts[str(numerical_path)]=sha(numerical_path);receipts[str(path)]=sha(path)
            requests[str(path)]={'path':path,'reference':torch.tensor([t['differentiable_score'] for t in pts]),
                'behavior':torch.tensor(response['sampled_processed_logprobs']),
                'free_tokens':numerical['free_tokens']}
        rows=sorted(reward['rows'],key=lambda r:r['sample'])
        for row in rows:
            if row['result']!=next(r for r in collection['results'] if r['sample']==row['sample']):
                raise ValueError('Reward substituted a different actor result')
            if row['semantic_score']!=.5*row['acquisition_supported']+.5*(row['acquisition_supported'] and row['report_supported']):
                raise ValueError('Reward component mismatch')
            if abs(row['total']-(row['semantic_score']-row['tool_cost']-row['frame_cost']+row['confirmed_fabrication_penalty']))>1e-9:
                raise ValueError('Frozen reward arithmetic differs')
        signal=group_signal(rows);advantages=semantic_group_advantages([r['total'] for r in rows])
        trajectories=[]
        if signal['eligible_for_update']:
            if advantages is None:raise ValueError('Artificial signal in an equal-score group')
            for row,advantage in zip(rows,advantages):
                calls=[requests[c['item_path']] for c in row['result']['requests']]
                trajectories.append({'source_id':row['source_id'],'sample':row['sample'],'reward':row['total'],
                    'advantage':advantage,'requests':calls,'free_tokens':sum(c['free_tokens'] for c in calls)})
        groups.append({'group':index,'source_id':rows[0]['source_id'],'signal':signal,'trajectories':trajectories})
    if len({g['source_id'] for g in groups})!=32:raise ValueError('Duplicate video group')
    # Stable ordering, unrelated to score. One pass, up to4 video groups/update.
    eligible=sorted([g for g in groups if g['trajectories']],key=lambda g:hashlib.sha256(g['source_id'].encode()).hexdigest())
    batches=[eligible[i:i+4] for i in range(0,len(eligible),4)]
    if len(batches)>8:raise ValueError('Trial exceeds the authorized update bound')
    return {'groups':groups,'batches':batches,'receipts':receipts,'parameter_sha':parameter_sha,'binding':initial}


def evaluate_batch(model,processor,trajectories):
    total=0.;kl_sum=0.;free_count=0;metrics=[]
    for t in trajectories:
        for r in t['requests']:
            with torch.no_grad():
                current,counts,_=score_saved_request(model,processor,r['path'])
                free=counts>1
                terms,ratio,kl=loss_terms(current,r['behavior'].to('cuda'),r['reference'].to('cuda'),t['advantage'])
            total+=float(terms[free].sum())/t['free_tokens']/len(trajectories)
            kl_sum+=float(kl[free].sum());free_count+=int(free.sum())
            metrics.append({'path':str(r['path']),'free_tokens':int(free.sum()),
                'ratio_min':float(ratio[free].min()),'ratio_max':float(ratio[free].max()),
                'mean_reference_KL_estimate':float(kl[free].mean())})
            del current,counts,terms,ratio,kl,free;torch.cuda.empty_cache()
    return {'fixed_sample_surrogate':total,'mean_reference_KL_estimate':kl_sum/free_count,'requests':metrics}


def save_committed_update(output,step,model,optimizer,parameter_sha):
    """Persist an executed update before post-checks; this is not a gate pass."""
    folder=Path(output)/f'update-{step:02d}';folder.mkdir()
    write(folder/'COMMITTED.json',{'optimizer_steps':step,'parameter_sha256':parameter_sha,
        'post_update_gate':'PENDING','candidate_selected':False})
    model.save_pretrained(folder/'adapter')
    torch.save(optimizer.state_dict(),folder/'optimizer.pt')
    return folder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ['trial','model','checkpoint','out']:p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--execute',action='store_true',help='Run bounded updates only after ALL zero-update gates pass')
    a=p.parse_args();a.trial=a.trial.resolve();updates=0;started=time.time()
    prepared=prepare(a.trial,a.model,a.checkpoint)
    protocol_path=a.trial/'OPTIMIZER_PROTOCOL_v2.json';protocol=load(protocol_path)
    if (protocol['revision']!=2 or protocol['advantage_standard_deviation_floor']!=.5
            or protocol['lr']!=1e-6 or protocol['clip']!=.2 or protocol['kl_beta']!=.01
            or protocol['max_updates']!=8 or protocol['groups_per_minibatch']!=4
            or protocol['epochs']!=1 or protocol['post_step_max_mean_reference_KL']!=.02):
        raise ValueError('Frozen optimizer protocol differs from actual implementation')
    prepared['receipts'][str(protocol_path)]=sha(protocol_path)
    a.out.mkdir(parents=True,exist_ok=False)
    with (a.out/'trainer_snapshot.py').open('xb') as snapshot:snapshot.write(Path(__file__).read_bytes())
    contract={'version':'semantic-request-GRPO-fixed-behavior-batch-v2','learning_rate':1e-6,
        'optimizer_protocol_sha256':sha(protocol_path),'trainer_source_sha256':sha(Path(__file__)),
        'clip_epsilon':.2,'reference_KL_coefficient':.01,'gradient_clip_norm':1.,'weight_decay':0.,
        'maximum_optimizer_steps':8,'planned_optimizer_steps':len(prepared['batches']),
        'video_groups_per_minibatch':4,'epochs_over_frozen_behavior_batch':1,
        'advantage_normalization':'group centered, standard deviation floor0.5; raw rewards unchanged',
        'advantage_standard_deviation_floor':.5,
        'independent_request_rng_schedule':SCHEDULE,
        'actual_behavior':'v4 full-FP32 real-tool independent samples; unchanged original denominator',
        'reference':'zero-update full-causal FP32 v4 scores, original actual request tokens',
        'first_and_later_step_gate':'finite gradients/parameters; changed LoRA; surrogate decrease; mean reference KL<=.02',
        'post_step_gate_uses_same_minibatch_not_generalization':True,
        'test_used':False,'model_binding':prepared['binding'],
        'group_signal':[{k:g[k] for k in ['group','source_id','signal']} for g in prepared['groups']],
        'frozen_dependency_sha256':prepared['receipts'],'original_checkpoint_not_written':True,
        'not_fresh_on_policy_after_first_update':True,'automatic_extra_training':False}
    write(a.out/'PLAN.json',contract)
    write(a.out/'ZERO_UPDATE_GATE_RECEIPT.json',{'status':'PASS','optimizer_steps':0,
        'videos':32,'independent_samples':128,'numeric_gradients_all_passed':True,
        'independent_request_rng_schedule':SCHEDULE,'actual_request_seed_uniqueness_checked':True,
        'semantic_reviews_complete_weak_reference':True,'informative_groups':sum(bool(g['trajectories']) for g in prepared['groups']),
        'parameter_sha256':prepared['parameter_sha'],'dependency_sha256':prepared['receipts']})
    if not a.execute:
        write(a.out/'PREPARED_ONLY.json',{'optimizer_steps':0,'planned_steps':len(prepared['batches'])});return
    try:
        if not prepared['batches']:
            write(a.out/'summary.json',{'status':'NO_LEARNING_SIGNAL_NO_UPDATE','optimizer_steps':0});return
        if torch.cuda.mem_get_info()[0]<40*1024**3:raise RuntimeError('Insufficient GPU headroom; do not interrupt peers')
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText,AutoProcessor
        from hf_head_precision import configure_head
        torch.set_num_threads(4)
        print(json.dumps({'stage':'LOADING_FROZEN_V4','optimizer_steps':0}),flush=True)
        processor=AutoProcessor.from_pretrained(a.model,local_files_only=True,trust_remote_code=True)
        base=AutoModelForImageTextToText.from_pretrained(a.model,dtype=torch.float32,local_files_only=True,
            trust_remote_code=True,attn_implementation='sdpa',low_cpu_mem_usage=True)
        model=PeftModel.from_pretrained(base,a.checkpoint,is_trainable=True).to('cuda')
        model._hf_compute_dtype='float32';configure_head(model,'float32')
        for name,param in model.named_parameters():param.requires_grad_('lora_' in name)
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads();model.train()
        if parameter_digest(model)!=prepared['parameter_sha']:raise ValueError('Actual initializer tensors differ from zero-update gate')
        params=[v for v in model.parameters() if v.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-6,weight_decay=0.)
        for groups in prepared['batches']:
            # Revalidate all frozen source records before each parameter update.
            if any(sha(path)!=expected for path,expected in prepared['receipts'].items()):
                raise ValueError('A frozen dependency changed before update')
            trajectories=[t for g in groups for t in g['trajectories']]
            request_total=sum(len(t['requests']) for t in trajectories)
            print(json.dumps({'stage':'PRE_UPDATE_EVALUATION','optimizer_steps':updates,
                'groups':[g['group'] for g in groups],'trajectories':len(trajectories),
                'requests':request_total}),flush=True)
            before=evaluate_batch(model,processor,trajectories)
            if updates==0:
                rechecked=0
                for t in trajectories:
                    for r in t['requests']:
                        with torch.no_grad():current,counts,_=score_saved_request(model,processor,r['path'])
                        error=(current-r['behavior'].to('cuda')).abs();free=counts>1
                        if float(error[free].mean())>.05 or float(error.max())>.2:
                            raise ValueError('First-step actual policy no longer passes unchanged numeric limits')
                        rechecked+=1
                        print(json.dumps({'stage':'INITIAL_POLICY_REQUEST_RECHECKED','optimizer_steps':0,
                            'completed_requests':rechecked,'total_requests':request_total}),flush=True)
                        del current,counts,error,free;torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            backward_requests=0
            for t in trajectories:
                for r in t['requests']:
                    current,counts,_=score_saved_request(model,processor,r['path'])
                    free=counts>1
                    if int(free.sum())!=r['free_tokens']:raise ValueError('Free token mask changed')
                    terms,_,_=loss_terms(current,r['behavior'].to('cuda'),r['reference'].to('cuda'),t['advantage'])
                    contribution=terms[free].sum()/t['free_tokens']/len(trajectories)
                    contribution.backward()
                    backward_requests+=1
                    print(json.dumps({'stage':'REWARD_GRADIENT_REQUEST_DONE','optimizer_steps':updates,
                        'completed_requests':backward_requests,'total_requests':request_total}),flush=True)
                    del current,counts,free,terms,contribution;torch.cuda.empty_cache()
            grads=[v.grad for v in params if v.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):raise ValueError('Nonfinite/missing gradient')
            if any(v.grad is not None for v in model.parameters() if not v.requires_grad):raise ValueError('Unexpected frozen-base gradient')
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
            if not 0<norm<float('inf'):raise ValueError('Invalid gradient norm')
            old_digest=parameter_digest(model);optimizer.step();updates+=1;optimizer.zero_grad(set_to_none=True)
            after_digest=parameter_digest(model)
            if old_digest==after_digest or not all(bool(torch.isfinite(v).all()) for v in params):
                raise ValueError('Nonfinite or unchanged parameters after update')
            folder=save_committed_update(a.out,updates,model,optimizer,after_digest)
            print(json.dumps({'stage':'POST_UPDATE_EVALUATION','optimizer_steps':updates,
                'gate_passed':'PENDING'}),flush=True)
            after=evaluate_batch(model,processor,trajectories)
            passed=(after['fixed_sample_surrogate']<before['fixed_sample_surrogate']
                and after['mean_reference_KL_estimate']<=.02)
            write(folder/'receipt.json',{'optimizer_steps':updates,'gate_passed':passed,
                'groups':[g['group'] for g in groups],'before':before,'after':after,
                'gradient_norm_before_clip':norm,'parameter_sha256_after':after_digest,
                'original_sampler_denominator_preserved':True,'same_batch_improvement_not_external_effect':True})
            print(json.dumps({'step':updates,'gate_passed':passed,'before':before['fixed_sample_surrogate'],
                'after':after['fixed_sample_surrogate'],'mean_reference_KL_estimate':after['mean_reference_KL_estimate']}),flush=True)
            if not passed:
                write(folder/'QUARANTINED.json',{'reason':'Post-update frozen gate failed; no automatic extra update'})
                raise RuntimeError('First/later step gate failed; stop and retain analysis')
        if binding(a.model,a.checkpoint)!=prepared['binding']:raise ValueError('Original checkpoint changed')
        write(a.out/'summary.json',{'status':'BOUNDED_UPDATES_COMPLETE_PENDING_DEVELOPMENT_EVALUATION',
            'optimizer_steps':updates,'test_used':False,'elapsed_seconds':time.time()-started,
            'original_checkpoint_unchanged':True,'semantic_reward_is_weak_reference':True,
            'paper_effect_not_established':True,'extra_training_started':False})
    except Exception as exc:
        pending=a.out/f'update-{updates:02d}'
        receipt=pending/'receipt.json'
        if updates and pending.exists() and not (receipt.exists() and load(receipt).get('gate_passed')):
            if not (pending/'QUARANTINED.json').exists():
                write(pending/'QUARANTINED.json',{'reason':repr(exc),'post_update_gate_not_passed':True})
        write(a.out/'FAILURE.json',{'error':repr(exc),'optimizer_steps_executed':updates,
            'automatic_retry':False,'original_checkpoint_not_written':True});raise


if __name__=='__main__':main()
