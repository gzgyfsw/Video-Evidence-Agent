#!/usr/bin/env python3
"""One bounded request-level GRPO update using actual HF real-tool samples.

Only the frozen weak source-frame citation proxy is optimized. This is an
engineering smoke test, not verified semantic-evidence learning or a new best
checkpoint. Original sampling probabilities remain the ratio denominator.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from check_hf_policy_alignment import binding, write, versions
from check_hf_policy_backward import parameter_digest
from hf_audited_frame_reward import PROTOCOL, reference, score
from hf_head_precision import configure_head
from hf_request_policy import read_bound_item, score_saved_request
from request_policy_replay import sha
from unified_student_contract import PUBLIC_CONTEXT_VERSION, public_context


def group_advantages(rewards):
    values=torch.tensor(rewards,dtype=torch.float64)
    if len(values)<2 or not bool(torch.isfinite(values).all()):raise ValueError('Need a finite sampled group')
    sd=values.std(unbiased=False)
    if float(sd)<1e-8:return None
    return ((values-values.mean())/(sd+1e-8)).tolist()


def loss_terms(current, behavior, frozen_reference, advantage, epsilon=.2, beta=.01):
    ratio=(current-behavior).exp()
    if not bool(torch.isfinite(ratio).all()):raise ValueError('Nonfinite policy ratio')
    gain=torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage)
    delta=frozen_reference-current
    kl=delta.exp()-delta-1
    return -gain+beta*kl,ratio,kl


def compute_autocast(compute_dtype):
    """Match the exact forward dtype used to collect each saved request."""
    if compute_dtype == 'bfloat16':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    if compute_dtype == 'float32':
        return nullcontext()
    raise ValueError('Unsupported saved sampling compute dtype')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rollouts',type=Path,required=True);ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reward-version',choices=['citation-v1','acquisition-citation-v2'],default='citation-v1')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False);started=time.time();updates=0
    try:
        if (args.rollouts/'QUARANTINED.json').exists():raise ValueError('Quarantined rollout cannot train')
        plan=json.loads((args.rollouts/'PLAN.json').read_text())
        sampling_contract=plan['sampling_contract']
        compute_dtype=sampling_contract['dtype']
        head_precision=sampling_contract.get('lm_head_compute','bfloat16')
        if compute_dtype not in {'bfloat16','float32'} or head_precision not in {'bfloat16','float32'}:
            raise ValueError('Unsupported saved sampling precision contract')
        replay=json.loads((args.rollouts/'train_mode_probability_replay_v2/summary.json').read_text())
        samples=json.loads((args.rollouts/'summary.json').read_text())
        if args.reward_version=='acquisition-citation-v2':
            from hf_audited_frame_reward_v2 import PROTOCOL as protocol,score as reward_score
            reward_file='FROZEN_WEAK_REWARD_v2.json'
        else:protocol,reward_score,reward_file=PROTOCOL,score,'FROZEN_WEAK_REWARD.json'
        frozen_path=args.rollouts/reward_file;frozen=json.loads(frozen_path.read_text())
        if replay['status']!='PASS' or len(replay['rows'])!=samples['model_requests']:
            raise ValueError('All real rollout requests must pass the unchanged alignment gate')
        if versions()!=samples['environment'] or binding(args.model,args.checkpoint)!=plan['model_binding']:
            raise ValueError('Sampling environment or initial weights differ')
        if frozen['protocol']!=protocol:raise ValueError('Reward protocol changed')
        metadata=json.loads((args.rollouts/'MANIFEST.json').read_text());groups=defaultdict(list)
        for row in frozen['rows']:
            result=row['result'];meta=next(m['source'] for m in metadata if m['source']['merged_id']==result['source_id'])
            ref=reference(meta)
            if ref!=row['reference'] or reward_score(result,ref)!=row['reward']:
                raise ValueError('Frozen weak reward no longer matches original references and pixels')
            if row['reward']['eligible']:groups[result['source_id']].append(row)
        trajectories=[]
        for source,rows in groups.items():
            adv=group_advantages([r['reward']['total'] for r in rows])
            if adv is None:continue
            for row,a in zip(rows,adv):trajectories.append({'source':source,'row':row,'advantage':a})
        if not trajectories:raise ValueError('No informative within-video group; no parameter update allowed')
        if len(trajectories)>4:raise ValueError('Smoke update bounded to at most4 actual rollouts')
        write(args.out/'PLAN.json',{'optimizer_steps':1,'learning_rate':1e-6,'clip_epsilon':.2,'beta':.01,
            'source_train_videos':len(groups),'sampled_trajectories':len(trajectories),
            'reward_protocol':protocol,'reward_file_sha256':sha(frozen_path),'model_binding':plan['model_binding'],
            'original_sampler_logprobs_used_as_denominator':True,'frozen_reference':'HF rescore before update, same initial Adapter',
            'request_histories_not_concatenated':True,'all_free_sampled_fields_credited':True,
            'policy_unchanged':True,'test_used':False,'semantic_evidence_success_not_verified':True})
        if torch.cuda.mem_get_info()[0]<40*1024**3:raise RuntimeError('Insufficient shared GPU headroom; do not interrupt peers')
        torch.set_num_threads(4)
        processor=AutoProcessor.from_pretrained(args.model,local_files_only=True,trust_remote_code=True)
        base=AutoModelForImageTextToText.from_pretrained(args.model,dtype=getattr(torch,compute_dtype),local_files_only=True,
            trust_remote_code=True,attn_implementation='sdpa',low_cpu_mem_usage=True)
        model=PeftModel.from_pretrained(base,args.checkpoint,is_trainable=True).to('cuda')
        model._hf_compute_dtype=compute_dtype
        configure_head(model,head_precision)
        for name,p in model.named_parameters():p.requires_grad_('lora_' in name)
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads();model.train();before=parameter_digest(model)
        if before!=replay['parameter_sha256_before']:
            raise ValueError('Actual initial policy tensors differ from the passed replay')
        pre=[];total_rollouts=len(trajectories)
        # Reference logits are computed separately. They are not copied over the
        # actual behavior logprobs, which always come from the saved sampler.
        for trajectory in trajectories:
            requests=[]
            for call in trajectory['row']['result']['requests']:
                path=Path(call['item_path']);item,request,response=read_bound_item(path)
                if (request.get('public_context_version')!=PUBLIC_CONTEXT_VERSION
                        or request['context']!=public_context(request['context'])):
                    raise ValueError('Request is missing the validated public-context boundary')
                text='\n'.join(b['text'] for b in request['content_order'] if b['type']=='text')
                if any(s in text for s in ['video_path','data/merged_video_safety_v1/','candidate_file']):
                    raise ValueError('Storage/source-label provenance leaked into the actual model prompt')
                if item['model_binding']!=plan['model_binding']:raise ValueError('Adapter switched within rollout')
                with torch.inference_mode(),compute_autocast(compute_dtype):
                    reference_lp,counts,_=score_saved_request(model,processor,path)
                behavior=torch.tensor(response['sampled_processed_logprobs'],device='cuda')
                free=counts>1;error=(reference_lp-behavior).abs()
                if (float(error[free].mean())>.05 or float(error.max())>.2):
                    raise ValueError('Current real request no longer passes frozen pre-update alignment limits')
                requests.append({'path':path,'reference':reference_lp.detach().cpu().clone(),
                    'behavior':behavior.cpu(),'free':free.cpu(),'free_tokens':int(free.sum()),
                    'pre_mean_free_logprob':float(reference_lp[free].mean())})
                del reference_lp,behavior,counts,free,error;torch.cuda.empty_cache()
            trajectory['requests']=requests;trajectory['free_tokens']=sum(r['free_tokens'] for r in requests)
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=1e-6,weight_decay=0.)
        optimizer.zero_grad(set_to_none=True);pre_loss=0.
        for trajectory in trajectories:
            for row in trajectory['requests']:
                with compute_autocast(compute_dtype):
                    current,_,_=score_saved_request(model,processor,row['path'])
                    free=row['free'].to('cuda')
                    terms,ratio,kl=loss_terms(current,row['behavior'].to('cuda'),row['reference'].to('cuda'),trajectory['advantage'])
                    contribution=terms[free].sum()/trajectory['free_tokens']/total_rollouts
                contribution.backward();pre_loss+=float(contribution.detach())
                pre.append({'request':str(row['path']),'advantage':trajectory['advantage'],
                    'free_tokens':row['free_tokens'],'ratio_min':float(ratio[free].min().detach()),
                    'ratio_max':float(ratio[free].max().detach()),'contribution':float(contribution.detach())})
                del current,terms,ratio,kl,contribution,free;torch.cuda.empty_cache()
        gradients=[p.grad for p in parameters if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):raise ValueError('Nonfinite/missing gradients; no update')
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
        if not 0<norm<float('inf'):raise ValueError('Invalid gradient norm; no update')
        optimizer.step();updates=1;optimizer.zero_grad(set_to_none=True)
        after=parameter_digest(model)
        if before==after:raise ValueError('Optimizer did not change policy parameters')
        post=[];post_loss=0.
        for trajectory in trajectories:
            delta=0.
            for row in trajectory['requests']:
                with torch.inference_mode(),compute_autocast(compute_dtype):
                    current,_,_=score_saved_request(model,processor,row['path'])
                    free=row['free'].to('cuda')
                    terms,ratio,kl=loss_terms(current,row['behavior'].to('cuda'),row['reference'].to('cuda'),trajectory['advantage'])
                post_loss+=float(terms[free].sum())/trajectory['free_tokens']/total_rollouts
                delta+=float((current-row['reference'].to('cuda'))[free].sum())/trajectory['free_tokens']
                post.append({'request':str(row['path']),'mean_free_KL_estimate':float(kl[free].mean()),
                    'ratio_min':float(ratio[free].min()),'ratio_max':float(ratio[free].max())})
                del current,free,terms,ratio,kl;torch.cuda.empty_cache()
            trajectory['post_mean_free_logprob_delta']=delta
        model.save_pretrained(args.out/'adapter_after_one_update')
        if binding(args.model,args.checkpoint)!=plan['model_binding']:raise ValueError('Original checkpoint overwritten')
        write(args.out/'summary.json',{'status':'ONE_UPDATE_COMPLETED_NOT_A_SELECTED_MODEL',
            'optimizer_steps':1,'learning_rate':1e-6,'source_train_videos':len(groups),
            'sampling_compute_dtype':compute_dtype,'sampling_head_precision':head_precision,
            'sampled_trajectories':total_rollouts,'actual_independent_requests':sum(len(t['requests']) for t in trajectories),
            'actual_completion_tokens':sum(len(read_bound_item(r['path'])[2]['completion_token_ids']) for t in trajectories for r in t['requests']),
            'credited_free_tokens':sum(t['free_tokens'] for t in trajectories),'pre_loss':pre_loss,'post_loss':post_loss,
            'fixed_sample_surrogate_improved':post_loss<pre_loss,'gradient_norm_before_clip':norm,
            'parameter_sha256_before':before,'parameter_sha256_after':after,
            'original_checkpoint_unchanged':True,'base_parameters_frozen':True,
            'per_trajectory':[{'source':t['source'],'sample':t['row']['result']['sample'],'reward':t['row']['reward']['total'],
                'advantage':t['advantage'],'mean_free_logprob_delta':t['post_mean_free_logprob_delta']} for t in trajectories],
            'pre_update_requests':pre,'post_update_requests':post,'environment':versions(),
            'weak_proxy_only':True,'semantic_evidence_improvement_verified':False,'evaluation_test_used':False,
            'checkpoint_selected':False,'elapsed_seconds':time.time()-started})
    except Exception as exc:
        write(args.out/'FAILURE.json',{'error':repr(exc),'automatic_retry':False,'test_used':False,
                                      'optimizer_steps_executed':updates,'original_checkpoint_not_written':True})
        raise


if __name__=='__main__':main()
