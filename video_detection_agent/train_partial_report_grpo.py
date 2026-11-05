"""Bounded partial-comparison report-policy pilot: prepare, zero-update check, one update.

The original four-attempt group is retained. Only factual description tokens in
the last ANSWER receive credit; excluded attempts receive neither policy nor KL
loss. Stored masked-temperature sampler logprobs remain the PPO denominator.
This is an exploratory partial-comparison objective, not vanilla full-group GRPO
or proof that local-Qwen factual judgments are true.
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from check_hf_policy_alignment import binding, fingerprint, versions, write
from check_hf_policy_backward import parameter_digest
from hf_request_policy import read_bound_item, prompt_batch, score_saved_request
from prepare_grpo_group_selection import ROOT
from request_policy_replay import sha


def load(path):
    return json.loads(Path(path).read_text())


def progress(out, stage, **kwargs):
    row={'stage':stage,'unix':time.time(),**kwargs}
    tmp=out/'progress.tmp';tmp.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(out/'progress.json');print(json.dumps(row,ensure_ascii=False),flush=True)


def gpu():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True)
    used,free,util=map(int,raw.strip().split(','))
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                      '--format=csv,noheader,nounits'],text=True).strip()
    return {'used_mib':used,'free_mib':free,'utilization_percent':util,'compute_processes':processes}


def description_ranges(text):
    """Return exact UTF-8 byte intervals inside JSON evidence-description strings.

    A lexical pass respects escaped quotes and key-like text inside values.
    Reconstructed values must exactly match the parsed evidence array.
    """
    parsed=json.loads(text)
    if parsed.get('action')!='ANSWER':raise ValueError('Only completed ANSWERs may be credited')
    strings=[];i=0
    while i<len(text):
        if text[i]!='"':i+=1;continue
        start=i;i+=1
        while i<len(text):
            if text[i]=='\\':i+=2;continue
            if text[i]=='"':break
            i+=1
        if i>=len(text):raise ValueError('Unterminated JSON string')
        end=i+1;strings.append((start,end,json.loads(text[start:end])));i=end
    result=[];decoded=[]
    for j,(start,end,value) in enumerate(strings[:-1]):
        k=end
        while k<len(text) and text[k].isspace():k+=1
        if value=='description' and k<len(text) and text[k]==':':
            a,b,description=strings[j+1]
            if text[k+1:a].strip():raise ValueError('Unexpected non-string description')
            result.append((len(text[:a+1].encode()),len(text[:b-1].encode())))
            decoded.append(description)
    if decoded!=[e['description'] for e in parsed['evidence']] or not result:
        raise ValueError('Description field coverage mismatch')
    return result


def byte_inverse():
    values=list(range(ord('!'),ord('~')+1))+list(range(0xa1,0xac+1))+list(range(0xae,0xff+1))
    chars=values[:];n=0
    for b in range(256):
        if b not in values:values.append(b);chars.append(256+n);n+=1
    return {chr(c):b for b,c in zip(values,chars)}


def description_token_mask(tokenizer, response, credited_claim_indices=None):
    """Require byte-exact actual-token decoding; never re-tokenize sampled text."""
    inverse=byte_inverse();special=set(tokenizer.all_special_ids);pieces=[]
    for token in response['completion_token_ids']:
        if token in special:pieces.append(b'');continue
        raw=tokenizer.convert_ids_to_tokens(token)
        try:pieces.append(bytes(inverse[c] for c in raw))
        except KeyError as exc:raise ValueError('Unsupported non-ByteLevel token') from exc
    raw=b''.join(pieces)
    text=raw.decode('utf-8')
    if text!=response['decoded_text_for_inspection_only']:
        raise ValueError('Actual token bytes do not reconstruct recorded output')
    ranges=description_ranges(text)
    if credited_claim_indices is not None:
        if any(not 0<=i<len(ranges) for i in credited_claim_indices):raise ValueError('Unknown credited claim index')
        ranges=[ranges[i] for i in credited_claim_indices]
    mask=[];offset=0
    for piece in pieces:
        end=offset+len(piece)
        # Boundary-spanning tokens are excluded rather than crediting a quote,
        # category, citation timestamp or structural token accidentally.
        mask.append(bool(piece) and any(offset>=a and end<=b for a,b in ranges))
        offset=end
    if ranges and not any(mask):raise ValueError('No whole description-content tokens')
    return torch.tensor(mask,dtype=torch.bool),ranges


def masked_loss(current, behavior, reference, advantage, selected, active, epsilon=.2, beta=.01):
    """An excluded attempt returns a graph-connected zero, including zero KL."""
    if not active:return current.sum()*0.0
    if not bool(selected.any()):raise ValueError('Active attempt lacks credited content tokens')
    ratio=(current[selected]-behavior[selected]).exp()
    if not bool(torch.isfinite(ratio).all()):raise ValueError('Nonfinite policy ratio')
    gain=torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage)
    delta=reference[selected]-current[selected]
    kl=torch.expm1(delta)-delta
    return (-gain+beta*kl).mean()


def prepare(args):
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
    source=args.source.resolve();candidates=[json.loads(l) for l in (source/'CANDIDATES.jsonl').read_text().splitlines()]
    if len(candidates)!=5:raise ValueError('Pilot frozen to the five audited partial groups')
    rows=[];expected=None;contract=None
    from review_grpo_selection_qwen import trace
    parent=Path(load(source/'PLAN.json')['source'])
    facts=load(parent/'FACT_RESULTS.json');conflicts=load(source/'CROSS_CONTEXT_CONFLICTS.json')
    selections={s['group']:s for s in load(parent/'SELECTION.json')}
    for group in candidates:
        if len(group['trajectories'])!=4:raise ValueError('Four original attempts required')
        net=[0]*4
        for edge in group['retained_comparisons']:
            net[edge['winner_sample']]+=1;net[edge['loser_sample']]-=1
        proxy=[.5+n/6 for n in net];sd=(sum((r-.5)**2 for r in proxy)/4)**.5
        advantages=[(r-.5)/sd for r in proxy]
        if any(abs(a-b)>1e-8 for a,b in zip(advantages,group['proposed_advantages'])):
            raise ValueError('Frozen advantages do not match verified comparisons')
        selection=selections[group['group']]
        summary=load(Path(selection['rollout_dir'])/'summary.json')
        for t in group['trajectories']:
            s=t['sample'];result=next(r for r in summary['results'] if r['sample']==s)
            verified=trace(selection,result)
            path=Path(result['requests'][-1]['item_path']);item,request,response=read_bound_item(path)
            if json.loads(response['decoded_text_for_inspection_only']).get('action')!='ANSWER':
                raise ValueError('Last request is not an actual ANSWER')
            expected=expected or item['model_binding'];contract=contract or item['sampling_contract']
            if item['model_binding']!=expected or item['sampling_contract']!=contract:
                raise ValueError('All five groups must use one original SFT behavior policy')
            active=group['proposed_loss_mask'][s]
            if active!=(net[s]!=0):raise ValueError('Attempt exclusion mask changed')
            judgments=[('UNKNOWN' if f"{group['group']}/{s}/{i}" in conflicts else
                facts[f"{group['group']}/{s}/{i}"]['judgment']) for i in range(len(verified['final_report']['evidence']))]
            wanted='SUPPORTED' if advantages[s]>0 else 'CONTRADICTED'
            credited=[i for i,j in enumerate(judgments) if active and j==wanted]
            if active and not credited:raise ValueError('Active relative preference lacks corresponding verified statements')
            rows.append({'group':group['group'],'sample':s,'source_id':group['source_id'],
                'item_path':str(path),'item_sha256':sha(path),'advantage':advantages[s],'active':active,
                'claim_judgments':judgments,'credited_claim_indices':credited,
                'actual_tool_calls':len(verified['tools']),'request_count':len(verified['turns']),
                'original_request_paths':[r['item_path'] for r in result['requests']],
                'sampler_environment':summary['environment']})
    if len(rows)!=20 or sum(r['active'] for r in rows)!=12:raise ValueError('Expected 20 attempts / 12 active')
    plan={'protocol':'partial-report-grpo-one-update-v1','script_sha256':sha(__file__),
        'source':str(source),'candidate_file_sha256':sha(source/'CANDIDATES.jsonl'),
        'source_plan_sha256':sha(source/'PLAN.json'),'rows':rows,'model_binding':expected,
        'fact_results_path':str(parent/'FACT_RESULTS.json'),'fact_results_sha256':sha(parent/'FACT_RESULTS.json'),
        'cross_context_conflicts_sha256':sha(source/'CROSS_CONTEXT_CONFLICTS.json'),
        'sampling_contract':contract,'environment':versions(),'optimizer_steps_maximum':1,
        'learning_rate':1e-6,'clip_epsilon':.2,'kl_beta':.01,'gradient_clip_norm':1.,
        'normalization':'mean per selected description token, then sum / all 20 original attempts',
        'credit_scope':'whole non-boundary tokens inside final ANSWER evidence.description strings AND original grammar has >1 allowed token',
        'claim_credit_rule':'positive advantage: only repeat-supported statements; negative advantage: only repeat-contradicted statements; UNKNOWN statements receive no direct token loss',
        'excluded_attempt_policy_and_KL_loss':'exact zero; no exclusion-based renormalization',
        'behavior_denominator':'original sampler processed temperature=0.7 grammar-masked probabilities',
        'grammar_version_compatibility':'Replay never compiles grammar: original SHA-bound per-token masks are used verbatim. Only xgrammar 0.1.23/0.1.25 may differ; torch/transformers/peft must match exactly.',
        'reference_policy':'unchanged original SFT; frozen pre-update rescores, not substituted for behavior',
        'thresholds':{'mean_free_logprob_error':.05,'maximum_logprob_error':.2},
        'model_improvement_claimed':False,'fixed120_used':False,'codex_semantic_labels_used':False,
        'new_train_validation_gate':'after one-update engineering and paired pilot evaluation; no blind 100-video expansion',
        'frozen_unix':time.time()}
    write(root/'PLAN.json',plan)
    (root/'code_snapshot').mkdir();(root/'code_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    progress(root,'PREPARED',groups=5,attempts=20,active_attempts=12,optimizer_steps=0)


def load_actor(plan):
    from peft import PeftModel
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from hf_head_precision import configure_head
    snap=gpu()
    if snap['compute_processes'] or snap['free_mib']<43000:
        raise RuntimeError('GPU resource conflict; do not interrupt peers')
    b=plan['model_binding'];model_path=Path(b['base_path']);checkpoint=Path(b['checkpoint_path'])
    if binding(model_path,checkpoint)!=b:raise ValueError('Original weights/config changed')
    if versions()!=plan['environment']:raise ValueError('Pilot software environment changed')
    for row in plan['rows']:
        old=row['sampler_environment'];now=plan['environment']
        if set(old)!=set(now) or any(old[k]!=now[k] for k in now if k!='xgrammar'):
            raise ValueError('Original model-computation software differs')
        if old['xgrammar']!=now['xgrammar'] and not {old['xgrammar'],now['xgrammar']}<={'0.1.23','0.1.25'}:
            raise ValueError('Unreviewed grammar-library version difference')
        # score_saved_request reads original SHA-bound bitmasks; it does not
        # import/compile xgrammar. Never regenerate masks with the newer library.
    torch.set_num_threads(4)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,trust_remote_code=True)
    decoder=load(model_path/'tokenizer.json')['decoder']
    if decoder['type']!='ByteLevel':raise ValueError('Unsupported tokenizer byte decoder')
    dtype=plan['sampling_contract']['dtype']
    base=AutoModelForImageTextToText.from_pretrained(model_path,dtype=getattr(torch,dtype),
        local_files_only=True,trust_remote_code=True,attn_implementation='sdpa',low_cpu_mem_usage=True)
    actor=PeftModel.from_pretrained(base,checkpoint,is_trainable=True).to('cuda')
    actor._hf_compute_dtype=dtype;configure_head(actor,plan['sampling_contract']['lm_head_compute'])
    for name,p in actor.named_parameters():p.requires_grad_('lora_' in name)
    for module in actor.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    actor.enable_input_require_grads();actor.train()
    return processor,actor


def bound_row(row,plan):
    path=Path(row['item_path'])
    if sha(path)!=row['item_sha256']:raise ValueError('Original item changed')
    item,request,response=read_bound_item(path)
    if item['model_binding']!=plan['model_binding'] or item['sampling_contract']!=plan['sampling_contract']:
        raise ValueError('Per-request policy binding changed')
    for im in request['images']:
        if sha(im['path'])!=im['sha256']:raise ValueError('Actual seen image changed')
    return path,item,request,response


def check(root,out,plan):
    started=time.time();out.mkdir(parents=True,exist_ok=False)
    progress(out,'LOADING_ORIGINAL_ACTOR',optimizer_steps=0)
    processor,actor=load_actor(plan);before=parameter_digest(actor);results=[]
    for index,row in enumerate(plan['rows']):
        progress(out,'REPLAY_AND_BACKWARD',completed=index,total=len(plan['rows']),group=row['group'],sample=row['sample'],optimizer_steps=0)
        path,item,request,response=bound_row(row,plan)
        batch=prompt_batch(request,processor,'cuda')
        if fingerprint(batch)!=response['prompt_tensor_receipt']:raise ValueError('Actual sampling/replay tensors differ')
        del batch
        content_mask,ranges=description_token_mask(processor.tokenizer,response,row['credited_claim_indices'])
        actor.zero_grad(set_to_none=True)
        current,counts,ids=score_saved_request(actor,processor,path)
        behavior=torch.tensor(response['sampled_processed_logprobs'],device='cuda')
        free=counts>1;selected=content_mask.to('cuda') & free;reference=current.detach().clone()
        error=(current.detach()-behavior).abs();mean=float(error[free].mean());maximum=float(error.max())
        if mean>plan['thresholds']['mean_free_logprob_error'] or maximum>plan['thresholds']['maximum_logprob_error']:
            raise ValueError('Original behavior/replay alignment failed')
        value=masked_loss(current,behavior,reference,row['advantage'],selected,row['active'],
            plan['clip_epsilon'],plan['kl_beta'])/len(plan['rows'])
        value.backward()
        gradients=[p.grad for p in actor.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):raise ValueError('Missing/nonfinite gradient')
        norm=math.sqrt(sum(float(g.detach().float().square().sum()) for g in gradients))
        if row['active'] and not norm>0:raise ValueError('Active attempt has no gradient')
        if not row['active'] and (norm!=0 or float(value.detach())!=0):raise ValueError('Skipped attempt contributes loss/gradient')
        if any(p.grad is not None for p in actor.parameters() if not p.requires_grad):raise ValueError('Base parameters received gradients')
        name=f"group-{row['group']:02d}-sample-{row['sample']}"
        np.savez_compressed(out/(name+'.npz'),reference=reference.cpu().numpy(),
            behavior=behavior.cpu().numpy(),selected=selected.cpu().numpy(),free=free.cpu().numpy())
        result={**row,'reference_path':str((out/(name+'.npz')).resolve()),
            'reference_sha256':sha(out/(name+'.npz')),'tokens':len(ids),'description_token_count':int(content_mask.sum()),
            'selected_free_tokens':int(selected.sum()),'description_byte_ranges':ranges,
            'mean_free_logprob_error':mean,'maximum_logprob_error':maximum,
            'gradient_norm':norm,'loss':float(value.detach()),'skipped_loss_and_gradient_exact_zero':not row['active'] and norm==0}
        results.append(result);write(out/(name+'.json'),result)
        actor.zero_grad(set_to_none=True)
        del current,counts,ids,behavior,selected,reference,error,value,gradients,free
        torch.cuda.empty_cache()
    after=parameter_digest(actor)
    if before!=after:raise ValueError('Zero-update check changed actor tensors')
    b=plan['model_binding']
    if binding(Path(b['base_path']),Path(b['checkpoint_path']))!=b:raise ValueError('Original checkpoint changed')
    write(out/'summary.json',{'status':'PASS','rows':results,'optimizer_created':False,'optimizer_steps':0,
        'parameter_sha256_before':before,'parameter_sha256_after':after,'environment':versions(),
        'active_attempts':sum(r['active'] for r in results),'excluded_attempts':sum(not r['active'] for r in results),
        'tool_context_attempts':sum(r['actual_tool_calls']>0 for r in results),
        'all_excluded_policy_and_KL_gradients_zero':True,'original_checkpoint_unchanged':True,
        'max_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'elapsed_seconds':time.time()-started})
    progress(out,'PASS',optimizer_steps=0,elapsed_seconds=time.time()-started)


def update_once(root,out,plan):
    checked=load(root/'zero_update_check/summary.json')
    if checked['status']!='PASS' or len(checked['rows'])!=20:raise ValueError('All 20 attempts must pass the zero-update check')
    out.mkdir(parents=True,exist_ok=False);started=time.time();steps=0
    try:
        progress(out,'LOADING_ORIGINAL_ACTOR',optimizer_steps=0)
        processor,actor=load_actor(plan);before=parameter_digest(actor)
        if before!=checked['parameter_sha256_before']:raise ValueError('Check/update initial actor tensors differ')
        parameters=[p for p in actor.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=plan['learning_rate'],weight_decay=0.)
        optimizer.zero_grad(set_to_none=True);pre_loss=0.;per=[]
        for index,row in enumerate(checked['rows']):
            if not row['active']:continue
            progress(out,'ACCUMULATING_ONE_UPDATE',request_index=index,group=row['group'],sample=row['sample'],optimizer_steps=0)
            path,item,request,response=bound_row(row,plan)
            if sha(row['reference_path'])!=row['reference_sha256']:raise ValueError('Frozen reference or mask changed')
            with np.load(row['reference_path']) as a:
                reference=torch.from_numpy(a['reference']).to('cuda');selected=torch.from_numpy(a['selected']).to('cuda')
                behavior=torch.from_numpy(a['behavior']).to('cuda')
            current,counts,ids=score_saved_request(actor,processor,path)
            free=counts>1;error=(current.detach()-behavior).abs()
            if float(error[free].mean())>.05 or float(error.max())>.2:raise ValueError('Pre-update policy no longer matches sampler')
            value=masked_loss(current,behavior,reference,row['advantage'],selected,True,
                plan['clip_epsilon'],plan['kl_beta'])/len(plan['rows'])
            value.backward();pre_loss+=float(value.detach())
            per.append({'group':row['group'],'sample':row['sample'],'advantage':row['advantage'],
                        'selected_tokens':int(selected.sum()),'pre_loss_contribution':float(value.detach())})
            del current,counts,ids,free,error,value,reference,selected,behavior;torch.cuda.empty_cache()
        if any(p.grad is not None for p in actor.parameters() if not p.requires_grad):raise ValueError('Base gradient leak')
        gradients=[p.grad for p in parameters if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):raise ValueError('Invalid accumulated gradient')
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,plan['gradient_clip_norm']))
        if not 0<norm<math.inf:raise ValueError('Invalid total gradient norm')
        progress(out,'EXECUTING_SINGLE_OPTIMIZER_STEP',optimizer_steps=0)
        optimizer.step();steps=1;optimizer.zero_grad(set_to_none=True)
        after=parameter_digest(actor)
        if after==before:raise ValueError('Single optimizer step did not change LoRA parameters')
        adapter=out/'adapter_after_one_update';actor.save_pretrained(adapter)
        torch.save(optimizer.state_dict(),out/'optimizer_state.pt')
        post_loss=0.;post=[]
        for row in checked['rows']:
            if not row['active']:
                post.append({'group':row['group'],'sample':row['sample'],'active':False,
                             'policy_loss_contribution':0.,'KL_loss_contribution':0.});continue
            path,_,_,_=bound_row(row,plan)
            with np.load(row['reference_path']) as a:
                reference=torch.from_numpy(a['reference']).to('cuda');selected=torch.from_numpy(a['selected']).to('cuda')
                behavior=torch.from_numpy(a['behavior']).to('cuda')
            with torch.inference_mode():
                current,_,_=score_saved_request(actor,processor,path)
                value=masked_loss(current,behavior,reference,row['advantage'],selected,True,
                    plan['clip_epsilon'],plan['kl_beta'])/len(plan['rows'])
                delta=float((current-reference)[selected].mean())
                ratio=(current-behavior)[selected].exp()
                d=(reference-current)[selected];kl=float((torch.expm1(d)-d).mean())
            post_loss+=float(value)
            post.append({'group':row['group'],'sample':row['sample'],'active':True,'advantage':row['advantage'],
                'mean_description_logprob_delta':delta,'mean_selected_KL':kl,
                'ratio_min':float(ratio.min()),'ratio_max':float(ratio.max())})
            del current,value,ratio,d,reference,selected,behavior;torch.cuda.empty_cache()
        b=plan['model_binding']
        if binding(Path(b['base_path']),Path(b['checkpoint_path']))!=b:raise ValueError('Original SFT checkpoint overwritten')
        summary={'status':'ONE_UPDATE_COMPLETED_NOT_A_SELECTED_MODEL','optimizer_steps':steps,
            'learning_rate':plan['learning_rate'],'groups':5,'original_attempts':20,'active_attempts':12,
            'pre_surrogate_loss':pre_loss,'post_surrogate_loss':post_loss,
            'fixed_sample_surrogate_improved':post_loss<pre_loss,'gradient_norm_before_clip':norm,
            'parameter_sha256_before':before,'parameter_sha256_after':after,
            'original_checkpoint_unchanged':True,'base_parameters_frozen':True,
            'zero_advantage_policy_and_KL_loss_skipped':True,'description_tokens_only':True,
            'pre_update_rows':per,'post_update_rows':post,'adapter_path':str(adapter.resolve()),
            'weak_reference_only':True,'semantic_improvement_verified':False,'fixed120_used':False,
            'elapsed_seconds':time.time()-started,'max_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
        write(out/'summary.json',summary);progress(out,'COMPLETE',optimizer_steps=steps,elapsed_seconds=time.time()-started)
    except Exception as exc:
        write(out/'FAILURE.json',{'error':repr(exc),'optimizer_steps_executed':steps,'automatic_retry':False,
              'original_checkpoint_not_written':True,'partial_new_adapter_not_selected':True})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','check','update'])
    p.add_argument('--source',type=Path,default=ROOT/'video_detection_agent/outputs/grpo_partial_fact_selection_20261030_v1')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.phase=='prepare':prepare(a);return
    root=a.out.resolve();plan=load(root/'PLAN.json')
    if sha(__file__)!=plan['script_sha256']:raise ValueError('Frozen pilot code changed')
    source=Path(plan['source'])
    if sha(source/'CANDIDATES.jsonl')!=plan['candidate_file_sha256'] or sha(source/'PLAN.json')!=plan['source_plan_sha256']:
        raise ValueError('Frozen audit/reward source changed')
    if sha(plan['fact_results_path'])!=plan['fact_results_sha256'] or sha(source/'CROSS_CONTEXT_CONFLICTS.json')!=plan['cross_context_conflicts_sha256']:
        raise ValueError('Frozen factual judgments changed')
    phase=root/('zero_update_check' if a.phase=='check' else 'one_update')
    try:
        if a.phase=='check':check(root,phase,plan)
        else:update_once(root,phase,plan)
    except Exception as exc:
        if phase.exists() and not (phase/'FAILURE.json').exists():
            write(phase/'FAILURE.json',{'error':repr(exc),'optimizer_steps_executed':0,
                'automatic_retry':False,'original_checkpoint_not_written':True})
        raise


if __name__=='__main__':main()
