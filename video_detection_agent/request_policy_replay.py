"""Exact single-request probability replay, never a concatenated conversation.

This is the shared batch/score interface for old/new policy evaluation. It does
not compute rewards or perform updates. Teacher text is never re-tokenized.
"""
import hashlib
import json
from pathlib import Path
import torch
from actor_request_credit import request_batch
from actor_policy_probabilities import allowed_from_bitmask, constrained_token_logprobs

VERSION='request-policy-replay-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_sampling(request):
    contract=request.get('effective_sampling_contract',{})
    expected={'version':1,'generation_config':'vllm','logprobs_mode':'processed_logprobs',
        'top_p':1.0,'top_k':-1,'min_p':0.0,'repetition_penalty':1.0,
        'presence_penalty':0.0,'frequency_penalty':0.0,'min_tokens':0,
        'ignore_eos':False,'logit_bias':{},'stop':[],'bad_words':[]}
    if any(contract.get(k)!=v for k,v in expected.items()):
        raise ValueError('Effective behavior distribution is missing or unsupported; never infer defaults')
    if type(request.get('temperature')) not in {int,float} or not 0<request['temperature']<=2:
        raise ValueError('Positive stochastic temperature required')
    if request.get('chat_template_kwargs')!={'enable_thinking':False}:
        raise ValueError('Unsupported/missing thinking contract')
    return contract


def pack(call, source, binding):
    if source.get('source_partition')!='train':raise ValueError('Only original Train requests allowed')
    if call['model']!=binding['model'] or not binding.get('weight_sha256'):
        raise ValueError('Actual sampling model/checkpoint not bound')
    request_path=Path(call['request_input_manifest'])
    if sha(request_path)!=call['request_input_sha256']:raise ValueError('Actual request changed')
    request=json.loads(request_path.read_text());validate_sampling(request)
    raw=call['raw'];choice=raw['choices'][0]
    response_path=Path(call['response_path'])
    if json.loads(response_path.read_text())!=raw:raise ValueError('Saved native response differs from packed response')
    if raw['model']!=binding['model'] or request['model']!=binding['model']:raise ValueError('Model identity mismatch')
    prefix=raw.get('prompt_token_ids');tokens=choice.get('token_ids')
    if not prefix or not tokens:raise ValueError('Native prompt/completion token IDs required')
    scores=(choice.get('logprobs') or {}).get('content')
    if not scores or len(scores)!=len(tokens):raise ValueError('Native per-token logprobs missing')
    entries=[{k:v for k,v in b.items() if k!='type'} for b in request['content_order'] if b['type']=='image_reference']
    if entries!=request['images']:raise ValueError('Image list/order mismatch')
    for im in entries:
        if sha(im['path'])!=im['sha256']:raise ValueError('Frozen image changed')
    return {'version':VERSION,'source_partition':'train','source_trajectory_id':source['source_trajectory_id'],
        'stage':call['stage'],'actor_model':binding['model'],'checkpoint_binding':binding,
        'request_path':str(request_path),'request_sha256':call['request_input_sha256'],
        'native_response_path':str(response_path),'native_response_sha256':sha(response_path),
        'prompt_token_ids':prefix,'completion_token_ids':tokens,
        'credited_completion_indices':list(range(len(tokens))),
        'native_processed_logprobs':[e['logprob'] for e in scores],
        'finish_reason':choice['finish_reason'],'fresh_single_user_context':True,
        'sampling_contract':request['effective_sampling_contract'],'temperature':request['temperature'],
        'request_provenance':source,'reward_not_computed':True,'optimizer_steps':0}


def replay_batch(item,processor,device='cpu'):
    if item.get('version')!=VERSION or item.get('source_partition')!='train':raise ValueError('Wrong replay version/split')
    request_path=Path(item['request_path'])
    if sha(request_path)!=item['request_sha256']:raise ValueError('Request changed')
    request=json.loads(request_path.read_text());validate_sampling(request)
    if (item['temperature']!=request['temperature'] or item['sampling_contract']!=request['effective_sampling_contract']
            or item['actor_model']!=request['model'] or item['checkpoint_binding']['model']!=request['model']):
        raise ValueError('Packed sampling/model identity differs from the actual request')
    if sha(item['native_response_path'])!=item['native_response_sha256']:raise ValueError('Native response changed')
    native=json.loads(Path(item['native_response_path']).read_text())
    choice=native['choices'][0]
    if (native.get('prompt_token_ids')!=item['prompt_token_ids'] or choice.get('token_ids')!=item['completion_token_ids']
            or [r['logprob'] for r in choice['logprobs']['content']]!=item['native_processed_logprobs']):
        raise ValueError('Replay IDs/probabilities differ from actual native response')
    batch=request_batch(request,processor)
    prompt=item['prompt_token_ids'];completion=item['completion_token_ids']
    if batch['input_ids'][0].tolist()!=prompt:raise ValueError('Native/local prefix tokens differ')
    indices=item['credited_completion_indices']
    if not indices or indices!=sorted(set(indices)) or indices[0]<0 or indices[-1]>=len(completion):raise ValueError('Invalid token credit indices')
    last=max(indices)
    # To predict y[j], input stops at y[j-1]. Images always stay in this request.
    batch['input_ids']=torch.tensor([prompt+completion[:last]],dtype=torch.long)
    for key in ['attention_mask','mm_token_type_ids']:
        if key in batch:
            batch[key]=torch.cat([batch[key],torch.full((1,last),1 if key=='attention_mask' else 0,dtype=batch[key].dtype)],dim=1)
    positions=torch.tensor([len(prompt)-1+i for i in indices],device=device)
    targets=torch.tensor([completion[i] for i in indices],device=device)
    return {k:v.to(device) for k,v in batch.items()},positions,targets


def score_request(model,processor,item,grammar,device='cuda',decode_mode='full_causal'):
    """Differentiable score for exactly ONE request; caller controls no_grad."""
    import numpy as np
    if grammar['request_sha256']!=item['request_sha256'] or sha(grammar['masks_path'])!=grammar['masks_sha256']:
        raise ValueError('Grammar masks bound to a different request')
    batch,positions,targets=replay_batch(item,processor,device)
    if 'pixel_values' in batch:batch['pixel_values']=batch['pixel_values'].to(next(model.parameters()).dtype)
    if decode_mode=='full_causal':
        output=model(**batch,use_cache=False,logits_to_keep=positions)
        logits=output.logits[0].float()
    elif decode_mode=='cached_teacher_forcing':
        n=len(item['prompt_token_ids']);completion=item['completion_token_ids'];wanted=set(item['credited_completion_indices'])
        initial={k:(v[:,:n] if k in {'input_ids','attention_mask','mm_token_type_ids'} else v) for k,v in batch.items()}
        if 'attention_mask' in initial and not bool((initial['attention_mask']==1).all()):
            raise ValueError('Cached replay currently supports only one unpadded request')
        output=model(**initial,use_cache=True,logits_to_keep=1)
        rows=[]
        for i in range(max(wanted)+1):
            if i in wanted:rows.append(output.logits[0,-1].float())
            if i==max(wanted):break
            cache=output.past_key_values
            if cache is None:raise ValueError('Model did not return its actual decode cache')
            # Single unpadded request: no decode attention mask is necessary.
            # Qwen3.5's direct forward otherwise builds multi-token RoPE positions
            # from the full mask instead of this one-token cache increment.
            output=model(input_ids=torch.tensor([[completion[i]]],device=device),
                mm_token_type_ids=torch.zeros((1,1),dtype=torch.long,device=device),
                past_key_values=cache,use_cache=True,logits_to_keep=1)
        logits=torch.stack(rows)
    else:raise ValueError('Unknown causal scoring mode')
    with np.load(grammar['masks_path']) as arrays:
        if arrays['indices'].tolist()!=item['credited_completion_indices'] or arrays['token_ids'].tolist()!=targets.tolist():
            raise ValueError('Mask/token alignment mismatch')
        allowed=allowed_from_bitmask(arrays['bitmask'],grammar['vocab_size'],device)
    if allowed.shape!=logits.shape:raise ValueError('Model/grammar vocab mismatch')
    processed=constrained_token_logprobs(logits,targets,allowed,item['temperature'])
    raw=logits.log_softmax(-1).gather(-1,targets[:,None]).squeeze(-1)
    return processed,raw
