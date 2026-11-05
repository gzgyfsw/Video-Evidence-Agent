"""Exact per-request Actor credit packing, without rewards or optimization.

Use native token IDs and original pixels. Never concatenate independent calls
into a fictional conversation or credit frozen observation/policy outputs.
"""
import hashlib
import json
from pathlib import Path
import re

from PIL import Image


def bytes_to_unicode():
    """The fixed256-byte alphabet used by the verified ByteLevel decoder."""
    byte_values=list(range(33,127))+list(range(161,173))+list(range(174,256))
    characters=byte_values.copy();extra=0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value);characters.append(256+extra);extra+=1
    return dict(zip(byte_values,map(chr,characters)))


def request_batch(request,processor):
    content=[];images=[]
    for block in request['content_order']:
        if block['type']=='text':content.append({'type':'text','text':block['text']})
        elif block['type']=='image_reference':
            file=Path(block['path'])
            if hashlib.sha256(file.read_bytes()).hexdigest()!=block['sha256']:
                raise ValueError('Actual request pixels changed')
            with Image.open(file) as original:image=original.convert('RGB')
            images.append(image);content.append({'type':'image','image':image})
        else:raise ValueError('Unknown original request content block')
    text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    return processor(text=[text],images=images,padding=False,return_tensors='pt')


def object_field_spans(text):
    decoder=json.JSONDecoder();position=0;result={}
    def space(pos):
        while pos<len(text) and text[pos].isspace():pos+=1
        return pos
    position=space(position)
    if text[position:position+1]!='{':raise ValueError('Tool payload must be the actual complete JSON object')
    position+=1
    while True:
        position=space(position)
        if text[position:position+1]=='}':position=space(position+1);break
        start=position;key,position=decoder.raw_decode(text,position)
        if not isinstance(key,str) or key in result:raise ValueError('Duplicate or non-string JSON field')
        position=space(position)
        if text[position:position+1]!=':':raise ValueError('Invalid field separator')
        _,position=decoder.raw_decode(text,space(position+1));result[key]=(start,position)
        position=space(position)
        if text[position:position+1]==',':position+=1
        elif text[position:position+1]!='}':raise ValueError('Invalid object separator')
    if position!=len(text):raise ValueError('Unexpected trailing tool payload text')
    return result


def actual_token_byte_spans(tokenizer,ids):
    decoder=json.loads(tokenizer.backend_tokenizer.decoder.__getstate__())
    if decoder.get('type')!='ByteLevel':raise ValueError('Unsupported tokenizer decoder; do not guess token boundaries')
    inverse={character:value for value,character in bytes_to_unicode().items()}
    chunks=[];spans=[];offset=0;special=set(tokenizer.all_special_ids)
    for token in ids:
        if token in special:chunk=b''
        else:
            piece=tokenizer.convert_ids_to_tokens(token)
            try:chunk=bytes(inverse[character] for character in piece)
            except (KeyError,TypeError) as exc:raise ValueError('Actual generated token bytes could not be recovered') from exc
        chunks.append(chunk);spans.append((offset,offset+len(chunk)));offset+=len(chunk)
    return b''.join(chunks),spans


def build_actor_credit(call,source_partition,actor_model,processor,checkpoint_binding):
    if source_partition!='train':raise ValueError('Dev/Test requests are forbidden as GRPO training inputs')
    if call['model']!=actor_model or call['stage'] not in {'route','tool_payload'}:return None
    if checkpoint_binding.get('model')!=actor_model or not checkpoint_binding.get('weight_sha256'):
        raise ValueError('Actual Actor checkpoint binding required')
    path=Path(call['request_input_manifest'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=call['request_input_sha256']:
        raise ValueError('Original model request changed')
    request=json.loads(path.read_text());raw=call['raw']
    if request['model']!=actor_model or request['stage']!=call['stage'] or raw['model']!=actor_model:
        raise ValueError('Request/response Actor role mismatch')
    prefix=raw.get('prompt_token_ids');completion=raw['choices'][0].get('token_ids')
    if not prefix or not completion:raise ValueError('Actual native prompt/completion IDs required; do not re-encode guessed history')
    batch=request_batch(request,processor)
    if batch['input_ids'][0].tolist()!=prefix:raise ValueError('Local multimodal prompt differs from actual native request')
    text=raw['choices'][0]['message']['content'] or ''
    if call['stage']=='route':
        match=re.search(r'"action"\s*:\s*"(ANSWER|JOINT_SEMANTIC_SCAN|TEMPORAL_EXPAND)"',text)
        if not match:raise ValueError('No model-selected route field; record invalid model output separately')
        character_spans=[match.span()];fields=['action']
    else:
        output=json.loads(text);action=output.get('action')
        fields={'JOINT_SEMANTIC_SCAN':['queries_by_risk'],
                'TEMPORAL_EXPAND':['timestamps','window','num_frames']}.get(action)
        if not fields:raise ValueError('Unexpected tool payload action')
        spans=object_field_spans(text)
        if set(spans)!=set(fields)|{'action'}:raise ValueError('Unrecognized or missing executed tool parameter fields')
        character_spans=[spans[field] for field in fields]
    wanted=[(len(text[:start].encode('utf-8')),len(text[:end].encode('utf-8'))) for start,end in character_spans]
    actual,offsets=actual_token_byte_spans(processor.tokenizer,completion);encoded=text.encode('utf-8')
    common=next((i for i,(a,b) in enumerate(zip(actual,encoded)) if a!=b),min(len(actual),len(encoded)))
    if any(end>common for _,end in wanted):raise ValueError('Actual token bytes do not cover the complete credited field')
    selected=[i for i,(a,b) in enumerate(offsets) if b>a and any(start<=a<b<=end for start,end in wanted)]
    if not selected:raise ValueError('No verified Actor decision tokens')
    return {'stage':call['stage'],'actor_model':actor_model,'source_partition':'train',
        'checkpoint_binding':checkpoint_binding,'request_path':str(path),'request_sha256':call['request_input_sha256'],
        'prompt_token_ids':prefix,'completion_token_ids':completion,
        'credited_completion_indices':selected,'supervised_fields':fields,
        'actual_completion_ids_used_without_reencoding':True,'fresh_single_user_context':True,
        'verified_multimodal_prompt_token_match':True,'frozen_helper_tokens_credited':0,
        'reward_not_computed':True,'optimizer_steps':0,'model_logprob_or_gradient_not_verified':True}
