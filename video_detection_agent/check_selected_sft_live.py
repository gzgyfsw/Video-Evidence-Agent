#!/usr/bin/env python3
"""Run the selected SFT modules through the original bounded thin tool loop."""
import argparse
import base64
import hashlib
import io
import json
import re
import time
from pathlib import Path
from openai import OpenAI
import jsonschema
from agent.qwen_backend import SAFETY_POLICY
from agent.tools import JointRiskSemanticLocator
from evaluate_thin_harness_teacher_pilot import run_one, _thin_action_schema

POLICY_SHA='45fb654deceb9edea3681dc87f2d352e164495b6a28165d84a27afd332bd6040'
ACTIONS={'ANSWER','JOINT_SEMANTIC_SCAN','TEMPORAL_EXPAND'}


def route_prompt(base, timestamps, state):
    return (base+'\n当前真实已见帧时间戳：'+json.dumps(timestamps)
            +'\n当前运行状态（历史工具记录不是视觉证据）：'+json.dumps(state,ensure_ascii=False))


def frame_time_label(index,timestamp):
    return f'IMAGE {index}: actual video timestamp = {json.dumps(timestamp)} seconds.'


def tool_payload_schema(action, anchors=None, frames_remaining=None):
    """Constrain transport types only; the model still selects every query/anchor."""
    fields={'JOINT_SEMANTIC_SCAN':['queries_by_risk'],
            'TEMPORAL_EXPAND':['timestamps','window','num_frames']}[action]
    properties=_thin_action_schema()['schema']['properties']
    schema={'type':'object','additionalProperties':False,
            'required':['action',*fields],
            'properties':{'action':{'type':'string','enum':[action]},
                          **{key:properties[key] for key in fields}}}
    if action=='JOINT_SEMANTIC_SCAN':
        # Empty risk maps are invalid in the actual tool interface.
        schema['properties']['queries_by_risk']={**properties['queries_by_risk'],
            'minProperties':1}
    elif anchors is not None:
        if not anchors:raise ValueError('temporal expansion has no known anchors')
        schema['properties']['timestamps']['items']={'type':'number','enum':anchors}
        if frames_remaining is not None:
            schema['properties']['num_frames']['maximum']=max(1,min(8,frames_remaining))
    return schema


def inspection_context(state):
    """Past inspection metadata, never prior guessed labels or similarity scores."""
    summary=state.get('history_summary',{})
    return {key:state.get(key) for key in ['duration','tools_remaining','frames_remaining',
                                          'all_acquired_timestamps']} | {
        'tool_call_count':summary.get('tool_call_count',0),
        'prior_tools':summary.get('prior_tools',[])}


def answer_payload_schema():
    properties=_thin_action_schema()['schema']['properties']
    fields=['category','severity','moderation_action','review_reason','rationale','evidence','risk_coverage']
    return {'type':'object','additionalProperties':False,'required':['action',*fields],
            'properties':{'action':{'type':'string','enum':['ANSWER']},
                          **{key:properties[key] for key in fields}}}


def observation_schema(frames, max_items=6, max_description_chars=220):
    if type(max_description_chars) is not int or not 1 <= max_description_chars <= 1024:
        raise ValueError('Observation description limit must be an integer from1 to1024')
    return {'type':'object','additionalProperties':False,'required':['evidence'],
            'properties':{'evidence':{'type':'array','minItems':0,'maxItems':max_items,
               'items':{'type':'object','additionalProperties':False,
                  'required':['timestamp','description'],'properties':{
                    'timestamp':{'type':'number','enum':[f.timestamp for f in frames]},
                    'description':{'type':'string','maxLength':max_description_chars}}}}}}


def witness_selection_schema(frame_count):
    if frame_count<1:raise ValueError('no actually seen images available')
    return {'type':'object','additionalProperties':False,'required':['witness_indices'],
        'properties':{'witness_indices':{'type':'array','minItems':1,'maxItems':min(3,frame_count),
           'items':{'type':'integer','enum':list(range(frame_count))}}}}


def policy_payload_schema(evidence_count=None, include_identity=False, evidence_first=False,
                          enforce_operational_tuple=False):
    schema=answer_payload_schema()
    for key in ['rationale','evidence']:
        schema['properties'].pop(key);schema['required'].remove(key)
    coverage=schema['properties']['risk_coverage']
    schema['properties']['risk_coverage']={'type':'object','additionalProperties':False,
        'required':list(coverage['properties']),
        'properties':{k:v['properties']['status'] for k,v in coverage['properties'].items()}}
    if evidence_count is not None:
        schema['properties']['evidence_index']={'type':['integer','null'],'enum':[None,*range(evidence_count)]}
        schema['required'].append('evidence_index')
    if include_identity:
        original=_thin_action_schema()['schema']['properties']
        for key in ['evidence_basis','identified_figure']:
            schema['properties'][key]={**original[key],'minLength':1}
            schema['required'].append(key)
    if evidence_first:
        order=['action','evidence_index','identified_figure','evidence_basis','category',
               'severity','moderation_action','review_reason','risk_coverage']
        order=[key for key in order if key in schema['properties']]
        if set(order)!=set(schema['properties']):raise ValueError('Unrecognized policy field order')
        schema['properties']={key:schema['properties'][key] for key in order}
        schema['required']=order
    if enforce_operational_tuple:
        harmful=['Sexual Content','Violence & Dangerous','Sensitive Figures & Ideologies']
        alternatives=[
            {'properties':{'category':{'const':'Harmless'},'severity':{'const':'L0'},
                'moderation_action':{'const':'ALLOW'},'review_reason':{'type':'null'}}},
            {'properties':{'category':{'enum':harmful},'severity':{'const':'L1'},
                'moderation_action':{'const':'REVIEW'},'review_reason':{'const':'CONFIRMED_L1'},
                **({'evidence_index':{'type':'integer'}} if evidence_count is not None else {})}},
            {'properties':{'category':{'enum':harmful},'severity':{'const':'L2'},
                'moderation_action':{'const':'BLOCK'},'review_reason':{'type':'null'},
                **({'evidence_index':{'type':'integer'}} if evidence_count is not None else {})}},
            {'properties':{'severity':{'type':'null'},'moderation_action':{'const':'REVIEW'},
                'review_reason':{'const':'EVIDENCE_INSUFFICIENT'}}}]
        # Native xgrammar may compile anyOf independently of its sibling
        # properties/required. Repeat the complete bounded object in every
        # branch, rather than relying on that implicit intersection. Otherwise
        # even evidence_index can become an unbounded integer at generation.
        from copy import deepcopy
        branches=[]
        for alternative in alternatives:
            if evidence_count==0 and alternative['properties'].get('evidence_index',{}).get('type')=='integer':
                continue  # Unsatisfiable: no selectable factual candidate exists.
            branch=deepcopy(schema)
            for key,constraint in alternative['properties'].items():
                # Do not retain a broader enum beside a narrower const/type.
                # Some native decoders prioritize enum and silently ignore
                # the intersecting restriction. Materialize the intersection.
                if key=='evidence_index' and constraint.get('type')=='integer':
                    branch['properties'][key]={'type':'integer',
                        'enum':[i for i in schema['properties'][key]['enum'] if i is not None]}
                else:
                    branch['properties'][key]=deepcopy(constraint)
            branches.append(branch)
        schema['anyOf']=branches
    return schema


def carry_original_identity_fields(final, transcript):
    """Copy existing model metadata, without generating a name or an assertion."""
    if final.get('stop_reason')!='MODEL_ANSWER':return
    proposals=[e['raw_decision'] for e in transcript if 'raw_decision' in e]
    if not proposals or proposals[-1].get('action')!='ANSWER':
        raise ValueError('Accepted answer lacks its original model proposal')
    for key in ['evidence_basis','identified_figure']:
        if key in proposals[-1]:final[key]=proposals[-1][key]


def parse_payload(text):
    text=text.strip()
    if text.startswith('```') and text.endswith('```'):
        text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    result=json.loads(text)
    if not isinstance(result,dict):raise ValueError('action payload must be an object')
    return result


def generation_settings(args,model,stage,index):
    """Default validation is unchanged; Train samplers opt into one Actor role."""
    actor_temperature=getattr(args,'actor_temperature',0)
    target=getattr(args,'sample_actor_model',None)
    if not 0<=actor_temperature<=2:raise ValueError('Actor sampling temperature must be between0 and2')
    if actor_temperature and not target:raise ValueError('Specify the single sampled Actor model alias')
    sampled=bool(actor_temperature and model==target and stage in {'route','tool_payload'})
    settings={'temperature':actor_temperature if sampled else 0,
              'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}}
    if sampled:settings['seed']=int(getattr(args,'sampling_seed',20261018))+index
    if getattr(args,'capture_token_ids',False):
        settings['extra_body'].update(return_token_ids=True,return_prompt_text=True)
    return settings


class SplitBrain:
    def __init__(self,args,out,prompt):
        self.args,self.out,self.prompt=args,out,prompt
        self.client=OpenAI(base_url=args.base_url,api_key='EMPTY',timeout=180,max_retries=0)
        self.trace=[]
        self.last_raw_response=None
        prompt_path=Path(args.contract)
        self.contract=json.loads(prompt_path.read_text())
        self.contract_sha256=hashlib.sha256(prompt_path.read_bytes()).hexdigest()

    def call_model(self,model,frames,prompt,stage,tokens):
        content=[{'type':'text','text':prompt}]
        images=[]
        for n,frame in enumerate(frames):
            image=frame.image.convert('RGB')
            if image.width*image.height>409600:
                scale=(409600/(image.width*image.height))**.5
                image=image.resize((max(1,int(image.width*scale)),max(1,int(image.height*scale))))
            buf=io.BytesIO();image.save(buf,format='JPEG',quality=95)
            data=buf.getvalue();digest=hashlib.sha256(data).hexdigest()
            path=self.out/'images'/(digest+'.jpg');path.parent.mkdir(exist_ok=True)
            if not path.exists():path.write_bytes(data)
            images.append({'path':str(path.resolve()),'sha256':digest,'timestamp':frame.timestamp})
            if getattr(self.args,'label_frame_times',False) and stage in {'witness_selection','evidence_observation','policy_payload','answer_payload'}:
                content.append({'type':'text','text':frame_time_label(n,frame.timestamp)})
            content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(data).decode()}})
        output_format=None
        if stage=='tool_payload' and self.args.tool_output_format=='json_schema':
            chosen=re.search(r'路由模型本轮已选择 (\w+)',prompt)
            if not chosen:raise ValueError('tool payload missing explicit model-selected action')
            output_format={'type':'json_schema','json_schema':{'name':'tool_payload','strict':True,
                           'schema':getattr(self,'tool_schema_override',None) or tool_payload_schema(chosen[1])}}
        if stage=='answer_payload' and self.args.answer_output_format=='json_schema':
            output_format={'type':'json_schema','json_schema':{'name':'answer_payload','strict':True,
                           'schema':answer_payload_schema()}}
        if stage in {'evidence_observation','policy_payload'}:
            schema=observation_schema(frames,1 if getattr(self.args,'single_witness',False) else 6,
                self.contract.get('observation_description_limit',220)) if stage=='evidence_observation' else policy_payload_schema(
                self.evidence_choice_count if getattr(self.args,'single_witness',False) else None,
                include_identity=self.contract.get('include_identity_fields',False),
                evidence_first=self.contract.get('policy_evidence_first',False),
                enforce_operational_tuple=self.contract.get('enforce_operational_tuple',False))
            output_format={'type':'json_schema','json_schema':{'name':stage,'strict':True,'schema':schema}}
        if stage=='witness_selection':
            output_format={'type':'json_schema','json_schema':{'name':stage,'strict':True,
                'schema':witness_selection_schema(len(frames))}}
        settings=generation_settings(self.args,model,stage,len(self.trace)+1)
        request={'model':model,'messages':[{'role':'user','content':content}],
                 'max_tokens':tokens,**settings}
        if output_format:request['response_format']=output_format
        request_dir=self.out/'request_inputs';request_dir.mkdir(exist_ok=True)
        request_path=request_dir/f'{len(self.trace)+1:04d}_{stage}.json'
        if request_path.exists():raise FileExistsError('Preserve the original request attempt; use an explicit recovery directory')
        order=[];image_index=0
        for block in content:
            if block['type']=='text':order.append(block)
            else:
                order.append({'type':'image_reference',**images[image_index]});image_index+=1
        request_input={'model':model,'stage':stage,'prompt':prompt,'content_order':order,
            'images':images,'temperature':settings['temperature'],'max_tokens':tokens,'response_format':output_format,
            'chat_template_kwargs':{'enable_thinking':False},'policy_sha256':POLICY_SHA,
            'contract_prompt_sha256':self.contract_sha256}
        if 'seed' in settings:request_input['seed']=settings['seed']
        if getattr(self.args,'capture_token_ids',False):
            request_input.update(return_token_ids=True,return_prompt_text=True)
        if stage=='policy_payload' and getattr(self,'evidence_candidate_manifest',None):
            candidate_path=self.evidence_candidate_manifest
            request_input['evidence_candidate_manifest']=str(candidate_path.resolve())
            request_input['evidence_candidate_manifest_sha256']=hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        request_path.write_text(json.dumps(request_input,ensure_ascii=False,indent=2))
        request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest()
        self.last_raw_response=None
        request_started=time.monotonic()
        try:response=self.client.chat.completions.create(**request)
        except Exception as exc:
            request_path.with_suffix('.failure.json').write_text(json.dumps({
                'request_input':str(request_path),'status':'TECHNICAL_FAILURE',
                'error':repr(exc),'elapsed_request_seconds':time.monotonic()-request_started,
                'not_a_semantic_judgment':True},ensure_ascii=False,indent=2))
            raise
        text=response.choices[0].message.content or ''
        self.last_raw_response=text
        item={'stage':stage,'model':model,'prompt':prompt,'images':images,'raw':response.model_dump(),
              'elapsed_request_seconds':time.monotonic()-request_started,
              'policy_sha256':POLICY_SHA,'owns_route':stage=='route','owns_tool_parameters':stage=='tool_payload',
              'contract_prompt_sha256':self.contract_sha256,'response_format':output_format}
        item['request_input_manifest']=str(request_path)
        item['request_input_sha256']=request_sha256
        if 'evidence_candidate_manifest' in request_input:
            for key in ['evidence_candidate_manifest','evidence_candidate_manifest_sha256']:
                item[key]=request_input[key]
        item['label_frame_times']=getattr(self.args,'label_frame_times',False)
        self.trace.append(item)
        with (self.out/'model_calls.jsonl').open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
        if getattr(self.args,'capture_token_ids',False):
            raw=item['raw']
            if not raw.get('prompt_token_ids') or not raw['choices'][0].get('token_ids'):
                raise RuntimeError('Actual request token IDs missing; raw request and response preserved')
        if stage!='route':
            if response.choices[0].finish_reason=='length':raise ValueError('payload truncated at generation token limit')
            parsed=parse_payload(text)
            if output_format:jsonschema.validate(parsed,output_format['json_schema']['schema'])
        return text

    def call(self,frames,runtime_prompt):
        state=json.loads(runtime_prompt.rsplit('当前状态：',1)[1])
        has_tools=bool(state.get('history_summary',{}).get('tool_call_count',0))
        has_tools=has_tools or any(e.get('executed_action') in ACTIONS-{'ANSWER'} for e in state['recent_tool_observations'])
        has_tools=has_tools or len(state['all_acquired_timestamps'])>4
        route_model=self.args.post_model if has_tools else self.args.initial_model
        # No ground-truth labels, future images or old final answer appear.
        prompt=route_prompt(self.prompt,[f.timestamp for f in frames],state)
        text=self.call_model(route_model,frames,prompt,'route',32)
        match=re.search(r'"action"\s*:\s*"([^"]+)"',text)
        if match is None or match[1] not in ACTIONS:
            raise ValueError('unparseable model-selected route')
        action=match[1]
        if action=='ANSWER':
            if getattr(self.args,'evidence_mode','full_answer')=='isolated':
                # This fresh request contains no policy, risk label, old answer,
                # route completion, or retrieved scores. Images are only those
                # already delivered to the actor at this stopping state.
                facts_model=getattr(self.args,'observation_model',None) or self.args.answer_model
                policy_frames=frames
                if getattr(self.args,'witness_first',False):
                    selection_prompt=SAFETY_POLICY+'\n'+self.contract['witness_selection_instruction']
                    selection_prompt+='\n实际取证过程（不是视觉证据）：'+json.dumps(inspection_context(state),ensure_ascii=False)
                    choice=parse_payload(self.call_model(self.args.answer_model,frames,
                        selection_prompt,'witness_selection',128))['witness_indices']
                    if (not choice or len(choice)>3 or len(set(choice))!=len(choice)
                        or any(type(i) is not int or not 0<=i<len(frames) for i in choice)):
                        raise ValueError('witness selection does not reference distinct actually seen frames')
                    policy_frames=[frames[i] for i in choice]
                    facts={'evidence':[]}
                    for frame in policy_frames:
                        observation_prompt=self.contract['observation_instruction']
                        observation_prompt+='\nImage order and exact timestamps: '+json.dumps([frame.timestamp])
                        observed=parse_payload(self.call_model(facts_model,[frame],
                            observation_prompt,'evidence_observation',384))
                        facts['evidence']+=observed['evidence']
                else:
                    observation_prompt=self.contract['observation_instruction']
                    observation_prompt+='\nImage order and exact timestamps: '+json.dumps([f.timestamp for f in frames])
                    facts=parse_payload(self.call_model(facts_model,frames,
                        observation_prompt,'evidence_observation',768))
                self.evidence_candidate_manifest=None
                if self.contract.get('evidence_unit')=='source_punctuation_segments':
                    from source_evidence_segments import segment_evidence
                    source_facts=facts['evidence']
                    candidates,bindings=segment_evidence(source_facts)
                    manifest={'version':'source-evidence-segments-v1','policy_call_index':len(self.trace)+1,
                        'source_observations':source_facts,'candidates':candidates,'bindings':bindings,
                        'factual_support':'UNVERIFIED_MODEL_OBSERVATIONS','new_visual_observations':False}
                    folder=self.out/'evidence_candidates';folder.mkdir(exist_ok=True)
                    self.evidence_candidate_manifest=folder/f'{len(self.trace)+1:04d}.json'
                    self.evidence_candidate_manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
                    facts={'evidence':candidates}
                policy_prompt=SAFETY_POLICY+'\n'+self.contract['policy_instruction']
                policy_prompt+='\n当前图片的独立观察（模型弱参考，仍需对照真实图片）：'+json.dumps(facts,ensure_ascii=False)
                policy_prompt+='\n已发生的取证过程（用于判断检查覆盖，不是视觉证据）：'+json.dumps(inspection_context(state),ensure_ascii=False)
                self.evidence_choice_count=len(facts['evidence'])
                decision=parse_payload(self.call_model(self.args.answer_model,policy_frames,
                    policy_prompt,'policy_payload',384))
                if decision.get('action')!='ANSWER':raise ValueError('policy module changed selected action')
                # Transport existing model outputs; never write or repair a fact
                # in Python, and do not let classification generate new facts.
                reported=facts['evidence']
                if getattr(self.args,'single_witness',False):
                    index=decision.pop('evidence_index')
                    if index is not None and (type(index) is not int or not 0<=index<len(reported)):
                        raise ValueError('policy selected a nonexistent factual observation')
                    reported=[] if index is None else [reported[index]]
                return {**decision,'evidence':reported,'rationale':None}
            final_prompt=self.prompt+'\n'+self.contract['answer_instruction']
            final_prompt+='\n当前真实已见帧时间戳：'+json.dumps([f.timestamp for f in frames])
            feedback=[e['rejected'] for e in state['recent_tool_observations'] if 'rejected' in e]
            if feedback:final_prompt+='\n此前格式检查反馈：'+json.dumps(feedback,ensure_ascii=False)
            result=self.call_model(self.args.answer_model,frames,final_prompt,'answer_payload',1024)
        else:
            instruction=self.contract[action]
            anchors=sorted(set([float(f.timestamp) for f in frames]
                +[float(t) for t in state.get('all_acquired_timestamps',[])]
                +[float(m['timestamp']) for e in state.get('recent_tool_observations',[])
                  for m in e.get('matches',[]) if isinstance(m.get('timestamp'),(int,float))]))
            self.tool_schema_override=tool_payload_schema(action,anchors,state.get('frames_remaining'))
            result=self.call_model(route_model,frames,prompt+'\n路由模型本轮已选择 '+action+
                '。现在仅补齐同一动作的真实工具参数，不改变动作。'+instruction,'tool_payload',512)
        parsed=parse_payload(result)
        if parsed.get('action')!=action:
            raise ValueError('payload executor changed the model-selected action')
        return parsed


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pack',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--base-url',required=True)
    ap.add_argument('--initial-model',default='initial')
    ap.add_argument('--post-model',default='post')
    ap.add_argument('--answer-model',default='answer')
    ap.add_argument('--contract',type=Path,default=Path(__file__).with_name('prompts')/'sft_live_contract_v2.json')
    ap.add_argument('--tool-output-format',choices=['unconstrained','json_schema'],default='json_schema')
    ap.add_argument('--answer-output-format',choices=['unconstrained','json_schema'],default='json_schema')
    ap.add_argument('--evidence-mode',choices=['full_answer','isolated'],default='full_answer')
    ap.add_argument('--observation-model',default=None,
                    help='Optional separately trained observation adapter on the same base; defaults to answer model')
    ap.add_argument('--label-frame-times',action='store_true')
    ap.add_argument('--single-witness',action='store_true',help='Policy selects one existing observation, never rewrites facts')
    ap.add_argument('--witness-first',action='store_true',help='Select real witness frames, inspect each independently, judge only the selected images')
    ap.add_argument('--capture-token-ids',action='store_true',help='Save actual native prompt/completion tokens; no sampling or gradients')
    ap.add_argument('--locator-model',required=True)
    ap.add_argument('--max-samples',type=int,default=8)
    ap.add_argument('--manifest',type=Path,default=None,
                    help='Frozen JSONL of Dev IDs; identity/path always resolved from pack Dev manifest')
    ap.add_argument('--num-shards',type=int,default=1)
    ap.add_argument('--shard-index',type=int,default=0)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    assert hashlib.sha256(SAFETY_POLICY.encode()).hexdigest()==POLICY_SHA,'frozen policy changed'
    args.out.mkdir(parents=True)
    (args.out/'runtime_config.json').write_text(json.dumps(vars(args),default=str,indent=2))
    (args.out/'runtime_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'contract_snapshot.json').write_bytes(args.contract.read_bytes())
    rows=[json.loads(l) for l in (args.pack/'dev_manifest.jsonl').open()]
    if args.manifest:
        requested=[json.loads(line)['merged_id'] for line in args.manifest.open() if line.strip()]
        catalog={r['merged_id']:r for r in rows}
        if len(requested)!=len(set(requested)):raise ValueError('duplicate live Dev ID')
        if not set(requested)<=set(catalog):raise ValueError('live manifest contains non-Dev ID')
        chosen=[catalog[i] for i in requested]
        selection='Externally frozen metadata-only Dev manifest; no selection by model outcomes'
    else:
        chosen=[]
        for route,category in [('DIRECT','Harmless'),('DIRECT','Sexual Content'),
                               ('ONE_TOOL','Harmless'),('ONE_TOOL','Sensitive Figures & Ideologies')]:
            group=sorted((r for r in rows if r['route_tier']==route and r['mapped_category']==category),
                         key=lambda r:(float(r['duration_seconds']),r['merged_id']))
            if not group:raise ValueError(f'missing live coverage cell {route}/{category}')
            chosen+=group[:1]
        chosen+=sorted((r for r in rows if r['route_tier']=='TWO_PLUS_TOOLS'),
                       key=lambda r:(float(r['duration_seconds']),r['merged_id']))[:4]
        chosen=chosen[:args.max_samples]
        selection='Legacy eight-case metadata-only smoke, not an accuracy estimate'
    if not 0<=args.shard_index<args.num_shards:raise ValueError('invalid shard index')
    chosen=[r for i,r in enumerate(chosen) if i%args.num_shards==args.shard_index]
    (args.out/'manifest.json').write_text(json.dumps({'rows':chosen,'selection':selection,
        'input_manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else None,
        'policy_sha256':POLICY_SHA},ensure_ascii=False,indent=2))
    example=json.loads(next((args.pack/'train_route_round1.jsonl').open()))
    prompt=example['prompt'].rsplit('\n当前真实已见帧时间戳：',1)[0]
    locator=JointRiskSemanticLocator(model_name=args.locator_model,device='cuda',embedding_cache_dir=str(args.out/'locator_cache'))
    result_rows=[]
    for row in chosen:
        case=args.out/row['merged_id'];case.mkdir()
        brain=SplitBrain(args,case,prompt)
        started=time.monotonic()
        try:
            final,transcript=run_one(row['video_path'],brain,locator,max_decisions=5,max_tools=3,
                                    max_total_frames=24,max_visible_frames=12,max_rejections=2,
                                    audit_only_safety=False)
            if brain.contract.get('include_identity_fields',False):
                carry_original_identity_fields(final,transcript)
            result={'id':row['merged_id'],'reference_route':row['route_tier'],
                    'final':final,'transcript':transcript,'error':None}
        except Exception as exc:
            result={'id':row['merged_id'],'reference_route':row['route_tier'],
                    'error':repr(exc),'transcript':getattr(exc,'transcript',[]),'final':None}
        result.update(model_requests=len(brain.trace),elapsed_seconds=time.monotonic()-started,
                      weak_reference_only=True,independent_gold=False)
        (case/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        result_rows.append(result)
        with (args.out/'results.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(row['merged_id'],result.get('final'),result['error'],flush=True)
        if result['error']:
            raise RuntimeError('Live smoke stopped at first failure; raw input/output saved for repair')
    (args.out/'COMPLETE.json').write_text(json.dumps({'rows':len(result_rows),'test_used':False,
          'framework_forced_routes':0,'new_semantic_evidence_reviews':0,'grpo_started':False},indent=2))


if __name__=='__main__':main()
