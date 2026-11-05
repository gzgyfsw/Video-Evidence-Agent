"""One Student, shared train/inference content. Importing never calls a model."""
import base64
import hashlib
import io
import json
import math
import copy
from pathlib import Path
from PIL import Image

VERSION = 'unified-student-next-step-v1'
POLICY_SHA = '45fb654deceb9edea3681dc87f2d352e164495b6a28165d84a27afd332bd6040'
INSTRUCTION = Path(__file__).with_name('prompts') / 'unified_student_v1.txt'
MAX_PIXELS = 409600
THINKING = False
PUBLIC_CONTEXT_VERSION = 'public-tool-results-v1'
UNLIMITED_BUDGET = {'max_tool_calls': None, 'max_frames': None}


def remaining_frames(budget, count):
    limit = budget.get('max_frames')
    return None if limit is None else max(0, limit - count)


def budget_exhausted(budget, calls, frames):
    tool_limit, frame_limit = budget.get('max_tool_calls'), budget.get('max_frames')
    return ((tool_limit is not None and calls >= tool_limit)
            or (frame_limit is not None and frames >= frame_limit))


def public_context(context):
    """Keep actual observations/history, exclude storage/audit provenance.

    Shared by SFT content and inference. Full tool receipts remain in the
    separate trajectory ledger; source paths may reveal dataset labels.
    """
    result = copy.deepcopy(context)
    for event in result['events']:
        if event['type'] == 'tool_result':
            returned = event['result']
            event['result'] = {k: v for k, v in returned.items() if k in {'frame_ids', 'candidates'}}
            if 'candidates' in event['result']:
                event['result']['candidates'] = [{k: v for k, v in c.items()
                    if k in {'timestamp', 'score', 'matched_query', 'risk_type'}}
                    for c in event['result']['candidates']]
    return result


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt(context, policy):
    if set(context) != {'duration_seconds', 'budget', 'events', 'visible_frame_ids'}:
        raise ValueError('Unknown context field; opaque trajectories are forbidden')
    if hashlib.sha256(policy.encode()).hexdigest() != POLICY_SHA:
        raise ValueError('Frozen policy changed')
    instruction = INSTRUCTION.read_text()
    if all(context['budget'].get(key) is None for key in UNLIMITED_BUDGET):
        instruction += '\n本次运行不设累计帧数、工具调用次数或决策轮数的成本上限。budget 中 null 表示无上限。根据证据缺口继续取证，根据证据是否充分决定结束，不要因为成本预算提前停止。\n'
    return instruction + '\n政策：\n' + policy + '\n此前已发生的交互：\n' + dumps(public_context(context))


def validate_output(v, context, images):
    """Check executability only, never assign semantic approval or new labels."""
    action = v.get('action')
    if action == 'JOINT_SEMANTIC_SCAN':
        if set(v) != {'action', 'queries_by_risk'}: raise ValueError('Incomplete search')
        q = v['queries_by_risk']
        if not isinstance(q, dict) or not q or set(q) - {'Sexual Content','Violence & Dangerous','Sensitive Figures & Ideologies'}:
            raise ValueError('Invalid risk map')
        if any(not isinstance(x,list) or not 1 <= len(x) <= 4 or any(not isinstance(t,str) or not t.strip() for t in x) for x in q.values()):
            raise ValueError('Invalid queries')
    elif action == 'TEMPORAL_EXPAND':
        if set(v) != {'action','timestamps','window','num_frames'}: raise ValueError('Incomplete expansion')
        if not isinstance(v['timestamps'],list) or not 1 <= len(v['timestamps']) <= 3 or any(type(t) not in {int,float} or not math.isfinite(t) or not 0 <= t <= context['duration_seconds'] for t in v['timestamps']):
            raise ValueError('Invalid timestamps')
        if type(v['window']) not in {int,float} or not .5 <= v['window'] <= 15: raise ValueError('Invalid window')
        if (type(v['num_frames']) is not int or v['num_frames'] < 1
                or (context['budget'].get('max_frames') is not None and v['num_frames'] > 8)):
            raise ValueError('Invalid frame count')
    elif action == 'ANSWER':
        required = {'action','category','severity','moderation_action','evidence'}
        if not required <= set(v) or set(v) - required - {'rationale','review_reason'}: raise ValueError('Invalid answer fields')
        if v['category'] not in {'Harmless','Sexual Content','Violence & Dangerous','Sensitive Figures & Ideologies',None}: raise ValueError('Invalid category')
        if v['severity'] not in {'L0','L1','L2',None} or v['moderation_action'] not in {'ALLOW','REVIEW','BLOCK'}: raise ValueError('Invalid operational fields')
        if v['moderation_action'] == 'REVIEW' and v.get('review_reason') not in {'CONFIRMED_L1','EVIDENCE_INSUFFICIENT'}: raise ValueError('Unaudited REVIEW reason cannot be inferred')
        if not isinstance(v['evidence'],list) or not v['evidence']: raise ValueError('No visible witness')
        for item in v['evidence']:
            if set(item) != {'timestamp','description'} or not isinstance(item['description'],str) or not item['description'].strip(): raise ValueError('Invalid evidence')
            if type(item['timestamp']) not in {int,float} or not any(abs(item['timestamp']-im['timestamp']) <= .001 for im in images): raise ValueError('Unseen citation')
    else: raise ValueError('Unknown action')
    if action != 'ANSWER':
        used = sum(e['type']=='decision' and e.get('action')!='ANSWER' for e in context['events'])
        if budget_exhausted(context['budget'], used, len(images)): raise ValueError('Budget exhausted')
        remaining = remaining_frames(context['budget'], len(images))
        if action=='TEMPORAL_EXPAND' and remaining is not None and v['num_frames'] > remaining: raise ValueError('Frame budget exceeded')


def content(context, policy, manifest, images=None, transport=False):
    if context['visible_frame_ids'] != [im['id'] for im in manifest] or not manifest: raise ValueError('Invalid visible image order')
    blocks = [{'type':'text','text':prompt(context,policy)}]
    for i,item in enumerate(manifest):
        path = Path(item['path'])
        if sha(path) != item['sha256']: raise ValueError('Audited pixels changed')
        with Image.open(path) as image:
            fmt = image.format
            if image.width*image.height > MAX_PIXELS: raise ValueError('Would resize audited pixels')
            rgb = image.convert('RGB')
        blocks.append({'type':'text','text':f"IMAGE {i}: id={item['id']}; timestamp={dumps(item['timestamp'])} seconds."})
        if transport:
            if fmt != 'JPEG': raise ValueError('Only frozen JPEG transport supported')
            blocks.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode()}})
        else:
            if images is not None and images[i].convert('RGB').tobytes() != rgb.tobytes(): raise ValueError('Training pixels differ')
            blocks.append({'type':'image','image':rgb if images is None else images[i]})
    return blocks


def freeze_runtime_image(image, timestamp, frame_id, out):
    """Freeze each new real frame once; training and inference reuse this JPEG."""
    image = image.convert('RGB')
    if image.width*image.height > MAX_PIXELS:
        scale = (MAX_PIXELS/(image.width*image.height))**.5
        image = image.resize((max(1,int(image.width*scale)),max(1,int(image.height*scale))))
    buf=io.BytesIO(); image.save(buf,format='JPEG',quality=95)
    data=buf.getvalue(); digest=hashlib.sha256(data).hexdigest()
    folder=Path(out); folder.mkdir(parents=True,exist_ok=True); path=folder/(digest+'.jpg')
    if path.exists() and path.read_bytes()!=data: raise ValueError('Hash collision')
    if not path.exists(): path.write_bytes(data)
    return {'id':frame_id,'timestamp':timestamp,'path':str(path.resolve()),'sha256':digest}


class UnifiedStudent:
    """Single model alias, no phase-specific Adapter. Real tools remain caller-owned.

    This next-step interface is not automatically connected to the legacy loop.
    Caller must append actual tool records and freeze new frames before retrying.
    """
    def __init__(self, client, model, processor, policy):
        self.client,self.model,self.processor,self.policy=client,model,processor,policy

    def prepare(self, context, manifest):
        local=[{'role':'user','content':content(context,self.policy,manifest)}]
        text=self.processor.apply_chat_template(local,tokenize=False,add_generation_prompt=True,enable_thinking=THINKING)
        images=[b['image'] for b in local[0]['content'] if b['type']=='image']
        ids=self.processor(text=[text],images=images,return_tensors='pt')['input_ids'][0].tolist()
        return {'model':self.model,'messages':[{'role':'user','content':content(context,self.policy,manifest,transport=True)}],
                'temperature':0,'max_tokens':1024,'extra_body':{
                    'chat_template_kwargs':{'enable_thinking':THINKING},'return_token_ids':True}},ids

    def next_step(self, context, manifest, out):
        out=Path(out); out.mkdir(parents=True,exist_ok=False)
        request,expected=self.prepare(context,manifest)
        (out/'request.json').write_text(json.dumps(request,ensure_ascii=False))
        (out/'expected_prompt_token_ids.json').write_text(json.dumps(expected))
        response=self.client.chat.completions.create(**request).model_dump()
        (out/'raw_response.json').write_text(json.dumps(response,ensure_ascii=False))
        if response.get('prompt_token_ids')!=expected: raise ValueError('Native train/inference prefix mismatch; do not execute')
        if response['choices'][0]['finish_reason']!='stop': raise ValueError('Incomplete output')
        value=json.loads(response['choices'][0]['message']['content'])
        validate_output(value,context,manifest)
        return value
