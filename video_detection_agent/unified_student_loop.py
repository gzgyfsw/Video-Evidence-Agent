"""Explicit single-Student loop; importing/exposing it never starts an experiment.

The injected executor owns real search/expansion and must return:
  (result_record, new_frozen_image_manifests).
No model can supply tool results. No adapters, policy heads or observers switch.
The old SplitBrain loop remains an independent baseline.
"""
import copy
import json
import math
from pathlib import Path
from unified_student_contract import dumps, validate_output, UNLIMITED_BUDGET


def validate_execution(value, context, images):
    validate_output(value,context,images)
    if any(type(im['timestamp']) not in {int,float} or not math.isfinite(im['timestamp']) or
           not 0 <= im['timestamp'] <= context['duration_seconds'] for im in images):
        raise ValueError('Invalid visible image timestamp')
    if value['action']=='TEMPORAL_EXPAND':
        known=[im['timestamp'] for im in images]
        for event in context['events']:
            if event['type']=='tool_result':
                known += [c['timestamp'] for c in event['result'].get('candidates',[])]
        if not all(any(abs(t-k)<=.0006 for k in known) for t in value['timestamps']):
            raise ValueError('Expansion anchor was not observed or located')
    if value['action']=='ANSWER':
        category,level,disposition=(value[k] for k in ['category','severity','moderation_action'])
        # Frozen legacy operational consistency; never fill or repair any field.
        if level=='L0' and (category!='Harmless' or disposition!='ALLOW'):
            raise ValueError('L0 requires Harmless/ALLOW')
        if level in {'L1','L2'} and (category in {'Harmless',None} or disposition!={'L1':'REVIEW','L2':'BLOCK'}[level]):
            raise ValueError('Invalid frozen severity/disposition tuple')
        if level is None and (disposition!='REVIEW' or value.get('review_reason')!='EVIDENCE_INSUFFICIENT'):
            raise ValueError('Unresolved output cannot be converted to ALLOW/BLOCK')


def run(student, duration_seconds, budget, initial_images, execute, out, max_decisions=None):
    # None means uncapped. Explicit historical budgets remain replayable.
    budget = copy.deepcopy(UNLIMITED_BUDGET if budget is None else budget)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    images=copy.deepcopy(initial_images)
    if not images or len({im['id'] for im in images})!=len(images):raise ValueError('Invalid initial images')
    events=[{'step_id':'initial','type':'observe','frame_ids':[im['id'] for im in images]}]
    configured_model=student.model;calls=0;seen_signatures=set()
    frame_limit = budget.get('max_frames')
    if frame_limit is not None and len(images)>frame_limit:raise ValueError('Initial frame budget exceeded')
    try:
        turn = 0
        while max_decisions is None or turn < max_decisions:
            turn += 1
            if student.model!=configured_model:raise ValueError('Student Adapter/model changed')
            context={'duration_seconds':duration_seconds,'budget':copy.deepcopy(budget),
                'events':copy.deepcopy(events),'visible_frame_ids':[im['id'] for im in images]}
            step_id=f'turn-{turn}'; step=out/step_id
            (out/f'{step_id}-context.json').write_text(json.dumps({'context':context,'images':images},ensure_ascii=False,indent=2))
            value=student.next_step(context,images,step)
            if student.model!=configured_model:raise ValueError('Student Adapter/model changed')
            validate_execution(value,context,images)
            (out/f'{step_id}-output.json').write_text(json.dumps(value,ensure_ascii=False,indent=2))
            if value['action']=='ANSWER':
                result={'status':'MODEL_ANSWER_UNREVIEWED','answer':value,'model':configured_model,
                    'tool_calls':calls,'decisions':turn,'semantic_support':'NOT_VERIFIED_BY_EXECUTION'}
                (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));return result
            signature=dumps(value)
            if signature in seen_signatures:raise ValueError('Duplicate tool call; no silent replacement')
            seen_signatures.add(signature)
            params={k:v for k,v in value.items() if k!='action'}
            events.append({'step_id':step_id,'type':'decision','action':value['action'],'parameters':params,'claims':[]})
            record,new_images=execute(value,copy.deepcopy(context),copy.deepcopy(images))
            # Persist actual return before validation, including failed attempts.
            (out/f'{step_id}-tool-return.json').write_text(json.dumps({'record':record,'images':new_images},ensure_ascii=False,indent=2))
            if record.get('error'):raise ValueError('Tool technical failure: '+str(record['error']))
            if set(record)-{'frame_ids','candidates','original_tool_record'}:raise ValueError('Unknown tool return fields')
            new_ids=[im['id'] for im in new_images]
            if record.get('frame_ids')!=new_ids or len(set(new_ids))!=len(new_ids) or set(new_ids)&{im['id'] for im in images}:
                raise ValueError('Tool image ledger mismatch')
            if frame_limit is not None and len(images)+len(new_images)>frame_limit:raise ValueError('Tool exceeded frame budget')
            calls+=1; images+=new_images
            # Match the time-ordered visible layout recorded by the old audits.
            images.sort(key=lambda im:(im['timestamp'],im['id']))
            events.append({'step_id':step_id+'-result','type':'tool_result','call_step_id':step_id,
                'tool':value['action'],'result':record,'frame_ids':new_ids})
        result={'status':'BUDGET_UNRESOLVED','model':configured_model,'tool_calls':calls,
            'successful_risk_discovery':False,'answer':None}
        (out/'result.json').write_text(json.dumps(result,indent=2));return result
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps({'status':'TECHNICAL_OR_INTERFACE_FAILURE',
            'error':repr(exc),'semantic_judgment':None,'events':events},ensure_ascii=False,indent=2))
        raise
