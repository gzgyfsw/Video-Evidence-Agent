"""Account verifiable Student failures without repairing or resampling outputs.

Valid trajectories use the original loop unchanged. Tool/transport/ledger/model
binding errors still raise. A Student-failure record is unresolved, never ALLOW
or a semantic judgment, and introduces no new reward term.
"""
import ast
import json
from pathlib import Path

from request_policy_replay import sha
from unified_student_contract import dumps, public_context
from unified_student_loop import run, validate_execution

PARAMETER_ERRORS = {
    'Incomplete search','Invalid risk map','Invalid queries','Incomplete expansion',
    'Invalid timestamps','Invalid window','Invalid frame count','Invalid answer fields',
    'Invalid category','Invalid operational fields','Unaudited REVIEW reason cannot be inferred',
    'No visible witness','Invalid evidence','Unseen citation','Unknown action','Budget exhausted',
    'Frame budget exceeded','Expansion anchor was not observed or located',
    'L0 requires Harmless/ALLOW','Invalid frozen severity/disposition tuple',
    'Unresolved output cannot be converted to ALLOW/BLOCK',
}


def recorded_message(value):
    for prefix in ['ValueError(', 'JSONDecodeError(']:
        if value.startswith(prefix) and value.endswith(')'):
            message=ast.literal_eval(value[len(prefix):-1])
            return message if isinstance(message,str) else None
    return None


def classify_recorded_failure(loop, calls, reader):
    """Return an outcome only when native records prove a Student-side error."""
    loop=Path(loop);failure_path=loop/'failure.json'
    if not calls or not failure_path.exists():return None
    failure=json.loads(failure_path.read_text());message=recorded_message(failure['error'])
    if message is None:return None
    bound=[]
    for call in calls:
        path=Path(call['item_path']);item,request,response=reader(path)
        if request['context']!=public_context(request['context']):
            raise ValueError('Recorded failure has private/unbound actor context')
        for image in request['images']:
            if sha(image['path'])!=image['sha256']:raise ValueError('Recorded failure pixels changed')
        if bound and (item['model_binding']!=bound[0][0]['model_binding']
                      or item['source_trajectory_id']!=bound[0][0]['source_trajectory_id']):
            raise ValueError('Failure request changed Student or source')
        bound.append((item,request,response,path))
    item,request,response,path=bound[-1]
    # A tool failure occurs after appending its attempted decision. A Student
    # validation/duplicate failure leaves exactly the request's input prefix.
    context=dict(request['context'],events=failure['events'])
    if public_context(context)!=request['context']:return None
    step=path.parent.name;context_path=loop/(step+'-context.json')
    logged=json.loads(context_path.read_text())
    if public_context(logged['context'])!=request['context'] or logged['images']!=request['images']:
        raise ValueError('Failure is not bound to the actual logged request prefix')
    text=response['decoded_text_for_inspection_only']
    try:value=json.loads(text)
    except json.JSONDecodeError:value=None
    reason=None
    if message=='Duplicate tool call; no silent replacement' and value is not None:
        validate_execution(value,request['context'],request['images'])
        saved_output=loop/(step+'-output.json')
        if not saved_output.exists() or json.loads(saved_output.read_text())!=value:return None
        if (loop/(step+'-tool-return.json')).exists():return None
        for event in request['context']['events']:
            if event['type']!='decision':continue
            prior={'action':event['action'],**event['parameters']}
            completed=any(e['type']=='tool_result' and e.get('call_step_id')==event['step_id']
                          for e in request['context']['events'])
            recorded_return=loop/(event['step_id']+'-tool-return.json')
            if completed and recorded_return.exists() and dumps(prior)==dumps(value):
                reason='DUPLICATE_TOOL_CALL';break
    elif message in PARAMETER_ERRORS and value is not None:
        try:validate_execution(value,request['context'],request['images'])
        except ValueError as exc:
            if str(exc)==message:reason='INVALID_ACTION_PARAMETERS'
    elif message=='Incomplete model JSON; saved output is not repaired' and response['finish_reason']!='stop':
        reason='INCOMPLETE_MODEL_OUTPUT'
    elif failure['error'].startswith('JSONDecodeError(') and value is None:
        reason='UNPARSEABLE_MODEL_OUTPUT'
    if reason is None:return None
    return {'status':'INVALID_STUDENT_ACTION','answer':None,'model':request['model'],
        'tool_calls':sum(e['type']=='tool_result' for e in request['context']['events']),
        'decisions':len(calls),'failure_reason':reason,'failed_output':value if value is not None else text,
        'failed_request_item':str(path.resolve()),'original_failure_path':str(failure_path.resolve()),
        'original_failure_sha256':sha(failure_path),'semantic_support':'NOT_VERIFIED_BY_EXECUTION',
        'unresolved':True,'resampled_or_repaired':False,'business_disposition_generated':False}


def run_accounted(student,duration_seconds,budget,initial_images,execute,out,max_decisions=None,*,reader):
    try:
        return run(student,duration_seconds,budget,initial_images,execute,out,max_decisions=max_decisions)
    except (ValueError,json.JSONDecodeError):
        result=classify_recorded_failure(Path(out),student.calls,reader)
        if result is None:raise
        path=Path(out)/'accounted_student_failure.json'
        with path.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
        return result
