"""Full frozen Dev comparison with explicit native Student-error accounting.

Same weak-reference metrics as v1; unresolved and invalid outcomes stay in all
1679 denominators. No semantic support is invented or inferred from labels.
"""
import json
from pathlib import Path

from full_dev_control_metrics import collect, metrics
from hf_frozen_dev_policy import read_dev_item
from request_policy_replay import sha
from student_action_outcomes import classify_recorded_failure
from recorded_rollout_recovery import recover_case


def load(path):return json.loads(path.read_text())


def validate_role_runtime(root, rows, frozen):
    current=sha(frozen/'RUNTIME_BINDINGS_v3.json')
    header=load(root/'RUNTIME_BINDING.json')
    if header['source_binding_sha256']!=current:
        raise ValueError('Accounted Dev role runtime differs')
    reuse_path=root/'REUSED_NATIVE_RECEIPTS.json'
    if header.get('predecessor_manifest_sha256')!=(sha(reuse_path) if reuse_path.exists() else None):
        raise ValueError('Predecessor reuse manifest changed')
    reused=load(reuse_path)['cases'] if reuse_path.exists() else {}
    manifest={r['source']['merged_id']:dict(r,sample=0) for r in load(frozen/'MANIFEST1679.json')}
    binding=load(root/'PLAN.json')['model_binding']
    for ident,row in rows.items():
        receipt=root/'rollouts'/ident/'receipt.json'
        if row.get('execution_runtime_binding_sha256')!=current:
            if ident not in reused or sha(receipt)!=reused[ident]['output_receipt_sha256']:
                raise ValueError('Unbound predecessor Dev receipt')
            predecessor=reused[ident]
            if sha(Path(predecessor['source_artifact']))!=predecessor['source_artifact_sha256']:
                raise ValueError('Predecessor Dev artifact changed')
        _,_,response=read_dev_item(Path(row['requests'][-1]['item_path']))
        native_case=Path(row['requests'][-1]['item_path']).parent.parent.parent
        actual=recover_case(native_case,manifest[ident],binding,read_dev_item,'dev')
        for field in ['status','answer','tool_calls','source_id','source_partition','actual_tools']:
            if actual[field]!=row[field]:raise ValueError('Dev outcome/cost differs from native trace: '+field)
        if row['status']=='MODEL_ANSWER_UNREVIEWED':
            if row['answer']!=json.loads(response['decoded_text_for_inspection_only']):
                raise ValueError('Dev answer differs from actual last completion')
        elif row['status']=='INVALID_STUDENT_ACTION':
            loop=Path(row['requests'][-1]['item_path']).parent.parent
            proof=classify_recorded_failure(loop,row['requests'],read_dev_item)
            if proof is None or proof['failure_reason']!=row['failure_reason'] or row['answer'] is not None:
                raise ValueError('Student failure cannot be proven from native Dev records')
        elif row['status']!='BUDGET_UNRESOLVED':
            raise ValueError('Unrecognised Dev outcome; no technical-failure-as-negative fallback')


def compare(frozen,baseline,candidate):
    plan=load(frozen/'PLAN.json');manifest=load(frozen/'MANIFEST1679.json')
    if len(manifest)!=1679 or sha(frozen/'MANIFEST1679.json')!=plan['manifest_sha256']:
        raise ValueError('Full original Dev1679 required')
    if sha(frozen/'WEAK_REFERENCES.json')!=plan['weak_reference_sha256']:
        raise ValueError('Weak references changed')
    expected={r['source']['merged_id'] for r in manifest}
    if len(expected)!=1679 or any(r['source']['frozen_split']!='dev' for r in manifest):
        raise ValueError('Dev identity/partition changed')
    references={r['source_id']:r for r in load(frozen/'WEAK_REFERENCES.json')}
    if set(references)!=expected:raise ValueError('Reference coverage differs')
    left,missing_left,left_prompt=collect(baseline,manifest)
    right,missing_right,right_prompt=collect(candidate,manifest)
    if missing_left or missing_right:
        return {'status':'INCOMPLETE_NOT_MAIN_CONTROL_RESULT','expected_per_role':1679,
            'completed':{'baseline':len(left),'candidate':len(right)},
            'missing':{'baseline':missing_left,'candidate':missing_right},'best_candidate_not_selected':True}
    validate_role_runtime(baseline,left,frozen);validate_role_runtime(candidate,right,frozen)
    for ident in expected:
        if left_prompt[ident]!=right_prompt[ident]:raise ValueError('Initial actual pixel/token tensors differ')
    contracts=[load(root/'PLAN.json') for root in [baseline,candidate]]
    for field in ['manifest_sha256','policy_sha256','budget','sampling','environment','version']:
        if contracts[0][field]!=contracts[1][field]:raise ValueError('Paired accounted evaluation differs: '+field)
    results={}
    for role,rows in [('baseline',left),('candidate',right)]:
        results[role]=metrics(rows,references)
        results[role]['invalid_student_actions']=sum(row['status']=='INVALID_STUDENT_ACTION' for row in rows.values())
        results[role]['failure_reason_counts']={reason:sum(row.get('failure_reason')==reason for row in rows.values())
            for reason in sorted({r['failure_reason'] for r in rows.values() if r.get('failure_reason')})}
    return {'status':'FULL_DEV_ACCOUNTED_OPERATIONAL_AND_WEAK_REFERENCE_CONTROL_COMPLETE',
        'count_per_role':1679,'metrics':results,'same_initial_pixel_token_tensors_verified':True,
        'student_failures_not_resampled':True,'technical_failures_not_semantic_negatives':True,
        'semantic_primary_evaluation_complete':False,'best_candidate_not_selected':True,
        'optimizer_steps_in_evaluation':0,'test_used':False,'weak_reference_only':True,
        'reason':'Full inference and weak-label agreement do not replace evidence correctness review'}
