"""Full frozen Dev comparison with explicit native Student-error accounting.

Same weak-reference metrics as v1; unresolved and invalid outcomes stay in all
frozen cohort denominators. No semantic support is invented or inferred from labels.
"""
import json
from pathlib import Path

from full_dev_control_metrics import collect, metrics
from hf_frozen_dev_policy import read_dev_item
from request_policy_replay import sha
from student_action_outcomes import classify_recorded_failure
from recorded_rollout_recovery import recover_case
from frozen_dev_manifest import load_frozen_dev, runtime_path
from full_dev_accounted_metrics import validate_role_runtime as validate_legacy_runtime


def load(path):return json.loads(path.read_text())


def validate_role_runtime(root, rows, frozen):
    current=sha(runtime_path(frozen))
    if any(sha(Path(path))!=expected for path,expected in load(runtime_path(frozen)).items()):
        raise ValueError('Frozen runtime source changed')
    header=load(root/'RUNTIME_BINDING.json')
    if header['source_binding_sha256']!=current:
        raise ValueError('Accounted Dev role runtime differs')
    reuse_path=root/'REUSED_NATIVE_RECEIPTS.json'
    if header.get('predecessor_manifest_sha256')!=(sha(reuse_path) if reuse_path.exists() else None):
        raise ValueError('Predecessor reuse manifest changed')
    reuse_record=load(reuse_path) if reuse_path.exists() else {}
    reused=reuse_record.get('cases',{})
    old_rows={}
    _,entries,_=load_frozen_dev(frozen)
    manifest={r['source']['merged_id']:dict(r,sample=0) for r in entries}
    binding=load(root/'PLAN.json')['model_binding']
    for ident,row in rows.items():
        receipt=root/'rollouts'/ident/'receipt.json'
        if row.get('execution_runtime_binding_sha256')!=current:
            if ident not in reused or sha(receipt)!=reused[ident]['output_receipt_sha256']:
                raise ValueError('Unbound predecessor Dev receipt')
            predecessor=reused[ident]
            if sha(Path(predecessor['source_artifact']))!=predecessor['source_artifact_sha256']:
                raise ValueError('Predecessor Dev artifact changed')
            old_rows[ident]=row
        _,_,response=read_dev_item(Path(row['requests'][-1]['item_path']))
        native_case=Path(row['requests'][-1]['item_path']).parent.parent.parent
        actual=recover_case(native_case,manifest[ident],binding,read_dev_item,'dev')
        for field in ['status','answer','tool_calls','source_id','source_partition','actual_tools']:
            if actual.get(field)!=row.get(field):raise ValueError('Dev outcome/cost differs from native trace: '+field)
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
    if old_rows:
        cert=Path(reuse_record['compatibility_certificate'])
        validation=Path(reuse_record['validation_receipt'])
        if sha(cert)!=reuse_record['compatibility_certificate_sha256'] or sha(validation)!=reuse_record['validation_receipt_sha256']:
            raise ValueError('Cohort reuse compatibility or validation proof changed')
        certificate=load(cert)
        if any(sha(Path(p))!=h for p,h in certificate['dependencies'].items()):
            raise ValueError('Cohort reuse generation compatibility dependencies changed')
        source=Path(reuse_record['source_role']);old_frozen=Path(reuse_record['source_frozen_root'])
        if (certificate['target_manifest_sha256']!=load(frozen/'PLAN.json')['manifest_sha256']
                or sha(source/'PLAN.json')!=reuse_record['source_role_plan_sha256']
                or sha(source/'RUNTIME_BINDING.json')!=reuse_record['source_role_runtime_binding_sha256']):
            raise ValueError('Cohort reuse source/target binding changed')
        approved=load(validation)['receipt_hashes']
        for ident,row in old_rows.items():
            original=source/'rollouts'/ident/'receipt.json'
            if sha(original)!=approved.get(str(original.resolve())) or load(original)!=row:
                raise ValueError('Reused native receipt was not approved by original compatibility validation')
        validate_legacy_runtime(source,old_rows,old_frozen)


def compare(frozen,baseline,candidate):
    plan,manifest,expected_count=load_frozen_dev(frozen)
    if sha(frozen/'WEAK_REFERENCES.json')!=plan['weak_reference_sha256']:
        raise ValueError('Weak references changed')
    expected={r['source']['merged_id'] for r in manifest}
    for role in [baseline,candidate]:
        if (role/'PLAN.json').exists():
            contract=load(role/'PLAN.json')
            if contract['manifest_sha256']!=plan['manifest_sha256']:
                raise ValueError('Role belongs to a different frozen cohort; explicit validated migration required')
    references={r['source_id']:r for r in load(frozen/'WEAK_REFERENCES.json')}
    if set(references)!=expected:raise ValueError('Reference coverage differs')
    left,missing_left,left_prompt=collect(baseline,manifest)
    right,missing_right,right_prompt=collect(candidate,manifest)
    if left:validate_role_runtime(baseline,left,frozen)
    if right:validate_role_runtime(candidate,right,frozen)
    if missing_left or missing_right:
        return {'status':'INCOMPLETE_NOT_MAIN_CONTROL_RESULT','expected_per_role':expected_count,
            'completed':{'baseline':len(left),'candidate':len(right)},
            'missing':{'baseline':missing_left,'candidate':missing_right},'best_candidate_not_selected':True}
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
        'count_per_role':expected_count,'metrics':results,'same_initial_pixel_token_tensors_verified':True,
        'student_failures_not_resampled':True,'technical_failures_not_semantic_negatives':True,
        'semantic_primary_evaluation_complete':False,'best_candidate_not_selected':True,
        'optimizer_steps_in_evaluation':0,'test_used':False,'weak_reference_only':True,
        'reason':'Full inference and weak-label agreement do not replace evidence correctness review'}
