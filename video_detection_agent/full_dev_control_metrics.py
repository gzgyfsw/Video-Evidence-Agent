"""Strict paired full-Dev coverage, causal input checks and weak-label metrics.

No model judgments are invented. Unreviewed evidence stays unreviewed; weak
reference agreement is auxiliary and never relabelled as true accuracy/FPR.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from hf_frozen_dev_policy import read_dev_item
from request_policy_replay import sha


def load(p):return json.loads(Path(p).read_text())


def collect(root,manifest):
    rows={};missing=[];prompt_receipts={}
    if not (root/'PLAN.json').exists():
        if any(root.glob('rollouts/*/receipt.json')):raise ValueError('Unbound Dev receipts without role contract')
        return {},[r['source']['merged_id'] for r in manifest],{}
    contract=load(root/'PLAN.json')
    for entry in manifest:
        ident=entry['source']['merged_id'];path=root/'rollouts'/ident/'receipt.json'
        if not path.exists():missing.append(ident);continue
        row=load(path)
        if (row['source_id']!=ident or row['seed']!=entry['seed'] or row['source_partition']!='dev' or row['model_binding']!=contract['model_binding']):
            raise ValueError('Paired Dev identity/seed/partition changed')
        if not row.get('requests'):raise ValueError('Dev receipt has no actual Student request')
        from hf_request_rng import request_seed
        from unified_student_contract import public_context
        for index,call in enumerate(row['requests']):
            bound,actual,answer=read_dev_item(call['item_path'])
            if (bound['model_binding']!=contract['model_binding'] or bound['source_trajectory_id']!=ident
                    or actual['seed']!=request_seed(entry['seed'],index)
                    or actual['policy_sha256']!=contract['policy_sha256']
                    or actual['context']!=public_context(actual['context'])):
                raise ValueError('Dev actual request source/model/policy/RNG/context differs')
            for im in actual['images']:
                if sha(im['path'])!=im['sha256']:raise ValueError('Frozen Dev pixels changed')
        item,request,response=read_dev_item(row['requests'][0]['item_path'])
        prompt_receipts[ident]=response['prompt_tensor_receipt']
        rows[ident]=row
    return rows,missing,prompt_receipts


def metrics(rows,references):
    count=len(references);confusion=Counter();answer_count=0;unresolved=0;tools=0;frames=0;harmful_reports_on_weak_normal=0;normal=0;resolved_agreement=0;resolved_risk_on_normal=0
    for ident,ref in references.items():
        row=rows[ident];answer=row.get('answer') or {};category=answer.get('category')
        if row['status']!='MODEL_ANSWER_UNREVIEWED':category=None
        if row['status']=='MODEL_ANSWER_UNREVIEWED':answer_count+=1
        is_unresolved=answer.get('review_reason')=='EVIDENCE_INSUFFICIENT' or category is None
        if is_unresolved:unresolved+=1
        if not is_unresolved and category==ref['weak_category']:resolved_agreement+=1
        confusion[(ref['weak_category'],category)]+=1
        tools+=len(row.get('actual_tools',[]))
        frames+=sum(len(t['returned_timestamps']) for t in row.get('actual_tools',[]))
        if ref['weak_category']=='Harmless':
            normal+=1
            if category not in {None,'Harmless'}:
                harmful_reports_on_weak_normal+=1
                if not is_unresolved:resolved_risk_on_normal+=1
    agreement=sum(v for (a,b),v in confusion.items() if a==b)
    per={}
    for c in sorted({r['weak_category'] for r in references.values()}):
        tp=confusion[(c,c)];fn=sum(v for (a,b),v in confusion.items() if a==c and b!=c);fp=sum(v for (a,b),v in confusion.items() if b==c and a!=c)
        per[c]={'count':tp+fn,'agreement':tp/(tp+fn),'weak_reference_F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    return {'denominator_all_frozen_dev':count,'weak_category_agreement':agreement/count,
            'resolved_weak_category_agreement_over_all_dev':resolved_agreement/count,
            'raw_category_agreement_includes_unresolved_guesses':True,
            'resolved_risk_reports_on_weak_harmless_reference':resolved_risk_on_normal,
            'weak_category_macro_F1':sum(v['weak_reference_F1'] for v in per.values())/len(per),
            'per_weak_category':per,'terminal_answers':answer_count,'unresolved':unresolved,
            'mean_tool_calls':tools/count,'mean_new_frames':frames/count,
            'weak_harmless_reference_count':normal,'risk_reports_on_weak_harmless_reference':harmful_reports_on_weak_normal,
            'true_harmless_FPR':None,'supported_risk_evidence_rate':None,'supported_report_rate':None,
            'semantic_audit_status':'NOT_FULLY_AUDITED','no_semantic_credit_for_unreviewed_evidence':True}


def compare(frozen,baseline,candidate):
    plan=load(frozen/'PLAN.json');manifest=load(frozen/'MANIFEST1679.json')
    if len(manifest)!=1679 or sha(frozen/'MANIFEST1679.json')!=plan['manifest_sha256']:
        raise ValueError('Need unchanged full original Dev1679')
    if sha(frozen/'WEAK_REFERENCES.json')!=plan['weak_reference_sha256']:raise ValueError('Frozen weak references changed')
    if len({r['source']['merged_id'] for r in manifest})!=1679 or any(r['source'].get('frozen_split')!='dev' for r in manifest):
        raise ValueError('Full Dev identity coverage or partition mismatch')
    references={r['source_id']:r for r in load(frozen/'WEAK_REFERENCES.json')}
    if set(references)!={r['source']['merged_id'] for r in manifest}:raise ValueError('Weak references omit Dev identities')
    left,missing_left,left_prompt=collect(baseline,manifest);right,missing_right,right_prompt=collect(candidate,manifest)
    if missing_left or missing_right:
        return {'status':'INCOMPLETE_NOT_MAIN_CONTROL_RESULT','expected_per_role':1679,
                'completed':{'baseline':len(left),'candidate':len(right)},
                'missing':{'baseline':missing_left,'candidate':missing_right},'best_candidate_not_selected':True}
    for ident in references:
        if left_prompt[ident]!=right_prompt[ident]:raise ValueError('Initial actual pixel/token tensors differ between roles: '+ident)
    runtime_hashes=[load(root/'RUNTIME_BINDING.json')['source_binding_sha256'] for root in [baseline,candidate]]
    if runtime_hashes[0]!=runtime_hashes[1] or runtime_hashes[0]!=sha(frozen/'RUNTIME_BINDINGS_v2.json'):
        raise ValueError('Paired full Dev inference implementation differs')
    contracts=[load(root/'PLAN.json') for root in [baseline,candidate]]
    for field in ['manifest_sha256','policy_sha256','budget','sampling','environment']:
        if contracts[0][field]!=contracts[1][field]:raise ValueError('Paired evaluation condition differs: '+field)
    return {'status':'FULL_DEV_OPERATIONAL_AND_WEAK_REFERENCE_CONTROL_COMPLETE',
            'count_per_role':1679,'same_initial_pixel_token_tensors_verified':True,
            'metrics':{'baseline':metrics(left,references),'candidate':metrics(right,references)},
            'test_used':False,'optimizer_steps_in_evaluation':0,'weak_reference_only':True,
            'semantic_primary_evaluation_complete':False,'best_candidate_not_selected':True,
            'reason':'Full inference and weak-label metrics do not replace evidence correctness review'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['frozen','baseline','candidate','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=compare(a.frozen,a.baseline,a.candidate)
    with a.out.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(result['status'])


if __name__=='__main__':main()
