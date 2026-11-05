"""Freeze trace-bound semantic reward components; no optimizer or model calls."""
import argparse
import json
from pathlib import Path
from prepare_unified_student_sft import covered, resolve
from review_hf_acquisition import digest
from request_policy_replay import sha
from hf_request_policy import read_bound_item
from unified_student_contract import POLICY_SHA


def group_signal(group):
    semantic=len({r['semantic_score'] for r in group})>1
    reward_var=len({r['total'] for r in group})>1
    penalty_var=len({r['confirmed_fabrication_penalty'] for r in group})>1
    all_reports_supported=bool(group) and all(r['report_supported'] and r['acquisition_supported'] for r in group)
    return {'samples':len(group),'semantic_variation':semantic,'reward_variation':reward_var,
        'confirmed_penalty_variation':penalty_var,'all_reports_supported':all_reports_supported,
        'eligible_for_update':len(group)==4 and reward_var and (semantic or penalty_var or all_reports_supported),
        'identical_score_group_retained':True,
        'cost_only_unknown_groups_excluded':True}


def report_supported(candidate, original, result, contradiction_reviews=None):
    if candidate.get('technical_failures') or any(x['severity']!='WARNING' for x in candidate.get('mechanical_issues',[])):
        raise ValueError('Technical/mechanical review failure; no score permitted')
    if candidate['original_sha256']!=digest(original):raise ValueError('Candidate/source canonical binding failed')
    expected_id=result['source_id']+f"::sample-{result['sample']}"
    if candidate['trajectory_id']!=expected_id or original['id']!=expected_id:
        raise ValueError('Report candidate belongs to another sampled trajectory')
    if original['student_outputs'][-1]!=result['answer']:
        raise ValueError('Reviewed answer differs from actual actor output')
    terminal=[c for c in candidate['checks'] if c['step_id']=='terminal-answer']
    if not terminal or any(c['status']!='COMPLETED' for c in terminal):raise ValueError('Report review incomplete')
    if any(c['provenance']['policy_sha256']!=POLICY_SHA for c in terminal):raise ValueError('Policy mismatch')
    disputed=[c for c in terminal if c['kind']=='claim' and c['subject'].get('kind') in {'fact','interpretation'}
              and c['review_judgment']=='NOT_SUPPORTED']
    if disputed:
        if contradiction_reviews is None or any(c['task_id'] not in contradiction_reviews for c in disputed):
            raise ValueError('Potential factual contradiction requires explicit confirmation before reward freezing')
        # Adjudication only chooses a penalty. It can never reverse the
        # original support verdict or turn a disputed report into full credit.
        return False
    if result['status']!='MODEL_ANSWER_UNREVIEWED':return False
    answer=result['answer']
    if answer['category'] in {None,'Harmless'} or answer.get('review_reason')=='EVIDENCE_INSUFFICIENT':return False
    # SFT markers conservatively suppress downstream supervision after an
    # uncertain query. The GRPO report reward asks whether the actual final
    # images support this report, independently of whether that query was good.
    # Recover only this reward view when the real prior steps explicitly have
    # no prose premises and *every* terminal claim/stop was directly supported.
    # Never change candidate labels or recover an unknown same-step premise.
    prior=[];found_terminal=False
    for step in original.get('steps',[]):
        if step.get('id')=='terminal-answer':found_terminal=True;break
        if step.get('type')=='decision':prior.append(step)
    independent_prefix=found_terminal and all(s.get('dependencies_complete') is True
        and s.get('claims',[])==[] and s.get('depends_on_claim_ids')==[] for s in prior)
    raw_terminal_supported=all(c.get('review_judgment',c['judgment'])=='SUPPORTED'
        for c in terminal if c['kind']!='blind')
    markers=candidate['supervision_markers']
    if independent_prefix and raw_terminal_supported:
        ids={c['task_id'] for c in terminal if c['kind']!='blind'}
        markers=[dict(m,supervise=True) if m['task_id'] in ids else m for m in markers]
    else:
        if not any(c['kind']=='stop' and c['judgment']=='SUPPORTED' for c in terminal):return False
    n=len(result['requests'])-1
    wanted={f'/student_outputs/{n}/action':answer['action']}
    wanted.update({f'/student_outputs/{n}/{k}':answer[k] for k in ['category','severity','moderation_action']})
    wanted.update({f'/student_outputs/{n}/evidence/{i}/description':e['description'] for i,e in enumerate(answer['evidence'])})
    for suffix,value in wanted.items():
        possible=[prefix+suffix for prefix in ['', '/claim_source_snapshot']]
        matched=False
        for pointer in possible:
            try:
                if resolve(original,pointer)!=value:continue
            except (KeyError,IndexError):continue
            if covered(markers,pointer,value):matched=True;break
        if not matched:return False
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['rollouts','report-review','acquisition','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--contradictions',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    load=lambda path:json.loads(path.read_text())
    rollout=load(a.rollouts/'summary.json');report=load(a.report_review/'summary.json')
    acq=load(a.acquisition/'summary.json');abind=load(a.acquisition/'binding.json')
    adjudications={}
    if a.contradictions:
        adjudicated=load(a.contradictions/'summary.json')
        if adjudicated['status']!='COMPLETED' or adjudicated['report_summary_sha256']!=sha(a.report_review/'summary.json'):
            raise ValueError('Contradiction confirmation incomplete or bound to different reports')
        adjudications=adjudicated['rows']
    if abind['rollout_summary_sha256']!=sha(a.rollouts/'summary.json') or abind['policy_sha256']!=POLICY_SHA:
        raise ValueError('Acquisition reward bound to a different rollout/policy')
    if report['state']!='COMPLETED' or acq['status']!='COMPLETED':raise ValueError('Review technical failure/incomplete')
    report_by_id={r['trajectory_id']:r for r in report['results']}
    rows=[]
    for result in rollout['results']:
        alias=next(x for x in acq['aliases'] if x['source_id']==result['source_id'] and x['sample']==result['sample'])
        audit=acq['tasks'][alias['task_id']]
        if sha(audit['raw_path'])!=audit['raw_sha256']:raise ValueError('Acquisition raw response changed')
        acquisition_input=load(a.acquisition/(alias['task_id']+'.input.json'))
        if digest(acquisition_input)!=audit['input_sha256']:raise ValueError('Acquisition input changed')
        _,req,_=read_bound_item(Path(result['requests'][-1]['item_path']))
        for im,expected in zip(acquisition_input['images'],req['images']):
            if {k:im[k] for k in ['id','timestamp','sha256']}!={k:expected[k] for k in ['id','timestamp','sha256']}:
                raise ValueError('Acquisition review is not of this actor seen set')
        if len(acquisition_input['images'])!=len(req['images']):raise ValueError('Acquisition image set differs')
        acquired=any(f['judgment']=='SUPPORTED' for f in audit['response']['findings'])
        key=result['source_id']+f"::sample-{result['sample']}"
        meta=report_by_id.get(key);good_report=False;candidate_hash=None;penalty_reviews={}
        if meta:
            path=Path(meta['candidate_file']);candidate=load(path);original=load(path.with_name('original.json'))
            for check in candidate['checks']:
                inp=load(path.parent/'tasks'/check['task_id']/'input.json')
                if digest(inp)!=check['provenance']['input_sha256']:raise ValueError('Accepted report input changed')
                rv=load(path.parent/'tasks'/check['task_id']/'review.json')
                if rv['input_sha256']!=digest(inp):raise ValueError('Report response/input binding differs')
                if any(im['sha256'] not in {x['sha256'] for x in req['images']} for im in inp['images']):
                    raise ValueError('Audit included unseen images')
                pending=adjudications.get(candidate['trajectory_id']+'::'+check['task_id'])
                if pending is not None:
                    obj={'claim':check['subject'],'images':inp['images'],
                         'original_input_sha256':digest(inp),'candidate_sha256':sha(path)}
                    if (pending['candidate_sha256']!=sha(path) or pending['input_sha256']!=digest(obj)
                        or sha(pending['raw_path'])!=pending['raw_sha256']
                        or pending['response']['status']!='COMPLETED'
                        or pending['original_report_not_upgraded'] is not True):
                        raise ValueError('Counterevidence confirmation lost its original object binding')
                    penalty_reviews[check['task_id']]=pending
            good_report=report_supported(candidate,original,result,penalty_reviews);candidate_hash=sha(path)
        elif result['status']=='MODEL_ANSWER_UNREVIEWED':raise ValueError('Terminal report has not been reviewed')
        tools=len(result['actual_tools']);new_frames=sum(len(t['returned_timestamps']) for t in result['actual_tools'])
        cost=.01*tools+.001*new_frames
        contradicted=any(x['response']['verdict']=='CONTRADICTED' for x in penalty_reviews.values())
        total=.5*acquired+.5*(good_report and acquired)-cost-float(contradicted)
        rows.append({'source_id':result['source_id'],'sample':result['sample'],'acquisition_supported':acquired,
            'report_supported':good_report,'semantic_score':.5*acquired+.5*(good_report and acquired),
            'tool_cost':.01*tools,'frame_cost':.001*new_frames,'total':total,
            'acquisition_task_id':alias['task_id'],'candidate_sha256':candidate_hash,
            'confirmed_fabrication_penalty':-float(contradicted),'fabrication_not_inferred_from_unknown':True,
            'contradiction_confirmation':penalty_reviews,
            'weak_reference_only':True,'result':result})
    signals={}
    for source in {r['source_id'] for r in rows}:
        group=[r for r in rows if r['source_id']==source]
        signals[source]=group_signal(group)
    (a.out/'FROZEN_SEMANTIC_REWARDS.json').write_text(json.dumps({'version':'semantic-evidence-acquisition-report-v1',
        'rollout_summary_sha256':sha(a.rollouts/'summary.json'),'report_summary_sha256':sha(a.report_review/'summary.json'),
        'acquisition_summary_sha256':sha(a.acquisition/'summary.json'),'rows':rows,'groups':signals,
        'report_reward_scope':'Actual terminal evidence/claims/stop; no whole-trajectory SFT upgrade. Prior unverified prose premises remain blocking.',
        'test_used':False,'optimizer_steps':0,'missing_contradiction_review_is_not_a_penalty':True},ensure_ascii=False,indent=2))
    print(json.dumps(signals))


if __name__=='__main__':main()
