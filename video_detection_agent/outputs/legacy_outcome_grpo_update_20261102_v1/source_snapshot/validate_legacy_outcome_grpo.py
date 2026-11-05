"""Frozen Train-only paired check for outcome/level/cost GRPO. No optimizer."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from check_hf_policy_alignment import binding
from collect_grpo_selection_batch import state
from prepare_grpo_group_selection import ROOT, digest, save
from prepare_legacy_train_grpo_pilot import identity, jsonl
from run_legacy_train_grpo_pilot import score_cohort


def read(p):return json.loads(Path(p).read_text())


def prepare(root,pilot):
    reserved=root/'reserved';original=read(reserved/'PLAN.json');selections=read(reserved/'SELECTION.json')
    training=read(pilot/'PLAN.json')
    merged={r['id']:r for r in jsonl(ROOT/'data/merged_video_safety_v1/manifest.jsonl')}
    trainkeys=set()
    for row in read(Path(training['cohort'])/'SELECTION.json'):trainkeys.update(identity(row['source'],merged))
    seen=set()
    for row in selections:
        keys=identity(row['source'],merged)
        if keys & (seen|trainkeys):raise ValueError('Validation overlaps pilot source/parent')
        seen.update(keys)
    plan={'protocol':'legacy-outcome-paired-Train-validation-v1','script_sha256':digest(__file__),
          'pilot':str(pilot),'pilot_plan_sha256':digest(pilot/'PLAN.json'),
          'reserved_plan_sha256':digest(reserved/'PLAN.json'),
          'reserved_selection_sha256':digest(reserved/'SELECTION.json'),
          'videos':20,'samples_per_video':4,'rollouts_per_model':80,'fresh_rollouts':160,
          'frozen_before_update_and_new_outputs':True,'optimizer_steps':0,'dev_test_used':False,
          'fixed120_identity_exclusion_only':True,'automatic_further_training':False,
          'readiness_gates':{
              'all_sampler_replay_mean_logprob_error_at_most':.05,
              'all_sampler_replay_max_logprob_error_at_most':.2,
              'zero_update_parameters_unchanged':True,
              'finite_nonzero_lora_gradients_and_no_base_gradients':True,
              'exactly_one_optimizer_update_saved_and_reloaded':True,
              'post_update_mean_reference_kl_at_most':.02,
              'paired_output_technical_failures':0,
              'L2_category_and_level_matches_must_not_decrease':True,
              'harmful_reference_predicted_L0_must_not_increase':True,
              'L0_reference_predicted_review_or_block_must_not_increase':True,
              'mean_task_reward_must_not_decrease':True,
              'visual_inputs_on_equal_successful_results_must_not_increase':True},
          'meaning':'Pass permits a bounded next GRPO experiment; does not establish quality gain or production readiness.'}
    save(root/'PLAN.json',plan)
    (root/'CODE_SNAPSHOT.py').write_bytes(Path(__file__).read_bytes())
    state(root/'RUN_PROGRESS.json',{'stage':'PREPARED_BEFORE_UPDATE','optimizer_steps':0,'updated_unix':time.time()})


def compare(root):
    data={}
    for side in ['sft','updated']:
        folder=root/('paired_'+side);score_cohort(folder)
        rows=[json.loads(s) for s in (folder/'TRACEABLE_REWARDS.jsonl').open()]
        valid=[r for r in rows if 'final_report' in r and r['reward']['total'] is not None]
        l2=[r for r in valid if r['reference_severity']=='L2' and not r['reference_boundary']]
        harmful=[r for r in valid if r['reference_category']!='Harmless']
        normal=[r for r in valid if r['reference_category']=='Harmless']
        data[side]={'attempts':len(rows),'technical_holds':len(rows)-len(valid),
            'category_matches':sum(r.get('category_match',False) for r in valid),
            'strict_joint_matches':sum(r.get('strict_joint_match',False) for r in valid),
            'explicit_L2_attempts':len(l2),'explicit_L2_category_and_level_matches':sum(r['strict_joint_match'] for r in l2),
            'harmful_predicted_L0':sum(r['final_report']['severity']=='L0' for r in harmful),
            'normal_predicted_review_or_block':sum(r['final_report']['severity']!='L0' for r in normal),
            'mean_task_reward':sum(r['reward']['task'] for r in valid)/len(valid) if valid else None,
            'all_visual_inputs':sum(r.get('visual_input_occurrences',0) for r in valid),
            'tool_calls':sum(len(r['actual_tools']) for r in rows)}
        data[side]['rows']={(r['source_id'],r['sample']):r for r in rows}
    a,b=data['sft'],data['updated'];reasons=[]
    if set(a['rows'])!=set(b['rows']):raise ValueError('Unpaired sample identities')
    if a['technical_holds'] or b['technical_holds']:reasons.append('TECHNICAL_HOLD')
    if b['explicit_L2_category_and_level_matches']<a['explicit_L2_category_and_level_matches']:reasons.append('L2_MATCHES_DECREASED')
    if b['harmful_predicted_L0']>a['harmful_predicted_L0']:reasons.append('RISK_ALLOW_INCREASED')
    if b['normal_predicted_review_or_block']>a['normal_predicted_review_or_block']:reasons.append('NORMAL_FALSE_ALARM_INCREASED')
    if a['mean_task_reward'] is None or b['mean_task_reward'] is None:reasons.append('MISSING_TASK_METRIC')
    elif b['mean_task_reward']+1e-12<a['mean_task_reward']:reasons.append('TASK_REWARD_DECREASED')
    common=[]
    for key,x in a['rows'].items():
        y=b['rows'][key]
        if (x['reward'].get('cost_eligible') and y['reward'].get('cost_eligible')
                and x['reward'].get('task')==y['reward'].get('task')):
            common.append((x['visual_input_occurrences'],y['visual_input_occurrences']))
    cost={'paired_equal_success_results':len(common),'sft_image_occurrences':sum(x for x,y in common),
          'updated_image_occurrences':sum(y for x,y in common)}
    if not common:reasons.append('NO_COMPARABLE_SUCCESS_FOR_COST')
    elif cost['updated_image_occurrences']>cost['sft_image_occurrences']:reasons.append('COMPARABLE_COST_INCREASED')
    for side in data:data[side].pop('rows')
    status='HOLD_BEFORE_EXPANSION' if reasons else 'READY_FOR_BOUNDED_GRPO_EXPERIMENT'
    summary={'status':status,'reasons':reasons,'models':data,'cost':cost,
        'new_optimizer_steps_in_validation':0,'effect_improvement_required_for_readiness':False,
        'dev_test_used':False,'fixed120_used':False,'weak_reference_only':True,
        'large_training_or_deployment_authorized_by_this_report':False}
    save(root/'SUMMARY.json',summary)
    lines=['# 一次GRPO更新后的独立Train配对检查','',f'结论：{status}。原因：{reasons}。','',
           '|指标|原SFT|一次更新后|','|---|---:|---:|']
    for key in ['attempts','category_matches','strict_joint_matches','explicit_L2_category_and_level_matches',
                'harmful_predicted_L0','normal_predicted_review_or_block','mean_task_reward','all_visual_inputs','tool_calls']:
        lines.append(f"|{key}|{data['sft'][key]}|{data['updated'][key]}|")
    lines+=['',f'相同达标结果成本对照：{cost}。',
            '参考是旧Train弱标签，非独立画面真值。20视频同源同种子、各4次，80次尝试不是80个独立视频。',
            'PASS只表示可以推进有上限的下一轮GRPO实验，不代表效果已提升；不自动启动扩容或更多更新。旧Dev/Test和固定120未用于本次奖励或比较。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return status


def run(root):
    plan=read(root/'PLAN.json');pilot=Path(plan['pilot']);reserved=root/'reserved'
    if digest(__file__)!=plan['script_sha256'] or digest(pilot/'PLAN.json')!=plan['pilot_plan_sha256']:
        raise ValueError('Frozen code/training plan changed')
    if digest(reserved/'PLAN.json')!=plan['reserved_plan_sha256'] or digest(reserved/'SELECTION.json')!=plan['reserved_selection_sha256']:
        raise ValueError('Preselected validation changed')
    update=read(pilot/'one_update/summary.json');reload=read(pilot/'reload_check/summary.json')
    if update['optimizer_steps']!=1 or not update['original_checkpoint_unchanged'] or reload['status']!='PASS':
        raise ValueError('Update/reload gate failed')
    if not 0<=update['mean_reference_kl']<=.02:raise ValueError('Reference KL exceeds frozen gate')
    template=read(reserved/'PLAN.json');original=template['model_binding']
    for side in ['sft','updated']:
        folder=root/('paired_'+side);folder.mkdir(exist_ok=False)
        checkpoint=Path(original['checkpoint_path'] if side=='sft' else update['adapter_path'])
        newbinding=binding(Path(original['base_path']),checkpoint)
        selection=[{**s,'rollout_dir':str(folder/'student_rollouts'/f"group-{s['group']:02d}")} for s in read(reserved/'SELECTION.json')]
        save(folder/'PLAN.json',{**template,'checkpoint_path':str(checkpoint),'model_binding':newbinding})
        save(folder/'SELECTION.json',selection)
        state(root/'RUN_PROGRESS.json',{'stage':'COLLECT_'+side.upper(),'optimizer_steps':0,'updated_unix':time.time()})
        with (root/('collect_'+side+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('collect_grpo_selection_batch.py')),
                            '--cohort',str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    status=compare(root)
    state(root/'RUN_PROGRESS.json',{'stage':'COMPLETE','result':status,'optimizer_steps':0,'updated_unix':time.time()})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=['prepare','run'])
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--pilot',type=Path,required=True)
    a=ap.parse_args();root=a.out.resolve()
    if a.phase=='prepare':prepare(root,a.pilot.resolve())
    else:
        try:run(root)
        except Exception as exc:
            state(root/'RUN_PROGRESS.json',{'stage':'STOPPED_FAILURE','error':repr(exc),'optimizer_steps':0})
            raise
