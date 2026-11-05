"""Collect fresh legacy-Train groups and score a draft reward; no optimizer."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from collect_grpo_selection_batch import state
from outcome_cost_reward_v1 import VERSION, reward
from prepare_grpo_group_selection import ROOT, digest
from review_grpo_selection_qwen import trace


def read(p): return json.loads(Path(p).read_text())


def score_cohort(root):
    plan = read(root/'PLAN.json')
    rows, groups = [], []
    for selected in read(root/'SELECTION.json'):
        source = selected['source']
        with Path(source['teacher_record_file']).open('rb') as handle:
            raw = next(data for i,data in enumerate(handle,1) if i == int(source['teacher_record_line']))
        import hashlib
        if hashlib.sha256(raw).hexdigest() != source['teacher_record_sha256']:
            raise ValueError('Teacher source changed')
        obj = json.loads(raw)
        if obj['id'] != source['teacher_id'] or obj['result']['final_action']['category'] != source['reference_category']:
            raise ValueError('Teacher identity/reference conflict')
        results = read(Path(selected['rollout_dir'])/'summary.json')['results']
        if sorted(r['sample'] for r in results) != [0,1,2,3]: raise ValueError('Four attempts missing')
        group_rows = []
        for result in results:
            row = {'group': selected['group'], 'sample': result['sample'], 'source_id': source['merged_id'],
                   'video_sha256': source['sha256'], 'source_video': source['video_path'],
                   'reference_category': source['reference_category'], 'reference_severity': source['reference_severity'],
                   'reference_boundary': source['reference_boundary'], 'trajectory_status': result['status'],
                   'receipt_path': str(Path(selected['rollout_dir'])/'rollouts'/
                       (source['merged_id']+f"-sample-{result['sample']}")/'receipt.json'),
                   'requests': result.get('requests', []), 'actual_tools': result.get('actual_tools', [])}
            try:
                verified = trace(selected, result)
                images, unique = 0, set()
                for call in result['requests']:
                    item = read(call['item_path'])
                    request = read(item['request_path'])
                    if item['model_binding'] != plan['model_binding']: raise ValueError('Wrong model binding')
                    images += len(request['images'])
                    unique.update(im['sha256'] for im in request['images'])
                answer = verified['final_report']
                if answer['moderation_action'] != {'L0':'ALLOW','L1':'REVIEW','L2':'BLOCK'}.get(answer['severity']):
                    raise ValueError('Final action/level inconsistent')
                score = reward(reference_category=source['reference_category'],
                    reference_severity=source['reference_severity'], predicted_category=answer['category'],
                    predicted_severity=answer['severity'], visual_inputs=images,
                    boundary=source['reference_boundary'])
                row.update(final_report=answer, reward=score, visual_input_occurrences=images,
                           unique_seen_image_hashes=len(unique),
                           category_match=answer['category']==source['reference_category'],
                           strict_joint_match=answer['category']==source['reference_category'] and answer['severity']==source['reference_severity'])
            except Exception as exc:
                # Do not invent a model-error score for unclassified execution or
                # provenance failures. Keep all four attempts in a held group.
                row.update(reward={'status':'HOLD','total':None,'reason':'TRACE_OR_EXECUTION_FAILURE'}, error=repr(exc))
            rows.append(row)
            group_rows.append(row)
        scores = [r['reward']['total'] for r in group_rows]
        tasks = [r['reward'].get('task') for r in group_rows]
        kind = 'HOLD' if None in scores else ('RESULT_DIFFERENCE' if len(set(tasks))>1 else (
               'COST_ONLY_DIFFERENCE' if len(set(scores))>1 else 'NO_REWARD_DIFFERENCE'))
        groups.append({'group':selected['group'],'source_id':source['merged_id'],
                       'rewards':scores,'task_scores':tasks,'kind':kind})
    summary = {'status':'COMPLETE_OFFLINE_REWARD_CHECK_NO_UPDATE','reward_version':VERSION,
        'groups':groups,'attempts':len(rows),'completed_reports':sum('final_report' in r for r in rows),
        'category_matches':sum(r.get('category_match',False) for r in rows),
        'strict_joint_matches':sum(r.get('strict_joint_match',False) for r in rows),
        'group_outcomes':dict(Counter(g['kind'] for g in groups)),
        'tool_actions':dict(Counter(t['executed_action'] for r in rows for t in r['actual_tools'])),
        'tool_count_histogram':dict(Counter(len(r['actual_tools']) for r in rows)),
        'predicted_levels':dict(Counter(r['final_report']['severity'] for r in rows if 'final_report' in r)),
        'reference_levels':dict(Counter(r['reference_severity'] for r in rows)),
        'cost_measure':'actual image occurrences in requests, including repeats; not visual tokens or KV work',
        'optimizer_steps':0,'dev_test_unchanged':True,'fixed120_identity_exclusion_only':True,
        'independent_visual_truth_claimed':False,'finished_unix':time.time()}
    for p,expected in plan['frozen_source_hashes'].items():
        if digest(p)!=expected: raise ValueError('Frozen source changed: '+p)
    state(root/'SUMMARY.json',summary)
    (root/'TRACEABLE_REWARDS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    outcomes=summary['group_outcomes']
    text = f'''# 旧 Train 真实 Agent 采样与奖励试点

来源为旧 Train 13431 条，选20条，每条独立4次；不覆盖历史结果。Dev/Test及固定120条仅提供身份排除信息，冻结文件哈希核对未变化。模型为冻结的20261027 SFT，所有阶段同一适配器；不设累计取帧、调用或决策上限。

已保存 {len(rows)} 条尝试，完成且可追溯报告 {summary['completed_reports']} 条。类别与旧弱参考一致 {summary['category_matches']} 条，严格类别+等级一致 {summary['strict_joint_matches']} 条。严格指标未采用L1/L2边界容忍，与奖励不是同一统计量；不能称独立人工准确率。

组结论：{json.dumps(outcomes,ensure_ascii=False)}。工具动作：{json.dumps(summary['tool_actions'],ensure_ascii=False)}。组内差异只是草案奖励信号，不代表已证明画面事实正确或模型已提升。无差异组不人为排序，技术/来源核对失败组HOLD。

奖励以正确类别+L2为优先；L1复核时类别影响较小；L0正确放行也有完整奖励；历史边界保留。任务分和成本分逐项保存。成本按实际请求中的图片呈现次数（含重复）计算，并非token、KV运算量或真实提速。GRPO标准化会放大细微差值，成本差异组在训练接入时仍须单独检查权重，不能仅凭成本项很小就声称影响很小。

本阶段优化器步骤为0。后续必须核对概率回放、助手动作掩码、组优势和梯度，再决定小更新范围。此报告不声称补帧、画面事实或成本收益已全面验证。

逐视频来源和4条原始轨迹见 SELECTION.json；逐轨迹回执、请求、工具返回、原参考、奖励分解见 TRACEABLE_REWARDS.jsonl；组统计见 SUMMARY.json；来源隔离见 SOURCE_AUDIT.json / QUARANTINE.json；时间与显存见 COLLECTION_PROGRESS.json 和各组summary.json。
'''
    (root/'REPORT_ZH.md').write_text(text)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort',type=Path,required=True)
    args=ap.parse_args(); root=args.cohort.resolve()
    lock=(root/'.pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    status={'stage':'PREPARING','pid':os.getpid(),'started_unix':started,'optimizer_steps':0}
    state(root/'RUN_PROGRESS.json',status)
    try:
        plan=read(root/'PLAN.json')
        for p,expected in plan['frozen_source_hashes'].items():
            if digest(p)!=expected:raise ValueError('Source changed: '+p)
        for row in plan['exclusion_files']:
            if digest(row['path'])!=row['sha256']:raise ValueError('Exclusion changed')
        snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=False)
        files=['prepare_legacy_train_grpo_pilot.py','run_legacy_train_grpo_pilot.py',
               'show_legacy_train_grpo_progress.py','collect_grpo_selection_batch.py','outcome_cost_reward_v1.py',
               'hf_unified_rollout.py','unified_student_contract.py','student_action_outcomes.py',
               'hf_request_policy.py','hf_request_rng.py','prompts/unified_student_v1.txt']
        hashes={}
        for name in files:
            src=Path(__file__).parent/name;dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dest);hashes[str(src)]=digest(src)
        state(root/'CODE_HASHES.json',hashes)
        status.update(stage='COLLECTING_REAL_TRAJECTORIES',updated_unix=time.time())
        state(root/'RUN_PROGRESS.json',status)
        with (root/'collector.log').open('x') as log:
            subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('collect_grpo_selection_batch.py')),
                '--cohort',str(root)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        for p,expected in hashes.items():
            if digest(p)!=expected:raise ValueError('Runtime source changed')
        status.update(stage='OFFLINE_REWARD_CHECK',updated_unix=time.time())
        state(root/'RUN_PROGRESS.json',status)
        summary=score_cohort(root)
        status.update(stage='COMPLETE_NO_OPTIMIZER_UPDATE',updated_unix=time.time(),
                      elapsed_seconds=time.time()-started,group_outcomes=summary['group_outcomes'])
        state(root/'RUN_PROGRESS.json',status)
    except Exception as exc:
        status.update(stage='STOPPED_FAILURE',error=repr(exc),updated_unix=time.time())
        state(root/'RUN_PROGRESS.json',status)
        raise


if __name__=='__main__':main()
