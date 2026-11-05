"""Display actual saved receipts; no model requests or experiment mutations."""
import argparse
import json
import os
import time
from pathlib import Path


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cohort',type=Path,default=Path(__file__).parent/'outputs/legacy_train_grpo_pilot_20261102_v1')
    args=ap.parse_args();root=args.cohort.resolve()
    plan=read(root/'PLAN.json');run=read(root/'RUN_PROGRESS.json');col=read(root/'COLLECTION_PROGRESS.json')
    receipts=list((root/'student_rollouts').glob('group-*/rollouts/*/receipt.json'))
    total=plan.get('rollouts',80);done=len(receipts);width=30;filled=int(width*done/total)
    groups=len(col.get('completed_groups',[]))
    alive=False
    if run.get('pid'):
        try:os.kill(run['pid'],0);alive=True
        except OSError:pass
    print('旧Train真实Agent采样 → 奖励验证（优化器未启动）')
    print('['+'#'*filled+'-'*(width-filled)+f'] {done}/{total} 条  {done/total:.1%}；完整组 {groups}/20')
    print('阶段:',run.get('stage','NOT_STARTED'),' | 采样:',col.get('stage','WAITING'),
          ' | 后台进程:', '运行中' if alive else '已结束/未启动')
    if 'group' in col:print('当前组:',col['group'],'最近完成sample:',col.get('sample','尚无'))
    if col.get('gpu'):print('最近GPU记录:',json.dumps(col['gpu'],ensure_ascii=False))
    if run.get('started_unix'):print('已运行:',round((time.time()-run['started_unix'])/60,1),'分钟')
    if run.get('error') or col.get('error'):print('错误:',run.get('error') or col.get('error'))
    summary=read(root/'SUMMARY.json')
    if summary:print('组结论:',json.dumps(summary['group_outcomes'],ensure_ascii=False))
    print('日志:',root/'collector.log')
    print('结果目录:',root)


if __name__=='__main__':main()
