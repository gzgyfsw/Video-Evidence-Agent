"""Read-only progress for engineering checks, one update and paired validation."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'outputs'


def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {}


def bar(done,total):
    n=int(30*done/max(total,1))
    return '['+'#'*n+'-'*(30-n)+f'] {done}/{total}'


def main():
    root=ROOT/'legacy_outcome_grpo_update_20261102_v1'
    validation=ROOT/'legacy_train_grpo_validation_20261102_v1'
    run=read(root/'RUN_PROGRESS.json');plan=read(root/'PLAN.json')
    alive=False
    if run.get('pid'):
        try:os.kill(run['pid'],0);alive=True
        except OSError:pass
    print('GRPO进入检查：6组 / 24条轨迹 /',plan.get('selected_model_requests','?'),'次真实模型请求')
    print('总阶段:',run.get('stage','NOT_STARTED'),'后台:', '运行中' if alive else '已结束/未启动')
    for name in ['zero_update_check','one_update','reload_check']:
        progress=read(root/name/'progress.json');summary=read(root/name/'summary.json')
        if not progress:continue
        text=bar(progress.get('completed',0),progress['total']) if progress.get('total') else ''
        print(name,summary.get('status') or progress.get('stage'),text)
    update=read(root/'one_update/summary.json')
    if update:print('实际更新次数:',update['optimizer_steps'],'参考KL:',update['mean_reference_kl'])
    v=read(validation/'RUN_PROGRESS.json')
    if v:print('独立Train对照:',v.get('stage'),v.get('result',''))
    for side in ['sft','updated']:
        folder=validation/('paired_'+side)
        if folder.exists():
            done=len(list((folder/'student_rollouts').glob('group-*/rollouts/*/receipt.json')))
            c=read(folder/'COLLECTION_PROGRESS.json')
            print(side,bar(done,80),c.get('stage','WAITING'))
    if run.get('error'):print('错误:',run['error'])
    result=read(validation/'SUMMARY.json')
    if result:print('最终结论:',result['status'],'原因:',result['reasons'])
    print('运行目录:',root)


if __name__=='__main__':main()
