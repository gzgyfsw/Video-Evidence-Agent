"""Sequential engineering gate, one update, reload, independent Train pairing."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from collect_grpo_selection_batch import state
from prepare_grpo_group_selection import ROOT, digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot',type=Path,required=True)
    ap.add_argument('--validation',type=Path,required=True)
    a=ap.parse_args();root=a.pilot.resolve();validation=a.validation.resolve()
    lock=(root/'.readiness.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();status={'stage':'STARTING','pid':os.getpid(),'started_unix':start,'max_optimizer_steps':1}
    state(root/'RUN_PROGRESS.json',status)
    try:
        snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=False)
        names=['run_legacy_grpo_readiness.py','train_legacy_outcome_grpo_pilot.py',
               'validate_legacy_outcome_grpo.py','show_legacy_outcome_grpo_progress.py',
               'train_teacher_category_grpo_pilot.py','train_partial_report_grpo.py',
               'hf_request_policy.py','outcome_cost_reward_v1.py','collect_grpo_selection_batch.py',
               'hf_unified_rollout.py','unified_student_contract.py','student_action_outcomes.py',
               'train_hf_request_grpo_smoke.py','run_legacy_train_grpo_pilot.py']
        hashes={}
        for name in names:
            src=Path(__file__).parent/name;shutil.copy2(src,snapshot/name);hashes[str(src)]=digest(src)
        state(root/'CODE_HASHES.json',hashes)
        for phase in ['check','update','reload','validation']:
            for path,h in hashes.items():
                if digest(path)!=h:raise ValueError('Frozen code changed: '+path)
            status.update(stage=phase.upper(),updated_unix=time.time())
            state(root/'RUN_PROGRESS.json',status)
            if phase=='validation':
                cmd=[sys.executable,'-u',str(Path(__file__).with_name('validate_legacy_outcome_grpo.py')),
                     'run','--out',str(validation),'--pilot',str(root)]
            else:
                cmd=[sys.executable,'-u',str(Path(__file__).with_name('train_legacy_outcome_grpo_pilot.py')),
                     phase,'--out',str(root)]
            with (root/(phase+'.log')).open('x') as log:
                subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((validation/'SUMMARY.json').read_text())
        status.update(stage='COMPLETE',result=result['status'],updated_unix=time.time(),
                      elapsed_seconds=time.time()-start,optimizer_steps=1)
        state(root/'RUN_PROGRESS.json',status)
        (root/'READINESS_RESULT_ZH.md').write_text(
            '# 正式GRPO前检查\n\n结论：'+result['status']+'。\n\n'
            '零更新概率/梯度检查、一次真实更新、保存后重新加载、独立Train配对比较均已执行。'
            '是否通过以validation/SUMMARY.json的冻结指标为准。PASS只允许推进有上限的下一轮训练实验，'
            '不是模型已提升或已可部署的证明。未自动扩容或追加更新。\n\n'
            '配对结果目录：'+str(validation)+'\n')
    except Exception as exc:
        status.update(stage='STOPPED_FAILURE',error=repr(exc),updated_unix=time.time())
        state(root/'RUN_PROGRESS.json',status)
        raise


if __name__=='__main__':main()
