import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_unified_actor import online_initial


class UnifiedActorDataTests(unittest.TestCase):
    def test_online_initial_exposes_only_real_initial_state(self):
        row={'id':'v','timestamps':[0,4,8,12],'source_partition':'train',
             'prompt':'FROZEN_POLICY\n当前真实已见帧时间戳：[0,4,8,12]',
             'target':'{"action":"ANSWER","rationale":"SECRET_FUTURE_LABEL"}'}
        changed=online_initial(row,{'duration_seconds':12})
        state=json.loads(changed['prompt'].split('当前运行状态（历史工具记录不是视觉证据）：')[1])
        self.assertEqual(state['tools_remaining'],3)
        self.assertEqual(state['frames_remaining'],20)
        self.assertEqual(state['history_summary']['prior_tools'],[])
        self.assertEqual(state['visible_timestamps'],[0,4,8,12])
        self.assertNotIn('SECRET_FUTURE_LABEL',changed['prompt'])
        self.assertEqual(changed['target'],row['target'])
        self.assertNotIn('当前运行状态',row['prompt'])


if __name__=='__main__':unittest.main()
