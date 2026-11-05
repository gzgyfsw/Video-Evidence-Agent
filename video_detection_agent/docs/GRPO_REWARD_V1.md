# GRPO Reward v1（冻结草案）

目标：最终分类和等级正确优先；仅在完全正确时奖励自主工具路线、证据落地和更低成本。
不使用 learned/VLM judge。基础设施失败返回 `None`，重试或剔除。

## 总式

`R = R_task + R_route + R_expand + R_grounding - R_cost`

非法 JSON 或无最终答案直接为 `-5.0`，不计算其他项。

### R_task

| 结果 | 奖励 |
|---|---:|
| 类别和等级均正确 | +4.0 |
| 类别正确、等级错误 | +1.5 |
| 有害类别之间混淆 | -1.0 |
| Harmless 误报为有害 | -2.0 |
| 有害漏判为 Harmless | -5.0 |
| JSON 非法或无终态 | -5.0 |
| 视频、工具或推理基础设施异常 | None（不训练） |

### R_route（仅严格正确）

- 初始证据满足框架直出条件，模型合法直接回答：`+0.3`；
- 模型自主调用 `JOINT_SEMANTIC_SCAN` 且最终严格正确：`+0.3`；
- 模型先错误直出，Python 守卫救回：`0`；
- Python 插入的工具动作：`0`。

不要求复制 Teacher 的工具次数。Harmless 不能初始直出，必须完成三类风险排除。

### R_expand（仅严格正确）

同时满足以下条件才为 `+0.2`：

1. `TEMPORAL_EXPAND` 由模型生成；
2. 冻结 Teacher 中间状态也标记当前证据不足；
3. 返回的新帧被后续回答真实使用。

### R_grounding（仅严格正确）

- 有害答案：奖励时间戳必须对应模型真实看过的帧；
- Harmless：真实检索后对 Sexual、Violence、Sensitive 三类均给出完整排除。

满足为 `+0.1`。只检查字段非空不算落地。

### R_cost（仅严格正确）

`min(0.30, 0.03 * num_tool_calls + 0.003 * num_extra_frames)`

初始 4 帧不计成本。工具成本始终远小于任务正确性奖励。

## 优势与 token 规则

- GRPO 组内优势只减均值、不除标准差；本地 TRL 0.29.1 对应 `scale_rewards="none"`；
- Python 插入的动作、工具结果及其 token 不计算 log-prob；
- 只有模型自己生成的 Assistant token 参与策略损失；
- Test 不用于奖励权重、训练轮数或 checkpoint 选择。

实现：`video_detection_agent/grpo_reward.py`。
