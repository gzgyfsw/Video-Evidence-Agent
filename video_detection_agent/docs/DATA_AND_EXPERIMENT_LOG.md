# 视频安全 Agent：数据与实验记录

> 状态：持续更新  

> 日报索引：[daily/README.md](daily/README.md)。最新记录：[2026-10-12](daily/2026-10-12.md)，包含方向变化、真实工程执行、THVL指标限制及后续计划。
> 本次记录日期：2026-09-05  
> 当前阶段：先冻结标签可信数据，再进行 Student 严格对照与 SFT/GRPO 划分。

> 完整的“从非 Agent 基线到 SFT、GRPO、消融和 THVL 外测”的时间线与统一结果表见：
> `outputs/thvl_current_protocol_20261010_v4/REPORT.md`。

## 1. 研究目标

构建能够根据视频难度自主决定是否调用工具、查看哪些时间点以及何时输出最终分类的
视频安全 Agent。类别空间固定为：

1. `Harmless`
2. `Sexual Content`
3. `Violence & Dangerous`
4. `Sensitive Figures & Ideologies`

风险等级为 `L0 / L1 / L2`。等级不参与“标签是否可信”的第一阶段验收，只在
Teacher–Student 严格对照、SFT 和 GRPO 阶段使用。

## 2. 当前原始主数据集

主清单：`data/merged_video_safety_v1/manifest.jsonl`

| 原始类别 | 数量 |
|---|---:|
| Harmless | 4,382 |
| Sexual Content | 5,645 |
| Violence & Dangerous | 4,775 |
| Sensitive Figures & Ideologies | 4,532 |
| **合计** | **19,334** |

当前主数据集中的视频均严格大于 10 秒。训练、验证和测试划分必须按照
`source_group` 分组，避免同一父视频或相关片段跨集合泄漏。

## 3. 标签可信池的验收规则

标签可信度必须与 Student 表现完全分开。当前确认的准入规则为：

```text
条件 A：主 Teacher（Qwen3.7-Plus）的类别与来源数据集原始类别一致；
或
条件 B：两个不同 Teacher 家族对同一样本给出相同的大类。
```

Teacher 家族及优先级：

1. 主 Teacher：Qwen3.7-Plus；
2. 第二 Teacher：Qwen3.8-27B；
3. 历史 Qwen3.6-27B 作为第二 Teacher 的历史补充结果。

标签验收只比较四分类大类，不要求两个 Teacher 的 `severity` 一致。若两个 Teacher
一致但与原始标签不同，则采用 Teacher 共识类别，同时保留原始标签和重标来源。

下列情况不得进入标签可信池：

- 没有主 Teacher 与原始标签一致，也没有两个不同 Teacher 的类别共识；
- 视频抽帧/解码失败，模型实际上没有获得视觉输入；
- 旧 AV1 解码失败集合 43 条，已按当前决定整体排除。

## 4. 标签可信池 v1 阶段性统计

生成脚本：`video_detection_agent/build_label_verified_pool.py`  
结果目录：`video_detection_agent/outputs/label_verified_pool_v1/`

历史结果全量扫描得到的阶段性候选数为 **16,801**：

| 最终 Teacher 验收类别 | 数量 |
|---|---:|
| Harmless | 6,339 |
| Sexual Content | 5,313 |
| Violence & Dangerous | 2,939 |
| Sensitive Figures & Ideologies | 2,210 |
| **合计** | **16,801** |

验收来源：

| 验收方式 | 数量 |
|---|---:|
| Qwen3.7-Plus 与原始标签一致 | 9,670 |
| Qwen3.7-Plus 与 Qwen3.8 类别共识 | 3,837 |
| Qwen3.8 与 Qwen3.6 类别共识 | 3,280 |
| Qwen3.7-Plus 与 Qwen3.6 类别共识 | 14 |
| 未满足标签可信规则 | 1,865 |
| 旧 AV1 集合排除 | 43 |

共有 3,895 条样本采用了与原始标签不同的 Teacher 共识类别。阶段性来源标签到最终
类别的矩阵如下，行是原始类别，列是最终 Teacher 验收类别：

| 原始类别 \ 最终类别 | Harmless | Sexual | Violence | Sensitive |
|---|---:|---:|---:|---:|
| Harmless | 3,538 | 120 | 178 | 106 |
| Sexual | 268 | 5,068 | 93 | 4 |
| Violence | 1,338 | 52 | 2,224 | 24 |
| Sensitive | 1,195 | 73 | 444 | 2,076 |

### 4.1 论文统计前必须修正的重复运行问题

16,801 是“扫描全部历史 Teacher 输出、同一 Teacher 家族保留其所有历史类别”的
阶段性上界，暂时不能直接写成论文最终数据量。同一样本存在多次 Teacher 运行，且部分
重复运行的类别互相冲突：

| Teacher 家族 | 有有效输出的样本 | 同一家族内部类别冲突 |
|---|---:|---:|
| Qwen3.7-Plus | 14,854 | 727 |
| Qwen3.8 | 8,957 | 485 |
| Qwen3.6 | 3,649 | 52 |

论文最终版不能从同一 Teacher 的多次输出中挑选最有利的一次。必须为每个 Teacher
家族确定唯一的 canonical run，建议优先级为：

1. 使用当前统一 4 帧非黑初始采样与完整 Agent 链路的最终重跑结果；
2. 若最终重跑缺失，使用明确标记为 recovery/final 的最后一次有效结果；
3. smoke、旧提示词、旧采样策略和中间失败重试不得覆盖最终结果。

完成 canonical run 固定后，重新生成标签可信池，届时的数量才可作为论文正式数据。

## 5. Direct / Adaptive / Active 的“是否需要补帧”定义

当前 Student 为 `Qwen3.5-4B`。比较目标是已验收 Teacher 的最终
`category + severity`，Student 的初始判断取独立保存的四帧 baseline。分组以
Teacher 的**实际完整轨迹是否调用工具**为准：

```text
Direct  = Teacher最终正确 ∧ Teacher.tool_calls == 0 ∧ Student初始类别和等级正确
Adaptive = Teacher最终正确 ∧ Teacher.tool_calls == 0 ∧ Student初始类别或等级错误
Active  = Teacher最终正确 ∧ Teacher.tool_calls >= 1
```

### 5.1 当前 Direct 的可执行判定

对同一条视频的同一组初始非黑均匀帧，记已验收 Teacher 的最终目标为
`(target_category, target_severity)`。仅当下列三项同时成立，样本才是 Direct：

```text
Teacher 最终输出 == ANSWER(target_category, target_severity)
Teacher.tool_calls == []
Student initial_pred / initial_severity == target_category / target_severity
```

Direct 的含义是：**Teacher 无需补帧，Student 也已经具备直接作答能力**。若 Teacher
不调用工具但 Student 初始判断错误，则是 Adaptive；只要 Teacher 的已验收完整轨迹
实际调用过工具，就是 Active。按当前安全流程，Harmless 必须经过联合风险排查，因此
不会被划入 Direct 或 Adaptive。

旧的“类别/等级一致且 Student 工具轮数、帧数不超过 Teacher”的四项条件保留为**执行
效率诊断**，不再用于定义 D/A/A。

Student 因 JSON 截断、格式或协议错误而失败，但 Teacher 轨迹完整且视频抽帧成功时：

- 样本仍可用于 Teacher 监督 SFT；
- 可用于带格式奖励的 GRPO；
- 单独标记为 `Teacher-only`；
- 不伪装成 Direct、Adaptive 或 Active。

若失败发生在视频解码、黑帧过滤或没有得到可见截图，则不纳入训练。

## 6. Sensitive 增量的 Student 阶段性实验

实验目录：`video_detection_agent/outputs/qwen35_sensitive_daa_delta_20260902/`

对 1,811 条 Teacher 已验收 Sensitive 样本运行完整 Qwen3.5-4B Agent 后：

| 状态 | 数量 |
|---|---:|
| 严格合格：类别、等级、工具数、截图数全部满足 | 728 |
| 需要学习 | 558 |
| Teacher-only：Student JSON/协议失败，非截图失败 | 525 |
| **合计** | **1,811** |

558 条需要学习的样本按 Teacher 是否调用工具进一步分为：

- Adaptive 候选：387；
- Active 候选：171。

这组结果仅代表 Sensitive 增量，不可外推为完整数据集的 Student 准确率或完整
Direct/Adaptive/Active 分布。

### 6.1 当前标签可信池上的 Student 覆盖与分类表现（阶段性）

统计基准为第 4 节的 16,801 条标签可信阶段性候选。只纳入
`joint_risk_v1_student` 的完整 Agent 输出；初始帧-only 结果不视为完整 Agent
轨迹。

| 指标 | 数量 / 比例 |
|---|---:|
| 标签可信池总数 | 16,801 |
| 已有完整 Student Agent 输出 | 3,157（18.79%） |
| Student 正常输出 | 2,469 |
| Student JSON/协议失败 | 688 |
| 正常输出中的类别正确数 | 1,939 |
| 正常输出条件下类别准确率 | 78.53% |
| 端到端类别准确率（失败也计错） | 61.42% |

分最终类别的当前覆盖和类别准确率：

| Teacher 验收类别 | 正确 | 错误 | Student 失败 | 正常输出条件准确率 |
|---|---:|---:|---:|---:|
| Harmless | 715 | 231 | 175 | 75.58% |
| Sexual Content | 36 | 15 | 71 | 70.59% |
| Violence & Dangerous | 229 | 99 | 45 | 69.82% |
| Sensitive Figures & Ideologies | 959 | 185 | 397 | 83.83% |

此表存在明显覆盖偏差：Sexual 仅有 122 条完整 Student 输出，不能用于宣称四类
最终能力。完整 Student Agent 尚未运行的标签可信样本为 13,644 条。

当前可同时取得 canonical Teacher 轨迹、完整 Student 轨迹、以及工具/帧数信息的
严格审计子集为 2,115 条。下表采用当时的旧“执行效率”标准，仅保留作历史记录：

| 严格分组 | Harmless | Sexual | Violence | Sensitive | 合计 |
|---|---:|---:|---:|---:|---:|
| Direct（四项全部满足） | 191 | 18 | 57 | 606 | 872 |
| Adaptive 候选（Teacher 未用工具） | 8 | 22 | 60 | 393 | 483 |
| Active 候选（Teacher 用过工具） | 67 | 1 | 11 | 145 | 224 |
| Teacher-only（非截图失败） | 60 | 62 | 17 | 397 | 536 |
| **合计** | **326** | **103** | **145** | **1,541** | **2,115** |

该表是严格资源约束下的已审计子集，不是 16,801 条全局最终 D/A/A 表。

## 7. 全量 Qwen3.5-4B Student 按“是否需要补帧”划分（当前正式统计）

统计日期：2026-09-05。`outputs/integrated_student_agent_v1/source.jsonl` 中有
**16,792** 条 canonical Teacher 标签，和同 ID 的 Qwen3.5-4B Student 输出。其中
3 条 Teacher 仅有最终标签而没有 `result/transcript`，不能判断是否需要工具，已标记为
Excluded。因此当前 D/A/A 有效轨迹为 **16,789** 条。

分组沿用第 5 节的实际工具调用定义：

```text
Direct  = Teacher零工具正确 ∧ Student初始正确
Adaptive = Teacher零工具正确 ∧ Student初始错误
Active  = Teacher使用工具后得到已验收终态
```

| 类型 \ 最终Teacher类别 | Harmless | Sexual | Violence | Sensitive | 合计 | 用途 |
|---|---:|---:|---:|---:|---:|---|
| Direct（Teacher零工具且Student初始正确） | 0 | 2,274 | 616 | 738 | **3,628** | GRPO |
| Adaptive | 0 | 438 | 736 | 534 | **1,708** | SFT + GRPO |
| Active | 6,339 | 2,600 | 1,581 | 933 | **11,453** | SFT + GRPO |
| **合计** | **6,339** | **5,312** | **2,933** | **2,205** | **16,789** | |

## 8. Student 训练版本迭代记录

统计日期：2026-09-12。以下表格只把已经明确冻结或完整评估过的版本作为正式记录。
`Dev1679` 是冻结验证集；`Test1679` 仍未使用。

### 8.1 冻结 Dev1679 主指标

| 版本 | 状态 | 大类准确率 | Macro-F1 | 大类+等级准确率 | 有害漏判率 | Harmless误报率 | 必要工具首步召回 | Direct首步工具率 | 证据利用率 | 平均帧数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 未微调 Qwen3.5-4B 初始4帧 | 基线参考 | 83.68% | 81.74% | 69.68% | 10.62% | 19.40% | 不适用 | 不适用 | 不适用 | 4.00 |
| 未微调 Qwen3.5-4B 完整Agent | SFT/GRPO直接基线 | 80.76% | 79.48% | 66.59% | 9.38% | 29.81% | 9.21% | 0.00% | 未记录 | 8.01 |
| SFT v5 checkpoint-50 | 已验收冷启动 | 87.97% | 87.64% | 81.24% | 3.73% | 21.61% | 60.53% | 2.75% | 83.73% | 7.85 |
| GRPO v1 | 已验收 | 87.97% | 87.64% | 81.30% | 3.73% | 21.61% | 62.94% | 3.03% | 83.46% | 7.89 |
| GRPO v2 | 当前正式最佳 | 87.73% | 87.35% | 81.18% | 3.25% | 22.08% | 63.16% | 3.03% | 87.41% | 7.94 |

说明：

- “初始4帧”不是完整 Agent，只用于衡量模型原始视觉分类能力。
- “完整Agent”才是 SFT 和 GRPO 的直接对照。
- GRPO v2 的数值来自 v2 composite frozen Dev1679 验收口径；它相对同口径 parent
  提升为：大类准确率 +0.48pp，Macro-F1 +0.35pp，大类+等级 +0.42pp，
  Harmless误报 -0.79pp，证据利用 +0.83pp，平均帧数 -0.07。

### 8.2 已尝试但未冻结的 GRPO v3

| 版本 | 评估集 | 状态 | 主要现象 |
|---|---|---|---|
| GRPO v3 iteration_005 | Dev1679 | 拒绝 | TEMPORAL_EXPAND 增多，必要时序召回从 60% 到 80%，但大类+等级下降 0.89pp，平均奖励下降，未通过冻结门槛。 |
| GRPO v3 iteration_012 historical_i5_strict | Dev64 | 通过小样本筛选 | 候选 `grpo-v3-i5-smoke_harmless_payload-s2` 在严格时序 stress Dev64 上大类准确率 70.31%，高于 parent 65.63%；必要时序召回 100%。 |
| GRPO v3 iteration_012 historical_i5_strict | Dev640/Dev1679 | 运行中 | Dev640 stress 正在评估；只有 Dev640 和完整 Dev1679 均通过后，才会冻结为 `best_grpo_qwen35_4b_agent_v3`。 |

当前正在运行的自动验收链不会接触 Test 集。若 Dev640 或完整 Dev1679 任一门槛失败，
仍保留 GRPO v2 作为正式最佳模型。

- Excluded：3 条；均为只有标签、缺少完整 Teacher 轨迹，不训练也不评估。
- SFT 候选：`Adaptive + Active = 13,161` 条。
- GRPO 候选：`Direct + Adaptive + Active = 16,789` 条。
- Teacher 零工具 `5,336` 条；Teacher 使用工具 `11,453` 条；有效轨迹中 Student
  初始类别与等级同时正确 `11,794` 条。
- Student 的 87 条类别别名输出已在 2026-09-05 进行确定性规范化
  （`Sexual → Sexual Content`、`Sensitive → Sensitive Figures & Ideologies`、
  `Harmless-L0 → Harmless`）；均非截断，轨迹完整。

可复现产物：`outputs/integrated_student_agent_v1/daa_need_frames_20260905/`，其中
`assignments.jsonl` 是逐样本分组，`direct.jsonl`、`adaptive.jsonl`、`active.jsonl`
为三组清单，`summary.json` 和 `report.md` 为汇总。

## 8. 后续实验顺序

1. 将当前已满足验收规则、但尚未并入 canonical source 的 Teacher 轨迹回收后，重建标签可信池；
2. 为这些新增 canonical Teacher 轨迹补齐 Qwen3.5-4B Student 完整 Agent 输出，再重算第 7 节；
3. 对尚未验收的 Teacher 结果继续进行第二 Teacher 复核；对从未运行或无完整轨迹的视频补跑 Teacher；
4. SFT 重点使用第 7 节的 Adaptive + Active；
5. GRPO 使用 Direct、Adaptive、Active，并按类别、等级、工具次数、帧数和格式设计奖励。

## 9. 当前产物

- 标签可信阶段性清单：`video_detection_agent/outputs/label_verified_pool_v1/label_verified.jsonl`
- 标签可信阶段性统计：`video_detection_agent/outputs/label_verified_pool_v1/summary.json`
- Sensitive 严格对照：`video_detection_agent/outputs/qwen35_sensitive_daa_delta_20260902/strict_teacher_student_audit.jsonl`
- Sensitive 严格统计：`video_detection_agent/outputs/qwen35_sensitive_daa_delta_20260902/strict_teacher_student_audit.summary.json`
- 当前全量 DAA 分组：`video_detection_agent/outputs/integrated_student_agent_v1/daa_need_frames_20260905/`

## 10. 冻结 Train / Dev / Test v1

生成日期：2026-09-05。生成脚本：
`video_detection_agent/build_frozen_agent_splits.py`。输出目录：
`video_detection_agent/outputs/agent_split_v1_20260905/`。

本划分只使用第 7 节中 **16,789 条具备完整 canonical Teacher 轨迹的已验收样本**。
另有 3 条只有已验收标签、没有 Teacher 轨迹，已作为 Excluded 移除。主 manifest 中
尚未通过 Teacher 标签验收的 2,542 条也不进入当前训练、验证或测试集。

划分原则：

1. 比例固定为 80% / 10% / 10%；
2. 使用 `source_group`、`parent_video_sha256` 和视频 SHA256 构造来源连通组件；
3. 同一父视频、同源片段或重复视频只能存在于一个 split；
4. 在来源隔离前提下，同时保持类别、Severity、D/A/A、时长、工具触发来源、Teacher
   验收来源和 Teacher 家族的分布；
5. Dev/Test 对 Base、zero-shot Agent、SFT 和 SFT+GRPO 始终使用同一批 ID；
6. Test 不用于选择 prompt、阈值、SFT epoch 或 GRPO reward；这些选择只看 Dev。

| Split | 数量 | 占比 | 当前协议轨迹子集 |
|---|---:|---:|---:|
| Train | 13,431 | 80.00% | 13,187 |
| Dev | 1,679 | 10.00% | 1,642 |
| Test | 1,679 | 10.00% | 1,640 |

Test 的类别与 D/A/A 分布：

| 类型 \ 类别 | Harmless | Sexual | Violence | Sensitive | 合计 |
|---|---:|---:|---:|---:|---:|
| Direct | 0 | 227 | 61 | 74 | 362 |
| Adaptive | 0 | 44 | 74 | 54 | 172 |
| Active | 633 | 260 | 159 | 93 | 1,145 |
| **合计** | **633** | **531** | **294** | **221** | **1,679** |

Train 中可用于 SFT 冷启动的 `Adaptive + Active` 共 **10,528 条**；Direct 2,903 条
保留给 GRPO 的“正确且避免不必要工具”目标。最终 SFT 可以只从 10,528 条 Train 候选
中再抽取约 1,000 条高质量轨迹，不能从 Dev/Test 抽取。

验收结果：Train、Dev、Test 的样本 ID、来源连通组件和视频 SHA256 两两零重叠；全部
16,789 条有效轨迹对应的本地视频路径均存在。`*_behavior_eval.jsonl` 是只保留当前 4 帧非黑初始采样且
只使用当前工具协议的评测视图；它不会改变主 Test 的最终分类评测口径。

## 11. SFT 前冻结的 Dev 基线

评测脚本：`video_detection_agent/evaluate_frozen_agent.py`。基线结果：
`video_detection_agent/outputs/dev_baseline_qwen35_4b_20260905/`。以下结果均来自冻结的
1,679 条 Dev，Test 尚未使用。

| 阶段 | 类别准确率 | Macro-F1 | 大类正确时的等级准确率 | 类别+等级严格准确率 | 有害漏成Harmless | Harmless误报有害 |
|---|---:|---:|---:|---:|---:|---:|
| 初始4帧、未使用工具 | 83.68% | 81.74% | 83.27% | 69.68% | 10.62% | 19.40% |
| 未微调完整Agent | 80.76% | 79.48% | 82.45% | 66.59% | 9.38% | 29.81% |

当前完整 Agent 虽把有害漏判率从 10.62% 降至 9.38%，但总体分类、严格准确率和
Harmless 误报均变差，说明工具轨迹会带来噪声，SFT 确有必要。

| D/A/A | 数量 | 初始严格准确率 | 完整Agent严格准确率 | Student工具调用率 |
|---|---:|---:|---:|---:|
| Direct | 363 | 100.00% | 88.15% | 0.00% / 46.56% |
| Adaptive | 172 | 0.00% | 33.72% | 1.16% / 43.02% |
| Active | 1,144 | 70.54% | 64.69% | 4.02% / 69.49% |

必要工具首步召回率为 **42/456=9.21%**：即 Teacher 明确要求工具的 Dev 状态中，
Student 在首步主动选择工具的比例。Student 平均使用 8.013 帧、平均工具调用次数为
0.618；Teacher 平均使用 8.602 帧、平均工具调用次数为 0.702。

### 11.1 固定解释口径

1. SFT 前后应直接比较“未微调完整 Agent”和“SFT 完整 Agent”；初始4帧只作为分类
   能力参考基线，不能替代 Agent 对照组；
2. D/A/A 由 Teacher 轨迹与未微调 Student 初始预测共同定义，因此 Direct 初始联合
   准确率 100%、Adaptive 初始联合准确率 0% 具有定义性，仅用于观察 SFT 后的迁移和退化；
3. Harmless 只允许 L0，因此大类已经判成 Harmless 时，条件等级准确率恒为 100%，
   不作为等级预测能力的主要证据；
4. 上表工具列按“模型原始首步主动调用率 / 框架守卫后的实际工具执行率”报告。前者衡量
   Router，后者衡量完整系统成本，二者不得混用。“平均工具调用次数”按每条轨迹中的真实
   工具执行次数求和后取平均，支持多次调用。当前 Dev Student 恰好只有 0/1 次调用：
   641 条为0次、1,038条为1次，所以 `1038/1679` 恰好等于平均调用次数0.618；
5. Dev 为 Direct 363、Adaptive 172、Active 1,144、Harmless 634；Test 为 Direct 362、
   Adaptive 172、Active 1,145、Harmless 633。二者相差一条来自正常分层划分，并非数据
   或分组发生变化。

SFT 通过标准暂不在本节固定；先保留以上基线，待训练方案和主要目标确定后再预注册。

### 11.2 SFT 主对照与工具证据利用口径

唯一主对照口径：**在相同冻结 Dev、相同 Agent 框架和相同推理配置下，比较未微调
Student 与 SFT Student 的分类、分级、工具决策及推理成本。** 初始4帧结果只作为纯分类
能力参考，不替代完整 Agent 对照。

对 Active/实际调用工具的轨迹固定报告：

1. 工具调用后是否形成合法最终答案；
2. 有害最终答案的时间戳证据是否对应工具实际新增的帧；
3. 初始类别或等级错误是否在工具调用后被纠正；
4. 是否调用了工具，但最终类别和等级完全沿用初始判断。

Dev 中只有 5 条 `TEMPORAL_EXPAND` 轨迹，样本量不足以单独给出稳定百分比。该工具先做
逐条轨迹检查和参数合法性测试，不能仅凭这 5 条宣称统计提升。

## 12. Agent SFT 冷启动的冻结验收标准

SFT 的目标不是在训练集上单纯提高分类准确率，而是让 Student 学会当前 Agent 协议：
在证据充分时直接回答，在证据不足时**自主**选择工具，读取真实工具结果后再作答。
未满足本节硬门槛前，不启动 GRPO。

### 12.1 训练管线门槛（必须全部通过）

1. Train、Dev、Test ID 和来源组件零泄漏；
2. 初始决策状态只含 4 张初始图片，不含未来工具结果；
3. 工具后状态只含此前真实执行的动作、真实工具结果和新增图片；
4. loss 只计算目标 Assistant JSON；system/user/tool-result token 全为 `-100`；
5. `JOINT_SEMANTIC_SCAN` 的完整 JSON（工具名及 `queries_by_risk` 参数）必须参与 loss；
6. 训练和推理均关闭 thinking，并使用相同首步 prompt、动作名和 JSON schema。

当前 v2 审计：800 条 Active 的原始 Teacher 首步中，338 条主动扫描，462 条先回答后由
框架守卫强制扫描；进入 v2 SFT 的实际首步目标已统一纠正为 800/800
`JOINT_SEMANTIC_SCAN`，0 条首步 `ANSWER`。800 条均为 4 张初始图片，首步 JSON 均只有
`action` 和 `queries_by_risk` 两个顶层字段。64 条真实 collator 抽查中，64/64 完整目标
参与 loss，32/32 扫描样本的工具名和参数均参与 loss。

当前主训练/推理链路使用 `JOINT_SEMANTIC_SCAN`，schema 为：

```json
{
  "action": "JOINT_SEMANTIC_SCAN",
  "queries_by_risk": {
    "Sexual Content": ["..."],
    "Violence & Dangerous": ["..."],
    "Sensitive Figures & Ideologies": ["..."]
  }
}
```

旧 `agent/actions.py` 中的 `SEMANTIC_LOCATE` 属于早期通用 Agent 接口，不是当前 Joint
Agent 的 SFT/推理入口；论文实验不得混用两套协议。

### 12.2 64 条 Router 同集过拟合门槛

先用 32 条扫描正样本和 32 条直接回答负样本做管线/容量诊断。扫描正样本包含 16 条
Harmless；直接回答负样本由 11 条严格 Direct 和 21 条 Adaptive 组成。所有样本只给
初始 4 帧。

| 指标 | 通过门槛 |
|---|---:|
| 合法 JSON | 100% |
| 首步动作准确率 | ≥95% |
| 扫描正样本召回率 | ≥95% |
| 直接回答负样本召回率 | ≥95% |
| ANSWER 样本类别+等级严格准确率 | ≥90% |

该实验在训练样本本身评估，只用于判断训练管线和现有 LoRA 容量，不作为泛化结果。
若通过，说明现有 LoRA 足以学习二选一 Router，后续应调整数据配比、首步 loss 权重或
采用 Router warm-up；若连同集都不能通过，才检查 LoRA 加载/可训练模块并考虑提高 rank。

### 12.3 冻结 Dev 上允许进入 GRPO 的硬门槛

所有指标必须使用同一冻结 Dev、相同 Agent 框架、相同初始 4 帧和相同推理参数。工具
自主性使用**框架守卫介入前的原始首步动作**计算，不能把框架强制调用冒充模型主动调用。

| 指标 | 未微调完整 Agent 基线 | SFT 通过门槛 |
|---|---:|---:|
| 合法结构化首步动作率 | 待完整记录 | ≥98% |
| 必要工具首步召回率 | 9.21%（42/456） | ≥40% |
| Direct 原始首步无意义工具调用率 | 0.00%（0/363） | ≤25% |
| 工具后合法复查/终态率 | 待统计 | ≥95% |
| 工具证据利用率 | 待统计 | ≥80% |
| 类别准确率 | 80.76% | ≥80.76% |
| Macro-F1 | 79.48% | ≥79.48% |
| 类别+等级严格准确率 | 66.59% | ≥66.59% |
| 有害漏成 Harmless | 9.38% | ≤9.38% |
| Harmless 误报有害 | 29.81% | ≤25% |
| 平均图片数 | 8.013 | ≤9.46 |

“工具证据利用率”的逐样本口径固定为：有害答案至少一条 grounded evidence 对应真实
新增帧时间戳；Harmless-L0 必须在真实工具结果后给出三类完整排除结果。仅调用工具但
答案完全沿用调用前判断，不计为利用。`TEMPORAL_EXPAND` 因 Dev 只有 5 条，暂要求逐条
参数合法、帧真实返回且后续回答成功，不单独设统计显著性门槛。

以上任一硬门槛失败，本轮只继续修正 SFT/Router，不进入 GRPO。初始 4 帧参考基线
（类别 83.68%、Macro-F1 81.74%、严格准确率 69.68%、Harmless 误报 19.40%）作为
进一步改善目标，但不替代“相同完整 Agent”主对照。

### 12.4 Router 64 条诊断结果（2026-09-05）

该实验从 `sft_qwen35_4b_video_trajectory_v2/final_adapter` 继续训练，保持 rank=16、
语言层 LoRA 范围和冻结视觉塔不变；只改变 64 条首步路由数据的配比，并重复训练 10
epoch（160 optimizer steps）。诊断权重与正式 v2 权重分目录保存。

旧评测曾把图片最长边固定缩到 384 并重新压缩为 JPEG，与 SFT 按 `384×384` 总像素预算
处理 PIL 图片的方式不一致，因此旧的 81.25% 结果作废。推理改为同一像素预算并使用
无损 PNG 后，先得到 95.31%；再从该诊断权重续训 48 个 optimizer step，只对初始
`ANSWER/JOINT_SEMANTIC_SCAN` 动作 token 增加 2.0 倍辅助 loss，最终结果如下。

| 指标 | 修正图像预处理后 | 动作 token 强化后 | 门槛 |
|---|---:|---:|---:|
| 合法 JSON | 100% | 100% | 100% |
| 首步动作准确率 | 95.31% | **98.44%（63/64）** | ≥95% |
| 扫描正样本召回率 | 96.88%（31/32） | **96.88%（31/32）** | ≥95% |
| 直接回答负样本召回率 | 93.75%（30/32） | **100%（32/32）** | ≥95% |
| ANSWER 类别+等级严格准确率 | 93.75% | **93.75%** | ≥90% |

结论：64 条 Router 同集门槛已全部通过，证明 rank=16、仅语言层 LoRA 足以学习当前
二选一路由，没必要因该问题扩大 LoRA 或做全参数微调。正式训练采用：1600 条 Train-only
Router warm-up（扫描/回答 800:800）→ 1200 条完整因果工具轨迹并重放 400 条 Direct
首步答案；完整轨迹阶段仍保持首步扫描/回答 800:800，避免遗忘后退化为全扫描。

### 12.5 Train-only Router warm-up 与早停选择（2026-09-05）

正式 Router warm-up 从 `sft_qwen35_4b_video_trajectory_v2/final_adapter` 继续，只对首步
`ANSWER/JOINT_SEMANTIC_SCAN` 动作值 token 计算 loss。训练集为 1,600 条 Train-only
状态，扫描/回答严格 800:800；Dev/Test 零泄漏。冻结 `router_dev256_v1` 同时覆盖
Direct、Adaptive、Teacher 主动工具和 Harmless 框架守卫四个单元，每单元 64 条。

| checkpoint | 动作准确率 | 必要工具召回 | Direct乱调用 | Harmless扫描 | 回答类别准确率 | 回答类别+等级 |
|---|---:|---:|---:|---:|---:|---:|
| step-50 | 78.78% | 42.11% | 3.23% | 87.30% | 95.65% | 73.91% |
| step-100 | 75.51% | 28.07% | 3.23% | 76.19% | 96.75% | 64.23% |
| **step-150** | **83.67%** | **56.14%** | **3.23%** | **88.89%** | 94.17% | 62.50% |
| step-200 | 81.63% | 50.88% | 3.23% | 85.71% | 94.17% | 63.33% |

四个 checkpoint 的合法 JSON 均为 100%。按“先满足必要工具召回与 Direct 成本门槛，
再比较动作准确率及回答质量”的预注册方向，选择 **step-150** 作为完整轨迹 SFT 起点，
不使用训练 loss 最低的 step-200。step-150 已完成 Router 冷启动，但回答严格准确率暂时
下降，因此还不能单独作为最终 SFT；后续必须通过完整因果轨迹训练恢复分类、分级和工具
结果利用能力，并重新通过冻结 Dev 完整 Agent 门槛。

### 12.6 轨迹 SFT、Router 刷新与 post-tool 定向修复（2026-09-05/06）

完整轨迹 v3 从 Router warm-up step-150 训练 200 optimizer steps。虽然回答类别准确率恢复
到约 98%，但冻结 Router Dev256 的必要工具首步召回退化为 0%，说明长答案 token 对首步
路由监督产生了严重遗忘；因此 v3 不作为最终检查点。

v4 从 v3 checkpoint-100 做 route-only 刷新。step-50 在 Router Dev256 上达到：合法 JSON
100%、必要工具召回 57.81%、Direct 多余调用 6.25%、Harmless 扫描召回 93.94%。但在冻结
Agent Dev64 上，有害漏判仍为 24.53%，说明 Router 已会调用工具，工具后的视觉复查仍不够
可靠。

v5 从 v4 step-50 继续，只监督 Active 轨迹的 `post_scan_assessment` Assistant JSON；历史
动作、用户 prompt、真实工具结果与所有未来信息均屏蔽，不再稀释首步与工具后判断两个目标。
800 条 Train-only Active 轨迹训练 100 optimizer steps，学习率 1e-5。冻结 Router Dev256
选择 step-50 后，在冻结 Agent Dev64 上得到：

| 指标 | v4 step-50 | **v5 step-50** | 门槛 |
|---|---:|---:|---:|
| 合法首步动作 | 历史结果未保留原始一次率 | **98.44%** | ≥98% |
| 必要工具首步召回 | 56.25% | **43.75%** | ≥40% |
| 类别准确率 | 78.12% | **92.19%** | ≥80.76% |
| Macro-F1 | 78.26% | **91.79%** | ≥79.48% |
| 类别+等级 | 68.75% | **76.56%** | ≥66.59% |
| 有害漏成 Harmless | 24.53% | **5.66%** | ≤9.38% |
| Harmless 误报有害 | 0% | **0%** | ≤25% |
| 工具后合法终态 | 100% | **100%** | ≥95% |
| 工具证据利用率 | 93.75% | **84.78%** | ≥80% |
| 平均图片数 | 9.047 | **8.891** | ≤9.46 |

v5 step-50 在 Dev64 的全部硬门槛上通过，并首次在该快速集形成了一次
`TEMPORAL_EXPAND` 轨迹。注意：当前 runner 是由模型先输出三类风险评估，再由 Python 在
`INSUFFICIENT/PRESENT` 时调度时间扩展，不能把它表述成模型自主 function call。它是当前
最佳 SFT 候选，但 Dev64 仅用于快速迭代；最终冻结前必须在
完整 1,679 条 Dev 上复验。完整 Dev 任务写入
`outputs/sft_qwen35_4b_posttool_v5/full_agent_dev1679/`，Test 仍保持封存。

### 12.7 v5 完整冻结 Dev 最终验收（2026-09-06）

`posttool_v5/checkpoint-50` 已完成全部 1,679 条冻结 Dev，结果覆盖 1,679 个唯一 ID，缺失、
越界、重复和运行错误均为 0。相对相同 Agent 框架、相同 4 帧初始输入和相同 Dev 上的未
微调 Qwen3.5-4B，结果如下：

| 指标 | 未微调完整 Agent | **SFT v5 step-50** | 变化 | 硬门槛 |
|---|---:|---:|---:|---:|
| 类别准确率 | 80.76% | **87.97%** | +7.21pp | ≥80.76% |
| Macro-F1 | 79.48% | **87.64%** | +8.16pp | ≥79.48% |
| 类别+等级严格准确率 | 66.59% | **81.24%** | +14.65pp | ≥66.59% |
| 有害漏成 Harmless | 9.38% | **3.73%** | -5.65pp | ≤9.38% |
| Harmless 误报有害 | 29.81% | **21.61%** | -8.20pp | ≤25% |
| 必要工具首步召回 | 9.21% | **60.53%（276/456）** | +51.32pp | ≥40% |
| Direct 原始首步乱调用 | 0.00% | **2.75%（10/363）** | +2.75pp | ≤25% |
| 合法结构化首步动作 | 历史基线未完整记录 | **98.57%** | — | ≥98% |
| 工具后合法终态 | 100% | **100%** | 0pp | ≥95% |
| 工具证据利用率 | 历史基线未按现口径记录 | **83.73%** | — | ≥80% |
| 平均图片数 | 8.013 | **7.854** | -0.159 | ≤9.46 |

11 项预注册硬门槛全部通过。模型原始首步主动输出 `JOINT_SEMANTIC_SCAN` 602 次；另有 412
次是安全框架在模型首步答案不满足“明确 L2 有害证据”时强制扫描，二者没有混算为自主
调用。完整 Dev 仅出现 1 次 `TEMPORAL_EXPAND`：请求时间点、4 秒窗口、4 帧返回和后续终态
均合法；最终答案未引用新增帧，因此该次在“工具证据利用率”中如实计为未利用。该样本量
不足以证明模型具备自主时间扩展能力，后续 GRPO 若要优化该能力，必须单独增加路由动作和
评测样本。

训练因果审计同时通过：v5 最终阶段只监督 800 条 Train-only Active 轨迹中的
`post_scan_assessment` Assistant JSON；初始状态只含 4 帧，未来工具结果泄漏为 0，最长
样本的非目标 token 均为 `-100`。Router 训练 1,600 个 ID、轨迹训练 1,200 个 ID，与 Dev
和 Test 的交集均为 0；完整评测结果与 Dev 1,679 个 ID 完全一致，与 Test 交集为 0。

因此该检查点正式冻结为 SFT 冷启动版本：
`outputs/best_sft_qwen35_4b_agent_v1/adapter`。复现实验所需的权重、数据、划分、脚本和指标
SHA-256 记录在 `outputs/best_sft_qwen35_4b_agent_v1/BEST_SFT.json`。截至冻结时 Test 未使用，
GRPO 未启动。

### 12.8 GRPO v1 启动与真实工具链 smoke（2026-09-06）

GRPO 从冻结的 SFT v5 step-50 LoRA 启动，Test 继续封存。没有使用 TRL 默认的文本工具循环，
因为该循环不能把 SigLIP/Temporal 新抽出的真实图片注入后续 VLM 回合。当前实现先由 vLLM
从 Train 视频采样多条完整 Agent 轨迹，真实执行 `JOINT_SEMANTIC_SCAN` 和框架调度的
`TEMPORAL_EXPAND`，再重建相同多轮消息与图片，仅对模型生成的 Assistant JSON token
计算 GRPO log-prob；Python 强制动作、工具结果和用户 prompt 均不参与梯度。

奖励函数冻结在 `grpo_reward.py`：类别+等级全对 +4，类别对等级错 +1.5，有害类互错 -1，
Harmless 误报有害 -2，有害漏成 Harmless -5，非法 JSON/无终态 -5；仅严格全对时再计算
自主路由、grounding 和工具/帧成本。优势采用同视频组均值中心化、**不除组内标准差**。
当前 `TEMPORAL_EXPAND` 仍由 Python 根据模型风险评估调度，因此不冒充自主调用，不发放
expand bonus。

工程 smoke 使用 10 个 Train 视频、每视频 4 条随机轨迹，共 40 条：无运行失败，34 条
类别+等级全对，6 条仅等级错；10 个组中 2 组存在奖励差异，8 条轨迹具有非零优势。一次
LoRA 更新的梯度范数为 0.6696，SFT 与更新后权重 SHA-256 不同，更新后可被 vLLM 正常加载。
同一批 10 条视频的确定性前后对照输出 10/10 完全一致，说明单步 smoke 没有破坏现有行为；
该权重只作训练管线验证，不作为正式候选。

正式 iteration-001 从原冻结 SFT v5 重新开始，使用 128 个未被 smoke 使用的 Train 视频、
每视频 4 条随机真实工具轨迹（512 条），按 10 个现有 D/A/A×类别桶近似均衡采样。Dev
验收规则已在 `docs/GRPO_ACCEPTANCE_V1.md` 预先冻结；完整 Dev 结果产生前不修改门槛。

### 12.9 GRPO iteration-001 更新与 Dev64 快检（2026-09-06）

512/512 条随机真实工具轨迹全部成功，运行失败为 0。128 个视频组中 64 组存在组内奖励
差异，产生 256 条非零优势轨迹；奖励分布为：393 条类别+等级全对、82 条类别对但等级错、
19 条有害漏成 Harmless、13 条有害类别互错、5 条 Harmless 误报有害。批次平均奖励
3.31456，64 个零方差组不参与策略梯度。

从冻结 SFT v5 执行 8 个 GRPO optimizer step，学习率 5e-6、KL beta 0.02、clip epsilon
0.2，每步 8 个视频组。训练平均 KL 为 0.000353，平均/最大梯度范数为 0.4265/0.5270，
训练耗时 839.2 秒。Assistant-only log-prob 审计通过；Python 强制动作与工具返回不参与
梯度，Test 未使用。更新后 Adapter 权重 SHA-256 为
`090af7e008b4737a0267b2f112a5911fb6ac1c9566197128adf37378d66d0469`，与冻结 SFT 权重不同。

冻结 Dev64 快检结果如下。该结果只用于决定是否值得运行完整 Dev，不用于最终接受检查点：

| 指标 | SFT v5 | GRPO iteration-001 | 变化 |
|---|---:|---:|---:|
| 类别准确率 | 92.19% | **93.75%** | +1.56pp |
| Macro-F1 | 91.79% | **93.36%** | +1.57pp |
| 类别+等级严格准确率 | 76.56% | **78.13%** | +1.56pp |
| 有害漏成 Harmless | 5.66% | 5.66% | 0pp |
| Harmless 误报有害 | 0% | 0% | 0pp |
| 必要工具首步召回 | 43.75% | 43.75% | 0pp |
| 工具证据利用率 | 84.78% | 82.98% | -1.80pp |
| 平均图片数 | 8.891 | 8.953 | +0.062 |
| 平均奖励 | 3.24131 | **3.31561** | +0.07430 |

64 条轨迹结构审计通过，运行错误、缺失、重复和越界 ID 均为 0。快检中一个 Adaptive 暴力
样本从 SFT 的错误 L1 且两次工具调用，改为正确 L2 且一次工具调用。由于分类、严格准确率
和奖励均改善，且安全漏判未变，iteration-001 已进入完整冻结 Dev1679 验收；正式结果出来
前不访问 Test，也不冻结该候选。

### 12.10 GRPO iteration-001 完整 Dev 拒绝与 iteration-002 调整（2026-09-06）

iteration-001 已完成冻结 Dev 全部 1,679 条，结果缺失、重复、越界和执行错误均为 0，结构
审计通过，但正式验收为 `GRPO_REJECTED`，没有冻结该 Adapter，也没有访问 Test。

| 指标 | 冻结 SFT v5 | GRPO iteration-001 | 变化 |
|---|---:|---:|---:|
| 类别准确率 | 87.97% | 87.31% | -0.66pp |
| Macro-F1 | 87.64% | 86.98% | -0.66pp |
| 类别+等级严格准确率 | 81.24% | 80.64% | -0.60pp |
| 有害漏成 Harmless | 3.73% | **3.35%** | -0.38pp |
| Harmless 误报有害 | 21.61% | 23.82% | +2.21pp |
| 必要工具首步召回 | 60.53% | 60.09% | -0.44pp |
| Direct 原始首步乱调用 | 2.75% | 2.75% | 0pp |
| 工具证据利用率 | 83.73% | 82.32% | -1.41pp |
| 平均图片数 | 7.854 | 7.896 | +0.042 |
| 平均奖励 | 3.31908 | 3.28656 | -0.03252 |

失败集中在 Harmless：召回从 78.39% 降到 76.18%，导致误报门槛失败；与此同时三类有害
漏判略降，说明更新并非整体失效，而是训练分布偏置。iteration-001 的 128 个训练视频按
10 个 D/A/A×类别桶近似均分，Harmless 仅 13 条（10.16%），而冻结 Train 中 Harmless
为 5,072/13,431（37.76%）。因此下一轮不修改冻结奖励、KL、clip 或学习率，只修正 Train
rollout 采样。

iteration-002 从原冻结 SFT v5 重新开始，不继承被拒绝的 iteration-001。清单使用冻结 Train
中 `(D/A/A, category)` 的自然联合比例进行最大余数整数配额：256 个未被 smoke/i1 使用的
Train 视频、每视频 4 条随机真实工具轨迹，共 1,024 条；类别为 Harmless 97、Sexual 82、
Violence 44、Sensitive 33，行为组为 Active 175、Adaptive 26、Direct 55。清单与 Dev、
Test、smoke 和 iteration-001 的视频 ID 交集均为 0；Test 继续封存。清单 SHA-256 为
`d52c5919f5b195837f0a38062c859c1472d959a63ef4492d0aba132aa01d90fb`。

### 12.11 GRPO iteration-002 Dev64 拒绝与受控低学习率重放（2026-09-06）

iteration-002 的 1,024/1,024 条随机轨迹均成功，无运行失败。256 个视频组中 136 组存在
奖励差异，产生 544 条非零优势轨迹；完整批次平均奖励为 3.35910。训练从冻结 SFT v5
重新开始，使用 17 个 optimizer step、学习率 5e-6、KL beta 0.02、clip epsilon 0.2。
训练平均 KL 为 0.000734，平均/最大梯度范数为 0.4523/0.8178，Assistant-only log-prob
审计通过，Test 未使用。

冻结 Dev64 快检结果如下：

| 指标 | SFT v5 | GRPO iteration-002 | 变化 |
|---|---:|---:|---:|
| 类别准确率 | 92.19% | 90.63% | -1.56pp |
| Macro-F1 | 91.79% | 90.15% | -1.64pp |
| 类别+等级严格准确率 | 76.56% | **78.13%** | +1.56pp |
| 有害漏成 Harmless | 5.66% | 7.55% | +1.89pp |
| Harmless 误报有害 | 0% | 0% | 0pp |
| 必要工具首步召回 | 43.75% | 43.75% | 0pp |
| 工具证据利用率 | 84.78% | 82.61% | -2.17pp |
| 平均图片数 | 8.891 | 8.891 | 0 |
| 平均奖励 | 3.24131 | 3.17963 | -0.06169 |

64 条结果完整且结构审计通过；首步路由在 SFT 与 iteration-002 之间 64/64 完全一致，最终
输出只有 3 条发生变化：两条 Violence 等级由错误改正，但一条原本正确的
Sensitive-L2 退化为 Harmless-L0。由于类别、漏判和主选择指标平均奖励同时退化，本候选
不进入完整 Dev，也不冻结。

iteration-003 采用受控重放：继续从原冻结 SFT v5 开始，完整复用 iteration-002 的清单、
轨迹、随机种子、batch 顺序、奖励、KL 和 clip，仅把学习率从 5e-6 降为 2e-6。该设计先
隔离“更新步长过大”因素，不引入另一批随机 rollout 的采样噪声；仍只使用 Train，Test
继续封存。

### 12.12 iteration-003 拒绝、评测稳定性诊断与终态 credit 修正（2026-09-06）

iteration-003 完成相同 17 步受控重放，平均 KL 由 iteration-002 的 0.000734 降为
0.000654；Adapter 相对 SFT 的权重 L2 位移从 0.13517 降为 0.05487，证明学习率降低确实
减小了参数位移。但 Dev64 类别准确率为 89.06%、严格准确率 71.88%、有害漏判 7.55%、
平均奖励 2.96695，均不满足晋级条件，因此不进入完整 Dev。

为排除 vLLM 并发下临界 token 的数值漂移，同一服务、同一 workers=8 配置额外重跑了未
变化的冻结 SFT v5。64 条中只有 2 条 L1/L2 边界样本翻转，类别输出 64/64 一致；汇总类别
准确率、严格准确率和有害漏判数与原 SFT 基线完全一致。由此确认 i2/i3 新增的类别错误不
是评测随机波动，只有个别 severity 边界需要在解释小样本结果时注明。

训练审计发现，iteration-002 的 307 条正优势轨迹中有 75 条是“模型首步错误/不充分回答，
Python 安全框架强制扫描后最终答对”。此前 GRPO 把终态奖励施加到轨迹中所有 Assistant
文本，因此这些轨迹会同时正向强化被框架救回的早期错误回答，属于 credit assignment
污染。iteration-004 不修改冻结标量奖励，只改变其梯度归因范围：完整因果历史仍全部输入，
但终态任务优势只计算最后一次 Assistant 决策的 log-prob；单回合 Direct 轨迹仍正常训练。
该模式由 `--logprob-scope final_assistant` 显式记录，16 项奖励/采样/credit 单测全部通过。
路由能力本轮由冻结 SFT 与 reference KL 保持；若终态修正通过，再单独实现基于 route
component 的首步动作 credit，不能再把一个终态标量无差别分给所有回合。

### 12.13 iteration-004 终态 credit 受控实验拒绝（2026-09-06）

iteration-004 完整复用 iteration-002 的 256 个 Train prompt、1,024 条真实工具轨迹、
随机种子、batch 顺序、2e-6 学习率、KL beta 与 clip，仅将 terminal trajectory advantage
从全部 Assistant 回合收缩到最后一个 Assistant 回合。17 个 optimizer step 全部完成，
训练平均 KL 为 0.000400，平均/最大梯度范数为 0.5265/0.8622；结构审计通过，Test 未使用。

冻结 Dev64 快检结果如下：

| 指标 | SFT v5 | GRPO iteration-004 | 变化 |
|---|---:|---:|---:|
| 类别准确率 | 92.19% | 92.19% | 0pp |
| Macro-F1 | 91.79% | 91.79% | 0pp |
| 类别+等级严格准确率 | 76.56% | 75.00% | -1.56pp |
| 有害漏成 Harmless | 5.66% | 5.66% | 0pp |
| Harmless 误报有害 | 0% | 0% | 0pp |
| 必要工具首步召回 | 43.75% | 43.75% | 0pp |
| Direct 原始首步乱调用 | 12.50% | 12.50% | 0pp |
| 工具证据利用率 | 84.78% | 82.98% | -1.80pp |
| 平均图片数 | 8.891 | 9.016 | +0.125 |
| 平均奖励 | 3.24131 | 3.19684 | -0.04447 |

候选与 SFT 只有 3 条最终标签发生变化，且全部是 Sexual Content 的 L1/L2 边界翻转：1 条
由错误改正，2 条由正确退化；四分类输出完全没有净变化。因此该候选不进入完整 Dev，也不
冻结。实验同时验证了诊断：只训练终态可以阻止框架救回的早期错误获得任务 credit，但不会
改善首步工具路由。下一轮必须采用分阶段的因果 credit：最终答案 token 接受终态任务回报；
首步路由/检索 query token 接受独立的 route return，且框架救回的非法或过早直答必须得到
负向首步回报，不能因最终被救回而获得正优势。

### 12.14 iteration-005 因果 route/query credit 实验拒绝（2026-09-06）

iteration-005 继续受控复用同一批 Train 轨迹，从冻结 SFT v5 独立启动。终态答案 advantage
权重设为 0；首步自主扫描或策略允许的明确 L2 直答继承 outcome-conditioned route return，
被框架救回的非法/过早 `ANSWER` 首步固定得到 -5。离线审计得到 132 个有路由差异的视频组、
528 条非零路由优势轨迹，框架救回且获得正 route advantage 的轨迹为 0。首步完整 JSON 与
query 接受 route credit，平均 121.0 个 route token；最终答案不接受策略 advantage，但保留
reference KL。17 步平均 KL 为 0.001017，训练结构正常，Test 未使用。

Dev64 中平均帧数从 8.891 降至 8.813、证据利用率从 84.78% 升至 88.89%，但必要工具召回和
Direct 路由均未变化；类别准确率从 92.19% 降到 85.94%，新增 3 条有害漏判，平均奖励下降
0.38278，故不进入完整 Dev。逐条变化表明，训练完整 query JSON 仍通过共享 LoRA 扰动了工具
后分类；同时每条轨迹约 440 个被选择的 Assistant token 会稀释 route token 的 policy loss。

下一轮保持冻结 scalar reward 与 causal route return 不变，只把 route credit 收缩到首步
`ANSWER/JOINT_SEMANTIC_SCAN` 动作值 token，并在非零 credit token 上求均值；其余最终答案
token 仅接受 KL 约束。训练同时保存多个中间 checkpoint，先用冻结 Router Dev 选择候选，
再运行完整 Agent Dev64，避免只看最后一步并减少无效全链路评测。

### 12.15 iteration-006 动作 token credit 与分阶段推理诊断（2026-09-06）

iteration-006 继续从冻结 SFT v5 独立启动并复用 iteration-002 的 256 个 Train 视频、1,024
条真实轨迹。因果 route return 与冻结奖励均未修改，但 route credit 从首步完整 JSON 收缩到
`action` 字段值 token；policy loss 只在非零 credit token 上求均值。学习率降为 5e-7，KL
beta 为 0.02，terminal advantage 权重为 0，并保存 step-4/8/12/16 与最终 step-17。132/256
个组存在 route return 差异，528 条 rollout 获得非零 route advantage；17 步平均 KL 为
0.000750，原始平均/最大梯度范数为 9.528/27.013，实际更新前按 1.0 裁剪。Test 未使用。

冻结 Router Dev256 上，step-4/8 的必要工具首步召回由 43.75% 升至 45.3125%（+1.5625pp），
Direct 无意义调用保持 4.6875%，结构化动作仍为 100%；回答类别准确率只变化 -0.013pp，严格
回答准确率变化 -0.173pp。因此动作 token 确实收到并响应了 GRPO 信号，但改变量很小。

同一 Adapter 直接承担完整 Agent 所有回合时，Dev64 仍未通过：step-4 的类别/严格准确率为
85.94%/71.88%，step-16 为 90.63%/78.13%，step-17 为 89.06%/76.56%；对应平均奖励分别为
2.76875、3.17902、3.10069，均低于冻结 SFT 的 3.24131。step-16 虽修正一个 Violence 等级，
也新增一个 Violence→Harmless 漏判。这证明即使梯度只落在动作 token，共享 LoRA 参数仍会
改变后续视觉分类分布，不能把 Router 改进和 post-tool 推理安全地放在同一 Adapter 回合中。

随后实现分阶段模型选择：初始路由使用 GRPO Adapter，扫描参数与工具后的视觉判断使用冻结
SFT v5。23 项奖励、credit 与阶段路由单测通过，每个 `model_turn` 显式记录实际模型。第一次
仅切换 post-tool 模型仍失败，因为 GRPO 首轮生成的 query 改变了 SigLIP 候选；第二次让 SFT
同时生成 query 后，独立重跑仍出现 query/生成漂移：GRPO 与 SFT 首步动作 64/64 相同，却有
3 条最终结果变化（1 条改善、2 条退化）。这不是可归因于路由的增益，故仍不进入完整 Dev。

后续验收改为**配对因果评测**：在同一视频上分别取得冻结 SFT 与 GRPO 的首步决策；若执行
分支相同，则共享同一次 SFT query、真实工具结果和 post-tool 判断；只有路由确实不同的样本
执行不同分支。这样下游采样/生成噪声不会掩盖少量 route action 变化。当前先在完整冻结 Dev
1,679 条上运行 step-4 首步配对审计，再决定是否需要新的 Train-only iteration。Test 继续封存。

### 12.16 iteration-006 step-4 完整路由与反事实 Agent 验收（2026-09-06）

step-4 已完成冻结 Dev 1,679/1,679 条首步动作评估，唯一 ID 和执行错误检查均通过。
相对冻结 SFT v5，合法结构化首步由 98.57% 升至 98.75%，必要工具首步召回由 60.53% 升至
61.18%（+0.658pp），Direct 首步工具率由 2.75% 升至 3.03%。完整集合仅有 16 条首步
`(action, category, severity)` 签名变化，其中真正改变安全框架执行分支的只有 4 条：
`DIRECT->SCAN` 3 条，`SCAN->DIRECT` 1 条。

为了不把 1,675 条相同分支的重复 SigLIP 检索或 VLM 生成漂移误算成 GRPO 效果，验收使用
冻结反事实协议：相同 `SCAN` 分支复用此前冻结且已审计的 SFT v5 真实工具结果；GRPO 合法
Direct 使用本次缓存首步答案；只有 3 条新增 `DIRECT->SCAN` 分支重新执行真实
`JOINT_SEMANTIC_SCAN` 和冻结 SFT post-tool 判断。完整反事实审计记录 1,679 个唯一 ID、0
错误、0 因果/来源违规，三条新增扫描均真实执行，Test 未使用。

快速首步脚本原先只检查扫描动作名，没有验证完整三类 `queries_by_risk`；统一到真实 Agent
口径后，step-4 有 21 条首步结构不完整（其中 20 条为扫描 query 映射不完整），由框架固定
风险查询安全兜底，因此不影响已执行的最终分类，但不能计作合法自主工具调用。缓存结果已
原子修复并保留修复前备份，相关修复清单和 SHA 记录在对应输出目录。

最终完整 Agent 指标保持类别准确率 87.969% 与 Macro-F1 87.643% 不变；严格准确率从
81.239% 升至 81.298%（仅净增 1 条，+0.060pp）；必要工具首步召回 +0.658pp；平均图片数
增加 0.009。进一步的奖励来源审计发现，不完整扫描 query 由框架固定查询救回后不应获得
自主路由奖励；不改奖励数值，仅修正来源判定并重新计算后，冻结 SFT 基线奖励为 3.315145，
step-4 为 3.318091（+0.002946）。所有安全硬门槛通过，但
奖励增量未达到预注册的 +0.003，且没有任何预注册“有意义改善”项达标，故结论为
`GRPO_REJECTED`，不冻结 step-4。当前继续用相同冻结 Dev 协议评估 step-16；Test 保持封存。

### 12.17 iteration-007 修正奖励血缘与启动前审计（2026-09-06）

iteration-006 训练时记录的 `grpo_reward.py` SHA-256 为
`04b08ea264ff39cf5801882187ca4de780748eead575bdc296f8420020520d29`，早于严格
`JOINT_SEMANTIC_SCAN` query schema 与自主调用来源判定修复；因此 iteration-006 的各检查点
只用于定位合适步数，不能作为最终可复现检查点冻结。iteration-007 必须从冻结 SFT v5 独立
重训，并记录修正后的奖励代码 SHA-256：
`6554431b618e753cb91f04926bd01a386b91c9fec4a02438a8fd631f049a84e9`。

为隔离奖励来源修复的影响，iteration-007 首轮严格复用 iteration-006 的 256 个 Train-only
视频和 1,024 条真实工具 rollout；manifest 与 rollout SHA-256 分别为
`d52c5919f5b195837f0a38062c859c1472d959a63ef4492d0aba132aa01d90fb` 和
`8a1bfa9877d80f57c340e2fc474f966b0ba9739ffd72fc52c1e66371dd7c6f73`。修正奖励的离线审计为：
1,024/1,024 条可训练、0 基础设施错误、824 条类别和等级完全正确，平均奖励 3.325117；
256 个 prompt group 中 160 个总奖励非零方差，640 条 rollout 有非零总奖励优势。训练仍采用
5e-7 学习率、KL beta 0.02、action-value token route credit、terminal advantage 权重 0，
保存 step-4/8/12/16 与最终 step-19（修正奖励产生 149 个有效路由组，按每步 8 组共 19
个 optimizer step）。管线启动前会校验上述三个 SHA，并在 iteration-006 诊断评估仍运行
时拒绝抢占 GPU。Test 继续封存。

### 12.18 iteration-006 step-16 诊断通过，但不作为最终冻结件（2026-09-06）

step-16 完成冻结 Dev 1,679/1,679 条严格首步评估。相对 SFT v5，合法结构化首步从
98.57% 升至 99.52%，必要工具首步召回从 60.53% 升至 62.94%（+2.412pp），Direct
原始首步工具率从 2.75% 小幅升至 3.03%。原始 `(action, category, severity)` 签名改变
42 条；进一步按安全框架真正执行的分支计算后，仅 14 条为 `DIRECT->SCAN`，其余为
`DIRECT->DIRECT` 651 条、`SCAN->SCAN` 1,014 条，没有 `SCAN->DIRECT`。比较工具现已同时
报告原始动作名和有效执行分支，防止把本就会被框架保护性扫描的 L1/无效直答误计为新增调用。

反事实完整 Agent 只对上述 14 条新增分支重新执行真实 SigLIP、补帧和冻结 SFT v5 工具后
判断；其余 1,665 条复用已审计分支。最终 1,679 条唯一 ID、0 错误、0 分支来源违规，轨迹
结构审计通过。相对冻结 SFT v5：类别准确率和 Macro-F1 不变，严格准确率 +0.060pp，
有害漏判与 Harmless 误报均不变，必要工具首步召回 +2.412pp，证据利用率 -0.459pp，
平均图片数 +0.057，平均奖励从 3.315145 升至 3.319127（+0.003982）。全部安全硬门槛通过，
奖励增量超过 +0.003 且必要工具召回提升超过 +1pp，预注册结论为 `GRPO_ACCEPTED`。

该结果证明 step-16 更新强度可用，但 iteration-006 的训练血缘记录的是旧奖励代码哈希，
不满足最终可复现冻结条件。因此它只作为 iteration-007 的步数选择依据；最终检查点必须由
修正奖励代码从冻结 SFT v5 重训，并重新完成同一冻结 Dev 验收。Test 仍未使用。

### 12.19 iteration-007 正式接受并冻结 GRPO v1（2026-09-06）

iteration-007 从冻结 SFT v5 独立重训，严格复用 256 个 Train-only 视频和 1,024 条真实工具
rollout，但使用修正后的奖励代码 SHA-256
`6554431b618e753cb91f04926bd01a386b91c9fec4a02438a8fd631f049a84e9`。离线奖励审计为
1,024/1,024 条可训练、0 基础设施错误、824 条类别和等级完全正确，平均奖励 3.325117。
训练得到 149 个有效路由组、596 条非零优势 rollout，共执行 19 个 optimizer step；学习率
5e-7、KL beta 0.02、action-value token route credit、terminal advantage 权重 0。平均 KL
为 0.000427，原始平均/最大梯度范数为 3.805/15.109，实际更新前按 1.0 裁剪。

冻结 Router Dev256 同口径比较 step-4/8/12/16/19 后，按预设规则选择 step-16：合法动作
100%，必要工具首步召回从 43.75% 升至 45.3125%，Direct 多余调用保持 4.6875%；回答类别
准确率变化 -0.027pp，回答类别+等级准确率从 78.862% 升至 82.645%。随后 step-16 完成完整
冻结 Dev 1,679/1,679 条首步评估，唯一 ID 1,679、错误 0。相对 SFT v5，合法结构化动作
由 98.571% 升至 99.940%，必要工具首步召回由 60.526% 升至 62.939%，Direct 原始首步工具
率由 2.755% 升至 3.030%。36 条原始签名发生变化；按安全框架真实执行分支计算为：
`DIRECT->DIRECT` 656 条、`SCAN->SCAN` 1,013 条、`DIRECT->SCAN` 9 条、`SCAN->DIRECT`
1 条。

配对反事实 Agent 验收只对 9 条新增扫描执行真实 SigLIP、补帧和冻结 SFT v5 复判；1,013
条原有扫描分支复用已审计 SFT v5 结果，657 条最终直答使用缓存的 GRPO 首轮结果。分支来源
审计为 1,679 个唯一 ID、0 错误、0 来源违规。最终完整 Agent 结果如下：

| 指标 | 冻结 SFT v5 | **GRPO v1 step-16** | 变化 |
|---|---:|---:|---:|
| 类别准确率 | 87.969% | **87.969%** | 0pp |
| Macro-F1 | 87.643% | **87.643%** | 0pp |
| 类别+等级严格准确率 | 81.239% | **81.298%** | +0.060pp |
| 有害漏成 Harmless | 3.732% | **3.732%** | 0pp |
| Harmless 误报有害 | 21.609% | **21.609%** | 0pp |
| 必要工具首步召回 | 60.526% | **62.939%** | +2.412pp |
| Direct 原始首步工具率 | 2.755% | **3.030%** | +0.275pp |
| 工具证据利用率 | 83.728% | **83.464%** | -0.264pp |
| 平均图片数 | 7.854 | **7.885** | +0.031 |
| 平均奖励 | 3.315145 | **3.321211** | +0.006066 |

全部预注册安全、结构与成本硬门槛通过；平均奖励增量超过 +0.003，且必要工具首步召回提升
超过 +1pp，结论为 `GRPO_ACCEPTED`。训练器、奖励、collator、manifest 与 rollout 的训练时
SHA 均与冻结时文件一致；manifest 生成配置进一步通过所选 ID 顺序、prompt 数、rollout 数
及冻结 Train SHA 四项检查。最终复合检查点冻结到
`outputs/best_grpo_qwen35_4b_agent_v1/`：初始路由使用 GRPO step-16 Adapter，query 生成及
所有工具后判断使用不可变 SFT v5 Adapter。项目测试 52/52 通过，Test 继续封存。

### 12.20 GRPO v2 多阶段真实工具链与 Temporal Router 修复（进行中，2026-09-07）

GRPO v2 将 Agent 扩展为真实三阶段因果轨迹：初始 4 帧后由模型选择直接回答或
`JOINT_SEMANTIC_SCAN`；真实 SigLIP 候选帧返回后，由模型再次选择 `ANSWER` 或
`TEMPORAL_EXPAND`；若扩帧，则模型在真实前后文帧返回后输出最终分类与等级。统一后的硬契约
为：任一风险已是 `PRESENT` 时立即回答；三类均 `ABSENT` 时回答 Harmless-L0；只有没有
`PRESENT` 且至少一类为 `INSUFFICIENT` 时才能扩帧。相关工具/奖励/阶段 credit 单测共
42 项通过。

warm-up v5 使用 56 条真实时序正例与 56 条可直接回答负例训练完整
`TEMPORAL_EXPAND` payload。状态集上能产生合法扩帧，但冻结 Dev64 上三个候选均被拒绝：
父模型类别/严格正确为 58/47、Harmless 误报为 0；候选最佳类别/严格正确仍为 58/47，
有害漏判从 5 降为 0，但 Harmless 误报升为 4。逐条审计显示共享 LoRA 把撑杆跳、普通人物、
正常脱衣等模糊画面推成 L1 风险，说明完整 payload 监督污染了分类边界。

warm-up v6 保持同一 56/56 样本，仅把监督收缩到 JSON `action` 字段，28 步训练正常完成。
严格 Dev24 状态门禁的最好 checkpoint 为 step-28：route accuracy 79.17%，Temporal recall
62.5%，Answer recall 87.5%，非法/错误 3 条；没有 checkpoint 同时满足 Temporal recall>0、
Answer recall>=93.75% 和非法/错误<=2，因此 v6 在进入 Dev64 前被拒绝。

当前 warm-up v7 不降低门槛，保留全部 56 个时序正例，并从 Train-only 池增加到 112 个
可直接回答负例，形成 1:2 的扩帧/回答先验。共 168 个唯一 Train ID，Dev/Test 重叠为 0，
所有状态只监督动作字段且未来扩帧结果泄漏为 0。训练、状态门禁、真实 Dev64、256 rollout
smoke 和 2,048 视频正式 GRPO 均由独立可续跑管理脚本串联；只有冻结 Dev64 同时通过分类、
分级、漏判、误报、真实自主扩帧、结构和帧成本门槛后，才会放行后续训练。Test 未运行推理，
继续封存。

状态评测过程中进一步发现历史冻结状态的提示文本仍将 `TEMPORAL_EXPAND.window` 限制为
0.5–8 秒，而项目原始函数调用契约、Qwen 后端及实际工具均为 0.5–15 秒；这会把模型生成的
10–13 秒合理上下文窗口错误计作非法动作。运行时代码已统一恢复为 0.5–15 秒，同时使用
`migrate_temporal_router_state_contract.py` 只迁移同一批 Dev24 状态中的 post-scan 指令文本；
样本 ID、视频、时间戳、真实 SigLIP 观察及监督目标均保持不变，迁移审计记录源/目标 SHA。
发现提示与校验器未同步后产生的中间状态评测已移动到备份目录，不参与 checkpoint 选择；
v7 正在统一契约下从头重跑状态门禁。

统一契约下的 Dev24 状态门禁已完成。step-34 在 8 个真实时序目标和 16 个可直接回答目标上取得
79.17% 总路由准确率、50.00% Temporal recall、93.75% Answer recall，非法/错误为 0，按预注册
门槛优于其他候选并被选中。随后在同一批 168 个 Train-only 状态上用迁移后的 15 秒提示回放：
总路由准确率 79.76%，Temporal recall 42.86%，Answer recall 98.21%，非法/错误 3，API 错误 0。
该结果证明模型已经学会自主请求真实时序上下文，同时没有在训练池上形成普遍的无意义扩帧。

Dev64 不复用历史 8 秒契约下的父模型缓存。冻结的 `router_dev64_v1.jsonl` 样本不变，但接受的
GRPO v1 父模型和 step-34/20/24 三个候选均在当前 0.5–15 秒契约下重新执行完整真实工具链；
只有相对该同口径父模型通过分类、等级、漏判、误报、时序动作、结构及帧成本门槛，才会生成
`DEV64_COMPLETE` 并放行 256 条 smoke rollout。Test 继续封存。

正式 GRPO 启动前的 credit 审计还发现：旧 `stagewise_causal_v2` 只把模型选择
`TEMPORAL_EXPAND` 的 post-scan 回合标成 temporal route，而同一状态下选择 `ANSWER` 的对照
动作被误当作普通最终回答。这样 temporal advantage 只能强化扩帧，不能通过组内相对优势抑制
无效扩帧。现已让 multistage runner 显式持久化每个模型回合的 `action_stage`，并将 post-scan
的 `ANSWER` 与 `TEMPORAL_EXPAND` 都归入 `temporal_route`；post-scan ANSWER 作为终止动作时，
同一 Assistant span 同时接受 temporal 与 terminal credit。新增反例测试后相关测试为 42/42，
该修复只作用于尚未开始的 smoke/正式 GRPO，不改动已经运行的冻结 Dev64 推理结果。

### 12.21 Severity SFT + route-weighted GRPO v3（运行中，2026-09-16）

本轮独立运行目录为
`outputs/severity_sft_grpo_v3_route_weighted/`，目标是对重新整理的 **4,400** 条训练候选完成
增量 SFT、真实工具链 GRPO 与冻结 Dev1679 验收。SFT 使用由这些候选展开的 **5,719** 个因果
决策状态；所有初始状态仅含 4 张非黑均匀帧，工具后状态只含此前动作和真实工具返回，loss 仅
作用于目标 Assistant JSON。SFT 候选 checkpoint 在 Dev64 上选择 `sft-severity-s450`，其冻结
Dev1679 结果为：类别准确率 **88.922%**、类别+等级准确率 **81.120%**、有害漏判 **75** 条、
Harmless 误报 **87** 条。

GRPO 使用 Train-only 的 800 个视频提示、每提示 4 个温度采样轨迹（共 3,200 条）；采样种子为
`20260915`，`top_k=3`，`temperature=0.7`，`top_p=0.95`，最大输出 4,096 token。只保留同一
视频中至少一条类别+等级正确且至少一条不正确的四轨迹组；按最终类别平衡，最多 512 组。Test ID
在 manifest 和 rollout 筛选时均被显式排除。

采样器使用一个进程共享的 `JointRiskSemanticLocator`，令同一视频的 4 个 rollout 复用候选帧
embedding，避免长视频重复解码造成内存累积；后续 Dev 多 checkpoint 评估启用持久 embedding
缓存，但不改变帧、提示、检索或模型语义。若最终视觉判断 JSON 缺少对已见帧的证据时间戳，只会
在相同帧和因果历史下执行一次格式修复回合；不注入标签，仍由相同证据校验器决定是否接受。

运行完成后由 `audit_severity_sft_grpo_v3.py` 自动验证：3,200 rollout 的 JSON/唯一 ID/错误数、
有效组完整性、Test 隔离、训练 adapter/checkpoint、SFT 与 GRPO 各 1,679 条冻结 Dev 覆盖，以及
`final_comparison.json` 中的 SFT/GRPO 同口径比较。

最终 GRPO 训练在 393 个有效提示组（1,572 条 rollout）上完成 50 个 optimizer updates；Dev64
从 step 10/20/30/40/50 和 final adapter 中选择 `grpo-severity-s30`。在同一冻结 Dev1679 上，结果为：

| 阶段 | 类别准确率 | 类别+等级准确率 | 有害漏判 | Harmless 误报 |
|---|---:|---:|---:|---:|
| SFT (`sft-severity-s450`) | 88.922% | 81.120% | 75 | 87 |
| SFT + GRPO (`grpo-severity-s30`) | **89.160%** | 81.120% | **73** | 87 |

因此本轮 `grpo_accepted=true`：类别+等级准确率未低于 SFT，且有害漏判减少 2 条。最终审计
`outputs/severity_sft_grpo_v3_route_weighted/final_artifact_audit.json` 已全部通过：两套 Dev 结果各
1,679 条且唯一、无结果错误、无 Test 样本；原始 rollout 为 3,200 条、无重复 ID、无错误。

完整 Dev 首轮发现本地 vLLM 对整个聊天历史（而不只是当前回合）施加 16 张图片上限：在
`4 initial + 8 locator + 4 temporal` 后，最终格式修复回合会达到 20 张。已修复为保留全部因果文本、
按新近程度压缩历史视觉块、并保证单次请求不超过 16 图；修复有单元测试覆盖。首轮唯一受影响的
样本以相同视频和因果状态在 retry 中重跑成功，不影响最终 Dev 的 1,679 条无错误覆盖。

### 12.22 Open Test Supplements v2 独立测试补充集（2026-09-20）

新增独立测试候选目录：
`data/open_test_supplements_v2/`。本轮没有下载新视频，下载流量为 **0 B**；全部样本来自主机
已有媒体的筛选与 Qwen3.7-Plus 首轮打标。该目录只作为后续独立测试/开放候选评估使用，**不得
并入 SFT、GRPO、Dev 或 Test 训练闭环**。

本地首轮成功标注批次如下：

| 批次 | 成功标注 |
|---|---:|
| 首批补充 | 405 |
| 第二批（已暂停） | 34 |
| 长暴力补充 | 35 |
| 短暴力测试补充 | 278 |
| 去重后合计 | **752** |

752 条本地首轮 Teacher 标签分布：

| 类别 | 数量 |
|---|---:|
| Harmless | 65 |
| Sexual Content | 62 |
| Violence & Dangerous | 612 |
| Sensitive Figures & Ideologies | 13 |

但当前 `manifest.jsonl` 中只有 **40** 条满足公开候选 provenance gate（MIT 或 CC BY 4.0），其
公开候选类别分布为 Harmless 8、Sexual 28、Violence 2、Sensitive 2。另有 **712** 条写入
`release_blockers.jsonl`，原因是仍需 license/provenance review。`internal_supplement_index.jsonl`
保留本地索引，`summary.json` 记录上述统计。

标签状态：Qwen3.7-Plus uniform-frame first pass，尚未经过人工或独立第二 Teacher 复核，因此不能
作为 benchmark ground truth。若后续用于论文开放测试，必须先完成：

1. 来源与再分发许可复核；
2. 独立 Teacher 或人工标签审计；
3. 与训练/Dev/Test 样本 ID、父视频、source_group 的泄漏检查。
# THVL-Bench 外部评测计划（2026-09-22）

已冻结 THVL-Bench 作为不参与训练和模型选择的长视频外部评测。目标是测量 Agent 对人工标注有害时间段的语义检索命中率、最终观察/证据召回率、危险放行率及取帧成本。完整执行方案、类别映射、对比实验和验收条件见 [THVL_EXTERNAL_EVAL_PLAN.md](THVL_EXTERNAL_EVAL_PLAN.md)。当前校准器任务结束并释放 GPU 后执行，避免并发 vLLM 争抢显存。

## 无泄漏轻量动作校准器（2026-09-23）

正式校准器只使用线上可获得的 Agent 轨迹特征。Logistic Regression 在 Train 2,400 条轨迹上拟合，Isotonic 与动作阈值只在 Dev 1,679 条上拟合，冻结 Test 1,679 条用于最终评测。`teacher_consensus`、`severity_mode`、`daa_group`、`tool_trigger_source`、`tool_signature` 和来源数据集等标注构造信息均未作为特征。

Dev 选出的安全工作点为 `min_safe=0.2, max_unsafe=0.2`。Test 结果：危险 ALLOW 从 59 降至 49，减少 10 条（16.95%）；额外转 REVIEW 31 条（1.85%）；`strict_or_review` 从 81.12% 降至 79.69%，`safe_action` 从 83.14% 降至 81.95%，overblock 保持 67。Test 的 unsafe-ALLOW 概率 AUC 为 0.8916，Brier 从 0.1547 改善至 0.0302，ECE 从 0.2119 改善至 0.0111。

结论：校准器能在独立 Test 上识别部分危险放行并将其转为 REVIEW，但不是无代价的全面提升；论文中应将其报告为安全—自动化折中，并给出完整阈值曲线，而不是只报告单一准确率。

## THVL 厚 Harness 同预算对照（2026-10-11）

实验目录：`video_detection_agent/outputs/thvl_thick_matched_20261011/`。冻结输入为 THVL-Bench 的
448 条视频和 1,097 段人工有害时间区间；该数据只作外部评测，不进入训练、奖励或 checkpoint 选择。

配置：Qwen3.5-4B + `paper-grpo` LoRA，初始 4 帧，最多 21 轮决策、20 次工具调用、128 张唯一帧；
每次时序扩展 4 帧，检索 Top-3，候选步长 4 秒、最多 64 帧，temperature=0、seed=42。厚 Harness
启用了初始 guard、固定查询回退和风险排除停止逻辑，因此本结果不是单变量 guard 消融。

验收：448/448 条通过结构校验，结果错误 0；`COMPLETE`、`VERIFIED.json` 和 `COMPARISON.json`
均已落盘。

| 指标 | 厚 Harness |
|---|---:|
| 工具调用 | 378/448（均为 `JOINT_SEMANTIC_SCAN`） |
| 触发来源 | 模型 213，框架 165 |
| 平均有效帧 | 9.94 |
| Exact 视觉/文本 observed recall | 55.85% |
| ±3 秒视觉/文本 observed recall | 64.26% |
| Exact evidence recall | 27.66% |
| ±3 秒 evidence recall | 30.21% |
| ALLOW / REVIEW / BLOCK | 97 / 191 / 160 |
| 同预算命中（厚 Agent / 均匀对照） | 525 / 580（940 个视觉/文本区间） |

结论：厚 Harness 比薄多轮的 Exact observed recall（37.23%）明显提高，但仍低于固定均匀 12
帧（67.45%）；平均帧数为 9.94，且本批没有实际使用 `TEMPORAL_EXPAND`。该结果用于外部取证
对照，不作为自定义 L1/L2 真值。

机器可读结果：

- `video_detection_agent/outputs/thvl_thick_matched_20261011/COMPARISON.json`
- `video_detection_agent/outputs/thvl_thick_matched_20261011/thick-multiloop/metrics_exact/summary.json`
- `video_detection_agent/outputs/thvl_thick_matched_20261011/thick-multiloop/metrics_tolerance3/summary.json`
- `video_detection_agent/outputs/thvl_thick_matched_20261011/REPORT.md`

### THVL 视频级引用证据与描述复核（2026-10-12，进行中）

不做四分类映射；按THVL原始人工区间核对真实已见引用，再用未微调本地Qwen3.6-27B独立看图
核对描述。408条视觉/文本适用视频中，至少一处引用精确命中：原厚版247（60.54%）、薄多轮119
（29.17%）、新厚多轮241（59.07%）、均匀12帧171（41.91%）。这些不是描述成立率或分类准确率。

四组2,682条引用去重为2,649个描述审核任务，优先1,075个已见且命中的任务。描述支持和有害事实
支持分别记录；模型审核仅作弱参考。首个管线测试发现审核员的影视豁免及未见即矛盾问题，已修正
审核提示词并将v2与测试产物隔离。动态结果及完整说明位于
`video_detection_agent/outputs/thvl_evidence_claim_audit_20261012_v2/REPORT.md`与`SUMMARY.json`。
