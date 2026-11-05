# THVL-Bench 外部评测计划

更新时间：2026-09-22

## 1. 任务定位

THVL-Bench（Temporal Harmful Video Localization Benchmark）用于评测长视频中稀疏有害片段的发现与时序定位能力，不作为当前四分类训练数据，也不用于选择或调节模型。

本实验冻结当前最佳 Video Safety Agent，在 THVL-Bench 全量视频上进行一次外部评测，验证核心研究问题：

> 在有限取帧预算下，Agent 能否通过语义检索和时序扩展找到人工标注的有害时间段，并避免将含有有害片段的长视频错误放行？

数据位置：

- THVL-Bench：`video_detection_agent/external_benchmarks/THVL-Bench`
- 运行目录：`video_detection_agent/outputs/thvl_external_eval_v1`
- 管理脚本：`video_detection_agent/manage_thvl_v3_external_eval.sh`

## 2. 数据特点

- 450 个有人工标注的真实长视频；本地数据目录可能包含额外元数据或文件，因此以解析后的 gold manifest 为准。
- 视频时长约 3–40 分钟。
- 1,099 个人工标注的有害片段。
- 每个片段含开始/结束时间戳、风险类别以及贡献模态。
- 11 类风险：Information Harm、Verbal Abuse、Hate、Bias、Addiction Harm、Sexual Harm、Physical Harm、Violence、Blood/Gore、Criminal Activity、Danger。

## 3. 与当前四分类的映射

仅对语义兼容的类别计算映射分类准确率：

| THVL 类别 | 当前类别 | 处理方式 |
|---|---|---|
| Sexual Harm | Sexual Content | 直接映射 |
| Physical Harm | Violence & Dangerous | 直接映射 |
| Violence | Violence & Dangerous | 直接映射 |
| Blood/Gore | Violence & Dangerous | 直接映射 |
| Danger | Violence & Dangerous | 直接映射 |
| Criminal Activity | Violence & Dangerous | 逐条按已有映射评测并单独报告限制 |
| Hate / Bias / Verbal Abuse / Information Harm / Addiction Harm | THVL-Other | 不强制映射，不计入四分类准确率 |

THVL 不用于评测 Sensitive Figures & Ideologies 的完整能力，也不把无法映射的类别错误塞入该类别。

## 4. 冻结条件

- 冻结基础模型：`Qwen3.5-4B`。
- 冻结当前最佳 LoRA：`outputs/action_aware_grpo_v1_review_conflicts/grpo/update/adapter`。
- 固定初始采样为 4 帧。
- 固定 `JOINT_SEMANTIC_SCAN` 的 `top_k=3`。
- 固定现有 `TEMPORAL_EXPAND` 策略。
- 不根据 THVL 结果选择 checkpoint、调阈值或修改提示词。
- THVL 结果只能用于外部泛化评测和后续 Train 难例方向分析。

## 5. 主要指标

### 时序取证指标

1. `segment_search_recall`：SigLIP 检索时间点命中人工风险片段的比例。
2. `segment_observed_recall`：Agent 最终看过的帧命中人工风险片段的比例。
3. `segment_evidence_recall`：最终证据时间戳命中人工风险片段的比例。
4. `video_any_search_hit_rate`：每个视频至少命中一个风险片段的比例。
5. 默认允许人工区间前后 3 秒容差，同时补充 0 秒和 5 秒容差敏感性结果。

### 安全决策指标

1. `unsafe_allow_rate`：含人工标注风险片段的视频被最终 ALLOW 的比例。
2. `REVIEW`、`BLOCK` 比例。
3. 可映射子集的类别准确率。

### 成本指标

1. 平均观察帧数。
2. 平均工具调用次数。
3. 不同帧预算下的风险片段召回率，形成准确性—成本曲线。

## 6. 对比实验

最终论文至少报告以下方法：

| 方法 | 用途 |
|---|---|
| Uniform-4 | 固定 4 帧、不调用工具的低成本基线 |
| Uniform-12 或同预算均匀采样 | 判断收益是否仅来自增加帧数 |
| 固定 SigLIP 检索 | 判断语义检索自身的贡献 |
| 当前 Agent | 验证可训练路由与时序扩展的综合收益 |
| Agent without TEMPORAL_EXPAND | 验证邻域补帧的贡献 |

所有方法应在相同视频、相同人工标注和可比帧预算下评测。

## 7. 执行顺序

当前轻量校准器任务仍占用 GPU。该任务完成并释放 GPU 后执行：

```bash
bash video_detection_agent/manage_thvl_v3_external_eval.sh start
```

实时查看：

```bash
watch -n 10 'bash video_detection_agent/manage_thvl_v3_external_eval.sh status'
```

查看详细日志：

```bash
bash video_detection_agent/manage_thvl_v3_external_eval.sh logs
```

主要结果文件：

- `video_detection_agent/outputs/thvl_external_eval_v1/v3-action-aware-final/results.jsonl`
- `video_detection_agent/outputs/thvl_external_eval_v1/v3-action-aware-final/thvl_metrics/summary.json`
- `video_detection_agent/outputs/thvl_external_eval_v1/v3-action-aware-final/thvl_metrics/per_video.jsonl`

## 8. 验收与后续决策

完成条件：

- 全量 gold manifest 中的视频均有有效结果或明确记录解码/基础设施失败。
- 汇报所有主要时序、安全和成本指标。
- 失败样本按“未检索到片段、检索到但判断错误、工具后仍放行、不可映射类别、解码失败”分组。
- 不用 THVL 结果回头选择当前 checkpoint。

结果用于决定后续 Train 难例补充方向：

- 检索召回低：优化动态 query、候选覆盖与长视频分段策略。
- 检索命中但证据召回低：优化 `TEMPORAL_EXPAND` 和证据引用。
- 看到了风险仍 ALLOW：补充可靠危险漏放难例并调整安全奖励。
- 帧数过高但召回无提升：加强成本约束和停止策略。

## 9. 论文中的定位

THVL-Bench 作为完全独立的长视频外部测试集，用于支持以下主张：

> 本方法不仅提高视频级安全分类，还能在长视频中以有限帧预算主动定位稀疏有害片段，并据此作出风险敏感的 ALLOW、REVIEW 或 BLOCK 决策。

## 10. 前置论文消融流水线

THVL 外测前先完成冻结 Test 上的内部消融。管理脚本为：

```bash
bash video_detection_agent/manage_paper_ablations_v1.sh status
```

该流水线等待无泄漏校准器完成后自动执行并支持断点续跑，包含：

- 最佳 GRPO 的固定 4 帧、固定 12 帧无工具基线；
- Base、SFT、SFT+GRPO 完整 Agent 对比；
- 去除 `TEMPORAL_EXPAND`；
- 去除初始 Harmless/L1 安全扫描约束；
- 去除 REVIEW 并将 L1 强制转为 BLOCK；
- SigLIP `top_k=1/3/5` 取帧预算对比；
- 完整 Agent 与无泄漏轻量校准器对比。

统一结果将写入：

- `video_detection_agent/outputs/paper_ablations_v1/summary.json`
- `video_detection_agent/outputs/paper_ablations_v1/report.md`
