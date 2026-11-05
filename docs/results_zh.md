# 当前结果表

本页区分“当前第四阶段的运行状态”和“已经完成审计的历史定量结果”。不同数据、参考标签和评测协议下的数字不能直接横向排名。

## 当前第四阶段状态

| 项目 | 规模或状态 | 结论边界 |
|---|---:|---|
| Unified Student SFT | 1,743 个视频、2,829 个监督状态、400 个优化步 | 已完成训练；从基座新建 LoRA |
| Outcome-Cost GRPO-53 | 938 组、1,688 条入选轨迹、53 个优化步 | `BOUNDED_GRPO_COMPLETE_NOT_PROMOTED`，仍是候选模型 |
| 配对 Dev | 1,915 个视频 | SFT/GRPO 使用相同来源、种子、工具、预算与处理协议；最终汇总指标尚未闭合 |
| 冻结 Test | 1,924 个视频 | 当前队列记录为等待配对 Dev 完成；不能提前报告 Test 性能 |

因此，目前可以报告训练已完成和冻结评测设计已经建立，但不能把 GRPO-53 写成正式优于 SFT，也不能给出尚未生成的 Dev/Test 最终数字。

## 已闭合：历史内部完整 Agent 对照

以下来自旧 `legacy_closeout` 的 1,679 条内部评测。类别和等级均相对弱参考，不代表逐视频独立人工真值；SFT 到最终 GRPO 之间还包含额外分级 SFT 与后续训练，差异不能全部归因于一次 GRPO 更新。

| 方法 | N | 类别一致 | 类别+等级一致 | 平均帧 | 平均工具调用 |
|---|---:|---:|---:|---:|---:|
| Base 完整 Agent | 1,679 | 67.24% | 55.57% | 未记录 | 1.69 |
| SFT 完整 Agent | 1,679 | 87.91% | 80.76% | 7.93 | 0.61 |
| GRPO 完整 Agent | 1,679 | **90.11%** | **81.06%** | 8.19 | 0.65 |
| 固定 4 帧（GRPO 权重） | 1,679 | 86.48% | 77.13% | 4.00 | 0 |
| 固定 12 帧（GRPO 权重） | 1,679 | 85.47% | 77.55% | 12.00 | 0 |
| 去 Temporal | 1,679 | 90.05% | **81.30%** | 8.17 | 0.65 |
| 去初始 Guard | 1,679 | 88.86% | 79.33% | 6.70 | 0.42 |

## 已闭合：分级修复 SFT → GRPO

该实验在同一冻结 Dev-1679 上对比，GRPO 使用 393 个有效提示组、1,572 条 rollout 和 50 个优化步，最终选择 step 30。

| 阶段 | 类别准确率 | 类别+等级准确率 | 有害漏判 | Harmless 误报 |
|---|---:|---:|---:|---:|
| SFT (`sft-severity-s450`) | 88.922% | 81.120% | 75 | 87 |
| SFT + GRPO (`grpo-severity-s30`) | **89.160%** | 81.120% | **73** | 87 |

该轮冻结结论为接受：严格准确率没有下降，有害漏判减少 2 条。它属于第三阶段的历史结果，不是当前第四阶段 GRPO-53 的最终结果。

## 已闭合：THVL 外部取证评测

THVL-Bench 含 448 个视频与 1,097 个人工有害时间区间，不参与训练或检查点选择。该实验评估的是取证覆盖，不是当前自定义 L1/L2 分类真值。

| 指标 | 厚 Harness Agent |
|---|---:|
| 结构校验通过 | 448 / 448 |
| 工具调用 | 378 / 448 |
| 平均有效帧 | 9.94 |
| Exact observed recall | 55.85% |
| ±3 秒 observed recall | 64.26% |
| Exact evidence recall | 27.66% |
| ±3 秒 evidence recall | 30.21% |
| ALLOW / REVIEW / BLOCK | 97 / 191 / 160 |

同预算下，厚 Harness Agent 命中 525 个视觉/文本区间，均匀采样对照命中 580 个；这说明主动取证框架已经可运行，但当时的取证召回仍未超过均匀基线。

## 权威来源

- [`HISTORICAL_EXPERIMENT_INDEX_20261102.md`](../video_detection_agent/docs/HISTORICAL_EXPERIMENT_INDEX_20261102.md)
- [`DATA_AND_EXPERIMENT_LOG.md`](../video_detection_agent/docs/DATA_AND_EXPERIMENT_LOG.md)
- 当前阶段训练与评测身份见 [`stages/04_policy_configurable_agent/README.md`](../stages/04_policy_configurable_agent/README.md)
