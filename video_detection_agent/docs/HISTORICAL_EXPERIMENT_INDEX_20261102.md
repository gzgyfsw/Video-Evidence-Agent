# 历史数据、训练、框架对照与消融索引

整理日期：2026-11-02。仅核对已保存记录并新增此索引；没有重跑实验、修改历史预测或改变划分。本文不是将所有旧实验合并成一套新结果。

## 建议先读的三份文档

1. [项目数据与实验总日志](DATA_AND_EXPERIMENT_LOG.md)：数据来源、标签可信池、D/A/A、Train/Dev/Test、SFT迭代、GRPO接受与失败过程。它是追加的历史时间线，各节有各自的模型、分母及协议，不能把最早章节当最新现状。
2. [历史实验链路与消融总账](../outputs/thvl_current_protocol_20261010_v4/REPORT.md)：从固定4帧Base、完整Agent、SFT、GRPO到厚薄Harness的连续梳理；含内部消融、GRPO-only、Action-aware、路由和LoRA小样本诊断。部分末尾状态更新较早，完成性需交叉核对下面的专门完成报告。
3. [旧方案统一复现严格指标总表](../outputs/legacy_closeout_20261007/strict_report.md)：旧内部N=1679的主对照。分类与严格类别+等级指标以这份及strict_summary.json为准，不与paper_ablations_v1较早重复运行取平均。

## 已完成的实验与原始证据

| 内容 | 工作记录/报告 | 数据、配置或逐视频结果 | 完成性与解释范围 |
|---|---|---|---|
| 旧Base/SFT/GRPO与推理消融 | [严格总表](../outputs/legacy_closeout_20261007/strict_report.md)、[完成审计](../outputs/legacy_closeout_20261007/COMPLETION_AUDIT.md) | [精确统计](../outputs/legacy_closeout_20261007/strict_summary.json)、[模型和代码冻结](../outputs/legacy_closeout_20261007/FREEZE.json)、各arm的results.jsonl/metrics目录 | 每arm覆盖1679；历史已暴露内部评测，不能称全新密封Test |
| 固定4/12帧、去Temporal、去初始guard、检索Top-1/Top-5 | 同上；[原始较早论文进度](../outputs/paper_ablations_v1/PROGRESS.md)仅作历史 | legacy_closeout中的uniform4-grpo、uniform12-grpo、no-temporal-grpo、no-initial-guard-grpo、budget-top1-grpo、budget-top5-grpo | 均已完成；早期PROGRESS仍含排队状态，不代表后来未完成 |
| 去REVIEW | [动作指标总表](../outputs/legacy_closeout_20261007/report.md)、no-review-force-block/metrics | 同一预测的后处理派生结果 | 不是独立模型训练或一次新的模型推理；Strict/Review不能混作严格类别+等级 |
| 不做SFT直接GRPO | [GRPO-only完成审计](../outputs/grpo_only_ablation_20261009/GRPO_ONLY_COMPLETION_AUDIT.md) | [四种训练方式对照](../outputs/grpo_only_ablation_20261009/grpo_only_summary.json)、raw_rollouts.jsonl、rollout_manifest.jsonl、update、full_dev1679 | N=1679完整结果；不等于已有SFT+GRPO模型去掉某个LoRA即可得到 |
| 原单轮/自主薄框架/规则厚框架/固定12帧 | [THVL厚薄对照报告](../outputs/thvl_thick_matched_20261011/REPORT.md) | [四组统计和结果哈希](../outputs/thvl_thick_matched_20261011/COMPARISON.json)、各arm逐视频metrics_exact/per_video.jsonl | 448条、同paper-grpo权重；厚薄预算上限相同，提示和策略同时变化，实际帧数不同；已完成 |
| THVL映射回旧四类的派生评分 | [四分类代理评测](../outputs/thvl_legacy_fourway_closeout_20261013/REPORT.md) | MAPPING.json、manifest.jsonl、四个arm的summary.json | 重用已完成预测，不是重跑；11类到4类为项目代理映射，不是原生四分类真值；不能评L1/L2 |
| 新旧Teacher轨迹比较 | [原16789条Teacher比较说明](../outputs/teacher_comparison_original16789_20261008/README.md) | comparison.csv、paired_records.jsonl、changed_results.jsonl、teacher_comparison.ipynb、summary.json | 16733个配对成功答案；旧/新提示、机制及部分模型不同，不能只归因于框架；初步案例未独立裁定画面真值 |
| 自主薄Harness Base/SFT全量对照 | [阶段状态日志](../experiments/thin_harness_week_20261001/STATUS.md) | [1678条配对指标](../experiments/thin_harness_week_20261001/eval/results_sharded_v2/base_vs_sft.full_dev.json)、同目录base-dev.jsonl和sft-dev.jsonl | 完整已跑；对应当时的SFT v1，不是20261027现有SFT；包含失败、恢复及准入未通过的记录 |

## 旧内部消融的关键数字

以下摘自legacy_closeout严格总表；“类别一致”及“类别+等级一致”均相对弱参考，不代表独立人工准确率。

| 方法 | N | 类别一致 | 严格类别+等级一致 | 平均帧 | 平均工具 |
|---|---:|---:|---:|---:|---:|
| Base完整Agent | 1679 | 67.24% | 55.57% | 原严格表记None | 1.69 |
| SFT完整Agent | 1679 | 87.91% | 80.76% | 7.93 | 0.61 |
| GRPO完整Agent | 1679 | 90.11% | 81.06% | 8.19 | 0.65 |
| 固定4帧GRPO权重 | 1679 | 86.48% | 77.13% | 4.00 | 0 |
| 固定12帧GRPO权重 | 1679 | 85.47% | 77.55% | 12.00 | 0 |
| 去Temporal | 1679 | 90.05% | 81.30% | 8.17 | 0.65 |
| 去初始guard | 1679 | 88.86% | 79.33% | 6.70 | 0.42 |

其他Top-K及动作处置指标见原总表，不在本索引中换口径重新排名。SFT→最终GRPO之间的权重继承含额外分级SFT和后续训练，不能把两行差异全归因于一次GRPO。

## 训练版本：不要把所有“GRPO”当成同一个模型

| 版本 | 权威记录 | 训练内容/规模 | 特别注意 |
|---|---|---|---|
| GRPO v1路由 | [奖励与信用规则](GRPO_REWARD_V1.md)、[准入合同](GRPO_ACCEPTANCE_V1.md)、总日志§12.19 | 训练初始路由；最终组合含冻结SFT工具后判断 | 文档中的原始奖励默认值不一定等于后来run的环境覆盖值 |
| **冻结GRPO v2多阶段包** | [完成审计](../outputs/best_grpo_qwen35_4b_agent_v2/COMPLETION_AUDIT.md)、[模型包清单](../outputs/best_grpo_qwen35_4b_agent_v2/BEST_GRPO_V2.json) | 8192条真实轨迹、2048组、正式42步；checkpoint-25被选中；216次模型Temporal调用 | **三个适配器组成的Agent包**，不是单LoRA；需按包绑定复现，不能套用后来单LoRA指标 |
| v3及后续定向Temporal尝试 | [v3研究与失败归因](../outputs/grpo_v3/README.md) | 定向时序路由、错误归因、困难样本及恢复记录 | README同时包含计划和实际结果；最终状态看具体iteration的acceptance/audit，不能据计划清单认定全部完成 |
| 分级修复SFT→GRPO | 总日志§12.21、[直接前后对照](../outputs/severity_sft_grpo_v3_route_weighted/final_comparison.json) | GRPO393组/1572轨迹，50步，选step30 | 直接SFT起点是severity-repair checkpoint450；类别88.922%→89.160%，严格联合不变 |
| action-aware后续GRPO | [选择记录](../outputs/action_aware_grpo_v1_review_conflicts/selection.json)、[实际训练配置](../outputs/action_aware_grpo_v1_review_conflicts/grpo/update/run_config.json) | 377组/1508轨迹，48步，从上述step30继续 | 与冻结v2三适配器包不是同一模型；对应legacy_closeout最终full-grpo路径 |

冻结v2完成报告确实记录过分类、严格联合及帧成本的小幅改善，同时有害漏判32→34，未隐去该取舍。这条证据与后来action-aware完整Agent平均帧数增加并不矛盾，因为比较的模型与协议不同。

## 数据、标签与来源清单

| 数据版本 | 记录位置 | 含义 |
|---|---|---|
| 原始冻结划分 | [旧Train/Dev/Test说明](../outputs/agent_split_v1_20260905_action_aware_v3_review_conflicts/README.md)、同目录train.jsonl/dev.jsonl/test.jsonl | Train13431 / Dev1679 / Test1679；集合规模不等于参与某次训练的视频数 |
| 类别验证规则 | build_label_verified_pool.py、outputs/label_verified_pool_v1/summary.json | Teacher与来源类别一致，或不同Teacher类别共识；不是逐视频人工确认，最初类别验证不包含等级 |
| 自主Teacher v3全池 | [最终汇总](../outputs/thin_teacher_v3_all_downloaded_20260928/canonical_summary.json)、canonical_index.jsonl指向原始分片行 | 33993库存，33862最终回答、131未解决；映射旧Train13426条中13388回答、38未解决，不可把映射总数当全部成功 |
| 新Teacher旧数据配对 | [比较说明](../outputs/teacher_comparison_original16789_20261008/README.md) | 按完整视频SHA配对；逐行来源、迁移矩阵、差异及未配对记录齐全 |
| 后来的扩充池 | [扩充划分说明](../outputs/expanded_thin_splits_v2_20261007_verified/README.md)、[严格>10秒候选说明](../outputs/weak_teacher_v3_pool_gt10s_20261008/README.md) | 不同范围/版本，不能与旧13431直接相加，也不等于已训练 |
| 后来的5627训练池 | [7077候选迭代协议](../outputs/agreed7077_sft_iteration_20261017_v2/PROTOCOL.md) | Train5627 / Dev731 / Test695，含跨划分隔离；保留为历史，当前用户要求优先旧池 |
| 当前SFT实际训练与权重 | [选择记录](../outputs/sft_validation_scale_20261027_v1/SELECTION.json)、[实际run配置](../outputs/sft_validation_scale_20261027_v1/train/run_config.json)、GRPO_STARTPOINT.json | 1743视频、2829状态、400步；实际adapter=null，从基座新建LoRA，不能根据较早计划宣称继承旧SFT |

## 日志里记录了哪些当时的工作

- [总日志](DATA_AND_EXPERIMENT_LOG.md)§2–7：来源标签、可信池、Base覆盖、D/A/A定义。
- 同文件§10–11：冻结划分、Base与SFT前的对照。
- 同文件§12.5–12.7：路由warm-up、轨迹SFT、router refresh、post-tool修复。
- 同文件§12.8–12.19：GRPO早期失败、奖励血缘、信用分配、配对反事实评估与冻结过程。
- 同文件§12.20–12.21：多阶段GRPO、分级修复和真实工具链采样。
- [2026-10-12工作日报](daily/2026-10-12.md)：框架转向、厚薄结果、取证/停止目标、数据口径和当天实际工程工作。
- [2026-10-13工作日报](daily/2026-10-13.md)：政策与事实验收边界、THVL旧四类代理映射收尾。
- [薄Harness阶段日志](../experiments/thin_harness_week_20261001/STATUS.md)：真实图片输入、loss mask、恢复对账、训练和拒绝原因；按日期阅读，不把旧“当前状态”标题当今天状态。

## 论文整理时的读取规则

1. 模型名不够，必须同时固定适配器路径/哈希、框架代码、政策、数据ID及评分口径。
2. 同一内部N=1679的重复结果优先采用统一冻结复现；原始paper_ablations进度保留但不平均、不择优拼接。
3. strict类别+等级、Strict/Review、Safe action是不同指标；“Overblock”不同报告分母/定义也不同。
4. THVL区间命中不等于分类分级正确；四类映射是派生代理评测。THVL后来已用于开发分析，不称新的密封外测，当前不得拿来筛训练样本或调奖励。
5. COMPLETE标记和逐视频结果优先于较早STATUS中的“排队/尚未完成”。模型包的实际冻结记录优先于综述中“最佳版本”的简写。
6. 旧历史预算只用于复现旧协议；当前新Agent按用户指令不设累计帧/工具/轮数成本上限，实际消耗照常记录。
7. 本索引未重新独立验算所有模型张量和全部实验结果；引用“完成”是根据专门报告及存在的结果/冻结产物核对，不把它说成新做了一轮全量复现实验。

本次相关的新试验另存：legacy_train_grpo_pilot_20261102_v1、legacy_outcome_grpo_update_20261102_v1、legacy_train_grpo_validation_20261102_v1；它们不是上述历史消融的一部分。
