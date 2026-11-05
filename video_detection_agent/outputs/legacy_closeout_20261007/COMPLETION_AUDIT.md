# 旧方案消融完成审计

审计时间：2026-10-09

## 已完成并验证的内部消融

以下实验均覆盖同一冻结内部评测集 `1679/1679`，并已生成结果与指标：

| 实验 | 结果 | VERIFIED | Metrics |
|---|---:|---:|---:|
| `full-base` | 1679/1679 | 是 | 是 |
| `full-sft` | 1679/1679 | 是 | 是 |
| `full-grpo` | 1679/1679 | 是 | 是 |
| `uniform4-grpo` | 1679/1679 | 是 | 是 |
| `uniform12-grpo` | 1679/1679 | 是 | 是 |
| `no-temporal-grpo` | 1679/1679 | 是 | 是 |
| `no-initial-guard-grpo` | 1679/1679 | 是 | 是 |
| `budget-top1-grpo` | 1679/1679 | 是 | 是 |
| `budget-top5-grpo` | 1679/1679 | 是 | 是 |
| `no-review-force-block` | 1679/1679 | 派生后处理，无独立 VERIFIED | 是 |

总汇总文件：

- `report.md`
- `strict_report.md`
- `summary.json`
- `strict_summary.json`

完成标记：

- `INTERNAL_REPLAY_COMPLETE`

## 当前结论

旧方案内部 replay 消融已经完成。`manage_legacy_ablations_only_20261008.sh status`
返回 `NOT_RUNNING`，并显示所有内部消融均为 `1679/1679`。

## 仍未完成但不属于本轮内部 replay 的事项

这些事项在 `PLAN.md` / `strict_report.md` 中仍被列为论文后续工作，不能用本轮内部消融结果代替：

1. THVL 等独立外部评测；
2. 随机种子复验；
3. 完整历史耗时与两种 REVIEW 原因审计。

补充：GRPO-only（不经过 SFT）消融已于 2026-10-09 完成，结果位于
`video_detection_agent/outputs/grpo_only_ablation_20261009/`，覆盖同一冻结 Dev 1679/1679，
并已生成 `GRPO_ONLY_COMPLETION_AUDIT.md` 和 action-aware 指标。

因此：当前已列入本项目的内部消融均已完成；随机种子复验、THVL 等独立外测和更细的历史审计仍属于后续论文工作，不影响本轮消融闭环。
