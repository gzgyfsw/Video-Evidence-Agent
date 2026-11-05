# Agent Teacher / Student 数据接收规则

状态：2026-08-18 固定。适用于固定 **4 张均匀非黑初始帧** 的 Agent 数据。

## 1. 视频准入

- 原始视频时长必须严格大于 10 秒；
- 不超过 10 秒的视频不直接训练。只有同一目标类别的短视频合成为大于 10 秒的新视频后才可使用；
- 合成样本必须保存所有父视频、拼接边界和共同的 `source_group`，划分时不得跨 split。

## 2. 记号

- `Y`：来源数据集原始类别；
- `T`：Qwen3.7-Plus Teacher 的最终类别、Severity 与工具轨迹；
- `C`：Codex 对视频帧的独立盲审类别；盲审时不可看到 `Y` 或 `T`；
- `S`：Qwen3.5-4B Student 在相同初始 4 帧下的类别与 Severity。

类别空间固定为 `Harmless`、`Sexual Content`、`Violence & Dangerous`、
`Sensitive Figures & Ideologies`。

## 3. Teacher 轨迹准入

Teacher 轨迹只在有完整、可解析的四类最终输出时进入判定。

| 条件 | 处理 | 接收来源标记 |
|---|---|---|
| `T.category == Y.category` | 直接纳入 | `source_teacher_match` |
| `T.category != Y.category` 且 `C.category == T.category` | 纳入，采用 Teacher 类别/Severity/轨迹 | `codex_teacher_recovered` |
| `T.category != Y.category` 且 `C.category != T.category` | 隔离，不训练 | `teacher_codex_disagreement` |
| Teacher 或 Codex 失败/无结论 | 隔离，不训练 | `audit_or_runtime_failure` |

因此，来源标签是直接一致时的第一道校验；发生冲突时，只有 Teacher 与
Codex 的**类别一致**才能回收。Codex 的 Severity 只记录审计信息，不作为
接收门槛。接收后的 `severity`、`tool_calls`、证据和轨迹始终取自 Teacher。

## 4. Student 对照（严格口径）

Student 的参照是已接收的 Teacher `T`，不是来源标签 `Y`。

```text
student_teacher_category_match = (S.category == T.category)
student_teacher_severity_match = (S.severity == T.severity)
student_teacher_strict_match = category_match and severity_match
```

只有 `student_teacher_strict_match=true` 才能称为 Student 与 Teacher “一致”。
类别相同但等级不同必须单独统计，不能算严格一致。

Student 与 Teacher 不一致**不**会使 Teacher 轨迹失效：它用于识别 Adaptive /
Active 样本，以及后续 SFT / GRPO 的学习空间；Teacher 轨迹的可用性只由第 3 节决定。

## 5. 每条输出必须保存的字段

```json
{
  "source_category": "...",
  "teacher_category": "...",
  "teacher_severity": "...",
  "teacher_acceptance": "source_teacher_match | codex_teacher_recovered | teacher_codex_disagreement | audit_or_runtime_failure",
  "codex_category": "... or null",
  "student_category": "... or null",
  "student_severity": "... or null",
  "student_teacher_category_match": true,
  "student_teacher_severity_match": true,
  "student_teacher_strict_match": true
}
```

后续所有“可训练数”“Teacher 正确数”和 Direct / Adaptive / Active 统计均须先使用
本规则过滤 Teacher，再按 Student–Teacher 严格一致口径计算。
