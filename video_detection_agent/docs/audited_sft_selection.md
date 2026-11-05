# 审核后训练筛选

`select_audited_sft.py` 读取已经闭合的 Qwen skill 审核，在独立目录输出当前
`train_sft_coldstart.py` 能读取的完整轨迹或最终回答 SFT 样本。原始轨迹、审核回答与
`candidate.json` 均只读；已有运行不会因规则变化而被改写。

```bash
python video_detection_agent/select_audited_sft.py \
  --run /path/to/completed-qwen-audit \
  --out /path/to/new-training-selection
```

准备下一批时可用 `prepare_qwen_skill_audit_batch.py --count N --out ... --select-only`
从当前 500 条池里做可复现的分层选取；它默认排除此前 Codex 33 条和本次 Qwen 10 条，
也可重复传 `--exclude-normalized` 排除之后的批次。默认仅选正向风险视频，
避免花审核调用在当前无法训练整段 Harmless 的案例上；校准审核时可加
`--include-harmless`。按输出的 Teacher 原文编写 `--atoms`
后再运行适配。当前 skill v1.4+ 还要求逐主张的显式原子化计划；准备模板并完成该计划后，
用 Qwen runner 的 `--claim-plan` 建立新审核。选取与适配都不请求模型。

筛选只接收 Train、来源视频与冻结图片哈希一致、全部关键审核项和字段均支持、
无技术或机械未知的完整候选。`Harmless/ALLOW` 由稀疏抽帧得出的整段无害结论
一律进入 `review_queue.jsonl`；只有可见正向风险证据的完整候选可导出。
即使旧版 skill 把这样的 Harmless 轨迹列为完整，训练筛选仍会拦截。
每条 evidence 原文的内容词须被独立审核的原子主张覆盖。导出的 target 只含
已审核的 action、category、severity、moderation_action、evidence、rationale，
不复制未经审核的附加字段或负面覆盖声明。原始判断仍是弱参考，不是人工金标。

输出：

- `sft_final_answer.jsonl`：当前单状态 SFT 训练器可读取的完整样本。每行有
  `prompt`、`target`、按审核时顺序排列的 `audit_images` 路径与 SHA256。
- `sft_full_trajectory.jsonl`：当前多轮 SFT 训练器可读取的因果轨迹；按历史
  顺序放入初始图片、原始工具决定、实际工具返回与新增图片、最终回答。工具分数
  仅作为返回文字，图片取审核时的冻结字节。不把两份文件合并训练，以免同一视频重复。
- `review_queue.jsonl`：每条未入选视频及全部拒绝原因。
- `local_fact_spans.jsonl`：已支持且仍可监督的局部事实片段与原文位置。
  当前训练器没有字段级 token loss mask，因此此文件不直接用于训练。
- `summary.json`：数量、来源和产物哈希。

训练器读取 `audit_images` 时验证图片字节哈希并从冻结图片解码，避免按视频时间戳
重新抽帧造成与审核画面不一致。若 `--max-pixels` 低于审核图片像素数，训练器会
拒绝静默缩小。使用前先读取审核产物的实际冻结图片尺寸，并将 `--max-pixels` 设为不小于
最大冻结图片像素数；本次 batch2 为 640×480，应使用 `--max-pixels 307200`。
程序不会启动训练；实际训练仍需沿用项目的模型、
优化器与资源配置。针对本次十视频 batch2，选择结果是两条正向风险最终回答样本、
两条完整轨迹样本，另八条进入复核队列。
这一选择只证明符合已实现的审核和格式约束；模型与 Teacher 同源，语义准确率仍需
独立抽样复核。导出的提示和目标属于审核后 SFT 合同；虽然当前训练器已能读取，
并入现有主 Agent 数据集之前仍须核对主 Agent 的输出字段及评测合同。
