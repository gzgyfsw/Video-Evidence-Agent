# Qwen3.8 执行 video-evidence-audit

入口：`video_detection_agent/run_qwen_skill_audit.py`。程序直接复用 skill 的
`prepare → export-qwen → accept → summarize`；只增加 Qwen HTTP 调用和执行留档。
不是另写一套审核政策，也不修改当前 Codex 审核或原 Teacher 轨迹。

## 准备与执行

输入仍为 skill 的 `video-evidence-input-v1` JSON/JSONL 合同。
使用与 Codex 审核相同的显式适配结果、媒体和政策；原仓库未知格式必须先显式适配，
不能直接把完整 Teacher transcript 当单次模型输入。程序不自行选择数据或扩大批次。

从项目根目录执行，下列路径和调用上限是占位示例，不对应任何已启动的实验。

```bash
python video_detection_agent/run_qwen_skill_audit.py prepare \
  --input /path/to/normalized.jsonl --media-root /path/to/media \
  --policy /path/to/provided-policy.md --policy-version supplied-v1 \
  --claim-plan /path/to/completed-claim-plan.json \
  --claim-batch-size 6 --claim-images cited \
  --out /path/to/new-qwen38-audit

python video_detection_agent/run_qwen_skill_audit.py inspect \
  --out /path/to/new-qwen38-audit

bash video_detection_agent/scripts/serve_qwen38_skill_audit.sh
```

另一个终端显式执行审核：

```bash
python video_detection_agent/run_qwen_skill_audit.py run \
  --out /path/to/new-qwen38-audit --max-calls 20 --workers 4
```

`prepare`、`inspect` 不请求模型，不加载模型、不下载模型。服务脚本使用本地
`Qwen3.8-27B`、exvllm 环境、bfloat16 权重，默认端口 8069，模型别名
`evidence-audit-qwen38-27b`。默认最多 8 路服务请求，runner 默认并发 4 条视频；
可用 `QWEN_AUDIT_MAX_SEQS` 调整服务并发。服务显式启用前缀缓存；仍保留 eager 执行，
不把未经实测的 CUDA Graph/量化优化算进吞吐承诺。
runner 在调用前检查 `/v1/models` 的模型 ID 与 `root` 路径，拒绝静默退回 Qwen3.6。
此检查验证服务声明与本地 checkpoint 元数据，不能证明服务内存中每个权重的真实性。

`--max-calls` 是整个输出目录累计调用上限，包含过去请求与两次尝试；重新运行时
不会清零。`inspect` 显示任务数及每任务最大图片数，便于在执行前制定预算。
默认每个请求最多 64 张图片、单对象 4096 输出 token、关闭 thinking；主张批次按对象数
分配输出上限，最多 16384 token。可以在 `prepare` 时
显式修改。完整前缀超过图片上限会停止，不丢帧；服务上下文超长或回答被截断记技术
失败，不缩短政策、图片或前缀。服务图片预处理上限 262144 pixels 会影响可辨细节；
更改服务预处理配置应使用新的审核版本，不能假定等同于 Codex 的视觉表现。

## 与 skill 保持一致的约束

- 每次准备冻结完整 skill 源文件、提示、协议、schema 和输入合同，并记录 SHA256。
  `run` 使用输出内的 `skill_snapshot`，后续修改已安装 skill 不影响旧运行。
- 查询与选点按调用前前缀审核；收益按返回后前缀审核；blind 隐藏 Teacher 文本；
  claim/stop 必须在相同前缀 blind 已完成后调用。每次 HTTP payload 只有一条新的
  user message，没有其他视频/时间步骤、过去审核回答、后续图片或共享聊天历史。
  可用 `--workers` 并行不同视频，每个视频内部仍按步骤顺序和盲观察依赖执行；累计
  调用预算在发请求前以互斥锁预留。
- 图片以 data URL 实际发送，图片 ID、顺序、原始字节和哈希留档。请求不附工具。
  单对象提示使用 skill 自带导出器；主张批次与最新版 codex_runner 一样共享
  原版 common/claim 提示并列出每项元数据。不增加分类政策或改变“支持”的含义。
- v2 将原版 schema 的字段形状投影到 API `json_schema` 输出约束，固定该任务 ID、
  输入 hash、实际型号和 backend，并约束 `uncertain` 为布尔值。传输层不支持的
  条件/唯一性规则仍由原版 `accept` 完整验证，不修改模型判断或补填回答字段。
- v3 从同一个历史前缀推导可引用工具 ID 并加入输出枚举；补充说明原版验收器已有的
  全附件观察、盲观察逐图记录和主张引用规则。Teacher 政策及原始语义提示保持原文，
  不强制支持或覆盖不支持/待定结论。SIGINT/SIGTERM 会停止调度并等待已发请求保存后退出。
- v5 对齐最新 skill：同一步、同种 claim、完全相同前缀才共享一次图片请求，
  默认每批最多 6 个对象；blind、query、effect、stop 单独调用。每个对象仍保留原始
  input hash、独立判断、引用、原版 accept 和监督标记。局部 claim 默认只发送全部
  原引用图片；原始前缀元数据与对象哈希保留，实际附件另存 manifest，不用邻帧替代。
  政策字段可使用原引用帧的证据子集；实际查看仍须覆盖全部原引用帧。新版接口提示不再
  错误要求所有 claim 重复查看全部历史图片。
- claim 默认 `--claim-observations compact`：其可见事实与局限留在逐项 reason，
  evidence_frame_ids 保留原合同；claim 的可选 observations 数组为空。
  blind 的完整逐图 observations 仍独立生成、验证和保存。可显式使用 detailed
  做格式对照，不能把重复描述的 token 消耗当成新增审核证据。
  API schema 对每项绑定 task_id/input hash；SUPPORT 的引用覆盖规则使用 anyOf
  转成服务支持的约束，原 skill.validate_review 仍是最终验收器。
- v5 的调用账本在启动时建立索引；每次 dispatch/accept 更新索引，避免上量时反复
  全目录扫描。缓存验证仍检查真实请求/响应/图片哈希。progress 默认同时报告
  completed_videos/total_videos，不能把完成对象数当可训练视频数。
- 模型 JSON 原文交给 skill 原版验收：不修复字段、不补引用、不改判断。
  `NOT_SUPPORTED` / `UNDETERMINED` 的完成回答不重试；首次技术失败立即停止新调度，
  已在途请求保存后退出。检查并修复错误后显式重跑同一个 run，只有失败对象可以
  进行第二次尝试；已完成语义结果不重跑。
  盲观察技术失败时后续 claim/stop 暂缓，不伪造观察记录。
- 原版汇总处理逐主张依赖、技术故障截断、完整/局部/暂缓候选及监督标记。
  所有结果保持 `weak_reference_only=true`、`independent_gold=false`，不启动训练。
- Qwen3.8 是本次用户指定的型号；不声称它已经被证明是最低价、最快或最准的模型。

输出包含原版 `manifest.json`、`cases/*`、`summary.json`，另有：

- `qwen_runner.json`：模型、endpoint、checkpoint 元数据、skill/runner 哈希及参数。
- `runner_snapshot.py`：本次实际 runner 的源码，后续 runner 迭代可用该冻结副本恢复旧运行。
- `qwen_calls/CASE/TASK/01|02/`：真实 `request.json`、未修改的
  `model-response.raw`、抽出的 `response.json`、调用事件/usage/耗时。
- 原版 `tasks/*/attempts/`：通过/失败验收记录、请求、接口返回及图片清单。
- `qwen_execution.json`、`qwen_progress.json`：累计调用数、预算、技术失败及进度。
- `executions/0001|0002/.../`：每次执行的不可覆盖报告与服务模型清单，保留恢复前的失败原因。

主张批次的目录是 `qwen_calls/CASE/batch-HASH/NN/`。event 的 `task_ids` 和
`input_sha256_by_task` 绑定全部原对象，一次真实 HTTP 调用只计一次预算。
`TASK.response.json` 是从原封 `model-response.raw` 解包的逐项 JSON；
`review_raw_sha256_by_task` 保存逐项哈希，训练筛选器 v7 支持这一溯源。
批次中遗漏、重复、哈希串换或不合法引用都产生技术失败；已合法完成的其他对象保留，
恢复只调用失败对象，不因其他项失败而重审整个批次。

## 上量的文本准备与性能检查

`edit_qwen_skill_claims.py` 直接使用冻结 skill 的 claim-editor 提示、literal_spans、
build_plan 和 claim_units 校验，由同一个 Qwen3.8 服务做文本拆分。它没有图片输入，
不判断视觉真值；同时额外校验每个子项完整保留原有依赖，防止清空依赖升级为独立事实。
原文不能翻译、改写或漏词；不合法输出首次停止，显式恢复最多两次。

```bash
python video_detection_agent/edit_qwen_skill_claims.py \
  --input /path/to/normalized.jsonl --out /path/to/new-editor \
  --max-calls 120 --timeout 600
# 然后在 prepare 使用 --claim-plan /path/to/new-editor/claim_plan.json

python video_detection_agent/profile_qwen_skill_audit.py \
  --run /path/to/old-frozen-audit \
  --output /path/to/new-profile.json
```

编辑器调用与视觉审核分别留档、分别设置硬上限，估算成本时应把两者相加。
性能检查不调用模型；它报告旧调用的真实耗时和新版排程请求数，
减少请求数不能直接换算为吞吐倍数。使用 `--compare-run` 可附同视频历史/新版的
实测用时，但不同并发、文本拆分和预热条件仍要注明。

2026-10-13 校准产物：`outputs/qwen_skill_audit_v5_speedcheck_20261013/`。
原 10 视频冻结对象的离线排程从 187 请求/1266 图片张次降为 63/414。
一条已审核视频的新版文本准备拆成 23 个对象，实际 6 次 HTTP 调用、全部验收闭合，
耗时 770.9 秒；历史同视频是 14 个对象/14 次调用、915.5 秒。
对象数和并发条件不同，不能把这一比值当全量吞吐保证。
同组 6 个主张重复逐图描述与紧凑输出的试验分别为 288.5/163.2 秒。
早期试验的引用遗漏和稀疏 schema 兼容失败均留档；改为完整 schema 分支后，
两个对象的真实服务校验耗时 63.7 秒，通过原版逐项验证。
试验中不同模型回答仍是弱参考；不因支持更多而宣称准确率提高。

## 恢复、锁与测试

重复 `run` 复用同模型且通过全部哈希检查的完成结果。原版机械 manifest 的
`model_calls=0` 表示原版 helper 没有发送请求；真实 Qwen 请求数以
`qwen_execution.json` 和 `qwen_calls/*/*/*/event.json` 为准。
模型完成但进程在验收前中断时，可从已保存的原始回答恢复，不再发请求。
全部任务已完成时，恢复检查不需要在线模型服务。
若在网络调用过程中中断且没有落盘回答，执行状态不确定，runner 停止并保留现场，
不会自动重新调用。不要删除事件来重新获得预算或重试机会。

同输出目录使用 `.qwen-runner.lock`，原版 helper 使用 `.writer.lock`。
进程崩溃留下锁时，先确认文件内 PID 已退出再手动移除相应锁；不能抢活跃任务的锁。
更换 checkpoint、模型、协议、政策、runner 或生成参数必须准备新的独立输出目录。

```bash
python /home/user/.codex/skills/video-evidence-audit/scripts/test_audit.py
python -m unittest discover -s video_detection_agent/tests -p test_qwen_skill_audit.py -v
```

测试使用合成图片与 MOCK HTTP 服务，验证协议约束、请求隔离、恢复、预算、重试及
原版验收链路。测试不证明 Qwen3.8 的语义准确率，也不证明它与 Codex 判断一致。
