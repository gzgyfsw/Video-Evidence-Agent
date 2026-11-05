# 默认分类政策 v2：实施记录

用户已明确授权先修改分类规则。本轮完成类别规则的文档化、按版本读取和兼容检查，未迁移数据或训练。

**后续决定：继续使用上一个冻结的提示词和边界进行比较。v2保留为未采用的备选，不用于当前实验。**
当前默认为legacy-safety-policy-v1，其政策内容与冻结THVL代码逐字一致。运行旧比较时不要设置VIDEO_SAFETY_POLICY_VERSION=default-category-policy-v2。

用户随后要求将当前旧版本也外置以便查看：原文已存于`policies/legacy_safety_policy_v1.md`，代码从该文档读取，
并以冻结哈希校验防止静默更改。中文阅读说明位于`policies/README.md`。这次仅改变存放位置，未采用v2边界。

## 已实现

- 分类主体存于`policies/default_category_policy_v2.md`，四个内部键保持兼容。
- `agent/moderation_policy.py`提供版本选择与中文展示名；`agent/qwen_backend.py`保留LEGACY_SAFETY_POLICY。
- 环境变量不设置时使用旧规则，避免旧实验续跑悄悄换政策。
- 显式启用v2时从政策文档读取分类主体，兼容入口仍附原等级合同，明确它不是未知等级分离的新协议。
- 自主Teacher运行配置增加policy_version，与已有safety_policy_sha256一起记录。
- 无效版本直接报错，不悄悄退回其他标准。

启用方式（只设置当前进程，未启动实验）：

```bash
VIDEO_SAFETY_POLICY_VERSION=default-category-policy-v2 PYTHONPATH=video_detection_agent .venv/bin/python -c 'from agent.qwen_backend import POLICY_VERSION, SAFETY_POLICY; print(POLICY_VERSION); print(SAFETY_POLICY)'
```

新实验须明确指定该版本并记录hash。更新标准时另建版本；不得在冻结实验中原地修改文件后继续把它称为同一政策。

## 规则对账

| 项目 | 原有条款／入口 | 本轮处理 |
|---|---|---|
| 赤膊、普通服装／身材 | SAFETY_POLICY已有衣服不能单独定性的约束 | 增补赤膊表演反例，要求姿态／构图／语境支持 |
| 毒品与医学注射 | 暴力危险有宽泛dangerous or illegal acts | 明确危险滥用／过量归属；物体不能独立证明毒品 |
| 蹦极 | 通用危险动作范围，未专门规定此活动 | 活动事实不能自动推成自伤／违规，需保护措施与危险行为依据 |
| 身份仇恨 | 原敏感类已覆盖身份歧视／去人化 | 展示名和中文说明明确范围，并区分普通辱骂与仇恨 |
| 高度知名政治人物 | 原政策中明确可识别且持续居中即可，通常中性L1 | 保留原项目范围，不把人物存在改写成仇恨／极端宣传 |
| 旗帜专项规则 | build_sft_coldstart_dataset、evaluate_remote_joint_teacher_agent、evaluate_multistage_video_agent含香港／西藏／台湾旗帜L2专项提示；自主thin入口无同一固定专项句 | 本轮不改专项句，文档要求显式标明来源和版本；入口间完整统一尚未做，不能宣称所有入口政策一致 |
| 体育／虚构 | 原有强力体育对抗纳入、明显非血腥游戏例外 | 保留明确条款；力度与例外不明时不猜 |
| 未覆盖风险 | 四类不能覆盖所有错误信息／辱骂／犯罪／成瘾 | 文档明确范围外，不能强塞分类；范围外字段的完整运行支持尚未实现 |
| L1与未知 | 原合同仍把mild/borderline/unresolved合并L1 | 本轮不改等级合同；split-l1-risk-and-uncertainty仍待实施，不把旧L1当确认轻微 |

## 无模型验收案例

以下为规则审阅案例，不是模型已答对的实验结果。

| 构造事实 | 规则预期 |
|---|---|
| 医学注射，未见滥用依据 | 不因针头确认吸毒 |
| 不明粉末或吸烟外观 | 不猜物质；必要时保留疑点补上下文 |
| 有上下文支持的毒品危险使用／过量 | 暴力与危险，等级另按阈值 |
| 单纯赤膊体育表演 | 不自动确认色情 |
| 蹦极且是否存在保护措施不明 | 不自动确认自伤；缺画面补证，缺阈值标政策边界 |
| 普通脏话 | 不自动归身份仇恨 |
| 有语境支持的身份去人化表达 | 敏感类候选，保留原文字及语境 |
| 不认识的西装人物 | 不猜高度知名政治人物身份 |
| 仅初始4帧未见风险 | 不构成完整排除结果 |
| 无直接危险且未覆盖的犯罪 | 范围外待处理，不伪造暴力证据 |

## 验证结果与限制

命令：

```bash
.venv/bin/python -m pytest -q video_detection_agent/tests/test_moderation_policy_versions.py video_detection_agent/tests/test_thin_teacher_v3_contract.py video_detection_agent/tests/test_agent.py video_detection_agent/tests/test_multistage_agent.py
```

实际结果：64 passed，9.59秒。验证旧SAFETY_POLICY与冻结THVL代码的字符串逐字一致、默认兼容、v2显式加载、
未知版本报错、现有Agent与thin合同回归。测试不证明真实模型已学会新分类规则。

原政策内容SHA256：`45fb654deceb9edea3681dc87f2d352e164495b6a28165d84a27afd332bd6040`。
本轮新政策与兼容等级拼接后的SHA256为`c627acacb0fa1964314d5e4ab0a71cd11401faaed31aeeb2201d73e2671ae65e`；
以后以启动时的SAFETY_POLICY_SHA256／运行safety_policy_sha256记录为准。

当前环境没有可调用的OpenSpec CLI；本轮读取现有规划并记录实施状态，未声称执行OpenSpec命令验证。
分类模块已可用，但完整未知等级／范围外结构、旧L1迁移、逐Train影响清单和真实模型验证仍未执行。
现有THVL审核进程、它的提示与原始结果未切换；未启动训练或新增批量模型调用。
