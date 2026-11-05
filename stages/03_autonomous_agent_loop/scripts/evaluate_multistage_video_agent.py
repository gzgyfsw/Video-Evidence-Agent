#!/usr/bin/env python3
"""Run a causal video Agent whose model owns both retrieval decisions.

GRPO v1 optimized only the initial ANSWER/JOINT_SEMANTIC_SCAN route while
Python scheduled temporal expansion.  This runner keeps the same safety guard
and real SigLIP/frame tools, but asks the policy to choose ANSWER or
TEMPORAL_EXPAND after seeing semantic candidates.  Framework fallbacks remain
auditable and never count as model-initiated actions.
"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import json
import re
from collections import OrderedDict
from pathlib import Path
import threading

from agent.qwen_backend import CATEGORIES, canonical_category
from agent.risk_queries import risk_queries
from agent.tools import JointRiskSemanticLocator, TemporalEvidenceExpander
from agent.video import (
    extract_frames,
    recover_frame_near_timestamp,
    uniform_frames,
    video_duration,
)
from build_sft_coldstart_dataset import RISKS, SAFETY_POLICY, initial_prompt, normalize_queries
from evaluate_causal_video_agent import (
    CausalBrain,
    ROOT,
    evenly_take,
    json_text,
    make_record,
    read_jsonl,
    valid_final,
    valid_initial_action,
    valid_initial_direct_harm,
)
from evaluate_remote_joint_teacher_agent import (
    evidence,
    merge_hits,
    normalize_assessments,
    normalize_dynamic_queries,
)


POLICY_MAX_OUTPUT_TOKENS = 1024


def decode_semantic_candidates(video_path: str, observations: list[dict]):
    """Decode every retriever hit that is available without failing the row.

    Damaged or unusual containers occasionally fail one random seek even when
    the other candidate timestamps decode correctly. Retry only the missing
    timestamps with the independent near-frame recovery path. If a timestamp
    still cannot be decoded, drop that observation and continue with the real
    frames that were obtained; never tell the VLM it saw an unavailable frame.
    """
    requested_times = [float(item["timestamp"]) for item in observations]
    decoded = extract_frames(video_path, requested_times, deduplicate=False)
    unused = set(range(len(decoded)))
    aligned_observations: list[dict] = []
    aligned_frames = []

    for observation, requested in zip(observations, requested_times):
        match = next(
            (
                index for index in unused
                if abs(float(decoded[index].timestamp) - requested) <= 1e-4
            ),
            None,
        )
        recovered = False
        if match is not None:
            frame = decoded[match]
            unused.remove(match)
        else:
            frame = recover_frame_near_timestamp(video_path, requested)
            recovered = frame is not None
        if frame is None:
            continue

        actual = float(frame.timestamp)
        item = copy.deepcopy(observation)
        if recovered or abs(actual - requested) > 1e-4:
            item["requested_timestamp"] = requested
            item["timestamp"] = actual
            item["decode_recovered"] = True
        aligned_observations.append(item)
        aligned_frames.append(frame)

    return aligned_observations, aligned_frames


def compact_policy_max_tokens(requested: int) -> int:
    """Bound compact action JSON below the HTTP timeout envelope.

    Router turns contain only a small JSON action.  Reusing the 4096-token
    allowance intended for historical free-form answers lets a malformed
    action ramble for longer than the 240-second request timeout, after which
    deterministic retries repeat the same failure.  The frozen final-answer
    helper already uses this bound.
    """
    return min(max(int(requested), 1), POLICY_MAX_OUTPUT_TOKENS)


def post_scan_action_prompt(timestamps: list[float]) -> str:
    """Require an explicit model action after real semantic retrieval."""
    return f"""你是视频安全Agent。你已经看到初始4帧和SigLIP返回的候选帧；query和相似度只用于定位，绝不是风险证据。
现在必须自主选择一个动作：
1. 证据已经足够：输出action=ANSWER，并给出category、severity、rationale、evidence、evidence_strength、evidence_basis、identified_figure。Harmless-L0还必须给出assessments，明确将三个风险类别均标为ABSENT。
2. 只有三个风险类别中没有任何PRESENT、且至少一个类别为INSUFFICIENT时，才输出action=TEMPORAL_EXPAND，并给出risk_types、timestamps、window、num_frames、assessments和rationale。risk_types只能列出status=INSUFFICIENT的类别；timestamps只能从当前候选时间戳选择；window范围0.5-15秒；num_frames范围1-4，优先最小够用值。
硬约束：任一类别为PRESENT时必须立即ANSWER，不得为了细分L1/L2、提高置信度或寻找更强证据而扩帧；三类均为ABSENT时必须ANSWER Harmless-L0；TEMPORAL_EXPAND不能与任何PRESENT状态同时出现。若现有事实只达到边界风险，直接ANSWER对应类别L1。
不得输出没有动作的中间分析。只有在没有已成立风险、但确实缺少时序上下文时才扩帧。
敏感旗帜涉及香港、西藏或台湾即Sensitive-L2。
assessments按三个类别分别给出PRESENT、ABSENT或INSUFFICIENT，每类包含status、candidate_timestamps和reason。
{SAFETY_POLICY}
当前可见时间戳：{[round(value, 3) for value in timestamps]}。只输出JSON。"""


def post_scan_contract_repair_prompt(candidate_times: list[float]) -> str:
    """Ask the policy to repair its own invalid tool/action payload once.

    This is schema feedback, not a framework-selected action: the model must
    still choose ANSWER versus TEMPORAL_EXPAND and supply a valid payload.
    """
    rounded = [round(value, 3) for value in candidate_times]
    return f"""上一条时序路由JSON未通过工具契约校验。请你自行修正并重新选择动作；Python不会替你选择动作、截断window、补时间戳或修改risk_types。
只能输出一个完整JSON对象：
- 若证据足够，输出合法action=ANSWER；
- 若确需前后文，输出action=TEMPORAL_EXPAND，并保证三类assessments齐全、没有PRESENT、至少一个INSUFFICIENT；risk_types恰好等于INSUFFICIENT类别；timestamps只能逐字复制合法候选时间戳；window必须在0.5至15.0之间；num_frames必须是1至4的整数。
合法候选时间戳：{rounded}。
重点检查上一条是否错误地把视频时长、时间点间隔或大于15的数写进window。只输出修正后的JSON。"""


def post_expand_answer_prompt(timestamps: list[float]) -> str:
    return f"""这是TEMPORAL_EXPAND返回真实前后文帧后的最终决策。现在必须输出action=ANSWER。
只依据整段对话中已经看到的图片判断，不得引用query或相似度作为证据。
输出字段：action,category,severity,rationale,evidence,evidence_strength,evidence_basis,identified_figure。
如果最终为Harmless-L0，还必须额外输出assessments，分别把Sexual Content、Violence & Dangerous、Sensitive Figures & Ideologies标为ABSENT并说明可见依据；缺少任一排除结论都属于无效答案。
如果为有害类别，evidence必须引用当前已见时间戳中的实际可见证据。敏感旗帜涉及香港、西藏或台湾即Sensitive-L2。
evidence.timestamp只能逐字复制“当前已见时间戳”列表中的数值；画面内字幕、播放器或计时器显示的时间不是采样时间戳，禁止引用。
{SAFETY_POLICY}
当前已见时间戳：{[round(value, 3) for value in timestamps]}。只输出JSON。"""


def post_scan_final_answer_prompt(timestamps: list[float]) -> str:
    """Ask a frozen answer policy to classify after the router stops sampling.

    The preceding ANSWER is a routing/control decision only.  Requiring a new
    independent final turn prevents temporal-router optimization from moving
    the moderation boundary used for category/severity decisions.
    """
    return f"""上一个Assistant输出的ANSWER只表示“现有帧已经足够、停止继续取帧”，不是最终类别标答。
现在请独立查看整段对话中已经获得的初始帧和SigLIP候选帧，忽略上一个Assistant提出的类别、等级和理由，重新完成最终视频安全判断。必须输出action=ANSWER。
只把图片中实际可见事实作为证据；query和相似度仅用于定位，绝不是风险证据。
输出字段：action,category,severity,rationale,evidence,evidence_strength,evidence_basis,identified_figure。
如果最终为Harmless-L0，还必须额外输出assessments，分别把Sexual Content、Violence & Dangerous、Sensitive Figures & Ideologies标为ABSENT并说明可见依据；缺少任一排除结论都属于无效答案。
如果为有害类别，evidence必须引用当前已见时间戳中的实际可见证据。敏感旗帜涉及香港、西藏或台湾即Sensitive-L2。
evidence.timestamp只能逐字复制“当前已见时间戳”列表中的数值；画面内字幕、播放器或计时器显示的时间不是采样时间戳，禁止引用。
{SAFETY_POLICY}
当前已见时间戳：{[round(value, 3) for value in timestamps]}。只输出JSON。"""


def frozen_answer_assessment_prompt(timestamps: list[float], *, after_temporal: bool) -> str:
    """Use the exact post-tool schema learned by the frozen SFT-v5 adapter.

    SFT-v5 was supervised on a bare ``assessments`` object with an optional
    nested ``final_answer``; the historical runner added ``ASSESS_RISKS`` only
    after generation.  Forcing that adapter to emit the newer top-level
    ``ANSWER`` schema is out-of-distribution and caused otherwise correct
    classifications to hallucinate timestamps from video-player overlays.
    """
    rounded = [round(value, 3) for value in timestamps]
    indexed = [f"第{index + 1}张图={value}s" for index, value in enumerate(rounded)]
    stage = (
        "这些图片已包含语义检索候选帧和TEMPORAL_EXPAND返回的真实前后文帧。"
        if after_temporal else
        "这些图片已包含初始帧和语义检索返回的真实候选帧。"
    )
    return f"""你是冻结的视频安全最终判断模型。{stage}
独立查看当前同时提供的所有图片，不得沿用路由模型先前提出的类别、等级或理由。
SigLIP query和相似度只用于定位，绝不是风险证据。

严格按SFT-v5原生协议输出一个JSON对象：
- 顶层必须包含assessments，分别对Sexual Content、Violence & Dangerous、Sensitive Figures & Ideologies给出status、candidate_timestamps和reason。
- status只能是PRESENT、ABSENT或INSUFFICIENT。本轮必须完成最终判断；如果某类风险仍证据不足，将其按可见证据的边界程度判为L1或排除，不得再请求工具。
- 顶层必须包含final_answer，其字段为category、severity、rationale、evidence、evidence_strength、evidence_basis、identified_figure。
- 任一类PRESENT时，final_answer必须是对应的有害类别L1/L2；三类全部ABSENT时，final_answer必须是Harmless-L0。
- 不要输出action字段；运行器会把该原生输出规范化为最终ANSWER。
- evidence必须是JSON数组，每项为{{"timestamp":秒数,"description":"该图中实际可见事实"}}。
- evidence.timestamp和candidate_timestamps只能逐字复制下面映射中的数值。画面内字幕、播放器、计时器或拍摄画面自带的数字不是采样时间戳，禁止引用。
- 图片与可引用时间戳按顺序一一对应：{indexed}。
- 敏感旗帜涉及香港、西藏或台湾即Sensitive-L2。
{SAFETY_POLICY}
可引用时间戳：{rounded}。只输出JSON。"""


def frozen_answer_grounding_repair_prompt(timestamps: list[float]) -> str:
    """Fresh visual retry for a helper that keeps copying on-screen clocks.

    The retry deliberately starts a new model context and resends every real
    image.  It therefore repairs only grounding/schema generation; it neither
    inherits a previous verdict nor maps an invalid number to a frame in
    Python.  The ordinary successful helper path is unchanged.
    """
    rounded = [round(value, 3) for value in timestamps]
    indexed = [f"第{index + 1}张图唯一对应{value}s" for index, value in enumerate(rounded)]
    return f"""你是冻结的视频安全最终判断模型。这是一次全新、独立的视觉判断；不存在可沿用的先前答案。
你必须重新查看同时提供的图片，完成三类风险评估和最终分类。

时间戳硬规则：
- 图片顺序与采样时间的唯一映射是：{indexed}。
- candidate_timestamps和evidence.timestamp只能从{rounded}中逐字复制。
- 画面内的字幕、播放器、计时器、镜头序号或其他数字全部不是采样时间，绝对不得输出。
- 如果风险事实出现在第N张图，必须使用上述映射中第N张图的时间。Python不会替你猜测或修改时间。

严格输出SFT-v5原生JSON：顶层只包含assessments和final_answer，不得输出action。
assessments必须完整包含Sexual Content、Violence & Dangerous、Sensitive Figures & Ideologies，
每类包含status、candidate_timestamps和reason。status只能是PRESENT、ABSENT或INSUFFICIENT。
final_answer必须包含category、severity、rationale、evidence、evidence_strength、evidence_basis和identified_figure。
任一类PRESENT时输出对应有害类别L1/L2并给出1至2条实际可见证据；三类全部ABSENT时输出Harmless-L0。
SigLIP query和相似度不是证据。敏感旗帜涉及香港、西藏或台湾即Sensitive-L2。
{SAFETY_POLICY}
只输出JSON。"""


def assessment_statuses(data: dict) -> tuple[dict, dict[str, str]]:
    assessments = normalize_assessments(data.get("assessments") or {})
    statuses = {
        risk: str((assessments.get(risk) or {}).get("status") or "").upper()
        for risk in RISKS
    }
    return assessments, statuses


def valid_post_scan_answer(data: dict, visible_times: list[float]) -> dict | None:
    if str(data.get("action") or "").upper() != "ANSWER":
        return None
    answer = valid_final(data, visible_times)
    if not answer:
        return None
    if answer["category"] == "Harmless":
        assessments, statuses = assessment_statuses(data)
        if len(assessments) != 3 or not all(statuses.get(risk) == "ABSENT" for risk in RISKS):
            return None
    return answer


def valid_post_scan_stop_route(data: dict, visible_times: list[float]) -> dict | None:
    """Validate that an ANSWER route is consistent with its own assessments.

    When a frozen helper owns the final verdict, the trainable model still owns
    stop-vs-expand.  It may stop only after all three risks are resolved: a
    harmful stop must mark its selected category PRESENT, while a Harmless stop
    must mark every risk ABSENT.  In particular, ANSWER plus INSUFFICIENT is an
    invalid route rather than permission to invoke the frozen answer helper.
    """
    answer = valid_post_scan_answer(data, visible_times)
    if not answer:
        return None
    assessments, statuses = assessment_statuses(data)
    if len(assessments) != len(RISKS):
        return None
    if any(statuses.get(risk) not in {"PRESENT", "ABSENT"} for risk in RISKS):
        return None
    if answer["category"] == "Harmless":
        return answer if all(statuses[risk] == "ABSENT" for risk in RISKS) else None
    return answer if statuses.get(answer["category"]) == "PRESENT" else None


def valid_post_scan_runtime_answer(
    data: dict,
    visible_times: list[float],
    *,
    independent_answer_model: bool,
) -> dict | None:
    """Select the validator that matches ownership of the final verdict.

    With an independent final-answer model, the trainable policy owns only the
    stop-versus-expand route and therefore must satisfy the strict assessment
    contract.  The legacy single-model path retains its historical terminal
    validator for reproducibility.
    """
    if independent_answer_model:
        return valid_post_scan_stop_route(data, visible_times)
    return valid_post_scan_answer(data, visible_times)


def valid_post_expand_answer(data: dict, visible_times: list[float]) -> dict | None:
    """The terminal v2 action is explicit and Harmless is fail-closed."""
    if str(data.get("action") or "").upper() != "ANSWER":
        return None
    answer = valid_final(data, visible_times)
    if not answer:
        return None
    if answer["category"] == "Harmless":
        assessments, statuses = assessment_statuses(data)
        if len(assessments) != 3 or not all(statuses.get(risk) == "ABSENT" for risk in RISKS):
            return None
    return answer


def valid_frozen_answer_output(data: dict, visible_times: list[float]) -> dict | None:
    """Validate both current ANSWER and frozen SFT-v5 legacy assessment JSON.

    SFT-v5 was deliberately trained to return ``ASSESS_RISKS`` with a nested
    ``final_answer`` after semantic retrieval. That is a valid frozen answer
    protocol, but it must never relax the trainable router's action contract.
    """
    def coerce_legacy_harmless_assessments(payload: dict) -> dict:
        """Canonicalize only the frozen SFT-v5 helper's legacy schema.

        The frozen checkpoint often emits ``"ABSENT - reason"`` instead of
        the newer nested assessment object.  The meaning is unambiguous, so
        normalize that representation here without weakening the trainable
        router's strict action validator.
        """
        candidate = dict(payload)
        if canonical_category(candidate.get("category")) != "Harmless":
            return candidate
        normalized = normalize_assessments(candidate.get("assessments") or {})
        canonical = {}
        for risk, assessment in normalized.items():
            item = dict(assessment)
            raw_status = str(item.get("status") or "").strip()
            match = re.match(
                r"^(PRESENT|ABSENT|INSUFFICIENT)\b\s*(?:[-:\u2014]\s*)?(.*)$",
                raw_status,
                flags=re.IGNORECASE,
            )
            if match:
                item["status"] = match.group(1).upper()
                if not item.get("reason") and match.group(2).strip():
                    item["reason"] = match.group(2).strip()
            item.setdefault("candidate_timestamps", [])
            canonical[risk] = item
        candidate["assessments"] = canonical
        return candidate

    def promote_misnested_final_answer(payload: dict) -> dict:
        """Recover one unambiguous frozen-helper brace-placement mistake.

        Rare SFT-v5 generations place ``final_answer`` one level too deep,
        either as a peer inside ``assessments`` or inside the last risk
        assessment.  The complete answer object is still present and this
        structural move has only one interpretation.  Keep the normalization
        private to the frozen helper boundary so trainable router actions
        remain subject to the strict protocol validators.
        """
        candidate = dict(payload)
        if isinstance(candidate.get("final_answer"), dict):
            return candidate
        raw_assessments = candidate.get("assessments")
        if not isinstance(raw_assessments, dict):
            return candidate
        assessments = dict(raw_assessments)
        found: list[tuple[str, str | None, dict]] = []
        peer = assessments.get("final_answer")
        if isinstance(peer, dict):
            found.append(("peer", None, peer))
        for key, value in assessments.items():
            if canonical_category(key) not in RISKS or not isinstance(value, dict):
                continue
            nested = value.get("final_answer")
            if isinstance(nested, dict):
                found.append(("risk", key, nested))
        if len(found) != 1:
            return candidate
        location, key, final_answer = found[0]
        if location == "peer":
            assessments.pop("final_answer", None)
        else:
            repaired_risk = dict(assessments[key])
            repaired_risk.pop("final_answer", None)
            assessments[key] = repaired_risk
        candidate["assessments"] = assessments
        candidate["final_answer"] = final_answer
        return candidate

    def recover_harmful_evidence_from_assessment(payload: dict) -> dict:
        """Reuse a matching PRESENT assessment when final evidence is omitted.

        A frozen helper can finish a valid JSON object early after writing the
        final category/severity/rationale but before repeating its already
        emitted evidence array.  Recovery is safe only when the final harmful
        category has a PRESENT assessment with a non-empty visual reason and
        candidate timestamps that snap to frames actually shown to the model.
        No verdict or severity is synthesized here.
        """
        candidate = dict(payload)
        final_answer = candidate.get("final_answer")
        if not isinstance(final_answer, dict):
            return candidate
        category = canonical_category(final_answer.get("category"))
        if category not in RISKS or evidence(final_answer, visible_times):
            return candidate
        assessments = normalize_assessments(candidate.get("assessments") or {})
        assessment = assessments.get(category)
        if not isinstance(assessment, dict):
            return candidate
        if str(assessment.get("status") or "").upper() != "PRESENT":
            return candidate
        reason = str(assessment.get("reason") or "").strip()
        timestamps = assessment.get("candidate_timestamps") or []
        if not reason or not isinstance(timestamps, list):
            return candidate
        recovered = evidence({
            "evidence": [
                {"timestamp": timestamp, "description": reason}
                for timestamp in timestamps
            ],
        }, visible_times)
        if not recovered:
            return candidate
        repaired_answer = dict(final_answer)
        repaired_answer["evidence"] = recovered
        candidate["final_answer"] = repaired_answer
        return candidate

    data = promote_misnested_final_answer(data)
    data = recover_harmful_evidence_from_assessment(data)
    current = coerce_legacy_harmless_assessments(data)
    answer = valid_post_expand_answer(current, visible_times)
    if answer:
        return answer
    # The historical evaluator injected ASSESS_RISKS into the transcript;
    # SFT-v5's actual supervised target was the bare object.  Accept both only
    # on this frozen-helper boundary.  Trainable router validators stay strict.
    if str(data.get("action") or "").upper() not in {"", "ASSESS_RISKS"}:
        return None
    nested = data.get("final_answer")
    if not isinstance(nested, dict):
        return None
    candidate = dict(nested)
    candidate["action"] = "ANSWER"
    if canonical_category(candidate.get("category")) == "Harmless":
        candidate["assessments"] = data.get("assessments") or {}
    candidate = coerce_legacy_harmless_assessments(candidate)
    return valid_post_expand_answer(candidate, visible_times)


def model_expand_request(
    data: dict,
    candidate_times: list[float],
    *,
    normalize_bounds: bool = False,
) -> dict | None:
    if str(data.get("action") or "").upper() != "TEMPORAL_EXPAND":
        return None
    # Temporal expansion is an evidence-acquisition action, not a way to
    # strengthen an already-established finding.  Keep the runtime contract
    # identical to the repaired warm-up data: every risk must be assessed, no
    # risk may already be PRESENT, and at least one risk must be INSUFFICIENT.
    assessments, statuses = assessment_statuses(data)
    if len(assessments) != len(RISKS):
        return None
    if any(statuses.get(risk) not in {"PRESENT", "ABSENT", "INSUFFICIENT"}
           for risk in RISKS):
        return None
    if any(statuses[risk] == "PRESENT" for risk in RISKS):
        return None
    insufficient_risks = {risk for risk in RISKS if statuses[risk] == "INSUFFICIENT"}
    if not insufficient_risks:
        return None
    raw_times = data.get("timestamps") or []
    if not isinstance(raw_times, list):
        return None
    selected = []
    for value in raw_times:
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            continue
        matched = next((known for known in candidate_times if abs(timestamp - known) <= 0.3), None)
        if matched is not None and matched not in selected:
            selected.append(float(matched))
    if not selected:
        return None
    raw_risks = data.get("risk_types") or []
    if isinstance(raw_risks, str):
        raw_risks = [raw_risks]
    risks = []
    for value in raw_risks:
        risk = canonical_category(value)
        if risk in RISKS and risk not in risks:
            risks.append(risk)
    # ``risk_types`` is part of the executable tool contract, not merely a
    # hint.  A strict equality check prevents the model from silently omitting
    # one of several unresolved risks and then expanding context for only a
    # convenient subset.
    if set(risks) != insufficient_risks:
        return None
    try:
        window = float(data.get("window"))
        num_frames = int(data.get("num_frames"))
    except (TypeError, ValueError):
        return None
    # The original Agent contract allows up to +/-15 seconds. Frame cost is
    # bounded independently by num_frames<=4, so a wider temporal window does
    # not increase the number of images sent to the VLM.
    normalized_parameters = {}
    if not 0.5 <= window <= 15.0:
        if not normalize_bounds:
            return None
        normalized_parameters["window"] = {"requested": window, "executed": min(max(window, 0.5), 15.0)}
        window = normalized_parameters["window"]["executed"]
    if not 1 <= num_frames <= 4:
        if not normalize_bounds:
            return None
        normalized_parameters["num_frames"] = {
            "requested": num_frames,
            "executed": min(max(num_frames, 1), 4),
        }
        num_frames = normalized_parameters["num_frames"]["executed"]
    request = {
        "action": "TEMPORAL_EXPAND",
        "risk_types": risks,
        "timestamps": selected[:2],
        "window": window,
        "num_frames": num_frames,
        "assessments": assessments,
        "rationale": str(data.get("rationale") or "")[:500],
        "provenance": "model",
        "model_initiated": True,
    }
    if normalized_parameters:
        request["normalized_parameters"] = normalized_parameters
    return request


def framework_expand_request(data: dict, hits, candidate_times: list[float]) -> dict:
    assessments, statuses = assessment_statuses(data)
    risks = [risk for risk in RISKS if statuses.get(risk) in {"PRESENT", "INSUFFICIENT"}]
    if not risks:
        risks = list(RISKS)
    selected = []
    for risk in risks:
        requested = (assessments.get(risk) or {}).get("candidate_timestamps") or []
        for value in requested:
            try:
                timestamp = float(value)
            except (TypeError, ValueError):
                continue
            matched = next((known for known in candidate_times if abs(timestamp - known) <= 0.3), None)
            if matched is not None and matched not in selected:
                selected.append(float(matched))
        if not selected:
            selected.extend(float(hit.timestamp) for hit in hits if hit.risk_type == risk)
    selected = list(dict.fromkeys(selected))[:2] or candidate_times[:2]
    return {
        "action": "TEMPORAL_EXPAND",
        "risk_types": risks,
        "timestamps": selected,
        "window": 4.0,
        "num_frames": 4,
        "automatic": True,
        "provenance": "framework",
        "reason": "post-scan policy did not return a valid grounded ANSWER or TEMPORAL_EXPAND action",
    }


def make_policy_failure_record(row: dict, args, brain: CausalBrain, transcript: list[dict],
                               reason: str, total_frames: int) -> dict:
    """Persist a model-policy failure as a trainable -5 rollout, not infra retry."""
    first = next((
        step["decision"] for step in transcript
        if isinstance(step.get("decision"), dict) and not step["decision"].get("automatic")
    ), {})
    initial_times = ((transcript[0].get("observation") or {}).get("timestamps") or [])
    first_answer = valid_final(first, initial_times)
    return {
        "id": row["id"],
        "source_id": row.get("source_id", row["id"]),
        "rollout_index": row.get("rollout_index"),
        "rollout_seed": row.get("rollout_seed", args.seed),
        "label": row["category"],
        "label_severity": row["severity"],
        "daa_group": row.get("daa_group"),
        "tool_trigger_source": row.get("tool_trigger_source"),
        "model": args.model,
        "initial_route_model": getattr(args, "initial_model", None) or args.model,
        "post_tool_model": getattr(args, "answer_model", None) or args.model,
        "answer_model": getattr(args, "answer_model", None),
        "initial_payload_model": None,
        "agent_version": args.agent_version,
        "requested_max_output_tokens": getattr(args, "max_output_tokens", None),
        "effective_policy_max_output_tokens": getattr(
            args, "effective_policy_max_output_tokens",
            compact_policy_max_tokens(args.max_output_tokens),
        ),
        "initial_action": str(first.get("action") or "").upper() or None,
        "initial_valid_json_action": valid_initial_action(first, initial_times),
        "initial_pred": first_answer.get("category") if first_answer else None,
        "initial_severity": first_answer.get("severity") if first_answer else None,
        "pred": None,
        "severity": None,
        "tool_calls": [step["tool"] for step in transcript if step.get("tool")],
        "result": {"final_action": {}, "transcript": transcript, "total_frames": total_frames},
        "model_turns": brain.rollout_turns,
        "usage_by_round": brain.usage,
        "raw_json_attempts": brain.raw_json_attempts,
        "raw_json_successes": brain.raw_json_successes,
        "turn_attempt_counts": brain.turn_attempt_counts,
        "raw_first_attempt_json_rate": (
            sum(value == 1 for value in brain.turn_attempt_counts) / len(brain.turn_attempt_counts)
            if brain.turn_attempt_counts else 0.0
        ),
        "policy_error": reason,
        "error": None,
    }


class InitialObservationCache:
    """Decode one initial observation per source video across rollout siblings."""

    def __init__(self, max_size: int = 16):
        self.max_size = max_size
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple, concurrent.futures.Future] = OrderedDict()

    def get(self, path: Path) -> tuple[float, tuple]:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        owner = False
        with self._lock:
            future = self._entries.get(key)
            if future is None:
                future = concurrent.futures.Future()
                self._entries[key] = future
                owner = True
            self._entries.move_to_end(key)
        if owner:
            try:
                value = (video_duration(str(path)), tuple(uniform_frames(str(path), 4)))
                future.set_result(value)
            except BaseException as exc:
                future.set_exception(exc)
                with self._lock:
                    self._entries.pop(key, None)
                raise
            with self._lock:
                while len(self._entries) > max(1, self.max_size):
                    oldest_key, oldest = next(iter(self._entries.items()))
                    if not oldest.done():
                        break
                    self._entries.pop(oldest_key)
        return future.result()


class SharedInitialRouteCache:
    """Generate one exact initial-route conversation snapshot per rollout group.

    Repeating an allegedly identical temperature-zero request through a
    continuously batched vLLM server is not guaranteed to be byte-identical.
    Temporal counterfactual GRPO requires a stronger invariant: sibling
    rollouts must start the trainable post-scan turn from the *same* causal
    prefix.  The first sibling therefore owns the real model call and the
    remaining siblings restore a deep copy of its complete conversation and
    audit state.
    """

    def __init__(self, expected_uses: int = 4):
        self.expected_uses = max(1, int(expected_uses))
        self._lock = threading.Lock()
        self._entries: dict[tuple, concurrent.futures.Future] = {}
        self._uses: dict[tuple, int] = {}

    def get_or_create(self, key: tuple, factory) -> dict:
        owner = False
        with self._lock:
            future = self._entries.get(key)
            if future is None:
                future = concurrent.futures.Future()
                self._entries[key] = future
                self._uses[key] = 0
                owner = True
        if owner:
            try:
                future.set_result(copy.deepcopy(factory()))
            except BaseException as exc:
                future.set_exception(exc)
        try:
            snapshot = copy.deepcopy(future.result())
        finally:
            with self._lock:
                self._uses[key] = self._uses.get(key, 0) + 1
                if self._uses[key] >= self.expected_uses:
                    self._entries.pop(key, None)
                    self._uses.pop(key, None)
        return snapshot


def initial_route_snapshot(brain: CausalBrain, first: dict) -> dict:
    """Capture every mutable field needed to resume one causal conversation."""
    return {
        "first": first,
        "messages": brain.messages,
        "rollout_turns": brain.rollout_turns,
        "usage": brain.usage,
        "raw_json_attempts": brain.raw_json_attempts,
        "raw_json_successes": brain.raw_json_successes,
        "turn_attempt_counts": brain.turn_attempt_counts,
        "seed_turn_index": brain.seed_turn_index,
    }


def restore_initial_route(brain: CausalBrain, snapshot: dict) -> dict:
    """Restore a private copy so later sibling turns cannot mutate the cache."""
    brain.messages = copy.deepcopy(snapshot["messages"])
    brain.rollout_turns = copy.deepcopy(snapshot["rollout_turns"])
    brain.usage = copy.deepcopy(snapshot["usage"])
    brain.raw_json_attempts = int(snapshot["raw_json_attempts"])
    brain.raw_json_successes = int(snapshot["raw_json_successes"])
    brain.turn_attempt_counts = copy.deepcopy(snapshot["turn_attempt_counts"])
    brain.seed_turn_index = int(snapshot["seed_turn_index"])
    return copy.deepcopy(snapshot["first"])


def call_independent_answer(
    row: dict,
    args,
    main_brain: CausalBrain,
    frames: list,
    prompt: str,
    visible_times: list[float],
) -> dict:
    """Run the frozen answer policy in a clean, independent visual context.

    Feeding the temporal router's free-form category/rationale into the answer
    model causes anchoring even when the prompt says to ignore it.  The answer
    policy therefore receives only real observations and tool results.  Its
    audited turn is attached to the rollout record, but excluded from the
    trainable policy transcript and GRPO log-probability.
    """
    if not args.answer_model:
        raise ValueError("independent answer requires --answer-model")
    # Final classification JSON is compact.  Keeping the helper at 1024
    # tokens prevents a malformed schema-repair turn from emitting thousands
    # of repetitive tokens and stalling a 16-worker rollout for minutes.
    helper_max_tokens = compact_policy_max_tokens(args.max_output_tokens)
    answer_seed = int(row.get("answer_seed", row.get("rollout_seed", args.seed)))
    helper = CausalBrain(
        args.answer_model, args.base_url, args.api_key, helper_max_tokens,
        0.0, 1.0, answer_seed + 100_003,
    )
    result = helper.call_turn(
        frames, prompt, policy_owned=False, action_stage="final_answer",
    )
    final = valid_frozen_answer_output(result, visible_times)
    if not final:
        result = helper.call_turn(
            [],
            "上一条最终判断未通过结构校验。不要调用任何工具，也不要沿用其他模型的判断。"
            "请恢复SFT-v5原生格式：不要输出action；顶层必须同时包含assessments和final_answer。"
            "category和severity必须合法；有害类别必须给出引用已见时间戳的可见证据；"
            "Harmless-L0必须将Sexual Content、Violence & Dangerous、Sensitive Figures & Ideologies"
            "三类全部明确标为ABSENT。assessments的每个值必须是JSON对象，"
            "禁止写成字符串，严格使用如下结构："
            "{\"assessments\":{" 
            "\"Sexual Content\":{\"status\":\"ABSENT\",\"candidate_timestamps\":[],\"reason\":\"可见依据\"},"
            "\"Violence & Dangerous\":{\"status\":\"ABSENT\",\"candidate_timestamps\":[],\"reason\":\"可见依据\"},"
            "\"Sensitive Figures & Ideologies\":{\"status\":\"ABSENT\",\"candidate_timestamps\":[],\"reason\":\"可见依据\"}},"
            "\"final_answer\":{\"category\":\"Harmless\",\"severity\":\"L0\",\"rationale\":\"依据\","
            "\"evidence\":[],\"evidence_strength\":\"clear\",\"evidence_basis\":\"visible_content\","
            "\"identified_figure\":null}}。"
            "evidence.timestamp必须逐字复制下列可引用时间戳中的一个数字；不得使用画面内字幕、"
            "播放器或计时器显示的数字。如果上一条证据时间不在列表中，必须改成与该可见图片"
            "对应的列表时间；无法对应时删除该证据，禁止编造。"
            f"可引用时间戳：{[round(value, 3) for value in visible_times]}。只输出JSON。",
            policy_owned=False,
            action_stage="final_answer",
        )
        final = valid_frozen_answer_output(result, visible_times)
    helpers = [(helper, "frozen_final_answer")]
    if not final:
        # A small fraction of otherwise valid frozen-helper answers repeatedly
        # copy a clock/subtitle number from inside the image as the evidence
        # timestamp.  Repeating the same conversation is deterministic and
        # cannot repair that error.  Start one fresh visual context, resend the
        # real frames, and require index-to-timestamp grounding.  Validation
        # remains fail-closed; Python never snaps or invents evidence times.
        grounding_helper = CausalBrain(
            args.answer_model, args.base_url, args.api_key, helper_max_tokens,
            0.0, 1.0, answer_seed + 200_003,
        )
        result = grounding_helper.call_turn(
            frames,
            frozen_answer_grounding_repair_prompt(visible_times),
            policy_owned=False,
            action_stage="final_answer_grounding_repair",
        )
        final = valid_frozen_answer_output(result, visible_times)
        helpers.append((grounding_helper, "frozen_final_answer_grounding_repair"))
    for answer_helper, turn_kind in helpers:
        for turn in answer_helper.rollout_turns:
            turn["exclude_from_policy_trajectory"] = True
            turn["independent_context"] = True
        main_brain.record_shadow_call(
            answer_helper, turn_kind=turn_kind, adopt_as_context=False,
        )
    if not final:
        raise RuntimeError(
            "frozen answer model remained invalid after conversational and fresh visual repairs: "
            + json_text(result)[:2000]
        )
    return final


def run_one(row: dict, args, locator: JointRiskSemanticLocator,
            initial_cache: InitialObservationCache,
            initial_route_cache: SharedInitialRouteCache | None = None) -> dict:
    args.effective_policy_max_output_tokens = compact_policy_max_tokens(
        args.max_output_tokens,
    )
    path = ROOT / row["local_path"]
    duration, cached_initial = initial_cache.get(path)
    initial = list(cached_initial)
    if len(initial) != 4:
        raise RuntimeError(f"decoded {len(initial)}/4 initial frames")
    # GRPO-v2 may specialize only the post-scan stop-vs-expand policy while
    # keeping the already accepted GRPO-v1 initial router frozen.  Both are
    # Qwen3.5-4B LoRA adapters and share one causal conversation; vLLM selects
    # the adapter per turn.  The frozen initial action remains visible context
    # but is excluded from the temporal adapter's GRPO log-probability.
    modular_initial = bool(args.initial_model)
    rollout_seed = int(row.get("rollout_seed", args.seed))
    initial_route_seed = int(row.get("initial_route_seed", rollout_seed))
    brain = CausalBrain(
        args.initial_model or args.model,
        args.base_url,
        args.api_key,
        args.effective_policy_max_output_tokens,
        args.temperature,
        args.top_p,
        initial_route_seed,
        post_tool_model=args.model if modular_initial else None,
    )
    initial_times = [frame.timestamp for frame in initial]
    transcript = [{"observation": {
        "duration": duration,
        "frames_seen": 4,
        "timestamps": initial_times,
        "initial_sampling": "uniform_4_non_black",
    }}]

    route_prompt = initial_prompt(initial_times)

    def generate_initial_route() -> dict:
        first_result = brain.call_turn(
            initial,
            route_prompt,
            action_stage="initial_route",
            policy_owned=not modular_initial,
            temperature_override=getattr(args, "initial_temperature", None),
        )
        return initial_route_snapshot(brain, first_result)

    share_initial = bool(row.get("v3_i4_shared_state")) and initial_route_cache is not None
    if share_initial:
        route_key = (
            str(path.resolve()),
            args.initial_model or args.model,
            initial_route_seed,
            getattr(args, "initial_temperature", None),
            args.top_p,
            args.effective_policy_max_output_tokens,
            route_prompt,
        )
        first = restore_initial_route(
            brain,
            initial_route_cache.get_or_create(route_key, generate_initial_route),
        )
    else:
        first = restore_initial_route(brain, generate_initial_route())
    # Persist semantic stage ownership explicitly.  The GRPO collator must
    # distinguish the initial route from the later ANSWER-vs-EXPAND route;
    # inferring solely from the chosen action would omit ANSWER counterfactuals.
    transcript.append({"decision": first})
    first_action = str(first.get("action") or "").upper()
    initial_answer = valid_final(first, initial_times)
    if first_action == "ANSWER" and valid_initial_direct_harm(initial_answer):
        return make_record(row, args, brain, transcript, initial_answer, 4)

    normalized = normalize_queries(first.get("queries_by_risk")) if first_action == "JOINT_SEMANTIC_SCAN" else None
    automatic_scan = None
    if not normalized:
        automatic_scan = {
            "action": "JOINT_SEMANTIC_SCAN",
            "automatic": True,
            "reason": "initial action could not safely bypass evidence acquisition",
        }
        transcript.append({"decision": automatic_scan})
    dynamic = normalize_dynamic_queries(
        first.get("queries_by_risk") or first.get("queries") or first.get("query") or [],
        canonical_category(first.get("category")),
    )
    if normalized:
        dynamic = normalized
    queries_by_risk = {risk: risk_queries(risk, dynamic.get(risk, [])) for risk in RISKS}
    if automatic_scan is not None:
        # A malformed or incomplete initial ANSWER is not an accepted causal
        # decision.  Keep the raw output in ``rollout_turns`` for audit, but
        # give the post-scan policy the same valid assistant transition it was
        # trained on: an actual scan action with the executed query payload.
        # Replacing it only after query normalization avoids the underspecified
        # ``{"action":"JOINT_SEMANTIC_SCAN"}`` context that caused the next
        # adapter to emit action-only or copied answers.
        safe_scan_context = {
            "action": "JOINT_SEMANTIC_SCAN",
            "queries_by_risk": queries_by_risk,
            "automatic": True,
        }
        for message in reversed(brain.messages):
            if message.get("role") == "assistant":
                message["content"] = json_text(safe_scan_context)
                break
    hits = locator(str(path), queries_by_risk, top_k_per_risk=args.top_k)
    merged = evenly_take(merge_hits(hits), 8)
    if not merged:
        raise RuntimeError("joint semantic scan returned no candidate frames")
    requested_candidate_count = len(merged)
    merged, candidates = decode_semantic_candidates(str(path), merged)
    if not candidates:
        raise RuntimeError("joint semantic scan candidates could not be decoded")
    candidate_times = [float(frame.timestamp) for frame in candidates]
    transcript.append({
        "tool": "JOINT_SEMANTIC_SCAN",
        "queries_by_risk": queries_by_risk,
        "observation": merged,
        "requested_candidate_count": requested_candidate_count,
        "decoded_candidate_count": len(candidates),
    })
    scan_state = {
        "executed_tool_call": {
            "tool": "JOINT_SEMANTIC_SCAN",
            "queries_by_risk": queries_by_risk,
            "automatic_framework_transition": automatic_scan,
        },
        "actual_tool_result": {
            "tool": "JOINT_SEMANTIC_SCAN",
            "observation": merged,
            "returned_frame_timestamps": candidate_times,
            "requested_candidate_count": requested_candidate_count,
            "decoded_candidate_count": len(candidates),
        },
    }
    visible_review = initial_times + candidate_times
    temporal_call_kwargs = {"action_stage": "temporal_route"}
    if row.get("temporal_route_seed") is not None:
        temporal_call_kwargs["seed_override"] = int(row["temporal_route_seed"])
    post_scan = brain.call_turn(
        candidates,
        "此前调用及真实工具结果如下。\n" + json_text(scan_state)
        + "\n\n" + post_scan_action_prompt(visible_review),
        **temporal_call_kwargs,
    )
    # Both legal actions at this state belong to the temporal routing policy.
    # ANSWER is the essential negative/control action for TEMPORAL_EXPAND and
    # must receive the same stagewise group-relative credit.
    repair_attempts = max(int(getattr(args, "post_scan_repair_attempts", 0)), 0)
    expand = None
    for repair_index in range(repair_attempts + 1):
        post_scan_action = str(post_scan.get("action") or "").upper()
        if (
            post_scan_action == "ANSWER"
            and args.answer_model
            and valid_post_scan_stop_route(post_scan, visible_review)
        ):
            # The trainable policy owns only the stop-vs-expand action here. A
            # frozen policy independently owns category/severity/evidence.
            transcript.append({"decision": {**post_scan, "provenance": "model_route"}})
            final_data = call_independent_answer(
                row, args, brain, initial + candidates,
                "已执行的真实语义检索如下。\n" + json_text(scan_state)
                + "\n\n" + frozen_answer_assessment_prompt(
                    visible_review, after_temporal=False,
                ),
                visible_review,
            )
            final = final_data
            transcript.append({"decision": {**final, "provenance": "frozen_answer_model"}})
            record = make_record(row, args, brain, transcript, final, 4 + len(candidates))
            record["answer_model"] = args.answer_model
            return record

        answer = valid_post_scan_runtime_answer(
            post_scan,
            visible_review,
            independent_answer_model=bool(args.answer_model),
        )
        if answer:
            transcript.append({"decision": {**post_scan, "provenance": "model"}})
            return make_record(row, args, brain, transcript, answer, 4 + len(candidates))

        expand = model_expand_request(
            post_scan,
            candidate_times,
            normalize_bounds=bool(getattr(args, "normalize_temporal_bounds", False)),
        )
        if expand:
            break
        if repair_index >= repair_attempts:
            break
        transcript.append({"decision": {
            **post_scan,
            "original_action": post_scan_action or None,
            "action": "INVALID_POST_SCAN",
            "provenance": "model_invalid_retryable",
        }})
        post_scan = brain.call_turn(
            [],
            post_scan_contract_repair_prompt(candidate_times),
            action_stage="temporal_route_repair",
            policy_owned=True,
        )

    if expand:
        transcript.append({"decision": expand})
    else:
        # A stage-specific temporal adapter can emit the correct action but an
        # incomplete payload (for example, only ``{"action":
        # "TEMPORAL_EXPAND"}``).  Do not invent timestamps or silently turn
        # the malformed object into a tool call.  When an independent answer
        # policy is available, use it on the real semantic-scan evidence as a
        # conservative fallback and record the route failure explicitly.
        if args.answer_model:
            transcript.append({"decision": {
                **post_scan,
                "original_action": str(post_scan.get("action") or "").upper() or None,
                "action": "INVALID_POST_SCAN_FALLBACK_ANSWER",
                "provenance": "model_invalid_fallback",
            }})
            fallback_final = call_independent_answer(
                row, args, brain, initial + candidates,
                "语义检索已执行，但后续路由输出不符合工具协议。"
                "不要调用工具；只依据当前真实图片和检索结果，直接给出最终判断。\n"
                + json_text(scan_state) + "\n\n"
                + frozen_answer_assessment_prompt(visible_review, after_temporal=False),
                visible_review,
            )
            transcript.append({"decision": {
                **fallback_final, "provenance": "fallback_answer_model",
            }})
            record = make_record(
                row, args, brain, transcript, fallback_final,
                4 + len(candidates),
            )
            record["answer_model"] = args.answer_model
            record["post_scan_fallback"] = "invalid_route_to_independent_answer"
            return record
        # Persist invalid/legacy policy output with a -5 reward.  A safety
        # fallback is useful in deployment, but using its eventual success as
        # GRPO supervision would teach the model to rely on Python rescue.
        transcript.append({"decision": {
            **post_scan,
            "original_action": str(post_scan.get("action") or "").upper() or None,
            "action": "INVALID_POST_SCAN",
            "provenance": "model_invalid",
        }})
        return make_policy_failure_record(
            row, args, brain, transcript,
            "post-scan response was neither a grounded ANSWER nor a valid TEMPORAL_EXPAND",
            4 + len(candidates),
        )

    expanded = TemporalEvidenceExpander()(
        str(path), expand["timestamps"], window=expand["window"], num_frames=expand["num_frames"],
    )
    if not expanded:
        raise RuntimeError("temporal expansion returned no frames")
    expanded_times = [frame.timestamp for frame in expanded]
    transcript.append({
        "tool": "TEMPORAL_EXPAND",
        "risk_types": expand["risk_types"],
        "observation": {"timestamps": expanded_times, "added_frames": len(expanded)},
    })
    temporal_state = {
        "executed_tool_call": {"tool": "TEMPORAL_EXPAND", "value": expand},
        "actual_tool_result": {
            "tool": "TEMPORAL_EXPAND",
            "observation": {"timestamps": expanded_times, "added_frames": len(expanded)},
        },
    }
    visible_final = visible_review + expanded_times
    if args.answer_model:
        final_data = call_independent_answer(
            row, args, brain, initial + candidates + expanded,
            "已执行的真实语义检索和时序扩展如下。只依据同时提供的真实图片完成判断。\n"
            + json_text({"semantic_scan": scan_state, "temporal_expand": temporal_state})
            + "\n\n" + frozen_answer_assessment_prompt(
                visible_final, after_temporal=True,
            ),
            visible_final,
        )
    else:
        final_data = brain.call_turn(
            expanded,
            "此前调用及真实工具结果如下。只依据已经返回的图片完成判断。\n"
            + json_text(temporal_state) + "\n\n" + post_expand_answer_prompt(visible_final),
            action_stage="final_answer",
        )
    final = final_data if args.answer_model else valid_post_expand_answer(final_data, visible_final)
    if not final:
        transcript.append({"decision": {"action": "INVALID_FINAL", **final_data}})
        return make_policy_failure_record(
            row, args, brain, transcript,
            "post-expand final answer was invalid or lacked visible evidence",
            4 + len(candidates) + len(expanded),
        )
    transcript.append({"decision": {**final, "provenance": "model"}})
    record = make_record(
        row, args, brain, transcript, final,
        4 + len(candidates) + len(expanded),
    )
    record["answer_model"] = args.answer_model
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry-file", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--initial-model",
        help=("optional frozen initial-route model; when set, --model owns "
              "only post-scan temporal routing and the initial turn is "
              "excluded from the trainable policy trajectory"),
    )
    parser.add_argument(
        "--answer-model",
        help=("optional frozen model that independently produces the final "
              "category/severity/evidence after the trainable policy chooses "
              "post-scan ANSWER or TEMPORAL_EXPAND"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8020/v1")
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--post-scan-repair-attempts",
        type=int,
        default=0,
        help=("number of model-owned semantic-contract repair turns after an "
              "invalid post-scan ANSWER/TEMPORAL_EXPAND payload"),
    )
    parser.add_argument(
        "--normalize-temporal-bounds",
        action="store_true",
        help=("execute an otherwise valid model-selected TEMPORAL_EXPAND after "
              "audibly clamping only window/num_frames to tool schema bounds"),
    )
    parser.add_argument("--locator-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--locator-model",
        default="google/siglip2-base-patch16-224",
        help="local path or model identifier used by the semantic locator",
    )
    parser.add_argument("--locator-cache-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--initial-temperature",
        type=float,
        default=None,
        help=("optional temperature for the frozen initial routing turn; "
              "use 0 for identical-prefix counterfactual rollout groups"),
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--max-new-samples", type=int, default=0,
        help=("process at most this many not-yet-written rows in this process; "
              "used for bounded-memory resumable batches"),
    )
    parser.add_argument("--agent-version", default="causal_video_trajectory_v2")
    args = parser.parse_args()
    args.effective_policy_max_output_tokens = compact_policy_max_tokens(
        args.max_output_tokens,
    )

    rows = read_jsonl(args.split)
    if args.max_samples:
        rows = rows[:args.max_samples]
    cached = {row["id"] for row in read_jsonl(args.output)}
    pending = [(index, row) for index, row in enumerate(rows, 1) if row["id"] not in cached]
    if args.max_new_samples:
        if args.max_new_samples < 0:
            raise ValueError("max_new_samples must be non-negative")
        pending = pending[:args.max_new_samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.workers > 1:
        from transformers import AutoModel, AutoProcessor
        _ = (AutoModel, AutoProcessor)
    locator = JointRiskSemanticLocator(
        model_name=args.locator_model,
        device=args.locator_device,
        embedding_cache_dir=str(args.locator_cache_dir) if args.locator_cache_dir else None,
    )
    initial_cache = InitialObservationCache()
    initial_route_cache = SharedInitialRouteCache(expected_uses=4)

    def work(item):
        index, row = item
        try:
            return "success", index, run_one(
                row, args, locator, initial_cache, initial_route_cache,
            )
        except Exception as exc:
            return "failed", index, {
                "id": row["id"],
                "source_id": row.get("source_id"),
                "rollout_index": row.get("rollout_index"),
                "rollout_seed": row.get("rollout_seed"),
                "index": index,
                "model": args.model,
                "initial_model": args.initial_model,
                "answer_model": args.answer_model,
                "agent_version": args.agent_version,
                "error": f"{type(exc).__name__}: {exc}",
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(work, item) for item in pending]
        for future in concurrent.futures.as_completed(futures):
            status, index, record = future.result()
            target = args.output if status == "success" else args.retry_file
            if target:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if status == "success":
                print(
                    f"[{index}/{len(rows)}] {record['id']} pred={record['pred']} "
                    f"severity={record['severity']} tools={record['tool_calls']} "
                    f"frames={record['result']['total_frames']}", flush=True,
                )
            else:
                print(f"[{index}/{len(rows)}] {record['id']} ERROR {record['error']}", flush=True)


if __name__ == "__main__":
    main()
