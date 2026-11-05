#!/usr/bin/env python3
"""Evaluate one-video causal Agent trajectories through an OpenAI-compatible VLM.

Unlike the historical stage-wise evaluator, this runner keeps the initial
frames, every Assistant action, and every real tool result in one conversation.
It therefore matches the protocol used by ``sft_video_trajectory_v1``.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import math
import threading
from pathlib import Path

from openai import OpenAI

from agent.qwen_backend import CATEGORIES, canonical_category
from agent.risk_queries import risk_queries
from agent.tools import JointRiskSemanticLocator, TemporalEvidenceExpander
from agent.video import extract_frames, uniform_frames, video_duration
from build_sft_coldstart_dataset import (
    RISKS,
    final_prompt,
    initial_prompt,
    normalize_queries,
    review_prompt,
)
from evaluate_remote_joint_teacher_agent import (
    canonical_severity,
    evidence,
    merge_hits,
    normalize_assessments,
    normalize_dynamic_queries,
    normalize_response,
)


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def evenly_take(values: list, limit: int) -> list:
    if len(values) <= limit:
        return values
    return [values[min(int((index + 0.5) * len(values) / limit), len(values) - 1)]
            for index in range(limit)]


def image_url(image, max_pixels: int = 384 * 384) -> str:
    """Match the pixel-budget preprocessing used by the SFT collator."""
    image = image.convert("RGB")
    pixels = image.width * image.height
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
        image = image.resize((
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        ))
    buffer = io.BytesIO()
    # Local vLLM transport can afford PNG; avoid introducing JPEG artifacts
    # that were absent from the PIL images used during SFT.
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class CausalBrain:
    """A controller whose history grows monotonically within one video.

    ``model`` owns the initial routing decision.  When ``post_tool_model`` is
    provided, every later visual assessment/final-answer turn uses that model
    while retaining the exact same causal conversation.  This allows a router
    adapter to be optimized without overwriting the frozen post-tool visual
    reasoning policy.
    """

    def __init__(self, model: str, base_url: str, api_key: str, max_tokens: int,
                 temperature: float = 0.0, top_p: float = 1.0, seed: int = 42,
                 post_tool_model: str | None = None):
        # Construct a short-lived HTTP client for each turn. Loading SigLIP can
        # take long enough that some httpx/OpenAI client versions close an idle
        # connection pool before the post-tool turn.
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.post_tool_model = post_tool_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.messages: list[dict] = []
        self.rollout_turns: list[dict] = []
        self.usage: list[dict] = []
        self.raw_json_attempts = 0
        self.raw_json_successes = 0
        self.turn_attempt_counts: list[int] = []
        # This is intentionally separate from rollout_turns: an optional
        # shadow payload call is recorded for audit, but replaces (rather than
        # extends) the causal initial context and must not perturb later seeds.
        self.seed_turn_index = 0

    def model_for_next_turn(self) -> str:
        """Return the stage-specific model without mutating conversation state."""
        if not self.rollout_turns:
            return self.model
        return self.post_tool_model or self.model

    def seed_for_attempt(
        self,
        turn_attempt: int,
        seed_override: int | None = None,
    ) -> int:
        """Use a reproducible but distinct seed for each syntax retry.

        A failed JSON attempt does not advance ``seed_turn_index``. Reusing the
        exact same seed on every retry can therefore reproduce one pathological
        truncated/repetitive completion forever.  Offset only retry attempts;
        successful first-attempt generations keep the historical seed exactly.
        """
        if turn_attempt < 1:
            raise ValueError("turn_attempt must be at least 1")
        base_seed = (
            int(seed_override)
            if seed_override is not None
            else self.seed + self.seed_turn_index
        )
        return base_seed + turn_attempt - 1

    def _history_with_image_budget(self, current_image_count: int, limit: int = 16) -> list[dict]:
        """Return causal history that fits the serving model's image limit.

        vLLM counts images in *all* chat turns, rather than only the newly
        supplied user turn.  A normal trajectory can therefore reach exactly
        sixteen images (4 initial + 8 locator + 4 temporal) and a same-frame
        final-answer repair would otherwise make the next request contain
        twenty.  Keep every earlier textual observation/decision, but retain
        visual blocks newest-first only while they fit alongside the current
        evidence images.  This is transport compaction, not a change to the
        causal state saved in ``self.messages`` or to the audit trajectory.
        """
        if current_image_count > limit:
            raise ValueError(
                f"current turn has {current_image_count} images; serving limit is {limit}"
            )
        remaining = limit - current_image_count
        compacted_reversed: list[dict] = []
        for message in reversed(self.messages):
            content = message.get("content")
            if not isinstance(content, list):
                compacted_reversed.append(message)
                continue
            kept_reversed: list[dict] = []
            for block in reversed(content):
                if isinstance(block, dict) and block.get("type") == "image_url":
                    if remaining <= 0:
                        continue
                    remaining -= 1
                kept_reversed.append(block)
            compacted_reversed.append({**message, "content": list(reversed(kept_reversed))})
        return list(reversed(compacted_reversed))

    def call_turn(
        self,
        frames,
        prompt: str,
        *,
        model_override: str | None = None,
        policy_owned: bool = True,
        action_stage: str | None = None,
        seed_override: int | None = None,
        temperature_override: float | None = None,
    ) -> dict:
        """Generate one causal Assistant turn.

        ``model_override`` supports a stage-specialized frozen answer policy
        without forking or rewriting conversation history. ``policy_owned``
        is persisted for GRPO: helper-model outputs remain visible causal
        context but must never receive the trainable policy's log-probability.
        """
        turn_model = model_override or self.model_for_next_turn()
        turn_temperature = (
            self.temperature
            if temperature_override is None
            else float(temperature_override)
        )
        last_error = None
        for turn_attempt, side in enumerate((384, 320, 256), 1):
            generation_seed = self.seed_for_attempt(turn_attempt, seed_override)
            content = [{"type": "text", "text": prompt}]
            content.extend({
                "type": "image_url",
                "image_url": {"url": image_url(frame.image, side * side)},
            } for frame in frames)
            user_message = {"role": "user", "content": content}
            request_messages = self._history_with_image_budget(len(frames)) + [user_message]
            self.raw_json_attempts += 1
            try:
                with OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=240) as client:
                    response = client.chat.completions.create(
                        model=turn_model,
                        messages=request_messages,
                        temperature=turn_temperature,
                        top_p=self.top_p,
                        seed=generation_seed,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"},
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                raw = response.choices[0].message.content
                parsed = normalize_response(json.loads(raw))
                self.messages.extend([user_message, {"role": "assistant", "content": raw}])
                rollout_turn = {
                    "model": turn_model,
                    "user_prompt": prompt,
                    "frame_timestamps": [float(frame.timestamp) for frame in frames],
                    "assistant_raw": raw,
                    "assistant_parsed": parsed,
                    "policy_owned": bool(policy_owned),
                    "generation_seed": generation_seed,
                    "generation_temperature": turn_temperature,
                    "generation_attempt": turn_attempt,
                }
                if action_stage:
                    rollout_turn["action_stage"] = action_stage
                self.rollout_turns.append(rollout_turn)
                self.raw_json_successes += 1
                self.turn_attempt_counts.append(turn_attempt)
                self.seed_turn_index += 1
                if response.usage:
                    self.usage.append(response.usage.model_dump())
                return parsed
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"causal policy failed after 3 attempts: {last_error}")

    def record_shadow_call(
        self,
        shadow: "CausalBrain",
        *,
        turn_kind: str,
        adopt_as_context: bool,
    ) -> None:
        """Record a second policy call and optionally adopt its causal context.

        The GRPO adapter may own only the discrete route while the frozen SFT
        policy supplies scan parameters.  Both calls remain visible in the
        audit trail; when the SFT payload is valid, later turns continue from
        the SFT conversation exactly as they did in the frozen baseline.
        """
        for item in shadow.rollout_turns:
            copied = dict(item)
            copied["turn_kind"] = turn_kind
            copied["adopted_as_context"] = adopt_as_context
            self.rollout_turns.append(copied)
        self.usage.extend(shadow.usage)
        self.raw_json_attempts += shadow.raw_json_attempts
        self.raw_json_successes += shadow.raw_json_successes
        self.turn_attempt_counts.extend(shadow.turn_attempt_counts)
        if adopt_as_context:
            self.messages = list(shadow.messages)
            self.seed_turn_index = shadow.seed_turn_index


def valid_final(data: dict, visible_times: list[float], *, include_action: bool = True) -> dict | None:
    if not isinstance(data, dict):
        return None
    category = canonical_category(data.get("category"))
    severity = canonical_severity(data.get("severity"), category)
    if category not in CATEGORIES or severity not in {"L0", "L1", "L2"}:
        return None
    grounded = evidence(data, visible_times)
    if category != "Harmless" and not grounded:
        return None
    answer = dict(data)
    if include_action:
        answer["action"] = "ANSWER"
    answer["category"] = category
    answer["severity"] = severity
    answer["evidence"] = grounded
    return answer


def valid_initial_direct_harm(answer: dict | None) -> bool:
    """Only unambiguous L2 harm may bypass the evidence-acquisition loop."""
    if not answer or answer.get("category") == "Harmless":
        return False
    return (
        answer.get("severity") == "L2"
        and str(answer.get("evidence_strength") or "").lower()
        in {"clear", "strong", "unambiguous"}
        and bool(answer.get("evidence"))
    )


def valid_initial_action(data: dict, visible_times: list[float]) -> bool:
    """Validate the complete first-turn schema, including scan parameters."""
    if not isinstance(data, dict):
        return False
    action = str(data.get("action") or "").upper()
    if action == "ANSWER":
        return valid_final(data, visible_times) is not None
    if action == "JOINT_SEMANTIC_SCAN":
        return bool(normalize_queries(data.get("queries_by_risk")))
    return False


def run_one(row: dict, args, locator: JointRiskSemanticLocator) -> dict:
    path = ROOT / row["local_path"]
    duration = video_duration(str(path))
    initial = uniform_frames(str(path), 4)
    if len(initial) != 4:
        raise RuntimeError(f"decoded {len(initial)}/4 initial frames")
    brain = CausalBrain(
        args.model,
        args.base_url,
        args.api_key,
        args.max_output_tokens,
        args.temperature,
        args.top_p,
        int(row.get("rollout_seed", args.seed)),
        args.post_tool_model,
    )
    initial_times = [frame.timestamp for frame in initial]
    transcript = [{"observation": {
        "duration": duration,
        "frames_seen": 4,
        "timestamps": initial_times,
        "initial_sampling": "uniform_4_non_black",
    }}]
    first = brain.call_turn(initial, initial_prompt(initial_times))
    transcript.append({"decision": first})
    first_action = str(first.get("action") or "").upper()
    initial_answer = valid_final(first, initial_times)
    if first_action == "ANSWER" and (
        valid_initial_direct_harm(initial_answer)
        or (args.disable_initial_safety_guard and initial_answer is not None)
    ):
        return make_record(row, args, brain, transcript, initial_answer, 4)

    automatic = None
    initial_scan_valid = bool(
        first_action == "JOINT_SEMANTIC_SCAN"
        and normalize_queries(first.get("queries_by_risk"))
    )
    if not initial_scan_valid:
        automatic = {
            "action": "JOINT_SEMANTIC_SCAN",
            "automatic": True,
            "reason": (
                "initial scan parameters were invalid and required fixed-query fallback"
                if first_action == "JOINT_SEMANTIC_SCAN"
                else "initial answer was Harmless, L1/ambiguous, or lacked clear grounded L2 evidence"
            ),
        }
        transcript.append({"decision": automatic})

    query_decision = first
    # Every row reaching this point will execute the joint scan.  Let the GRPO
    # adapter own only that route choice; use the frozen SFT initial turn as
    # both the query-parameter source and the causal context for all later SFT
    # turns.  This also covers an unsafe/insufficient initial ANSWER rescued by
    # the framework, avoiding hidden GRPO-text influence after the route.
    if args.initial_payload_model:
        payload_brain = CausalBrain(
            args.initial_payload_model,
            args.base_url,
            args.api_key,
            args.max_output_tokens,
            args.temperature,
            args.top_p,
            int(row.get("rollout_seed", args.seed)),
        )
        payload = payload_brain.call_turn(initial, initial_prompt(initial_times))
        payload_is_scan = str(payload.get("action") or "").upper() == "JOINT_SEMANTIC_SCAN"
        brain.record_shadow_call(
            payload_brain,
            turn_kind="initial_scan_payload",
            adopt_as_context=True,
        )
        transcript.append({
            "planner_decision": payload,
            "planner_model": args.initial_payload_model,
            "planner_proposed_scan": payload_is_scan,
            "adopted_as_context": True,
        })
        query_decision = payload

    dynamic = normalize_dynamic_queries(
        query_decision.get("queries_by_risk")
        or query_decision.get("queries")
        or query_decision.get("query")
        or [],
        canonical_category(query_decision.get("category")),
    )
    normalized = normalize_queries(query_decision.get("queries_by_risk"))
    if normalized:
        dynamic = normalized
    queries_by_risk = {risk: risk_queries(risk, dynamic.get(risk, [])) for risk in RISKS}
    hits = locator(str(path), queries_by_risk, top_k_per_risk=args.top_k)
    merged = evenly_take(merge_hits(hits), 8)
    if not merged:
        raise RuntimeError("joint semantic scan returned no candidate frames")
    requested_candidate_times = [float(item["timestamp"]) for item in merged]
    candidates = extract_frames(str(path), requested_candidate_times, deduplicate=False)
    if not candidates:
        raise RuntimeError(
            f"decoded 0/{len(requested_candidate_times)} candidate frames"
        )
    # A damaged GOP near one requested timestamp must not discard all other
    # valid visual evidence from the video.  Keep the frames that actually
    # decoded and align the locator metadata to their real timestamps.
    candidate_times = [float(frame.timestamp) for frame in candidates]
    if len(candidates) != len(requested_candidate_times):
        merged = [
            {
                **min(
                    merged,
                    key=lambda item: abs(float(item["timestamp"]) - timestamp),
                ),
                "timestamp": timestamp,
            }
            for timestamp in candidate_times
        ]
    transcript.append({
        "tool": "JOINT_SEMANTIC_SCAN",
        "queries_by_risk": queries_by_risk,
        "observation": merged,
    })
    scan_state = {
        "executed_tool_call": {
            "tool": "JOINT_SEMANTIC_SCAN",
            "queries_by_risk": queries_by_risk,
            "automatic_framework_transition": automatic,
        },
        "actual_tool_result": {
            "tool": "JOINT_SEMANTIC_SCAN",
            "observation": merged,
            "returned_frame_timestamps": candidate_times,
        },
    }
    visible_review = initial_times + candidate_times
    assessment = brain.call_turn(
        candidates,
        "此前调用及真实工具结果如下。query和分数仅用于定位，不是风险证据。\n"
        + json_text(scan_state) + "\n\n" + review_prompt(visible_review),
    )
    transcript.append({"decision": {"action": "ASSESS_RISKS", **assessment}})

    raw_final = assessment.get("final_answer")
    if isinstance(raw_final, list):
        raw_final = next((item for item in raw_final if isinstance(item, dict)), None)
    scan_final = valid_final(raw_final, visible_review) if isinstance(raw_final, dict) else None
    if scan_final:
        return make_record(row, args, brain, transcript, scan_final, 4 + len(candidates))

    assessments = normalize_assessments(assessment.get("assessments") or {})
    statuses = {risk: str((assessments.get(risk) or {}).get("status") or "").upper()
                for risk in RISKS}
    if len(assessments) == 3 and all(statuses[risk] == "ABSENT" for risk in RISKS):
        final = {
            "action": "ANSWER",
            "category": "Harmless",
            "severity": "L0",
            "rationale": "联合候选视觉核验未发现三类风险证据。",
            "evidence": [],
        }
        transcript.append({"decision": {**final, "automatic": True}})
        return make_record(row, args, brain, transcript, final, 4 + len(candidates))

    if args.disable_temporal_expand:
        # Ablation: force a terminal decision from exactly the initial and
        # semantic-retrieval evidence.  The model may not request or observe
        # neighbouring temporal context.
        no_expand_data = brain.call_turn(
            candidates,
            "消融实验已禁用TEMPORAL_EXPAND。不得请求更多帧；请仅依据已经观察到的初始帧和"
            "语义候选帧给出最终答案。\n\n" + final_prompt(visible_review),
            action_stage="no_temporal_expand_final",
        )
        final = valid_final(no_expand_data, visible_review)
        if not final:
            repair_data = brain.call_turn(
                candidates,
                "上一份JSON未通过证据格式校验。TEMPORAL_EXPAND仍被禁用；有害结论必须引用"
                "已观察时间戳。\n"
                + json_text({"visible_timestamps": visible_review}) + "\n\n"
                + final_prompt(visible_review),
                action_stage="no_temporal_expand_repair",
            )
            final = valid_final(repair_data, visible_review)
        if not final:
            raise RuntimeError("no-temporal ablation answer is invalid or lacks visible evidence")
        transcript.append({"decision": {**final, "ablation": "no_temporal_expand"}})
        return make_record(row, args, brain, transcript, final, 4 + len(candidates))

    insufficient = [risk for risk in RISKS if statuses.get(risk) in {"INSUFFICIENT", "PRESENT"}]
    if not insufficient:
        insufficient = list(RISKS)
    expand_times = []
    for risk in insufficient:
        requested = (assessments.get(risk) or {}).get("candidate_timestamps") or []
        valid = [float(value) for value in requested
                 if any(abs(float(value) - known) <= 0.3 for known in candidate_times)]
        if not valid:
            valid = [hit.timestamp for hit in hits if hit.risk_type == risk][:2]
        expand_times.extend(valid[:2])
    expand_times = list(dict.fromkeys(expand_times)) or candidate_times[:2]
    expand_decision = {
        "action": "TEMPORAL_EXPAND",
        "risk_types": insufficient,
        "timestamps": expand_times,
        "window": 4.0,
        "num_frames": 4,
    }
    transcript.append({"decision": expand_decision})
    expanded = TemporalEvidenceExpander()(str(path), expand_times, window=4.0, num_frames=4)
    if not expanded:
        raise RuntimeError("temporal expansion returned no frames")
    expanded_times = [frame.timestamp for frame in expanded]
    transcript.append({
        "tool": "TEMPORAL_EXPAND",
        "risk_types": insufficient,
        "observation": {"timestamps": expanded_times, "added_frames": len(expanded)},
    })
    temporal_state = {
        "executed_tool_call": {"tool": "TEMPORAL_EXPAND", "value": expand_decision},
        "actual_tool_result": {
            "tool": "TEMPORAL_EXPAND",
            "observation": {"timestamps": expanded_times, "added_frames": len(expanded)},
        },
    }
    visible_final = visible_review + expanded_times
    final_data = brain.call_turn(
        expanded,
        "此前调用及真实工具结果如下。只依据已经返回的图片完成判断。\n"
        + json_text(temporal_state) + "\n\n" + final_prompt(visible_final),
    )
    final = valid_final(final_data, visible_final)
    if not final:
        # A model may return syntactically valid JSON while omitting a visible
        # timestamp for a harmful conclusion.  This is a protocol failure, not
        # a reason to discard otherwise decoded evidence or to fabricate a
        # label.  Give it one causal, same-frame repair turn.  The preceding
        # answer remains in history, no gold label is provided, and the normal
        # validator still rejects unsupported conclusions.
        repair_data = brain.call_turn(
            expanded,
            "你的上一份最终 JSON 未通过证据格式校验。请仅依据本轮已返回的图片重新输出最终答案。"
            "若结论不是 Harmless，evidence 必须引用下列已观察时间戳中的至少一个，且不要编造时间。\n"
            + json_text({"visible_timestamps": visible_final}) + "\n\n"
            + final_prompt(visible_final),
        )
        transcript.append({"decision": {"action": "FINAL_EVIDENCE_REPAIR", **repair_data}})
        final = valid_final(repair_data, visible_final)
    if not final:
        raise RuntimeError("final answer is invalid or lacks visible evidence")
    transcript.append({"decision": final})
    return make_record(row, args, brain, transcript, final, 4 + len(candidates) + len(expanded))


def make_record(row: dict, args, brain: CausalBrain, transcript: list[dict],
                final: dict, total_frames: int) -> dict:
    first = next((step["decision"] for step in transcript
                  if isinstance(step.get("decision"), dict)
                  and not step["decision"].get("automatic")), {})
    first_answer = valid_final(first, transcript[0]["observation"]["timestamps"])
    first_action = str(first.get("action") or "").upper() or None
    initial_valid_action = valid_initial_action(
        first, transcript[0]["observation"]["timestamps"],
    )
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
        "post_tool_model": getattr(args, "post_tool_model", None) or args.model,
        "initial_payload_model": getattr(args, "initial_payload_model", None),
        "agent_version": getattr(args, "agent_version", "causal_video_trajectory_v1"),
        "requested_max_output_tokens": getattr(args, "max_output_tokens", None),
        "effective_policy_max_output_tokens": getattr(
            args, "effective_policy_max_output_tokens",
            getattr(args, "max_output_tokens", None),
        ),
        "initial_action": first_action,
        "initial_valid_json_action": initial_valid_action,
        "initial_pred": first_answer.get("category") if first_answer else None,
        "initial_severity": first_answer.get("severity") if first_answer else None,
        "pred": final["category"],
        "severity": final["severity"],
        "tool_calls": [step["tool"] for step in transcript if step.get("tool")],
        "result": {"final_action": final, "transcript": transcript, "total_frames": total_frames},
        "model_turns": brain.rollout_turns,
        "usage_by_round": brain.usage,
        "raw_json_attempts": brain.raw_json_attempts,
        "raw_json_successes": brain.raw_json_successes,
        "turn_attempt_counts": brain.turn_attempt_counts,
        "raw_first_attempt_json_rate": (
            sum(value == 1 for value in brain.turn_attempt_counts)
            / len(brain.turn_attempt_counts)
            if brain.turn_attempt_counts else 0.0
        ),
        "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry-file", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--post-tool-model",
        help=("optional model used after the initial routing turn; defaults "
              "to --model for the historical single-policy protocol"),
    )
    parser.add_argument(
        "--initial-payload-model",
        help=("optional frozen model that supplies JOINT_SEMANTIC_SCAN query "
              "parameters after --model selects the scan route"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8015/v1")
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--locator-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--locator-cache-dir", type=Path,
        help=(
            "Optional persistent cache for SigLIP candidate embeddings. This only "
            "avoids recomputing deterministic video features across resumed or "
            "multi-checkpoint evaluations; it does not change retrieval inputs."
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent-version", default="causal_video_trajectory_v1")
    parser.add_argument(
        "--disable-temporal-expand", action="store_true",
        help=("Ablation: after semantic retrieval, force a final answer without "
              "observing neighbouring temporal frames."),
    )
    parser.add_argument(
        "--disable-initial-safety-guard", action="store_true",
        help=("Ablation: accept any valid initial ANSWER, including Harmless or "
              "L1, instead of forcing joint risk evidence acquisition."),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--daa-group", choices=("Direct", "Adaptive", "Active"))
    args = parser.parse_args()
    rows = read_jsonl(args.split)
    if args.category:
        rows = [row for row in rows if row.get("category") == args.category]
    if args.daa_group:
        rows = [row for row in rows if row.get("daa_group") == args.daa_group]
    if args.max_samples:
        rows = rows[:args.max_samples]
    cached = {row["id"] for row in read_jsonl(args.output)}
    pending = [(index, row) for index, row in enumerate(rows, 1) if row["id"] not in cached]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Transformers exposes Auto classes through a lazy module.  Resolving
    # them concurrently in several first-use locator threads can race and
    # intermittently raise ``cannot import name AutoModel``.  Warm the lazy
    # exports once in the main thread before the worker pool starts.
    if args.workers > 1:
        from transformers import AutoModel, AutoProcessor
        _ = (AutoModel, AutoProcessor)
    # ``JointRiskSemanticLocator`` is deliberately thread-safe: it serializes
    # only SigLIP inference and shares in-flight/disk candidate embeddings.
    # A locator per worker defeats that cache.  In particular, four GRPO
    # siblings can otherwise decode the same long source into hundreds of
    # full-resolution PIL images independently, causing unbounded host-memory
    # growth before any result is written.  One process-wide locator lets
    # sibling rollouts await one candidate-embedding Future instead.
    locator = JointRiskSemanticLocator(
        device=args.locator_device,
        embedding_cache_dir=(str(args.locator_cache_dir) if args.locator_cache_dir else None),
    )

    def work(item):
        index, row = item
        try:
            return "success", index, run_one(row, args, locator)
        except Exception as exc:
            return "failed", index, {
                "id": row["id"], "index": index,
                "error": f"{type(exc).__name__}: {exc}",
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Persist each completed video immediately.  Ordered ``executor.map``
        # can hold many finished videos behind one very long decode; a later
        # interruption would then needlessly recompute those hidden results.
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
                    f"frames={record['result']['total_frames']}",
                    flush=True,
                )
            else:
                print(f"[{index}/{len(rows)}] {record['id']} ERROR {record['error']}", flush=True)


if __name__ == "__main__":
    main()
