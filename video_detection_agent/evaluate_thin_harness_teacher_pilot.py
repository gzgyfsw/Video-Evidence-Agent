#!/usr/bin/env python3
"""Run a remote visual Teacher in an autonomous, bounded tool loop.

This runner records raw proposals and Harness rejections so
that autonomy can be measured independently of the final disposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import time
from pathlib import Path

from agent.qwen_backend import CATEGORIES, SAFETY_POLICY, POLICY_VERSION, canonical_category
from agent.tools import JointRiskSemanticLocator, TemporalEvidenceExpander
from agent.video import extract_frames, uniform_frames, video_duration
from evaluate_dataset import read_jsonl, video_path
from evaluate_remote_joint_teacher_agent import (
    RISKS, RemoteBrain, canonical_severity, merge_hits,
    normalize_assessments, normalize_dynamic_queries, normalize_response,
)


class LocalBrain:
    """Transformers fallback for local Teacher models when vLLM cannot serve them."""

    def __init__(self, model_path, max_output_tokens=4096):
        import torch
        from json_repair import repair_json
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.repair_json = repair_json
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, low_cpu_mem_usage=True,
        ).eval()
        self.max_output_tokens = max_output_tokens

    def call(self, frames, prompt):
        messages = [{"role": "user", "content": [
            *[{"type": "image", "image": frame.image} for frame in frames],
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.processor(
            text=[text], images=[frame.image for frame in frames],
            padding=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_output_tokens, do_sample=False,
            )
        output = self.processor.decode(
            generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if not match:
            raise ValueError(f"local policy returned invalid JSON: {output[:500]!r}")
        raw_json = match.group(0)
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            parsed = json.loads(self.repair_json(raw_json))
        return normalize_response(parsed)


def _visible_frames(initial, acquired, limit=12):
    """Prefer fresh evidence, but keep the original sparse global context."""
    ordered = list({round(frame.timestamp, 6): frame for frame in initial + acquired}.values())
    if len(ordered) <= limit:
        return sorted(ordered, key=lambda frame: frame.timestamp)
    keep = initial[:4] + acquired[-(limit - min(4, len(initial))):]
    return sorted({round(frame.timestamp, 6): frame for frame in keep}.values(),
                  key=lambda frame: frame.timestamp)


def _coverage(raw):
    mapped = normalize_assessments(raw)
    return {risk: str((mapped.get(risk) or {}).get("status", "NOT_ASSESSED")).upper()
            for risk in RISKS}


def _infer_missing_answer_fields(proposal):
    """Bridge old category/severity answers into the newer action contract."""
    if not isinstance(proposal, dict) or str(proposal.get("action", "")).upper() != "ANSWER":
        return []
    normalized = []
    category = canonical_category(proposal.get("category"))
    severity = canonical_severity(proposal.get("severity"), category)
    if not proposal.get("moderation_action"):
        if category == "Harmless" and severity == "L0":
            proposal["moderation_action"] = "ALLOW"
            normalized.append("NORMALIZED_MISSING_MODERATION_ACTION")
        elif category in RISKS and severity == "L1":
            proposal["moderation_action"] = "REVIEW"
            normalized.append("NORMALIZED_MISSING_MODERATION_ACTION")
        elif category in RISKS and severity == "L2":
            proposal["moderation_action"] = "BLOCK"
            normalized.append("NORMALIZED_MISSING_MODERATION_ACTION")
    if str(proposal.get("moderation_action", "")).upper() == "REVIEW" and not proposal.get("review_reason"):
        if category in RISKS and severity == "L1":
            proposal["review_reason"] = "CONFIRMED_L1"
            normalized.append("NORMALIZED_MISSING_REVIEW_REASON")
        else:
            proposal["review_reason"] = "EVIDENCE_INSUFFICIENT"
            normalized.append("NORMALIZED_MISSING_REVIEW_REASON")
    return normalized


def _thin_action_schema():
    risk_coverage_item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {
                "type": "string",
                "enum": ["PRESENT", "ABSENT", "INSUFFICIENT", "NOT_ASSESSED"],
            },
            "reason": {"type": "string"},
        },
        "required": ["status", "reason"],
    }
    return {
        "name": "thin_harness_action",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["ANSWER", "JOINT_SEMANTIC_SCAN", "TEMPORAL_EXPAND"],
                },
                "moderation_action": {
                    "type": "string",
                    "enum": ["ALLOW", "REVIEW", "BLOCK"],
                },
                "category": {
                    "type": ["string", "null"],
                    "enum": [*CATEGORIES, None],
                },
                "severity": {
                    "type": ["string", "null"],
                    "enum": ["L0", "L1", "L2", None],
                },
                "review_reason": {
                    "type": ["string", "null"],
                    "enum": ["CONFIRMED_L1", "EVIDENCE_INSUFFICIENT", None],
                },
                "rationale": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "timestamp": {"type": "number"},
                            "description": {"type": "string"},
                        },
                        "required": ["timestamp", "description"],
                    },
                },
                "risk_coverage": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {risk: risk_coverage_item for risk in RISKS},
                },
                "queries_by_risk": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        risk: {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 4,
                        }
                        for risk in RISKS
                    },
                },
                "timestamps": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 1,
                    "maxItems": 3,
                },
                "window": {"type": "number", "minimum": 0.5, "maximum": 15},
                "num_frames": {"type": "integer", "minimum": 1, "maximum": 8},
                "evidence_strength": {"type": ["string", "null"]},
                "evidence_basis": {"type": ["string", "null"]},
                "identified_figure": {"type": ["string", "null"]},
                "notes": {"type": ["string", "null"]},
            },
            "required": ["action"],
        },
    }


def _strict_evidence(proposal, visible_times):
    """Resolve citations only to frames actually sent in this decision."""
    raw = proposal.get("evidence") or []
    if not isinstance(raw, list):
        return [], "evidence must be a list"
    grounded = []
    for item in raw:
        rounded_tolerance = 0.0006
        if isinstance(item, str):
            # Some JSON-only backends serialize a citation as a timestamped
            # sentence.  This is a mechanical syntax conversion, not a new
            # visual claim: the raw proposal remains in the transcript and
            # the timestamp must still resolve to an actually displayed frame.
            match = re.fullmatch(
                r"\s*(?:Frame\s+(?:at\s+|\d+\s*\(\s*)?)?"
                r"(\d+(?:\.(\d+))?)\s*s\s*\)?\s*:\s*(.+?)\s*",
                item, flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                item = {"timestamp": float(match.group(1)),
                        "description": match.group(3).strip()}
                if match.group(2):
                    rounded_tolerance = min(0.0051, max(0.0006,
                                                        0.5 * 10 ** -len(match.group(2)) + 0.0001))
        if isinstance(item, dict) and not str(item.get("description", "")).strip():
            # Equivalent visible-fact fields used by some local-model outputs.
            # Keep the raw model field in the transcript; only the accepted
            # citation uses the canonical description key.
            for alias in ("observation", "detail", "fact"):
                if str(item.get(alias, "")).strip():
                    item = {**item, "description": item[alias]}
                    break
        if not isinstance(item, dict) or not str(item.get("description", "")).strip():
            return [], "evidence item requires a visible-fact description"
        try:
            timestamp_raw = item["timestamp"]
            timestamp = float(timestamp_raw)
        except (TypeError, ValueError, KeyError):
            return [], "evidence timestamp is invalid"
        if rounded_tolerance <= 0.0006:
            text = str(timestamp_raw)
            if "." in text:
                decimals = len(text.split(".", 1)[1].rstrip("0"))
                if decimals:
                    rounded_tolerance = min(
                        0.051,
                        max(0.0006, 0.5 * 10 ** -decimals + 0.0001),
                    )
        if not visible_times:
            return [], "evidence timestamp has no displayed frame"
        nearest = min(visible_times, key=lambda known: abs(known - timestamp))
        if abs(nearest - timestamp) > rounded_tolerance:
            return [], "evidence timestamp was not displayed in this decision"
        if nearest not in [e["timestamp"] for e in grounded]:
            grounded.append({"timestamp": nearest, "description": item["description"]})
    return grounded, None


def _append_unique_frames(acquired, fresh, initial, remaining):
    known = {round(frame.timestamp, 3) for frame in initial + acquired}
    added = []
    for frame in fresh:
        key = round(frame.timestamp, 3)
        if key not in known and len(added) < remaining:
            known.add(key)
            added.append(frame)
    acquired.extend(added)
    return added


def _history_summary(transcript):
    """Keep a bounded, explicitly non-visual summary of prior actions."""
    tools = [entry for entry in transcript if "tool" in entry]
    decisions = [entry["raw_decision"] for entry in transcript if "raw_decision" in entry]
    return {
        "tool_call_count": len(tools),
        "prior_tools": [
            {"tool": entry["tool"],
             "risk_types": list((entry.get("queries_by_risk") or {}).keys()),
             "queries": {risk: items[:2] for risk, items in
                         (entry.get("queries_by_risk") or {}).items()},
             "anchors": entry.get("anchors", []),
             "added_timestamps": [round(t, 3) for t in entry.get("added_timestamps", [])]}
            for entry in tools[-8:]
        ],
        "earlier_tool_count": max(len(tools) - 8, 0),
        "prior_model_notes": [str(item.get("rationale", ""))[:160]
                              for item in decisions[-4:] if item.get("rationale")],
    }


def _transient_api_error(message):
    lowered = message.casefold()
    if "sampletimeouterror" in lowered or "sample exceeded" in lowered:
        return False
    return any(token in lowered for token in (
        "connection error", "apiconnectionerror", "timed out", "timeout",
        "rate limit", "error code: 429", "error code: 500", "error code: 502",
        "error code: 503", "error code: 504",
    ))


def _completed_ids(output, retry_file, *, skip_previous_errors=False,
                   retry_incomplete_output=False):
    records = read_jsonl(output) if output.exists() else []
    if retry_incomplete_output:
        done = {row["id"] for row in records
                if row.get("trajectory_status") == "ANSWERED" and
                not row.get("error") and
                ((row.get("result") or {}).get("final_action") or {}).get("action") == "ANSWER"}
    else:
        done = {row["id"] for row in records}
    if skip_previous_errors and retry_file.exists():
        done.update(row["id"] for row in read_jsonl(retry_file)
                    if not _transient_api_error(str(row.get("error", ""))) and
                    (not retry_incomplete_output or
                     "no visible initial frames" not in str(row.get("error", "")).casefold()))
    return done


class SampleTimeoutError(TimeoutError):
    pass


class TrajectoryRunError(RuntimeError):
    def __init__(self, message, transcript):
        super().__init__(message)
        self.transcript = transcript


class sample_timeout:
    def __init__(self, seconds: int | None):
        self.seconds = int(seconds or 0)
        self.previous_handler = None

    def __enter__(self):
        if self.seconds <= 0:
            return self
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._raise_timeout)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.previous_handler)
        return False

    def _raise_timeout(self, signum, frame):
        raise SampleTimeoutError(f"sample exceeded {self.seconds}s")


def run_one(path, brain, locator, *, max_decisions=5, max_tools=3,
            max_total_frames=24, top_k=2, max_rejections=2,
            max_visible_frames=12, audit_only_safety=False):
    if max_decisions < 1 or max_tools < 0 or max_total_frames < 4:
        raise ValueError("invalid decision, tool, or total-frame budget")
    if not 4 <= max_visible_frames <= 12:
        raise ValueError("max_visible_frames must be 4-12 for this visual backend")
    duration = video_duration(path)
    initial = uniform_frames(path, 4)
    if not initial:
        raise RuntimeError("no visible initial frames")
    acquired = []
    transcript = [{"observation": {"duration": duration,
                                    "initial_timestamps": [f.timestamp for f in initial]}}]
    scan_signatures = set()
    expanded_signatures = set()
    tool_calls = 0
    rejections = 0
    all_hits = []
    audit_flags = []

    for turn in range(1, max_decisions + 1):
        visible = _visible_frames(initial, acquired, limit=max_visible_frames)
        visible_times = [frame.timestamp for frame in visible]
        frame_remaining = max_total_frames - len(initial) - len(acquired)
        prompt = f"""你是视频安全审核的自主Teacher。以下图片都来自同一个视频，按时间顺序排列。
每轮只选择一个动作：
1. ANSWER：输出最终category、severity、moderation_action、review_reason、rationale、evidence、risk_coverage。
2. JOINT_SEMANTIC_SCAN：输出queries_by_risk，按需选择风险类别，每类1-4条具体可见的英文query；工具会在全视频搜索并返回真实候选帧。
3. TEMPORAL_EXPAND：输出timestamps、window、num_frames；用于查看已见疑点附近的动作或语境。
你自主决定是否继续取证与何时停止。SigLIP的query和分数不是风险证据；不能引用未实际显示的帧。
历史摘要只是此前调用和模型笔记，不能替代本轮图片作证；all_acquired_timestamps也不代表本轮全部图片都可见。
REVIEW必须区分review_reason=CONFIRMED_L1（轻度风险已确认）与EVIDENCE_INSUFFICIENT（取证后仍无法确定）。
若当前证据不足，优先考虑能否用一次工具解决；无效重复检索会被拒绝。所有已见证据来自图片，不保证覆盖整个视频。
特别注意：初始4帧没有看见风险，不等于全视频三个风险均不存在。要输出Harmless/ALLOW，你必须有足够的全视频排查依据；对可能包含未采到事件的普通或长视频，主动使用JOINT_SEMANTIC_SCAN检索未排除的风险，并按需用TEMPORAL_EXPAND验证。只有确实能从现有画面判断内容始终相同、无需再取证时，才可直接ALLOW。不要为了省工具而把NOT_ASSESSED填成ABSENT；也不要为了避免工具调用而滥用REVIEW。
{SAFETY_POLICY}
输出一个JSON对象，字段action，以及动作所需字段。ANSWER还须输出risk_coverage，三个键为{RISKS}，每项status只能是PRESENT、ABSENT、INSUFFICIENT、NOT_ASSESSED，并说明reason。
类别只能是{list(CATEGORIES)}。Harmless=L0/ALLOW；有害L1/REVIEW；有害L2/BLOCK；证据不足的REVIEW可不填category和severity。
当前状态：{json.dumps({"turn": turn, "duration": round(duration, 3), "visible_timestamps": [round(x, 3) for x in visible_times], "all_acquired_timestamps": [round(f.timestamp, 3) for f in initial + acquired], "tools_remaining": max_tools - tool_calls, "frames_remaining": frame_remaining, "recent_tool_observations": transcript[-3:], "history_summary": _history_summary(transcript)}, ensure_ascii=False)}"""
        try:
            proposal = brain.call(visible, prompt)
        except Exception as exc:
            transcript.append({"turn": turn, "visible_timestamps": visible_times,
                               "backend_error": f"{type(exc).__name__}: {exc}",
                               "raw_response": getattr(brain, "last_raw_response", None)})
            raise TrajectoryRunError(str(exc), transcript) from exc
        transcript.append({"turn": turn, "raw_decision": proposal,
                           "visible_timestamps": visible_times})
        action = str(proposal.get("action", "")).upper() if isinstance(proposal, dict) else ""
        rejection = None

        if action == "ANSWER":
            normalization_flags = _infer_missing_answer_fields(proposal)
            disposition = str(proposal.get("moderation_action", "")).upper()
            category = canonical_category(proposal.get("category"))
            severity = canonical_severity(proposal.get("severity"), category)
            grounded, evidence_error = _strict_evidence(proposal, visible_times)
            coverage = _coverage(proposal.get("risk_coverage"))
            review_reason = str(proposal.get("review_reason", "")).upper()
            if evidence_error:
                rejection = evidence_error
            elif disposition not in {"ALLOW", "REVIEW", "BLOCK"}:
                rejection = "invalid moderation_action"
            elif disposition == "ALLOW" and (category != "Harmless" or severity != "L0"):
                rejection = "ALLOW requires Harmless-L0"
            elif disposition == "ALLOW" and not audit_only_safety and not all(value == "ABSENT" for value in coverage.values()):
                rejection = "ALLOW requires explicit exclusion assessments for all three risks"
            elif disposition == "BLOCK" and (category not in RISKS or severity != "L2" or (not audit_only_safety and not grounded)):
                rejection = "BLOCK requires harmful L2 category and grounded evidence"
            elif disposition == "REVIEW" and review_reason not in {"CONFIRMED_L1", "EVIDENCE_INSUFFICIENT"}:
                rejection = "REVIEW requires a valid review_reason"
            elif disposition == "REVIEW" and review_reason == "CONFIRMED_L1" and (
                category not in RISKS or severity != "L1" or (not audit_only_safety and not grounded)
            ):
                rejection = "CONFIRMED_L1 requires grounded harmful L1 evidence"
            elif disposition == "REVIEW" and review_reason == "EVIDENCE_INSUFFICIENT" and severity is not None:
                rejection = "EVIDENCE_INSUFFICIENT cannot claim a confirmed severity"
            else:
                if disposition == "ALLOW" and not all(value == "ABSENT" for value in coverage.values()):
                    audit_flags.append("ALLOW_WITH_INCOMPLETE_RISK_EXCLUSION")
                if disposition == "ALLOW" and tool_calls == 0:
                    audit_flags.append("ALLOW_WITHOUT_TOOL")
                if disposition == "BLOCK" and not grounded:
                    audit_flags.append("BLOCK_WITHOUT_VISIBLE_EVIDENCE")
                if review_reason == "CONFIRMED_L1" and not grounded:
                    audit_flags.append("CONFIRMED_L1_WITHOUT_VISIBLE_EVIDENCE")
                audit_flags.extend(flag for flag in normalization_flags if flag not in audit_flags)
                final = {"action": "ANSWER", "moderation_action": disposition,
                         "category": category if category in CATEGORIES else None,
                         "severity": severity if severity in {"L0", "L1", "L2"} else None,
                         "review_reason": review_reason if disposition == "REVIEW" else None,
                         "risk_coverage": coverage, "evidence": grounded,
                         "rationale": proposal.get("rationale"),
                         "stop_reason": "MODEL_ANSWER", "audit_flags": audit_flags}
                transcript.append({"executed_action": "ANSWER", "accepted_answer": final,
                                   "audit_flags": audit_flags})
                return final, transcript

        elif action == "JOINT_SEMANTIC_SCAN":
            raw_queries = proposal.get("queries_by_risk")
            queries = normalize_dynamic_queries(raw_queries)
            queries = {risk: items for risk, items in queries.items() if items}
            signature = tuple(sorted((risk, q.casefold().strip())
                                     for risk, items in queries.items() for q in items))
            if tool_calls >= max_tools or frame_remaining <= 0:
                rejection = "tool or frame budget exhausted"
            elif not isinstance(raw_queries, dict) or any(
                canonical_category(risk) not in RISKS or not isinstance(items, list) or
                not 1 <= len(items) <= 4 or
                any(not isinstance(item, str) or not item.strip() or len(item) > 240
                    for item in items)
                for risk, items in raw_queries.items()
            ):
                rejection = "invalid queries_by_risk schema"
            elif not signature:
                rejection = "JOINT_SEMANTIC_SCAN requires at least one query"
            elif signature in scan_signatures:
                rejection = "duplicate semantic scan"
            else:
                scan_signatures.add(signature)
                hits = locator(path, queries, top_k_per_risk=min(top_k, frame_remaining))
                merged = merge_hits(hits)
                chosen_times = [x["timestamp"] for x in merged]
                old = {round(f.timestamp, 3) for f in initial + acquired}
                new_times = [x for x in chosen_times if round(x, 3) not in old][:frame_remaining]
                fresh = _append_unique_frames(acquired, extract_frames(path, new_times),
                                              initial, frame_remaining)
                all_hits.extend(merged)
                tool_calls += 1
                transcript.append({"executed_action": action, "tool": action,
                                   "queries_by_risk": queries,
                                   "matches": merged, "added_timestamps": [f.timestamp for f in fresh]})

        elif action == "TEMPORAL_EXPAND":
            try:
                timestamps = [float(x) for x in proposal.get("timestamps", [])]
                window = float(proposal.get("window", 2.0))
                count = int(proposal.get("num_frames", 4))
            except (TypeError, ValueError):
                timestamps, window, count = [], 0, 0
            anchors = [f.timestamp for f in initial + acquired]
            known_hits = [x["timestamp"] for x in all_hits]
            if tool_calls >= max_tools or frame_remaining <= 0:
                rejection = "tool or frame budget exhausted"
            elif not timestamps or len(timestamps) > 3 or not (0.5 <= window <= 15):
                rejection = "invalid temporal anchors or window"
            elif not (1 <= count <= min(8, frame_remaining)):
                rejection = "invalid num_frames"
            elif not all(any(abs(t - known) <= 0.0006 for known in anchors + known_hits)
                         for t in timestamps):
                rejection = "temporal anchor was not observed"
            else:
                signature = (tuple(round(t, 3) for t in timestamps),
                             round(window, 3), count)
                if signature in expanded_signatures:
                    rejection = "duplicate temporal expansion"
                else:
                    expanded_signatures.add(signature)
                    fresh = _append_unique_frames(
                        acquired,
                        TemporalEvidenceExpander()(path, timestamps, window=window,
                                                   num_frames=count),
                        initial, frame_remaining,
                    )
                    tool_calls += 1
                    transcript.append({"executed_action": action, "tool": action,
                                       "anchors": timestamps,
                                       "window": window, "requested_frames": count,
                                       "added_timestamps": [f.timestamp for f in fresh]})
        else:
            rejection = "invalid action"

        if rejection:
            rejections += 1
            transcript.append({"rejected": {"reason": rejection, "action": action}})
            if rejections >= max_rejections:
                break

    stop_reason = "MAX_REJECTIONS" if rejections >= max_rejections else "MAX_DECISIONS"
    final = {"action": None, "moderation_action": None, "review_reason": None,
             "category": None, "severity": None, "stop_reason": stop_reason,
             "audit_flags": audit_flags}
    transcript.append({"stop_reason": stop_reason})
    return final, transcript


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--local-model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--retry-file", type=Path)
    parser.add_argument("--skip-previous-errors", action="store_true",
                        help="one-pass mode: leave prior failed samples in the retry ledger")
    parser.add_argument("--retry-incomplete-output", action="store_true",
                        help="for fallback batches, retry prior stopped trajectories")
    parser.add_argument("--pause-file", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--completion-marker", type=Path)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--sample-timeout-seconds", type=int, default=0,
                        help="optional wall-clock timeout per sample; timed-out samples go to retry file")
    parser.add_argument("--transient-retries", type=int, default=4,
                        help="retries for temporary API connectivity errors before pausing")
    parser.add_argument("--locator-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=5)
    parser.add_argument("--max-tools", type=int, default=3)
    parser.add_argument("--max-total-frames", type=int, default=24)
    parser.add_argument("--max-visible-frames", type=int, default=12)
    parser.add_argument("--max-rejections", type=int, default=2)
    parser.add_argument("--agent-version", default="thin_harness_teacher_v2")
    parser.add_argument("--audit-only-safety", action="store_true",
                        help="record unsupported safety answers for offline audit without replacing them")
    args = parser.parse_args()
    runs = read_jsonl(args.runs)
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        parser.error("shard-index must be within num-shards")
    runs = [row for index, row in enumerate(runs) if index % args.num_shards == args.shard_index]
    manifest = {row["id"]: row for row in read_jsonl(args.data)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    retry_file = args.retry_file or args.output.with_name(args.output.stem + ".retry.jsonl")
    done = _completed_ids(args.output, retry_file,
                          skip_previous_errors=args.skip_previous_errors,
                          retry_incomplete_output=args.retry_incomplete_output)
    if args.local_model:
        brain = LocalBrain(args.local_model, max_output_tokens=args.max_output_tokens)
    else:
        brain = RemoteBrain(args.model, base_url=args.base_url, api_key=args.api_key,
                            max_output_tokens=args.max_output_tokens, strict_images=True,
                            response_schema=_thin_action_schema())
    locator = JointRiskSemanticLocator(device=args.locator_device)
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    policy_sha256 = hashlib.sha256(SAFETY_POLICY.encode("utf-8")).hexdigest()
    run_config = {"max_decisions": args.max_decisions,
                  "policy_version": POLICY_VERSION,
                  "max_tools": args.max_tools,
                  "max_total_frames": args.max_total_frames,
                  "max_visible_frames": args.max_visible_frames,
                  "max_rejections": args.max_rejections,
                  "top_k": args.top_k,
                  "audit_only_safety": args.audit_only_safety,
                  "runner_sha256": runner_sha256,
                  "response_schema_sha256": getattr(brain, "response_schema_hash", None),
                  "safety_policy_sha256": policy_sha256}
    attempted = 0
    completed = True
    for index, row in enumerate(runs, 1):
        if args.pause_file and args.pause_file.exists():
            print("pause requested", flush=True)
            completed = False
            break
        if args.max_samples is not None and attempted >= args.max_samples:
            completed = False
            break
        if row["id"] in done:
            continue
        attempted += 1
        started = time.monotonic()
        response_model_start = len(getattr(brain, "response_models", []))
        print(f"[{index}/{len(runs)}] {row['id']} start", flush=True)
        try:
            with sample_timeout(args.sample_timeout_seconds):
                for retry_index in range(args.transient_retries + 1):
                    try:
                        final, transcript = run_one(
                            video_path(manifest[row["id"]]), brain, locator,
                            max_decisions=args.max_decisions, max_tools=args.max_tools,
                            max_total_frames=args.max_total_frames, top_k=args.top_k,
                            max_visible_frames=args.max_visible_frames,
                            max_rejections=args.max_rejections,
                            audit_only_safety=args.audit_only_safety,
                        )
                        break
                    except Exception as transient_exc:
                        if not _transient_api_error(str(transient_exc)) or retry_index == args.transient_retries:
                            raise
                        delay = min(10 * (2 ** retry_index), 60)
                        print(f"[{index}/{len(runs)}] {row['id']} temporary API failure; "
                              f"retry {retry_index + 1}/{args.transient_retries} in {delay}s", flush=True)
                        time.sleep(delay)
            record = {"id": row["id"], "label": row.get("category") or row.get("label"),
                      "label_severity": row.get("severity") or row.get("label_severity"),
                      "model": args.model, "agent_version": args.agent_version,
                      "response_models": getattr(brain, "response_models", [])[response_model_start:],
                      "trajectory_status": "ANSWERED" if final.get("action") == "ANSWER" else "STOPPED_WITHOUT_ANSWER",
                      "run_config": run_config,
                      "result": {"final_action": final, "transcript": transcript,
                                 "tool_calls": [x["tool"] for x in transcript if "tool" in x],
                                 "decision_count": sum("raw_decision" in x for x in transcript),
                                 "unique_frames_acquired": len({round(t, 3) for x in transcript
                                    for t in (x.get("observation", {}).get("initial_timestamps", []) +
                                              x.get("added_timestamps", []))})},
                      "elapsed_seconds": round(time.monotonic() - started, 3)}
        except Exception as exc:
            record = {"id": row["id"], "model": args.model,
                      "agent_version": args.agent_version,
                      "response_models": getattr(brain, "response_models", [])[response_model_start:],
                      "run_config": run_config,
                      "result": {"transcript": getattr(exc, "transcript", []),
                                 "tool_calls": [step["tool"] for step in
                                                getattr(exc, "transcript", []) if "tool" in step]},
                      "error": f"{type(exc).__name__}: {exc}",
                      "elapsed_seconds": round(time.monotonic() - started, 3)}
            if _transient_api_error(record["error"]):
                print(f"[{index}/{len(runs)}] {row['id']} transient API error; retry-file and continue: "
                      f"{record['error']}", flush=True)
        destination = retry_file if "error" in record else args.output
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        if "error" not in record:
            done.add(row["id"])
        final = (record.get("result") or {}).get("final_action") or {}
        print(f"[{index}/{len(runs)}] {row['id']} action={final.get('moderation_action')} "
              f"tools={(record.get('result') or {}).get('tool_calls')} "
              f"error={record.get('error')}", flush=True)
    if completed and args.completion_marker:
        args.completion_marker.parent.mkdir(parents=True, exist_ok=True)
        args.completion_marker.write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
