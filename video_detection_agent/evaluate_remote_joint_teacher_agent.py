#!/usr/bin/env python3
"""Generate trajectories for the joint three-risk Teacher Agent."""

import argparse
import base64
import hashlib
import concurrent.futures
import io
import json
import os
import re
import threading
import time
from pathlib import Path

from openai import OpenAI

from agent.qwen_backend import CATEGORIES, SAFETY_POLICY, canonical_category
from agent.risk_queries import RISK_QUERY_BANK, risk_queries
from agent.tools import JointRiskSemanticLocator, TemporalEvidenceExpander
from agent.video import extract_frames, uniform_frames, video_duration
from evaluate_dataset import read_jsonl, video_path

RISKS = list(CATEGORIES[1:])


def canonical_severity(value, category=None):
    """Normalize controller variants to the project's L0/L1/L2 schema."""
    if canonical_category(category) == "Harmless":
        return "L0"
    match = re.search(r"L\s*([0-3])", str(value or ""), re.IGNORECASE)
    if not match:
        return value
    level = min(int(match.group(1)), 2)
    return f"L{level}"


def image_url(image, side=640):
    image = image.convert("RGB")
    image.thumbnail((side, side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=78, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


class RemoteBrain:
    def __init__(self, model, base_url=None, api_key=None, max_output_tokens=4096,
                 strict_images=False, response_schema=None):
        """OpenAI-compatible visual controller (302.ai or a local vLLM server)."""
        key = api_key or os.environ.get("API_KEY_302")
        if not key:
            env = Path(__file__).resolve().parents[1] / ".env.302"
            for line in env.read_text().splitlines() if env.exists() else []:
                name, sep, value = line.partition("=")
                if sep and name.strip() == "API_KEY_302":
                    key = value.strip().strip("'\"")
        if not key and not base_url:
            raise RuntimeError("API_KEY_302 is required")
        self.api_key = key or "local-vllm"
        self.base_url = base_url or "https://api.302.ai/v1"
        self.model = model
        self.is_local_vllm = bool(base_url)
        self.max_output_tokens = max_output_tokens
        self.strict_images = strict_images
        self.response_schema = response_schema
        self.response_schema_hash = (
            hashlib.sha256(json.dumps(response_schema, sort_keys=True).encode("utf-8")).hexdigest()
            if response_schema else None
        )
        self.usage = []
        self.response_models = []
        self.last_raw_response = None

    def call(self, frames, prompt):
        last = None
        self.last_raw_response = None
        image_attempts = ((12, 640), (12, 640), (12, 640)) if self.strict_images else ((12, 640), (8, 576), (4, 512))
        for limit, side in image_attempts:
            try:
                chosen = frames
                if len(chosen) > limit:
                    chosen = [chosen[min(int((i + .5) * len(chosen) / limit), len(chosen)-1)]
                              for i in range(limit)]
                content = [{"type": "text", "text": prompt}]
                content += [{"type": "image_url", "image_url": {"url": image_url(f.image, side)}}
                            for f in chosen]
                request_started = time.monotonic()
                print(f"model_request start model={self.model} frames={len(chosen)}", flush=True)
                # Match the causal evaluator: don't reuse an idle connection
                # across image retrieval/model loading between Agent turns.
                # The caller owns retries and its sample deadline.
                with OpenAI(api_key=self.api_key, base_url=self.base_url,
                            timeout=180, max_retries=0) as client:
                    response_format = (
                        {"type": "json_schema", "json_schema": self.response_schema}
                        if self.response_schema else {"type": "json_object"}
                    )
                    stream = client.chat.completions.create(
                        model=self.model, messages=[{"role": "user", "content": content}],
                        temperature=0, max_tokens=self.max_output_tokens,
                        response_format=response_format,
                        stream=True, stream_options={"include_usage": True},
                        extra_body=(
                            {"chat_template_kwargs": {"enable_thinking": False}}
                            if self.is_local_vllm else {"enable_thinking": False}
                        ),
                    )
                    parts = []
                    response_model = None
                    last_progress = request_started
                    for chunk in stream:
                        response_model = chunk.model or response_model
                        if chunk.usage:
                            self.usage.append(chunk.usage.model_dump())
                        if chunk.choices and chunk.choices[0].delta.content:
                            parts.append(chunk.choices[0].delta.content)
                            self.last_raw_response = "".join(parts)
                        now = time.monotonic()
                        if now - last_progress >= 30:
                            print(f"model_request progress seconds={now-request_started:.1f} "
                                  f"characters={len(self.last_raw_response or '')}", flush=True)
                            last_progress = now
                self.response_models.append(response_model or self.model)
                self.last_raw_response = "".join(parts)
                print(f"model_request done seconds={time.monotonic()-request_started:.1f} "
                      f"characters={len(self.last_raw_response)}", flush=True)
                return json.loads(self.last_raw_response)
            except Exception as exc:
                # A timed-out sample must leave this fallback loop immediately.
                # Otherwise the one-shot SIGALRM is swallowed here and smaller
                # image retries continue without the per-sample deadline.
                cause = exc
                seen = set()
                while cause is not None and id(cause) not in seen:
                    seen.add(id(cause))
                    if isinstance(cause, TimeoutError):
                        # The SDK can wrap the SIGALRM exception inside an
                        # APIConnectionError. Preserve that deadline too.
                        raise cause
                    cause = cause.__cause__ or cause.__context__
                if type(exc).__name__ == "APITimeoutError":
                    raise
                last = exc
        detail = f"; underlying={type(last.__cause__).__name__}: {str(last.__cause__)[:240]}" if last.__cause__ else ""
        raise RuntimeError(f"remote joint policy failed after 3 attempts: {last}{detail}")


def initial_count(duration):
    # Every video starts from the same deliberately sparse global view.  The
    # Agent must use tools, rather than a growing initial image budget, when
    # four visible frames are insufficient.
    return 4


def merge_hits(hits, separation=0.75):
    groups = []
    for hit in sorted(hits, key=lambda item: item.timestamp):
        if not groups or hit.timestamp - groups[-1][-1].timestamp > separation:
            groups.append([hit])
        else:
            groups[-1].append(hit)
    merged = []
    for group in groups:
        best = max(group, key=lambda item: item.score)
        merged.append({
            "timestamp": best.timestamp,
            "risks": sorted({item.risk_type for item in group}),
            "matches": [{"risk_type": item.risk_type, "query": item.matched_query,
                         "score": item.score} for item in group],
        })
    return merged


def evidence(data, visible):
    if not isinstance(data, dict) or not visible:
        return []
    result = []
    raw = data.get("evidence", []) or []
    # Some otherwise valid teachers return a timestamped prose string instead of
    # the requested list. Recover only timestamps that can be snapped to a frame
    # that was actually shown; the query text or SigLIP score is never accepted.
    if isinstance(raw, str):
        prose = raw.strip()
        recovered = []
        for token in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", prose):
            timestamp = float(token)
            nearest = min(visible, key=lambda known: abs(known - timestamp))
            tolerance = max(.5, (max(visible) - min(visible)) / max(len(visible) - 1, 1) * .35)
            if abs(nearest - timestamp) <= tolerance and nearest not in [item["timestamp"] for item in recovered]:
                recovered.append({"timestamp": nearest, "description": prose})
            if len(recovered) == 2:
                break
        return recovered
    if isinstance(raw, dict):
        raw = [raw]
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            timestamp = float(item["timestamp"])
            nearest = min(visible, key=lambda known: abs(known - timestamp))
            # The prompt exposes rounded timestamps, while decoded video timestamps can
            # differ slightly. Snap evidence to the closest actually visible frame.
            tolerance = max(.5, (max(visible) - min(visible)) / max(len(visible) - 1, 1) * .35)
            if abs(nearest - timestamp) <= tolerance and str(item.get("description", "")).strip():
                result.append({"timestamp": nearest, "description": item["description"]})
        except Exception:
            pass
    return result[:2]


def normalize_response(data):
    """Tolerate common JSON shapes returned by remote controller models."""
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        dictionaries = [item for item in data if isinstance(item, dict)]
        if len(dictionaries) == 1:
            return dictionaries[0]
        return {"action": "JOINT_SEMANTIC_SCAN", "queries_by_risk": data}
    return {}


def normalize_dynamic_queries(dynamic, hinted_risk=None):
    """Convert dict/list/string query output into the canonical per-risk mapping."""
    result = {risk: [] for risk in RISKS}
    if isinstance(dynamic, dict):
        for key, value in dynamic.items():
            risk = canonical_category(key)
            if risk not in result:
                continue
            values = value if isinstance(value, list) else [value]
            result[risk].extend(str(item).strip() for item in values if str(item).strip())
        return result
    if isinstance(dynamic, list):
        for item in dynamic:
            if isinstance(item, dict):
                risk = canonical_category(item.get("risk_type") or item.get("category"))
                values = item.get("queries") or item.get("query") or []
                values = values if isinstance(values, list) else [values]
                if risk in result:
                    result[risk].extend(str(value).strip() for value in values if str(value).strip())
            elif hinted_risk in result and str(item).strip():
                result[hinted_risk].append(str(item).strip())
    elif hinted_risk in result and isinstance(dynamic, str) and dynamic.strip():
        result[hinted_risk].append(dynamic.strip())
    return result


def normalize_assessments(value):
    """Normalize model-produced risk assessments to {canonical_risk: dict}."""
    result = {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        items = []
        for item in value:
            if not isinstance(item, dict):
                continue
            risk = item.get("risk_type") or item.get("category") or item.get("risk")
            items.append((risk, item))
    else:
        items = []
    aliases = {
        "sexual_content": "Sexual Content",
        "violence_dangerous": "Violence & Dangerous",
        "sensitive_figures_ideologies": "Sensitive Figures & Ideologies",
    }
    for key, assessment in items:
        normalized_key = aliases.get(str(key).strip().lower(), canonical_category(key))
        if normalized_key not in RISKS:
            continue
        if isinstance(assessment, str):
            assessment = {"status": assessment}
        if isinstance(assessment, dict):
            result[normalized_key] = assessment
    return result


def classify_initial_frames(initial, brain):
    """Record the no-tool four-frame baseline inside the full Student run."""
    prompt = f"""你是视频内容安全分类器。当前图片是同一视频均匀抽取的4张初始帧。
禁止请求工具或更多帧，只根据当前可见内容给出当前最佳判断。
类别只能是 Harmless、Sexual Content、Violence & Dangerous、Sensitive Figures & Ideologies。
等级规则：Harmless只能L0；轻微或边界风险L1；明确风险L2。
{SAFETY_POLICY}
只输出JSON：{{"category":"...","severity":"L0/L1/L2","rationale":"不超过30字的可见证据"}}"""
    data = normalize_response(brain.call(initial, prompt))
    category = canonical_category(data.get("category"))
    severity = canonical_severity(data.get("severity"), category)
    if category not in CATEGORIES:
        raise ValueError(f"invalid initial category: {data.get('category')!r}")
    if severity not in {"L0", "L1", "L2"}:
        raise ValueError(f"invalid initial severity: {data.get('severity')!r}")
    return {
        "category": category,
        "severity": severity,
        "rationale": str(data.get("rationale") or "").strip(),
    }


def run_one(
    path, brain, locator, top_k, agent_role="Teacher", *, duration=None, initial=None,
    evidence_protocol_v2=False,
):
    duration = video_duration(path) if duration is None else duration
    initial = uniform_frames(path, initial_count(duration)) if initial is None else initial
    if not initial:
        raise RuntimeError(
            "no visible frames after decoding/black-frame filtering; "
            "refusing to run the Teacher without visual evidence"
        )
    transcript = [{"observation": {"duration": duration, "frames_seen": len(initial),
                   "timestamps": [f.timestamp for f in initial],
                   "initial_sampling": "uniform_4_non_black"}}]
    initial_prompt = f"""你是视频安全Joint {agent_role} Agent。图片按时间顺序排列。
先判断当前帧是否已有明确风险证据。若有，输出action=ANSWER；若没有，输出action=JOINT_SEMANTIC_SCAN，并为三个风险类别各生成1-4条结合当前场景的、具体可见的英文检索query。固定query库会由Python自动补充，不要照抄通用词。
不得因为当前没看到就直接Harmless。敏感旗帜规则：只要确认旗帜涉及香港、西藏或台湾，Sensitive-L2。
{SAFETY_POLICY}
输出JSON。ANSWER字段：action,category,severity,rationale,evidence,evidence_strength,evidence_basis,identified_figure。扫描字段：action,queries_by_risk，键必须是{RISKS}。时间戳：{[round(f.timestamp,3) for f in initial]}"""
    first = normalize_response(brain.call(initial, initial_prompt))
    transcript.append({"decision": first})
    if str(first.get("action", "")).upper() == "ANSWER":
        ev = evidence(first, [f.timestamp for f in initial])
        direct_category = canonical_category(first.get("category"))
        direct_severity = canonical_severity(first.get("severity"), direct_category)
        direct_strength = str(first.get("evidence_strength") or "").lower()
        # Initial L1/ambiguous output is a hypothesis that still requires the
        # joint evidence workflow. Only clear, grounded L2 harm may terminate
        # immediately from the four-frame observation.
        if (direct_category in RISKS and direct_severity == "L2"
                and direct_strength in {"clear", "strong", "unambiguous"} and ev):
            final = dict(first); final["category"] = direct_category; final["evidence"] = ev
            final["severity"] = direct_severity
            return final, transcript, len(initial), []
        # A harmful prediction without grounded visible evidence is a hypothesis,
        # not a usable answer. Continue through the joint evidence workflow.
        transcript.append({"decision": {"action": "JOINT_SEMANTIC_SCAN",
                           "automatic": True,
                           "reason": "direct answer was not clear grounded L2 evidence"}})

    dynamic = normalize_dynamic_queries(first.get("queries_by_risk") or first.get("queries") or
                                        first.get("query") or [],
                                        canonical_category(first.get("category")))
    queries_by_risk = {risk: risk_queries(risk, dynamic[risk]) for risk in RISKS}
    hits = locator(path, queries_by_risk, top_k_per_risk=top_k)
    merged = merge_hits(hits)
    transcript.append({"tool": "JOINT_SEMANTIC_SCAN", "queries_by_risk": queries_by_risk,
                       "observation": merged})
    candidate_times = [item["timestamp"] for item in merged]
    candidates = extract_frames(path, candidate_times)
    review_frames = sorted(initial + candidates, key=lambda f: f.timestamp)
    review_prompt = f"""独立查看当前初始帧和联合检索候选帧。SigLIP query与分数不是证据。
分别判断三个类别为PRESENT、ABSENT或INSUFFICIENT。PRESENT必须有当前可见时间戳证据；单帧缺少动作/语境时为INSUFFICIENT。若任一PRESENT，同时输出final_answer(category,severity,rationale,evidence,evidence_strength,evidence_basis,identified_figure)。若无PRESENT，输出assessments，每类含status、candidate_timestamps和reason。三类ABSENT可同时输出final_answer=Harmless-L0。敏感旗帜涉及香港、西藏或台湾即Sensitive-L2。
final_answer必须与assessments同级，evidence必须是数组，每项为{{"timestamp":秒数,"description":"当前图片中实际可见的证据"}}。
{SAFETY_POLICY}
当前时间戳：{[round(f.timestamp,3) for f in review_frames]}。只输出JSON。"""
    assessment = normalize_response(brain.call(review_frames, review_prompt))
    transcript.append({"decision": {"action": "ASSESS_RISKS", **assessment}})
    final_data = assessment.get("final_answer")
    # Qwen occasionally nests final_answer inside assessments despite the schema.
    # Accept that recoverable shape rather than silently converting PRESENT harm
    # into Harmless.
    if not final_data and isinstance(assessment.get("assessments"), dict):
        final_data = assessment["assessments"].get("final_answer")
    if isinstance(final_data, list):
        final_data = next((item for item in final_data if isinstance(item, dict)), None)
    if not isinstance(final_data, dict):
        final_data = None
    assessments = normalize_assessments(assessment.get("assessments") or {})
    all_risks_absent = all(
        str((assessments.get(risk) or {}).get("status", "")).upper() == "ABSENT"
        for risk in RISKS
    )
    provisional = None
    if final_data:
        final = {"action": "ANSWER", **final_data}
        final["category"] = canonical_category(final.get("category"))
        final["severity"] = canonical_severity(final.get("severity"), final["category"])
        final["evidence"] = evidence(final, [f.timestamp for f in review_frames])
        provisional = final
        # v2 has an explicit evidence-acquisition policy.  A benign outcome is
        # valid only after all three risks were structurally excluded.  L1 is
        # a review hypothesis, so it receives temporal context before it can
        # become a final decision.  Clear grounded L2 evidence may stop here.
        if not evidence_protocol_v2 and (final["category"] == "Harmless" or final["evidence"]):
            return final, transcript, len(review_frames), []
        if evidence_protocol_v2:
            if final["category"] == "Harmless" and all_risks_absent:
                return final, transcript, len(review_frames), []
            if (final["category"] in RISKS and final["severity"] == "L2"
                    and final["evidence"]):
                return final, transcript, len(review_frames), []

    if evidence_protocol_v2:
        # PRESENT L1 and every non-ABSENT assessment require temporal
        # verification.  This is deliberately stricter than v1, which
        # frequently terminated on a single ambiguous candidate frame.
        insufficient = [risk for risk in RISKS
                        if str((assessments.get(risk) or {}).get("status", "")).upper()
                        != "ABSENT"]
        if provisional and provisional["category"] in RISKS and provisional["severity"] == "L1":
            if provisional["category"] not in insufficient:
                insufficient.append(provisional["category"])
    else:
        insufficient = [risk for risk in RISKS
                        if str((assessments.get(risk) or {}).get("status", "")).upper()
                        in {"INSUFFICIENT", "PRESENT"}]
    if not insufficient:
        if evidence_protocol_v2 and not all_risks_absent:
            raise RuntimeError("v2 cannot ALLOW without structured three-risk exclusion")
        final = {"action": "ANSWER", "category": "Harmless", "severity": "L0",
                 "rationale": "联合候选视觉核验未发现三类风险证据。", "evidence": []}
        transcript.append({"decision": final})
        return final, transcript, len(review_frames), []

    expand_times = []
    for risk in insufficient:
        requested = (assessments.get(risk) or {}).get("candidate_timestamps") or []
        valid = [float(t) for t in requested if any(abs(float(t)-known) <= .3 for known in candidate_times)]
        if not valid:
            valid = [h.timestamp for h in hits if h.risk_type == risk][:2]
        expand_times.extend(valid[:2])
    expand_times = list(dict.fromkeys(expand_times))
    if evidence_protocol_v2 and not expand_times:
        # A malformed assessment must not silently skip temporal verification.
        # The best merged candidates still provide real, visible context.
        expand_times = candidate_times[:2]
    if not expand_times:
        raise RuntimeError("temporal verification requested without candidate timestamps")
    expanded = TemporalEvidenceExpander()(path, expand_times, window=4.0, num_frames=min(16, 4*len(insufficient)))
    transcript.append({"decision": {"action": "TEMPORAL_EXPAND", "risk_types": insufficient,
                       "timestamps": expand_times, "window": 4.0, "num_frames": min(16,4*len(insufficient))}})
    transcript.append({"tool": "TEMPORAL_EXPAND", "risk_types": insufficient,
                       "observation": {"timestamps": [f.timestamp for f in expanded],
                                       "added_frames": len(expanded)}})
    final_frames = sorted(initial + candidates + expanded, key=lambda f: f.timestamp)
    final_prompt = f"""这是定向补帧后的最终视觉判断。只依据当前图片，对三类给出最终状态。任一类有达到L1/L2的实际可见证据则ANSWER该类别；三类均无证据则ANSWER Harmless-L0。不得引用query或分数。敏感旗帜涉及香港、西藏或台湾即Sensitive-L2。输出JSON：category,severity,rationale,evidence,evidence_strength,evidence_basis,identified_figure。{SAFETY_POLICY}\n时间戳：{[round(f.timestamp,3) for f in final_frames]}"""
    data = brain.call(final_frames, final_prompt)
    final = {"action": "ANSWER", **data}
    final["category"] = canonical_category(final.get("category"))
    final["severity"] = canonical_severity(final.get("severity"), final["category"])
    final["evidence"] = evidence(final, [f.timestamp for f in final_frames])
    if final["category"] != "Harmless" and not final["evidence"]:
        raise RuntimeError("final harmful answer lacks visible evidence")
    transcript.append({"decision": final})
    return final, transcript, len(final_frames), [f.timestamp for f in expanded]


def process_row(index, row, manifest, args, brain, locator, top_k):
    """Process one sample; callers serialize JSONL writes in the main thread."""
    usage_start = len(brain.usage)
    initial_baseline = None
    initial_error = None
    initial_usage = []
    try:
        path = video_path(manifest[row["id"]])
        prepared_duration = None
        prepared_initial = None
        if args.agent_role == "Student":
            prepared_duration = video_duration(path)
            prepared_initial = uniform_frames(path, initial_count(prepared_duration))
            if not prepared_initial:
                raise RuntimeError(
                    "no visible frames after decoding/black-frame filtering; "
                    "refusing to run the Student without visual evidence"
                )
            baseline_usage_start = len(brain.usage)
            try:
                initial_baseline = classify_initial_frames(prepared_initial, brain)
            except Exception as exc:
                initial_error = f"{type(exc).__name__}: {exc}"
            initial_usage = brain.usage[baseline_usage_start:]
        final, transcript, total_frames, added = run_one(
        path, brain, locator, top_k, args.agent_role,
            duration=prepared_duration, initial=prepared_initial,
            evidence_protocol_v2=args.evidence_protocol_v2,
        )
        record = {"id": row["id"], "label": row.get("label", row.get("category")),
                  "label_severity": row.get("label_severity", row.get("severity")),
                  "reference_tool_calls": row.get("tool_calls") or [],
                  "reference_total_frames": (row.get("result") or {}).get("total_frames"),
                  "model": args.model,
                  "agent_version": (
                      f"joint_risk_v2_{args.agent_role.lower()}"
                      if args.evidence_protocol_v2 else f"joint_risk_v1_{args.agent_role.lower()}"
                  ), "pred": final.get("category"),
                  "severity": final.get("severity"), "final_action": "ANSWER",
                  "tool_calls": [step["tool"] for step in transcript if "tool" in step],
                  "result": {"final_action": final, "transcript": transcript,
                             "total_frames": total_frames, "added_timestamps": added},
                  "usage_by_round": brain.usage[usage_start:]}
        if args.agent_role == "Student":
            record.update({
                "initial_pred": initial_baseline.get("category") if initial_baseline else None,
                "initial_severity": initial_baseline.get("severity") if initial_baseline else None,
                "initial_rationale": initial_baseline.get("rationale") if initial_baseline else None,
                "initial_error": initial_error,
                "initial_usage_by_round": initial_usage,
            })
    except Exception:
        # A failed sample must never enter the success JSONL: cached IDs are
        # derived from that file, so writing it there would permanently skip it.
        raise
    return index, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--agent-role", choices=("Teacher", "Student"), default="Teacher")
    parser.add_argument("--base-url", default=None,
                        help="Optional OpenAI-compatible endpoint, e.g. http://127.0.0.1:8008/v1")
    parser.add_argument("--api-key", default=None,
                        help="API key for --base-url; local vLLM accepts any non-empty value")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--locator-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retry-file", type=Path,
                        help="Separate queue for samples that exhausted all retries")
    parser.add_argument("--pause-file", type=Path)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--evidence-protocol-v2", action="store_true",
                        help="Require three-risk exclusion for ALLOW and temporal verification for L1/insufficient evidence.")
    args = parser.parse_args()
    rows = read_jsonl(args.runs)
    if args.max_samples: rows = rows[:args.max_samples]
    manifest = {r["id"]: r for r in read_jsonl(args.data)}
    cached = {r["id"] for r in read_jsonl(args.output)} if args.output.exists() else set()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = [(index, row) for index, row in enumerate(rows, 1) if row["id"] not in cached]
    local = threading.local()

    def work(item):
        if args.pause_file and args.pause_file.exists():
            return None
        if not hasattr(local, "brain"):
            local.brain = RemoteBrain(
                args.model, base_url=args.base_url, api_key=args.api_key,
                max_output_tokens=args.max_output_tokens,
            )
            local.locator = JointRiskSemanticLocator(device=args.locator_device)
        # RemoteBrain.call already degrades image count 12 -> 8 -> 4. These
        # whole-sample retries additionally reduce retrieval/output cost, then
        # move SigLIP off CUDA for the final recovery attempt.
        plans = (
            (args.top_k, args.max_output_tokens, args.locator_device),
            (max(1, args.top_k - 1), min(args.max_output_tokens, 3072), args.locator_device),
            (1, min(args.max_output_tokens, 2048), args.locator_device),
            (1, min(args.max_output_tokens, 2048), "cpu"),
        )
        recovery_history = []
        for attempt, (top_k, max_tokens, device) in enumerate(plans, 1):
            try:
                if local.locator.device != device:
                    local.locator = JointRiskSemanticLocator(device=device)
                local.brain.max_output_tokens = max_tokens
                index, record = process_row(
                    item[0], item[1], manifest, args, local.brain, local.locator, top_k,
                )
                record["recovery_attempt"] = attempt
                record["recovery_degraded"] = attempt > 1
                record["recovery_history"] = recovery_history
                return "success", index, record
            except Exception as exc:
                message = str(exc)
                lowered = message.lower()
                event = {
                    "attempt": attempt,
                    "device": device,
                    "top_k": top_k,
                    "max_output_tokens": max_tokens,
                    "error_type": type(exc).__name__,
                    "error": message[:2000],
                    "suspected_oom": (
                        "out of memory" in lowered or "cuda oom" in lowered
                        or type(exc).__name__ == "OutOfMemoryError"
                    ),
                }
                recovery_history.append(event)
                print(
                    f"[{item[0]}/{len(rows)}] {item[1]['id']} retry "
                    f"attempt={attempt} device={device} top_k={top_k} "
                    f"error_type={event['error_type']} suspected_oom={event['suspected_oom']} "
                    f"error={event['error']}",
                    flush=True,
                )
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(min(2 ** attempt, 10))
        failure = {"id": item[1]["id"], "index": item[0], "attempts": len(plans),
                   "recovery_history": recovery_history, "retry_required": True}
        return "failed", item[0], failure

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(work, pending):
            if result is None:
                continue
            status, index, record = result
            if status == "failed":
                if args.retry_file:
                    args.retry_file.parent.mkdir(parents=True, exist_ok=True)
                    with args.retry_file.open("a") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{index}/{len(rows)}] {record['id']} retry-required "
                      f"attempts={record['attempts']}", flush=True)
                continue
            with args.output.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False)+"\n")
            print(f"[{index}/{len(rows)}] {record['id']} pred={record.get('pred')} severity={record.get('severity')} tools={record.get('tool_calls')} error={record.get('error')}", flush=True)


if __name__ == "__main__": main()
