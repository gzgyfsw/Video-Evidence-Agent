"""Deterministic GRPO v1 reward for the video-safety Agent.

The reward deliberately contains no learned judge.  Infrastructure failures
return ``None`` and must be retried or removed rather than optimized against.
All route, grounding, temporal, and cost terms are enabled only after an exact
category+severity answer, so efficiency cannot dominate safety correctness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os


HARMFUL_CATEGORIES = {
    "Sexual Content",
    "Violence & Dangerous",
    "Sensitive Figures & Ideologies",
}
VALID_CATEGORIES = HARMFUL_CATEGORIES | {"Harmless"}


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


EXACT_REWARD = _float_env("VIDEO_GRPO_EXACT_REWARD", 4.0)
CATEGORY_ONLY_REWARD = _float_env("VIDEO_GRPO_CATEGORY_ONLY_REWARD", 1.5)
HARMFUL_MISS_REWARD = _float_env("VIDEO_GRPO_HARMFUL_MISS_REWARD", -5.0)
HARMLESS_FALSE_POSITIVE_REWARD = _float_env("VIDEO_GRPO_HARMLESS_FP_REWARD", -2.0)
HARMFUL_CATEGORY_CONFUSION_REWARD = _float_env("VIDEO_GRPO_CATEGORY_CONFUSION_REWARD", -1.0)
INVALID_REWARD = _float_env("VIDEO_GRPO_INVALID_REWARD", -5.0)
REVIEW_EXACT_REWARD = _float_env("VIDEO_GRPO_REVIEW_EXACT_REWARD", 4.0)
REVIEW_SAFE_REWARD = _float_env("VIDEO_GRPO_REVIEW_SAFE_REWARD", 2.0)
REVIEW_CATEGORY_REWARD = _float_env("VIDEO_GRPO_REVIEW_CATEGORY_REWARD", 2.5)
REVIEW_WRONG_CATEGORY_REWARD = _float_env("VIDEO_GRPO_REVIEW_WRONG_CATEGORY_REWARD", -1.0)
REVIEW_ALLOW_REWARD = _float_env("VIDEO_GRPO_REVIEW_ALLOW_REWARD", -5.0)
ROUTE_BONUS = _float_env("VIDEO_GRPO_ROUTE_BONUS", 0.3)
TEMPORAL_BONUS = _float_env("VIDEO_GRPO_TEMPORAL_BONUS", 0.2)
GROUNDING_BONUS = _float_env("VIDEO_GRPO_GROUNDING_BONUS", 0.1)
TOOL_COST = _float_env("VIDEO_GRPO_TOOL_COST", 0.03)
EXTRA_FRAME_COST = _float_env("VIDEO_GRPO_EXTRA_FRAME_COST", 0.003)
MAX_COST = _float_env("VIDEO_GRPO_MAX_COST", 0.30)


@dataclass(frozen=True)
class RewardInput:
    gold_category: str
    gold_severity: str
    pred_category: str | None = None
    pred_severity: str | None = None
    valid_json: bool = True
    has_final_answer: bool = True
    infrastructure_error: bool = False

    # Assistant provenance. Framework-inserted actions must never set these.
    valid_direct_answer: bool = False
    framework_rescued: bool = False
    model_initiated_joint_scan: bool = False
    model_initiated_temporal_expand: bool = False

    # Temporal bonus is fail-closed: all three conditions are required.
    teacher_state_requires_temporal_context: bool = False
    temporal_result_was_consumed: bool = False

    # For harmful answers this means every rewarded evidence timestamp belongs
    # to a frame actually shown. For Harmless it means the three risks were
    # explicitly excluded after real retrieval.
    evidence_is_grounded: bool = False

    num_tool_calls: int = 0
    num_extra_frames: int = 0

    # Optional action-aware labels.  Clean rows keep the historical strict
    # category+severity objective.  Boundary/ambiguous rows optimize the
    # operational moderation action, because independent Teachers disagree on
    # a unique severity or even category.
    severity_mode: str = "clean"
    target_action: str | None = None
    acceptable_actions: tuple[str, ...] = ()
    acceptable_severities: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewardBreakdown:
    total: float | None
    task: float | None
    route: float = 0.0
    expand: float = 0.0
    grounding: float = 0.0
    cost: float = 0.0
    exact: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def compute_reward(item: RewardInput) -> RewardBreakdown:
    """Compute the frozen scalar reward and an auditable decomposition."""
    if item.infrastructure_error:
        return RewardBreakdown(total=None, task=None, reason="infrastructure_error")
    if not item.valid_json or not item.has_final_answer:
        return RewardBreakdown(total=INVALID_REWARD, task=INVALID_REWARD, reason="invalid_or_missing_final_answer")

    category_correct = item.pred_category == item.gold_category
    severity_correct = item.pred_severity == item.gold_severity
    pred_action = action_from_prediction(item.pred_category, item.pred_severity)
    mode = item.severity_mode or "clean"
    if mode in {"boundary", "ambiguous"}:
        acceptable = set(item.acceptable_actions or ("REVIEW", "BLOCK"))
        target = item.target_action or "REVIEW"
        if pred_action == "ALLOW":
            return RewardBreakdown(total=REVIEW_ALLOW_REWARD, task=REVIEW_ALLOW_REWARD, reason=f"{mode}_unsafe_allow")
        if mode == "boundary":
            if category_correct and pred_action == target:
                task = REVIEW_EXACT_REWARD
                exact = True
            elif category_correct and pred_action in acceptable:
                task = REVIEW_CATEGORY_REWARD
                exact = False
            elif pred_action in acceptable:
                task = REVIEW_WRONG_CATEGORY_REWARD
                exact = False
            else:
                task = HARMFUL_CATEGORY_CONFUSION_REWARD
                exact = False
        else:
            if pred_action == target:
                task = REVIEW_EXACT_REWARD
                exact = True
            elif pred_action in acceptable:
                task = REVIEW_SAFE_REWARD
                exact = False
            else:
                task = HARMFUL_CATEGORY_CONFUSION_REWARD
                exact = False
        if not exact:
            return RewardBreakdown(total=task, task=task, reason=f"{mode}_action_partial")
        return _with_route_terms(item, task, exact=True, reason=f"{mode}_review")

    exact = category_correct and severity_correct
    if exact:
        task = EXACT_REWARD
    elif category_correct:
        return RewardBreakdown(total=CATEGORY_ONLY_REWARD, task=CATEGORY_ONLY_REWARD, reason="category_only")
    elif item.gold_category in HARMFUL_CATEGORIES and item.pred_category == "Harmless":
        return RewardBreakdown(total=HARMFUL_MISS_REWARD, task=HARMFUL_MISS_REWARD, reason="harmful_miss")
    elif item.gold_category == "Harmless" and item.pred_category in HARMFUL_CATEGORIES:
        return RewardBreakdown(
            total=HARMLESS_FALSE_POSITIVE_REWARD,
            task=HARMLESS_FALSE_POSITIVE_REWARD,
            reason="harmless_false_positive",
        )
    else:
        return RewardBreakdown(
            total=HARMFUL_CATEGORY_CONFUSION_REWARD,
            task=HARMFUL_CATEGORY_CONFUSION_REWARD,
            reason="harmful_category_confusion",
        )

    return _with_route_terms(item, task, exact=exact, reason="exact")


def action_from_prediction(category: str | None, severity: str | None) -> str | None:
    if category == "Harmless" and severity == "L0":
        return "ALLOW"
    if category in HARMFUL_CATEGORIES and severity == "L1":
        return "REVIEW"
    if category in HARMFUL_CATEGORIES and severity == "L2":
        return "BLOCK"
    return None


def _with_route_terms(item: RewardInput, task: float, *, exact: bool, reason: str) -> RewardBreakdown:
    route = 0.0
    if item.valid_direct_answer and not item.framework_rescued:
        route = ROUTE_BONUS
    elif item.model_initiated_joint_scan:
        route = ROUTE_BONUS

    expand = TEMPORAL_BONUS if (
        item.model_initiated_temporal_expand
        and item.teacher_state_requires_temporal_context
        and item.temporal_result_was_consumed
    ) else 0.0
    grounding = GROUNDING_BONUS if item.evidence_is_grounded else 0.0

    if item.num_tool_calls < 0 or item.num_extra_frames < 0:
        raise ValueError("tool and extra-frame counts must be non-negative")
    cost = min(MAX_COST, TOOL_COST * item.num_tool_calls + EXTRA_FRAME_COST * item.num_extra_frames)
    total = task + route + expand + grounding - cost
    return RewardBreakdown(
        total=round(total, 12),
        task=task,
        route=route,
        expand=expand,
        grounding=grounding,
        cost=cost,
        exact=exact,
        reason=reason,
    )


def _timestamps(value) -> list[float]:
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        raw = item.get("timestamp") if isinstance(item, dict) else item
        try:
            output.append(float(raw))
        except (TypeError, ValueError):
            pass
    return output


def _three_risks_absent(transcript: list[dict]) -> bool:
    for step in transcript:
        decision = step.get("decision") if isinstance(step, dict) else None
        if not isinstance(decision, dict):
            continue
        # v1 persisted the post-tool assessment as ASSESS_RISKS and let the
        # framework materialize Harmless.  In v2 the model owns the post-scan
        # action and can return ANSWER(Harmless) together with the same three
        # explicit ABSENT assessments.  Both are equally grounded; only the
        # provenance and assessments matter here.
        if decision.get("action") not in {"ASSESS_RISKS", "ANSWER"}:
            continue
        assessments = decision.get("assessments") or {}
        if isinstance(assessments, list):
            assessments = {
                item.get("category") or item.get("risk"): item
                for item in assessments if isinstance(item, dict)
            }
        if all(
            str((assessments.get(risk) or {}).get("status") or "").upper() == "ABSENT"
            for risk in HARMFUL_CATEGORIES
        ):
            return True
    return False


def _valid_joint_scan(decision: dict) -> bool:
    if not isinstance(decision, dict) or decision.get("action") != "JOINT_SEMANTIC_SCAN":
        return False
    value = decision.get("queries_by_risk")
    if not isinstance(value, dict):
        return False
    for risk in HARMFUL_CATEGORIES:
        raw = value.get(risk)
        if raw is None:
            raw = value.get(risk.lower().replace(" & ", "_").replace(" ", "_"))
        values = raw if isinstance(raw, list) else [raw] if raw else []
        if not any(str(item or "").strip() for item in values):
            return False
    return True


def reward_input_from_record(meta: dict, record: dict) -> RewardInput:
    """Convert a persisted Agent trajectory into the frozen reward schema.

    Model provenance is fail-closed. In particular, legacy/current runners that
    schedule ``TEMPORAL_EXPAND`` in Python do not receive the autonomous expand
    bonus merely because the action appears in the transcript.
    """
    result = record.get("result") or {}
    transcript = result.get("transcript") or []
    final = result.get("final_action") or {}
    pred_category = record.get("pred") or final.get("category")
    pred_severity = record.get("severity") or final.get("severity")
    decisions = [
        step.get("decision") for step in transcript
        if isinstance(step, dict) and isinstance(step.get("decision"), dict)
    ]
    first_model = next((decision for decision in decisions if not decision.get("automatic")), {})
    framework_rescued = any(
        decision.get("automatic") and decision.get("action") == "JOINT_SEMANTIC_SCAN"
        for decision in decisions
    )
    model_joint = _valid_joint_scan(first_model) and not framework_rescued
    model_temporal = any(
        decision.get("action") == "TEMPORAL_EXPAND"
        and (
            decision.get("provenance") == "model"
            or decision.get("model_initiated") is True
        )
        for decision in decisions
    )
    initial_times = []
    observed_times = []
    temporal_times = []
    for step in transcript:
        if not isinstance(step, dict):
            continue
        observation = step.get("observation")
        if not step.get("tool") and isinstance(observation, dict) and observation.get("timestamps") is not None:
            values = _timestamps(observation.get("timestamps"))
            if not initial_times:
                initial_times = values
            observed_times.extend(values)
        elif step.get("tool"):
            if isinstance(observation, dict):
                values = _timestamps(observation.get("timestamps"))
            else:
                values = _timestamps(observation)
            observed_times.extend(values)
            if step.get("tool") == "TEMPORAL_EXPAND":
                temporal_times.extend(values)
    final_evidence = final.get("evidence") or []
    evidence_times = _timestamps(final_evidence)
    if pred_category == "Harmless":
        grounded = _three_risks_absent(transcript)
    else:
        grounded = bool(evidence_times) and all(
            any(abs(cited - shown) <= 0.5 for shown in observed_times)
            for cited in evidence_times
        )
    temporal_consumed = any(
        abs(cited - shown) <= 0.5
        for cited in evidence_times
        for shown in temporal_times
    ) or (pred_category == "Harmless" and bool(temporal_times) and grounded)
    first_evidence_times = _timestamps(first_model.get("evidence") or [])
    strength = str(first_model.get("evidence_strength") or "").lower()
    valid_direct = (
        first_model.get("action") == "ANSWER"
        and first_model.get("category") in HARMFUL_CATEGORIES
        and first_model.get("severity") == "L2"
        and strength in {"clear", "strong", "unambiguous"}
        and bool(first_evidence_times)
        and all(any(abs(cited - shown) <= 0.5 for shown in initial_times) for cited in first_evidence_times)
        and not framework_rescued
    )
    total_frames = result.get("total_frames")
    initial_count = len(initial_times) or 4
    extra_frames = max(int(total_frames) - initial_count, 0) if total_frames is not None else 0
    tool_calls = record.get("tool_calls") or [
        step.get("tool") for step in transcript if isinstance(step, dict) and step.get("tool")
    ]
    has_final = pred_category in VALID_CATEGORIES and pred_severity in {"L0", "L1", "L2"}
    prior_signal = meta.get("v3_prior_signal") or {}
    # V3 mines a stronger Train-only counterfactual label than the historical
    # Teacher action alone: if a prior real-tool rollout was correct only with
    # temporal context (jointly or at category level), expansion is known to
    # be useful for this source.  A prior temporal-harm signal deliberately
    # does not qualify.  The +0.2 bonus remains fail-closed because the current
    # rollout must still initiate the action itself, consume its frames, and
    # finish exactly correct.
    reference_requires_temporal = (
        "TEMPORAL_EXPAND" in (meta.get("teacher_tool_calls") or [])
        or bool(prior_signal.get("prior_temporal_joint_help"))
        or bool(prior_signal.get("prior_temporal_category_help"))
    )
    return RewardInput(
        gold_category=meta.get("category") or record.get("label"),
        gold_severity=meta.get("severity") or record.get("label_severity"),
        pred_category=pred_category,
        pred_severity=pred_severity,
        valid_json=has_final,
        has_final_answer=has_final,
        infrastructure_error=bool(record.get("error")),
        valid_direct_answer=valid_direct,
        framework_rescued=framework_rescued,
        model_initiated_joint_scan=model_joint,
        model_initiated_temporal_expand=model_temporal,
        teacher_state_requires_temporal_context=reference_requires_temporal,
        temporal_result_was_consumed=temporal_consumed,
        evidence_is_grounded=grounded,
        num_tool_calls=len(tool_calls),
        num_extra_frames=extra_frames,
        severity_mode=meta.get("severity_mode") or "clean",
        target_action=meta.get("target_action"),
        acceptable_actions=tuple(meta.get("acceptable_actions") or ()),
        acceptable_severities=tuple(meta.get("acceptable_severities") or ()),
    )
