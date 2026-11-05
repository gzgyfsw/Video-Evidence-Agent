from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .actions import ActionType, AgentAction
from .core import Expander, Locator
from .qwen_backend import CATEGORIES
from .risk_queries import risk_queries
from .types import LocatedFrame, VideoFrame
from .tools import MotionEventLocator
from .video import extract_frames, uniform_frames, video_duration


HARMFUL_CATEGORIES = (
    "Sexual Content",
    "Violence & Dangerous",
    "Sensitive Figures & Ideologies",
)


SPECIFIC_INTIMATE_AREA_TERMS = (
    "breast", "nipple", "buttock", "crotch", "genital", "groin", "thigh",
    "胸部", "乳房", "乳头", "臀部", "屁股", "裆部", "胯部", "生殖器", "阴部", "大腿",
)


def has_specific_intimate_area_evidence(evidence: list[dict]) -> bool:
    """Reject generic claims such as 'sexualized framing' or 'body curves'."""
    descriptions = " ".join(str(item.get("description", "")).lower() for item in evidence)
    return any(term in descriptions for term in SPECIFIC_INTIMATE_AREA_TERMS)


class VisualPolicy(Protocol):
    def decide(self, frames: list[VideoFrame], state: dict,
               allowed_actions: list[ActionType]) -> AgentAction: ...


@dataclass
class PolicyAgentConfig:
    # A fixed sparse first look makes the controller's later tool decisions
    # meaningful for both short and long videos.
    initial_frames: int | None = 4
    short_video_seconds: float = 10.0
    medium_video_seconds: float = 60.0
    long_video_seconds: float = 300.0
    short_initial_frames: int = 4
    medium_initial_frames: int = 8
    long_initial_frames: int = 12
    very_long_initial_frames: int = 16
    top_k: int = 5
    default_window: float = 5.0
    default_action_frames: int = 8
    max_frames_per_action: int = 16
    max_total_frames: int = 64
    max_global_sample_calls: int = 2
    max_locate_calls: int = 3
    max_expand_calls: int = 3
    max_steps: int = 16
    motion_top_k: int = 3
    motion_window: float = 4.0
    motion_action_frames: int = 15
    max_global_context_frames: int = 8
    suggestive_sexual_review_score: float = 0.05


@dataclass
class PolicyAgentResult:
    final_action: AgentAction
    transcript: list[dict] = field(default_factory=list)
    initial_timestamps: list[float] = field(default_factory=list)
    added_timestamps: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "final_action": self.final_action.to_dict(),
            "transcript": self.transcript,
            "initial_timestamps": self.initial_timestamps,
            "added_timestamps": self.added_timestamps,
            "total_frames": len(self.initial_timestamps) + len(self.added_timestamps),
        }


class AgentRunError(RuntimeError):
    def __init__(self, message: str, transcript: list[dict], initial: list[VideoFrame],
                 added_timestamps: list[float]):
        super().__init__(message)
        self.transcript = transcript
        self.initial_timestamps = [frame.timestamp for frame in initial]
        self.added_timestamps = added_timestamps

    def to_dict(self) -> dict:
        return {
            "error": str(self), "transcript": self.transcript,
            "initial_timestamps": self.initial_timestamps,
            "added_timestamps": self.added_timestamps,
            "total_frames": len(self.initial_timestamps) + len(self.added_timestamps),
        }


class ToolCallingVideoSafetyAgent:
    """Visual evidence-driven, budgeted tool loop without confidence or REVIEW."""

    def __init__(self, policy: VisualPolicy, locator: Locator, expander: Expander,
                 config: PolicyAgentConfig | None = None,
                 motion_locator: MotionEventLocator | None = None):
        self.policy = policy
        self.locator = locator
        self.expander = expander
        self.config = config or PolicyAgentConfig()
        self.motion_locator = motion_locator

    @staticmethod
    def _evenly_limit(frames: list[VideoFrame], count: int) -> list[VideoFrame]:
        if len(frames) <= count:
            return frames
        indices = [min(int((index + 0.5) * len(frames) / count), len(frames) - 1)
                   for index in range(count)]
        return [frames[index] for index in indices]

    def _initial_frame_count(self, duration: float) -> int:
        cfg = self.config
        if cfg.initial_frames is not None:
            return cfg.initial_frames
        if duration <= cfg.short_video_seconds:
            return cfg.short_initial_frames
        if duration <= cfg.medium_video_seconds:
            return cfg.medium_initial_frames
        if duration <= cfg.long_video_seconds:
            return cfg.long_initial_frames
        return cfg.very_long_initial_frames

    def run(self, video_path: str) -> PolicyAgentResult:
        cfg = self.config
        duration = video_duration(video_path)
        initial = uniform_frames(video_path, min(self._initial_frame_count(duration), cfg.max_total_frames))
        frames = list(initial)
        global_context = self._evenly_limit(list(initial), cfg.max_global_context_frames)
        added_timestamps: list[float] = []
        transcript: list[dict] = [{"observation": {
            "duration": duration, "frames_seen": len(frames),
            "timestamps": [frame.timestamp for frame in frames],
            "initial_sampling": "uniform_4_non_black" if cfg.initial_frames == 4 else (
                "manual" if cfg.initial_frames is not None else "duration_adaptive"
            ),
        }}]
        risk_checks = {category: "not_checked" for category in HARMFUL_CATEGORIES}
        located: list[LocatedFrame] = []
        located_risk: str | None = None
        pending_checked_risk: str | None = None
        forced_risk_type: str | None = None
        locator_max_scores: dict[str, float] = {}
        phase = "initial"
        global_calls = locate_calls = expand_calls = 0

        for _ in range(cfg.max_steps):
            acquired_count = len(initial) + len(added_timestamps)
            remaining_frames = max(cfg.max_total_frames - acquired_count, 0)
            sorted_times = sorted(frame.timestamp for frame in frames)
            gaps = [b - a for a, b in zip(sorted_times, sorted_times[1:])]
            state = {
                "phase": phase,
                "duration": round(duration, 3),
                "video_length_bucket": (
                    "short" if duration <= cfg.short_video_seconds else
                    "medium" if duration <= cfg.medium_video_seconds else
                    "long" if duration <= cfg.long_video_seconds else "very_long"
                ),
                "frames_seen": len(frames),
                "total_frames_acquired": acquired_count,
                "frame_timestamps": [round(frame.timestamp, 3) for frame in frames],
                "approx_global_stride": round(duration / max(len(frames), 1), 3),
                "max_observed_gap": round(max(gaps) if gaps else duration, 3),
                "frame_budget_remaining": remaining_frames,
                "max_frames_per_action": cfg.max_frames_per_action,
                "global_sample_calls_remaining": cfg.max_global_sample_calls - global_calls,
                "locate_calls_remaining": cfg.max_locate_calls - locate_calls,
                "expand_calls_remaining": cfg.max_expand_calls - expand_calls,
                "risk_checks": risk_checks,
                "harmless_allowed": all(value == "no_evidence" for value in risk_checks.values()),
                "unchecked_risks": [name for name, value in risk_checks.items()
                                    if value != "no_evidence"],
                "required_risk_type": forced_risk_type,
                "inspected_risk_type": (
                    pending_checked_risk if phase == "risk_inspected" else None
                ),
                "located_risk_type": located_risk if phase == "located" else None,
                "locator_candidates": (
                    [{"timestamp": round(item.timestamp, 3), "score": round(float(item.score), 6)}
                     for item in located]
                    if phase == "located" else []
                ),
                "tool_history": transcript,
            }
            state["allowed_answer_categories"] = (
                list(CATEGORIES) if state["harmless_allowed"] else list(HARMFUL_CATEGORIES)
            )
            if phase == "risk_inspected":
                allowed = [ActionType.ANSWER]
                if pending_checked_risk is not None:
                    allowed.append(ActionType.MARK_NO_EVIDENCE)
            elif phase == "located":
                # Retrieval produces search leads, not visible evidence. The
                # policy decides whether and how much temporal context to inspect.
                allowed = [ActionType.ANSWER]
                if located and remaining_frames > 0 and expand_calls < cfg.max_expand_calls:
                    allowed.append(ActionType.TEMPORAL_EXPAND)
            elif state["harmless_allowed"]:
                allowed = [ActionType.ANSWER]
            elif forced_risk_type is not None:
                allowed = [ActionType.SEMANTIC_LOCATE]
            else:
                allowed = [ActionType.ANSWER]
                if locate_calls < cfg.max_locate_calls:
                    allowed.append(ActionType.SEMANTIC_LOCATE)
                if (remaining_frames > 0
                        and global_calls < cfg.max_global_sample_calls):
                    allowed.append(ActionType.SAMPLE_GLOBAL)

            action = self.policy.decide(frames, state, allowed)
            transcript.append({"decision": action.to_dict()})
            if action.action not in allowed:
                transcript.append({"rejected": {
                    "reason": "action is invalid in the current phase",
                    "action": action.action.value,
                    "allowed": [item.value for item in allowed],
                }})
                continue

            if action.action == ActionType.ANSWER:
                if action.category not in CATEGORIES:
                    transcript.append({"rejected": "ANSWER requires a valid category"})
                    continue
                if action.severity not in {"L0", "L1", "L2"}:
                    transcript.append({"rejected": "ANSWER requires severity L0, L1, or L2"})
                    continue
                if (action.category != "Harmless" and action.severity == "L0"
                        and not state["harmless_allowed"]):
                    # Some controllers express "this remaining risk is absent"
                    # as Risk-L0. Treat that as a request to inspect that risk,
                    # never as a final harmful label or a repeated rejection.
                    candidate = (action.category if action.category in state["unchecked_risks"]
                                 else state["unchecked_risks"][0])
                    forced_risk_type = candidate
                    transcript.append({"controller_transition": {
                        "from": f"ANSWER({action.category}, L0)",
                        "to": "REQUIRE_SEMANTIC_CHECK",
                        "risk_type": forced_risk_type,
                    }})
                    continue
                if ((action.category == "Harmless" and action.severity != "L0")
                        or (action.category != "Harmless" and action.severity == "L0")):
                    transcript.append({"rejected": {
                        "reason": "Harmless requires L0; harmful categories require L1-L2",
                        "category": action.category,
                        "severity": action.severity,
                    }})
                    continue
                if action.category == "Harmless" and not state["harmless_allowed"]:
                    if phase == "risk_inspected" and pending_checked_risk is not None:
                        action = AgentAction(
                            ActionType.MARK_NO_EVIDENCE,
                            risk_type=pending_checked_risk,
                            rationale=action.rationale,
                        )
                        transcript.append({"controller_transition": {
                            "from": "ANSWER(Harmless)",
                            "to": "MARK_NO_EVIDENCE",
                            "risk_type": pending_checked_risk,
                        }})
                    elif phase == "located" and located_risk is not None:
                        transcript.append({"rejected": {
                            "reason": "locator candidates are search leads; inspect them with TEMPORAL_EXPAND before excluding this risk",
                            "risk_type": located_risk,
                            "candidate_timestamps": [item.timestamp for item in located],
                        }})
                        continue
                    else:
                        forced_risk_type = next(
                            name for name, value in risk_checks.items() if value != "no_evidence"
                        )
                        transcript.append({"controller_transition": {
                            "from": "ANSWER(Harmless)",
                            "to": "REQUIRE_SEMANTIC_CHECK",
                            "risk_type": forced_risk_type,
                        }})
                        continue
                else:
                    if action.category != "Harmless":
                        visible = [frame.timestamp for frame in frames]
                        validated = []
                        for item in action.evidence or []:
                            timestamp = float(item.get("timestamp", -1.0))
                            nearest = min(visible, key=lambda known: abs(known - timestamp))
                            description = str(item.get("description", "")).strip()
                            if abs(nearest - timestamp) <= 0.26 and description:
                                validated.append({"timestamp": nearest, "description": description})
                        strength_allowed = (
                            action.evidence_strength == "clear"
                            or (action.severity == "L1"
                                and action.evidence_strength == "ambiguous")
                        )
                        if not strength_allowed or not validated:
                            transcript.append({"rejected": {
                                "reason": "harmful ANSWER requires timestamped evidence; only L1 may use ambiguous evidence_strength",
                                "evidence_strength": action.evidence_strength,
                                "visible_timestamps": visible,
                            }})
                            continue
                        if action.category == "Sexual Content":
                            allowed_bases = {
                                "revealing_attire", "nudity", "sexual_action", "provocative_pose",
                                "intimate_area_emphasis", "fetishized_presentation",
                            }
                            if action.evidence_basis not in allowed_bases:
                                transcript.append({"rejected": {
                                    "reason": "Sexual Content requires a valid concrete evidence_basis",
                                    "evidence_basis": action.evidence_basis,
                                }})
                                continue
                            if (action.evidence_basis == "revealing_attire"
                                    and action.severity != "L1"):
                                transcript.append({"rejected": {
                                    "reason": "revealing_attire is a boundary basis valid only for L1",
                                    "severity": action.severity,
                                }})
                                continue
                            if (action.evidence_basis == "intimate_area_emphasis"
                                    and not has_specific_intimate_area_evidence(validated)):
                                transcript.append({"rejected": {
                                    "reason": "intimate_area_emphasis requires a specifically named visible intimate area",
                                    "evidence": validated,
                                }})
                                continue
                            suggestive_bases = {
                                "provocative_pose", "intimate_area_emphasis",
                                "fetishized_presentation",
                            }
                            locator_score = locator_max_scores.get("Sexual Content")
                            if (action.evidence_basis in suggestive_bases
                                    and locator_score is not None
                                    and locator_score < cfg.suggestive_sexual_review_score):
                                # SigLIP is a ranking tool, not a calibrated
                                # classifier. Low support marks the trajectory
                                # for teacher review but must not veto visible
                                # evidence or create an ANSWER/reject loop.
                                transcript.append({"quality_flag": {
                                    "reason": "low_suggestive_sexual_locator_support",
                                    "evidence_basis": action.evidence_basis,
                                    "locator_max_score": locator_score,
                                    "review_score": cfg.suggestive_sexual_review_score,
                                }})
                        if action.category == "Sensitive Figures & Ideologies":
                            sensitive_bases = {
                                "named_prominent_figure", "extremist_content",
                                "identity_discrimination", "cult_content",
                            }
                            if action.evidence_basis not in sensitive_bases:
                                transcript.append({"rejected": {
                                    "reason": "Sensitive answer requires a valid evidence_basis",
                                    "evidence_basis": action.evidence_basis,
                                }})
                                continue
                            if action.evidence_basis == "named_prominent_figure":
                                name = (action.identified_figure or "").strip()
                                generic = {
                                    "unknown", "unspecified", "political leader",
                                    "politician", "public figure", "government official",
                                }
                                if not name or name.lower() in generic:
                                    transcript.append({"rejected": {
                                        "reason": "political-figure evidence must name the identified person",
                                        "identified_figure": action.identified_figure,
                                    }})
                                    continue
                                identity_text = " ".join(
                                    [action.rationale or ""]
                                    + [item["description"] for item in validated]
                                ).lower()
                                resemblance_terms = {
                                    "相似", "类似", "疑似", "可能是", "看起来像",
                                    "resembles", "resemblance", "looks like", "possibly",
                                    "appears to be", "similar to",
                                }
                                if any(term in identity_text for term in resemblance_terms):
                                    transcript.append({"rejected": {
                                        "reason": "named political identity cannot be based on resemblance or guessing",
                                        "identified_figure": action.identified_figure,
                                    }})
                                    continue
                        action.evidence = validated
                    return PolicyAgentResult(action, transcript, [f.timestamp for f in initial], added_timestamps)

            if action.action == ActionType.SAMPLE_GLOBAL:
                requested = action.num_frames or cfg.default_action_frames
                count = min(max(int(requested), 1), cfg.max_frames_per_action, remaining_frames)
                candidates = uniform_frames(video_path, acquired_count + count)
                known = [frame.timestamp for frame in initial] + added_timestamps
                added = [frame for frame in candidates
                         if all(abs(frame.timestamp - timestamp) > 1e-3 for timestamp in known)]
                added = self._evenly_limit(added, count)
                frames.extend(added)
                frames.sort(key=lambda frame: frame.timestamp)
                added_timestamps.extend(frame.timestamp for frame in added)
                global_context = self._evenly_limit(frames, cfg.max_global_context_frames)
                frames = list(global_context)
                global_calls += 1
                transcript.append({"tool": "SAMPLE_GLOBAL", "observation": {
                    "requested_frames": requested, "added_frames": len(added),
                    "timestamps": [frame.timestamp for frame in added],
                }})
                phase = "global_sampled"
                continue

            if action.action == ActionType.SEMANTIC_LOCATE:
                if action.risk_type not in HARMFUL_CATEGORIES or not action.query:
                    transcript.append({"rejected": "SEMANTIC_LOCATE requires harmful risk_type and query"})
                    continue
                if risk_checks[action.risk_type] == "no_evidence":
                    transcript.append({"rejected": "risk_type was already checked with no evidence"})
                    continue
                if forced_risk_type is not None and action.risk_type != forced_risk_type:
                    transcript.append({"rejected": {
                        "reason": "SEMANTIC_LOCATE must check required_risk_type",
                        "required_risk_type": forced_risk_type,
                        "received_risk_type": action.risk_type,
                    }})
                    continue
                located_risk = action.risk_type
                pending_checked_risk = None
                forced_risk_type = None
                queries = risk_queries(action.risk_type, action.query)
                located = self.locator(
                    video_path, queries, cfg.top_k, risk_type=action.risk_type
                )
                locator_max_scores[action.risk_type] = max(
                    (float(item.score) for item in located), default=0.0
                )
                locate_calls += 1
                transcript.append({"tool": "SEMANTIC_LOCATE", "risk_type": located_risk,
                                   "queries": queries,
                                   "observation": [vars(item) for item in located]})
                motion_events: list[LocatedFrame] = []
                if located_risk == "Violence & Dangerous" and self.motion_locator is not None:
                    motion_events = self.motion_locator(
                        video_path, cfg.motion_top_k, risk_type=located_risk
                    )
                    transcript.append({"tool": "MOTION_LOCATE", "risk_type": located_risk,
                                       "observation": [vars(item) for item in motion_events]})
                # Keep all retrieval candidates available for the next policy
                # step. Qwen, rather than Python, now schedules TEMPORAL_EXPAND.
                if motion_events:
                    located.extend(motion_events)
                phase = "located"
                continue

            if action.action == ActionType.TEMPORAL_EXPAND:
                if not located or located_risk is None:
                    transcript.append({"rejected": "TEMPORAL_EXPAND requires locator results"})
                    continue
                available = [item.timestamp for item in located]
                timestamps = action.timestamps or available
                timestamps = [t for t in timestamps if any(abs(t - known) < 0.25 for known in available)] or available
                window = min(max(float(action.window or cfg.default_window), 0.5), 15.0)
                requested = action.num_frames or cfg.default_action_frames
                count = min(max(int(requested), 1), cfg.max_frames_per_action, remaining_frames)
                expanded = self.expander(video_path, timestamps, window, count + len(frames))
                known = [frame.timestamp for frame in frames]
                expanded = [frame for frame in expanded
                            if all(abs(frame.timestamp - timestamp) > 1e-3 for timestamp in known)]
                expanded = self._evenly_limit(expanded, count)
                frames.extend(expanded)
                frames.sort(key=lambda frame: frame.timestamp)
                added_timestamps.extend(frame.timestamp for frame in expanded)
                expand_calls += 1
                pending_checked_risk = located_risk
                transcript.append({"tool": "TEMPORAL_EXPAND", "risk_type": located_risk,
                                   "observation": {"requested_frames": requested,
                                                   "added_frames": len(expanded),
                                                   "timestamps": [f.timestamp for f in expanded]}})
                located = []
                located_risk = None
                phase = "risk_inspected"
                continue

            if action.action == ActionType.MARK_NO_EVIDENCE:
                if action.risk_type not in HARMFUL_CATEGORIES:
                    transcript.append({"rejected": "MARK_NO_EVIDENCE requires a harmful risk_type"})
                    continue
                if pending_checked_risk is not None and action.risk_type != pending_checked_risk:
                    transcript.append({"rejected": "MARK_NO_EVIDENCE must match the just-inspected risk"})
                    continue
                if pending_checked_risk is None:
                    transcript.append({"rejected": "MARK_NO_EVIDENCE requires semantic retrieval and temporal inspection"})
                    continue
                if risk_checks[action.risk_type] == "no_evidence":
                    transcript.append({"rejected": "risk_type was already checked with no evidence"})
                    continue
                risk_checks[action.risk_type] = "no_evidence"
                transcript.append({"tool": "MARK_NO_EVIDENCE", "risk_type": action.risk_type,
                                   "basis": "targeted_inspection"})
                pending_checked_risk = None
                located = []
                located_risk = None
                frames = list(global_context)
                phase = "risk_checked"

        raise AgentRunError(
            "visual agent exhausted its step budget without a valid final ANSWER",
            transcript, initial, added_timestamps,
        )
