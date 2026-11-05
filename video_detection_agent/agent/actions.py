from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    ANSWER = "ANSWER"
    SAMPLE_GLOBAL = "SAMPLE_GLOBAL"
    SEMANTIC_LOCATE = "SEMANTIC_LOCATE"
    TEMPORAL_EXPAND = "TEMPORAL_EXPAND"
    MARK_NO_EVIDENCE = "MARK_NO_EVIDENCE"


MODERATION_ACTION_BY_SEVERITY = {
    "L0": "ALLOW",
    "L1": "REVIEW",
    "L2": "BLOCK",
}


@dataclass
class AgentAction:
    action: ActionType
    category: str | None = None
    severity: str | None = None
    risk_type: str | None = None
    query: str | list[str] | None = None
    timestamps: list[float] | None = None
    window: float | None = None
    num_frames: int | None = None
    rationale: str | None = None
    evidence: list[dict] | None = None
    evidence_strength: str | None = None
    evidence_basis: str | None = None
    identified_figure: str | None = None

    def to_dict(self) -> dict:
        result = {"action": self.action.value}
        for key in ("category", "severity", "risk_type", "query", "timestamps", "window", "num_frames", "rationale", "evidence", "evidence_strength", "evidence_basis", "identified_figure"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.severity in MODERATION_ACTION_BY_SEVERITY:
            result["moderation_action"] = MODERATION_ACTION_BY_SEVERITY[self.severity]
        return result
