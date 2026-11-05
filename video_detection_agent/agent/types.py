from dataclasses import dataclass, field
from typing import Any


@dataclass
class VideoFrame:
    timestamp: float
    image: Any


@dataclass
class LocatedFrame:
    timestamp: float
    score: float
    matched_query: str | None = None
    risk_type: str | None = None


@dataclass
class ClassificationResult:
    pred: str
    probabilities: dict[str, float]

    @property
    def confidence(self) -> float:
        return float(self.probabilities[self.pred])

    @property
    def margin(self) -> float:
        values = sorted(self.probabilities.values(), reverse=True)
        return float(values[0] - values[1]) if len(values) > 1 else float(values[0])


@dataclass
class AgentResult:
    initial: ClassificationResult
    final: ClassificationResult
    triggered: bool
    trigger_reasons: list[str] = field(default_factory=list)
    evidence_query: str | None = None
    located: list[LocatedFrame] = field(default_factory=list)
    initial_timestamps: list[float] = field(default_factory=list)
    added_timestamps: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        def cls(result: ClassificationResult) -> dict:
            return {
                "pred": result.pred,
                "confidence": result.confidence,
                "margin": result.margin,
                "probabilities": result.probabilities,
            }

        return {
            "initial": cls(self.initial),
            "final": cls(self.final),
            "triggered": self.triggered,
            "trigger_reasons": self.trigger_reasons,
            "evidence_query": self.evidence_query,
            "located": [vars(item) for item in self.located],
            "initial_timestamps": self.initial_timestamps,
            "added_timestamps": self.added_timestamps,
            "total_frames": len(self.initial_timestamps) + len(self.added_timestamps),
        }
