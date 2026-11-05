from dataclasses import dataclass
from typing import Callable, Protocol

from .types import AgentResult, ClassificationResult, LocatedFrame, VideoFrame
from .video import uniform_frames


class Classifier(Protocol):
    def classify(self, frames: list[VideoFrame]) -> ClassificationResult: ...


class Locator(Protocol):
    def __call__(self, video_path: str, query: str | list[str], top_k: int,
                 risk_type: str | None = None) -> list[LocatedFrame]: ...


class Expander(Protocol):
    def __call__(self, video_path: str, timestamps: list[float], window: float, num_frames: int) -> list[VideoFrame]: ...


@dataclass
class AgentConfig:
    initial_frames: int = 4
    confidence_threshold: float = 0.90
    margin_threshold: float = 0.15
    harmless_risk_threshold: float = 0.55
    top_k: int = 5
    window: float = 5.0
    expansion_frames: int = 16
    risk_probe_query: str = "weapon, fighting, blood, nudity, sexual act, dangerous action, cult symbol, extremist propaganda, famous political leader, racial discrimination"


def default_query(result: ClassificationResult) -> str:
    ranked = sorted(result.probabilities, key=result.probabilities.get, reverse=True)[:2]
    hints = {
        "Harmless": "weapon, fighting, blood, nudity, sexual act, dangerous action, cult symbol, extremist propaganda, famous political leader, racial discrimination",
        "Sexual Content": "nudity, sexualized pose, sexual act, explicit body exposure",
        "Violence & Dangerous": "weapon, knife, fighting, assault, blood, injury, dangerous action",
        "Sensitive Figures & Ideologies": "cult symbol, extremist propaganda, famous political leader, racial discrimination symbol or text",
    }
    return ", ".join(dict.fromkeys(hints.get(name, name) for name in ranked))


class RuleBasedVideoSafetyAgent:
    def __init__(
        self,
        classifier: Classifier,
        locator: Locator,
        expander: Expander,
        config: AgentConfig | None = None,
        query_generator: Callable[[ClassificationResult], str] = default_query,
    ):
        self.classifier = classifier
        self.locator = locator
        self.expander = expander
        self.config = config or AgentConfig()
        self.query_generator = query_generator

    def run(self, video_path: str) -> AgentResult:
        cfg = self.config
        initial_frames = uniform_frames(video_path, cfg.initial_frames)
        initial = self.classifier.classify(initial_frames)
        reasons = []
        if initial.confidence < cfg.confidence_threshold:
            reasons.append("low_confidence")
        if initial.margin < cfg.margin_threshold:
            reasons.append("low_margin")

        located: list[LocatedFrame] = []
        query = None
        if initial.pred == "Harmless":
            located = self.locator(video_path, cfg.risk_probe_query, cfg.top_k)
            if located and located[0].score >= cfg.harmless_risk_threshold:
                reasons.append("harmless_but_high_risk_retrieval")

        if not reasons:
            return AgentResult(
                initial=initial, final=initial, triggered=False,
                initial_timestamps=[frame.timestamp for frame in initial_frames], located=located,
            )

        query = self.query_generator(initial)
        if not located or query != cfg.risk_probe_query:
            located = self.locator(video_path, query, cfg.top_k)
        added = self.expander(
            video_path, [item.timestamp for item in located], cfg.window,
            cfg.expansion_frames + len(initial_frames),
        )
        initial_times = [frame.timestamp for frame in initial_frames]
        added = [
            frame for frame in added
            if all(abs(frame.timestamp - timestamp) > 1e-3 for timestamp in initial_times)
        ]
        if len(added) > cfg.expansion_frames:
            indices = [min(int((index + 0.5) * len(added) / cfg.expansion_frames), len(added) - 1)
                       for index in range(cfg.expansion_frames)]
            added = [added[index] for index in indices]
        final = self.classifier.classify(initial_frames + added) if added else initial
        return AgentResult(
            initial=initial, final=final, triggered=True, trigger_reasons=reasons,
            evidence_query=query, located=located,
            initial_timestamps=[frame.timestamp for frame in initial_frames],
            added_timestamps=[frame.timestamp for frame in added],
        )
