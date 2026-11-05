from .types import ClassificationResult, LocatedFrame, VideoFrame


class MockClassifier:
    """Deterministic backend for an end-to-end control-flow smoke test."""

    def classify(self, frames: list[VideoFrame]) -> ClassificationResult:
        if len(frames) <= 16:
            return ClassificationResult("Harmless", {"Harmless": .62, "Sexual Content": .08, "Violence & Dangerous": .25, "Sensitive Figures & Ideologies": .05})
        return ClassificationResult("Violence & Dangerous", {"Harmless": .04, "Sexual Content": .03, "Violence & Dangerous": .91, "Sensitive Figures & Ideologies": .02})


class MockLocator:
    def __call__(self, video_path: str, query: str | list[str], top_k: int,
                 risk_type: str | None = None) -> list[LocatedFrame]:
        return [LocatedFrame(timestamp=2.0 + i, score=.9 - i * .05) for i in range(top_k)]
