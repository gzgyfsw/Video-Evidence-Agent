from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict, defaultdict
from pathlib import Path
import hashlib
import os
import threading

from .types import LocatedFrame, VideoFrame
from .video import extract_frames, uniform_frames, video_duration


@dataclass
class RiskSemanticLocator:
    model_name: str = "google/siglip2-base-patch16-224"
    candidate_frames: int | None = None
    device: str | None = None
    candidate_stride_seconds: float = 2.0
    min_candidate_frames: int = 16
    max_candidate_frames: int = 256
    min_time_separation_seconds: float = 0.5

    def __post_init__(self):
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def candidate_frame_count(self, duration: float) -> int:
        if self.candidate_frames is not None:
            return self.candidate_frames
        if self.candidate_stride_seconds <= 0:
            raise ValueError("candidate_stride_seconds must be positive")
        candidate_count = int(-(-duration // self.candidate_stride_seconds))
        return min(max(candidate_count, self.min_candidate_frames), self.max_candidate_frames)

    def __call__(self, video_path: str, query: str | list[str], top_k: int = 5,
                 risk_type: str | None = None) -> list[LocatedFrame]:
        queries = [query] if isinstance(query, str) else list(query)
        queries = list(dict.fromkeys(item.strip().rstrip(". ") for item in queries if item.strip()))
        if not queries:
            raise ValueError("at least one query is required")
        self._load()
        import torch
        duration = video_duration(video_path)
        candidate_count = self.candidate_frame_count(duration)
        frames = uniform_frames(video_path, candidate_count)
        if not frames:
            return []
        text_queries = [f"This image shows {item}." for item in queries]
        inputs = self._processor(
            text=text_queries, images=[frame.image for frame in frames],
            # Dynamic queries originate from the controller.  A malformed or
            # unusually verbose query must not make the whole video fail when
            # SigLIP batches text features of different lengths.
            padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
            score_matrix = outputs.logits_per_image.sigmoid().float().cpu()
        scores, query_indices = score_matrix.max(dim=1)
        count = min(max(int(top_k), 0), len(frames))
        ranked = torch.argsort(scores, descending=True).tolist()
        selected: list[int] = []
        for index in ranked:
            if all(abs(frames[index].timestamp - frames[known].timestamp) >= self.min_time_separation_seconds
                   for known in selected):
                selected.append(index)
            if len(selected) == count:
                break
        # Very short videos may not contain enough separated timestamps.
        if len(selected) < count:
            selected.extend(index for index in ranked if index not in selected)
            selected = selected[:count]
        return [
            LocatedFrame(
                timestamp=frames[index].timestamp,
                score=float(scores[index]),
                matched_query=queries[int(query_indices[index])],
                risk_type=risk_type,
            )
            for index in selected
        ]


@dataclass
class JointRiskSemanticLocator(RiskSemanticLocator):
    """Encode video candidates once and retrieve top-k independently per risk."""

    embedding_cache_size: int = 32
    embedding_cache_dir: str | None = None

    def __post_init__(self):
        super().__post_init__()
        self._embedding_cache: OrderedDict[tuple, object] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._inference_lock = threading.RLock()

    def _disk_embedding_path(self, key: tuple) -> Path | None:
        if not self.embedding_cache_dir:
            return None
        digest = hashlib.sha256(repr((self.model_name, key)).encode("utf-8")).hexdigest()
        return Path(self.embedding_cache_dir) / f"{digest}.pt"

    def _candidate_embeddings(self, video_path: str):
        """Return normalized image embeddings, cached across rollout siblings."""
        import torch

        path = Path(video_path)
        stat = path.stat()
        duration = video_duration(video_path)
        candidate_count = self.candidate_frame_count(duration)
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, candidate_count)
        owner = False
        with self._cache_lock:
            future = self._embedding_cache.get(key)
            if future is None:
                from concurrent.futures import Future
                future = Future()
                self._embedding_cache[key] = future
                owner = True
            self._embedding_cache.move_to_end(key)
        if not owner:
            return future.result()
        try:
            disk_path = self._disk_embedding_path(key)
            value = None
            if disk_path is not None and disk_path.is_file():
                saved = torch.load(disk_path, map_location="cpu", weights_only=True)
                saved_times = saved.get("timestamps") if isinstance(saved, dict) else None
                saved_embeddings = saved.get("embeddings") if isinstance(saved, dict) else None
                if (isinstance(saved_times, list) and isinstance(saved_embeddings, torch.Tensor)
                        and saved_embeddings.ndim == 2
                        and len(saved_times) == saved_embeddings.shape[0]):
                    value = ([float(item) for item in saved_times], saved_embeddings.float())
            if value is None:
                frames = uniform_frames(video_path, candidate_count)
                if not frames:
                    value = ([], None)
                else:
                # Video decoding happens outside the model lock so another
                # source can prepare its candidates while the GPU is busy.
                    with self._inference_lock:
                        image_inputs = self._processor(
                            images=[frame.image for frame in frames], return_tensors="pt",
                        ).to(self.device)
                        image_kwargs = {
                            name: tensor for name, tensor in image_inputs.items()
                            if name in {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
                        }
                        with torch.inference_mode():
                            output = self._model.get_image_features(**image_kwargs)
                            embeddings = output.pooler_output.float()
                            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
                            embeddings = embeddings.cpu()
                    value = ([float(frame.timestamp) for frame in frames], embeddings)
                if disk_path is not None and value[1] is not None:
                    disk_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = disk_path.with_suffix(
                        f".tmp-{os.getpid()}-{threading.get_ident()}.pt"
                    )
                    torch.save({"timestamps": value[0], "embeddings": value[1]}, temporary)
                    os.replace(temporary, disk_path)
        except BaseException as exc:
            future.set_exception(exc)
            with self._cache_lock:
                self._embedding_cache.pop(key, None)
            raise
        future.set_result(value)
        with self._cache_lock:
            self._embedding_cache.move_to_end(key)
            while len(self._embedding_cache) > max(1, self.embedding_cache_size):
                oldest_key, oldest = next(iter(self._embedding_cache.items()))
                if not oldest.done():
                    break
                self._embedding_cache.pop(oldest_key)
        return value

    def __call__(self, video_path: str, queries_by_risk: dict[str, list[str]],
                 top_k_per_risk: int = 3) -> list[LocatedFrame]:
        cleaned = {
            risk: list(dict.fromkeys(item.strip().rstrip(". ") for item in queries if item.strip()))
            for risk, queries in queries_by_risk.items()
        }
        flattened = [(risk, query) for risk, queries in cleaned.items() for query in queries]
        if not flattened:
            raise ValueError("at least one joint query is required")
        # Transformers/PyTorch model objects are shared by rollout threads.
        with self._inference_lock:
            self._load()
        import torch
        timestamps, image_embeddings = self._candidate_embeddings(video_path)
        if image_embeddings is None:
            return []
        # Serialize only model inference. Candidate decoding is coordinated by
        # an in-flight Future and can overlap for different source videos.
        with self._inference_lock:
            text_inputs = self._processor(
                text=[f"This image shows {query}." for _, query in flattened],
                # Keep controller-generated queries within SigLIP's context
                # window; otherwise one long query makes `input_ids` ragged.
                padding="max_length", truncation=True, return_tensors="pt",
            ).to(self.device)
            text_kwargs = {
                key: value for key, value in text_inputs.items()
                if key in {"input_ids", "attention_mask", "position_ids"}
            }
            with torch.inference_mode():
                text_output = self._model.get_text_features(**text_kwargs)
                text_embeddings = text_output.pooler_output.float()
                text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
                matrix = image_embeddings.to(self.device) @ text_embeddings.t()
                matrix = (
                    matrix * self._model.logit_scale.float().exp()
                    + self._model.logit_bias.float()
                ).sigmoid().cpu()
            query_columns: dict[str, list[int]] = defaultdict(list)
            for column, (risk, _) in enumerate(flattened):
                query_columns[risk].append(column)
            output: list[LocatedFrame] = []
            for risk, columns in query_columns.items():
                risk_scores, local_indices = matrix[:, columns].max(dim=1)
                ranked = torch.argsort(risk_scores, descending=True).tolist()
                selected: list[int] = []
                for index in ranked:
                    if all(abs(timestamps[index] - timestamps[known])
                           >= self.min_time_separation_seconds for known in selected):
                        selected.append(index)
                    if len(selected) == min(max(int(top_k_per_risk), 0), len(timestamps)):
                        break
                for index in selected:
                    column = columns[int(local_indices[index])]
                    output.append(LocatedFrame(
                        timestamp=timestamps[index],
                        score=float(risk_scores[index]),
                        matched_query=flattened[column][1],
                        risk_type=risk,
                    ))
            return output


@dataclass
class MotionEventLocator:
    """Cheap low-resolution scan for brief temporal changes missed by sparse VLM frames."""

    samples_per_second: float = 4.0
    min_time_separation_seconds: float = 1.5
    resize_width: int = 160
    resize_height: int = 90

    def __call__(self, video_path: str, top_k: int = 5,
                 risk_type: str | None = None) -> list[LocatedFrame]:
        if self.samples_per_second <= 0:
            raise ValueError("samples_per_second must be positive")
        import cv2
        import numpy as np

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        previous = None
        changes: list[tuple[float, float]] = []
        # Seeking separately for every 0.25 s sample is prohibitively slow on
        # long GOP videos: each seek decodes from a preceding keyframe.  Walk
        # the stream once and retain the same approximately uniform sample
        # cadence instead.  This is both faster and less prone to returning a
        # neighbouring keyframe for many requested timestamps.
        sample_stride = max(1, int(round(fps / self.samples_per_second)))
        frame_index = 0
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                if frame_index % sample_stride:
                    frame_index += 1
                    continue
                timestamp = frame_index / fps
                gray = cv2.resize(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                    (self.resize_width, self.resize_height),
                )
                if previous is not None:
                    # Consecutive retained frames are separated by the normal
                    # approximately 4 Hz detector cadence.
                    score = float(np.mean(cv2.absdiff(gray, previous)) / 255.0)
                    changes.append((float(timestamp), score))
                previous = gray
                frame_index += 1
        finally:
            capture.release()

        selected: list[tuple[float, float]] = []
        for timestamp, score in sorted(changes, key=lambda item: item[1], reverse=True):
            if all(abs(timestamp - known) >= self.min_time_separation_seconds
                   for known, _ in selected):
                selected.append((timestamp, score))
            if len(selected) == max(int(top_k), 0):
                break
        return [LocatedFrame(timestamp, score, "high temporal visual change", risk_type)
                for timestamp, score in selected]


@dataclass
class TemporalEvidenceExpander:
    def intervals(self, video_path: str, timestamps: list[float], window: float) -> list[tuple[float, float]]:
        duration = video_duration(video_path)
        raw = sorted((max(0.0, t - window), min(duration, t + window)) for t in timestamps)
        merged: list[list[float]] = []
        for start, end in raw:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return [(start, end) for start, end in merged]

    def __call__(
        self, video_path: str, timestamps: list[float], window: float = 5.0, num_frames: int = 16
    ) -> list[VideoFrame]:
        if window < 0 or num_frames <= 0:
            raise ValueError("window must be non-negative and num_frames must be positive")
        intervals = self.intervals(video_path, timestamps, window)
        if not intervals:
            return []
        lengths = [max(end - start, 1e-6) for start, end in intervals]
        total = sum(lengths)
        timestamps_out = []
        for index in range(num_frames):
            target = (index + 0.5) * total / num_frames
            elapsed = 0.0
            for (start, end), length in zip(intervals, lengths):
                if target <= elapsed + length:
                    timestamps_out.append(start + (target - elapsed) / length * (end - start))
                    break
                elapsed += length
        return extract_frames(video_path, timestamps_out)

    def bursts(
        self, video_path: str, timestamps: list[float], window: float = 2.0,
        num_frames: int = 8, max_events: int = 3,
    ) -> list[VideoFrame]:
        """Sample short dense bursts around separate events without merging them."""
        if window <= 0 or num_frames <= 0 or max_events <= 0:
            raise ValueError("window, num_frames and max_events must be positive")
        duration = video_duration(video_path)
        events = []
        for timestamp in timestamps:
            timestamp = min(max(float(timestamp), 0.0), duration)
            if all(abs(timestamp - known) >= 0.5 for known in events):
                events.append(timestamp)
            if len(events) == max_events:
                break
        if not events:
            return []
        base, remainder = divmod(num_frames, len(events))
        output = []
        for index, event in enumerate(events):
            count = base + (1 if index < remainder else 0)
            # A large change often occurs at impact or at the immediate cut after
            # an event, so retain the lead-up instead of sampling a wide symmetric interval.
            start, end = max(0.0, event - window), min(duration, event)
            if count == 1:
                output.append((start + end) / 2.0)
            else:
                output.extend(start + i * (end - start) / (count - 1) for i in range(count))
        # Overlapping lead-up windows often produce the same timestamps. Keep
        # their temporal order but never spend VLM budget on duplicate frames.
        output = list(dict.fromkeys(round(timestamp, 9) for timestamp in output))
        return extract_frames(video_path, output)


def risk_semantic_locator(video_path: str, query: str | list[str], top_k: int = 5,
                          risk_type: str | None = None, **kwargs) -> list[LocatedFrame]:
    return RiskSemanticLocator(**kwargs)(video_path, query, top_k, risk_type=risk_type)


def temporal_evidence_expander(
    video_path: str, timestamps: list[float], window: float = 5.0, num_frames: int = 16
) -> list[VideoFrame]:
    return TemporalEvidenceExpander()(video_path, timestamps, window, num_frames)
