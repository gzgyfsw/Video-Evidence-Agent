import os
import shutil
import subprocess
import concurrent.futures
import json
from fractions import Fraction
from io import BytesIO
from functools import lru_cache
from pathlib import Path

from .types import VideoFrame


def is_near_black_frame(image, *, max_luminance: float = 12.0,
                        dark_pixel_limit: int = 24,
                        min_dark_fraction: float = 0.985) -> bool:
    """Return True only for a nearly blank black/fade frame.

    This deliberately does *not* reject ordinary dark/night scenes: a frame is
    skipped only when it is both very dark on average and almost all of its
    pixels are dark.  The initial observation should spend its small image
    budget on visible content, not fade-in/fade-out frames.
    """
    import numpy as np

    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if pixels.size == 0:
        return True
    peak = pixels.max(axis=2)
    return bool(pixels.mean() <= max_luminance and (peak <= dark_pixel_limit).mean() >= min_dark_fraction)


def _open_video(video_path: str):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video decoding requires opencv-python") from exc
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video metadata: fps={fps}, frames={frame_count}")
    return cv2, capture, fps, frame_count


@lru_cache(maxsize=1)
def _nvdec_ffmpeg() -> str | None:
    """Whether the system FFmpeg can decode with NVIDIA NVDEC.

    OpenCV in this environment is CPU-only even though the host FFmpeg and
    NVIDIA driver expose CUDA decoding.  Keep this probe cached and make the
    CPU path a transparent fallback for portability.
    """
    # Sparse random access (the normal 4-frame policy) is faster through the
    # CPU decoder: NVDEC needs a new FFmpeg process and keyframe seek for each
    # timestamp.  Enable it explicitly for dense/sequential extraction jobs.
    if os.environ.get("VIDEO_DECODE_BACKEND", "cpu").lower() not in {"gpu", "nvdec", "cuda"}:
        return None
    candidates = [
        os.environ.get("VIDEO_GPU_FFMPEG"),
        "/usr/bin/ffmpeg",  # Host FFmpeg is CUDA-enabled; Conda's often is not.
        shutil.which("ffmpeg"),
    ]
    for ffmpeg in dict.fromkeys(item for item in candidates if item):
        try:
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-hwaccels"], capture_output=True,
                text=True, timeout=5, check=False,
            )
            if "cuda" in result.stdout.lower().split():
                return ffmpeg
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def _nvdec_available() -> bool:
    return _nvdec_ffmpeg() is not None


def _gpu_decode_frame(video_path: str, timestamp: float, width: int, height: int):
    """Decode one random-access frame with NVDEC and return a PIL image.

    ``hwdownload`` converts the decoded CUDA surface to RGB only after NVDEC;
    no CPU codec work is performed.  The caller validates the payload and
    falls back to OpenCV for unusual codecs or FFmpeg builds.
    """
    from PIL import Image

    ffmpeg = _nvdec_ffmpeg()
    if not ffmpeg or width <= 0 or height <= 0:
        return None
    command = [
        ffmpeg, "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-ss", f"{timestamp:.6f}", "-i", video_path, "-frames:v", "1", "-an",
        "-vf", "hwdownload,format=nv12,format=rgb24", "-f", "rawvideo", "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    expected = width * height * 3
    if result.returncode or len(result.stdout) != expected:
        return None
    return Image.frombytes("RGB", (width, height), result.stdout)


@lru_cache(maxsize=1024)
def _needs_software_ffmpeg_fallback(video_path: str) -> bool:
    """OpenCV builds can open AV1/VP9 containers but fail to decode their pixels."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return False
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", video_path],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip().lower() in {"av1", "vp9"}


def _software_ffmpeg_frame(video_path: str, timestamp: float):
    """Decode a single frame without NVDEC when OpenCV lacks the video codec."""
    # The Conda FFmpeg on this host advertises AV1 but cannot decode these
    # files; the system FFmpeg can.  Allow an explicit override elsewhere.
    ffmpeg = (os.environ.get("VIDEO_CPU_FFMPEG") or
              ("/usr/bin/ffmpeg" if os.path.isfile("/usr/bin/ffmpeg") else None) or
              shutil.which("ffmpeg"))
    if not ffmpeg:
        return None
    try:
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-ss", f"{timestamp:.6f}", "-i", video_path,
             "-frames:v", "1", "-an", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
            capture_output=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode or not result.stdout:
        return None
    try:
        from PIL import Image
        with Image.open(BytesIO(result.stdout)) as image:
            return image.convert("RGB").copy()
    except (OSError, ValueError):
        return None


def _software_video_metadata(video_path: str):
    """Read stream timing without first opening a codec OpenCV may lack."""
    path=Path(video_path)
    if not path.is_file():raise FileNotFoundError(path)
    ffprobe=('/usr/bin/ffprobe' if os.path.isfile('/usr/bin/ffprobe') else shutil.which('ffprobe'))
    if not ffprobe:raise RuntimeError('FFprobe required for software-only video codec')
    result=subprocess.run([ffprobe,'-v','error','-select_streams','v:0','-show_entries',
        'stream=avg_frame_rate,r_frame_rate,nb_frames,duration,width,height','-of','json',str(path)],
        capture_output=True,text=True,timeout=10,check=True)
    stream=json.loads(result.stdout)['streams'][0];fps=0.0
    for key in ['avg_frame_rate','r_frame_rate']:
        try:fps=float(Fraction(stream.get(key,'0/1')))
        except (ValueError,ZeroDivisionError,TypeError):continue
        if fps>0:break
    try:frame_count=int(stream.get('nb_frames','0'))
    except (ValueError,TypeError):frame_count=0
    if frame_count<=0 and fps>0:
        frame_count=round(float(stream.get('duration',0))*fps)
    width=int(stream.get('width',0));height=int(stream.get('height',0))
    if fps<=0 or frame_count<=0 or width<=0 or height<=0:
        raise RuntimeError('Invalid software-only video stream metadata')
    return fps,frame_count,width,height


def _video_timing(video_path: str):
    if _needs_software_ffmpeg_fallback(video_path):
        fps,frame_count,_,_=_software_video_metadata(video_path)
    else:
        _,capture,fps,frame_count=_open_video(video_path)
        capture.release()
    return fps,frame_count


def video_duration(video_path: str) -> float:
    fps,frame_count=_video_timing(video_path)
    return frame_count / fps


def extract_frames(
    video_path: str, timestamps: list[float], *, deduplicate: bool = True
) -> list[VideoFrame]:
    software_only = _needs_software_ffmpeg_fallback(video_path)
    if software_only:
        cv2,capture=None,None
        fps,frame_count,width,height=_software_video_metadata(video_path)
    else:
        cv2, capture, fps, frame_count = _open_video(video_path)
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    duration = frame_count / fps
    use_nvdec = not software_only and _nvdec_available()
    requests: list[tuple[int, float, int]] = []
    seen_frame_indices: set[int] = set()
    for request_index, timestamp in enumerate(timestamps):
        timestamp = min(max(float(timestamp), 0.0), max(duration - 1.0 / fps, 0.0))
        # Different requested timestamps can decode to the same physical frame,
        # especially when several expansion windows overlap in a short video.
        frame_index = min(max(int(round(timestamp * fps)), 0), frame_count - 1)
        if deduplicate and frame_index in seen_frame_indices:
            continue
        seen_frame_indices.add(frame_index)
        requests.append((request_index, frame_index / fps, frame_index))

    parallel_min = int(os.environ.get("VIDEO_PARALLEL_DECODE_MIN_FRAMES", "64"))
    max_workers = min(int(os.environ.get("VIDEO_DECODE_WORKERS", "8")), len(requests))
    if not use_nvdec and len(requests) >= parallel_min and max_workers > 1:
        # Dense locator scans on long videos otherwise spend minutes doing
        # hundreds of random seeks through one decoder.  Split the ordered
        # requests into contiguous ranges; every timestamp and decoded pixel
        # remains identical to the serial OpenCV path.
        if capture is not None:capture.release()
        chunk_size = (len(requests) + max_workers - 1) // max_workers
        chunks = [requests[index:index + chunk_size]
                  for index in range(0, len(requests), chunk_size)]

        def decode_chunk(chunk):
            chunk_cv2, chunk_capture = None, None
            if not software_only:
                chunk_cv2, chunk_capture, _, _ = _open_video(video_path)
            decoded = []
            try:
                for request_index, timestamp, _ in chunk:
                    if software_only:
                        image = _software_ffmpeg_frame(video_path, timestamp)
                        if image is None:
                            continue
                    else:
                        chunk_capture.set(chunk_cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                        ok, bgr = chunk_capture.read()
                        if not ok:
                            continue
                        rgb = chunk_cv2.cvtColor(bgr, chunk_cv2.COLOR_BGR2RGB)
                        from PIL import Image
                        image = Image.fromarray(rgb)
                    decoded.append((request_index, VideoFrame(
                        timestamp=timestamp, image=image,
                    )))
            finally:
                if chunk_capture is not None:
                    chunk_capture.release()
            return decoded

        decoded = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            for result in executor.map(decode_chunk, chunks):
                decoded.extend(result)
        return [frame for _, frame in sorted(decoded, key=lambda item: item[0])]

    frames: list[VideoFrame] = []
    try:
        for _, timestamp, _ in requests:
            if software_only:
                image = _software_ffmpeg_frame(video_path, timestamp)
                if image is not None:
                    frames.append(VideoFrame(timestamp=timestamp, image=image))
                continue
            if use_nvdec:
                image = _gpu_decode_frame(video_path, timestamp, width, height)
                if image is not None:
                    frames.append(VideoFrame(timestamp=timestamp, image=image))
                    continue
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, bgr = capture.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            from PIL import Image
            frames.append(VideoFrame(timestamp=timestamp, image=Image.fromarray(rgb)))
    finally:
        if capture is not None:capture.release()
    return frames


def recover_frame_near_timestamp(
    video_path: str,
    timestamp: float,
    *,
    max_backtrack_seconds: float = 1.0,
) -> VideoFrame | None:
    """Deterministically recover one frame when OpenCV random seeking fails.

    Some older or damaged H.264 containers can decode a timestamp during a
    sequential/batched pass but fail when a fresh OpenCV decoder seeks to the
    same position.  Training trajectory replay must preserve the image count,
    so try the requested frame and a small number of *earlier* neighbouring
    frames, first with OpenCV and then with FFmpeg's independent demuxer.

    The fallback never moves forward and never searches farther than
    ``max_backtrack_seconds``.  Returning the actual recovered timestamp keeps
    the substitution auditable instead of pretending the exact request was
    decoded.
    """
    if max_backtrack_seconds < 0:
        raise ValueError("max_backtrack_seconds must be non-negative")
    fps,frame_count=_video_timing(video_path)
    duration = frame_count / fps
    requested = min(max(float(timestamp), 0.0), max(duration - 1.0 / fps, 0.0))

    offsets = [0.0, 1.0 / fps, 2.0 / fps, 3.0 / fps, 0.1, 0.2, 0.5, 0.75, 1.0]
    candidates: list[float] = []
    for offset in offsets:
        if offset > max_backtrack_seconds + 1e-9:
            continue
        candidate = max(0.0, requested - offset)
        if not any(abs(candidate - known) < 1e-9 for known in candidates):
            candidates.append(candidate)

    for candidate in candidates:
        decoded = extract_frames(video_path, [candidate], deduplicate=False)
        if decoded:
            return decoded[0]

    ffmpeg = os.environ.get("VIDEO_CPU_FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    for candidate in candidates:
        command = [
            ffmpeg, "-v", "error", "-ss", f"{candidate:.6f}",
            "-i", video_path, "-frames:v", "1", "-an",
            "-f", "image2pipe", "-vcodec", "png", "pipe:1",
        ]
        try:
            result = subprocess.run(command, capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode or not result.stdout:
            continue
        try:
            from PIL import Image
            with Image.open(BytesIO(result.stdout)) as image:
                return VideoFrame(timestamp=candidate, image=image.convert("RGB").copy())
        except (OSError, ValueError):
            continue
    return None


def uniform_frames(video_path: str, num_frames: int) -> list[VideoFrame]:
    """Uniformly sample visible frames, excluding near-blank black frames.

    We first try the exact evenly spaced timestamps.  Only if one of those is
    a black transition frame do we decode a denser uniform candidate set and
    choose evenly across the remaining visible frames.  This preserves the
    usual boundary sampling for normal videos without spending a VLM image on
    a fade frame.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    fps,frame_count=_video_timing(video_path)
    last_timestamp = max((frame_count - 1) / fps, 0.0)
    if num_frames == 1:
        timestamps = [last_timestamp / 2]
    else:
        # Boundary frames are mandatory: image-to-video clips can contain risk
        # only in the source frame before animation quickly changes the scene.
        timestamps = [index * last_timestamp / (num_frames - 1) for index in range(num_frames)]
    sampled = extract_frames(video_path, timestamps)
    visible = [frame for frame in sampled if not is_near_black_frame(frame.image)]
    # A decoder can miss one requested timestamp even when none of the frames
    # it did return is black.  In that case we still need the denser fallback
    # below; otherwise callers unexpectedly receive 3/4 initial frames.
    if len(sampled) == num_frames and len(visible) == num_frames:
        return visible

    # Search alternative positions only when the original uniform sample
    # contained a blank frame.  Include the original positions so that a
    # non-black boundary remains eligible.
    # Some archive videos contain a long black leader followed by only a short
    # visible segment.  Twenty probes can find fewer than ``num_frames`` even
    # though that segment contains hundreds of decodable frames.  This denser
    # fallback is used only after the cheap boundary sample has already found
    # a black/missing frame, so normal videos keep the original fast path.
    candidate_count = max(num_frames * 20, 80)
    candidate_timestamps = [index * last_timestamp / (candidate_count - 1)
                            for index in range(candidate_count)]
    candidates = extract_frames(video_path, sorted(set(timestamps + candidate_timestamps)))
    visible = [frame for frame in candidates if not is_near_black_frame(frame.image)]
    if not visible:
        # A genuinely all-black video has no meaningful visual observation.
        # Returning no frames lets the caller mark it as undecodable instead
        # of silently providing black images to the model.
        return []
    if len(visible) <= num_frames:
        return visible
    indices = [min(round(index * (len(visible) - 1) / (num_frames - 1)), len(visible) - 1)
               for index in range(num_frames)] if num_frames > 1 else [len(visible) // 2]
    return [visible[index] for index in indices]
