"""Replay requested timestamps when OpenCV lacks the source codec."""
from pathlib import Path
from agent.types import VideoFrame
from agent.video import extract_frames, _software_ffmpeg_frame


def recorded_frames(video_path, timestamps):
    if not Path(video_path).is_file():
        raise FileNotFoundError(video_path)
    try:
        frames = extract_frames(video_path, timestamps, deduplicate=False)
        if len(frames) == len(timestamps):
            return frames
    except RuntimeError:
        pass
    # FFmpeg can decode AV1 files that this OpenCV build cannot even open.
    # Keep each original request. Never substitute an earlier/later timestamp.
    result = []
    for timestamp in timestamps:
        image = _software_ffmpeg_frame(video_path, float(timestamp))
        if image is None:
            raise RuntimeError(f'Cannot replay requested timestamp {timestamp}: {video_path}')
        frame = VideoFrame(timestamp=float(timestamp), image=image)
        frame.replay_decoder = 'ffmpeg_requested_timestamp_actual_pts_not_verified'
        result.append(frame)
    return result
