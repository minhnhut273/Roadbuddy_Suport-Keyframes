from __future__ import annotations
from dataclasses import dataclass
import cv2


@dataclass
class SampledFrame:
    frame_idx: int
    time_sec: float
    image: any


def sample_video_uniform(
    video_path: str,
    fps_sample: float = 2.0,
    max_frames: int | None = None,
) -> list[SampledFrame]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    frame_step = max(int(round(native_fps / fps_sample)), 1)

    sampled: list[SampledFrame] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            time_sec = frame_idx / native_fps
            sampled.append(SampledFrame(frame_idx=frame_idx, time_sec=time_sec, image=frame))
            if max_frames is not None and len(sampled) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return sampled