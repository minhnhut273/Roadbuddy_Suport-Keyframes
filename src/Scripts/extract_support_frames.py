from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import cv2


def load_train_json(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "data" in obj:
        return obj["data"]
    if isinstance(obj, list):
        return obj

    raise ValueError("Unsupported train.json format")


def resolve_video_path(video_root: Path, raw_video_path: str) -> Path:
    raw_path = Path(raw_video_path)

    # Nếu đã là absolute path
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    # Nếu train.json ghi kiểu dataset/videos/xxx.mp4 mà video_root đã là .../dataset/videos
    if raw_path.parts[:2] == ("dataset", "videos"):
        candidate = video_root / raw_path.name
        if candidate.exists():
            return candidate

    # fallback: ghép trực tiếp
    candidate = video_root / raw_path
    if candidate.exists():
        return candidate

    # fallback cuối: chỉ lấy tên file
    candidate = video_root / raw_path.name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Cannot resolve video path: {raw_video_path} with video_root={video_root}")


def extract_frame_at_time(video_path: Path, time_sec: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = int(round(time_sec * fps))
    frame_idx = max(0, min(frame_idx, max(total_frames - 1, 0)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Cannot read frame at {time_sec:.3f}s from {video_path}")

    actual_time = frame_idx / fps
    return frame, frame_idx, actual_time, fps


def save_support_frames(
    train_json: Path,
    video_root: Path,
    output_dir: Path,
    limit: int | None = None,
    sample_id: str | None = None,
) -> None:
    data = load_train_json(train_json)

    if sample_id is not None:
        data = [x for x in data if x.get("id") == sample_id]

    if limit is not None:
        data = data[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for item in data:
        sid = item.get("id", "unknown")
        qtype = item.get("type", "")
        raw_video_path = item.get("video_path", "")
        support_frames = item.get("support_frames", [])

        if not raw_video_path or not support_frames:
            continue

        try:
            video_path = resolve_video_path(video_root, raw_video_path)
        except Exception as e:
            print(f"[WARN] Skip {sid}: {e}")
            continue

        sample_out = output_dir / sid
        sample_out.mkdir(parents=True, exist_ok=True)

        for i, t in enumerate(support_frames, start=1):
            try:
                frame, frame_idx, actual_time, fps = extract_frame_at_time(video_path, float(t))
            except Exception as e:
                print(f"[WARN] Failed {sid} support[{i}]={t}: {e}")
                continue

            filename = f"support_{i}_gt_{float(t):.3f}s_actual_{actual_time:.3f}s_fidx_{frame_idx}.jpg"
            save_path = sample_out / filename
            cv2.imwrite(str(save_path), frame)

            manifest.append({
                "id": sid,
                "type": qtype,
                "video_path": str(video_path),
                "support_index": i,
                "gt_time_sec": float(t),
                "actual_time_sec": actual_time,
                "frame_idx": frame_idx,
                "fps": fps,
                "image_path": str(save_path),
            })

            print(f"[OK] {sid} -> {save_path.name}")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Saved support frames to: {output_dir}")
    print(f"Saved manifest to     : {manifest_path}")
    print(f"Total extracted       : {len(manifest)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract support frames from videos using train.json")
    parser.add_argument("--train_json", required=True, help="Path to train.json")
    parser.add_argument("--video_root", required=True, help="Path to video root")
    parser.add_argument("--output_dir", required=True, help="Directory to save support frame images")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit number of samples")
    parser.add_argument("--sample_id", default=None, help="Optional single sample id, e.g. train_0019")
    args = parser.parse_args()

    save_support_frames(
        train_json=Path(args.train_json),
        video_root=Path(args.video_root),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        sample_id=args.sample_id,
    )


if __name__ == "__main__":
    main()