#!/usr/bin/env python
from __future__ import annotations

import yaml

from src.utils.io import read_json
from support_frame_scorer_type_aware import SupportFrameScorer


def main() -> None:
    with open("configs/selector.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    result_json = read_json("outputs_batch_50/train_0047/result.json")
    sample = result_json["sample"]
    selector_result = result_json["selector_result"]

    question_type = sample.get("type")
    support_frames = sample.get("support_frames", [])
    video_path = sample["video_abspath"]
    candidate_times = [f["time_sec"] for f in selector_result.get("all_frames", [])]

    print("🎯 Demo: Type-aware Support Frame Scoring")
    print("=" * 80)
    print(f"Sample ID      : {sample['id']}")
    print(f"Question       : {sample['question']}")
    print(f"Question Type  : {question_type}")
    print(f"Support Frames : {support_frames}")
    print(f"Candidates     : {len(candidate_times)} frames")
    print()

    scorer = SupportFrameScorer(cfg)
    result = scorer.score_support_frames(
        video_path=video_path,
        support_times=support_frames,
        candidate_times=candidate_times,
        question_type=question_type,
    )

    print(result["comparison"]["summary"])
    for i, sf in enumerate(result["support_frames"], start=1):
        print(
            f"  Support {i}: t={sf['time_sec']:.6f}s | score={sf['score']:.6f} | "
            f"rank={sf.get('rank_in_candidates')} | status={sf.get('status')}"
        )


if __name__ == "__main__":
    main()
