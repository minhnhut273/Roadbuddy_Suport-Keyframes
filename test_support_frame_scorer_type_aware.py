#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.data.train_loader import load_train_data, build_video_abspath, infer_question_type
from src.selector.pipeline import KeyframeSelectorPipeline
from src.utils.io import write_json
from support_frame_scorer_type_aware import SupportFrameScorer


def print_result(sample: dict[str, Any], result: dict[str, Any]) -> None:
    print("=" * 90)
    print("TYPE-AWARE SUPPORT FRAME SCORING")
    print("=" * 90)
    print(f"Sample ID      : {sample['id']}")
    print(f"Question       : {sample['question']}")
    print(f"Question Type  : {result.get('question_type')}")
    print(f"Support Frames : {sample.get('support_frames', [])}")
    print()

    print("Support Frames:")
    for i, sf in enumerate(result["support_frames"], start=1):
        print(
            f"  {i}. t={sf['time_sec']:.6f}s | score={sf['score']:.6f} | "
            f"rank={sf.get('rank_in_candidates')} | status={sf.get('status')}"
        )
        for k, v in sf["components"].items():
            print(f"      {k:18s}: {v:.4f}")
    print()

    comp = result["comparison"]
    print("Comparison:")
    print(f"  Total support       : {comp['total_support_frames']}")
    print(f"  Total candidates    : {comp['total_candidate_frames']}")
    print(f"  Avg support score   : {comp['avg_support_score']:.6f}")
    print(f"  Avg candidate score : {comp['avg_candidate_score']:.6f}")
    print(f"  Max candidate score : {comp['max_candidate_score']:.6f}")
    print(f"  Min candidate score : {comp['min_candidate_score']:.6f}")
    print(f"  Summary             : {comp['summary']}")
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Type-aware support frame scorer")
    parser.add_argument("--config", default="configs/selector.yaml")
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--no_pipeline", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = load_train_data(args.train_json)
    if args.sample_id is not None:
        sample = next((x for x in data if x.get("id") == args.sample_id), None)
        if sample is None:
            raise ValueError(f"Cannot find sample_id={args.sample_id}")
    elif args.sample_index is not None:
        sample = data[args.sample_index]
    else:
        sample = data[0]

    video_path = build_video_abspath(args.video_root, sample["video_path"])
    question_type = sample.get("type") or infer_question_type(sample)
    support_times = sample.get("support_frames", [])

    candidate_times: list[float] = []
    selector_result = None
    if not args.no_pipeline:
        pipeline = KeyframeSelectorPipeline(args.config)
        selector_result = pipeline.run(video_path, question_type=question_type)
        candidate_times = [f["time_sec"] for f in selector_result.get("all_frames", [])]

    scorer = SupportFrameScorer(cfg)
    result = scorer.score_support_frames(
        video_path=video_path,
        support_times=support_times,
        candidate_times=candidate_times,
        question_type=question_type,
    )

    print_result(sample, result)

    if args.output_json:
        payload = {
            "sample": {
                "id": sample["id"],
                "question": sample["question"],
                "type": question_type,
                "support_frames": support_times,
                "video_path": sample["video_path"],
                "video_abspath": video_path,
            },
            "selector_result": selector_result,
            "support_frames_scoring": result,
        }
        write_json(payload, args.output_json)
        print(f"Saved JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
