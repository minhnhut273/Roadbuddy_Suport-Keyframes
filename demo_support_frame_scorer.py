#!/usr/bin/env python
"""
Quick demo script để test support frame scoring functionality
Sử dụng result.json từ outputs_batch_50
"""

from __future__ import annotations
from pathlib import Path
import yaml
import json

from src.eval.support_frame_scorer import SupportFrameScorer
from src.utils.io import read_json


def main():
    # Load config
    with open("configs/selector.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load result from existing output
    result_json = read_json("outputs_batch_50/train_0047/result.json")
    
    sample = result_json["sample"]
    selector_result = result_json["selector_result"]
    
    support_frames = sample.get("support_frames", [])
    video_path = sample["video_abspath"]

    print(f"🎯 Demo: Support Frame Scoring")
    print(f"=" * 80)
    print(f"Sample ID    : {sample['id']}")
    print(f"Question     : {sample['question']}")
    print(f"Support Frames: {support_frames}")
    print(f"Video Path   : {video_path}")
    print()

    if not support_frames:
        print("⚠️  No support frames in this sample")
        return

    # Score support frames
    print("🔄 Scoring support frames...")
    scorer = SupportFrameScorer(cfg)
    
    # Convert selector_result frames to simple format for comparison
    candidate_frames = [
        {
            "score": f["score"],
            "components": f["components"]
        }
        for f in selector_result["all_frames"]
    ]
    
    support_scoring_result = scorer.score_support_frames(
        video_path=video_path,
        support_times=support_frames,
        all_candidate_frames=candidate_frames,
    )
    print("✓ Scoring complete\n")

    # Print results
    print("=" * 90)
    print("📊 SUPPORT FRAME SCORING RESULTS")
    print("=" * 90)
    print()

    # Support frames
    print("📍 Support Frames Details:")
    if support_scoring_result["support_frames"]:
        for i, sf in enumerate(support_scoring_result["support_frames"], start=1):
            rank = sf.get("rank_in_candidates", "?")
            status = sf.get("status", "N/A")
            
            status_emoji = "✓" if status == "excellent" else "◐" if status == "good" else "✗"
            print(f"\n  {status_emoji} Support Frame {i}")
            print(f"      Time: {sf['time_sec']:.6f}s")
            print(f"      Score: {sf['score']:.6f}")
            print(f"      Rank: #{rank} (of {len(candidate_frames)} candidates)")
            print(f"      Status: {status}")
            print(f"      Components:")
            for name, val in sf["components"].items():
                print(f"        • {name:20s}: {val:.4f}")
    print()

    # Comparison
    comparison = support_scoring_result["comparison"]
    print("📈 Comparison with Candidates:")
    print(f"  Total Support Frames  : {comparison['total_support_frames']}")
    print(f"  Total Candidate Frames: {comparison['total_candidate_frames']}")
    print(f"  Avg Support Score     : {comparison['avg_support_score']:.6f}")
    print(f"  Max Candidate Score   : {comparison['max_candidate_score']:.6f}")
    print(f"  Min Candidate Score   : {comparison['min_candidate_score']:.6f}")
    print()

    # Summary
    summary = comparison["summary"]
    if "✓" in summary:
        emoji = "🟢"
    elif "~" in summary:
        emoji = "🟡"
    else:
        emoji = "🔴"

    print(f"{emoji} Summary: {summary}")
    print("=" * 90)


if __name__ == "__main__":
    main()
