from __future__ import annotations

import argparse
from pathlib import Path
import json
import yaml

from src.data.train_loader import load_train_data, build_video_abspath
from src.selector.pipeline import KeyframeSelectorPipeline
from src.eval.support_frame_scorer import SupportFrameScorer


def infer_question_type_from_sample(sample: dict) -> str | None:
    return sample.get("type")


def print_support_frame_comparison(
    sample_id: str,
    question: str,
    support_frames: list[float],
    candidate_scores: list[float],
    support_scoring_result: dict,
) -> None:
    print()
    print("=" * 90)
    print("🎯 SUPPORT FRAME SCORING COMPARISON")
    print("=" * 90)
    print(f"Sample ID    : {sample_id}")
    print(f"Question     : {question}")
    print(f"Support Times: {support_frames}")
    print()

    print("📍 Support Frames:")
    if support_scoring_result["support_frames"]:
        for i, sf in enumerate(support_scoring_result["support_frames"], start=1):
            score = sf["score"]
            rank = sf.get("rank_in_candidates", "?")
            status = sf.get("status", "N/A")
            status_emoji = "✓" if status == "excellent" else "◐" if status == "good" else "✗"

            print(f"  {status_emoji} Support Frame {i}")
            print(f"      GT Time     : {sf['time_sec']:.6f}s")
            print(f"      Actual Time : {sf.get('actual_time_sec', sf['time_sec']):.6f}s")
            print(f"      Frame Idx   : {sf.get('frame_idx', -1)}")
            print(f"      Score       : {score:.6f}")
            print(f"      Rank        : #{rank} (out of {len(candidate_scores)} candidates)")
            print(f"      Status      : {status}")
            print(f"      Components:")
            for comp_name, comp_val in sf["components"].items():
                print(f"        - {comp_name}: {comp_val:.4f}")
            print()
    else:
        print("  ⚠️ No support frames could be scored")
        print()

    print("📊 Candidate Frames:")
    print(f"  Total candidates: {len(candidate_scores)}")
    if candidate_scores:
        sorted_scores = sorted(candidate_scores, reverse=True)
        print(f"  Top 4 scores: {[f'{s:.6f}' for s in sorted_scores[:4]]}")
        print(f"  Score range: {min(candidate_scores):.6f} - {max(candidate_scores):.6f}")
        print(f"  Average: {sum(candidate_scores) / len(candidate_scores):.6f}")
    print()

    comparison = support_scoring_result["comparison"]
    print("📈 Comparison Summary:")
    print(f"  Avg Support Score  : {comparison['avg_support_score']:.6f}")
    print(f"  Max Candidate Score: {comparison['max_candidate_score']:.6f}")
    print(f"  Min Candidate Score: {comparison['min_candidate_score']:.6f}")
    print()
    print(comparison["summary"])
    print("=" * 90)
    print()


def main():
    parser = argparse.ArgumentParser(description="Test support frame scorer")
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
    print(f"✓ Loaded {len(data)} samples from {args.train_json}")

    if args.sample_id is not None:
        sample = next((x for x in data if x.get("id") == args.sample_id), None)
        if sample is None:
            raise ValueError(f"Cannot find sample_id={args.sample_id}")
    elif args.sample_index is not None:
        sample = data[args.sample_index]
    else:
        sample = data[0]

    support_frames = sample.get("support_frames", [])
    question_type = infer_question_type_from_sample(sample)
    video_path = build_video_abspath(args.video_root, sample["video_path"])

    print(f"✓ Using sample      : {sample['id']}")
    print(f"✓ Video             : {video_path}")
    print(f"✓ Question Type     : {question_type}")
    print(f"✓ Support Frames    : {support_frames}")

    if not support_frames:
        print("⚠️ No support frames found in this sample")
        return

    all_candidate_frames = []
    candidate_scores = []

    if not args.no_pipeline:
        print("\n🔄 Running candidate selector pipeline...")
        pipeline = KeyframeSelectorPipeline(args.config)
        selector_result = pipeline.run(str(video_path), question_type=question_type)

        all_candidate_frames = selector_result["all_frames"]
        candidate_scores = [x["score"] for x in all_candidate_frames]

        print(f"✓ Selected frames   : {len(selector_result['selected_frames'])}")
        print(f"✓ All candidates    : {len(all_candidate_frames)}")
    else:
        print("\n⊘ Skipping pipeline (--no_pipeline)")

    print("\n🔄 Scoring support frames...")
    scorer = SupportFrameScorer(cfg)
    result = scorer.score_support_frames(
        video_path=str(video_path),
        support_times=support_frames,
        question_type=question_type,
        all_candidate_frames=all_candidate_frames if all_candidate_frames else None,
    )
    print("✓ Support frames scored")

    print_support_frame_comparison(
        sample_id=sample["id"],
        question=sample["question"],
        support_frames=support_frames,
        candidate_scores=candidate_scores,
        support_scoring_result=result,
    )

    if args.output_json:
        output_data = {
            "sample": {
                "id": sample["id"],
                "question": sample["question"],
                "type": question_type,
                "support_frames": support_frames,
                "video_path": sample["video_path"],
                "video_abspath": str(video_path),
            },
            "support_frames_scoring": result,
            "candidate_frames_stats": {
                "total": len(candidate_scores),
                "scores": candidate_scores,
                "average": sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0.0,
                "max": max(candidate_scores) if candidate_scores else 0.0,
                "min": min(candidate_scores) if candidate_scores else 0.0,
            },
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Results saved to: {out_path}")


if __name__ == "__main__":
    main()