from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import yaml

from src.data.train_loader import load_train_data, build_video_abspath, infer_question_type
from src.selector.pipeline import KeyframeSelectorPipeline
from src.eval.eval_selector import evaluate_selection
from src.eval.support_frame_scorer import SupportFrameScorer
from src.utils.io import ensure_dir, write_json


def save_selected_images(selected_frames: list[dict], output_dir: Path) -> None:
    ensure_dir(output_dir)
    for rank, fr in enumerate(selected_frames, start=1):
        image = fr["image"]
        fname = f"top{rank}_{fr['time_sec']:.3f}s_score_{fr['score']:.4f}.jpg"
        cv2.imwrite(str(output_dir / fname), image)


def strip_images(obj: dict) -> dict:
    cleaned = {
        "video_path": obj["video_path"],
        "sampled_frames": obj["sampled_frames"],
        "selected_frames": [],
        "all_frames": [],
    }

    for fr in obj["selected_frames"]:
        copied = {k: v for k, v in fr.items() if k != "image"}
        cleaned["selected_frames"].append(copied)

    for fr in obj["all_frames"]:
        copied = {k: v for k, v in fr.items() if k != "image"}
        cleaned["all_frames"].append(copied)

    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/selector.yaml")
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--sample_id", default=None, help="Ví dụ train_0001")
    parser.add_argument("--sample_index", type=int, default=None, help="0-based index")
    parser.add_argument("--output_dir", default="outputs/run_selector")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

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
    support_frames = sample.get("support_frames", [])
    question_type = infer_question_type(sample)

    pipeline = KeyframeSelectorPipeline(args.config)
    print("Resolved video path:", video_path)

    result = pipeline.run(video_path, question_type=question_type)

    pred_times = [x["time_sec"] for x in result["selected_frames"]]

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    eval_metrics = evaluate_selection(
        pred_times=pred_times,
        support_frames=support_frames,
        tolerances=cfg["eval_tolerances_sec"],
    )

    # Score support frames và so sánh với candidate frames
    support_scorer = SupportFrameScorer(cfg)
    support_scoring_result = support_scorer.score_support_frames(
        video_path=video_path,
        support_times=support_frames,
        all_candidate_frames=result["all_frames"],
    )

    sample_out_dir = output_dir / sample["id"]
    ensure_dir(sample_out_dir)

    if cfg.get("save_top_frames", True):
        save_selected_images(result["selected_frames"], sample_out_dir / "frames")

    json_obj = {
        "sample": {
            "id": sample["id"],
            "question": sample["question"],
            "choices": sample["choices"],
            "answer": sample.get("answer"),
            "type": question_type,
            "video_path": sample["video_path"],
            "video_abspath": video_path,
            "support_frames": support_frames,
        },
        "selector_result": strip_images(result),
        "support_frames_scoring": support_scoring_result,
        "evaluation": eval_metrics,
    }

    write_json(json_obj, sample_out_dir / "result.json")

    print("=" * 80)
    print(f"Sample ID       : {sample['id']}")
    print(f"Question Type   : {question_type}")
    print(f"Question        : {sample['question']}")
    print(f"Support Frames  : {support_frames}")
    print(f"Candidate Times : {pred_times}")
    print(f"Evaluation      : {eval_metrics}")
    print()
    print("Support Frame Scoring:")
    print(f"  Total Support Frames    : {support_scoring_result['comparison']['total_support_frames']}")
    if support_scoring_result['support_frames']:
        for i, sf in enumerate(support_scoring_result['support_frames'], start=1):
            rank_info = f" (Rank #{sf.get('rank_in_candidates', '?')})" if 'rank_in_candidates' in sf else ""
            status = f" [{sf.get('status', 'N/A')}]" if 'status' in sf else ""
            print(
                f"    Support Frame {i}: t={sf['time_sec']:.4f}s, "
                f"score={sf['score']:.4f}{rank_info}{status}"
            )
    print(f"  Avg Support Score       : {support_scoring_result['comparison']['avg_support_score']:.4f}")
    print(f"  Comparison Summary      : {support_scoring_result['comparison']['summary']}")
    print()
    print(f"Saved to        : {sample_out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()