from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.data.train_loader import load_train_data, build_video_abspath
from src.selector.pipeline import KeyframeSelectorPipeline
from src.eval.support_frame_scorer import SupportFrameScorer

from src.utils.io import ensure_dir, write_json


def save_image(path: Path, image_bgr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)


def extract_frame_at_time(video_path: str, time_sec: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, None, -1, -1.0

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = int(round(time_sec * fps))
        frame_idx = max(0, min(frame_idx, max(total_frames - 1, 0)))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        actual_time = frame_idx / fps
        return ok, frame, frame_idx, actual_time
    finally:
        cap.release()


def summarize_support_scoring(support_scoring: dict) -> dict:
    support_frames = support_scoring.get("support_frames", [])
    comparison = support_scoring.get("comparison", {})

    if not support_frames:
        return {
            "support_count": 0,
            "avg_support_score": 0.0,
            "best_support_rank": None,
            "weak_support_count": 0,
            "good_or_better_count": 0,
            "summary": comparison.get("summary", "No support frames"),
        }

    ranks = [x.get("rank_in_candidates") for x in support_frames if x.get("rank_in_candidates") is not None]
    weak_count = sum(1 for x in support_frames if x.get("status") == "weak")
    good_count = sum(1 for x in support_frames if x.get("status") in ("good", "excellent"))

    return {
        "support_count": len(support_frames),
        "avg_support_score": float(comparison.get("avg_support_score", 0.0)),
        "best_support_rank": min(ranks) if ranks else None,
        "weak_support_count": weak_count,
        "good_or_better_count": good_count,
        "summary": comparison.get("summary", ""),
    }


def aggregate_compare(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_samples": 0}

    def avg(key: str) -> float:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    support_rank_vals = [int(r["best_support_rank"]) for r in rows if r.get("best_support_rank") not in (None, "")]
    weak_cnt = sum(int(r.get("weak_support_count", 0)) for r in rows)
    good_cnt = sum(int(r.get("good_or_better_count", 0)) for r in rows)

    return {
        "num_samples": len(rows),
        "candidate_metrics_avg": {
            "hit@1_tol_0_25": avg("hit@1_tol_0_25"),
            "hit@k_tol_0_25": avg("hit@k_tol_0_25"),
            "recall_tol_0_25": avg("recall_tol_0_25"),
            "hit@1_tol_0_5": avg("hit@1_tol_0_5"),
            "hit@k_tol_0_5": avg("hit@k_tol_0_5"),
            "recall_tol_0_5": avg("recall_tol_0_5"),
            "hit@1_tol_1_0": avg("hit@1_tol_1_0"),
            "hit@k_tol_1_0": avg("hit@k_tol_1_0"),
            "recall_tol_1_0": avg("recall_tol_1_0"),
            "mean_min_distance": avg("mean_min_distance"),
        },
        "support_metrics_avg": {
            "avg_support_score": avg("avg_support_score"),
            "avg_candidate_score_mean": avg("avg_candidate_score"),
            "max_candidate_score_mean": avg("max_candidate_score"),
            "min_candidate_score_mean": avg("min_candidate_score"),
            "mean_best_support_rank": (sum(support_rank_vals) / len(support_rank_vals)) if support_rank_vals else None,
        },
        "support_status_totals": {
            "weak_support_count": weak_cnt,
            "good_or_better_count": good_cnt,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run full train set: candidate frames + support frames + per-sample result.json + results.csv + compare.json"
    )
    parser.add_argument("--config", default="configs/selector.yaml")
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = load_train_data(args.train_json)
    start = args.start_index
    end = len(data) if args.limit is None else min(len(data), start + args.limit)
    data = data[start:end]

    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    pipeline = KeyframeSelectorPipeline(args.config)
    support_scorer = SupportFrameScorer(cfg)

    csv_rows: list[dict[str, Any]] = []

    for idx, sample in enumerate(data, start=start):
        sample_id = sample["id"]
        qtype = sample.get("type")
        question = sample.get("question", "")
        support_times = sample.get("support_frames", [])

        video_path = build_video_abspath(args.video_root, sample["video_path"])
        sample_out = out_root / sample_id
        candidate_out = sample_out / "candidate_frames"
        support_out = sample_out / "support_frames"
        ensure_dir(candidate_out)
        ensure_dir(support_out)

        print("=" * 100)
        print(f"[{idx}] Processing {sample_id}")
        print(f"  Type          : {qtype}")
        print(f"  Question      : {question}")
        print(f"  Support Frames: {support_times}")
        print(f"  Video         : {video_path}")

        # 1) Run candidate selector
        selector_result = pipeline.run(str(video_path), question_type=qtype)

        # Save selected candidate frame images
        for rank, fr in enumerate(selector_result["selected_frames"], start=1):
            img = fr.get("image")
            if img is None:
                continue
            save_image(
                candidate_out / f"top{rank}_{fr['time_sec']:.3f}s_score_{fr['score']:.4f}.jpg",
                img,
            )

        # 2) Save support frame images
        support_manifest = []
        for sidx, t in enumerate(support_times, start=1):
            ok, frame, frame_idx, actual_time = extract_frame_at_time(str(video_path), float(t))
            if not ok or frame is None:
                continue
            save_path = support_out / f"support{sidx}_gt_{float(t):.3f}s_actual_{actual_time:.3f}s_fidx_{frame_idx}.jpg"
            save_image(save_path, frame)
            support_manifest.append(
                {
                    "support_index": sidx,
                    "gt_time_sec": float(t),
                    "actual_time_sec": float(actual_time),
                    "frame_idx": int(frame_idx),
                    "image_path": str(save_path),
                }
            )

        # 3) Evaluate candidate selector
        pred_times = [x["time_sec"] for x in selector_result["selected_frames"]]
        eval_result = evaluate_selected_times(
            pred_times=pred_times,
            support_times=support_times,
            tolerances=cfg.get("eval_tolerances_sec", [0.25, 0.5, 1.0]),
        )

        # 4) Score support frames using current type-aware + ROI-aware scorer
        support_scoring = support_scorer.score_support_frames(
            video_path=str(video_path),
            support_times=support_times,
            question_type=qtype,
            all_candidate_frames=selector_result["all_frames"],
        )
        support_summary = summarize_support_scoring(support_scoring)

        # 5) Per-sample result.json
        selector_result_json = make_selector_result_json_safe(selector_result)

        sample_result = {
            "sample": {
                **sample,
                "video_abspath": str(video_path),
            },
            "selector_result": selector_result_json,
            "support_frame_images": support_manifest,
            "support_frames_scoring": support_scoring,
            "evaluation": eval_result,
        }
        write_json(sample_result, sample_out / "result.json")

        # 6) Row for results.csv
        cand_scores = [x["score"] for x in selector_result["all_frames"]]
        row = {
            "id": sample_id,
            "type": qtype,
            "video_path": sample["video_path"],
            "support_frames": json.dumps(support_times, ensure_ascii=False),
            "candidate_times": json.dumps(pred_times, ensure_ascii=False),
            "question": question,
            "hit@1_tol_0_25": eval_result.get("hit@1_tol_0_25", 0),
            "hit@k_tol_0_25": eval_result.get("hit@k_tol_0_25", 0),
            "recall_tol_0_25": eval_result.get("recall_tol_0_25", 0.0),
            "hit@1_tol_0_5": eval_result.get("hit@1_tol_0_5", 0),
            "hit@k_tol_0_5": eval_result.get("hit@k_tol_0_5", 0),
            "recall_tol_0_5": eval_result.get("recall_tol_0_5", 0.0),
            "hit@1_tol_1_0": eval_result.get("hit@1_tol_1_0", 0),
            "hit@k_tol_1_0": eval_result.get("hit@k_tol_1_0", 0),
            "recall_tol_1_0": eval_result.get("recall_tol_1_0", 0.0),
            "mean_min_distance": eval_result.get("mean_min_distance", 0.0),
            "num_candidates": len(selector_result["all_frames"]),
            "avg_candidate_score": (sum(cand_scores) / len(cand_scores)) if cand_scores else 0.0,
            "max_candidate_score": max(cand_scores) if cand_scores else 0.0,
            "min_candidate_score": min(cand_scores) if cand_scores else 0.0,
            "support_count": support_summary["support_count"],
            "avg_support_score": support_summary["avg_support_score"],
            "best_support_rank": support_summary["best_support_rank"],
            "weak_support_count": support_summary["weak_support_count"],
            "good_or_better_count": support_summary["good_or_better_count"],
            "support_summary": support_summary["summary"],
        }
        csv_rows.append(row)

    # 7) Save results.csv
    csv_path = out_root / "results.csv"
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    # 8) Save compare.json
    compare = aggregate_compare(csv_rows)
    compare["start_index"] = start
    compare["end_index_exclusive"] = end
    compare["results_csv"] = str(csv_path)
    write_json(compare, out_root / "compare.json")

    print("=" * 100)
    print("DONE")
    print(f"Samples      : {len(csv_rows)}")
    print(f"Results CSV  : {csv_path}")
    print(f"Compare JSON : {out_root / 'compare.json'}")



def evaluate_selected_times(
    pred_times: list[float],
    support_times: list[float],
    tolerances: list[float],
) -> dict:
    if not support_times:
        out = {}
        for tol in tolerances:
            tol_key = str(tol).replace(".", "_")
            out[f"hit@1_tol_{tol_key}"] = 0
            out[f"hit@k_tol_{tol_key}"] = 0
            out[f"recall_tol_{tol_key}"] = 0.0
        out["mean_min_distance"] = None
        return out

    if not pred_times:
        out = {}
        for tol in tolerances:
            tol_key = str(tol).replace(".", "_")
            out[f"hit@1_tol_{tol_key}"] = 0
            out[f"hit@k_tol_{tol_key}"] = 0
            out[f"recall_tol_{tol_key}"] = 0.0
        out["mean_min_distance"] = float("inf")
        return out

    def min_dist_to_support(t: float) -> float:
        return min(abs(t - s) for s in support_times)

    out = {}

    top1_time = pred_times[0] if pred_times else None

    for tol in tolerances:
        tol_key = str(tol).replace(".", "_")

        hit1 = 0
        if top1_time is not None and min_dist_to_support(top1_time) <= tol:
            hit1 = 1

        hitk = 1 if any(min_dist_to_support(t) <= tol for t in pred_times) else 0

        covered = sum(
            1 for s in support_times
            if any(abs(t - s) <= tol for t in pred_times)
        )
        recall = covered / len(support_times) if support_times else 0.0

        out[f"hit@1_tol_{tol_key}"] = hit1
        out[f"hit@k_tol_{tol_key}"] = hitk
        out[f"recall_tol_{tol_key}"] = recall

    mean_min_distance = sum(min_dist_to_support(t) for t in pred_times) / len(pred_times)
    out["mean_min_distance"] = mean_min_distance

    return out


def make_selector_result_json_safe(selector_result: dict) -> dict:
    def strip_image_from_frame_list(frames: list[dict]) -> list[dict]:
        out = []
        for fr in frames:
            item = {}
            for k, v in fr.items():
                if k == "image":
                    continue
                item[k] = v
            out.append(item)
        return out

    return {
        "video_path": selector_result.get("video_path"),
        "sampled_frames": selector_result.get("sampled_frames"),
        "selected_frames": strip_image_from_frame_list(selector_result.get("selected_frames", [])),
        "all_frames": strip_image_from_frame_list(selector_result.get("all_frames", [])),
        "coarse_peaks": selector_result.get("coarse_peaks", []),
    }

if __name__ == "__main__":
    main()