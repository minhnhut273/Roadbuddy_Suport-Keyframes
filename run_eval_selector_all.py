from __future__ import annotations
import argparse
from pathlib import Path
import csv
import yaml
import cv2

from src.data.train_loader import load_train_data, build_video_abspath, infer_question_type
from src.selector.pipeline import KeyframeSelectorPipeline
from src.eval.eval_selector import evaluate_selection
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
        cleaned["selected_frames"].append({k: v for k, v in fr.items() if k != "image"})

    for fr in obj["all_frames"]:
        cleaned["all_frames"].append({k: v for k, v in fr.items() if k != "image"})

    return cleaned


def mean_ignore_none(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/selector.yaml")
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--output_dir", default="outputs_batch")
    parser.add_argument("--limit", type=int, default=1450, help="Số mẫu chạy, mặc định 50")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--save_frames", action="store_true", help="Lưu top frame cho từng sample")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tolerances = cfg["eval_tolerances_sec"]
    pipeline = KeyframeSelectorPipeline(args.config)
    data = load_train_data(args.train_json)

    start = max(args.start_index, 0)
    end = min(start + args.limit, len(data))
    subset = data[start:end]

    rows = []
    metric_keys = []
    all_metric_values: dict[str, list[float | None]] = {}

    for idx, sample in enumerate(subset, start=start):
        sample_id = sample["id"]
        qtype = infer_question_type(sample)
        support_frames = sample.get("support_frames", [])
        video_path = build_video_abspath(args.video_root, sample["video_path"])

        print(f"[{idx}] Running {sample_id} ...")
        print(f"Resolved video path: {video_path}")

        result = pipeline.run(video_path, question_type=qtype)
        pred_times = [x["time_sec"] for x in result["selected_frames"]]
        metrics = evaluate_selection(
            pred_times=pred_times,
            support_frames=support_frames,
            tolerances=tolerances,
        )

        if not metric_keys:
            metric_keys = list(metrics.keys())
            for k in metric_keys:
                all_metric_values[k] = []

        for k in metric_keys:
            all_metric_values[k].append(metrics.get(k))

        sample_out_dir = output_dir / sample_id
        ensure_dir(sample_out_dir)

        if args.save_frames and cfg.get("save_top_frames", True):
            save_selected_images(result["selected_frames"], sample_out_dir / "frames")

        sample_json = {
            "sample": {
                "id": sample["id"],
                "question": sample["question"],
                "choices": sample["choices"],
                "answer": sample.get("answer"),
                "type": qtype,
                "video_path": sample["video_path"],
                "video_abspath": video_path,
                "support_frames": support_frames,
            },
            "selector_result": strip_images(result),
            "evaluation": metrics,
        }
        write_json(sample_json, sample_out_dir / "result.json")

        row = {
            "sample_index": idx,
            "sample_id": sample_id,
            "type": qtype,
            "video_path": sample["video_path"],
            "support_frames": str(support_frames),
            "pred_times": str(pred_times),
            "question": sample["question"],
        }
        row.update(metrics)
        rows.append(row)

    summary = {
        "num_samples": len(rows),
        "start_index": start,
        "end_index_exclusive": end,
        "averages": {},
    }

    for k, vals in all_metric_values.items():
        summary["averages"][k] = mean_ignore_none(vals)

    write_json(summary, output_dir / "summary.json")

    csv_path = output_dir / "results.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "sample_index", "sample_id", "type", "video_path",
        "support_frames", "pred_times", "question"
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 90)
    print("BATCH EVALUATION DONE")
    print(f"Samples         : {len(rows)}")
    print(f"Saved CSV       : {csv_path}")
    print(f"Saved summary   : {output_dir / 'summary.json'}")
    print("Average metrics:")
    for k, v in summary["averages"].items():
        print(f"  {k}: {v}")

    print("Primary candidate-selector metrics:")
    for k in ["hit@k_tol_0_5", "hit@k_tol_1_0", "recall_tol_0_5", "mean_min_distance"]:
        if k in summary["averages"]:
            print(f"  {k}: {summary['averages'][k]}")

    print("=" * 90)


if __name__ == "__main__":
    main()
