# from __future__ import annotations

# import argparse
# import json
# from pathlib import Path
# from typing import Any
# import csv

# import cv2
# import numpy as np


# def read_json(path: Path) -> dict[str, Any]:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def ensure_dir(path: Path) -> None:
#     path.mkdir(parents=True, exist_ok=True)


# def extract_frame_at_time(video_path: str, time_sec: float):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return False, None, -1, -1.0

#     try:
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         if fps <= 0:
#             fps = 25.0

#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_idx = int(round(time_sec * fps))
#         frame_idx = max(0, min(frame_idx, max(total_frames - 1, 0)))

#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#         ok, frame = cap.read()
#         actual_time = frame_idx / fps
#         return ok, frame, frame_idx, actual_time
#     finally:
#         cap.release()


# def make_panel(
#     left_img,
#     right_img,
#     left_title: str,
#     right_title: str,
#     sample_id: str,
#     question_type: str,
#     question: str,
# ):
#     h = max(left_img.shape[0], right_img.shape[0])
#     w1 = int(left_img.shape[1] * h / left_img.shape[0])
#     w2 = int(right_img.shape[1] * h / right_img.shape[0])

#     left_resized = cv2.resize(left_img, (w1, h))
#     right_resized = cv2.resize(right_img, (w2, h))

#     gap = 20
#     header_h = 120
#     footer_h = 80
#     canvas_h = header_h + h + footer_h
#     canvas_w = w1 + gap + w2

#     canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

#     canvas[header_h:header_h + h, 0:w1] = left_resized
#     canvas[header_h:header_h + h, w1 + gap:w1 + gap + w2] = right_resized

#     cv2.putText(canvas, f"{sample_id} | {question_type}", (20, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

#     question_line = question[:110]
#     cv2.putText(canvas, question_line, (20, 65),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)

#     cv2.putText(canvas, left_title, (20, header_h - 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 70, 180), 2, cv2.LINE_AA)
#     cv2.putText(canvas, right_title, (w1 + gap + 20, header_h - 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 120, 60), 2, cv2.LINE_AA)

#     return canvas


# def safe_div(a: float, b: float) -> float:
#     return a / b if b else 0.0


# def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
#     if not rows:
#         return {"num_samples": 0}

#     def avg(key: str) -> float:
#         vals = [float(r[key]) for r in rows if r.get(key) is not None]
#         return sum(vals) / len(vals) if vals else 0.0

#     total_pairs = sum(int(r["num_support_used"]) for r in rows)

#     before = sum(int(r["before_count"]) for r in rows)
#     after = sum(int(r["after_count"]) for r in rows)
#     exact = sum(int(r["exact_count"]) for r in rows)

#     return {
#         "num_samples": len(rows),
#         "num_support_pairs": total_pairs,
#         "metrics_avg": {
#             "hit@1_tol_0_25": avg("hit@1_tol_0_25"),
#             "hit@k_tol_0_25": avg("hit@k_tol_0_25"),
#             "recall_tol_0_25": avg("recall_tol_0_25"),
#             "hit@1_tol_0_5": avg("hit@1_tol_0_5"),
#             "hit@k_tol_0_5": avg("hit@k_tol_0_5"),
#             "recall_tol_0_5": avg("recall_tol_0_5"),
#             "hit@1_tol_1_0": avg("hit@1_tol_1_0"),
#             "hit@k_tol_1_0": avg("hit@k_tol_1_0"),
#             "recall_tol_1_0": avg("recall_tol_1_0"),
#             "mean_min_distance": avg("mean_min_distance"),
#         },
#         "nearest_candidate_position_vs_support": {
#             "before_count": before,
#             "after_count": after,
#             "exact_count": exact,
#             "before_pct": safe_div(before, total_pairs),
#             "after_pct": safe_div(after, total_pairs),
#             "exact_pct": safe_div(exact, total_pairs),
#         },
#     }


# def main():
#     parser = argparse.ArgumentParser(
#         description="Summarize full eval by question type and compare nearest candidate vs support."
#     )
#     parser.add_argument("--output_dir", required=True, help="e.g. outputs_batch_full_updated")
#     parser.add_argument("--pairs_out_dir", required=True, help="where to save support-vs-nearest-candidate image pairs")
#     parser.add_argument(
#         "--candidate_source",
#         choices=["selected_frames", "all_frames"],
#         default="selected_frames",
#         help="Use final selected candidates or all candidate frames",
#     )
#     args = parser.parse_args()

#     output_dir = Path(args.output_dir)
#     pairs_out_dir = Path(args.pairs_out_dir)
#     ensure_dir(pairs_out_dir)

#     result_files = sorted(output_dir.glob("train_*/result.json"))
#     if not result_files:
#         print(f"❌ No result.json found in {output_dir}")
#         return

#     rows: list[dict[str, Any]] = []
#     rows_by_type: dict[str, list[dict[str, Any]]] = {}

#     for rf in result_files:
#         data = read_json(rf)

#         sample = data.get("sample", {})
#         selector_result = data.get("selector_result", {})
#         evaluation = data.get("evaluation", {})

#         sample_id = sample.get("id", "unknown")
#         qtype = sample.get("type", "other")
#         question = sample.get("question", "")
#         support_times = [float(x) for x in sample.get("support_frames", [])]
#         video_path = sample.get("video_abspath") or selector_result.get("video_path")

#         candidate_frames = selector_result.get(args.candidate_source, [])
#         candidate_times = [float(x["time_sec"]) for x in candidate_frames]

#         before_count = 0
#         after_count = 0
#         exact_count = 0
#         min_abs_dist = None
#         nearest_candidate_time_global = None
#         nearest_support_time_global = None

#         pair_dir = pairs_out_dir / sample_id
#         ensure_dir(pair_dir)

#         used_support = 0

#         for i, st in enumerate(support_times, start=1):
#             if not candidate_times:
#                 continue

#             nearest_idx = min(range(len(candidate_times)), key=lambda j: abs(candidate_times[j] - st))
#             cand_time = candidate_times[nearest_idx]
#             delta = cand_time - st
#             abs_delta = abs(delta)

#             if min_abs_dist is None or abs_delta < min_abs_dist:
#                 min_abs_dist = abs_delta
#                 nearest_candidate_time_global = cand_time
#                 nearest_support_time_global = st

#             if delta < 0:
#                 before_count += 1
#                 rel = "before"
#             elif delta > 0:
#                 after_count += 1
#                 rel = "after"
#             else:
#                 exact_count += 1
#                 rel = "exact"

#             if video_path:
#                 ok_s, support_img, s_idx, s_actual = extract_frame_at_time(video_path, st)
#                 ok_c, cand_img, c_idx, c_actual = extract_frame_at_time(video_path, cand_time)

#                 if ok_s and ok_c and support_img is not None and cand_img is not None:
#                     panel = make_panel(
#                         left_img=support_img,
#                         right_img=cand_img,
#                         left_title=f"Support | gt={st:.3f}s | actual={s_actual:.3f}s | idx={s_idx}",
#                         right_title=f"Nearest candidate ({rel}) | pred={cand_time:.3f}s | actual={c_actual:.3f}s | idx={c_idx}",
#                         sample_id=sample_id,
#                         question_type=qtype,
#                         question=question,
#                     )
#                     out_name = f"{sample_id}_support{i}_{rel}_delta_{delta:+.3f}.jpg"
#                     cv2.imwrite(str(pair_dir / out_name), panel)

#             used_support += 1

#         row = {
#             "id": sample_id,
#             "type": qtype,
#             "question": question,
#             "num_support_used": used_support,
#             "nearest_support_time": nearest_support_time_global,
#             "nearest_candidate_time": nearest_candidate_time_global,
#             "nearest_abs_delta": min_abs_dist if min_abs_dist is not None else None,
#             "before_count": before_count,
#             "after_count": after_count,
#             "exact_count": exact_count,
#             "hit@1_tol_0_25": evaluation.get("hit@1_tol_0_25", 0),
#             "hit@k_tol_0_25": evaluation.get("hit@k_tol_0_25", 0),
#             "recall_tol_0_25": evaluation.get("recall_tol_0_25", 0.0),
#             "hit@1_tol_0_5": evaluation.get("hit@1_tol_0_5", 0),
#             "hit@k_tol_0_5": evaluation.get("hit@k_tol_0_5", 0),
#             "recall_tol_0_5": evaluation.get("recall_tol_0_5", 0.0),
#             "hit@1_tol_1_0": evaluation.get("hit@1_tol_1_0", 0),
#             "hit@k_tol_1_0": evaluation.get("hit@k_tol_1_0", 0),
#             "recall_tol_1_0": evaluation.get("recall_tol_1_0", 0.0),
#             "mean_min_distance": evaluation.get("mean_min_distance", 0.0),
#         }
#         rows.append(row)
#         rows_by_type.setdefault(qtype, []).append(row)

#     # Save CSV
#     csv_path = output_dir / "summary_by_type_and_nearest_direction.csv"
#     with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
#         writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#         writer.writeheader()
#         writer.writerows(rows)

#     # Save global + per-type JSON
#     summary = {
#         "candidate_source": args.candidate_source,
#         "num_samples": len(rows),
#         "global_summary": summarize_rows(rows),
#         "summary_by_type": {
#             qtype: summarize_rows(type_rows)
#             for qtype, type_rows in rows_by_type.items()
#         },
#         "csv_file": str(csv_path),
#         "pairs_out_dir": str(pairs_out_dir),
#     }

#     json_path = output_dir / "summary_by_type_and_nearest_direction.json"
#     with json_path.open("w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2, ensure_ascii=False)

#     print("=" * 90)
#     print("DONE")
#     print(f"Saved CSV   : {csv_path}")
#     print(f"Saved JSON  : {json_path}")
#     print(f"Saved pairs : {pairs_out_dir}")
#     print("=" * 90)

#     global_pos = summary["global_summary"]["nearest_candidate_position_vs_support"]
#     print("GLOBAL nearest-candidate-vs-support:")
#     print(f"  before: {global_pos['before_count']} ({global_pos['before_pct']:.2%})")
#     print(f"  after : {global_pos['after_count']} ({global_pos['after_pct']:.2%})")
#     print(f"  exact : {global_pos['exact_count']} ({global_pos['exact_pct']:.2%})")

#     print("\nBY TYPE:")
#     for qtype, info in summary["summary_by_type"].items():
#         pos = info["nearest_candidate_position_vs_support"]
#         m = info["metrics_avg"]
#         print(f"- {qtype}")
#         print(f"    samples={info['num_samples']}, support_pairs={info['num_support_pairs']}")
#         print(f"    hit@k@0.5={m['hit@k_tol_0_5']:.4f}, hit@k@1.0={m['hit@k_tol_1_0']:.4f}, mean_min_distance={m['mean_min_distance']:.4f}")
#         print(f"    before={pos['before_pct']:.2%}, after={pos['after_pct']:.2%}, exact={pos['exact_pct']:.2%}")


# if __name__ == "__main__":
#     main()


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import csv

import cv2
import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def make_panel(
    left_img,
    right_img,
    left_title: str,
    right_title: str,
    sample_id: str,
    question_type: str,
    question: str,
):
    h = max(left_img.shape[0], right_img.shape[0])
    w1 = int(left_img.shape[1] * h / left_img.shape[0])
    w2 = int(right_img.shape[1] * h / right_img.shape[0])

    left_resized = cv2.resize(left_img, (w1, h))
    right_resized = cv2.resize(right_img, (w2, h))

    gap = 20
    header_h = 120
    footer_h = 80
    canvas_h = header_h + h + footer_h
    canvas_w = w1 + gap + w2

    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    canvas[header_h:header_h + h, 0:w1] = left_resized
    canvas[header_h:header_h + h, w1 + gap:w1 + gap + w2] = right_resized

    cv2.putText(canvas, f"{sample_id} | {question_type}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

    question_line = question[:110]
    cv2.putText(canvas, question_line, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)

    cv2.putText(canvas, left_title, (20, header_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 70, 180), 2, cv2.LINE_AA)
    cv2.putText(canvas, right_title, (w1 + gap + 20, header_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 120, 60), 2, cv2.LINE_AA)

    return canvas


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def safe_mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def safe_median(vals: list[float]) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    n = len(vals)
    if n % 2 == 1:
        return float(vals[n // 2])
    return float((vals[n // 2 - 1] + vals[n // 2]) / 2.0)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_samples": 0}

    def avg(key: str) -> float:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    total_pairs = sum(int(r["num_support_used"]) for r in rows)

    before = sum(int(r["before_count"]) for r in rows)
    after = sum(int(r["after_count"]) for r in rows)
    exact = sum(int(r["exact_count"]) for r in rows)

    nearest_abs_vals = []
    best_abs_vals = []

    best_before = 0
    best_after = 0
    best_exact = 0

    for r in rows:
        nearest_abs_list = r.get("nearest_abs_deltas", [])
        best_abs_list = r.get("best_abs_deltas", [])
        nearest_abs_vals.extend(nearest_abs_list)
        best_abs_vals.extend(best_abs_list)

        best_before += int(r.get("best_before_count", 0))
        best_after += int(r.get("best_after_count", 0))
        best_exact += int(r.get("best_exact_count", 0))

    return {
        "num_samples": len(rows),
        "num_support_pairs": total_pairs,
        "metrics_avg": {
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
        "nearest_candidate_position_vs_support": {
            "before_count": before,
            "after_count": after,
            "exact_count": exact,
            "before_pct": safe_div(before, total_pairs),
            "after_pct": safe_div(after, total_pairs),
            "exact_pct": safe_div(exact, total_pairs),
        },
        "nearest_support_distance_stats": {
            "mean_nearest_support_distance": safe_mean(nearest_abs_vals),
            "median_nearest_support_distance": safe_median(nearest_abs_vals),
        },
        "best_frame_vs_support": {
            "before_count": best_before,
            "after_count": best_after,
            "exact_count": best_exact,
            "before_pct": safe_div(best_before, total_pairs),
            "after_pct": safe_div(best_after, total_pairs),
            "exact_pct": safe_div(best_exact, total_pairs),
            "mean_best_frame_distance": safe_mean(best_abs_vals),
            "median_best_frame_distance": safe_median(best_abs_vals),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize full eval by question type and compare nearest candidate vs support."
    )
    parser.add_argument("--output_dir", required=True, help="e.g. outputs_batch_full_updated")
    parser.add_argument("--pairs_out_dir", required=True, help="where to save support-vs-nearest-candidate image pairs")
    parser.add_argument(
        "--candidate_source",
        choices=["selected_frames", "all_frames"],
        default="selected_frames",
        help="Use final selected candidates or all candidate frames",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pairs_out_dir = Path(args.pairs_out_dir)
    ensure_dir(pairs_out_dir)

    result_files = sorted(output_dir.glob("train_*/result.json"))
    if not result_files:
        print(f"No result.json found in {output_dir}")
        return

    rows: list[dict[str, Any]] = []
    rows_by_type: dict[str, list[dict[str, Any]]] = {}

    for rf in result_files:
        data = read_json(rf)

        sample = data.get("sample", {})
        selector_result = data.get("selector_result", {})
        evaluation = data.get("evaluation", {})

        sample_id = sample.get("id", "unknown")
        qtype = sample.get("type", "other")
        question = sample.get("question", "")
        support_times = [float(x) for x in sample.get("support_frames", [])]
        video_path = sample.get("video_abspath") or selector_result.get("video_path")

        candidate_frames = selector_result.get(args.candidate_source, [])
        candidate_times = [float(x["time_sec"]) for x in candidate_frames]

        before_count = 0
        after_count = 0
        exact_count = 0

        best_before_count = 0
        best_after_count = 0
        best_exact_count = 0

        nearest_candidate_time_global = None
        nearest_support_time_global = None
        min_abs_dist = None

        nearest_abs_deltas = []
        best_abs_deltas = []

        pair_dir = pairs_out_dir / sample_id
        ensure_dir(pair_dir)

        used_support = 0

        best_time = float(candidate_times[0]) if candidate_times else None

        for i, st in enumerate(support_times, start=1):
            if not candidate_times:
                continue

            nearest_idx = min(range(len(candidate_times)), key=lambda j: abs(candidate_times[j] - st))
            cand_time = candidate_times[nearest_idx]
            delta = cand_time - st
            abs_delta = abs(delta)
            nearest_abs_deltas.append(abs_delta)

            if min_abs_dist is None or abs_delta < min_abs_dist:
                min_abs_dist = abs_delta
                nearest_candidate_time_global = cand_time
                nearest_support_time_global = st

            if delta < 0:
                before_count += 1
                rel = "before"
            elif delta > 0:
                after_count += 1
                rel = "after"
            else:
                exact_count += 1
                rel = "exact"

            if best_time is not None:
                best_delta = best_time - st
                best_abs_deltas.append(abs(best_delta))
                if best_delta < 0:
                    best_before_count += 1
                elif best_delta > 0:
                    best_after_count += 1
                else:
                    best_exact_count += 1

            if video_path:
                ok_s, support_img, s_idx, s_actual = extract_frame_at_time(video_path, st)
                ok_c, cand_img, c_idx, c_actual = extract_frame_at_time(video_path, cand_time)

                if ok_s and ok_c and support_img is not None and cand_img is not None:
                    panel = make_panel(
                        left_img=support_img,
                        right_img=cand_img,
                        left_title=f"Support | gt={st:.3f}s | actual={s_actual:.3f}s | idx={s_idx}",
                        right_title=f"Nearest candidate ({rel}) | pred={cand_time:.3f}s | actual={c_actual:.3f}s | idx={c_idx}",
                        sample_id=sample_id,
                        question_type=qtype,
                        question=question,
                    )
                    out_name = f"{sample_id}_support{i}_{rel}_delta_{delta:+.3f}.jpg"
                    cv2.imwrite(str(pair_dir / out_name), panel)

            used_support += 1

        row = {
            "id": sample_id,
            "type": qtype,
            "question": question,
            "num_support_used": used_support,
            "nearest_support_time": nearest_support_time_global,
            "nearest_candidate_time": nearest_candidate_time_global,
            "nearest_abs_delta": min_abs_dist if min_abs_dist is not None else None,
            "nearest_abs_deltas": nearest_abs_deltas,
            "best_abs_deltas": best_abs_deltas,
            "before_count": before_count,
            "after_count": after_count,
            "exact_count": exact_count,
            "best_before_count": best_before_count,
            "best_after_count": best_after_count,
            "best_exact_count": best_exact_count,
            "hit@1_tol_0_25": evaluation.get("hit@1_tol_0_25", 0),
            "hit@k_tol_0_25": evaluation.get("hit@k_tol_0_25", 0),
            "recall_tol_0_25": evaluation.get("recall_tol_0_25", 0.0),
            "hit@1_tol_0_5": evaluation.get("hit@1_tol_0_5", 0),
            "hit@k_tol_0_5": evaluation.get("hit@k_tol_0_5", 0),
            "recall_tol_0_5": evaluation.get("recall_tol_0_5", 0.0),
            "hit@1_tol_1_0": evaluation.get("hit@1_tol_1_0", 0),
            "hit@k_tol_1_0": evaluation.get("hit@k_tol_1_0", 0),
            "recall_tol_1_0": evaluation.get("recall_tol_1_0", 0.0),
            "mean_min_distance": evaluation.get("mean_min_distance", 0.0),
        }
        rows.append(row)
        rows_by_type.setdefault(qtype, []).append(row)

    csv_path = output_dir / "summary_by_type_and_nearest_direction.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "type", "question", "num_support_used",
                "nearest_support_time", "nearest_candidate_time", "nearest_abs_delta",
                "before_count", "after_count", "exact_count",
                "best_before_count", "best_after_count", "best_exact_count",
                "hit@1_tol_0_25", "hit@k_tol_0_25", "recall_tol_0_25",
                "hit@1_tol_0_5", "hit@k_tol_0_5", "recall_tol_0_5",
                "hit@1_tol_1_0", "hit@k_tol_1_0", "recall_tol_1_0",
                "mean_min_distance",
            ]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "id": r["id"],
                "type": r["type"],
                "question": r["question"],
                "num_support_used": r["num_support_used"],
                "nearest_support_time": r["nearest_support_time"],
                "nearest_candidate_time": r["nearest_candidate_time"],
                "nearest_abs_delta": r["nearest_abs_delta"],
                "before_count": r["before_count"],
                "after_count": r["after_count"],
                "exact_count": r["exact_count"],
                "best_before_count": r["best_before_count"],
                "best_after_count": r["best_after_count"],
                "best_exact_count": r["best_exact_count"],
                "hit@1_tol_0_25": r["hit@1_tol_0_25"],
                "hit@k_tol_0_25": r["hit@k_tol_0_25"],
                "recall_tol_0_25": r["recall_tol_0_25"],
                "hit@1_tol_0_5": r["hit@1_tol_0_5"],
                "hit@k_tol_0_5": r["hit@k_tol_0_5"],
                "recall_tol_0_5": r["recall_tol_0_5"],
                "hit@1_tol_1_0": r["hit@1_tol_1_0"],
                "hit@k_tol_1_0": r["hit@k_tol_1_0"],
                "recall_tol_1_0": r["recall_tol_1_0"],
                "mean_min_distance": r["mean_min_distance"],
            })

    summary = {
        "candidate_source": args.candidate_source,
        "num_samples": len(rows),
        "global_summary": summarize_rows(rows),
        "summary_by_type": {
            qtype: summarize_rows(type_rows)
            for qtype, type_rows in rows_by_type.items()
        },
        "csv_file": str(csv_path),
        "pairs_out_dir": str(pairs_out_dir),
    }

    json_path = output_dir / "summary_by_type_and_nearest_direction.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 90)
    print("DONE")
    print(f"Saved CSV   : {csv_path}")
    print(f"Saved JSON  : {json_path}")
    print(f"Saved pairs : {pairs_out_dir}")
    print("=" * 90)

    global_pos = summary["global_summary"]["nearest_candidate_position_vs_support"]
    best_pos = summary["global_summary"]["best_frame_vs_support"]

    print("GLOBAL nearest-candidate-vs-support:")
    print(f"  before: {global_pos['before_count']} ({global_pos['before_pct']:.2%})")
    print(f"  after : {global_pos['after_count']} ({global_pos['after_pct']:.2%})")
    print(f"  exact : {global_pos['exact_count']} ({global_pos['exact_pct']:.2%})")
    print(f"  mean_nearest_support_distance   : {summary['global_summary']['nearest_support_distance_stats']['mean_nearest_support_distance']:.4f}")
    print(f"  median_nearest_support_distance : {summary['global_summary']['nearest_support_distance_stats']['median_nearest_support_distance']:.4f}")

    print("GLOBAL best-frame-vs-support:")
    print(f"  before: {best_pos['before_count']} ({best_pos['before_pct']:.2%})")
    print(f"  after : {best_pos['after_count']} ({best_pos['after_pct']:.2%})")
    print(f"  exact : {best_pos['exact_count']} ({best_pos['exact_pct']:.2%})")
    print(f"  mean_best_frame_distance   : {best_pos['mean_best_frame_distance']:.4f}")
    print(f"  median_best_frame_distance : {best_pos['median_best_frame_distance']:.4f}")

    print("\nBY TYPE:")
    for qtype, info in summary["summary_by_type"].items():
        pos = info["nearest_candidate_position_vs_support"]
        best = info["best_frame_vs_support"]
        m = info["metrics_avg"]
        nearest_stats = info["nearest_support_distance_stats"]

        print(f"- {qtype}")
        print(f"    samples={info['num_samples']}, support_pairs={info['num_support_pairs']}")
        print(f"    hit@k@0.5={m['hit@k_tol_0_5']:.4f}, hit@k@1.0={m['hit@k_tol_1_0']:.4f}, mean_min_distance={m['mean_min_distance']:.4f}")
        print(f"    nearest before={pos['before_pct']:.2%}, after={pos['after_pct']:.2%}, exact={pos['exact_pct']:.2%}")
        print(f"    mean_nearest_support_distance={nearest_stats['mean_nearest_support_distance']:.4f}, median={nearest_stats['median_nearest_support_distance']:.4f}")
        print(f"    best before={best['before_pct']:.2%}, after={best['after_pct']:.2%}, exact={best['exact_pct']:.2%}")
        print(f"    mean_best_frame_distance={best['mean_best_frame_distance']:.4f}, median={best['median_best_frame_distance']:.4f}")


if __name__ == "__main__":
    main()