from __future__ import annotations
from typing import Sequence


def hit_at_k(pred_times: Sequence[float], gt_times: Sequence[float], tol: float = 0.5) -> int:
    return int(any(abs(p - g) <= tol for p in pred_times for g in gt_times))


def mean_min_distance(pred_times: Sequence[float], gt_times: Sequence[float]) -> float | None:
    pred_times = list(pred_times)
    gt_times = list(gt_times)
    if not pred_times or not gt_times:
        return None

    distances = []
    for p in pred_times:
        distances.append(min(abs(p - g) for g in gt_times))
    return float(sum(distances) / len(distances))


def recall_at_tol(pred_times: Sequence[float], gt_times: Sequence[float], tol: float = 0.5) -> float:
    gt_times = list(gt_times)
    if not gt_times:
        return 0.0

    covered = 0
    for g in gt_times:
        if any(abs(p - g) <= tol for p in pred_times):
            covered += 1
    return covered / len(gt_times)


def evaluate_selection(
    pred_times: Sequence[float],
    support_frames: Sequence[float],
    tolerances: Sequence[float],
) -> dict:
    out = {}

    pred_times = list(pred_times)
    support_frames = list(support_frames)

    for tol in tolerances:
        tol_str = str(tol).replace(".", "_")
        out[f"hit@1_tol_{tol_str}"] = hit_at_k(pred_times[:1], support_frames, tol=tol)
        out[f"hit@k_tol_{tol_str}"] = hit_at_k(pred_times, support_frames, tol=tol)
        out[f"recall_tol_{tol_str}"] = recall_at_tol(pred_times, support_frames, tol=tol)

    out["mean_min_distance"] = mean_min_distance(pred_times, support_frames)
    return out