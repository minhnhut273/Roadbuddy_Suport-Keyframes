"""
Metrics for evaluating weight quality.

Focus on ranking metrics (not regression metrics) since we care about
whether support frames rank higher, not their absolute scores.
"""

from __future__ import annotations

import numpy as np
from typing import Any


def hit_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 1) -> float:
    if len(y_true) == 0:
        return 0.0
    top_k_idx = np.argsort(-y_scores)[:k]
    return float(1.0 if np.any(y_true[top_k_idx] == 1) else 0.0)


def mean_reciprocal_rank(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Reciprocal rank of the highest-ranked support frame.
    Standard MRR for one query/sample:
      MRR = 1 / rank_of_first_relevant_item
    """
    support_indices = np.where(y_true == 1)[0]
    if len(support_indices) == 0:
        return 0.0

    ranking = np.argsort(-y_scores)

    # map item index -> 1-based rank
    rank_map = {int(idx): pos + 1 for pos, idx in enumerate(ranking)}

    support_ranks = [rank_map[int(idx)] for idx in support_indices]
    first_relevant_rank = min(support_ranks)

    return float(1.0 / first_relevant_rank)


def average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Average Precision (AP) - area under precision-recall curve.
    
    Args:
        y_true: binary labels
        y_scores: predicted scores
    
    Returns:
        AP (0.0 to 1.0)
    """
    num_support = np.sum(y_true)
    if num_support == 0:
        return 0.0
    
    # Sort by scores (descending)
    sorted_idx = np.argsort(-y_scores)
    sorted_y = y_true[sorted_idx]
    
    # Compute precision at each recall level
    precisions = []
    for i, label in enumerate(sorted_y):
        if label == 1:
            precision_at_i = np.sum(sorted_y[:i+1]) / (i + 1)
            precisions.append(precision_at_i)
    
    return float(np.mean(precisions) if precisions else 0.0)

def mean_min_distance(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """
    Mean rank position of support frames in descending score order.
    Lower is better.
    """
    support_indices = np.where(y_true == 1)[0]
    if len(support_indices) == 0:
        return 0.0

    ranking = np.argsort(-y_scores)

    # map item index -> rank position
    rank_map = {int(idx): int(pos) for pos, idx in enumerate(ranking)}

    support_ranks = [rank_map[int(idx)] for idx in support_indices]
    return float(np.mean(support_ranks)) if support_ranks else 0.0



def pairwise_accuracy(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """
    For each (support, non-support) pair, check if support scores higher.
    
    Args:
        y_true: binary labels
        y_scores: predicted scores
    
    Returns:
        Accuracy (0.0 to 1.0)
    """
    support_idx = np.where(y_true == 1)[0]
    non_support_idx = np.where(y_true == 0)[0]
    
    if len(support_idx) == 0 or len(non_support_idx) == 0:
        return 0.0
    
    correct = 0
    total = 0
    
    for s_idx in support_idx:
        for ns_idx in non_support_idx:
            if y_scores[s_idx] > y_scores[ns_idx]:
                correct += 1
            total += 1
    
    return float(correct / total) if total > 0 else 0.0




def nearest_candidate_index(y_scores: np.ndarray) -> int:
    """Index of top-1 candidate by score."""
    if len(y_scores) == 0:
        return -1
    return int(np.argsort(-y_scores)[0])


def asymmetric_nearest_distance(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    times: np.ndarray,
    after_penalty: float = 1.5,
) -> float:
    """
    Distance from top-1 predicted frame to nearest support frame, with asymmetric penalty:
    - before support: normal penalty
    - after support: larger penalty

    Lower is better.
    """
    support_idx = np.where(y_true == 1)[0]
    if len(support_idx) == 0 or len(y_scores) == 0:
        return 0.0

    pred_idx = nearest_candidate_index(y_scores)
    pred_time = float(times[pred_idx])

    support_times = [float(times[i]) for i in support_idx]

    best_cost = None
    for st in support_times:
        delta = pred_time - st
        if delta <= 0:
            cost = abs(delta)               # before: phạt bình thường
        else:
            cost = after_penalty * abs(delta)  # after: phạt nặng hơn

        if best_cost is None or cost < best_cost:
            best_cost = cost

    return float(best_cost if best_cost is not None else 0.0)

















def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> dict[str, float]:
    """
    Compute all ranking metrics.
    
    Args:
        y_true: binary labels
        y_scores: predicted scores
    
    Returns:
        dict with all metrics
    """
    return {
        "hit@1": hit_at_k(y_true, y_scores, k=1),
        "hit@3": hit_at_k(y_true, y_scores, k=3),
        "hit@5": hit_at_k(y_true, y_scores, k=5),
        "mrr": mean_reciprocal_rank(y_true, y_scores),
        "ap": average_precision(y_true, y_scores),
        "mean_min_distance": mean_min_distance(y_true, y_scores),
        "pairwise_accuracy": pairwise_accuracy(y_true, y_scores),
    }


def compute_metrics_with_times(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    times: np.ndarray,
    after_penalty: float = 1.5,
) -> dict[str, float]:
    """
    Compute ranking metrics + asymmetric nearest distance.
    """
    out = compute_metrics(y_true, y_scores)
    out["asym_nearest_distance"] = asymmetric_nearest_distance(
        y_true=y_true,
        y_scores=y_scores,
        times=times,
        after_penalty=after_penalty,
    )
    return out


def format_metrics(metrics: dict[str, float]) -> str:
    """Pretty print metrics"""
    lines = []
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:20s}: {v:.4f}")
        else:
            lines.append(f"  {k:20s}: {v}")
    return "\n".join(lines)
