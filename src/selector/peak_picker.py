from __future__ import annotations
from typing import Sequence


def temporal_nms(
    times_sec: Sequence[float],
    scores: Sequence[float],
    top_k: int,
    min_gap_sec: float,
) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []

    for idx in order:
        t = times_sec[idx]
        ok = True
        for kept_idx in keep:
            if abs(times_sec[kept_idx] - t) < min_gap_sec:
                ok = False
                break
        if ok:
            keep.append(idx)
        if len(keep) >= top_k:
            break

    return sorted(keep, key=lambda i: times_sec[i])