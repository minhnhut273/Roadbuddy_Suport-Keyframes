from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math


@dataclass
class FrameScoreComponents:
    sharpness: float
    edge_density: float
    brightness: float
    novelty: float
    center_bias: float


@dataclass
class ScoredFrame:
    frame_idx: int
    time_sec: float
    image: any
    components: FrameScoreComponents
    score: float


def minmax_normalize(values: Sequence[float]) -> list[float]:
    values = list(values)
    if not values:
        return []

    vmin = min(values)
    vmax = max(values)

    if math.isclose(vmin, vmax):
        return [0.5 for _ in values]

    return [(v - vmin) / (vmax - vmin) for v in values]


def compute_center_bias(index: int, total: int) -> float:
    if total <= 1:
        return 1.0
    x = index / (total - 1)
    return 1.0 - abs(x - 0.5) * 2.0


def weighted_score(
    sharpness: float,
    edge_density: float,
    brightness: float,
    novelty: float,
    center_bias: float,
    weights: dict[str, float],
) -> float:
    return float(
        weights.get("sharpness", 0.0) * sharpness
        + weights.get("edge_density", 0.0) * edge_density
        + weights.get("brightness", 0.0) * brightness
        + weights.get("novelty", 0.0) * novelty
        + weights.get("center_bias", 0.0) * center_bias
    )