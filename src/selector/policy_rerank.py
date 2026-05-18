from __future__ import annotations

from typing import Any
import cv2
import numpy as np


TYPE_TO_POLICY = {
    "sign_identification": "sign_policy",
    "information_reading": "sign_policy",
    "navigation": "sign_policy",
    "other": "sign_policy",

    "rule_compliance": "lane_policy",
    "verification": "lane_policy",

    "object_presence": "coverage_policy",
    "counting": "coverage_policy",
}


def get_policy(question_type: str | None) -> str:
    if not question_type:
        return "sign_policy"
    return TYPE_TO_POLICY.get(question_type, "sign_policy")


def normalize_list(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def crop_roi(img: np.ndarray, roi: list[float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = int(roi[0] * w)
    y1 = int(roi[1] * h)
    x2 = int(roi[2] * w)
    y2 = int(roi[3] * h)

    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return img[y1:y2, x1:x2]


def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 160)
    return float(np.mean(edges > 0))


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(gray: np.ndarray) -> float:
    return float(np.mean(gray) / 255.0)


def estimate_text_score(img: np.ndarray, roi_regions: list[list[float]]) -> float:
    vals = []
    for roi in roi_regions:
        patch = crop_roi(img, roi)
        if patch.size == 0:
            vals.append(0.0)
            continue
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        e = edge_density(gray)
        s = min(laplacian_sharpness(gray) / 500.0, 1.0)
        b = brightness_score(gray)
        vals.append(0.45 * e + 0.45 * s + 0.10 * b)
    return float(max(vals) if vals else 0.0)


def estimate_sign_area_score(img: np.ndarray, roi_regions: list[list[float]]) -> float:
    vals = []
    for roi in roi_regions:
        patch = crop_roi(img, roi)
        if patch.size == 0:
            vals.append(0.0)
            continue
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        area_ratio = float(np.mean(th > 0))

        target = 0.18
        score = 1.0 - min(abs(area_ratio - target) / target, 1.0)
        vals.append(score)
    return float(max(vals) if vals else 0.0)


def estimate_temporal_stability_score(
    frame: dict[str, Any],
    all_frames: list[dict[str, Any]],
    time_window_sec: float = 0.6,
) -> float:
    t0 = float(frame["time_sec"])
    neighbors = [
        f for f in all_frames
        if abs(float(f["time_sec"]) - t0) <= time_window_sec
    ]
    if len(neighbors) <= 1:
        return 0.0

    scores = [float(f.get("score", 0.0)) for f in neighbors]
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    return mean_score - 0.5 * std_score


def rerank_sign_policy(
    selected_frames: list[dict[str, Any]],
    all_frames: list[dict[str, Any]],
    roi_regions: list[list[float]],
    top_k: int,
    candidate_pool_size: int,
    local_window_sec: float,
    rerank_weights: dict[str, float],
) -> list[dict[str, Any]]:
    if not selected_frames:
        return selected_frames

    best_t = float(selected_frames[0]["time_sec"])
    local_pool = [
        f for f in all_frames
        if abs(float(f["time_sec"]) - best_t) <= local_window_sec
    ]

    candidate_pool = sorted(
        local_pool if local_pool else all_frames,
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True,
    )[:candidate_pool_size]

    visual_vals = []
    text_vals = []
    area_vals = []
    temp_vals = []

    for frame in candidate_pool:
        img = frame.get("image")
        visual_vals.append(float(frame.get("score", 0.0)))

        if img is None:
            text_vals.append(0.0)
            area_vals.append(0.0)
        else:
            text_vals.append(estimate_text_score(img, roi_regions))
            area_vals.append(estimate_sign_area_score(img, roi_regions))

        temp_vals.append(estimate_temporal_stability_score(frame, all_frames))

    visual_n = normalize_list(visual_vals)
    text_n = normalize_list(text_vals)
    area_n = normalize_list(area_vals)
    temp_n = normalize_list(temp_vals)

    wv = float(rerank_weights.get("visual_score", 0.55))
    wt = float(rerank_weights.get("text_score", 0.20))
    wa = float(rerank_weights.get("sign_area_score", 0.15))
    ws = float(rerank_weights.get("temporal_stability_score", 0.10))

    out = []
    for i, frame in enumerate(candidate_pool):
        final_score = (
            wv * visual_n[i]
            + wt * text_n[i]
            + wa * area_n[i]
            + ws * temp_n[i]
        )
        item = dict(frame)
        item["rerank_visual_score"] = visual_n[i]
        item["rerank_text_score"] = text_n[i]
        item["rerank_sign_area_score"] = area_n[i]
        item["rerank_temporal_stability_score"] = temp_n[i]
        item["rerank_final_score"] = final_score
        out.append(item)

    out.sort(key=lambda x: x["rerank_final_score"], reverse=True)
    return out[:top_k]


def rerank_frames_by_policy(
    question_type: str | None,
    selected_frames: list[dict[str, Any]],
    all_frames: list[dict[str, Any]],
    roi_regions: list[list[float]],
    top_k: int,
    rerank_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not rerank_cfg:
        return selected_frames
    if not rerank_cfg.get("enabled", False):
        return selected_frames

    policy = get_policy(question_type)
    policy_cfg = rerank_cfg.get("policy_rerank", {}).get(policy, {})
    if not policy_cfg.get("enabled", False):
        return selected_frames

    if policy == "sign_policy":
        return rerank_sign_policy(
            selected_frames=selected_frames,
            all_frames=all_frames,
            roi_regions=roi_regions,
            top_k=top_k,
            candidate_pool_size=int(policy_cfg.get("candidate_pool_size", 12)),
            local_window_sec=float(policy_cfg.get("local_window_sec", 1.2)),
            rerank_weights=policy_cfg.get("weights", {}),
        )

    return selected_frames