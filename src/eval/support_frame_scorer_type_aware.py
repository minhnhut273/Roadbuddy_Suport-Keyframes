from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.perception.blur import (
    laplacian_variance,
    canny_edge_density,
    brightness_score,
    novelty_score,
    roi_feature_max,
)
from src.selector.scorer import minmax_normalize, compute_center_bias


@dataclass
class _FrameFeatures:
    time_sec: float
    frame_idx: int
    raw_sharpness: float
    raw_edge_density: float
    raw_brightness: float
    raw_novelty: float
    raw_center_bias: float
    raw_roi_sharpness: float
    raw_roi_edge_density: float
    raw_roi_brightness: float


class SupportFrameScorer:
    """
    Type-aware support frame scorer.

    Điểm khác với phiên bản cũ:
    1. Dùng đúng policy theo question_type (type -> policy -> ROI/weights)
    2. Recompute RAW features cho candidate frames từ video + candidate times
    3. Normalize support + candidate trên cùng một thang raw feature
    4. Normalize ROI features trên thang ROI riêng, không dùng chung với global features
    """

    def __init__(self, config: dict[str, Any]):
        self.cfg = config

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "SupportFrameScorer":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    def _get_policy_config(self, question_type: str | None) -> dict[str, Any]:
        if not self.cfg.get("type_aware", False):
            return {
                "weights": self.cfg.get("weights", {}),
                "roi_regions": self.cfg.get("roi_regions", []),
            }

        type_to_policy = self.cfg.get("type_to_policy", {})
        policy_configs = self.cfg.get("policy_configs", {})
        policy_name = type_to_policy.get(question_type)

        if policy_name and policy_name in policy_configs:
            policy = policy_configs[policy_name]
            return {
                "weights": policy.get("weights", self.cfg.get("weights", {})),
                "roi_regions": policy.get("roi_regions", self.cfg.get("roi_regions", [])),
            }

        return {
            "weights": self.cfg.get("weights", {}),
            "roi_regions": self.cfg.get("roi_regions", []),
        }

    def _extract_frame_at_time(self, video_path: str | Path, time_sec: float) -> tuple[bool, Any, int, float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, None, -1, -1.0

        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if native_fps <= 0:
                native_fps = 25.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = int(round(time_sec * native_fps))
            if total_frames > 0:
                frame_idx = max(0, min(frame_idx, total_frames - 1))
            else:
                frame_idx = max(0, frame_idx)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            actual_time = frame_idx / native_fps
            return ret, frame, frame_idx, actual_time
        finally:
            cap.release()

    def _compute_features(
        self,
        video_path: str | Path,
        times_sec: list[float],
        roi_regions: list[list[float]],
    ) -> list[_FrameFeatures]:
        if not times_sec:
            return []

        extracted: list[tuple[float, int, Any]] = []
        for t in times_sec:
            ok, frame, frame_idx, actual_time = self._extract_frame_at_time(video_path, t)
            if not ok or frame is None:
                print(f"[WARN] Cannot extract frame at requested={t:.6f}s from {video_path}")
                continue
            extracted.append((actual_time, frame_idx, frame))

        if not extracted:
            return []

        extracted.sort(key=lambda x: x[0])
        results: list[_FrameFeatures] = []
        n = len(extracted)
        prev_frame = None

        for i, (actual_time, frame_idx, frame) in enumerate(extracted):
            results.append(
                _FrameFeatures(
                    time_sec=actual_time,
                    frame_idx=frame_idx,
                    raw_sharpness=laplacian_variance(frame),
                    raw_edge_density=canny_edge_density(frame),
                    raw_brightness=brightness_score(frame),
                    raw_novelty=novelty_score(frame, prev_frame),
                    raw_center_bias=compute_center_bias(i, n),
                    raw_roi_sharpness=roi_feature_max(frame, roi_regions, laplacian_variance),
                    raw_roi_edge_density=roi_feature_max(frame, roi_regions, canny_edge_density),
                    raw_roi_brightness=roi_feature_max(frame, roi_regions, brightness_score),
                )
            )
            prev_frame = frame

        return results

    def _normalize_jointly(
        self,
        support_features: list[_FrameFeatures],
        candidate_features: list[_FrameFeatures],
    ) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        all_feats = support_features + candidate_features
        if not all_feats:
            return [], []

        keys = [
            "raw_sharpness",
            "raw_edge_density",
            "raw_brightness",
            "raw_novelty",
            "raw_center_bias",
            "raw_roi_sharpness",
            "raw_roi_edge_density",
            "raw_roi_brightness",
        ]

        normalized_by_key: dict[str, list[float]] = {}
        for k in keys:
            values = [float(getattr(x, k)) for x in all_feats]
            normalized_by_key[k] = minmax_normalize(values)

        def pack(start: int, end: int) -> list[dict[str, float]]:
            rows: list[dict[str, float]] = []
            for idx in range(start, end):
                rows.append({
                    "sharpness": normalized_by_key["raw_sharpness"][idx],
                    "edge_density": normalized_by_key["raw_edge_density"][idx],
                    "brightness": normalized_by_key["raw_brightness"][idx],
                    "novelty": normalized_by_key["raw_novelty"][idx],
                    "center_bias": normalized_by_key["raw_center_bias"][idx],
                    "roi_sharpness": normalized_by_key["raw_roi_sharpness"][idx],
                    "roi_edge_density": normalized_by_key["raw_roi_edge_density"][idx],
                    "roi_brightness": normalized_by_key["raw_roi_brightness"][idx],
                })
            return rows

        s_norm = pack(0, len(support_features))
        c_norm = pack(len(support_features), len(all_feats))
        return s_norm, c_norm

    def _weighted_score(self, components: dict[str, float], weights: dict[str, float]) -> float:
        return float(
            weights.get("sharpness", 0.0) * components["sharpness"]
            + weights.get("edge_density", 0.0) * components["edge_density"]
            + weights.get("brightness", 0.0) * components["brightness"]
            + weights.get("novelty", 0.0) * components["novelty"]
            + weights.get("center_bias", 0.0) * components["center_bias"]
            + weights.get("roi_sharpness", 0.0) * components["roi_sharpness"]
            + weights.get("roi_edge_density", 0.0) * components["roi_edge_density"]
            + weights.get("roi_brightness", 0.0) * components["roi_brightness"]
        )

    def score_support_frames(
        self,
        video_path: str | Path,
        support_times: list[float],
        candidate_times: list[float] | None = None,
        question_type: str | None = None,
    ) -> dict[str, Any]:
        policy_cfg = self._get_policy_config(question_type)
        weights = policy_cfg.get("weights", {})
        roi_regions = policy_cfg.get("roi_regions", [])

        if not support_times:
            return {
                "support_frames": [],
                "candidate_frames": [],
                "comparison": {
                    "total_support_frames": 0,
                    "total_candidate_frames": 0,
                    "avg_support_score": 0.0,
                    "avg_candidate_score": 0.0,
                    "max_candidate_score": 0.0,
                    "min_candidate_score": 0.0,
                    "summary": "No support frames provided",
                },
            }

        support_features = self._compute_features(video_path, support_times, roi_regions)
        candidate_features = self._compute_features(video_path, candidate_times or [], roi_regions)

        if not support_features:
            return {
                "support_frames": [],
                "candidate_frames": [],
                "comparison": {
                    "total_support_frames": len(support_times),
                    "total_candidate_frames": len(candidate_times or []),
                    "avg_support_score": 0.0,
                    "avg_candidate_score": 0.0,
                    "max_candidate_score": 0.0,
                    "min_candidate_score": 0.0,
                    "summary": "Failed to extract any support frames",
                },
            }

        support_norm, cand_norm = self._normalize_jointly(support_features, candidate_features)

        final_support = []
        for feat, comps in zip(support_features, support_norm):
            final_support.append({
                "time_sec": feat.time_sec,
                "frame_idx": feat.frame_idx,
                "score": self._weighted_score(comps, weights),
                "components": comps,
                "raw_features": {
                    "sharpness": feat.raw_sharpness,
                    "edge_density": feat.raw_edge_density,
                    "brightness": feat.raw_brightness,
                    "novelty": feat.raw_novelty,
                    "center_bias": feat.raw_center_bias,
                    "roi_sharpness": feat.raw_roi_sharpness,
                    "roi_edge_density": feat.raw_roi_edge_density,
                    "roi_brightness": feat.raw_roi_brightness,
                },
            })

        final_candidates = []
        for feat, comps in zip(candidate_features, cand_norm):
            final_candidates.append({
                "time_sec": feat.time_sec,
                "frame_idx": feat.frame_idx,
                "score": self._weighted_score(comps, weights),
                "components": comps,
                "raw_features": {
                    "sharpness": feat.raw_sharpness,
                    "edge_density": feat.raw_edge_density,
                    "brightness": feat.raw_brightness,
                    "novelty": feat.raw_novelty,
                    "center_bias": feat.raw_center_bias,
                    "roi_sharpness": feat.raw_roi_sharpness,
                    "roi_edge_density": feat.raw_roi_edge_density,
                    "roi_brightness": feat.raw_roi_brightness,
                },
            })

        candidate_scores = [x["score"] for x in final_candidates]
        if candidate_scores:
            for sf in final_support:
                rank = sum(1 for cs in candidate_scores if cs >= sf["score"]) + 1
                sf["rank_in_candidates"] = rank
                sf["status"] = "excellent" if rank <= 2 else "good" if rank <= 4 else "weak"
        else:
            for sf in final_support:
                sf["rank_in_candidates"] = None
                sf["status"] = "standalone"

        support_scores = [x["score"] for x in final_support]
        avg_support = sum(support_scores) / len(support_scores) if support_scores else 0.0
        avg_candidate = sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0.0
        max_candidate = max(candidate_scores) if candidate_scores else 0.0
        min_candidate = min(candidate_scores) if candidate_scores else 0.0

        if not candidate_scores:
            summary = f"Support frames scored successfully with type-aware policy '{question_type}'."
        elif avg_support > max_candidate:
            summary = (
                f"✓ Avg support ({avg_support:.4f}) > max candidate ({max_candidate:.4f}). "
                f"Support looks stronger than candidate pool."
            )
        elif avg_support >= avg_candidate:
            summary = (
                f"~ Avg support ({avg_support:.4f}) >= avg candidate ({avg_candidate:.4f}), "
                f"but not above top candidate ({max_candidate:.4f})."
            )
        elif avg_support > min_candidate:
            summary = (
                f"~ Avg support ({avg_support:.4f}) is within candidate range "
                f"({min_candidate:.4f} - {max_candidate:.4f})."
            )
        else:
            summary = (
                f"✗ Avg support ({avg_support:.4f}) < all candidates range "
                f"({min_candidate:.4f} - {max_candidate:.4f})."
            )

        return {
            "question_type": question_type,
            "policy": {
                "weights": weights,
                "roi_regions": roi_regions,
            },
            "support_frames": final_support,
            "candidate_frames": final_candidates,
            "comparison": {
                "total_support_frames": len(final_support),
                "total_candidate_frames": len(final_candidates),
                "avg_support_score": float(avg_support),
                "avg_candidate_score": float(avg_candidate),
                "max_candidate_score": float(max_candidate),
                "min_candidate_score": float(min_candidate),
                "summary": summary,
            },
        }
