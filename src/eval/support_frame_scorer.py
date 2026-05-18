from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from src.perception.blur import (
    laplacian_variance,
    canny_edge_density,
    brightness_score,
    roi_feature_max,
)
from src.selector.scorer import minmax_normalize


class SupportFrameScorer:
    """
    Score support frames using the SAME family of features as the candidate selector,
    while respecting the CURRENT type-aware policy (ROI + weights).

    Key design choices:
    - Recompute RAW features for both support frames and candidate frames.
    - Normalize support + candidate together on the SAME raw scale.
    - Normalize ROI features on their OWN scale (separate from global features).
    - Use policy selected from question_type, matching the pipeline spirit.
    """

    def __init__(self, config: dict):
        self.cfg = config or {}

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------
    def _get_policy_config(self, question_type: str | None) -> dict:
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

    # ------------------------------------------------------------------
    # Frame extraction helpers
    # ------------------------------------------------------------------
    def _extract_frame_at_time(
        self,
        video_path: str | Path,
        time_sec: float,
    ) -> tuple[bool, Any, int, float]:
        """
        Returns:
            (success, frame_bgr, frame_idx, actual_time_sec)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, None, -1, -1.0

        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if native_fps <= 0:
                native_fps = 25.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = int(round(time_sec * native_fps))
            frame_idx = max(0, min(frame_idx, max(total_frames - 1, 0)))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            actual_time = frame_idx / native_fps if native_fps > 0 else time_sec
            return bool(ret), frame, frame_idx, actual_time
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Raw feature extraction
    # ------------------------------------------------------------------
    def _compute_raw_features(self, frame, roi_regions: list[list[float]]) -> dict[str, float]:
        return {
            "sharpness": float(laplacian_variance(frame)),
            "edge_density": float(canny_edge_density(frame)),
            "brightness": float(brightness_score(frame)),
            "roi_sharpness": float(roi_feature_max(frame, roi_regions, laplacian_variance)),
            "roi_edge_density": float(roi_feature_max(frame, roi_regions, canny_edge_density)),
            "roi_brightness": float(roi_feature_max(frame, roi_regions, brightness_score)),
        }

    def _normalize_feature_map(self, values: list[float]) -> list[float]:
        return minmax_normalize(values)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def score_support_frames(
        self,
        video_path: str | Path,
        support_times: list[float],
        question_type: str | None = None,
        all_candidate_frames: list[dict] | None = None,
    ) -> dict:
        """
        Args:
            video_path: absolute/relative path to video
            support_times: support timestamps from train.json
            question_type: type of question for choosing policy
            all_candidate_frames: pipeline candidates; each item may contain:
                - image (preferred, BGR ndarray in-memory)
                - time_sec (fallback, to re-extract from video)
                - frame_idx (optional)
                - score (optional; not trusted for normalization)
        Returns:
            dict with support_frames[] and comparison{}
        """
        policy_cfg = self._get_policy_config(question_type)
        weights = policy_cfg.get("weights", {})
        roi_regions = policy_cfg.get("roi_regions", [])

        if not support_times:
            return {
                "support_frames": [],
                "comparison": {
                    "total_support_frames": 0,
                    "total_candidate_frames": 0,
                    "avg_support_score": 0.0,
                    "max_candidate_score": 0.0,
                    "min_candidate_score": 0.0,
                    "all_support_better_than_candidates": False,
                    "summary": "No support frames provided",
                },
            }

        # --------------------------------------------------------------
        # 1) Extract & compute RAW support features
        # --------------------------------------------------------------
        support_items: list[dict] = []
        for t in support_times:
            ok, frame, frame_idx, actual_time = self._extract_frame_at_time(video_path, float(t))
            if not ok or frame is None:
                print(f"[WARN] Cannot extract support frame at {t:.6f}s")
                continue

            raw = self._compute_raw_features(frame, roi_regions)
            support_items.append(
                {
                    "kind": "support",
                    "time_sec": float(t),
                    "actual_time_sec": float(actual_time),
                    "frame_idx": int(frame_idx),
                    "raw": raw,
                }
            )

        if not support_items:
            return {
                "support_frames": [],
                "comparison": {
                    "total_support_frames": len(support_times),
                    "total_candidate_frames": 0,
                    "avg_support_score": 0.0,
                    "max_candidate_score": 0.0,
                    "min_candidate_score": 0.0,
                    "all_support_better_than_candidates": False,
                    "summary": "Failed to extract any support frames",
                },
            }

        # --------------------------------------------------------------
        # 2) Recompute RAW candidate features
        # --------------------------------------------------------------
        candidate_items: list[dict] = []
        if all_candidate_frames:
            for cand in all_candidate_frames:
                frame = cand.get("image", None)

                if frame is None:
                    # fallback: re-extract from video using time_sec
                    time_sec = cand.get("time_sec", None)
                    if time_sec is None:
                        continue
                    ok, frame, frame_idx, actual_time = self._extract_frame_at_time(video_path, float(time_sec))
                    if not ok or frame is None:
                        continue
                else:
                    frame_idx = int(cand.get("frame_idx", -1))
                    actual_time = float(cand.get("time_sec", -1.0))

                raw = self._compute_raw_features(frame, roi_regions)
                candidate_items.append(
                    {
                        "kind": "candidate",
                        "time_sec": float(cand.get("time_sec", actual_time if actual_time >= 0 else 0.0)),
                        "actual_time_sec": float(actual_time if actual_time >= 0 else cand.get("time_sec", 0.0)),
                        "frame_idx": int(frame_idx),
                        "raw": raw,
                    }
                )

        # --------------------------------------------------------------
        # 3) Joint normalization on SAME raw scale
        # --------------------------------------------------------------
        all_items = candidate_items + support_items

        feat_names = [
            "sharpness",
            "edge_density",
            "brightness",
            "roi_sharpness",
            "roi_edge_density",
            "roi_brightness",
        ]

        norm_tables: dict[str, list[float]] = {}
        for feat in feat_names:
            vals = [it["raw"][feat] for it in all_items]
            norm_tables[feat] = self._normalize_feature_map(vals)

        for idx, it in enumerate(all_items):
            it["norm"] = {feat: norm_tables[feat][idx] for feat in feat_names}

            # novelty / center_bias for isolated support scoring:
            # keep neutral values because we do not reconstruct sequence-local context here
            novelty_val = 0.5
            center_bias_val = 0.5

            score = (
                weights.get("sharpness", 0.0) * it["norm"]["sharpness"]
                + weights.get("edge_density", 0.0) * it["norm"]["edge_density"]
                + weights.get("brightness", 0.0) * it["norm"]["brightness"]
                + weights.get("novelty", 0.0) * novelty_val
                + weights.get("center_bias", 0.0) * center_bias_val
                + weights.get("roi_sharpness", 0.0) * it["norm"]["roi_sharpness"]
                + weights.get("roi_edge_density", 0.0) * it["norm"]["roi_edge_density"]
                + weights.get("roi_brightness", 0.0) * it["norm"]["roi_brightness"]
            )
            it["score"] = float(score)

        # --------------------------------------------------------------
        # 4) Build support output + rank against candidates
        # --------------------------------------------------------------
        candidate_scores = [it["score"] for it in candidate_items]
        support_frames_out: list[dict] = []

        for sf in support_items:
            score = sf["score"]
            rank = None
            status = None
            if candidate_scores:
                rank = sum(1 for cs in candidate_scores if cs >= score) + 1
                status = "excellent" if rank <= 2 else "good" if rank <= 4 else "weak"

            support_frames_out.append(
                {
                    "time_sec": sf["time_sec"],
                    "actual_time_sec": sf["actual_time_sec"],
                    "frame_idx": sf["frame_idx"],
                    "score": float(score),
                    "components": {
                        "sharpness": float(sf["norm"]["sharpness"]),
                        "edge_density": float(sf["norm"]["edge_density"]),
                        "brightness": float(sf["norm"]["brightness"]),
                        "roi_sharpness": float(sf["norm"]["roi_sharpness"]),
                        "roi_edge_density": float(sf["norm"]["roi_edge_density"]),
                        "roi_brightness": float(sf["norm"]["roi_brightness"]),
                    },
                    "rank_in_candidates": rank,
                    "status": status,
                }
            )

        # --------------------------------------------------------------
        # 5) Comparison summary
        # --------------------------------------------------------------
        support_scores = [x["score"] for x in support_frames_out]
        avg_support = sum(support_scores) / len(support_scores) if support_scores else 0.0
        max_candidate = max(candidate_scores) if candidate_scores else 0.0
        min_candidate = min(candidate_scores) if candidate_scores else 0.0

        all_better = (
            all(s >= max_candidate for s in support_scores)
            if candidate_scores and support_scores
            else False
        )

        if not candidate_scores:
            summary = (
                f"Support frames scored successfully. Avg support score: {avg_support:.4f}"
            )
        elif all_better:
            summary = (
                f"✓ All support frames ({len(support_scores)}) are BETTER than "
                f"top candidate ({max_candidate:.4f}). Avg support: {avg_support:.4f}"
            )
        elif avg_support > max_candidate:
            summary = (
                f"✓ Average support score ({avg_support:.4f}) is BETTER than "
                f"top candidate ({max_candidate:.4f})"
            )
        elif avg_support > min_candidate:
            summary = (
                f"~ Support frames (avg {avg_support:.4f}) are within candidate range "
                f"({min_candidate:.4f} - {max_candidate:.4f})"
            )
        else:
            summary = (
                f"✗ Support frames (avg {avg_support:.4f}) are WEAKER than "
                f"all candidates ({min_candidate:.4f} - {max_candidate:.4f})"
            )

        return {
            "support_frames": support_frames_out,
            "comparison": {
                "total_support_frames": len(support_times),
                "total_candidate_frames": len(candidate_scores),
                "avg_support_score": float(avg_support),
                "max_candidate_score": float(max_candidate),
                "min_candidate_score": float(min_candidate),
                "all_support_better_than_candidates": bool(all_better),
                "summary": summary,
            },
        }
