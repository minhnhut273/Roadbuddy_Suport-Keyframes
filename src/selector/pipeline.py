# from __future__ import annotations

# from pathlib import Path
# import yaml
# import cv2

# from src.video.sampler import sample_video_uniform
# from src.perception.blur import (
#     laplacian_variance,
#     canny_edge_density,
#     brightness_score,
#     novelty_score,
#     roi_feature_max,
# )
# from src.selector.scorer import (
#     FrameScoreComponents,
#     ScoredFrame,
#     minmax_normalize,
#     compute_center_bias,
# )
# from src.selector.peak_picker import temporal_nms
# # from src.selector.policy_rerank import rerank_frames_by_policy

# # Tùy chọn: chỉ dùng khi bật rerank
# try:
#     from src.perception.detector import TrafficDetector
# except Exception:
#     TrafficDetector = None

# try:
#     from src.perception.ocr_reader import OCRReader
# except Exception:
#     OCRReader = None


# class KeyframeSelectorPipeline:
#     def __init__(self, config_path: str | Path):
#         with open(config_path, "r", encoding="utf-8") as f:
#             self.cfg = yaml.safe_load(f)

#         self.use_rerank = bool(self.cfg.get("use_rerank", False))
#         self.detector = None
#         self.ocr = None

#         if self.use_rerank:
#             detector_model = self.cfg.get("detector_model")

#             if detector_model and TrafficDetector is not None:
#                 try:
#                     self.detector = TrafficDetector(detector_model)
#                 except Exception as e:
#                     print(f"[WARN] Cannot load detector model: {detector_model}")
#                     print(f"[WARN] Fallback to OCR-only / selector-only. Error: {e}")
#                     self.detector = None

#             if OCRReader is not None:
#                 try:
#                     self.ocr = OCRReader(
#                         langs=["vi", "en"],
#                         gpu=bool(self.cfg.get("ocr_gpu", False)),
#                     )
#                 except Exception as e:
#                     print(f"[WARN] Cannot initialize OCR. Error: {e}")
#                     self.ocr = None

#     def _trim_frames(self, frames: list, trim_ratio: float) -> list:
#         if not frames:
#             return frames

#         n_total = len(frames)
#         if n_total < 10:
#             return frames

#         start_idx = int(n_total * trim_ratio)
#         end_idx = int(n_total * (1.0 - trim_ratio))
#         trimmed = frames[start_idx:end_idx]
#         return trimmed if trimmed else frames

#     def _get_policy_config(self, question_type: str | None) -> dict:
#         if not self.cfg.get("type_aware", False):
#             return {
#                 "top_k": self.cfg.get("top_k", 4),
#                 "temporal_nms_gap_sec": self.cfg.get("temporal_nms_gap_sec", 0.5),
#                 "refine_window_sec": self.cfg.get("refine_window_sec", 1.0),
#                 "fps_refine": self.cfg.get("fps_refine", 6.0),
#                 "roi_regions": self.cfg.get("roi_regions", []),
#                 "weights": self.cfg.get("weights", {}),
#             }

#         type_to_policy = self.cfg.get("type_to_policy", {})
#         policy_configs = self.cfg.get("policy_configs", {})

#         policy_name = type_to_policy.get(question_type)
#         if policy_name and policy_name in policy_configs:
#             policy = policy_configs[policy_name]
#             return {
#                 "top_k": policy.get("top_k", self.cfg.get("top_k", 4)),
#                 "temporal_nms_gap_sec": policy.get(
#                     "temporal_nms_gap_sec",
#                     self.cfg.get("temporal_nms_gap_sec", 0.5),
#                 ),
#                 "refine_window_sec": policy.get(
#                     "refine_window_sec",
#                     self.cfg.get("refine_window_sec", 1.0),
#                 ),
#                 "fps_refine": policy.get(
#                     "fps_refine",
#                     self.cfg.get("fps_refine", 6.0),
#                 ),
#                 "roi_regions": policy.get("roi_regions", self.cfg.get("roi_regions", [])),
#                 "weights": policy.get("weights", self.cfg.get("weights", {})),
#             }

#         return {
#             "top_k": self.cfg.get("top_k", 4),
#             "temporal_nms_gap_sec": self.cfg.get("temporal_nms_gap_sec", 0.5),
#             "refine_window_sec": self.cfg.get("refine_window_sec", 1.0),
#             "fps_refine": self.cfg.get("fps_refine", 6.0),
#             "roi_regions": self.cfg.get("roi_regions", []),
#             "weights": self.cfg.get("weights", {}),
#         }

#     def _sample_video_window(
#         self,
#         video_path: str,
#         start_sec: float,
#         end_sec: float,
#         fps_sample: float,
#         max_frames: int | None = None,
#     ):
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise RuntimeError(f"Cannot open video: {video_path}")

#         native_fps = cap.get(cv2.CAP_PROP_FPS)
#         if native_fps <= 0:
#             native_fps = 25.0

#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         duration_sec = total_frames / native_fps if total_frames > 0 else 0.0

#         start_sec = max(0.0, start_sec)
#         end_sec = min(duration_sec, end_sec)

#         if end_sec <= start_sec:
#             cap.release()
#             return []

#         start_frame = max(int(start_sec * native_fps), 0)
#         end_frame = min(
#             int(end_sec * native_fps),
#             total_frames - 1 if total_frames > 0 else int(end_sec * native_fps),
#         )

#         frame_step = max(int(round(native_fps / fps_sample)), 1)

#         sampled = []
#         cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#         frame_idx = start_frame

#         while frame_idx <= end_frame:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             if (frame_idx - start_frame) % frame_step == 0:
#                 time_sec = frame_idx / native_fps
#                 sampled.append(
#                     type(
#                         "SampledFrameLike",
#                         (),
#                         {
#                             "frame_idx": frame_idx,
#                             "time_sec": time_sec,
#                             "image": frame,
#                         },
#                     )
#                 )
#                 if max_frames is not None and len(sampled) >= max_frames:
#                     break

#             frame_idx += 1

#         cap.release()
#         return sampled

#     def _deduplicate_by_time(
#         self,
#         scored_frames: list[ScoredFrame],
#         tol: float = 1e-6,
#     ) -> list[ScoredFrame]:
#         if not scored_frames:
#             return []

#         scored_frames = sorted(scored_frames, key=lambda x: (x.time_sec, -x.score))
#         deduped = [scored_frames[0]]

#         for fr in scored_frames[1:]:
#             if abs(fr.time_sec - deduped[-1].time_sec) > tol:
#                 deduped.append(fr)
#             elif fr.score > deduped[-1].score:
#                 deduped[-1] = fr

#         return deduped

#     def _score_sampled_frames(
#         self,
#         frames: list,
#         question_type: str | None = None,
#     ) -> list[ScoredFrame]:
#         if not frames:
#             return []

#         policy_cfg = self._get_policy_config(question_type)
#         weights = dict(policy_cfg.get("weights", {}))
#         rois = policy_cfg.get("roi_regions", [])

#         raw_sharpness = []
#         raw_edges = []
#         raw_brightness = []
#         raw_novelty = []

#         raw_roi_sharpness = []
#         raw_roi_edges = []
#         raw_roi_brightness = []

#         prev_image = None
#         for fr in frames:
#             img = fr.image

#             raw_sharpness.append(laplacian_variance(img))
#             raw_edges.append(canny_edge_density(img))
#             raw_brightness.append(brightness_score(img))
#             raw_novelty.append(novelty_score(img, prev_image))

#             raw_roi_sharpness.append(roi_feature_max(img, rois, laplacian_variance))
#             raw_roi_edges.append(roi_feature_max(img, rois, canny_edge_density))
#             raw_roi_brightness.append(roi_feature_max(img, rois, brightness_score))

#             prev_image = img

#         norm_sharpness = minmax_normalize(raw_sharpness)
#         norm_edges = minmax_normalize(raw_edges)
#         norm_brightness = minmax_normalize(raw_brightness)
#         norm_novelty = minmax_normalize(raw_novelty)

#         norm_roi_sharpness = minmax_normalize(raw_roi_sharpness)
#         norm_roi_edges = minmax_normalize(raw_roi_edges)
#         norm_roi_brightness = minmax_normalize(raw_roi_brightness)

#         n = len(frames)
#         scored_frames: list[ScoredFrame] = []

#         for i, fr in enumerate(frames):
#             c_bias = compute_center_bias(i, n)

#             comps = FrameScoreComponents(
#                 sharpness=norm_sharpness[i],
#                 edge_density=norm_edges[i],
#                 brightness=norm_brightness[i],
#                 novelty=norm_novelty[i],
#                 center_bias=c_bias,
#             )

#             score = (
#                 weights.get("sharpness", 0.0) * norm_sharpness[i]
#                 + weights.get("edge_density", 0.0) * norm_edges[i]
#                 + weights.get("brightness", 0.0) * norm_brightness[i]
#                 + weights.get("novelty", 0.0) * norm_novelty[i]
#                 + weights.get("center_bias", 0.0) * c_bias
#                 + weights.get("roi_sharpness", 0.0) * norm_roi_sharpness[i]
#                 + weights.get("roi_edge_density", 0.0) * norm_roi_edges[i]
#                 + weights.get("roi_brightness", 0.0) * norm_roi_brightness[i]
#             )

#             scored_frames.append(
#                 ScoredFrame(
#                     frame_idx=fr.frame_idx,
#                     time_sec=fr.time_sec,
#                     image=fr.image,
#                     components=comps,
#                     score=float(score),
#                 )
#             )

#         return scored_frames

#     def _rerank_candidates(self, selected_frames: list[dict]) -> list[dict]:
#         if not self.use_rerank:
#             return selected_frames

#         rerank_top_m = int(self.cfg.get("rerank_top_m", len(selected_frames)))
#         rerank_top_m = min(rerank_top_m, len(selected_frames))

#         weights = self.cfg.get("rerank_weights", {})
#         subset = selected_frames[:rerank_top_m]
#         reranked = []

#         for fr in subset:
#             image = fr["image"]

#             det_conf = 0.0
#             det_area = 0.0
#             ocr_conf = 0.0
#             ocr_text_len = 0.0

#             crops = []

#             if self.detector is not None:
#                 boxes = self.detector.detect(image)
#                 if boxes:
#                     det_conf = max(b.conf for b in boxes)
#                     det_area = max(b.area_ratio for b in boxes)
#                     best_box = max(boxes, key=lambda b: (b.conf, b.area_ratio))
#                     crops.append(self.detector.crop_with_pad(image, best_box))

#             if self.ocr is not None:
#                 ocr_img = crops[0] if crops else image
#                 ocr_info = self.ocr.read_text(ocr_img)
#                 ocr_conf = float(ocr_info["mean_conf"])
#                 ocr_text_len = min(float(ocr_info["text_len"]) / 30.0, 1.0)

#             rerank_score = (
#                 weights.get("base_selector_score", 0.40) * fr["score"]
#                 + weights.get("det_conf", 0.25) * det_conf
#                 + weights.get("det_area", 0.15) * min(det_area / 0.05, 1.0)
#                 + weights.get("ocr_conf", 0.10) * ocr_conf
#                 + weights.get("ocr_text_len", 0.10) * ocr_text_len
#             )

#             fr_copy = dict(fr)
#             fr_copy["rerank"] = {
#                 "det_conf": det_conf,
#                 "det_area": det_area,
#                 "ocr_conf": ocr_conf,
#                 "ocr_text_len": ocr_text_len,
#                 "rerank_score": rerank_score,
#             }
#             fr_copy["score"] = rerank_score
#             reranked.append(fr_copy)

#         if rerank_top_m < len(selected_frames):
#             reranked.extend(selected_frames[rerank_top_m:])

#         reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)
#         return reranked

#     def run(self, video_path: str, question_type: str | None = None) -> dict:
#         policy_cfg = self._get_policy_config(question_type)

#         fps_sample = float(self.cfg.get("fps_sample", 3.0))
#         fps_coarse = float(self.cfg.get("fps_coarse", fps_sample))
#         fps_refine = float(policy_cfg.get("fps_refine", self.cfg.get("fps_refine", 6.0)))

#         max_frames = int(self.cfg.get("max_frames_per_video", 96))
#         top_k = int(policy_cfg.get("top_k", self.cfg.get("top_k", 4)))
#         min_gap_sec = float(
#             policy_cfg.get(
#                 "temporal_nms_gap_sec",
#                 self.cfg.get("temporal_nms_gap_sec", 0.5),
#             )
#         )
#         coarse_top_m = int(self.cfg.get("coarse_top_m", 2))
#         refine_window_sec = float(
#             policy_cfg.get(
#                 "refine_window_sec",
#                 self.cfg.get("refine_window_sec", 1.0),
#             )
#         )
#         trim_ratio = float(policy_cfg.get("trim_ratio", self.cfg.get("trim_ratio", 0.10)))

#         coarse_frames = sample_video_uniform(
#             video_path=video_path,
#             fps_sample=fps_coarse,
#             max_frames=max_frames,
#         )

#         if not coarse_frames:
#             return {
#                 "video_path": video_path,
#                 "sampled_frames": 0,
#                 "selected_frames": [],
#                 "all_frames": [],
#                 "coarse_peaks": [],
#             }

#         coarse_frames = self._trim_frames(coarse_frames, trim_ratio)
#         coarse_scored = self._score_sampled_frames(
#             coarse_frames,
#             question_type=question_type,
#         )

#         coarse_times = [x.time_sec for x in coarse_scored]
#         coarse_scores = [x.score for x in coarse_scored]

#         coarse_keep_idx = temporal_nms(
#             times_sec=coarse_times,
#             scores=coarse_scores,
#             top_k=coarse_top_m,
#             min_gap_sec=max(min_gap_sec, refine_window_sec * 0.75),
#         )
#         coarse_selected = [coarse_scored[i] for i in coarse_keep_idx]

#         fine_candidates: list[ScoredFrame] = []

#         for peak in coarse_selected:
#             win_start = peak.time_sec - refine_window_sec
#             win_end = peak.time_sec + refine_window_sec

#             local_frames = self._sample_video_window(
#                 video_path=video_path,
#                 start_sec=win_start,
#                 end_sec=win_end,
#                 fps_sample=fps_refine,
#                 max_frames=max_frames,
#             )
#             if not local_frames:
#                 continue

#             local_scored = self._score_sampled_frames(
#                 local_frames,
#                 question_type=question_type,
#             )
#             fine_candidates.extend(local_scored)

#         if not fine_candidates:
#             fine_candidates = coarse_scored[:]

#         merged_candidates = coarse_scored + fine_candidates
#         merged_candidates = self._deduplicate_by_time(merged_candidates)

#         merged_times = [x.time_sec for x in merged_candidates]
#         merged_scores = [x.score for x in merged_candidates]

#         final_keep_idx = temporal_nms(
#             times_sec=merged_times,
#             scores=merged_scores,
#             top_k=top_k,
#             min_gap_sec=min_gap_sec,
#         )
#         final_selected = [merged_candidates[i] for i in final_keep_idx]

#         final_selected_dicts = [
#             {
#                 "frame_idx": x.frame_idx,
#                 "time_sec": x.time_sec,
#                 "score": x.score,
#                 "components": {
#                     "sharpness": x.components.sharpness,
#                     "edge_density": x.components.edge_density,
#                     "brightness": x.components.brightness,
#                     "novelty": x.components.novelty,
#                     "center_bias": x.components.center_bias,
#                 },
#                 "image": x.image,
#             }
#             for x in final_selected
#         ]

#         final_selected_dicts = self._rerank_candidates(final_selected_dicts)


#         rerank_cfg = self.cfg.get("rerank", {})

#         all_frames_dicts = [
#             {
#                 "frame_idx": x.frame_idx,
#                 "time_sec": x.time_sec,
#                 "score": x.score,
#                 "components": {
#                     "sharpness": x.components.sharpness,
#                     "edge_density": x.components.edge_density,
#                     "brightness": x.components.brightness,
#                     "novelty": x.components.novelty,
#                     "center_bias": x.components.center_bias,
#                 },
#                 "image": x.image,
#             }
#             for x in merged_candidates
#         ]

#         # final_selected_dicts = rerank_frames_by_policy(
#         #     question_type=question_type,
#         #     selected_frames=final_selected_dicts,
#         #     all_frames=all_frames_dicts,
#         #     roi_regions=policy_cfg.get("roi_regions", []),
#         #     top_k=top_k,
#         #     rerank_cfg=rerank_cfg,
#         # )


#         # return {
#         #     "video_path": video_path,
#         #     "sampled_frames": len(coarse_frames),
#         #     "selected_frames": final_selected_dicts,
#         #     "all_frames": [
#         #         {
#         #             "frame_idx": x.frame_idx,
#         #             "time_sec": x.time_sec,
#         #             "score": x.score,
#         #             "components": {
#         #                 "sharpness": x.components.sharpness,
#         #                 "edge_density": x.components.edge_density,
#         #                 "brightness": x.components.brightness,
#         #                 "novelty": x.components.novelty,
#         #                 "center_bias": x.components.center_bias,
#         #             },
#         #             "image": x.image,
#         #         }
#         #         for x in merged_candidates
#         #     ],
#         #     "coarse_peaks": [
#         #         {
#         #             "frame_idx": x.frame_idx,
#         #             "time_sec": x.time_sec,
#         #             "score": x.score,
#         #         }
#         #         for x in coarse_selected
#         #     ],
#         # }





#         final_selected_dicts = [
#             {
#                 "frame_idx": x.frame_idx,
#                 "time_sec": x.time_sec,
#                 "score": x.score,
#                 "components": {
#                     "sharpness": x.components.sharpness,
#                     "edge_density": x.components.edge_density,
#                     "brightness": x.components.brightness,
#                     "novelty": x.components.novelty,
#                     "center_bias": x.components.center_bias,
#                 },
#                 "image": x.image,
#             }
#             for x in final_selected
#         ]

#         all_frames_dicts = [
#             {
#                 "frame_idx": x.frame_idx,
#                 "time_sec": x.time_sec,
#                 "score": x.score,
#                 "components": {
#                     "sharpness": x.components.sharpness,
#                     "edge_density": x.components.edge_density,
#                     "brightness": x.components.brightness,
#                     "novelty": x.components.novelty,
#                     "center_bias": x.components.center_bias,
#                 },
#                 "image": x.image,
#             }
#             for x in merged_candidates
#         ]

#         rerank_cfg = self.cfg.get("rerank", {})

#         final_selected_dicts = rerank_frames_by_policy(
#             question_type=question_type,
#             selected_frames=final_selected_dicts,
#             all_frames=all_frames_dicts,
#             roi_regions=policy_cfg.get("roi_regions", []),
#             top_k=top_k,
#             rerank_cfg=rerank_cfg,
#         )

#         return {
#             "video_path": video_path,
#             "sampled_frames": len(coarse_frames),
#             "selected_frames": final_selected_dicts,
#             "all_frames": all_frames_dicts,
#             "coarse_peaks": [
#                 {
#                     "frame_idx": x.frame_idx,
#                     "time_sec": x.time_sec,
#                     "score": x.score,
#                 }
#                 for x in coarse_selected
#             ],
#         }






from __future__ import annotations

from pathlib import Path
import yaml
import cv2

from src.video.sampler import sample_video_uniform
from src.perception.blur import (
    laplacian_variance,
    canny_edge_density,
    brightness_score,
    novelty_score,
    roi_feature_max,
)
from src.selector.scorer import (
    FrameScoreComponents,
    ScoredFrame,
    minmax_normalize,
    compute_center_bias,
)
from src.selector.peak_picker import temporal_nms

# Tùy chọn: chỉ dùng khi bật rerank cũ
try:
    from src.perception.detector import TrafficDetector
except Exception:
    TrafficDetector = None

try:
    from src.perception.ocr_reader import OCRReader
except Exception:
    OCRReader = None


class KeyframeSelectorPipeline:
    def __init__(self, config_path: str | Path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.use_rerank = bool(self.cfg.get("use_rerank", False))
        self.detector = None
        self.ocr = None

        if self.use_rerank:
            detector_model = self.cfg.get("detector_model")

            if detector_model and TrafficDetector is not None:
                try:
                    self.detector = TrafficDetector(detector_model)
                except Exception as e:
                    print(f"[WARN] Cannot load detector model: {detector_model}")
                    print(f"[WARN] Fallback to OCR-only / selector-only. Error: {e}")
                    self.detector = None

            if OCRReader is not None:
                try:
                    self.ocr = OCRReader(
                        langs=["vi", "en"],
                        gpu=bool(self.cfg.get("ocr_gpu", False)),
                    )
                except Exception as e:
                    print(f"[WARN] Cannot initialize OCR. Error: {e}")
                    self.ocr = None

    def _trim_frames(self, frames: list, trim_ratio: float) -> list:
        if not frames:
            return frames

        n_total = len(frames)
        if n_total < 10:
            return frames

        start_idx = int(n_total * trim_ratio)
        end_idx = int(n_total * (1.0 - trim_ratio))
        trimmed = frames[start_idx:end_idx]
        return trimmed if trimmed else frames

    def _get_policy_config(self, question_type: str | None) -> dict:
        if not self.cfg.get("type_aware", False):
            return {
                "top_k": self.cfg.get("top_k", 4),
                "temporal_nms_gap_sec": self.cfg.get("temporal_nms_gap_sec", 0.5),
                "refine_window_sec": self.cfg.get("refine_window_sec", 1.0),
                "fps_refine": self.cfg.get("fps_refine", 6.0),
                "roi_regions": self.cfg.get("roi_regions", []),
                "weights": self.cfg.get("weights", {}),
                "trim_ratio": self.cfg.get("trim_ratio", 0.10),
            }

        type_to_policy = self.cfg.get("type_to_policy", {})
        policy_configs = self.cfg.get("policy_configs", {})

        policy_name = type_to_policy.get(question_type)
        if policy_name and policy_name in policy_configs:
            policy = policy_configs[policy_name]
            return {
                "top_k": policy.get("top_k", self.cfg.get("top_k", 4)),
                "temporal_nms_gap_sec": policy.get(
                    "temporal_nms_gap_sec",
                    self.cfg.get("temporal_nms_gap_sec", 0.5),
                ),
                "refine_window_sec": policy.get(
                    "refine_window_sec",
                    self.cfg.get("refine_window_sec", 1.0),
                ),
                "fps_refine": policy.get(
                    "fps_refine",
                    self.cfg.get("fps_refine", 6.0),
                ),
                "roi_regions": policy.get("roi_regions", self.cfg.get("roi_regions", [])),
                "weights": policy.get("weights", self.cfg.get("weights", {})),
                "trim_ratio": policy.get("trim_ratio", self.cfg.get("trim_ratio", 0.10)),
            }

        return {
            "top_k": self.cfg.get("top_k", 4),
            "temporal_nms_gap_sec": self.cfg.get("temporal_nms_gap_sec", 0.5),
            "refine_window_sec": self.cfg.get("refine_window_sec", 1.0),
            "fps_refine": self.cfg.get("fps_refine", 6.0),
            "roi_regions": self.cfg.get("roi_regions", []),
            "weights": self.cfg.get("weights", {}),
            "trim_ratio": self.cfg.get("trim_ratio", 0.10),
        }

    def _sample_video_window(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        fps_sample: float,
        max_frames: int | None = None,
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0:
            native_fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / native_fps if total_frames > 0 else 0.0

        start_sec = max(0.0, start_sec)
        end_sec = min(duration_sec, end_sec)

        if end_sec <= start_sec:
            cap.release()
            return []

        start_frame = max(int(start_sec * native_fps), 0)
        end_frame = min(
            int(end_sec * native_fps),
            total_frames - 1 if total_frames > 0 else int(end_sec * native_fps),
        )

        frame_step = max(int(round(native_fps / fps_sample)), 1)

        sampled = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % frame_step == 0:
                time_sec = frame_idx / native_fps
                sampled.append(
                    type(
                        "SampledFrameLike",
                        (),
                        {
                            "frame_idx": frame_idx,
                            "time_sec": time_sec,
                            "image": frame,
                        },
                    )
                )
                if max_frames is not None and len(sampled) >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        return sampled

    def _deduplicate_by_time(
        self,
        scored_frames: list[ScoredFrame],
        tol: float = 1e-6,
    ) -> list[ScoredFrame]:
        if not scored_frames:
            return []

        scored_frames = sorted(scored_frames, key=lambda x: (x.time_sec, -x.score))
        deduped = [scored_frames[0]]

        for fr in scored_frames[1:]:
            if abs(fr.time_sec - deduped[-1].time_sec) > tol:
                deduped.append(fr)
            elif fr.score > deduped[-1].score:
                deduped[-1] = fr

        return deduped

    def _score_sampled_frames(
        self,
        frames: list,
        question_type: str | None = None,
    ) -> list[ScoredFrame]:
        if not frames:
            return []

        policy_cfg = self._get_policy_config(question_type)
        weights = dict(policy_cfg.get("weights", {}))
        rois = policy_cfg.get("roi_regions", [])

        raw_sharpness = []
        raw_edges = []
        raw_brightness = []
        raw_novelty = []

        raw_roi_sharpness = []
        raw_roi_edges = []
        raw_roi_brightness = []

        prev_image = None
        for fr in frames:
            img = fr.image

            raw_sharpness.append(laplacian_variance(img))
            raw_edges.append(canny_edge_density(img))
            raw_brightness.append(brightness_score(img))
            raw_novelty.append(novelty_score(img, prev_image))

            raw_roi_sharpness.append(roi_feature_max(img, rois, laplacian_variance))
            raw_roi_edges.append(roi_feature_max(img, rois, canny_edge_density))
            raw_roi_brightness.append(roi_feature_max(img, rois, brightness_score))

            prev_image = img

        norm_sharpness = minmax_normalize(raw_sharpness)
        norm_edges = minmax_normalize(raw_edges)
        norm_brightness = minmax_normalize(raw_brightness)
        norm_novelty = minmax_normalize(raw_novelty)

        norm_roi_sharpness = minmax_normalize(raw_roi_sharpness)
        norm_roi_edges = minmax_normalize(raw_roi_edges)
        norm_roi_brightness = minmax_normalize(raw_roi_brightness)

        n = len(frames)
        scored_frames: list[ScoredFrame] = []

        for i, fr in enumerate(frames):
            c_bias = compute_center_bias(i, n)

            comps = FrameScoreComponents(
                sharpness=norm_sharpness[i],
                edge_density=norm_edges[i],
                brightness=norm_brightness[i],
                novelty=norm_novelty[i],
                center_bias=c_bias,
            )

            score = (
                weights.get("sharpness", 0.0) * norm_sharpness[i]
                + weights.get("edge_density", 0.0) * norm_edges[i]
                + weights.get("brightness", 0.0) * norm_brightness[i]
                + weights.get("novelty", 0.0) * norm_novelty[i]
                + weights.get("center_bias", 0.0) * c_bias
                + weights.get("roi_sharpness", 0.0) * norm_roi_sharpness[i]
                + weights.get("roi_edge_density", 0.0) * norm_roi_edges[i]
                + weights.get("roi_brightness", 0.0) * norm_roi_brightness[i]
            )

            scored_frames.append(
                ScoredFrame(
                    frame_idx=fr.frame_idx,
                    time_sec=fr.time_sec,
                    image=fr.image,
                    components=comps,
                    score=float(score),
                )
            )

        return scored_frames

    def _rerank_candidates(self, selected_frames: list[dict]) -> list[dict]:
        if not self.use_rerank:
            return selected_frames

        rerank_top_m = int(self.cfg.get("rerank_top_m", len(selected_frames)))
        rerank_top_m = min(rerank_top_m, len(selected_frames))

        weights = self.cfg.get("rerank_weights", {})
        subset = selected_frames[:rerank_top_m]
        reranked = []

        for fr in subset:
            image = fr["image"]

            det_conf = 0.0
            det_area = 0.0
            ocr_conf = 0.0
            ocr_text_len = 0.0

            crops = []

            if self.detector is not None:
                boxes = self.detector.detect(image)
                if boxes:
                    det_conf = max(b.conf for b in boxes)
                    det_area = max(b.area_ratio for b in boxes)
                    best_box = max(boxes, key=lambda b: (b.conf, b.area_ratio))
                    crops.append(self.detector.crop_with_pad(image, best_box))

            if self.ocr is not None:
                ocr_img = crops[0] if crops else image
                ocr_info = self.ocr.read_text(ocr_img)
                ocr_conf = float(ocr_info["mean_conf"])
                ocr_text_len = min(float(ocr_info["text_len"]) / 30.0, 1.0)

            rerank_score = (
                weights.get("base_selector_score", 0.40) * fr["score"]
                + weights.get("det_conf", 0.25) * det_conf
                + weights.get("det_area", 0.15) * min(det_area / 0.05, 1.0)
                + weights.get("ocr_conf", 0.10) * ocr_conf
                + weights.get("ocr_text_len", 0.10) * ocr_text_len
            )

            fr_copy = dict(fr)
            fr_copy["rerank"] = {
                "det_conf": det_conf,
                "det_area": det_area,
                "ocr_conf": ocr_conf,
                "ocr_text_len": ocr_text_len,
                "rerank_score": rerank_score,
            }
            fr_copy["score"] = rerank_score
            reranked.append(fr_copy)

        if rerank_top_m < len(selected_frames):
            reranked.extend(selected_frames[rerank_top_m:])

        reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)
        return reranked

    def run(self, video_path: str, question_type: str | None = None) -> dict:
        policy_cfg = self._get_policy_config(question_type)

        fps_sample = float(self.cfg.get("fps_sample", 3.0))
        fps_coarse = float(self.cfg.get("fps_coarse", fps_sample))
        fps_refine = float(policy_cfg.get("fps_refine", self.cfg.get("fps_refine", 6.0)))

        max_frames = int(self.cfg.get("max_frames_per_video", 96))
        top_k = int(policy_cfg.get("top_k", self.cfg.get("top_k", 4)))
        min_gap_sec = float(
            policy_cfg.get(
                "temporal_nms_gap_sec",
                self.cfg.get("temporal_nms_gap_sec", 0.5),
            )
        )
        coarse_top_m = int(self.cfg.get("coarse_top_m", 2))
        refine_window_sec = float(
            policy_cfg.get(
                "refine_window_sec",
                self.cfg.get("refine_window_sec", 1.0),
            )
        )
        trim_ratio = float(policy_cfg.get("trim_ratio", self.cfg.get("trim_ratio", 0.10)))

        coarse_frames = sample_video_uniform(
            video_path=video_path,
            fps_sample=fps_coarse,
            max_frames=max_frames,
        )

        if not coarse_frames:
            return {
                "video_path": video_path,
                "sampled_frames": 0,
                "selected_frames": [],
                "all_frames": [],
                "coarse_peaks": [],
            }

        coarse_frames = self._trim_frames(coarse_frames, trim_ratio)
        coarse_scored = self._score_sampled_frames(
            coarse_frames,
            question_type=question_type,
        )

        coarse_times = [x.time_sec for x in coarse_scored]
        coarse_scores = [x.score for x in coarse_scored]

        coarse_keep_idx = temporal_nms(
            times_sec=coarse_times,
            scores=coarse_scores,
            top_k=coarse_top_m,
            min_gap_sec=max(min_gap_sec, refine_window_sec * 0.75),
        )
        coarse_selected = [coarse_scored[i] for i in coarse_keep_idx]

        fine_candidates: list[ScoredFrame] = []

        for peak in coarse_selected:
            win_start = peak.time_sec - refine_window_sec
            win_end = peak.time_sec + refine_window_sec

            local_frames = self._sample_video_window(
                video_path=video_path,
                start_sec=win_start,
                end_sec=win_end,
                fps_sample=fps_refine,
                max_frames=max_frames,
            )
            if not local_frames:
                continue

            local_scored = self._score_sampled_frames(
                local_frames,
                question_type=question_type,
            )
            fine_candidates.extend(local_scored)

        if not fine_candidates:
            fine_candidates = coarse_scored[:]

        merged_candidates = coarse_scored + fine_candidates
        merged_candidates = self._deduplicate_by_time(merged_candidates)

        merged_times = [x.time_sec for x in merged_candidates]
        merged_scores = [x.score for x in merged_candidates]

        final_keep_idx = temporal_nms(
            times_sec=merged_times,
            scores=merged_scores,
            top_k=top_k,
            min_gap_sec=min_gap_sec,
        )
        final_selected = [merged_candidates[i] for i in final_keep_idx]

        final_selected_dicts = [
            {
                "frame_idx": x.frame_idx,
                "time_sec": x.time_sec,
                "score": x.score,
                "components": {
                    "sharpness": x.components.sharpness,
                    "edge_density": x.components.edge_density,
                    "brightness": x.components.brightness,
                    "novelty": x.components.novelty,
                    "center_bias": x.components.center_bias,
                },
                "image": x.image,
            }
            for x in final_selected
        ]

        final_selected_dicts = self._rerank_candidates(final_selected_dicts)

        return {
            "video_path": video_path,
            "sampled_frames": len(coarse_frames),
            "selected_frames": final_selected_dicts,
            "all_frames": [
                {
                    "frame_idx": x.frame_idx,
                    "time_sec": x.time_sec,
                    "score": x.score,
                    "components": {
                        "sharpness": x.components.sharpness,
                        "edge_density": x.components.edge_density,
                        "brightness": x.components.brightness,
                        "novelty": x.components.novelty,
                        "center_bias": x.components.center_bias,
                    },
                    "image": x.image,
                }
                for x in merged_candidates
            ],
            "coarse_peaks": [
                {
                    "frame_idx": x.frame_idx,
                    "time_sec": x.time_sec,
                    "score": x.score,
                }
                for x in coarse_selected
            ],
        }