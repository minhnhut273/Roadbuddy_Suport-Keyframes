from __future__ import annotations
import cv2
import numpy as np


def laplacian_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def canny_edge_density(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float((edges > 0).mean())


def brightness_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean() / 255.0)


def novelty_score(image: np.ndarray, prev_image: np.ndarray | None) -> float:
    if prev_image is None:
        return 1.0
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    diff = cv2.absdiff(gray1, gray2)
    return float(diff.mean() / 255.0)


def crop_roi(image: np.ndarray, roi: list[float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(roi[0] * w)))
    y1 = max(0, min(h - 1, int(roi[1] * h)))
    x2 = max(x1 + 1, min(w, int(roi[2] * w)))
    y2 = max(y1 + 1, min(h, int(roi[3] * h)))
    return image[y1:y2, x1:x2]


def roi_feature_max(image: np.ndarray, rois: list[list[float]], feature_fn) -> float:
    if not rois:
        return float(feature_fn(image))

    vals = []
    for roi in rois:
        patch = crop_roi(image, roi)
        if patch is None or patch.size == 0:
            continue
        vals.append(float(feature_fn(patch)))

    if not vals:
        return float(feature_fn(image))
    return float(max(vals))