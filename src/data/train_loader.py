from __future__ import annotations
from pathlib import Path
from typing import Any

from src.utils.io import read_json


def load_train_data(path: str | Path) -> list[dict[str, Any]]:
    obj = read_json(path)

    if isinstance(obj, dict) and "data" in obj:
        data = obj["data"]
    elif isinstance(obj, list):
        data = obj
    else:
        raise ValueError(f"Unsupported train json format: {path}")

    if not isinstance(data, list):
        raise ValueError("Expected 'data' to be a list")

    return data


from pathlib import Path

def build_video_abspath(video_root: str | Path, relative_video_path: str) -> str:
    video_root = Path(video_root)
    rel = Path(relative_video_path)

    parts = list(rel.parts)

    # bỏ các prefix thường gặp trong json
    if len(parts) >= 2 and parts[0] == "dataset" and parts[1] == "videos":
        rel = Path(*parts[2:])
    elif len(parts) >= 2 and parts[0] in ("train", "public_test") and parts[1] == "videos":
        rel = Path(*parts[2:])

    return str((video_root / rel).resolve())


def infer_question_type(sample: dict[str, Any]) -> str:
    if "type" in sample and sample["type"]:
        return str(sample["type"])

    q = str(sample.get("question", "")).lower()

    if "đúng hay sai" in q or "phải không" in q or "đúng không" in q:
        return "verification"
    if "có xuất hiện" in q or "có đèn" in q or "có biển" in q:
        return "object_presence"
    if "bao nhiêu" in q or "tốc độ" in q or "khoảng cách" in q:
        return "information_reading"
    if "hướng nào" in q or "đi theo hướng nào" in q or "muốn đi" in q:
        return "navigation"
    if "có mấy" in q or "bao nhiêu biển" in q or "đếm" in q:
        return "counting"
    if "được phép" in q or "có được" in q:
        return "rule_compliance"

    return "sign_identification"