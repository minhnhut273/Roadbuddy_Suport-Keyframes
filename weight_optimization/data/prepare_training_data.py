"""
Prepare training data from outputs_batch_50 results.

Reads result.json files, extracts features, and creates (X, y) pairs where:
- X: normalized features from selected_frames
- y: binary label (1 if time matches support_frames, 0 otherwise)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
from dataclasses import dataclass, asdict



@dataclass
class TrainingItem:
    """Single training example"""
    question_type: str
    time_sec: float
    is_support: bool
    features: dict[str, float]  # raw features
    normalized_features: dict[str, float]  # normalized
    frame_idx: int
    sample_id: str


def load_result_files(output_dir: Path) -> list[dict]:
    """Load all result.json from output_dir/train_*/"""
    results = []
    output_dir = Path(output_dir)
    
    for train_dir in sorted(output_dir.glob("train_*")):
        result_file = train_dir / "result.json"
        if result_file.exists():
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"[WARN] Cannot load {result_file}: {e}")
    
    return results


def extract_features(results: list[dict]) -> list[TrainingItem]:
    """
    Extract features from results.
    
    For each sample:
      - Get support_frames timestamps (ground truth)
      - Get all_frames from selector result
      - For each frame, mark if it matches support frame (with tolerance)
    """
    items = []
    support_time_tolerance = 0.1  # 100ms tolerance for matching
    
    for result in results:
        sample_id = result.get("sample", {}).get("id", "unknown")
        question_type = result.get("sample", {}).get("type", "other")
        support_frames = result.get("sample", {}).get("support_frames", [])
        all_frames = result.get("selector_result", {}).get("all_frames", [])
        
        if not all_frames:
            print(f"[WARN] No frames in {sample_id}")
            continue
        
        # Convert support frames to set for fast lookup
        support_times_set = set(float(t) for t in support_frames)
        
        for frame in all_frames:
            time_sec = float(frame.get("time_sec", 0))
            frame_idx = int(frame.get("frame_idx", 0))
            
            # Check if this time matches any support frame
            is_support = any(
                abs(time_sec - st) < support_time_tolerance 
                for st in support_times_set
            )
            
            # Extract raw features
            components = frame.get("components", {})
            features = {
                "sharpness": float(components.get("sharpness", 0)),
                "edge_density": float(components.get("edge_density", 0)),
                "brightness": float(components.get("brightness", 0)),
                "novelty": float(components.get("novelty", 0)),
                "center_bias": float(components.get("center_bias", 0)),
                "roi_sharpness": float(components.get("roi_sharpness", 0)),
                "roi_edge_density": float(components.get("roi_edge_density", 0)),
                "roi_brightness": float(components.get("roi_brightness", 0)),
            }
            
            item = TrainingItem(
                question_type=question_type,
                time_sec=time_sec,
                is_support=is_support,
                features=features,
                normalized_features={},  # Will be filled later
                frame_idx=frame_idx,
                sample_id=sample_id,
            )
            items.append(item)
    
    return items


def normalize_features(items: list[TrainingItem]) -> list[TrainingItem]:
    """Min-max normalize features across all items"""
    feature_names = [
        "sharpness",
        "edge_density",
        "brightness",
        "novelty",
        "center_bias",
        "roi_sharpness",
        "roi_edge_density",
        "roi_brightness",
    ]
    
    for feat_name in feature_names:
        vals = [item.features[feat_name] for item in items]
        min_val = min(vals)
        max_val = max(vals)
        
        # Avoid division by zero
        if max_val - min_val < 1e-6:
            for item in items:
                item.normalized_features[feat_name] = 0.5
        else:
            for item in items:
                norm = (item.features[feat_name] - min_val) / (max_val - min_val)
                item.normalized_features[feat_name] = float(norm)
    
    return items


def split_data(
    items: list[TrainingItem],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[TrainingItem], list[TrainingItem], list[TrainingItem]]:
    np.random.seed(seed)

    by_sample: dict[str, list[TrainingItem]] = {}
    for item in items:
        by_sample.setdefault(item.sample_id, []).append(item)

    sample_ids = list(by_sample.keys())
    np.random.shuffle(sample_ids)

    n = len(sample_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(sample_ids[:n_train])
    val_ids = set(sample_ids[n_train:n_train + n_val])
    test_ids = set(sample_ids[n_train + n_val:])

    train, val, test = [], [], []
    for sid, group_items in by_sample.items():
        if sid in train_ids:
            train.extend(group_items)
        elif sid in val_ids:
            val.extend(group_items)
        else:
            test.extend(group_items)

    return train, val, test


def compute_statistics(items: list[TrainingItem]) -> dict[str, Any]:
    """Compute dataset statistics"""
    support_count = sum(1 for item in items if item.is_support)
    total_count = len(items)
    
    by_type = {}
    for item in items:
        if item.question_type not in by_type:
            by_type[item.question_type] = {"total": 0, "support": 0}
        by_type[item.question_type]["total"] += 1
        if item.is_support:
            by_type[item.question_type]["support"] += 1
    
    return {
        "total_samples": total_count,
        "support_samples": support_count,
        "non_support_samples": total_count - support_count,
        "support_ratio": support_count / total_count if total_count > 0 else 0,
        "by_question_type": by_type,
        "feature_names": [
            "sharpness",
            "edge_density",
            "brightness",
            "novelty",
            "center_bias",
            "roi_sharpness",
            "roi_edge_density",
            "roi_brightness",
        ],
    }


def save_dataset(
    train: list[TrainingItem],
    val: list[TrainingItem],
    test: list[TrainingItem],
    output_file: Path,
) -> None:
    """Save dataset to JSON"""
    dataset = {
        "train": [asdict(item) for item in train],
        "val": [asdict(item) for item in val],
        "test": [asdict(item) for item in test],
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Dataset saved to {output_file}")
    print(f"   Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from batch results")
    parser.add_argument("--output_dir", required=True, help="Path to outputs_batch_50 or similar")
    parser.add_argument("--output_file", default="../outputs/metrics/training_data.json")
    parser.add_argument("--summary_file", default="../outputs/metrics/training_data_summary.json")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Output dir not found: {output_dir}")
        return
    
    print("[1/4] Loading result files...")
    results = load_result_files(output_dir)
    print(f"  Loaded {len(results)} results")
    
    print("[2/4] Extracting features...")
    items = extract_features(results)
    print(f"  Extracted {len(items)} frame items")
    
    print("[3/4] Normalizing features...")
    items = normalize_features(items)
    
    print("[4/4] Splitting train/val/test...")
    train, val, test = split_data(items)
    
    # Save dataset
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(train, val, test, output_file)
    
    # Save summary
    all_stats = compute_statistics(items)
    summary = {
        "total_statistics": all_stats,
        "train_statistics": compute_statistics(train),
        "val_statistics": compute_statistics(val),
        "test_statistics": compute_statistics(test),
        "split_ratios": {
            "train": len(train) / len(items),
            "val": len(val) / len(items),
            "test": len(test) / len(items),
        }
    }
    
    summary_file = Path(args.summary_file)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary saved to {summary_file}")
    print(f"\nDataset Summary:")
    print(f"  Total: {all_stats['total_samples']} samples")
    print(f"  Support: {all_stats['support_samples']} ({all_stats['support_ratio']:.1%})")
    print(f"  Question types: {list(all_stats['by_question_type'].keys())}")


if __name__ == "__main__":
    main()
