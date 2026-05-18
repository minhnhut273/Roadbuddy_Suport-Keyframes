"""
Ablation study: analyze feature importance by dropping/zeroing features.
"""

from __future__ import annotations

import json
import numpy as np
import sys
from pathlib import Path
from typing import Any
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from optimize.metrics import compute_metrics


def load_weights_from_yaml(config_file: Path) -> dict[str, float]:
    """Load weights from YAML config"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if isinstance(config, dict) and "weights" in config:
        return config["weights"]
    elif isinstance(config, dict):
        return config
    else:
        return {}


def load_training_data(data_file: Path) -> tuple[list, list, list]:
    """Load train/val/test from prepared data file"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data["train"], data["val"], data["test"]


def evaluate_weights(
    items: list[dict[str, Any]],
    weights: dict[str, float],
    feature_names: list[str] = None,
) -> dict[str, float]:
    """Compute metrics for weights on items"""
    if feature_names is None:
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
    
    scores = []
    labels = []
    
    for item in items:
        features = item["normalized_features"]
        score = sum(weights.get(fname, 0.0) * features.get(fname, 0.0)
                   for fname in feature_names)
        scores.append(score)
        labels.append(1.0 if item["is_support"] else 0.0)
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    return compute_metrics(labels, scores)


def ablation_single_feature(
    items: list[dict[str, Any]],
    weights: dict[str, float],
    feature_to_drop: str,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Evaluate impact of dropping a single feature.
    
    Returns drop-in performance (lower is worse).
    """
    weights_ablated = weights.copy()
    weights_ablated[feature_to_drop] = 0.0
    
    return evaluate_weights(items, weights_ablated, feature_names)


def ablation_feature_groups(
    items: list[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, dict[str, float]]:
    """
    Analyze importance of feature groups (non-ROI vs ROI features).
    """
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
    
    # Group features
    groups = {
        "all_features": feature_names,
        "global_only": ["sharpness", "edge_density", "brightness"],
        "roi_only": ["roi_sharpness", "roi_edge_density", "roi_brightness"],
        "temporal": ["novelty"],
        "spatial": ["center_bias"],
        "global_plus_roi": [
            "sharpness", "edge_density", "brightness",
            "roi_sharpness", "roi_edge_density", "roi_brightness"
        ],
    }
    
    results = {}
    for group_name, group_features in groups.items():
        weights_group = {fname: weights.get(fname, 0.0) for fname in group_features}
        results[group_name] = evaluate_weights(items, weights_group, group_features)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ablation study for feature importance")
    parser.add_argument("--weights_file", required=True)
    parser.add_argument("--data_file", default="../outputs/metrics/training_data.json")
    parser.add_argument("--output_dir", default="../outputs/ablation_results")
    args = parser.parse_args()
    
    # Load data
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return
    
    print("[1/3] Loading data and weights...")
    train_items, val_items, test_items = load_training_data(data_file)
    
    weights_file = Path(args.weights_file)
    if not weights_file.exists():
        print(f"❌ Weights file not found: {weights_file}")
        return
    
    weights = load_weights_from_yaml(weights_file)
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
    
    # Baseline metrics
    baseline_metrics = evaluate_weights(test_items, weights, feature_names)
    baseline_pairwise_acc = baseline_metrics["pairwise_accuracy"]
    
    print("[2/3] Computing single-feature ablations...")
    single_ablations = {}
    for feature in feature_names:
        ablated_metrics = ablation_single_feature(test_items, weights, feature, feature_names)
        drop_in_accuracy = baseline_pairwise_acc - ablated_metrics["pairwise_accuracy"]
        
        single_ablations[feature] = {
            "baseline_accuracy": baseline_pairwise_acc,
            "ablated_accuracy": ablated_metrics["pairwise_accuracy"],
            "drop_in_accuracy": drop_in_accuracy,
            "importance": drop_in_accuracy,  # higher = more important
            "all_metrics": ablated_metrics,
        }
    
    print("[3/3] Computing feature group analysis...")
    group_analysis = ablation_feature_groups(test_items, weights)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Single feature importance
    importance_file = output_dir / "feature_importance.json"
    with open(importance_file, 'w', encoding='utf-8') as f:
        json.dump(single_ablations, f, indent=2)
    print(f"✅ Feature importance saved to {importance_file}")
    
    # Group analysis
    group_file = output_dir / "group_analysis.json"
    with open(group_file, 'w', encoding='utf-8') as f:
        json.dump(group_analysis, f, indent=2)
    print(f"✅ Group analysis saved to {group_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (based on pairwise accuracy drop)")
    print("=" * 60)
    
    # Sort by importance
    sorted_features = sorted(single_ablations.items(),
                            key=lambda x: x[1]["importance"],
                            reverse=True)
    
    for feature, ablation in sorted_features:
        importance = ablation["importance"]
        drop = ablation["drop_in_accuracy"]
        weight = weights.get(feature, 0.0)
        
        print(f"{feature:20s} | weight={weight:.4f} | importance={importance:.4f} | drop={drop:.4f}")
    
    print("\n" + "=" * 60)
    print("FEATURE GROUP ANALYSIS (Pairwise Accuracy)")
    print("=" * 60)
    for group_name, metrics in group_analysis.items():
        acc = metrics.get("pairwise_accuracy", 0.0)
        print(f"{group_name:20s}: {acc:.4f}")


if __name__ == "__main__":
    main()
