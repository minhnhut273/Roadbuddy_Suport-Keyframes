"""
Evaluate optimized weights on test set and compare with baseline.
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
from optimize.metrics import compute_metrics, format_metrics, compute_metrics_with_times



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


def filter_items_by_policy(
    items: list[dict[str, Any]],
    policy_name: str | None,
) -> list[dict[str, Any]]:
    if policy_name is None:
        return items
    out = []
    for item in items:
        qtype = item.get("question_type", "other")
        mapped = TYPE_TO_POLICY.get(qtype, "sign_policy")
        if mapped == policy_name:
            out.append(item)
    return out


def load_weights_from_yaml(config_file: Path) -> dict[str, float]:
    """Load weights from YAML config"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"  [DEBUG] load_weights_from_yaml reading: {config_file}")
    print(f"  [DEBUG] config type: {type(config)}")

    if isinstance(config, dict):
        print(f"  [DEBUG] top-level keys: {list(config.keys())}")

    if isinstance(config, dict) and "weights" in config:
        print("  [DEBUG] using config['weights']")
        return config["weights"]
    elif isinstance(config, dict):
        print("  [DEBUG] using full config as weights dict")
        return config
    else:
        print("  [DEBUG] returning empty dict")
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
    after_penalty: float = 1.5,
) -> dict[str, float]:
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

    grouped = {}
    for item in items:
        grouped.setdefault(item["sample_id"], []).append(item)

    all_metrics = []

    for sample_id, group_items in grouped.items():
        y_true = []
        y_scores = []
        times = []

        for item in group_items:
            features = item["normalized_features"]
            score = sum(weights.get(fname, 0.0) * features.get(fname, 0.0)
                        for fname in feature_names)
            y_scores.append(score)
            y_true.append(1.0 if item["is_support"] else 0.0)
            times.append(float(item["time_sec"]))

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        times = np.array(times)

        if np.sum(y_true) == 0:
            continue

        all_metrics.append(
            compute_metrics_with_times(
                y_true=y_true,
                y_scores=y_scores,
                times=times,
                after_penalty=after_penalty,
            )
        )

    if not all_metrics:
        return {}

    keys = all_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}

def compare_weights(
    baseline_weights: dict[str, float],
    optimized_weights: dict[str, float],
) -> dict[str, Any]:
    """Compare two sets of weights"""
    feature_names = list(baseline_weights.keys())
    
    comparison = {}
    for fname in feature_names:
        baseline_w = baseline_weights.get(fname, 0.0)
        optimized_w = optimized_weights.get(fname, 0.0)
        delta = optimized_w - baseline_w
        pct_change = (delta / baseline_w * 100) if baseline_w != 0 else 0.0
        
        comparison[fname] = {
            "baseline": baseline_w,
            "optimized": optimized_w,
            "delta": delta,
            "pct_change": pct_change,
        }
    
    return comparison



def load_baseline_weights_from_config(
    config_path: Path,
    policy_name: str | None = None,
) -> dict[str, float]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if policy_name is None:
        return cfg.get("weights", {})

    policy_cfg = cfg.get("policy_configs", {}).get(policy_name, {})
    return policy_cfg.get("weights", cfg.get("weights", {}))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate optimized weights")
    parser.add_argument("--weights_file", required=True,
                       help="Path to optimized weights YAML")
    parser.add_argument("--baseline_config", default="../configs/selector.yaml")
    parser.add_argument("--data_file", default="../outputs/metrics/training_data.json")
    parser.add_argument("--output_file", default="../outputs/metrics/evaluation_results.json")
    parser.add_argument("--after_penalty", type=float, default=1.5)

    parser.add_argument(
        "--policy",
        choices=["sign_policy", "lane_policy", "coverage_policy"],
        default=None,
        help="Evaluate on a specific policy subset only",
    )

    args = parser.parse_args()
    
    # Load data
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return
    
    print("[1/4] Loading data and weights...")
    train_items, val_items, test_items = load_training_data(data_file)
    
    train_items = filter_items_by_policy(train_items, args.policy)
    val_items = filter_items_by_policy(val_items, args.policy)
    test_items = filter_items_by_policy(test_items, args.policy)

    print(f"  Policy filter: {args.policy if args.policy else 'ALL'}")
    print(f"  Filtered Test size: {len(test_items)}")



    # Load baseline weights
# Load baseline weights
    baseline_config = Path(args.baseline_config)
    print(f"  Baseline config path: {baseline_config.resolve()}")
    print(f"  Baseline config exists: {baseline_config.exists()}")

    baseline_weights = {}
    if baseline_config.exists():
        with open(baseline_config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            print(f"  Baseline config top-level keys: {list(cfg.keys()) if isinstance(cfg, dict) else type(cfg)}")
        baseline_weights = load_baseline_weights_from_config(
            baseline_config,
            policy_name=args.policy,
        )
        print(f"  Loaded baseline weights keys: {list(baseline_weights.keys())}")
        print(f"  Loaded baseline weights values: {baseline_weights}")
    else:
        print("  WARNING: baseline config file not found")
    
    # Load optimized weights
    weights_file = Path(args.weights_file)
    if not weights_file.exists():
        print(f"❌ Weights file not found: {weights_file}")
        return
    
    optimized_weights = load_weights_from_yaml(weights_file)
    print(f"  Optimized weights loaded: {list(optimized_weights.keys())}")
    
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
    
    # Evaluate
    print("[2/4] Evaluating on test set...")
    baseline_metrics = evaluate_weights(test_items, baseline_weights, feature_names, after_penalty=args.after_penalty)
    optimized_metrics = evaluate_weights(test_items, optimized_weights, feature_names, after_penalty=args.after_penalty)

    print("[3/4] Computing comparison...")
    comparison = compare_weights(baseline_weights, optimized_weights)
    
    # Compute deltas
    metric_deltas = {}
    for metric_name in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric_name]
        optimized_val = optimized_metrics[metric_name]
        delta = optimized_val - baseline_val
        pct_change = (delta / baseline_val * 100) if baseline_val != 0 else 0.0
        
        metric_deltas[metric_name] = {
            "baseline": baseline_val,
            "optimized": optimized_val,
            "delta": delta,
            "pct_change": pct_change,
        }
    
    # Save results
    print("[4/4] Saving results...")
    results = {
        "baseline_weights": baseline_weights,
        "optimized_weights": optimized_weights,
        "weight_comparison": comparison,
        "baseline_metrics": baseline_metrics,
        "optimized_metrics": optimized_metrics,
        "metric_deltas": metric_deltas,
        "test_set_size": len(test_items),
    }
    
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f" Results saved to {output_file}\n")
    
    # Print summary
    print("=" * 60)
    print("WEIGHT COMPARISON")
    print("=" * 60)
    for fname, comp in comparison.items():
        print(f"{fname:20s}: {comp['baseline']:.4f} -> {comp['optimized']:.4f} ({comp['pct_change']:+.1f}%)")
    
    print("\n" + "=" * 60)
    print("BASELINE METRICS (Test Set)")
    print("=" * 60)
    print(format_metrics(baseline_metrics))
    
    print("\n" + "=" * 60)
    print("OPTIMIZED METRICS (Test Set)")
    print("=" * 60)
    print(format_metrics(optimized_metrics))
    
    print("\n" + "=" * 60)
    print("METRIC CHANGES")
    print("=" * 60)
    for metric_name, delta in metric_deltas.items():
        sign = "✅" if delta['delta'] > 0 else "❌"
        print(f"{sign} {metric_name:20s}: {delta['baseline']:.4f} -> {delta['optimized']:.4f} ({delta['pct_change']:+.1f}%)")


if __name__ == "__main__":
    main()
