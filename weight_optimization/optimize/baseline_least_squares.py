"""
Baseline: Linear Least Squares weight optimization.

Treat it as a simple regression problem:
  - Support frames target score = 1.0
  - Non-support frames target score = 0.0
  
This is a BASELINE approach that may not be ideal for ranking problem,
but provides a simple comparison point.
"""

from __future__ import annotations

import numpy as np
import json
import sys
from pathlib import Path
from typing import Any
import yaml

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_metrics, format_metrics


class LeastSquaresOptimizer:
    """
    Simple least squares approach for weight optimization.
    
    Solves: argmin ||Xw - y||^2 + lambda ||w - w_baseline||^2
    
    Note: This assumes support frames should score exactly 1.0 and
    non-support should score 0.0, which is a simplification.
    """
    
    def __init__(
        self,
        feature_names: list[str],
        baseline_weights: dict[str, float] | None = None,
        l2_lambda: float = 0.1,
    ):
        self.feature_names = feature_names
        self.l2_lambda = l2_lambda
        
        if baseline_weights:
            self.baseline_weights = np.array([baseline_weights.get(fname, 1.0 / len(feature_names))
                                              for fname in feature_names])
        else:
            self.baseline_weights = np.ones(len(feature_names)) / len(feature_names)
    
    def fit(
        self,
        train_items: list[dict[str, Any]],
        verbose: bool = True,
    ) -> dict[str, float]:
        """
        Fit weights using least squares.
        
        Args:
            train_items: training set (with normalized_features and is_support)
            verbose: print info
        
        Returns:
            dict with fitted weights
        """
        # Build feature matrix
        X = []
        y = []
        
        for item in train_items:
            features = item["normalized_features"]
            feature_vec = np.array([features.get(fname, 0.0) for fname in self.feature_names])
            X.append(feature_vec)
            y.append(1.0 if item["is_support"] else 0.0)
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        if verbose:
            print(f"Fitting on {len(X)} samples with {X.shape[1]} features")
        
        # Add regularization term: (w - w_baseline)
        # Augment X and y
        n_samples = X.shape[0]
        X_aug = np.vstack([X, np.sqrt(self.l2_lambda) * np.eye(len(self.feature_names))])
        y_aug = np.vstack([y, np.sqrt(self.l2_lambda) * self.baseline_weights.reshape(-1, 1)])
        
        # Solve least squares: (X'X + lambda*I)w = X'y + lambda*w_baseline
        # Using numpy's least squares solver
        w, residuals, rank, s = np.linalg.lstsq(X_aug, y_aug, rcond=None)
        
        weights = {fname: float(w[i, 0]) for i, fname in enumerate(self.feature_names)}
        
        # Normalize to sum=1 for interpretability
        total = sum(weights.values())
        if total > 0:
            weights = {fname: v / total for fname, v in weights.items()}
        
        return weights
    
    def evaluate(
        self,
        items: list[dict[str, Any]],
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Evaluate weights on a dataset"""
        scores = []
        labels = []
        
        for item in items:
            features = item["normalized_features"]
            score = sum(weights.get(fname, 0.0) * features.get(fname, 0.0)
                       for fname in self.feature_names)
            scores.append(score)
            labels.append(1.0 if item["is_support"] else 0.0)
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        return compute_metrics(labels, scores)


def load_training_data(data_file: Path) -> tuple[list, list, list]:
    """Load train/val/test from prepared data file"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data["train"], data["val"], data["test"]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline: Least squares weight optimization")
    parser.add_argument("--data_file", default="../outputs/metrics/training_data.json")
    parser.add_argument("--baseline_config", default="../configs/selector.yaml")
    parser.add_argument("--output_dir", default="../outputs/weights")
    parser.add_argument("--l2_lambda", type=float, default=0.1)
    args = parser.parse_args()
    
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("Run: python data/prepare_training_data.py first")
        return
    
    print("[1/3] Loading training data...")
    train_items, val_items, test_items = load_training_data(data_file)
    print(f"  Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")
    
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
            baseline_weights = cfg.get("weights", {})
            print(f"  Loaded baseline weights keys: {list(baseline_weights.keys())}")
            print(f"  Loaded baseline weights values: {baseline_weights}")
    else:
        print("  WARNING: baseline config file not found")
    
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
    
    print("[2/3] Fitting least squares optimizer...")
    optimizer = LeastSquaresOptimizer(
        feature_names=feature_names,
        baseline_weights=baseline_weights,
        l2_lambda=args.l2_lambda,
    )
    
    weights = optimizer.fit(train_items)
    
    print("[3/3] Saving results...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save optimized weights
    weights_file = output_dir / "optimized_weights_leastsquares.yaml"
    with open(weights_file, 'w', encoding='utf-8') as f:
        yaml.dump(weights, f)
    print(f"✅ Weights saved to {weights_file}")
    
    # Evaluate on validation and test
    val_metrics = optimizer.evaluate(val_items, weights)
    test_metrics = optimizer.evaluate(test_items, weights)
    
    print("\n📊 Validation Results:")
    print(format_metrics(val_metrics))
    
    print("\n📊 Test Results:")
    print(format_metrics(test_metrics))
    
    # Save metrics
    metrics_file = output_dir.parent / "metrics" / "baseline_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            "method": "least_squares",
            "baseline_weights": baseline_weights,
            "optimized_weights": weights,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        }, f, indent=2)
    print(f"\n✅ Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
