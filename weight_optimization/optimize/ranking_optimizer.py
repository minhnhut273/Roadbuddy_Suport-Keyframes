"""
Ranking-based weight optimizer using pairwise loss.

Key insight: Instead of fitting support frames ≈ 1.0, we enforce that
support frames score HIGHER than non-support frames (ranking).

Loss: Pairwise ranking loss
  L = sum over (support, non-support) pairs: max(0, margin - score_support + score_nonsupport)
"""

from __future__ import annotations

import numpy as np
import json
import sys
from pathlib import Path
from typing import Any
import yaml
import random

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_metrics, compute_metrics_with_times, format_metrics


def group_items_by_sample(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(item["sample_id"], []).append(item)
    return grouped


def compute_grouped_metrics(
    items: list[dict[str, Any]],
    score_fn,
    after_penalty: float = 1.5,
) -> dict[str, float]:
    grouped = group_items_by_sample(items)
    all_metrics = []

    for sample_id, group_items in grouped.items():
        y_true = np.array([1.0 if x["is_support"] else 0.0 for x in group_items])
        y_scores = np.array([score_fn(x["normalized_features"]) for x in group_items])
        times = np.array([float(x["time_sec"]) for x in group_items])

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
        return {
            "hit@1": 0.0,
            "hit@3": 0.0,
            "hit@5": 0.0,
            "mrr": 0.0,
            "ap": 0.0,
            "mean_min_distance": 0.0,
            "pairwise_accuracy": 0.0,
            "asym_nearest_distance": 0.0,
        }

    keys = all_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}

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


def get_policy_from_item(item: dict[str, Any]) -> str:
    """Get policy from item based on question_type"""
    qtype = item.get("question_type", "other")
    return TYPE_TO_POLICY.get(qtype, "sign_policy")


def filter_items_by_policy(
    items: list[dict[str, Any]],
    policy_name: str | None,
) -> list[dict[str, Any]]:
    if policy_name is None:
        return items

    out = []
    for item in items:
        policy = get_policy_from_item(item)
        if policy == policy_name:
            out.append(item)
    return out

class RankingWeightOptimizer:
    """
    Optimize weights using pairwise ranking loss.
    
    Features:
      - Per-question-type weights (optional)
      - Pairwise ranking loss
      - L2 regularization to stay close to baseline
      - Early stopping on validation metric
    """
    
    def __init__(
        self,
        feature_names: list[str],
        baseline_weights: dict[str, float] | None = None,
        learning_rate: float = 0.01,
        margin: float = 0.1,
        l2_lambda: float = 0.01,
        per_question_type: bool = False,
        pairs_per_support: int = 32,
        hard_negative_ratio: float = 0.5,
    ):
        self.feature_names = feature_names
        self.learning_rate = learning_rate
        self.margin = margin
        self.l2_lambda = l2_lambda
        self.per_question_type = per_question_type
        self.pairs_per_support = pairs_per_support
        self.hard_negative_ratio = hard_negative_ratio
        
        # Initialize weights (uniform or from baseline)
        if baseline_weights:
            self.weights = {fname: baseline_weights.get(fname, 1.0 / len(feature_names))
                           for fname in feature_names}
        else:
            self.weights = {fname: 1.0 / len(feature_names) for fname in feature_names}
        
        self.baseline_weights = self.weights.copy()
        self.history = []
    
    def _compute_score(self, features: dict[str, float]) -> float:
        """Compute weighted score for features"""
        score = sum(self.weights.get(fname, 0.0) * features.get(fname, 0.0)
                   for fname in self.feature_names)
        return float(score)
    
    def _temporal_closeness(self, t_support: float, t_neg: float) -> float:
        """Compute temporal closeness score (higher = closer)"""
        return 1.0 / (abs(t_support - t_neg) + 1e-6)
    
    def _policy_hard_negative_score(
        self,
        support_item: dict[str, Any],
        neg_item: dict[str, Any],
    ) -> float:
        """
        Compute hard negative score. Higher = harder negative.
        Combines current model score, temporal proximity, and policy-specific signals.
        """
        policy = get_policy_from_item(support_item)

        support_time = float(support_item["time_sec"])
        neg_time = float(neg_item["time_sec"])

        neg_features = neg_item["normalized_features"]
        current_score = self._compute_score(neg_features)
        temporal = self._temporal_closeness(support_time, neg_time)

        if policy == "sign_policy":
            return (
                0.50 * current_score +
                0.30 * temporal +
                0.20 * neg_features.get("roi_edge_density", 0.0)
            )

        if policy == "lane_policy":
            return (
                0.40 * current_score +
                0.30 * temporal +
                0.30 * neg_features.get("edge_density", 0.0)
            )

        if policy == "coverage_policy":
            return (
                0.50 * current_score +
                0.30 * temporal +
                0.20 * neg_features.get("roi_sharpness", 0.0)
            )

        # Default fallback
        return 0.5 * current_score + 0.5 * temporal
    
    def _sample_negatives_for_support(
        self,
        support_item: dict[str, Any],
        nonsupport_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Sample hard and random negatives for a support frame.
        
        Returns a mix of:
          - Top hard negatives (ranked by difficulty)
          - Random negatives from remaining pool
        
        Mix ratio controlled by hard_negative_ratio.
        """
        if not nonsupport_items:
            return []

        k_total = min(self.pairs_per_support, len(nonsupport_items))
        if k_total <= 0:
            return []

        k_hard = int(round(k_total * self.hard_negative_ratio))
        k_hard = min(k_hard, len(nonsupport_items))
        k_rand = k_total - k_hard

        # Rank negatives by hard score
        ranked = sorted(
            nonsupport_items,
            key=lambda x: self._policy_hard_negative_score(support_item, x),
            reverse=True,
        )

        hard_negs = ranked[:k_hard]

        remaining = ranked[k_hard:]
        if k_rand > 0 and remaining:
            rand_negs = random.sample(remaining, min(k_rand, len(remaining)))
        else:
            rand_negs = []

        return hard_negs + rand_negs
    
    def _compute_pairwise_loss(
        self,
        grouped_items: dict[str, list[dict[str, Any]]],
    ) -> float:
        """
        Pairwise ranking loss within each sample_id only.
        """
        total_loss = 0.0
        num_pairs = 0

        for sample_id, items in grouped_items.items():
            support_items = [x for x in items if x["is_support"]]
            nonsupport_items = [x for x in items if not x["is_support"]]

            if not support_items or not nonsupport_items:
                continue

            for s_item in support_items:
                sampled_negatives = self._sample_negatives_for_support(
                    support_item=s_item,
                    nonsupport_items=nonsupport_items,
                )

                score_s = self._compute_score(s_item["normalized_features"])

                for ns_item in sampled_negatives:
                    score_ns = self._compute_score(ns_item["normalized_features"])
                    pair_loss = max(0.0, self.margin - score_s + score_ns)
                    total_loss += pair_loss
                    num_pairs += 1

        avg_loss = total_loss / num_pairs if num_pairs > 0 else 0.0

        l2_loss = sum(
            (self.weights[fname] - self.baseline_weights[fname]) ** 2
            for fname in self.feature_names
        )

        return avg_loss + self.l2_lambda * l2_loss
    
    def _compute_gradient(
        self,
        grouped_items: dict[str, list[dict[str, Any]]],
    ) -> dict[str, float]:
        """
        Compute approximate gradient using finite differences.
        """
        eps = 1e-4
        grads = {}

        base_loss = self._compute_pairwise_loss(grouped_items)

        for fname in self.feature_names:
            self.weights[fname] += eps
            loss_plus = self._compute_pairwise_loss(grouped_items)
            self.weights[fname] -= eps

            grad = (loss_plus - base_loss) / eps
            grads[fname] = grad

        return grads
    
    def train(
        self,
        train_items: list[dict[str, Any]],
        val_items: list[dict[str, Any]],
        epochs: int = 50,
        verbose: bool = True,
        after_penalty: float = 1.5,
    ) -> dict[str, Any]:
        """
        Train weights on training set with early stopping on validation.
        
        Args:
            train_items: training set (dicts with normalized_features and is_support)
            val_items: validation set
            epochs: max number of epochs
            verbose: print progress
        
        Returns:
            training history dict
        """
        train_grouped = group_items_by_sample(train_items)
        val_grouped = group_items_by_sample(val_items)

        num_train_support = sum(sum(1 for x in items if x["is_support"]) for items in train_grouped.values())
        num_train_non = sum(sum(1 for x in items if not x["is_support"]) for items in train_grouped.values())

        num_val_support = sum(sum(1 for x in items if x["is_support"]) for items in val_grouped.values())
        num_val_non = sum(sum(1 for x in items if not x["is_support"]) for items in val_grouped.values())

        print(f"Training on {len(train_grouped)} samples | {num_train_support} support, {num_train_non} non-support frames")
        print(f"Validating on {len(val_grouped)} samples | {num_val_support} support, {num_val_non} non-support frames")
        
        best_metric = -np.inf
        patience = 10
        patience_counter = 0
        

        best_metrics = None
        best_weights = self.weights.copy()

        for epoch in range(epochs):
            # Compute gradients
            grads = self._compute_gradient(train_grouped)
            
            # Gradient descent step
            for fname in self.feature_names:
                self.weights[fname] -= self.learning_rate * grads.get(fname, 0.0)
            
            # Ensure weights remain non-negative and normalized
            min_weight = min(self.weights.values())
            if min_weight < 0:
                # Shift all weights
                offset = abs(min_weight) + 1e-3
                for fname in self.feature_names:
                    self.weights[fname] += offset
            
            # Normalize to sum=1 (optional, for interpretability)
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for fname in self.feature_names:
                    self.weights[fname] /= total_weight
            
            # Evaluate on validation
            # val_scores = np.array([self._compute_score(item["normalized_features"])
            #                        for item in val_items])
            # val_labels = np.array([float(item["is_support"]) for item in val_items])
            # val_metrics = compute_metrics(val_labels, val_scores)

            val_metrics = compute_grouped_metrics(
                val_items,
                self._compute_score,
                after_penalty=after_penalty,
            )    

            # Use pairwise accuracy as metric
            current_metric = (
                0.20 * val_metrics["hit@1"] +
                0.30 * val_metrics["hit@3"] +
                0.50 * val_metrics["hit@5"] -
                0.05 * val_metrics["asym_nearest_distance"]
            )
            self.history.append({
                "epoch": epoch,
                "metrics": val_metrics,
                "weights": self.weights.copy(),
            })
            
            # Early stopping
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                best_weights = self.weights.copy()
                best_metrics = val_metrics.copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            if verbose:
                print(
                    f"Epoch {epoch+1}: objective={current_metric:.4f} | "
                    f"hit@1={val_metrics['hit@1']:.4f} | "
                    f"hit@3={val_metrics['hit@3']:.4f} | "
                    f"hit@5={val_metrics['hit@5']:.4f} | "
                    f"asym_dist={val_metrics['asym_nearest_distance']:.4f}"
                )
        
        # Restore best weights
        self.weights = best_weights
        
        return {
            "final_weights": self.weights,
            "best_metric": best_metric,
            "best_metrics": best_metrics,
            "epochs_trained": len(self.history),
            "history": self.history,
        }


def load_training_data(data_file: Path) -> tuple[list, list, list]:
    """Load train/val/test from prepared data file"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data["train"], data["val"], data["test"]

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
    
    parser = argparse.ArgumentParser(description="Train weights using ranking loss")
    parser.add_argument("--data_file", default="../outputs/metrics/training_data.json")
    parser.add_argument("--baseline_config", default="../configs/selector.yaml")
    parser.add_argument("--output_dir", default="../outputs/weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--pairs_per_support", type=int, default=32)
    parser.add_argument("--after_penalty", type=float, default=1.5)
    parser.add_argument("--hard_negative_ratio", type=float, default=0.5)

    parser.add_argument(
        "--policy",
        choices=["sign_policy", "lane_policy", "coverage_policy"],
        default=None,
        help="Train weights for a specific policy only",
    )

    args = parser.parse_args()
    
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("Run: python data/prepare_training_data.py first")
        return
    
    print("[1/3] Loading training data...")
    train_items, val_items, test_items = load_training_data(data_file)
    

    train_items = filter_items_by_policy(train_items, args.policy)
    val_items = filter_items_by_policy(val_items, args.policy)
    test_items = filter_items_by_policy(test_items, args.policy)

    print(f"  Policy filter: {args.policy if args.policy else 'ALL'}")
    print(f"  Filtered Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")


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


            # baseline_weights = cfg.get("weights", {})


            baseline_weights = load_baseline_weights_from_config(
                baseline_config,
                policy_name=args.policy,
            )
            print(f"  Loaded baseline weights keys: {list(baseline_weights.keys())}")
            print(f"  Loaded baseline weights values: {baseline_weights}")


            # print(f"  Loaded baseline weights keys: {list(baseline_weights.keys())}")
            # print(f"  Loaded baseline weights values: {baseline_weights}")
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
    
    print("[2/3] Training optimizer...")
    optimizer = RankingWeightOptimizer(
        feature_names=feature_names,
        baseline_weights=baseline_weights,
        learning_rate=args.learning_rate,
        margin=args.margin,
        pairs_per_support=args.pairs_per_support,
        hard_negative_ratio=args.hard_negative_ratio,
    )
    
    result = optimizer.train(train_items, val_items, epochs=args.epochs,after_penalty=args.after_penalty,)
    
    print("[3/3] Saving results...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save optimized weights

    suffix = args.policy if args.policy else "global"
    weights_file = output_dir / f"optimized_weights_{suffix}.yaml"

    log_file = output_dir / f"training_log_{suffix}.json"

    with open(weights_file, 'w', encoding='utf-8') as f:
        yaml.dump(result["final_weights"], f)
    print(f" Weights saved to {weights_file}")
    
    # Save training log
    # log_file = output_dir / "training_log_ranking.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            "final_weights": result["final_weights"],
            "baseline_weights": baseline_weights,
            "best_metric": result["best_metric"],
            "epochs_trained": result["epochs_trained"],
            "history": [
                {**h, "weights": h["weights"]}
                for h in result["history"]
            ]
        }, f, indent=2)
    print(f" Log saved to {log_file}")
    
    print("\n Best Validation Results:")
    if result.get("best_metrics") is not None:
        print(format_metrics(result["best_metrics"]))


if __name__ == "__main__":
    main()
