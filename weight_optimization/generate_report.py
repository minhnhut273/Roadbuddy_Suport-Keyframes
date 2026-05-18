"""
Visualization and summary of optimization results.

Generates human-readable reports from JSON outputs.
"""

import json
import argparse
from pathlib import Path
from typing import Any


def load_json(file: Path) -> dict:
    """Load JSON file"""
    if not file.exists():
        print(f"⚠️  File not found: {file}")
        return {}
    
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_section(title: str, width: int = 70):
    """Print formatted section header"""
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def print_subsection(title: str, width: int = 70):
    """Print formatted subsection header"""
    print(f"\n{'-'*width}")
    print(f"  {title}")
    print(f"{'-'*width}")


def summarize_training_data(summary_file: Path):
    """Summarize training data statistics"""
    data = load_json(summary_file)
    if not data:
        return
    
    print_section("TRAINING DATA SUMMARY")
    
    total = data.get("total_statistics", {})
    print(f"\nTotal Samples: {total.get('total_samples', 'N/A')}")
    print(f"  - Support frames: {total.get('support_samples', 'N/A')} ({total.get('support_ratio', 0):.1%})")
    print(f"  - Non-support:    {total.get('non_support_samples', 'N/A')} ({1 - total.get('support_ratio', 0):.1%})")
    
    print(f"\nBy Question Type:")
    by_type = total.get('by_question_type', {})
    for qtype, counts in by_type.items():
        support = counts.get('support', 0)
        total_q = counts.get('total', 0)
        pct = (support / total_q * 100) if total_q > 0 else 0
        print(f"  {qtype:25s}: {total_q:4d} samples ({support:3d} support, {pct:5.1f}%)")
    
    # Split statistics
    print(f"\nData Split:")
    splits = data.get("split_ratios", {})
    for split_name, ratio in splits.items():
        print(f"  {split_name:10s}: {ratio:.1%}")


def summarize_evaluation(eval_file: Path):
    """Summarize evaluation results"""
    data = load_json(eval_file)
    if not data:
        return
    
    print_section("EVALUATION RESULTS")
    
    # Weights comparison
    print_subsection("Weight Changes")
    comparison = data.get("weight_comparison", {})
    
    print(f"{'Feature':<20} {'Baseline':<15} {'Optimized':<15} {'Change':<15}")
    print("-" * 65)
    
    for fname in [
    "sharpness",
    "edge_density",
    "brightness",
    "novelty",
    "center_bias",
    "roi_sharpness",
    "roi_edge_density",
    "roi_brightness",
    ]:
        if fname in comparison:
            comp = comparison[fname]
            baseline = comp.get("baseline", 0)
            optimized = comp.get("optimized", 0)
            pct_change = comp.get("pct_change", 0)
            
            sign = "→" if pct_change >= 0 else "←"
            print(f"{fname:<20} {baseline:<15.4f} {optimized:<15.4f} {sign} {pct_change:+6.1f}%")
    
    # Metrics comparison
    print_subsection("Performance Metrics")
    
    baseline_m = data.get("baseline_metrics", {})
    optimized_m = data.get("optimized_metrics", {})
    deltas = data.get("metric_deltas", {})
    
    metrics_order = ["hit@1", "hit@3", "hit@5", "mrr", "pairwise_accuracy"]
    
    print(f"{'Metric':<20} {'Baseline':<15} {'Optimized':<15} {'Change':<15} {'Status'}")
    print("-" * 80)
    
    for metric in metrics_order:
        if metric in baseline_m:
            baseline = baseline_m[metric]
            optimized = optimized_m.get(metric, 0)
            delta = optimized - baseline
            pct = (delta / baseline * 100) if baseline != 0 else 0
            
            status = "✅" if delta > 0 else ("➡️ " if abs(delta) < 0.001 else "❌")
            
            print(f"{metric:<20} {baseline:<15.4f} {optimized:<15.4f} {pct:+6.1f}% {status}")
    
    print(f"\nTest Set Size: {data.get('test_set_size', 'N/A')} samples")


def summarize_ablation(importance_file: Path):
    """Summarize ablation study results"""
    data = load_json(importance_file)
    if not data:
        return
    
    print_section("ABLATION STUDY (Feature Importance)")
    
    print(f"{'Feature':<20} {'Baseline Acc':<15} {'Drop in Acc':<15} {'Importance':<15}")
    print("-" * 65)
    
    # Sort by importance
    sorted_items = sorted(data.items(),
                         key=lambda x: x[1].get("importance", 0),
                         reverse=True)
    
    for feature, ablation in sorted_items:
        baseline_acc = ablation.get("baseline_accuracy", 0)
        drop = ablation.get("drop_in_accuracy", 0)
        importance = ablation.get("importance", 0)
        
        # Bar chart
        bar_len = int(importance * 50)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        
        print(f"{feature:<20} {baseline_acc:<15.4f} {drop:<15.4f} {importance:.4f}")
        print(f"{'':20} {bar}")


def generate_report(output_dir: Path):
    """Generate complete HTML-style report"""
    
    summary_file = output_dir / "metrics" / "training_data_summary.json"
    eval_file = output_dir / "metrics" / "evaluation_results.json"
    importance_file = output_dir / "ablation_results" / "feature_importance.json"
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "WEIGHT OPTIMIZATION REPORT" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    
    summarize_training_data(summary_file)
    summarize_evaluation(eval_file)
    summarize_ablation(importance_file)
    
    print_section("RECOMMENDATIONS")
    
    # Read metrics for recommendations
    eval_data = load_json(eval_file)
    if eval_data:
        deltas = eval_data.get("metric_deltas", {})
        hit1_delta = deltas.get("hit@1", {}).get("pct_change", 0)
        
        if hit1_delta > 2:
            print("\n✅ POSITIVE RESULT: Optimized weights show improvement!")
            print("   Recommendation: Use optimized_weights_ranking.yaml in production")
        elif hit1_delta > 0:
            print("\n⚠️  MARGINAL IMPROVEMENT: Small gains observed")
            print("   Recommendation: Review ablation study to understand which features help")
        else:
            print("\n❌ NO IMPROVEMENT: Weights did not improve performance")
            print("   Recommendation: Check if support frames correlate with visual features")
            print("                   Consider alternative supervision signals or feature engineering")
    
    # Read importance for recommendations
    ablation_data = load_json(importance_file)
    if ablation_data:
        top_features = sorted(ablation_data.items(),
                            key=lambda x: x[1].get("importance", 0),
                            reverse=True)[:3]
        
        print("\n📊 TOP IMPORTANT FEATURES:")
        for feat, data_item in top_features:
            importance = data_item.get("importance", 0)
            print(f"   1. {feat}: {importance:.4f}")
    
    print("\n" + "=" * 70)
    print("Report generated successfully!")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate optimization summary report")
    parser.add_argument("--output_dir", default="./outputs",
                       help="Output directory containing results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return
    
    generate_report(output_dir)


if __name__ == "__main__":
    main()
