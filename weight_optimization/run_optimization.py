"""
Main entry point for weight optimization pipeline.

Orchestrates: data preparation → optimization → evaluation → ablation.
"""

import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success/failure"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
        if result.returncode != 0:
            print(f"❌ Failed with return code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Weight Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_optimization.py --all                  # Run full pipeline
  python run_optimization.py --prepare              # Only prepare data
  python run_optimization.py --method ranking       # Optimize with ranking
  python run_optimization.py --method baseline      # Optimize with least squares
  python run_optimization.py --evaluate optimized_weights_ranking.yaml
        """,
    )
    
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--prepare", action="store_true", help="Prepare training data only")
    parser.add_argument("--method", choices=["ranking", "baseline", "both"],
                       help="Optimization method")
    parser.add_argument("--evaluate", metavar="WEIGHTS_FILE", help="Evaluate weights file")
    parser.add_argument("--ablate", metavar="WEIGHTS_FILE", help="Run ablation study")
    
    parser.add_argument("--output_dir", default="outputs_batch_50",
                       help="Path to batch results directory")
    parser.add_argument("--train_json", default=None,
                       help="Path to train.json (optional)")
    parser.add_argument("--baseline_config", default="../configs/selector.yaml",
                       help="Path to baseline selector config")
    

    parser.add_argument(
        "--policy",
        choices=["sign_policy", "lane_policy", "coverage_policy", "all_policies"],
        default=None,
        help="Run optimization for a specific policy or all policies",
    )

    args = parser.parse_args()
    
    # Default to ranking method if --all but no method specified
    if args.all and args.method is None:
        args.method = "ranking"
    
    # Determine what to run
    run_prepare = args.all or args.prepare
    run_optimize = args.all or args.method is not None
    run_evaluate = args.all or args.evaluate is not None
    run_ablate = args.all or args.ablate is not None
    
    if not any([run_prepare, run_optimize, run_evaluate, run_ablate]):
        parser.print_help()
        print("\n❌ Please specify at least one action (--all, --prepare, --method, --evaluate, --ablate)")
        return
    
    print("🚀 Weight Optimization Pipeline")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Baseline config: {args.baseline_config}")
    
    success_log = []
    
    # Step 1: Prepare data
    if run_prepare:
        cmd = [
            "python", "data/prepare_training_data.py",
            "--output_dir", args.output_dir,
            "--output_file", "outputs/metrics/training_data.json",
            "--summary_file", "outputs/metrics/training_data_summary.json",
        ]
        if args.train_json:
            cmd.extend(["--train_json", args.train_json])
        
        if run_command(cmd, "Step 1: Prepare Training Data"):
            success_log.append("✅ Data preparation")
        else:
            success_log.append("❌ Data preparation")
            if args.all:
                print("\n⚠️  Stopping due to failure. Cannot continue without training data.")
                return
    
    # Step 2: Optimize weights
    if run_optimize:
        ranking_ok = True

        if args.method in ["ranking", "both"]:
            cmd = [
                "python", "optimize/ranking_optimizer.py",
                "--data_file", "outputs/metrics/training_data.json",
                "--baseline_config", args.baseline_config,
                "--output_dir", "outputs/weights",
                "--epochs", "100",
            ]
            ranking_ok = run_command(cmd, "Step 2a: Ranking-based Optimization")
            if ranking_ok:
                success_log.append("✅ Ranking optimization")
            else:
                success_log.append("❌ Ranking optimization")
                if args.all and args.method == "ranking":
                    print("\n⚠️ Stopping due to ranking optimization failure.")
                    return
        
        if args.method in ["baseline", "both"]:
            cmd = [
                "python", "optimize/baseline_least_squares.py",
                "--data_file", "outputs/metrics/training_data.json",
                "--baseline_config", args.baseline_config,
                "--output_dir", "outputs/weights",
            ]
            if run_command(cmd, "Step 2b: Baseline (Least Squares)"):
                success_log.append("✅ Baseline optimization")
            else:
                success_log.append("❌ Baseline optimization")
    
    # Step 3: Evaluate
    if run_evaluate:
        weights_file = Path(args.evaluate) if args.evaluate else Path("outputs/weights/optimized_weights_ranking.yaml")
        if not weights_file.exists():
            print(f"⚠️ Skip evaluation because weights file does not exist: {weights_file}")
        else:
            cmd = [
                "python", "eval/evaluate_weights.py",
                "--weights_file", weights_file,
                "--baseline_config", args.baseline_config,
                "--data_file", "outputs/metrics/training_data.json",
                "--output_file", "outputs/metrics/evaluation_results.json",
            ]
            if run_command(cmd, "Step 3: Evaluate Optimized Weights"):
                success_log.append("✅ Evaluation")
            else:
                success_log.append("❌ Evaluation")
    
    # Step 4: Ablation
    if run_ablate:
        weights_file = args.ablate if args.ablate else "outputs/weights/optimized_weights_ranking.yaml"
        cmd = [
            "python", "eval/ablation_study.py",
            "--weights_file", weights_file,
            "--data_file", "outputs/metrics/training_data.json",
            "--output_dir", "outputs/ablation_results",
        ]
        if run_command(cmd, "Step 4: Ablation Study"):
            success_log.append("✅ Ablation study")
        else:
            success_log.append("❌ Ablation study")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 PIPELINE SUMMARY")
    print(f"{'='*60}")
    for status in success_log:
        print(status)
    
    print(f"\n📂 Output files:")
    print(f"   Weights:    weight_optimization/outputs/weights/")
    print(f"   Metrics:    weight_optimization/outputs/metrics/")
    print(f"   Ablation:   weight_optimization/outputs/ablation_results/")
    
    print(f"\n✅ Pipeline completed!")


if __name__ == "__main__":
    main()
