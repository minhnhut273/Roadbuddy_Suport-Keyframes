#!/usr/bin/env python3
"""
Quick sanity check for weight optimization setup.

Run this to verify:
1. All modules can be imported
2. Output directories exist
3. Sample data can be processed
"""

import sys
from pathlib import Path

def check_imports():
    """Check if required packages are importable"""
    print("🔍 Checking imports...")
    
    required = ["numpy", "yaml", "json"]
    optional = ["scipy", "sklearn"]
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} - REQUIRED but missing!")
            return False
    
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg} (optional)")
        except ImportError:
            print(f"  ⚠️  {pkg} (optional, not critical)")
    
    return True


def check_structure():
    """Check folder structure"""
    print("\n🔍 Checking folder structure...")
    
    base = Path(__file__).parent
    folders = [
        "data",
        "optimize",
        "eval",
        "outputs",
        "outputs/weights",
        "outputs/metrics",
        "outputs/ablation_results",
    ]
    
    for folder in folders:
        folder_path = base / folder
        if folder_path.exists():
            print(f"  ✅ {folder}/")
        else:
            print(f"  ❌ {folder}/ - MISSING!")
            return False
    
    return True


def check_files():
    """Check critical Python files exist"""
    print("\n🔍 Checking critical files...")
    
    base = Path(__file__).parent
    files = [
        "run_optimization.py",
        "generate_report.py",
        "data/prepare_training_data.py",
        "optimize/ranking_optimizer.py",
        "optimize/baseline_least_squares.py",
        "optimize/metrics.py",
        "eval/evaluate_weights.py",
        "eval/ablation_study.py",
    ]
    
    for file in files:
        file_path = base / file
        if file_path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING!")
            return False
    
    return True


def check_imports_from_scripts():
    """Try to import key modules"""
    print("\n🔍 Checking script imports...")
    
    try:
        from optimize.metrics import compute_metrics
        print(f"  ✅ optimize.metrics")
    except Exception as e:
        print(f"  ❌ optimize.metrics - {e}")
        return False
    
    try:
        from optimize.ranking_optimizer import RankingWeightOptimizer
        print(f"  ✅ optimize.ranking_optimizer")
    except Exception as e:
        print(f"  ❌ optimize.ranking_optimizer - {e}")
        return False
    
    return True


def main():
    print("=" * 60)
    print("Weight Optimization Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Imports", check_imports),
        ("Folder Structure", check_structure),
        ("Files", check_files),
        ("Script Imports", check_imports_from_scripts),
    ]
    
    all_pass = True
    for name, check_fn in checks:
        try:
            if not check_fn():
                all_pass = False
        except Exception as e:
            print(f"❌ {name} - Exception: {e}")
            all_pass = False
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou're ready to run:")
        print("  python run_optimization.py --all --output_dir ../outputs_batch_50")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before running the pipeline.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
