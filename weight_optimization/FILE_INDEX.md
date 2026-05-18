# Weight Optimization Pipeline - File Index

```
weight_optimization/
│
├── 📖 README.md              ← START HERE (full documentation)
├── 🚀 QUICKSTART.md          ← Quick commands and examples
├── ⚙️  CONFIG.md              ← Configuration and tuning guide
├── 📋 FILE_INDEX.md          ← This file
│
├── run_optimization.py       ← MAIN ENTRY POINT
├── generate_report.py        ← View results (human-readable)
│
├── 📁 data/
│   ├── __init__.py
│   └── prepare_training_data.py    (Step 1: Data preparation)
│
├── 📁 optimize/
│   ├── __init__.py
│   ├── metrics.py                  (Ranking metrics: hit@k, MRR, AP, etc)
│   ├── ranking_optimizer.py        (Step 2a: Main optimizer - USE THIS)
│   └── baseline_least_squares.py   (Step 2b: Baseline for comparison)
│
├── 📁 eval/
│   ├── __init__.py
│   ├── evaluate_weights.py         (Step 3: Test set evaluation)
│   └── ablation_study.py           (Step 4: Feature importance)
│
└── 📁 outputs/
    ├── weights/                    ← MAIN OUTPUT (use these weights)
    │   ├── optimized_weights_ranking.yaml
    │   └── training_log_ranking.json
    ├── metrics/                    ← RESULTS & ANALYSIS
    │   ├── training_data_summary.json
    │   ├── evaluation_results.json
    │   └── baseline_metrics.json
    └── ablation_results/           ← FEATURE INSIGHTS
        ├── feature_importance.json
        └── group_analysis.json
```

## 📌 Key Files by Use Case

### For Quick Start
1. Read: `QUICKSTART.md`
2. Run: `python run_optimization.py --all --output_dir ../outputs_batch_50`
3. View: `python generate_report.py`

### For Understanding Approach
1. Read: `README.md` (technical details)
2. Read: `CONFIG.md` (parameters and tuning)
3. Review: `optimize/ranking_optimizer.py` (algorithm)

### For Analyzing Results
1. Check: `outputs/metrics/evaluation_results.json` (main metrics)
2. Check: `outputs/ablation_results/feature_importance.json` (which features matter)
3. Check: `outputs/weights/optimized_weights_ranking.yaml` (use-ready weights)

### For Reproducing Experiments
1. Use: `run_optimization.py` with specific arguments
2. Save outputs to timestamped folder
3. Compare: `outputs/metrics/evaluation_results.json` across runs

### For Debugging
1. Check data: `python data/prepare_training_data.py` alone
2. Check optimizer: `python optimize/ranking_optimizer.py` with `--verbose`
3. Check evaluation: `python eval/evaluate_weights.py` with specific weights file

## 🎯 Typical Workflow

### Scenario 1: First Time Run
```bash
cd weight_optimization

# Prepare data
python data/prepare_training_data.py --output_dir ../outputs_batch_50

# Optimize weights
python optimize/ranking_optimizer.py

# Evaluate
python eval/evaluate_weights.py --weights_file outputs/weights/optimized_weights_ranking.yaml

# View results
python generate_report.py
```

### Scenario 2: Quick Full Pipeline
```bash
python run_optimization.py --all --output_dir ../outputs_batch_50
python generate_report.py
```

### Scenario 3: Compare Multiple Methods
```bash
# Ranking method
python run_optimization.py --method ranking

# Least squares baseline
python run_optimization.py --method baseline

# Compare both
python eval/evaluate_weights.py --weights_file outputs/weights/optimized_weights_ranking.yaml
python eval/evaluate_weights.py --weights_file outputs/weights/optimized_weights_leastsquares.yaml
```

### Scenario 4: Deep Analysis
```bash
# Run full pipeline
python run_optimization.py --all

# Generate summary
python generate_report.py

# Deep dive into feature importance
python -c "import json; print(json.dumps(json.load(open('outputs/ablation_results/feature_importance.json')), indent=2))"
```

## 📊 Output Summary

| File | Format | Purpose | Size |
|------|--------|---------|------|
| `optimized_weights_ranking.yaml` | YAML | Use-ready weights | ~1 KB |
| `training_data.json` | JSON | Train/val/test split | ~10 MB |
| `evaluation_results.json` | JSON | Performance metrics | ~10 KB |
| `feature_importance.json` | JSON | Ablation results | ~5 KB |
| `training_log_ranking.json` | JSON | Loss curves | ~100 KB |

## ✅ Checklist for Running

- [ ] Have `outputs_batch_50/` folder with `train_*/result.json` files
- [ ] Have `configs/selector.yaml` with baseline weights
- [ ] Python environment set up with dependencies (numpy, yaml, etc)
- [ ] ~2 GB disk space for training data
- [ ] 5-10 minutes for full pipeline

## 🔗 Cross-References

- Algorithm details → `optimize/ranking_optimizer.py` + `README.md`
- Metrics explanation → `optimize/metrics.py` + `CONFIG.md`
- Data format → `data/prepare_training_data.py` + `TrainingItem` dataclass
- Examples → `QUICKSTART.md`

## 📝 Notes

- All outputs go to `outputs/` folder (relative to script location)
- Can run steps independently or full pipeline with `run_optimization.py`
- Results are JSON/YAML - easy to parse and integrate
- Reports are text-based - easy to copy to documentation

---

**Last Updated**: May 2026
**Pipeline Version**: 1.0
**Status**: Tested and ready for production
