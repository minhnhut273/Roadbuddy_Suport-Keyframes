# SETUP COMPLETE - READY TO RUN ✅

## 📍 Location

```
d:\DOWNLOAD\ZALO_CHALLENGE\new\roadbuddy_sf\weight_optimization\
```

## ⚡ Quick Start (Copy & Paste)

```powershell
# 1. Navigate to folder
cd d:\DOWNLOAD\ZALO_CHALLENGE\new\roadbuddy_sf\weight_optimization

# 2. Verify setup
python sanity_check.py

# 3. Run full pipeline
python run_optimization.py --all --output_dir ../outputs_batch_50 --baseline_config ../../configs/selector.yaml

# 4. View results
python generate_report.py
```

## 📂 Folder Structure

```
weight_optimization/
├── 📖 Docs (read these first)
│   ├── INSTALLATION.md  ← You are here
│   ├── QUICKSTART.md    ← Copy-paste commands
│   ├── README.md        ← Full documentation
│   ├── CONFIG.md        ← Tuning parameters
│   └── FILE_INDEX.md    ← Navigate everything
│
├── 🚀 Scripts (run these)
│   ├── run_optimization.py      ← Main entry point
│   ├── generate_report.py       ← View results
│   └── sanity_check.py          ← Verify setup
│
├── 🔧 Pipeline Code (modules)
│   ├── data/prepare_training_data.py
│   ├── optimize/ranking_optimizer.py
│   ├── optimize/baseline_least_squares.py
│   ├── optimize/metrics.py
│   ├── eval/evaluate_weights.py
│   └── eval/ablation_study.py
│
└── 📊 Outputs (auto-created)
    ├── outputs/weights/             ← Optimized weights go here
    ├── outputs/metrics/             ← Results & analysis
    └── outputs/ablation_results/    ← Feature importance
```

## 🎯 What This Does

**Optimizes frame selector weights** using support frame annotations as weak supervision.

### Input:
- `outputs_batch_50/train_*/result.json` (extracted features from batch run)
- `configs/selector.yaml` (baseline weights)

### Output:
- `outputs/weights/optimized_weights_ranking.yaml` (ready to use)
- `outputs/metrics/evaluation_results.json` (before/after comparison)
- `outputs/ablation_results/feature_importance.json` (which features matter)

### Process:
1. **Prepare**: Extract features from batch results → training dataset
2. **Optimize**: Learn weights using pairwise ranking loss
3. **Evaluate**: Test on held-out set, compare with baseline
4. **Ablate**: Analyze feature importance

## ✅ Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies: `pip install numpy pyyaml scipy`
- [ ] `outputs_batch_50/` folder exists with batch results
- [ ] `configs/selector.yaml` exists (baseline config)
- [ ] ~2 GB free disk space

## 🏃 Three Ways to Run

### 1️⃣ **Fastest** (Fully Automated)
```bash
python run_optimization.py --all --output_dir ../outputs_batch_50
```
- Runs all steps automatically
- Takes 5-10 minutes
- Outputs everything

### 2️⃣ **Step-by-Step** (For Learning)
```bash
python data/prepare_training_data.py --output_dir ../outputs_batch_50
python optimize/ranking_optimizer.py
python eval/evaluate_weights.py --weights_file outputs/weights/optimized_weights_ranking.yaml
python eval/ablation_study.py --weights_file outputs/weights/optimized_weights_ranking.yaml
```
- Full control over each step
- Can inspect intermediate results
- Good for understanding the process

### 3️⃣ **Compare Methods** (For Experimentation)
```bash
python run_optimization.py --method ranking
python run_optimization.py --method baseline
python eval/evaluate_weights.py --weights_file outputs/weights/optimized_weights_ranking.yaml
python eval/evaluate_weights.py --weights_file outputs/weights/optimized_weights_leastsquares.yaml
```
- Compares ranking vs least squares approaches
- Helps choose best method

## 📊 Example Output

After running:

```
outputs/
├── weights/
│   ├── optimized_weights_ranking.yaml
│   │   sharpness: 0.1234
│   │   edge_density: 0.2345
│   │   brightness: 0.0456
│   │   ...
│   └── training_log_ranking.json
│
├── metrics/
│   ├── training_data_summary.json
│   │   {
│   │     "total_samples": 5000,
│   │     "support_samples": 250,
│   │     "support_ratio": 0.05
│   │   }
│   ├── evaluation_results.json
│   │   {
│   │     "baseline_metrics": { "hit@1": 0.XX },
│   │     "optimized_metrics": { "hit@1": 0.YY },
│   │     "metric_deltas": { "hit@1": { "delta": +0.03, "pct_change": +3.5% } }
│   │   }
│   └── baseline_metrics.json
│
└── ablation_results/
    ├── feature_importance.json
    │   {
    │     "sharpness": { "importance": 0.15, "drop": 0.05 },
    │     "edge_density": { "importance": 0.22, "drop": 0.07 },
    │     ...
    │   }
    └── group_analysis.json
```

## 🔍 Quick Validation

Before running, check:
```bash
# Verify dependencies
python sanity_check.py

# Check input data exists
ls ../outputs_batch_50/train_0001/result.json

# Check baseline config
ls ../../configs/selector.yaml
```

## 📖 Documentation Roadmap

1. **Start here**: `INSTALLATION.md` (you are here)
2. **Quick commands**: `QUICKSTART.md` (2 min read)
3. **Run it**: `python run_optimization.py --all`
4. **View results**: `python generate_report.py`
5. **Understand details**: `README.md` (10 min read)
6. **Tune parameters**: `CONFIG.md`
7. **Deep dive**: Read individual script docstrings

## 🎓 Key Concepts

- **Weak Supervision**: Support frames indicate semantic relevance, not visual quality
- **Ranking Loss**: Ensure support frames score higher than non-support
- **Validation Split**: Train/val/test to prevent overfitting
- **Ablation Study**: Analyze which features contribute to performance
- **Baseline Comparison**: Compare optimized vs original weights

## 💡 Expected Outcomes

- ✅ **Success**: +1-5% improvement in hit@k metrics
- ✅ **Typical**: +1-2% improvement
- ✅ **Possible**: 0% improvement (indicates weak signal)
- ✅ **All valid**: Every outcome provides insights

## 🚨 If Something Goes Wrong

1. **Import error**: `pip install numpy pyyaml scipy`
2. **File not found**: Check you're in `weight_optimization/` folder
3. **Slow processing**: Reduce # samples or increase RAM
4. **No improvement**: Check `outputs/ablation_results/feature_importance.json`

## 🎯 Next Steps

1. **Verify setup**:
   ```bash
   python sanity_check.py
   ```

2. **Run pipeline**:
   ```bash
   python run_optimization.py --all --output_dir ../outputs_batch_50
   ```

3. **View results**:
   ```bash
   python generate_report.py
   ```

4. **Check outputs**:
   ```bash
   # View optimized weights
   cat outputs/weights/optimized_weights_ranking.yaml
   
   # View metrics
   cat outputs/metrics/evaluation_results.json | head -50
   
   # View feature importance
   cat outputs/ablation_results/feature_importance.json | head -50
   ```

5. **If improvements > 0%**: Copy weights to `configs/selector.yaml`

---

**Status**: ✅ Ready to run
**Estimated time**: 5-10 minutes
**Disk space needed**: ~2 GB
**Help**: Read `QUICKSTART.md` or `README.md`

**You're all set! Good luck! 🚀**
