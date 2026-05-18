# ✅ Weight Optimization Pipeline - Setup Complete!

## 📦 What Was Created

A complete, production-ready pipeline for optimizing frame selector weights using support frame annotations as weak supervision.

```
weight_optimization/
├── 📖 Documentation (Read These First)
│   ├── README.md              (comprehensive guide - 8 min read)
│   ├── QUICKSTART.md          (copy-paste commands - 2 min read)
│   ├── SETUP.md               (installation guide)
│   ├── CONFIG.md              (tuning parameters)
│   ├── FILE_INDEX.md          (navigate all files)
│   └── INSTALLATION.md        (this file)
│
├── 🚀 Main Scripts
│   ├── run_optimization.py    ← RUN THIS (orchestrates everything)
│   ├── generate_report.py     (view results in human-readable format)
│   └── sanity_check.py        (verify setup works)
│
├── 🔧 Core Pipeline
│   ├── data/
│   │   └── prepare_training_data.py     (extract features from batch results)
│   ├── optimize/
│   │   ├── metrics.py                   (ranking metrics: hit@k, MRR, AP, etc)
│   │   ├── ranking_optimizer.py         (main method using pairwise ranking loss)
│   │   └── baseline_least_squares.py    (simple regression baseline)
│   └── eval/
│       ├── evaluate_weights.py          (test set evaluation & comparison)
│       └── ablation_study.py            (feature importance analysis)
│
└── 📊 Output Directories (Auto-Created)
    ├── outputs/weights/                 (use-ready optimized weights)
    ├── outputs/metrics/                 (evaluation results & analysis)
    └── outputs/ablation_results/        (feature importance insights)
```

## 🎯 Three Ways to Get Started

### ⚡ Fastest (5 minutes)
```bash
cd weight_optimization
python sanity_check.py              # Verify setup
python run_optimization.py --all --output_dir ../outputs_batch_50
python generate_report.py           # View results
```

### 📚 Recommended (10 minutes)
1. Read: `QUICKSTART.md`
2. Read: First section of `README.md`
3. Run: `python run_optimization.py --all ...`
4. View: `python generate_report.py`
5. Analyze: `outputs/metrics/evaluation_results.json`

### 🔬 Deep Dive (30 minutes)
1. Read: `README.md` (full documentation)
2. Read: `CONFIG.md` (tuning guide)
3. Run pipeline step-by-step (see below)
4. Analyze ablation study results
5. Experiment with parameters

## 🏃 Step-by-Step Execution

### Option 1: Full Pipeline (Recommended)
```bash
cd weight_optimization
python run_optimization.py --all \
  --output_dir ../outputs_batch_50 \
  --baseline_config ../../configs/selector.yaml
```

### Option 2: Manual Steps (For Learning/Debugging)

#### Step 1: Prepare Data
```bash
python data/prepare_training_data.py \
  --output_dir ../outputs_batch_50 \
  --output_file outputs/metrics/training_data.json
```
**Output**: `training_data.json` (~10-50 MB)

#### Step 2: Optimize Weights
```bash
python optimize/ranking_optimizer.py \
  --data_file outputs/metrics/training_data.json \
  --baseline_config ../../configs/selector.yaml \
  --epochs 100
```
**Output**: `optimized_weights_ranking.yaml`

#### Step 3: Evaluate Results
```bash
python eval/evaluate_weights.py \
  --weights_file outputs/weights/optimized_weights_ranking.yaml \
  --data_file outputs/metrics/training_data.json \
  --output_file outputs/metrics/evaluation_results.json
```
**Output**: Metrics comparison + analysis

#### Step 4: Feature Importance (Optional)
```bash
python eval/ablation_study.py \
  --weights_file outputs/weights/optimized_weights_ranking.yaml \
  --data_file outputs/metrics/training_data.json \
  --output_dir outputs/ablation_results/
```
**Output**: Feature importance analysis

## 📊 Key Output Files

After running, check these files:

1. **Optimized Weights** (Use-ready)
   - File: `outputs/weights/optimized_weights_ranking.yaml`
   - Format: Simple YAML, copy to `configs/selector.yaml`
   - Example:
     ```yaml
     sharpness: 0.1234
     edge_density: 0.2345
     brightness: 0.0456
     ```

2. **Main Results** (Performance metrics)
   - File: `outputs/metrics/evaluation_results.json`
   - Contains: baseline vs optimized metrics, weight changes
   - Key metrics: hit@1, hit@3, hit@5, pairwise_accuracy

3. **Feature Importance** (Which features matter)
   - File: `outputs/ablation_results/feature_importance.json`
   - Shows: Impact of each feature on ranking quality

4. **Data Summary** (Dataset statistics)
   - File: `outputs/metrics/training_data_summary.json`
   - Shows: Number of samples, support frame ratio, distribution

## ✨ Feature Highlights

✅ **Ranking-Based Approach** - Treats it as ranking problem, not regression
✅ **Weak Supervision** - Works with semantic labels, not pixel-level annotations
✅ **Per-Question-Type Support** - Can optimize different weights for different question types
✅ **Validation Split** - Includes train/val/test split to prevent overfitting
✅ **Comprehensive Metrics** - Hit@k, MRR, pairwise accuracy, etc.
✅ **Feature Ablation** - Analyze which features matter
✅ **Comparison vs Baseline** - Automatic before/after comparison
✅ **Human-Readable Reports** - Easy to share and understand results

## 🔍 Quick Validation Checks

Before running, verify:

```bash
cd weight_optimization

# 1. Check imports work
python sanity_check.py

# 2. Check baseline config exists
ls -la ../../configs/selector.yaml

# 3. Check batch results exist  
ls -la ../outputs_batch_50/train_0001/result.json

# 4. Check disk space
df -h
```

## 💾 Expected Resource Usage

- **Data size**: 10-50 MB (depends on # samples)
- **Processing time**: 5-10 minutes
- **RAM**: 2-4 GB
- **Disk**: 1-2 GB (for outputs)

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "No module named 'numpy'" | `pip install numpy pyyaml scipy` |
| "outputs_batch_50 not found" | Make sure you're in correct directory |
| "No improvement in metrics" | ✅ Normal! Check ablation study for insights |
| "Script crashes on large data" | Reduce sample size or increase RAM |
| "Can't find evaluate.py" | Make sure you're in `weight_optimization/` folder |

## 🎓 What You'll Learn

By running this pipeline, you'll understand:

1. **How to optimize heuristic weights** using weak supervision
2. **Ranking losses** vs regression losses
3. **Feature importance analysis** via ablation
4. **Train/val/test splits** to prevent overfitting
5. **Comparing multiple optimization approaches** (ranking vs least squares)

## 📈 Expected Outcomes

- ✅ Optimized weights file (ready to use)
- ✅ Performance metrics (before/after comparison)
- ✅ Feature importance ranking
- ✅ Detailed analysis report

### Realistic Results
- **Best case**: +3-5% improvement in hit@k metrics
- **Typical case**: +1-2% improvement
- **Possible case**: No improvement (indicates weak signal)
- **All outcomes are valid** and provide insights!

## 🎯 Next Steps

1. **Run the pipeline**: `python run_optimization.py --all`
2. **Review results**: `python generate_report.py`
3. **Check file sizes**: `ls -lah outputs/`
4. **If improvement >0%**: Copy weights to `configs/selector.yaml`
5. **If improvement ≤0%**: Review ablation study, check data quality

## 📞 Need Help?

Refer to:
1. `QUICKSTART.md` - Quick copy-paste commands
2. `README.md` - Full technical documentation
3. `CONFIG.md` - Parameter tuning guide
4. `FILE_INDEX.md` - Navigate all components
5. Script docstrings - Implementation details

---

**You're all set! 🚀**

```bash
cd weight_optimization
python run_optimization.py --all --output_dir ../outputs_batch_50
python generate_report.py
```

Good luck! 🎯
