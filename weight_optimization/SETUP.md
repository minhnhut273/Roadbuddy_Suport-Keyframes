# Setup & Installation Guide

## Prerequisites

- Python 3.8+
- Virtual environment or conda (recommended)
- ~2 GB free disk space

## Step 1: Install Dependencies

```bash
# From project root
pip install numpy pyyaml scipy scikit-learn

# Or if using conda
conda install numpy pyyaml scipy scikit-learn
```

## Step 2: Verify Setup

```bash
cd weight_optimization
python sanity_check.py
```

Expected output:
```
✅ ALL CHECKS PASSED!
You're ready to run:
  python run_optimization.py --all --output_dir ../outputs_batch_50
```

## Step 3: Prepare Input Data

Make sure you have:

```
project/
├── outputs_batch_50/
│   ├── train_0001/
│   │   └── result.json
│   ├── train_0002/
│   │   └── result.json
│   └── ... (up to train_0050)
│
├── configs/
│   └── selector.yaml        (with baseline weights)
│
└── weight_optimization/     (this folder)
```

If you don't have `outputs_batch_50/`, run the batch selector first:
```bash
cd ..
python run_eval_selector_all.py
# Wait for completion (~30 min)
```

## Step 4: Run Pipeline

```bash
cd weight_optimization

# Full pipeline
python run_optimization.py --all \
  --output_dir ../outputs_batch_50 \
  --baseline_config ../../configs/selector.yaml

# Monitor progress in console
```

Expected duration: **5-10 minutes** depending on number of samples

## Step 5: View Results

```bash
python generate_report.py
```

Or manually check files:
```bash
# View optimized weights
cat outputs/weights/optimized_weights_ranking.yaml

# View metrics
python -c "import json; print(json.load(open('outputs/metrics/evaluation_results.json')))" | head -50

# View feature importance
python -c "import json; print(json.load(open('outputs/ablation_results/feature_importance.json')))" | head -50
```

## Troubleshooting

### "No module named 'numpy'"
```bash
pip install numpy
```

### "No such file or directory: ../outputs_batch_50"
Make sure:
1. You're in the `weight_optimization/` directory
2. `outputs_batch_50/` exists one level up
3. Check path with `ls -la ../outputs_batch_50`

### "Permission denied" on Linux/Mac
```bash
chmod +x sanity_check.py
chmod +x run_optimization.py
```

### "Memory error" or "Killed"
The dataset might be too large. Try:
1. Reduce number of samples in `outputs_batch_50/`
2. Run on a machine with more RAM
3. Contact support

### Script runs but "No metrics improvement"
This is expected! Support frames are weak supervision. Review:
1. Ablation study results (`feature_importance.json`)
2. Data statistics (`training_data_summary.json`)
3. Whether features correlate with semantic labels

## Optional: Advanced Setup

### For GPU acceleration (optional)
```bash
# Only needed if you want to speed up large datasets
pip install torch  # or tensorflow
```

### For visualization (optional)
```bash
pip install matplotlib seaborn
```

### For integration testing
```bash
cd weight_optimization
python -m pytest test_*.py  # if tests exist
```

## Verifying Installation

Quick test without data:
```bash
cd weight_optimization

# Import check
python -c "from optimize.metrics import compute_metrics; print('✅ Imports OK')"

# Sanity check
python sanity_check.py

# Help text
python run_optimization.py --help
```

## Next Steps

After installation:
1. Read `QUICKSTART.md` for command examples
2. Run `python run_optimization.py --all`
3. Check results with `python generate_report.py`
4. Review `README.md` for detailed documentation

---

**If you get stuck, check:**
1. QUICKSTART.md - Common issues
2. CONFIG.md - Parameter tuning
3. Script docstrings - Implementation details
4. JSON output files - Data validation
