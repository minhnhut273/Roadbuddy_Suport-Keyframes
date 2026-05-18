# Quick Start Guide

## Setup

```bash
cd weight_optimization
```

## Full Pipeline (Recommended)

```bash
python run_optimization.py --all \
  --output_dir ../outputs_batch_50 \
  --baseline_config ../../configs/selector.yaml
```

This will:
1. ✅ Prepare training data from batch results
2. ✅ Optimize weights (ranking-based)
3. ✅ Evaluate on test set
4. ✅ Run ablation study
5. ✅ Generate all output files

**Time**: ~5-10 minutes

## Step-by-Step

### 1. Prepare Data
```bash
python data/prepare_training_data.py \
  --output_dir ../outputs_batch_50 \
  --output_file outputs/metrics/training_data.json
```

**Output**: `outputs/metrics/training_data.json`

### 2. Optimize with Ranking Loss
```bash
python run_optimization.py --method ranking
```

**Output**: `outputs/weights/optimized_weights_ranking.yaml`

### 3. Compare with Baseline (Optional)
```bash
python run_optimization.py --method baseline
```

**Output**: `outputs/weights/optimized_weights_leastsquares.yaml`

### 4. Evaluate Results
```bash
python run_optimization.py --evaluate outputs/weights/optimized_weights_ranking.yaml
```

**Output**: `outputs/metrics/evaluation_results.json`

### 5. Ablation Study
```bash
python run_optimization.py --ablate outputs/weights/optimized_weights_ranking.yaml
```

**Output**: `outputs/ablation_results/`

## Quick Checks

### View training data summary
```bash
python -c "import json; d = json.load(open('outputs/metrics/training_data_summary.json')); print('Total:', d['total_statistics']['total_samples'], '| Support:', d['total_statistics']['support_samples'])"
```

### View optimized weights
```bash
python -c "import yaml; w = yaml.safe_load(open('outputs/weights/optimized_weights_ranking.yaml')); print('\n'.join([f'{k}: {v:.4f}' for k,v in sorted(w.items())]))"
```

### View evaluation results
```bash
python -c "import json; r = json.load(open('outputs/metrics/evaluation_results.json')); print('Baseline Hit@1:', r['baseline_metrics']['hit@1']); print('Optimized Hit@1:', r['optimized_metrics']['hit@1'])"
```

### View feature importance
```bash
python -c "import json; a = json.load(open('outputs/ablation_results/feature_importance.json')); print('\n'.join([f'{k}: importance={v[\"importance\"]:.4f}' for k,v in sorted(a.items(), key=lambda x: x[1]['importance'], reverse=True)]))"
```

## Troubleshooting

### Error: "No such file or directory: ../outputs_batch_50"
Make sure you're running from inside `weight_optimization/` directory:
```bash
cd weight_optimization
python run_optimization.py --all ...
```

### Error: "Cannot extract support frame"
This is a warning, not an error. Some support frame timestamps might be outside video duration.

### No improvement in metrics
This is expected with weak supervision. Support frames are semantic, not visual quality labels. Try:
- Check ablation study to see which features matter
- Verify data is loaded correctly
- Try different learning rates or margin values

## Output Files

```
weight_optimization/
├── outputs/
│   ├── weights/
│   │   ├── optimized_weights_ranking.yaml        ← Use this
│   │   ├── optimized_weights_leastsquares.yaml   (optional)
│   │   └── training_log_ranking.json
│   ├── metrics/
│   │   ├── training_data_summary.json            (data stats)
│   │   ├── evaluation_results.json               (comparison)
│   │   └── baseline_metrics.json
│   └── ablation_results/
│       ├── feature_importance.json               (key insights)
│       └── group_analysis.json
```

## How to Use Optimized Weights

Copy optimized weights back to main config:

```yaml
# selector.yaml
weights:
  sharpness: 0.XXXX      # from optimized_weights_ranking.yaml
  edge_density: 0.XXXX
  brightness: 0.XXXX
  ...
```

Then test with your pipeline to see if performance improves.

## Expected Outcomes

- **Improvement**: +1-5% on ranking metrics (realistic)
- **No change**: Possible if support frames don't correlate with features
- **Decrease**: Also possible - indicates weak supervision challenge

All outcomes are valid and provide insights into whether visual features match semantic labels.

## Key Files to Review

1. `README.md` - Full documentation
2. `outputs/metrics/evaluation_results.json` - Main results
3. `outputs/ablation_results/feature_importance.json` - Which features matter
4. `outputs/weights/optimized_weights_ranking.yaml` - Use-ready weights

Good luck! 🎯
