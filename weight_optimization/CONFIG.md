# Configuration Template for Weight Optimization

## Overview

This folder contains an end-to-end pipeline for optimizing frame selector weights
using support frame annotations as weak supervision.

### Pipeline Components

- **data/prepare_training_data.py** - Extract features from batch results
- **optimize/ranking_optimizer.py** - Ranking-based weight optimization (RECOMMENDED)
- **optimize/baseline_least_squares.py** - Regression baseline
- **optimize/metrics.py** - Ranking metrics computation
- **eval/evaluate_weights.py** - Evaluate weights on test set
- **eval/ablation_study.py** - Feature importance analysis
- **run_optimization.py** - Main orchestration script
- **generate_report.py** - Generate summary report

### Quick Start

```bash
# From weight_optimization/ directory

# Full pipeline
python run_optimization.py --all \
  --output_dir ../outputs_batch_50 \
  --baseline_config ../../configs/selector.yaml

# Then view results
python generate_report.py --output_dir ./outputs
```

### Configuration

#### input_dir
Path to batch results (e.g., `outputs_batch_50/`). Should contain:
- `train_0001/result.json`
- `train_0002/result.json`
- etc.

#### baseline_config
Path to baseline selector config (e.g., `configs/selector.yaml`). Used for:
- Initial weight values (L2 regularization anchors)
- Feature names
- Policy selection

#### output_dir
Where to save results:
- `outputs/weights/` - Optimized weights
- `outputs/metrics/` - Evaluation results
- `outputs/ablation_results/` - Feature importance

### Key Parameters

#### Ranking Optimizer
- `learning_rate` (default 0.01) - Gradient descent step size
  - Increase if convergence is slow
  - Decrease if loss is unstable
- `margin` (default 0.1) - Pairwise ranking margin
  - How much higher should support frames score than non-support
  - Increase to enforce stricter ranking
- `l2_lambda` (default 0.01) - Regularization strength
  - Stay close to baseline weights
  - Increase to prevent large weight changes

#### Least Squares Baseline
- `l2_lambda` (default 0.1) - Regularization strength

### Metrics Explained

- **hit@k** - Fraction of support frames in top-k predictions (higher is better)
- **mrr** - Mean Reciprocal Rank (higher is better)
- **ap** - Average Precision (higher is better)
- **pairwise_accuracy** - Fraction of (support, non-support) pairs correctly ranked (higher is better)
- **mean_min_distance** - Avg distance of support frame from nearest non-support (lower is better)

### Output Files

#### Weights
```yaml
# optimized_weights_ranking.yaml
sharpness: 0.1234
edge_density: 0.2345
brightness: 0.0456
novelty: 0.0000
center_bias: 0.0234
```

#### Metrics (evaluation_results.json)
```json
{
  "baseline_metrics": { "hit@1": 0.XX, "mrr": 0.YY, ... },
  "optimized_metrics": { "hit@1": 0.XX, "mrr": 0.YY, ... },
  "metric_deltas": { "hit@1": { "delta": +0.03, "pct_change": +3.5 }, ... }
}
```

#### Ablation (feature_importance.json)
```json
{
  "sharpness": {
    "importance": 0.1234,
    "drop_in_accuracy": 0.0456,
    ...
  },
  ...
}
```

### Interpreting Results

#### Good signs:
- ✅ hit@1, hit@3 improved by +2-5%
- ✅ pairwise_accuracy improved
- ✅ One or two features show high importance

#### Cautionary signs:
- ⚠️ Minimal improvement (<1%)
- ⚠️ Large variance in metrics
- ⚠️ All features equally important (possible overfit)

#### Bad signs:
- ❌ Metrics decreased significantly
- ❌ Weights diverged far from baseline (check margin/learning_rate)
- ❌ Zero importance for all features (data issue?)

### Troubleshooting

**Q: "Validation metrics don't improve"**
A: This is expected with weak supervision. Support frames indicate semantic usefulness,
   not visual quality. Verify ablation study shows non-zero importance for some features.

**Q: "Weights diverged from baseline"**
A: Increase `l2_lambda` to enforce regularization. Or decrease `learning_rate`.

**Q: "Memory error"**
A: Reduce number of samples or batch size. Check data file size.

**Q: "Can't find output files"**
A: Make sure you're running from `weight_optimization/` directory. Check output_dir path.

### Next Steps

1. Review `outputs/metrics/evaluation_results.json` for overall performance
2. Review `outputs/ablation_results/feature_importance.json` for feature insights
3. If improvement >0%, copy weights to `configs/selector.yaml` and test
4. If improvement ≤0%, investigate:
   - Are support frames semantically aligned with visual quality?
   - Do ROI features matter?
   - Consider collecting more data or refining supervision

### References

- Ranking losses: Learning to Rank literature
- Weak supervision: Snorkel, Data Programming
- Weight optimization: Hyperparameter tuning

---

**Note**: This is a research tool for understanding weight-feature relationships.
Results depend heavily on quality of support frame annotations.
