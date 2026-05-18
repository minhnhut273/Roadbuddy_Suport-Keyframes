# Weight Optimization from Support Frames

## Mục tiêu

Tối ưu hóa các trọng số (`weights`) trong `selector.yaml` bằng cách học từ **support frames** (weak supervision).

## Quan điểm kỹ thuật

### Bản chất của bài toán:
- **Weak supervision**: Support frames là tín hiệu semantic (frame hữu ích từ góc độ ngữ nghĩa), không phải gold label cho visual quality
- **Ranking problem**: Mục tiêu không phải fit support frames ≈ 1.0, mà là đảm bảo support frames xếp cao trong ranking

### Rủi ro:
- Overfit vào annotation style: Support frame không nhất thiết có sharpness cao nhất
- Validation leak nếu không split train/val
- Feature không correlate với semantic signal

## Cấu trúc

```
weight_optimization/
├── README.md                           (đây)
├── data/
│   └── prepare_training_data.py        (lấy features từ output_batch_50)
├── optimize/
│   ├── ranking_optimizer.py            (pairwise ranking loss)
│   ├── baseline_least_squares.py       (baseline so sánh)
│   └── metrics.py                      (hit@k, mean_min_distance, v.v.)
├── eval/
│   ├── evaluate_weights.py             (test weights trên validation set)
│   └── ablation_study.py               (test per-feature importance)
├── outputs/
│   ├── weights/                        (optimized weights)
│   ├── metrics/                        (performance metrics)
│   └── ablation_results/               (ablation study results)
└── run_optimization.py                 (entry point)
```

## Pipeline

### 1. Prepare Training Data
```bash
python data/prepare_training_data.py \
  --output_dir ../outputs_batch_50 \
  --train_json <path_to_train.json> \
  --output metrics/training_data_summary.json
```

**Output**:
- `X_train`: (N, 8) - normalized features [sharpness, edge_density, brightness, roi_sharpness, roi_edge_density, roi_brightness, ...]
- `y_support`: binary labels (1 = support frame, 0 = others)
- Split: 60% train, 20% val, 20% test

### 2. Optimize Weights
```bash
python run_optimization.py \
  --method ranking \
  --epochs 100 \
  --learning_rate 0.01 \
  --output_dir ./outputs/weights
```

**Output**:
- `optimized_weights_{question_type}.yaml` - weights per question type
- `training_log_{method}.json` - loss curve, metrics during training

### 3. Evaluate on Validation Set
```bash
python eval/evaluate_weights.py \
  --weights outputs/weights/optimized_weights_all.yaml \
  --validation_data metrics/training_data_summary.json \
  --output metrics/evaluation_results.json
```

**Output**:
- Hit@1, Hit@3, Hit@5 (% support frames trong top-k)
- Mean Reciprocal Rank (MRR)
- Comparison vs baseline weights

### 4. Ablation Study (optional)
```bash
python eval/ablation_study.py \
  --weights outputs/weights/optimized_weights_all.yaml \
  --validation_data metrics/training_data_summary.json \
  --output ablation_results/
```

**Output**:
- Per-feature importance: impact khi drop từng weight group
- Sensitivity analysis

## Expected Outcomes

- ✅ Optimized weights file (YAML format)
- ✅ Metrics report showing Hit@k improvement
- ✅ Ablation report showing feature importance
- ✅ Comparison graph: baseline vs optimized

### Realistic Expectations
- Improvement có thể nhỏ (1-5%) do weak supervision
- Có thể không cải thiện nếu features không correlate với semantic signal
- Mục đích chính: verify hypothesis + identify important features

## Cách chạy

### Automatic (recommended)
```bash
python run_optimization.py --all
```

### Step-by-step
```bash
python data/prepare_training_data.py
python run_optimization.py --method ranking
python eval/evaluate_weights.py
python eval/ablation_study.py
```

## Output Files

```
outputs/
├── weights/
│   ├── optimized_weights_all.yaml
│   ├── optimized_weights_rule_compliance.yaml
│   ├── optimized_weights_sign_identification.yaml
│   └── training_log_ranking.json
├── metrics/
│   ├── training_data_summary.json          (data statistics)
│   ├── evaluation_results.json              (hit@k, MRR, etc)
│   └── comparison_vs_baseline.json          (delta metrics)
└── ablation_results/
    ├── feature_importance.json
    └── sensitivity_analysis.json
```

## Quick Test

```bash
# Xem tóm tắt data
python -c "import json; print(json.load(open('outputs/metrics/training_data_summary.json')))"

# Xem optimized weights
python -c "import yaml; print(yaml.safe_load(open('outputs/weights/optimized_weights_all.yaml')))"

# Xem metrics
python -c "import json; print(json.load(open('outputs/metrics/evaluation_results.json')))"
```

## Notes

- Tất cả metrics có 95% confidence intervals (nếu có)
- Per-question-type analysis để tránh cộng gộp tín hiệu
- Validation set không được dùng trong optimization (avoid leak)
