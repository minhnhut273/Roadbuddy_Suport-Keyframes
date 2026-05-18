# 🎬 RoadBuddy Support Frame Selection and Optimization

## 📖 Giới thiệu đầy đủ

Đây là một **hệ thống chọn keyframes thông minh** từ video hỗ trợ lái xe an toàn. Repo chứa đầy đủ code để:

1. **🎯 Chọn Support Keyframes** từ video dùng multiple visual features (sharpness, edge density, brightness, ROI-based features)
2. **📊 Đánh giá Chất lượng** của selection so với ground-truth support frames
3. **⚙️ Tối ưu hóa Trọng số** của các visual features dùng weak supervision từ support frames
4. **📈 Phân tích Chi tiết** kết quả theo loại câu hỏi (sign identification, rule compliance, object presence, v.v.)

### 🎯 Bài toán chính

Một video dashcam thường chứa nhiều frame, nhưng chỉ một số frame là "support frames" – những frame hữu ích từ góc độ **semantic** (trả lời câu hỏi liên quan đến video). Bài toán là:

- **Input**: Video, metadata (question, support frame indices)
- **Output**: Top-K keyframes được ranking bằng visual scores + semantic signals
- **Weak Supervision**: Chúng ta dùng support frame times để học trọng số của các visual features

### 🚀 Workflow chính gồm 3 bước

| Bước | Công cụ | Mục đích |
|------|---------|---------|
| 1️⃣ **Selection & Evaluation** | `run_eval_selector_all.py` | Chạy selector trên toàn bộ dataset, tạo output results |
| 2️⃣ **Analysis** | `src/eval/analyze_eval_by_type.py` | Phân tích chi tiết theo loại câu hỏi, tạo báo cáo & hình ảnh minh họa |
| 3️⃣ **Weight Optimization** | `weight_optimization/run_optimization.py` | Tối ưu trọng số visual features dùng support frames |

## 📁 Cấu trúc thư mục chi tiết

```
roadbuddy_sf/
│
├── 🔧 configs/
│   └── selector.yaml               ⭐ Cấu hình chính (weights, ROI, policies, tolerances)
│
├── 📦 src/
│   ├── eval/                       📊 Đánh giá & Phân tích
│   │   ├── eval_selector.py        (Tính hit@k, recall, mean_distance)
│   │   ├── support_frame_scorer.py (Scoring logic)
│   │   └── analyze_eval_by_type.py ⭐ Phân tích chi tiết theo câu hỏi loại
│   │
│   ├── selector/                   🎯 Core selector pipeline
│   │   ├── pipeline.py             (Chạy coarse → refine → rank)
│   │   ├── scorer.py               (Tính visual scores)
│   │   ├── policy_rerank.py        (Rerank policy)
│   │   └── peak_picker.py          (Chọn đỉnh NMS)
│   │
│   ├── data/                       📥 Tải dữ liệu
│   │   └── train_loader.py         (Load train.json, video paths)
│   │
│   ├── perception/                 👁️ Feature extraction
│   │   ├── detector.py             (Detect objects)
│   │   ├── ocr_reader.py           (Read text từ video)
│   │   └── blur.py                 (Detect blurred frames)
│   │
│   ├── utils/
│   │   └── io.py                   (Save/load JSON, create dirs)
│   │
│   ├── video/
│   │   └── sampler.py              (Sample frames từ video)
│   │
│   └── Scripts/                    🔨 Tiện ích không bắt buộc
│       ├── download_videos.py
│       └── extract_support_frames.py
│
├── ⚙️ weight_optimization/         Tối ưu trọng số visual features
│   ├── data/
│   │   └── prepare_training_data.py (Chuẩn bị X, y từ outputs)
│   │
│   ├── optimize/
│   │   ├── ranking_optimizer.py    (Pairwise ranking loss)
│   │   ├── baseline_least_squares.py (Baseline comparison)
│   │   └── metrics.py              (Hit@k, MRR, v.v.)
│   │
│   ├── eval/
│   │   ├── evaluate_weights.py     (Kiểm tra weights tối ưu)
│   │   └── ablation_study.py       (Phân tích feature importance)
│   │
│   ├── README.md                   (Hướng dẫn chi tiết optimization)
│   └── run_optimization.py         ⭐ Entry point chính
│
├── 📂 outputs_batch_last/          Output từ run_eval_selector_all.py
│   ├── train_0001/ ... train_9999/ (Kết quả từng sample)
│   ├── results.csv                 (Summary dạng CSV)
│   ├── summary.json                (Metrics tổng hợp)
│   └── summary_by_type_and_nearest_direction.json (Phân tích chi tiết)
│
├── 📄 run_eval_selector_all.py    ⭐ Entry point: Selection & Evaluation
├── run_selector.py                (Single video selector - không bắt buộc)
├── run_full_support_candidate_compare.py (Comparison script)
├── .gitignore
├── .venv/                          (Python virtual env)
└── README.md                       (Tài liệu này)
```

### 🎯 Các file QUAN TRỌNG NHẤT để tập trung

| File | Mục đích | Loại |
|------|---------|------|
| `configs/selector.yaml` | Cấu hình weights, ROI, policies | ⭐⭐⭐ Config |
| `run_eval_selector_all.py` | Chạy selector batch & tạo outputs | ⭐⭐⭐ Script chính |
| `src/eval/analyze_eval_by_type.py` | Phân tích kết quả chi tiết | ⭐⭐⭐ Analysis |
| `weight_optimization/run_optimization.py` | Tối ưu trọng số | ⭐⭐⭐ Script chính |
| `src/selector/pipeline.py` | Core logic selector | ⭐⭐ Core |
| `weight_optimization/data/prepare_training_data.py` | Chuẩn bị data cho optimization | ⭐⭐ Support |
| `weight_optimization/optimize/ranking_optimizer.py` | Logic học trọng số | ⭐⭐ Optimize |

## 🚀 Hướng dẫn chi tiết từng bước

### 🔌 Chuẩn bị môi trường

```powershell
# 1. Kích hoạt virtual environment
.venv\Scripts\Activate.ps1

# 2. Cài đặt dependencies (nếu chưa có)
pip install -r requirements.txt
# (hoặc tự cài: opencv-python, pyyaml, numpy, scipy, scikit-learn, v.v.)
```

---

### **Bước 1️⃣: Chạy Selector & Tạo Output** (`run_eval_selector_all.py`)

#### 🎯 Mục đích
Chạy selector trên toàn bộ dataset, tính toán scores cho mỗi frame và tạo output evaluation.

#### 📥 Input cần thiết
- `train_json`: File JSON chứa danh sách sample (id, video_path, support_frames, question, v.v.)
- `video_root`: Thư mục chứa các video
- `config` (optional): Đường dẫn đến `configs/selector.yaml`

#### 📊 Output sinh ra
```
outputs_batch/
├── train_0001/
│   ├── result.json          (Kết quả selection & evaluation)
│   └── frames/              (Top selected frames - nếu --save_frames)
├── train_0002/
│   ├── result.json
│   └── frames/
│   ...
├── summary.json             (Metrics trung bình)
├── results.csv              (CSV summary)
└── summary.json
```

#### 💻 Cách chạy

**Chạy đơn giản (100 mẫu):**
```powershell
python run_eval_selector_all.py \
  --train_json data/train.json \
  --video_root D:/videos \
  --output_dir outputs_batch \
  --limit 100
```

**Chạy toàn bộ với lưu frames:**
```powershell
python run_eval_selector_all.py \
  --train_json data/train.json \
  --video_root D:/videos \
  --config configs/selector.yaml \
  --output_dir outputs_batch \
  --limit 1450 \
  --start_index 0 \
  --save_frames
```

**Tiếp tục từ sample thứ 500:**
```powershell
python run_eval_selector_all.py \
  --train_json data/train.json \
  --video_root D:/videos \
  --output_dir outputs_batch_continue \
  --limit 500 \
  --start_index 500
```

#### 📊 Kết quả mong đợi
```
Mẫu được xử lý: 100/100
Saved CSV       : outputs_batch/results.csv
Saved summary   : outputs_batch/summary.json

Primary metrics:
  hit@k_tol_0_5: 0.75
  hit@k_tol_1_0: 0.85
  recall_tol_0_5: 0.70
  mean_min_distance: 0.35
```

---

### **Bước 2️⃣: Phân tích Chi tiết** (`src/eval/analyze_eval_by_type.py`)

#### 🎯 Mục đích
- Phân tích kết quả **theo loại câu hỏi** (sign_identification, rule_compliance, object_presence, v.v.)
- So sánh vị trí nearest candidate vs support frame
- Tạo hình ảnh so sánh support frame vs predicted candidate frame
- Sinh báo cáo JSON & CSV chi tiết

#### 📥 Input
- `output_dir`: Thư mục chứa `train_*/result.json` (từ bước 1)

#### 📊 Output sinh ra
```
outputs_batch/
├── summary_by_type_and_nearest_direction.csv
├── summary_by_type_and_nearest_direction.json  (Báo cáo chi tiết)
└── nearest_pairs_selected/
    ├── train_0001/
    │   ├── train_0001_support1_before_delta_-0.123.jpg
    │   └── train_0001_support2_after_delta_+0.456.jpg
    ...
```

#### 💻 Cách chạy

```powershell
# Phân tích từ outputs_batch
python src/eval/analyze_eval_by_type.py \
  --output_dir outputs_batch \
  --pairs_out_dir outputs_batch/nearest_pairs_selected \
  --candidate_source selected_frames

# Hoặc với all_frames
python src/eval/analyze_eval_by_type.py \
  --output_dir outputs_batch \
  --pairs_out_dir outputs_batch/nearest_pairs_all \
  --candidate_source all_frames
```

#### 📈 Kết quả mong đợi
```
DONE
Saved CSV   : outputs_batch/summary_by_type_and_nearest_direction.csv
Saved JSON  : outputs_batch/summary_by_type_and_nearest_direction.json
Saved pairs : outputs_batch/nearest_pairs_selected

GLOBAL nearest-candidate-vs-support:
  before: 150 (30.00%)
  after : 250 (50.00%)
  exact : 100 (20.00%)
  mean_nearest_support_distance   : 0.3456

BY TYPE:
- sign_identification
    samples=200, support_pairs=400
    hit@k@0.5=0.7500, hit@k@1.0=0.8500, mean_min_distance=0.3456
    nearest before=28.00%, after=52.00%, exact=20.00%
...
```

---

### **Bước 3️⃣: Tối ưu Trọng số** (`weight_optimization/run_optimization.py`)

#### 🎯 Mục đích
Học trọng số tối ưu cho các visual features (sharpness, edge_density, brightness, ROI features) dùng support frames làm signal.

#### 🔄 Sub-steps
1. **Prepare**: Chuẩn bị training data từ outputs batch
2. **Optimize**: Chọn method (ranking hoặc baseline)
3. **Evaluate**: Kiểm tra weights tối ưu trên validation set
4. **Ablate** (optional): Phân tích feature importance

#### 💻 Cách chạy

**Chạy toàn bộ pipeline (recommended):**
```powershell
cd weight_optimization
python run_optimization.py --all
```

**Chỉ các bước cụ thể:**
```powershell
# Chỉ chuẩn bị data
python run_optimization.py --prepare

# Tối ưu với ranking
python run_optimization.py --method ranking

# Tối ưu baseline (least squares)
python run_optimization.py --method baseline

# Cả hai methods
python run_optimization.py --method both

# Đánh giá weights tối ưu
python run_optimization.py --evaluate outputs/weights/optimized_weights_ranking.yaml

# Ablation study
python run_optimization.py --ablate outputs/weights/optimized_weights_ranking.yaml
```

#### 📂 Output sinh ra
```
weight_optimization/outputs/
├── weights/
│   ├── optimized_weights_ranking.yaml    (Weights tối ưu - ranking)
│   ├── optimized_weights_baseline.yaml   (Weights baseline)
│   └── training_log_ranking.json         (Loss curve)
│
├── metrics/
│   ├── training_data.json                (Features & labels)
│   ├── training_data_summary.json        (Data statistics)
│   └── evaluation_results.json           (Hit@k, MRR, v.v.)
│
└── ablation_results/
    ├── feature_importance.json
    └── sensitivity_analysis.json
```

#### 📊 Kết quả mong đợi
```
🚀 Weight Optimization Pipeline
   Output dir: outputs_batch_50

Step 1: Prepare Training Data
  ✅ Data preparation

Step 2a: Ranking-based Optimization
  ✅ Ranking optimization

Step 3: Evaluate Optimized Weights
  ✅ Evaluation

📊 PIPELINE SUMMARY
✅ Data preparation
✅ Ranking optimization
✅ Evaluation

📂 Output files:
   Weights:    weight_optimization/outputs/weights/
   Metrics:    weight_optimization/outputs/metrics/
   Ablation:   weight_optimization/outputs/ablation_results/

✅ Pipeline completed!
```

## ⚙️ Cấu hình Selector (`configs/selector.yaml`)

File này quyết định **cách thức chọn frame** và **cách tính score**.

### 🎯 Các config quan trọng

| Config | Mục đích | Giá trị mẫu |
|--------|---------|----------|
| **fps_sample** | FPS lấy mẫu ban đầu | 3.0 |
| **fps_coarse** | FPS ở bước coarse ranking | 2.0 |
| **fps_refine** | FPS ở bước refine | 6.0 |
| **max_frames_per_video** | Tối đa frames xử lý | 96 |
| **top_k** | Số lượng frame chọn cuối cùng | 10 |
| **temporal_nms_gap_sec** | Khoảng cách tối thiểu giữa frames | 0.5s |

### 🎯 Trọng số Visual Features

```yaml
weights:
  sharpness: 0.1195          # Độ sắc nét toàn ảnh
  edge_density: 0.0997       # Mật độ cạnh
  brightness: 0.0394        # Độ sáng
  novelty: 0.0000027        # Mức độ khác biệt với frame kế tiếp
  center_bias: 0.0298       # Ưu tiên frame ở giữa video
  roi_sharpness: 0.3407     # Độ sắc nét trong ROI (vùng quan trọng)
  roi_edge_density: 0.3007  # Mật độ cạnh trong ROI
  roi_brightness: 0.0702    # Độ sáng trong ROI
```

🎨 **ROI (Region of Interest)** là các vùng quan trọng được định nghĩa bằng tọa độ `[x_min, y_min, x_max, y_max]` (normalized 0-1).

### 🎯 Question Type Policies

Bạn có thể có **policy riêng** cho từng loại câu hỏi:

```yaml
type_to_policy:
  sign_identification: sign_policy
  information_reading: sign_policy
  navigation: sign_policy
  
  rule_compliance: lane_policy
  verification: lane_policy
  
  object_presence: coverage_policy
  counting: coverage_policy
```

Mỗi policy có config riêng (weights, ROI, top_k, v.v.) được định nghĩa trong `policy_configs`.

### 📋 Ví dụ policy cho sign_identification

```yaml
policy_configs:
  sign_policy:
    top_k: 10
    temporal_nms_gap_sec: 0.40
    roi_regions:
      - [0.30, 0.00, 0.72, 0.34]  # Top region
      - [0.52, 0.00, 1.00, 0.42]  # Right region
      - [0.22, 0.14, 0.82, 0.56]  # Center region
    weights:
      sharpness: 0.0628
      roi_sharpness: 0.3801
      roi_edge_density: 0.4000
      ...
```

---

## 💡 Quick Reference - Các lệnh thường dùng

### 🎬 Selection

```powershell
# Chạy selector trên 100 mẫu
python run_eval_selector_all.py --train_json data/train.json --video_root D:/videos --output_dir outputs_batch --limit 100

# Chạy selector trên toàn bộ dataset với lưu frames
python run_eval_selector_all.py --train_json data/train.json --video_root D:/videos --output_dir outputs_batch --limit 1450 --save_frames
```

### 📊 Analysis

```powershell
# Phân tích từ outputs
python src/eval/analyze_eval_by_type.py --output_dir outputs_batch --pairs_out_dir outputs_batch/nearest_pairs_selected
```

### ⚙️ Optimization

```powershell
# Chạy toàn bộ optimization pipeline
cd weight_optimization
python run_optimization.py --all

# Hoặc từng bước
python run_optimization.py --prepare
python run_optimization.py --method ranking
python run_optimization.py --evaluate outputs/weights/optimized_weights_ranking.yaml
```

---

## 🧪 Test / Demo nhanh

Nếu bạn chỉ muốn kiểm tra xem code có chạy không:

```powershell
# Chạy selector trên 10 mẫu
python run_eval_selector_all.py \
  --train_json data/train.json \
  --video_root D:/videos \
  --output_dir test_outputs \
  --limit 10

# Phân tích kết quả
python src/eval/analyze_eval_by_type.py \
  --output_dir test_outputs \
  --pairs_out_dir test_outputs/pairs
```

---

## 📊 Hiểu về Output

### `result.json` (từ bước 1)

Mỗi sample sinh ra file `train_XXXX/result.json`:

```json
{
  "sample": {
    "id": "train_0001",
    "question": "Is there a stop sign?",
    "type": "sign_identification",
    "support_frames": [2.5, 5.3, 8.1],
    "video_path": "videos/train_0001.mp4"
  },
  "selector_result": {
    "selected_frames": [
      {"time_sec": 2.48, "score": 0.92},
      {"time_sec": 5.31, "score": 0.89},
      {"time_sec": 8.09, "score": 0.87}
    ],
    "all_frames": [...]
  },
  "evaluation": {
    "hit@k_tol_0_5": 1,
    "hit@k_tol_1_0": 1,
    "recall_tol_0_5": 1.0,
    "mean_min_distance": 0.05
  }
}
```

### `summary_by_type_and_nearest_direction.json` (từ bước 2)

Tổng hợp theo loại câu hỏi:

```json
{
  "global_summary": {
    "num_samples": 500,
    "metrics_avg": {
      "hit@k_tol_0_5": 0.75,
      "hit@k_tol_1_0": 0.85
    },
    "nearest_candidate_position_vs_support": {
      "before_count": 150,
      "before_pct": 0.30,
      "after_count": 250,
      "after_pct": 0.50,
      "exact_count": 100,
      "exact_pct": 0.20
    }
  },
  "summary_by_type": {
    "sign_identification": {...},
    "rule_compliance": {...},
    "object_presence": {...}
  }
}
```

---

## 🔑 Các Metric chính

| Metric | Ý nghĩa |
|--------|---------|
| **hit@k_tol_X** | % support frames nằm trong top-k selected frames (tolerance X giây) |
| **recall_tol_X** | Proportion of support frames được cover |
| **mean_min_distance** | Khoảng cách trung bình nhỏ nhất từ candidate đến support frame |
| **MRR** | Mean Reciprocal Rank - rank trung bình của support frame |

---

## 📝 Các file hỗ trợ (optional)

## 📝 Các file hỗ trợ (optional)

Nếu bạn chỉ tập trung vào 3 bước chính, bạn có thể **không cần** quan tâm đến:

- `demo_support_frame_scorer.py`
- `demo_support_frame_scoring_type_aware.py`
- `run_selector.py` (chạy single video)
- `run_full_support_candidate_compare.py`
- `test_support_frame_scorer.py` / `test_support_frame_scorer_type_aware.py`
- `weight_optimization/sanity_check.py`
- `weight_optimization/generate_weight_opt_commands.py`
- `weight_optimization/generate_report.py`
- `src/Scripts/` (download & extract utilities)

---

## 🆘 Troubleshooting

### ❌ Lỗi: "Video file not found"
- Kiểm tra đường dẫn `video_root` có tồn tại không
- Kiểm tra `train_json` có path video đúng không

### ❌ Lỗi: "CUDA out of memory" / "Memory error"
- Giảm `max_frames_per_video` trong `selector.yaml`
- Giảm `--limit` khi chạy `run_eval_selector_all.py`
- Giảm `fps_sample`, `fps_coarse`, `fps_refine` để sample ít frames hơn

### ❌ Optimization không cải thiện
- Weak supervision từ support frames có thể không correlate với visual features
- Thử điều chỉnh learning rate, epochs
- Kiểm tra dữ liệu training có đủ không

### ❌ Script báo lỗi module không tìm thấy
```powershell
# Đảm bảo bạn trong thư mục gốc project
cd D:\DOWNLOAD\ZALO_CHALLENGE\new\roadbuddy_sf

# Cài lại dependencies
pip install opencv-python pyyaml numpy scipy scikit-learn
```

---

## 📞 Ghi chú quan trọng

### ✅ Những điểm cần lưu ý

1. **Support frames là weak supervision**: Chúng là semantic signal (frame hữu ích), không nhất thiết optimal visual quality
2. **Multiple ROI regions**: Các region khác nhau cho sign, lane, objects
3. **Type-aware policies**: Policy khác nhau cho loại câu hỏi khác nhau
4. **Validation leak**: Weight optimization có chia train/val/test để tránh overfit
5. **Optimization realistic**: Cải thiện có thể nhỏ (1-5%) vì dữ liệu weak

### 📊 Workflow tiêu chuẩn

```
1. Chuẩn bị data (train.json + video_root)
   ↓
2. Chạy: python run_eval_selector_all.py
   ↓
3. Phân tích: python src/eval/analyze_eval_by_type.py
   ↓
4. Tối ưu: cd weight_optimization && python run_optimization.py --all
   ↓
5. Kiểm tra kết quả trong outputs/weights/, outputs/metrics/
```

---

## 📚 Tài liệu thêm

- `weight_optimization/README.md` - Hướng dẫn chi tiết optimization
- `configs/selector.yaml` - Các parameter có sẵn
- Output JSON files - Chứa kết quả chi tiết mỗi sample

---

## 🎉 Tóm tắt nhanh

| Cần làm gì? | Chạy lệnh nào? |
|------------|---------------|
| Chạy selector trên dataset | `python run_eval_selector_all.py --train_json ... --video_root ...` |
| Phân tích kết quả chi tiết | `python src/eval/analyze_eval_by_type.py --output_dir outputs_batch` |
| Tối ưu trọng số | `cd weight_optimization && python run_optimization.py --all` |
| Xem metrics | Mở `outputs_batch/summary.json` hoặc CSV |
| Xem support vs predicted frames | Mở ảnh trong `nearest_pairs_selected/` |
| Cập nhật trọng số selector | Sửa `weights` trong `configs/selector.yaml` |

---

**Happy coding! 🚀**
