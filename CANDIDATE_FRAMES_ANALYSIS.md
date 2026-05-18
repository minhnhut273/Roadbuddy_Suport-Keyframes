# Phân Tích Cách Lấy Candidate Frames

## 📋 Tóm Tắt Quy Trình

Repository này sử dụng **2-stage pipeline** để lấy candidate frames (khung hình chính) từ video:
1. **Coarse Stage**: Lấy các peak candidate từ toàn bộ video
2. **Fine/Refine Stage**: Lấy chi tiết từ các cửa sổ thời gian xung quanh các peak

---

## 🎬 Quy Trình Chi Tiết

### **STAGE 1: COARSE STAGE (Quét Tổng Quát)**

**Entry Point**: `KeyframeSelectorPipeline.run()` → `src/selector/pipeline.py`

```
1. Lấy mẫu video ở fps_coarse (thường 2.0 fps)
   ↓
2. Cắt bỏ 10% đầu và cuối video (trim_ratio)
   ↓
3. Tính điểm cho mỗi frame dựa trên các feature
   ↓
4. Lọc top-2 peak dùng Temporal NMS (Non-Maximum Suppression)
```

#### **Chi Tiết Step 1: Sampling Video**
```python
coarse_frames = sample_video_uniform(
    video_path=video_path,
    fps_sample=fps_coarse,      # Default: 2.0 fps
    max_frames=max_frames        # Default: 96 frames
)
```
- **Mục đích**: Giảm chi phí tính toán bằng cách chỉ lấy mẫu video ở tần suất thấp
- **Kết quả**: Danh sách frame được lấy mẫu đều từ video

#### **Chi Tiết Step 2: Trimming**
```python
coarse_frames = self._trim_frames(coarse_frames, trim_ratio=0.1)
```
- **Mục đích**: Loại bỏ phần đầu và cuối video (thường chứa credits, black frames)
- **Kết quả**: Frames từ 10% đến 90% của video

#### **Chi Tiết Step 3: Frame Scoring**

Mỗi frame được ghi điểm dựa trên **8 thành phần**:

| Thành Phần | Hàm | Ý Nghĩa |
|-----------|------|--------|
| **sharpness** | `laplacian_variance()` | Độ sắc nét - tính variance của Laplacian filter |
| **edge_density** | `canny_edge_density()` | Mật độ cạnh - % pixel được Canny detector phát hiện |
| **brightness** | `brightness_score()` | Độ sáng - giá trị trung bình pixel (0-1) |
| **novelty** | `novelty_score()` | Tính mới - khác biệt frame hiện tại vs frame trước |
| **center_bias** | `compute_center_bias()` | Sở thích ở giữa video - hàm U(x) = 1 - \|x - 0.5\| * 2 |
| **roi_sharpness** | `roi_feature_max()` | Độ sắc nét tối đa trong ROIs |
| **roi_edge_density** | `roi_feature_max()` | Mật độ cạnh tối đa trong ROIs |
| **roi_brightness** | `roi_feature_max()` | Độ sáng tối đa trong ROIs |

**Công thức tính điểm:**
```python
score = (
    0.12 * sharpness +
    0.10 * edge_density +
    0.04 * brightness +
    0.00 * novelty +
    0.03 * center_bias +
    0.34 * roi_sharpness +      # Trọng số cao nhất!
    0.30 * roi_edge_density +
    0.07 * roi_brightness
)
```

**ROI Regions** (vùng quan tâm trong frame - thường là khu vực chứa biển báo):
```yaml
roi_regions:
  - [0.30, 0.00, 0.70, 0.35]   # Phía trên giữa
  - [0.55, 0.00, 1.00, 0.40]   # Phía trên phải
  - [0.25, 0.20, 0.75, 0.60]   # Giữa frame
```

#### **Chi Tiết Step 4: Temporal NMS**

```python
coarse_keep_idx = temporal_nms(
    times_sec=coarse_times,
    scores=coarse_scores,
    top_k=coarse_top_m,           # Default: 2
    min_gap_sec=0.375             # (temporal_nms_gap_sec * 0.75)
)
```

- **Thuật toán**: 
  1. Sắp xếp frames theo điểm giảm dần
  2. Lưu frame có điểm cao nhất
  3. Bỏ qua frames nằm trong 0.375s của frame được lưu
  4. Lặp lại cho đến khi có 2 frame hoặc hết danh sách
  5. Sắp xếp lại theo thời gian

- **Kết quả**: Top-2 peak candidates

---

### **STAGE 2: REFINE STAGE (Làm Chi Tiết)**

Với mỗi peak từ stage 1:

```python
for peak in coarse_selected:  # Duyệt 2 peak từ stage 1
    win_start = peak.time_sec - refine_window_sec    # -1.0s
    win_end = peak.time_sec + refine_window_sec      # +1.0s
    
    # Lấy mẫu kỹ lưỡng hơn trong cửa sổ ±1.0s
    local_frames = self._sample_video_window(
        video_path=video_path,
        start_sec=win_start,
        end_sec=win_end,
        fps_sample=fps_refine,  # Default: 6.0 fps
        max_frames=96
    )
    
    # Tính điểm lại
    local_scored = self._score_sampled_frames(local_frames)
    fine_candidates.extend(local_scored)
```

**Quy Trình**:
```
Coarse peak 1 [±1.0s window] → Fine candidates 1
    ↓
Coarse peak 2 [±1.0s window] → Fine candidates 2
    ↓
Kết hợp: coarse_scored + fine_candidates
    ↓
Deduplicatè frames nằm quá gần nhau (< 1e-6s)
    ↓
Temporal NMS cuối cùng để lấy top-4
```

---

## 🎯 Kết Quả Cuối Cùng

### **Cấu Trúc Output**

```python
{
    "video_path": "...",
    "sampled_frames": 63,  # Tổng frames được lấy mẫu ở stage 1
    
    "selected_frames": [   # Top-4 final candidates
        {
            "frame_idx": 150,
            "time_sec": 5.2,
            "score": 0.85,
            "components": {
                "sharpness": 0.92,
                "edge_density": 0.15,
                "brightness": 0.60,
                "novelty": 0.02,
                "center_bias": 0.80,
            },
            "image": <BGR numpy array>
        },
        # ...3 khung hình khác
    ],
    
    "all_frames": [...],        # Tất cả merged candidates
    "coarse_peaks": [...]       # Top-2 coarse peaks
}
```

---

## ⚙️ Cấu Hình Theo Loại Câu Hỏi (Type-Aware)

Repository hỗ trợ **policies khác nhau** tùy loại câu hỏi:

```yaml
type_to_policy:
  sign_identification: sign_policy      # Loại: Nhận diện biển báo
  information_reading: sign_policy      # Loại: Đọc thông tin
  navigation: sign_policy               # Loại: Điều hướng
  
  rule_compliance: lane_policy          # Loại: Tuân thủ quy tắc
  verification: lane_policy             # Loại: Xác nhận
  
  object_presence: coverage_policy      # Loại: Sự hiện diện đối tượng
  counting: coverage_policy             # Loại: Đếm

policy_configs:
  sign_policy:
    top_k: 4
    temporal_nms_gap_sec: 0.40          # Khác với default 0.5s
    refine_window_sec: 0.85             # Khác với default 1.0s
```

---

## 📊 Biểu Đồ Quy Trình

```
┌─────────────────────────────────┐
│     Video (3 phút)              │
└────────────────┬────────────────┘
                 │
         ┌───────▼────────┐
         │ Coarse Sampling│ (fps_coarse=2.0)
         │ Max 96 frames  │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Trim 10%      │
         │  đầu & cuối    │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Score mỗi     │ (8 features)
         │  frame         │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Temporal NMS  │ (top_k=2)
         │  min_gap=0.37s │
         └───────┬────────┘
                 │
        ┌────────┴────────┐
        │                 │
    ┌───▼────┐        ┌──▼────┐
    │  Peak 1│        │ Peak 2 │
    └───┬────┘        └──┬─────┘
        │                 │
    ┌───▼─────────────────▼────┐
    │ Fine Refine              │ (fps_refine=6.0)
    │ ±1.0s window mỗi peak    │
    └───┬─────────────────────┐
        │                     │
    ┌───▼────┐           ┌───▼────┐
    │Fine 1  │           │Fine 2  │
    └───┬────┘           └───┬────┘
        │                    │
        └────────┬───────────┘
                 │
         ┌───────▼────────┐
         │ Merge + Dedup  │
         │ All candidates │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │ Final Temporal │ (top_k=4)
         │ NMS            │
         └───────┬────────┘
                 │
         ┌───────▼────────────┐
         │ ✓ Selected Frames  │
         │ (4 khung chính)    │
         └────────────────────┘
```

---

## 🚀 Điều Chỉnh & Tối Ưu Hóa

### **Nếu muốn frames tập trung ở biển báo:**
```yaml
weights:
  roi_sharpness: 0.40      # Tăng từ 0.34
  roi_edge_density: 0.35   # Tăng từ 0.30
  roi_brightness: 0.10     # Tăng từ 0.07
```

### **Nếu muốn frames cách xa nhau:**
```yaml
temporal_nms_gap_sec: 1.0  # Tăng từ 0.5s
```

### **Nếu muốn frames sắc nét hơn:**
```yaml
weights:
  sharpness: 0.20          # Tăng từ 0.12
```

### **Nếu muốn xem xét toàn bộ video:**
```yaml
trim_ratio: 0.0            # Từ 0.1 (loại 10%)
```

---

## 📁 File Chính

| File | Mục đích |
|------|---------|
| [src/selector/pipeline.py](../src/selector/pipeline.py) | Pipeline chính - lấy candidate frames |
| [src/selector/scorer.py](../src/selector/scorer.py) | Tính điểm frame |
| [src/selector/peak_picker.py](../src/selector/peak_picker.py) | Temporal NMS algorithm |
| [src/perception/blur.py](../src/perception/blur.py) | Tính các feature (sharpness, edges, brightness) |
| [configs/selector.yaml](../configs/selector.yaml) | Cấu hình |
| [run_selector.py](../run_selector.py) | Script chạy pipeline |

---

## 💡 Ví Dụ Chạy

```bash
# Chạy với một video cụ thể
python run_selector.py \
    --config configs/selector.yaml \
    --train_json data/train.json \
    --video_root /path/to/videos \
    --sample_id train_0019 \
    --output_dir outputs/my_results
```

**Output**:
```
outputs/my_results/
├── train_0019/
│   ├── result.json              # Metadata + điểm
│   └── frames/
│       ├── top1_5.200s_score_0.8523.jpg
│       ├── top2_6.450s_score_0.8401.jpg
│       ├── top3_8.120s_score_0.8290.jpg
│       └── top4_10.580s_score_0.8175.jpg
```

---

## 🔍 Hiểu Rõ Score Components

**Ví dụ frame có:**
- Biển báo sắc nét ở ROI → `roi_sharpness = 0.95`
- Nhiều cạnh/chi tiết → `roi_edge_density = 0.45`
- Độ sáng tốt → `roi_brightness = 0.75`
- Ở giữa video → `center_bias = 0.95`

**Score = 0.34×0.95 + 0.30×0.45 + 0.07×0.75 + 0.03×0.95 = 0.52**

---

## 📈 Performance Considerations

| Stage | Frames | FPS | Chi Phí |
|-------|--------|-----|---------|
| Coarse | ~63 | 2.0 | Thấp ⚡ |
| Refine | ~120 | 6.0 | Trung bình ⚡⚡ |
| **Total** | ~183 | Mix | **Vừa phải** |

Với cách này, video 3 phút sẽ chỉ xử lý ~183 frames thay vì 5400 frames (30fps×180s), **tiết kiệm 96% chi phí**!

