# Support Frame Scoring Implementation Guide

## ✅ Những Gì Đã Được Thực Hiện

### 1. **Core Module: `src/eval/support_frame_scorer.py`** 
Lớp `SupportFrameScorer` để tính điểm support frames:
- ✓ Extract frame từ video tại support timestamps
- ✓ Tính 8 features (sharpness, edges, brightness, ROI variants)
- ✓ Normalize features chung với candidate frames
- ✓ Tính weighted score sử dụng cùng config như candidate frames
- ✓ So sánh rank với tất cả candidate frames
- ✓ Tạo detailed comparison report

### 2. **Integration into Main Flow**
Sửa `run_selector.py`:
- ✓ Import `SupportFrameScorer`
- ✓ Tính điểm support frames tự động
- ✓ Lưu kết quả vào `result.json`
- ✓ In detail report ra console

### 3. **Documentation**
- ✓ `SUPPORT_FRAME_SCORING_FLOW.md` - Hướng dẫn chi tiết
- ✓ `TEST_GUIDE.md` - Guide test (file này)

### 4. **Test Script: `test_support_frame_scorer.py`**
Script standalone để test & demonstrate:
- ✓ Pretty print results
- ✓ Flexible arguments (sample_id, sample_index, etc.)
- ✓ Optional pipeline running
- ✓ Save results to JSON
- ✓ Compare support vs candidate frames

---

## 🚀 Quick Start

### **Cách 1: Run với Main Pipeline**

```bash
python run_selector.py \
    --config configs/selector.yaml \
    --train_json data/train.json \
    --video_root src/video/dataset \
    --sample_id train_0047
```

**Output sẽ bao gồm:**
```
Support Frame Scoring:
  Total Support Frames    : 1
    Support Frame 1: t=1.3323s, score=0.7234 (Rank #3) [good]
  Avg Support Score       : 0.7234
  Comparison Summary      : ~ Support frames (avg 0.7234) are within candidate range (...)
```

### **Cách 2: Test Specific Sample**

```bash
python test_support_frame_scorer.py \
    --config configs/selector.yaml \
    --train_json data/train.json \
    --video_root src/video/dataset \
    --sample_id train_0047 \
    --output_json output_comparison.json
```

### **Cách 3: Test Without Pipeline** (Chỉ score support)

```bash
python test_support_frame_scorer.py \
    --config configs/selector.yaml \
    --train_json data/train.json \
    --video_root src/video/dataset \
    --sample_id train_0047 \
    --no_pipeline
```

### **Cách 4: Test by Index**

```bash
python test_support_frame_scorer.py \
    --config configs/selector.yaml \
    --train_json data/train.json \
    --video_root src/video/dataset \
    --sample_index 0
```

---

## 📊 Output Format

### **Example JSON Output**

```json
{
  "sample": {
    "id": "train_0047",
    "question": "Trong video có xuất hiện biển báo hiệu lệnh không?",
    "type": "sign_identification",
    "support_frames": [1.332302],
    "video_path": "train/videos/cf8d446a_035_clip_003_0012_0019_Y.mp4"
  },
  "selector_result": {
    "selected_frames": [
      {
        "frame_idx": 4,
        "time_sec": 0.13333333,
        "score": 0.9954819915588017,
        "components": { ... }
      },
      ...
    ],
    "all_frames": [ ... ]
  },
  "support_frames_scoring": {
    "support_frames": [
      {
        "time_sec": 1.332302,
        "score": 0.7234,
        "components": {
          "sharpness": 0.85,
          "edge_density": 0.45,
          "brightness": 0.72,
          "roi_sharpness": 0.91,
          "roi_edge_density": 0.48,
          "roi_brightness": 0.78
        },
        "rank_in_candidates": 3,
        "status": "good"
      }
    ],
    "comparison": {
      "total_support_frames": 1,
      "total_candidate_frames": 87,
      "avg_support_score": 0.7234,
      "max_candidate_score": 0.9954,
      "min_candidate_score": 0.3421,
      "all_support_better_than_candidates": false,
      "summary": "~ Support frames (avg 0.7234) are within candidate range..."
    }
  },
  "evaluation": { ... }
}
```

---

## 📋 Output Interpretation

### **Status Levels**

- **✓ excellent** (rank ≤ 2) - Support frame tốt hơn top-2 candidates
- **◐ good** (rank ≤ 4) - Support frame tốt hơn top-4 candidates  
- **✗ weak** (rank > 4) - Support frame yếu hơn top-4 candidates

### **Comparison Summary**

| Pattern | Meaning |
|---------|---------|
| ✓ All support... BETTER | Tất cả support frames tốt hơn tất cả candidates |
| ✓ Average support... BETTER | Avg support tốt hơn max candidate |
| ~ Support frames... within | Support nằm giữa min/max candidates |
| ✗ Support frames... WEAKER | Support yếu hơn tất cả candidates |

---

## 🔍 Key Features

### **1. Normalization Chung**
- Support frames được normalize dùng cùng scale với candidate frames
- Đảm bảo comparison công bằng dù support frames được extract riêng biệt

### **2. ROI-Aware Scoring**
- Support frames được evaluated trên cùng ROI regions như candidates
- Hữu ích cho tasks như nhận diện biển báo (tập trung vào ROI)

### **3. Rank Comparison**
- Mỗi support frame được rank so với tất cả candidate frames
- Giúp hiểu vị trí tương đối của support frames

### **4. Detailed Components**
- Xem chi tiết từng feature component
- Debug tại sao support frame có score cao/thấp

---

## 💡 Use Cases

### **Use Case 1: Validate Pipeline Quality**
```bash
# Test trên nhiều samples
for i in {0..49}; do
  python test_support_frame_scorer.py \
    --train_json data/train.json \
    --video_root src/video/dataset \
    --sample_index $i \
    --output_json results/sample_$i.json
done

# Analyze results
python analyze_support_vs_candidates.py
```

### **Use Case 2: Debug Specific Sample**
```bash
python test_support_frame_scorer.py \
    --sample_id train_0047 \
    --train_json data/train.json \
    --video_root src/video/dataset \
    --output_json debug_train_0047.json
```

### **Use Case 3: Hyperparameter Tuning**
```python
# Edit configs/selector.yaml
# Adjust weights, ROI regions, etc.
# Then test
python run_selector.py ... # Test with new config
```

---

## 🧪 Example Test Run

```bash
$ python test_support_frame_scorer.py \
    --config configs/selector.yaml \
    --train_json data/train.json \
    --video_root src/video/dataset \
    --sample_id train_0047

✓ Loaded 50 samples from data/train.json
✓ Using sample: train_0047
✓ Video: D:\...\cf8d446a_035_clip_003_0012_0019_Y.mp4
✓ Question Type: sign_identification
✓ Support Frames: [1.332302]

🔄 Running candidate frame selector pipeline...
✓ Found 4 selected frames
✓ Analyzed 87 total candidate frames

🔄 Scoring support frames...
✓ Support frames scored

==========================================================================================
🎯 SUPPORT FRAME SCORING COMPARISON
==========================================================================================
Sample ID    : train_0047
Question     : Trong video có xuất hiện biển báo hiệu lệnh không?

📍 Support Frames (từ file train):
  ◐ Support Frame 1
      Time: 1.332302s
      Score: 0.723401
      Rank: #3 (out of 87 candidates)
      Status: good
      Components:
        - sharpness: 0.8534
        - edge_density: 0.4512
        - brightness: 0.7245
        - roi_sharpness: 0.9123
        - roi_edge_density: 0.4801
        - roi_brightness: 0.7834

📊 Candidate Frames (từ pipeline):
  Total candidates: 87
  Top 4 scores: ['0.9954819915588017', '0.6710210414896577', '0.6060941068004286', '0.5701814126968384']
  Score range: 0.339302 - 0.995482
  Average: 0.549123

📈 Comparison Summary:
  Avg Support Score: 0.723401
  Max Candidate Score: 0.995482
  Min Candidate Score: 0.339302

🟡 ~ Support frames (avg 0.723401) are within candidate range (0.339302 - 0.995482)
==========================================================================================
```

---

## 📁 Files Modified/Created

| File | Type | Description |
|------|------|-------------|
| `src/eval/support_frame_scorer.py` | NEW | Core scorer module |
| `run_selector.py` | MODIFIED | Added support frame scoring |
| `test_support_frame_scorer.py` | NEW | Test script with pretty output |
| `SUPPORT_FRAME_SCORING_FLOW.md` | NEW | Detailed documentation |
| `SUPPORT_FRAME_SCORING_IMPLEMENTATION_GUIDE.md` | NEW | This file |

---

## ⚠️ Important Notes

1. **Normalization**: Support frames score có thể khác nếu có/không có candidate frames để normalize
   - Với candidates: normalize chung scale
   - Không candidates: normalize chỉ trong support frames

2. **Features Not Applicable**: 
   - Novelty score (không có frame trước) → sử dụng 0.5
   - Center bias (không có context vị trí) → sử dụng 0.5

3. **Performance**: 
   - Mỗi support frame cần extract từ video (I/O bound)
   - Với nhiều support frames hoặc video lớn, có thể chậm

4. **Dependencies**:
   - OpenCV (cv2) - extract frames
   - NumPy - tính toán
   - YAML - config
   - Các modules từ src/

---

## 🐛 Troubleshooting

### **"Cannot extract frame at X.Xs"**
- Video path sai hoặc video corrupt
- Check `video_abspath` trong output

### **Score quá cao/thấp so với expected**
- Check normalization scale (có candidate frames hay không?)
- Verify config weights

### **Rank luôn #1 khi không có candidates**
- Bình thường - không có competitors
- Chạy cùng pipeline để có candidates

---

## 📞 Questions?

Refer to `SUPPORT_FRAME_SCORING_FLOW.md` cho detailed explanation hoặc review code comments trong `src/eval/support_frame_scorer.py`.

