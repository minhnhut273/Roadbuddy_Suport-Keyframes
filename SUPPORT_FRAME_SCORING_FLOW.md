# Support Frame Scoring Flow

## 📌 Giới Thiệu

Module `SupportFrameScorer` tính điểm cho **support frames** (frame được cung cấp trong file train) nhằm **so sánh với candidate frames** được pipeline lấy ra.

Điều này giúp:
1. **Đánh giá chất lượng pipeline** - liệu pipeline có lấy được frames tốt hơn support frames không?
2. **Phát hiện vấn đề** - nếu support frames có điểm cao hơn candidates, có thể pipeline cần tối ưu
3. **Benchmark** - so sánh trực tiếp điểm support frame vs candidate frames

---

## 🎯 Cách Hoạt Động

### **Quy Trình Chính**

```
Input: Video + Support Frame Times (từ train JSON)
          ↓
    Extract Frame tại mỗi support time
          ↓
    Tính 8 feature cho mỗi frame (giống như candidate scoring)
          ↓
    Normalize features dùng chung scale với candidate frames
          ↓
    Tính final score = weighted sum of features
          ↓
    So sánh với candidate frames scores
          ↓
    Output: Detailed comparison + Summary
```

### **Các Features Được Tính**

Giống như candidate frame scorer:

| Feature | Weight (Config) | Ý Nghĩa |
|---------|-----------------|---------|
| **sharpness** | 0.12 | Độ sắc nét toàn frame |
| **edge_density** | 0.10 | Mật độ cạnh toàn frame |
| **brightness** | 0.04 | Độ sáng toàn frame |
| **novelty** | 0.00 | Khác biệt vs frame trước (N/A cho support) |
| **center_bias** | 0.03 | Sở thích ở giữa video (N/A cho support) |
| **roi_sharpness** | 0.34 | Độ sắc nét tối đa trong ROIs |
| **roi_edge_density** | 0.30 | Mật độ cạnh tối đa trong ROIs |
| **roi_brightness** | 0.07 | Độ sáng tối đa trong ROIs |

---

## 📊 Output Chi Tiết

### **1. Per-Frame Information**

```json
{
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
  ]
}
```

**Giải thích:**
- `rank_in_candidates`: Vị trí xếp hạng so với tất cả candidate frames (1 = tốt nhất)
- `status`: 
  - `excellent`: rank ≤ 2 (tốt hơn top 2 candidates)
  - `good`: rank ≤ 4 (tốt hơn top 4 candidates)
  - `weak`: rank > 4 (yếu hơn top 4 candidates)

### **2. Comparison Statistics**

```json
{
  "comparison": {
    "total_support_frames": 1,
    "total_candidate_frames": 87,
    "avg_support_score": 0.7234,
    "max_candidate_score": 0.9954,
    "min_candidate_score": 0.3421,
    "all_support_better_than_candidates": false,
    "summary": "✗ Support frames (avg 0.7234) are WEAKER than all candidates (0.3421 - 0.9954)"
  }
}
```

**Summary Messages:**
- ✓ `all_support_better_than_candidates`: Tất cả support frames tốt hơn tất cả candidates
- ✓ Avg support tốt hơn max candidate
- ~ Support frames nằm trong range của candidates
- ✗ Support frames yếu hơn tất cả candidates

---

## 💻 Cách Sử Dụng

### **1. Tích Hợp với `run_selector.py`** (Sẵn có)

Module đã được tích hợp vào `run_selector.py`. Khi chạy:

```bash
python run_selector.py \
    --config configs/selector.yaml \
    --train_json data/train.json \
    --video_root /path/to/videos \
    --sample_id train_0019
```

**Output sẽ bao gồm:**
```
Support Frame Scoring:
  Total Support Frames    : 1
    Support Frame 1: t=1.3323s, score=0.7234 (Rank #3) [good]
  Avg Support Score       : 0.7234
  Comparison Summary      : ~ Support frames (avg 0.7234) are within candidate range (0.3421 - 0.9954)
```

### **2. Sử Dụng Standalone**

```python
from src.eval.support_frame_scorer import SupportFrameScorer
import yaml

# Load config
with open("configs/selector.yaml") as f:
    cfg = yaml.safe_load(f)

# Tạo scorer
scorer = SupportFrameScorer(cfg)

# Tính điểm support frames
result = scorer.score_support_frames(
    video_path="/path/to/video.mp4",
    support_times=[1.332, 5.2, 8.1],  # Support frame times
    all_candidate_frames=candidate_frames_list  # Optional
)

# Xem kết quả
print(result["comparison"]["summary"])
for sf in result["support_frames"]:
    print(f"  Time: {sf['time_sec']:.4f}s, Score: {sf['score']:.4f}")
```

### **3. So Sánh Chi Tiết**

```python
# So sánh điểm trực tiếp
support_scores = [sf["score"] for sf in result["support_frames"]]
candidate_scores = [c["score"] for c in all_candidate_frames]

print(f"Support:  avg={sum(support_scores)/len(support_scores):.4f}, max={max(support_scores):.4f}")
print(f"Candidate: avg={sum(candidate_scores)/len(candidate_scores):.4f}, max={max(candidate_scores):.4f}")
```

---

## 📈 Phân Tích Kết Quả

### **Trường Hợp 1: Support frames tốt hơn candidates**

✓ **Điểm tích cực:**
- Support frames được chọn bởi con người/expert
- Pipeline lấy được frames tốt

✗ **Điểm tiêu cực:**
- Nếu support frames tốt hơn significant, có thể pipeline cần tối ưu
- Check weights - có thể feature nào được ưu tiên quá cao/quá thấp

### **Trường Hợp 2: Support frames yếu hơn candidates**

✓ **Điểm tích cực:**
- Pipeline lấy được frames tốt hơn support frames
- Pipeline hoạt động hiệu quả

✗ **Điểm tiêu cực:**
- Nếu support frames quá yếu, có thể support frames không phải lựa chọn tốt nhất
- Có thể cần review data labeling

### **Trường Hợp 3: Support frames gần bằng candidates**

~ **Nhận xét:**
- Pipeline hoạt động tương đương với support frames
- Cân bằng tốt giữa chất lượng và đa dạng frame

---

## 🔧 Tối Ưu Hóa

### **Nếu Support Frames Luôn Yếu Hơn Candidates**

Có thể:
1. **Tăng trọng số ROI** - support frames có thể không focus vào ROI
2. **Check video quality** - có thể video có vấn đề ở support times
3. **Review data labeling** - support frames có thể được chọn không optimal

### **Nếu Support Frames Luôn Tốt Hơn Candidates**

Có thể:
1. **Tăng độ "gắt gao"** của candidate selection
2. **Adjust weights** - ưu tiên features mà support frames mạnh
3. **Kiểm tra sampling strategy** - có thể candidate frame sampling miss đi best frames

---

## 📁 File Liên Quan

| File | Mục Đích |
|------|---------|
| [src/eval/support_frame_scorer.py](../src/eval/support_frame_scorer.py) | Core scorer module |
| [run_selector.py](../run_selector.py) | Main script tích hợp |
| [src/selector/scorer.py](../src/selector/scorer.py) | Candidate frame scorer (reference) |
| [configs/selector.yaml](../configs/selector.yaml) | Cấu hình weights |

---

## 🧪 Test Example

```python
# test_support_frame_scorer.py
from src.eval.support_frame_scorer import SupportFrameScorer
from src.data.train_loader import load_train_data, build_video_abspath
import yaml

# Load sample
with open("configs/selector.yaml") as f:
    cfg = yaml.safe_load(f)

data = load_train_data("data/train.json")
sample = data[0]

# Score support frames
scorer = SupportFrameScorer(cfg)
video_path = build_video_abspath("src/video/dataset", sample["video_path"])

result = scorer.score_support_frames(
    video_path=video_path,
    support_times=sample.get("support_frames", [])
)

print(f"Support frames: {len(result['support_frames'])}")
print(f"Summary: {result['comparison']['summary']}")
```

---

## 💡 Use Cases

### **1. Pipeline Validation**
```python
# Kiểm tra tất cả samples, xem bao nhiêu % support frames yếu hơn candidates
for sample in data:
    result = scorer.score_support_frames(video_path, sample.get("support_frames", []))
    if result["comparison"]["all_support_better_than_candidates"]:
        print(f"⚠️  {sample['id']}: Support frames tốt hơn")
```

### **2. Data Quality Check**
```python
# Tìm support frames có điểm quá thấp
weak_supports = []
for sample in data:
    result = scorer.score_support_frames(...)
    for sf in result["support_frames"]:
        if sf["score"] < 0.4:
            weak_supports.append((sample['id'], sf))
```

### **3. Hyperparameter Tuning**
```python
# So sánh kết quả với weights khác nhau
for sharpness_weight in [0.10, 0.15, 0.20]:
    cfg["weights"]["sharpness"] = sharpness_weight
    scorer = SupportFrameScorer(cfg)
    # ... test all samples ...
```

---

## 📝 Notes

- Support frames được extract lần đầu tiên, có thể có I/O overhead
- Caching frame extraction có thể tăng performance nếu cần run nhiều lần
- Novelty score không áp dụng cho support frames (sử dụng 0.5 - neutral)
- Center bias không áp dụng cho support frames (sử dụng 0.5 - neutral)

