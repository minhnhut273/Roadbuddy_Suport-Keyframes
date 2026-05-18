# Support Frame Scoring (Type-Aware, ROI-Aware)

## Mục tiêu
Bản này tính điểm support frame theo **cùng tinh thần kỹ thuật hiện tại** của selector:
- dùng đúng `question_type`
- map `type -> policy`
- dùng đúng `weights` và `roi_regions` theo policy
- recompute RAW features cho support frames và candidate frames từ video
- normalize support + candidate trên cùng thang raw feature
- normalize riêng cho global features và ROI features

## Files
- `support_frame_scorer_type_aware.py`
- `test_support_frame_scorer_type_aware.py`
- `demo_support_frame_scoring_type_aware.py`

## Cách chạy

### 1) Test 1 sample + chạy pipeline
```bash
python test_support_frame_scorer_type_aware.py \
  --config configs/selector.yaml \
  --train_json data/train.json \
  --video_root src/video/dataset \
  --sample_id train_0047 \
  --output_json output_support_compare.json
```

### 2) Chỉ score support frame, không chạy pipeline
```bash
python test_support_frame_scorer_type_aware.py \
  --config configs/selector.yaml \
  --train_json data/train.json \
  --video_root src/video/dataset \
  --sample_id train_0047 \
  --no_pipeline
```

### 3) Demo nhanh từ `result.json`
```bash
python demo_support_frame_scoring_type_aware.py
```

## Output
JSON sẽ gồm:
- `sample`
- `selector_result`
- `support_frames_scoring`

Trong đó `support_frames_scoring` có:
- `question_type`
- `policy`
- `support_frames`
- `candidate_frames`
- `comparison`

## Ghi chú
Bản này phù hợp hơn bản cũ vì không còn:
- trộn raw support với normalized candidate
- normalize ROI bằng global scale
- bỏ qua type-aware policy
