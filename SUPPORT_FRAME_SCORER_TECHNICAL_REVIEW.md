# Technical Checklist: Support Frame Scoring Implementation Review

**Date**: May 9, 2026  
**Module**: `src/eval/support_frame_scorer.py`  
**Status**: ⚠️ Concept Valid, Implementation Incomplete  
**Severity**: High - Scores not directly comparable with pipeline

---

## Executive Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| **Concept** | ✅ Valid | Aligns with pipeline scoring philosophy |
| **Architecture** | ⚠️ Partial | Missing type-awareness, feature structure |
| **Normalization** | ❌ Incorrect | Mixing normalized & raw values |
| **ROI Handling** | ❌ Incorrect | Wrong scale for ROI features |
| **Policy Handling** | ❌ Missing | No type-aware policy support |
| **Score Compatibility** | ❌ Not Guaranteed | Cannot directly compare with pipeline scores |

---

## Detailed Checklist

### ✅ CORRECT: Core Concept

- [x] Purpose is sound: extract support frames, score them, compare with candidates
- [x] Feature set matches pipeline: sharpness, edge_density, brightness, roi_*, novelty, center_bias
- [x] Uses weighted sum scoring like pipeline
- [x] Provides ranking & status labels
- [x] Module separation is clean

**Assessment**: Good foundational design. Ready for fixes.

---

### ❌ INCORRECT: Normalization Strategy

**Location**: `SupportFrameScorer.score_support_frames()` lines 165-210

**Issue**:

```python
# CURRENT (WRONG)
if all_candidate_frames:
    cand_sharpness = [c["components"]["sharpness"] for c in all_candidate_frames]  # ← Already normalized [0-1]
    
    support_raw_sharp = [s["raw_sharpness"] for s in scored_support]  # ← Raw Laplacian values
    
    all_sharp = cand_sharpness + support_raw_sharp  # ❌ Mixing scales!
    
    sharp_min, sharp_max = min(all_sharp), max(all_sharp)
    # Result: scale is distorted completely
```

**Why It's Wrong**:

| Type | Example Value | Meaning |
|------|---------------|---------|
| `cand_sharpness` | `0.75` | Already normalized to [0, 1] |
| `support_raw_sharpness` | `1250.5` | Raw Laplacian variance |
| **Combined min** | `0.75` | Meaningless |
| **Combined max** | `1250.5` | Meaningless |
| **Normalized result** | Wrong | Scale is completely distorted |

**Impact**:
- Support frame score cannot be directly compared with pipeline scores
- Ranking and status labels are unreliable
- Could show support frame as "good" when it's actually "weak" or vice versa

**Needs to be fixed**: ❌ **CRITICAL**

**How to fix**:
- Option A: Store raw features in candidate frames from pipeline
- Option B: Recalculate raw features for all candidate frames before comparing
- Option C: Don't use `all_candidate_frames`, only rely on intrinsic support frame scoring

---

### ❌ INCORRECT: ROI Feature Normalization Scale

**Location**: `SupportFrameScorer.score_support_frames()` lines 223-226

**Issue**:

```python
# CURRENT (WRONG)
roi_sharp_norm = normalize_sharp(sf["raw_roi_sharpness"])       # ← Using global sharpness scale
roi_edges_norm = normalize_edges(sf["raw_roi_edge_density"])   # ← Using global edges scale
roi_bright_norm = normalize_bright(sf["raw_roi_brightness"])   # ← Using global brightness scale
```

**Why It's Wrong**:

ROI features have different statistical distributions than global features:

```
Global sharpness range:    [0, 2000]
ROI sharpness range:       [0, 500]   ← Different scale!

If normalize_sharp was built from [0, 2000], then applying it to ROI value 400:
  normalized = (400 - 0) / (2000 - 0) = 0.2

But if ROI scale is really [0, 500], should be:
  normalized = (400 - 0) / (500 - 0) = 0.8

Result: ROI sharpness is underestimated by 4x
```

**Impact**:
- ROI features (which have 64% combined weight in config) are misscaled
- Support frames with strong ROI content get unfairly low scores
- Comparison with pipeline is skewed

**Needs to be fixed**: ❌ **CRITICAL**

**How to fix**:
- Create separate normalize functions for ROI features
- Build ROI normalize scale from ROI raw values only
- Result: 6 independent feature groups instead of 3

---

### ❌ MISSING: Type-Aware Policy Support

**Location**: `SupportFrameScorer.__init__()` lines 20-31

**Issue**:

```python
# CURRENT (WRONG)
def __init__(self, config: dict):
    self.weights = config.get("weights", {})              # ← Global weights only
    self.roi_regions = config.get("roi_regions", [])     # ← Global ROI only
    
# Missing:
# - question_type parameter
# - type_to_policy lookup
# - policy_configs access
```

**Pipeline Does This** (reference: `src/selector/pipeline.py`):

```python
def _get_policy_config(self, question_type: str | None) -> dict:
    if not self.cfg.get("type_aware", False):
        return {
            "top_k": self.cfg.get("top_k", 4),
            "weights": self.cfg.get("weights", {}),
            "roi_regions": self.cfg.get("roi_regions", []),
        }
    
    type_to_policy = self.cfg.get("type_to_policy", {})
    policy_name = type_to_policy.get(question_type)
    
    if policy_name in self.cfg.get("policy_configs", {}):
        return policy_configs[policy_name]
    
    return default_config
```

**Why It's Wrong**:

From `configs/selector.yaml`:

```yaml
type_aware: true

type_to_policy:
  sign_identification: sign_policy
  rule_compliance: lane_policy
  object_presence: coverage_policy

policy_configs:
  sign_policy:
    top_k: 4
    temporal_nms_gap_sec: 0.40        # Different from global 0.5!
    weights:
      roi_sharpness: 0.40             # Different from global 0.34!
```

**Current SupportFrameScorer**:
- Always uses global weights (roi_sharpness: 0.34)
- Always uses global roi_regions
- Ignores sign_policy weights (roi_sharpness: 0.40)

**Impact**:
- If pipeline is scoring with `sign_policy`, but scorer uses global config, they're using different weights
- Support frame could score 0.65 with global weights, but 0.72 with sign_policy weights
- Comparison is meaningless when policies don't match

**Example mismatch**:

```
Sample train_0047:
  question_type: "sign_identification"
  pipeline uses: sign_policy weights (roi_sharpness: 0.40)
  support scorer uses: global weights (roi_sharpness: 0.34)
  
Result:
  support_frame score: 0.5 (using wrong weights)
  vs
  pipeline scores: calculated with different weights
  
Comparison is INVALID ❌
```

**Needs to be fixed**: ❌ **CRITICAL**

**How to fix**:
- Add `question_type` parameter to `score_support_frames()`
- Copy `_get_policy_config()` logic from pipeline
- Use correct policy weights & ROI regions based on question type

---

### ❌ INCORRECT: Test Script Candidate Frame Structure

**Location**: `test_support_frame_scorer.py` lines 168-170

**Issue**:

```python
# CURRENT (WRONG)
all_candidate_frames=[{"score": s} for s in candidate_scores] if candidate_scores else None

# Scorer expects:
# c["components"]["sharpness"]
# c["components"]["edge_density"]
# c["components"]["brightness"]

# But test passes only:
# {"score": 0.995}  ← Missing components!
```

**Why It's Wrong**:

```python
# In SupportFrameScorer.score_support_frames():
if all_candidate_frames:
    cand_sharpness = [c["components"]["sharpness"] for c in all_candidate_frames]
    # ↑ Will throw KeyError: 'components' if structure is wrong
```

**Impact**:
- Test script will crash when trying to normalize
- Or silently produce wrong results if error handling differs
- Cannot validate normalization fixes

**Needs to be fixed**: ❌ **IMPORTANT**

**How to fix**:
```python
# Instead of:
all_candidate_frames=[{"score": s} for s in candidate_scores]

# Should be:
all_candidate_frames = result["all_frames"]  # Contains full structure with components
```

---

### ⚠️ ACCEPTABLE BUT NOT IDEAL: Novelty & Center_bias = 0.5

**Location**: `SupportFrameScorer.score_support_frames()` lines 231-233

**Current**:
```python
score = (
    ...
    + self.weights.get("novelty", 0.0) * 0.5  # Using neutral value
    + self.weights.get("center_bias", 0.0) * 0.5
    ...
)
```

**Assessment**: 

| Aspect | Evaluation |
|--------|-----------|
| **Novelty = 0.5** | ✅ Reasonable - support frame has no prev frame context |
| **Center_bias = 0.5** | ⚠️ Debatable - support frame time is not "centered" in video |
| **vs Pipeline** | ❌ Pipeline uses actual novelty & center_bias from sequence |

**Example mismatch**:

```
Support frame at t=1.3s in 8.0s video:
  Pipeline would compute: center_bias = 1 - |1.3/8.0 - 0.5| * 2 = 0.675
  Support scorer assigns: center_bias = 0.5
  Difference: 0.175 in bias component
```

**Impact**: Minor - only affects up to 3% of final score (0.03 weight for center_bias)

**Verdict**: ⚠️ **Acceptable as temporary workaround**, but not perfect for score equivalence

**Could improve by**:
- Computing actual center_bias from support_time and video_duration
- Leaving novelty as 0.5 (no history available)

---

### ❌ INCOMPLETE: Integration with run_selector.py

**Location**: `run_selector.py` lines 74-79

**Current**:
```python
support_scoring_result = support_scorer.score_support_frames(
    video_path=video_path,
    support_times=support_frames,
    all_candidate_frames=result["all_frames"],  # ← Correct structure
)
```

**Assessment**: 

This part is actually **correct** - it passes `result["all_frames"]` which has full structure.

However, it **doesn't pass `question_type`**, so the fix for issue #3 would require:

```python
support_scoring_result = support_scorer.score_support_frames(
    video_path=video_path,
    support_times=support_frames,
    all_candidate_frames=result["all_frames"],
    question_type=question_type,  # ← Need to add this
)
```

---

## Summary Table

| Issue | Severity | Impact | Fix Type |
|-------|----------|--------|----------|
| **Normalize mix raw+normalized** | 🔴 Critical | Scores incomparable | Major refactor |
| **ROI scale wrong** | 🔴 Critical | ROI underestimated 4-10x | Major refactor |
| **No type-aware policy** | 🔴 Critical | Wrong weights used | Add logic |
| **Test script structure** | 🟡 Important | Test broken | Minor fix |
| **Novelty/CenterBias = 0.5** | 🟡 Minor | Up to 3% error | Enhancement |

---

## Recommendations for Code Review / Team Discussion

**Current State**:
> "Ý tưởng support frame scoring là phù hợp với mục tiêu phân tích pipeline, vì nó cho phép so sánh trực tiếp support frame trong dữ liệu với candidate frames mà bộ chọn khung hình sinh ra."

**Limitation**:
> "Tuy nhiên, implementation hiện tại chưa hoàn toàn tương thích với cơ chế scoring chính của selector:
> 1. Đang trộn candidate features đã normalize với support features thô
> 2. Chưa normalize riêng cho ROI features theo scale của ROI
> 3. Chưa dùng đúng policy theo loại câu hỏi (type-aware)
> 4. Cấu trúc test script không khớp với module requirements"

**Current Capability**:
> "Module hiện phù hợp để debug định tính (qualitative debugging), nhưng chưa đủ chính xác để kết luận định lượng mạnh (quantitative comparison) về việc support frame 'tốt hơn' hay 'yếu hơn' candidate frames."

**Required Fixes** (Priority Order):

1. 🔴 **Refactor normalization** - Separate raw values, normalize together with support raw
2. 🔴 **Separate ROI scale** - Build normalize functions for ROI features independently  
3. 🔴 **Add type-aware policy** - Accept question_type, lookup policy_configs like pipeline
4. 🟡 **Fix test integration** - Pass full candidate frame structure with components
5. 🟡 **Improve center_bias** - Calculate actual center_bias from support time & duration

**Effort Estimate**:
- Fixes 1-3: ~2 hours refactoring + validation
- Fixes 4-5: ~30 minutes

**Result After Fix**:
- ✅ Support frame scores directly comparable with pipeline scores
- ✅ Scores use same normalization scale and feature space
- ✅ Respects type-aware policies (sign_policy, lane_policy, coverage_policy)
- ✅ Reliable for quantitative comparisons

---

## Decision Point

**Option A**: Proceed with current implementation as "qualitative debug tool" 
- Pro: Ready to use now
- Con: Scores are not accurate for making conclusions

**Option B**: Fix implementation to be "quantitatively correct"
- Pro: Scores are directly comparable with pipeline
- Con: Takes ~2-3 hours to implement & validate

**Recommendation**: **Option B** - Given that purpose is to validate pipeline quality, accuracy is worth the time investment.

---

## Sign-off

**Review Date**: May 9, 2026  
**Reviewed By**: Technical Analysis  
**Status**: Ready for Fix Implementation  
**Next Step**: Code refactoring with fixes 1-5 above

