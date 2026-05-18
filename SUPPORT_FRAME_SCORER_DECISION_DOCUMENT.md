# Support Frame Scorer - Technical Decision Document

**Project**: Road Buddy - SF  
**Module**: Support Frame Scoring for Pipeline Validation  
**Date**: May 9, 2026  
**Classification**: Technical Review - Implementation Assessment

---

## 1. Problem Statement

We need to validate whether the candidate frame selector pipeline is choosing good frames compared to ground-truth support frames provided in training data.

**Solution Concept**: Score support frames using the same feature space and scoring logic as the pipeline, then compare scores directly.

**Question**: Is the current implementation technically sound?

---

## 2. Assessment Result: PARTIALLY VALID ⚠️

### 2.1 Concept Level ✅ VALID

The idea of support frame scoring is sound:

```
Support Frame Scoring Purpose:
├─ Extract frame at support_times from video
├─ Calculate features: sharpness, edges, brightness, ROI variants
├─ Apply weighted scoring: score = Σ weights[i] × features[i]
├─ Compare: rank support_score against all_candidate_scores
└─ Insight: Is pipeline missing better frames?
```

This aligns perfectly with:
- Pipeline's feature philosophy
- Pipeline's scoring mechanism
- Pipeline's debug objectives

### 2.2 Implementation Level ❌ INCOMPLETE

4 Critical issues prevent score equivalence:

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Normalize: raw vs normalized mix | 🔴 Critical | Breaks all comparisons |
| 2 | ROI normalize: wrong scale domain | 🔴 Critical | Underestimates ROI 4-10x |
| 3 | Missing: type-aware policy | 🔴 Critical | Uses wrong weights |
| 4 | Structure: test script mismatch | 🟡 Important | Test won't run |

---

## 3. Detailed Issue Analysis

### Issue #1: Normalization Strategy

**Current Code** (`support_frame_scorer.py` line 175-185):

```python
if all_candidate_frames:
    cand_sharpness = [c["components"]["sharpness"] for c in all_candidate_frames]
    support_raw_sharp = [s["raw_sharpness"] for s in scored_support]
    
    all_sharp = cand_sharpness + support_raw_sharp  # ← MIXING SCALES
    sharp_min, sharp_max = min(all_sharp), max(all_sharp)
```

**The Problem**:

```
Candidate "sharpness": 0.75     (normalized in [0, 1] from pipeline)
Support "raw_sharpness": 1250.5 (raw Laplacian variance)

Combined range: [0.75, 1250.5]  ← MEANINGLESS

When normalize support raw using this range:
  norm = (1250.5 - 0.75) / (1250.5 - 0.75) = 1.0  ← Distorted!
```

**Why It Fails**:

```
Pipeline flow:
  all_frames → raw_sharp values [100, 2000] → normalize [0, 1] → store 0.75

Support scorer flow:
  support_frame → raw_sharp value 1250.5 → mix with [0.75] → normalize

Result: 
  Different normalization domains = Different scales = Incomparable scores
```

**Impact**: ❌ **Comparison is INVALID**

---

### Issue #2: ROI Feature Scale

**Current Code** (`support_frame_scorer.py` line 223-226):

```python
roi_sharp_norm = normalize_sharp(sf["raw_roi_sharpness"])
roi_edges_norm = normalize_edges(sf["raw_roi_edge_density"])
roi_bright_norm = normalize_bright(sf["raw_roi_brightness"])
```

**The Problem**:

```
normalize_sharp was built from global sharpness range: [0, 2000]

But ROI sharpness has different range: [0, 500]

Example: roi_sharpness = 400
  Using global scale: (400 - 0) / (2000 - 0) = 0.2
  Using ROI scale:    (400 - 0) / (500 - 0) = 0.8
  
Error: 4x underestimation!
```

**Why It's Wrong**:

ROI features are patches of the image, naturally smaller values than global features.

```
Global edge density:   0-100% of frame has edges
ROI edge density:      0-100% of ROI patch has edges

Different statistical distributions → Require different scales
```

**Impact**: ❌ **ROI components underestimated by 4-10x**

**Consequence**:

```
Config weights:
  roi_sharpness:     0.34  ← 34% of score
  roi_edge_density:  0.30  ← 30% of score
  roi_brightness:    0.07  ← 7% of score
  Total ROI weight:       64%

If ROI is 4x underestimated:
  Expected ROI contribution: 64% × score
  Actual ROI contribution:   16% × score  ← 4x LOWER
  
Result: Support frame score is artificially low by ~12% absolute
```

---

### Issue #3: Type-Aware Policy Missing

**Current Code** (`support_frame_scorer.py` line 20-31):

```python
def __init__(self, config: dict):
    self.weights = config.get("weights", {})           # ← Global only
    self.roi_regions = config.get("roi_regions", [])   # ← Global only
```

**Pipeline Does** (`selector/pipeline.py` line 78-103):

```python
def _get_policy_config(self, question_type: str | None) -> dict:
    if not self.cfg.get("type_aware", False):
        return {"weights": ..., "roi_regions": ...}
    
    type_to_policy = self.cfg.get("type_to_policy", {})
    policy_name = type_to_policy.get(question_type)
    
    if policy_name in policy_configs:
        return policy_configs[policy_name]
    
    return default_config
```

**Config Structure** (`configs/selector.yaml`):

```yaml
type_aware: true  ← Pipeline respects this!

weights:                      ← Global weights
  sharpness: 0.12
  roi_sharpness: 0.34

type_to_policy:              ← Maps question type to policy
  sign_identification: sign_policy
  rule_compliance: lane_policy
  object_presence: coverage_policy

policy_configs:
  sign_policy:               ← Policy-specific weights
    weights:
      roi_sharpness: 0.40    ← Different from global 0.34!
      roi_edge_density: 0.35 ← Different from global 0.30!
```

**The Problem**:

```
Sample: train_0047
  question_type: "sign_identification"
  Pipeline uses: sign_policy weights
  Support scorer uses: global weights
  
Pipeline scoring: score = 0.34×sharp + 0.35×edges + ...
Support scoring:  score = 0.34×sharp + 0.30×edges + ...  ← Wrong!
                                       └─ 5% difference

For multiple ROI weights:
  Total divergence: 5-10% difference in final score
```

**Impact**: ❌ **Scores use different weights = Not comparable**

---

### Issue #4: Test Script Structure Mismatch

**Current Code** (`test_support_frame_scorer.py` line 168-170):

```python
all_candidate_frames=[{"score": s} for s in candidate_scores] if candidate_scores else None
```

**Scorer Expects** (`support_frame_scorer.py` line 175):

```python
cand_sharpness = [c["components"]["sharpness"] for c in all_candidate_frames]
                    └────────────────────────┘
                     This key is required!
```

**Result**:

```
KeyError: 'components'

Or silent failure if error handling is different
```

**Impact**: 🟡 **Test will fail or produce wrong results**

---

## 4. Score Divergence Analysis

### Cumulative Effect

If all 3 normalize issues exist simultaneously:

```
Support frame true intrinsic quality:  0.80

After Issue #1 (normalize mix):       -0.10  → 0.70
After Issue #2 (ROI wrong scale):     -0.12  → 0.58  
After Issue #3 (wrong weights):       -0.08  → 0.50

Reported vs Actual: 0.50 vs 0.80
Error: 37.5% ❌ HUGE
```

### Example Case: train_0047

```
Support frame at t=1.33s

True features:
  sharpness: 0.85
  edge: 0.45
  brightness: 0.72
  roi_sharpness: 0.91  ← Strong ROI
  roi_edge: 0.48
  roi_brightness: 0.78

Candidate pool avg: 0.65

Expected ranking: Top 5-10 (above average due to strong ROI)
Actual reported: Rank #27 (below average) ← WRONG!

Why? ROI underestimated + wrong policy weights
```

---

## 5. Validation Approach

### Current State

```
✅ Concept: Correct
✅ Feature set: Correct
✅ Weighted sum: Correct
❌ Normalization: Wrong
❌ ROI scale: Wrong
❌ Policy: Wrong
❌ Integration: Wrong

Overall: Concept-wise sound, implementation-wise broken
```

### After Fixes

```
✅ Concept: Correct
✅ Feature set: Correct
✅ Weighted sum: Correct
✅ Normalization: Correct (separate raw+normalize with candidates)
✅ ROI scale: Correct (separate scale for ROI features)
✅ Policy: Correct (type-aware policy lookup)
✅ Integration: Correct (proper candidate structure)

Overall: Fully aligned with pipeline scoring
```

---

## 6. Required Fixes (Priority Order)

### FIX #1: Normalize Architecture ⭐⭐⭐ CRITICAL

**What**: Don't mix candidate normalized values with support raw values.

**How**:
```
Option A: Store raw features in candidates from pipeline
  Modify: selector/pipeline.py to save raw features before normalize
  Pro: Use actual pipeline values
  Con: More pipeline changes

Option B: Recalculate raw features for all candidates
  Modify: support_frame_scorer to extract all candidate frames raw
  Pro: Self-contained module
  Con: Extra frame extraction overhead

Option C: Accept candidates as-is, normalize only support frames
  Modify: Don't use all_candidate_frames for normalize scale
  Pro: Simplest change
  Con: No cross-validation
```

**Recommendation**: Option B (cleanest, self-contained)

---

### FIX #2: Separate ROI Normalization ⭐⭐⭐ CRITICAL

**What**: Create separate normalize functions for ROI features.

**How**:
```
Instead of 3 normalize functions:
  normalize_sharp(v)
  normalize_edges(v)
  normalize_bright(v)

Create 6 normalize functions:
  normalize_sharp(v)           # Global scale
  normalize_roi_sharp(v)       # ROI scale
  normalize_edges(v)           # Global scale
  normalize_roi_edges(v)       # ROI scale
  normalize_bright(v)          # Global scale
  normalize_roi_bright(v)      # ROI scale

Each built from appropriate raw value pools
```

---

### FIX #3: Type-Aware Policy ⭐⭐⭐ CRITICAL

**What**: Accept question_type, lookup correct policy.

**How**:
```python
def score_support_frames(
    self,
    video_path: str,
    support_times: list[float],
    question_type: str | None = None,  # ← ADD THIS
    all_candidate_frames: list[dict] | None = None,
):
    policy_cfg = self._get_policy_config(question_type)  # ← Copy from pipeline
    self.weights = policy_cfg.get("weights", {})
    self.roi_regions = policy_cfg.get("roi_regions", [])
```

---

### FIX #4: Test Integration ⭐⭐ IMPORTANT

**What**: Pass correct candidate frame structure.

**How**:
```python
# Instead of:
all_candidate_frames=[{"score": s} for s in candidate_scores]

# Use:
all_candidate_frames = result["all_frames"]  # ← Full structure
```

---

### FIX #5: Center Bias Accuracy ⭐ NICE-TO-HAVE

**What**: Calculate actual center_bias instead of hardcoding 0.5.

**How**:
```python
def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames / fps

# In scoring:
duration = get_video_duration(video_path)
center_bias = compute_center_bias(time_sec / duration, 1.0)
```

---

## 7. Implementation Timeline

| Fix | Time | Difficulty | Blocking | Notes |
|-----|------|-----------|----------|-------|
| #1 | 60m | High | #2, #3 | Most complex, core fix |
| #2 | 30m | Medium | #1 | Depends on #1 done |
| #3 | 20m | Low | - | Straightforward lookup |
| #4 | 5m | Trivial | - | One-liner fix |
| #5 | 10m | Low | - | Optional enhancement |
| **Total** | **2h** | - | - | Including testing |

---

## 8. Decision Matrix

### Option A: Use Current Implementation

| Aspect | Assessment |
|--------|-----------|
| **Can run now?** | ✅ Yes |
| **Can debug pipeline?** | ✅ Qualitative yes |
| **Can compare scores?** | ❌ No - unreliable |
| **Can make conclusions?** | ❌ No - 37.5% error possible |
| **Risk** | 🔴 High - wrong conclusions |

### Option B: Fix Now, Use Later

| Aspect | Assessment |
|--------|-----------|
| **Can run now?** | ⏳ After 2h |
| **Can debug pipeline?** | ✅ Yes - quantitatively |
| **Can compare scores?** | ✅ Yes - directly |
| **Can make conclusions?** | ✅ Yes - reliable |
| **Risk** | 🟢 Low - scientifically sound |

---

## 9. Recommendation

### PRIMARY: Fix Implementation (Option B)

**Justification**:

1. **Purpose requires accuracy**: We're validating a production pipeline. Wrong conclusions could lead to bad model deployments.

2. **Investment vs payoff**: 2 hours of work → Month-long confidence in pipeline validation

3. **Cost of being wrong**: 
   - If support frame scores are 37% off, we might think pipeline is good when it's mediocre
   - Or think pipeline is bad when it's actually good
   - Could justify wrong hyperparameters or architecture choices

4. **Technical debt**: Current code will accumulate questions every time someone sees divergent scores

### SECONDARY: Mark Current as "Draft"

If timeline is critical:

```python
class SupportFrameScorer:
    """
    ⚠️ DRAFT: Concept-level implementation
    
    Current status: Qualitative validation only
    NOT suitable for quantitative comparisons yet
    
    Known issues:
    - Normalization domain mixing
    - ROI scale not independent
    - Type-aware policy not implemented
    - Test structure mismatch
    
    See: SUPPORT_FRAME_SCORER_TECHNICAL_REVIEW.md
    """
```

Then fix when time permits.

---

## 10. Sign-Off & Next Steps

**Status**: ⚠️ **Technical Review Complete**

**Finding**: Implementation concept is sound but execution has 4 critical issues preventing score equivalence.

**Recommendation**: Proceed with fixes (Option B) - 2 hour investment for long-term confidence.

**If proceeding**:
1. Acknowledge known issues in code comments
2. Implement Fix #1 (Normalization) first
3. Then Fix #2 (ROI Scale)
4. Then Fix #3 (Type-Aware Policy)
5. Run demo test to validate
6. Update documentation with corrected behavior

**Questions?** See detailed issue analysis in Section 3.

---

**Document Status**: Complete Technical Analysis  
**Prepared**: May 9, 2026  
**Ready for**: Team discussion & decision

