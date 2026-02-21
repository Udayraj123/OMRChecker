# Detection Fusion - Decision Logic

**Status**: Advanced Detection System
**Python Reference**: `src/processors/detection/fusion/detection_fusion.py`

---

## Overview

This document explains the decision-making logic in Detection Fusion, covering fusion strategy selection, confidence-based decisions, and discrepancy flagging criteria.

---

## Fusion Strategy Selection

### When to Use Each Strategy

#### 1. Confidence-Weighted Fusion (Default)

**Use When**:
- You need the highest accuracy possible
- Discrepancies must be flagged for review
- Both traditional and ML detections are available
- Audit trail is important (knowing when/why ML was used)

**Don't Use When**:
- Processing speed is critical (adds ~10% overhead for discrepancy checks)
- ML model is untrusted (might override traditional incorrectly)
- You want fully deterministic results

**Decision Tree**:
```
Choose Confidence-Weighted IF:
  ├─> Accuracy > Speed? YES
  ├─> Need audit trail? YES
  ├─> ML model is reliable? YES
  └─> Result: Use confidence_weighted
```

**Example Configuration**:
```python
# Production environment with high accuracy requirements
fusion = DetectionFusion(
    fusion_strategy="confidence_weighted",
    discrepancy_threshold=0.3  # Flag if confidence gap > 0.3
)
```

---

#### 2. ML Fallback Fusion

**Use When**:
- Traditional detection is primary method
- ML is only for edge cases (low confidence)
- Processing speed matters (ML rarely called)
- You trust traditional detection for typical cases

**Don't Use When**:
- ML model is more accurate than traditional (use confidence_weighted instead)
- You need to know all disagreements (this strategy doesn't flag agreement cases)
- Traditional detection is known to be unreliable

**Decision Tree**:
```
Choose ML Fallback IF:
  ├─> Traditional generally reliable? YES
  ├─> Speed > Accuracy? YES
  ├─> ML only for safety net? YES
  └─> Result: Use ml_fallback
```

**Example Configuration**:
```python
# High-volume processing with quality sheets
fusion = DetectionFusion(
    fusion_strategy="ml_fallback"
)
# ML only runs when traditional confidence < 0.6
```

---

#### 3. Traditional Primary Fusion

**Use When**:
- Debugging ML models (want to see all differences)
- Collecting ML comparison data for model improvement
- You must use traditional results but want ML insights
- Regulatory/compliance requires deterministic traditional detection

**Don't Use When**:
- You actually want to use ML results (defeats the purpose)
- Not interested in ML comparison data

**Decision Tree**:
```
Choose Traditional Primary IF:
  ├─> Debugging ML model? YES
  OR
  ├─> Collecting ML training data? YES
  OR
  ├─> Must use traditional only? YES
  └─> Result: Use traditional_primary
```

**Example Configuration**:
```python
# ML model evaluation mode
fusion = DetectionFusion(
    fusion_strategy="traditional_primary"
)
# Results always use traditional, but ML differences logged
```

---

### Strategy Comparison Matrix

| Criteria | Confidence-Weighted | ML Fallback | Traditional Primary |
|----------|---------------------|-------------|---------------------|
| **Accuracy** | Highest | High | Medium-High |
| **Speed** | Medium | Fast | Fastest |
| **ML Usage** | When higher confidence | When trad low confidence | Never (informational only) |
| **Discrepancies Flagged** | Disagreements | None | All differences (informational) |
| **Audit Trail** | Detailed | Minimal | Full comparison |
| **Deterministic** | No | Mostly | Yes |
| **Use Case** | Production (accuracy) | Production (speed) | Debugging/Evaluation |

---

## Confidence-Based Decisions

### Confidence Thresholds

```python
HIGH_CONFIDENCE = 0.85  # Both methods confident
LOW_CONFIDENCE = 0.6    # Fallback threshold
```

**Rationale**:
- **0.85**: High confidence threshold
  - Based on empirical testing showing >95% accuracy above this value
  - Reduces false positives from low-quality detections

- **0.6**: Low confidence threshold
  - Indicates detection uncertainty (alignment issues, faded marks)
  - Triggers ML fallback in `ml_fallback` strategy

---

### Confidence-Weighted Fusion Decision Matrix

| Traditional Conf | ML Conf | Agreement | Decision | Discrepancy |
|-----------------|---------|-----------|----------|-------------|
| >0.85 | >0.85 | Yes | Use Traditional | No |
| >0.85 | >0.85 | No | Use Higher Conf | Yes (high_conf_disagreement) |
| >0.85 | <0.85 | Any | Use Traditional | No |
| <0.6 | >0.85 | Any | Use ML | No |
| <0.6 | <0.6 | Any | Use Traditional | No (future: flag both_low_conf) |
| 0.6-0.85 | 0.6-0.85 | Any | Use Traditional | No |

---

### Decision Examples

#### Example 1: High Confidence Agreement

```python
# Input
field_id = "Q1"
trad_conf = 0.88
ml_conf = 0.92
trad_response = ["A"]
ml_response = ["A"]

# Decision Logic
both_high_confidence = (0.88 > 0.85) and (0.92 > 0.85)  # True
responses_agree = True

# Output
decision = "Use Traditional"  # More tested, agreement confirms accuracy
discrepancy = None
rationale = "Both methods agree with high confidence"
```

---

#### Example 2: High Confidence Disagreement

```python
# Input
field_id = "Q5"
trad_conf = 0.89
ml_conf = 0.86
trad_response = ["C"]
ml_response = ["D"]

# Decision Logic
both_high_confidence = (0.89 > 0.85) and (0.86 > 0.85)  # True
responses_agree = False

# Output
decision = "Use Traditional (higher confidence: 0.89 > 0.86)"
discrepancy = {
    "field_id": "Q5",
    "traditional": {"response": ["C"], "confidence": 0.89},
    "ml": {"response": ["D"], "confidence": 0.86},
    "reason": "high_confidence_disagreement"
}
rationale = "Flag for manual review - both methods confident but disagree"
```

---

#### Example 3: Low Traditional, High ML

```python
# Input
field_id = "Q8"
trad_conf = 0.52
ml_conf = 0.91
trad_response = ["B"]
ml_response = ["B"]

# Decision Logic
low_trad = 0.52 < 0.6  # True
high_ml = 0.91 > 0.85  # True

# Output
decision = "Use ML"
discrepancy = None
rationale = "Traditional low confidence (alignment issue?), ML robust"
logger.debug("Using ML for Q8: trad_conf=0.52, ml_conf=0.91")
```

---

#### Example 4: Medium Confidence

```python
# Input
field_id = "Q10"
trad_conf = 0.75
ml_conf = 0.78
trad_response = ["A"]
ml_response = ["C"]

# Decision Logic
both_high_confidence = False  # Both < 0.85
low_trad_high_ml = False      # trad >= 0.6

# Output
decision = "Use Traditional (default path)"
discrepancy = None
rationale = "Medium confidence, traditional preferred as default"
```

---

#### Example 5: Both Low Confidence

```python
# Input
field_id = "Q12"
trad_conf = 0.45
ml_conf = 0.51
trad_response = ["A", "B"]  # Multi-marked
ml_response = ["A", "C"]

# Decision Logic
both_low = (0.45 < 0.6) and (0.51 < 0.6)  # True

# Output (Current)
decision = "Use Traditional (default)"
discrepancy = None

# Output (Future Enhancement)
decision = "Use Traditional (flagged for review)"
discrepancy = {
    "field_id": "Q12",
    "reason": "both_low_confidence",
    "needs_manual_review": True,
    "traditional": {"response": ["A", "B"], "confidence": 0.45},
    "ml": {"response": ["A", "C"], "confidence": 0.51}
}
rationale = "Both methods uncertain - likely scanning/alignment issue"
```

---

## ML Fallback Fusion Decision Matrix

| Traditional Conf | Decision | ML Used? |
|-----------------|----------|----------|
| ≥ 0.6 | Use Traditional | No |
| < 0.6 | Use ML | Yes |

**Simple Logic**:
```python
if traditional_confidence >= 0.6:
    return traditional
else:
    return ml  # Fallback
```

**No Discrepancies**: This strategy never creates discrepancies.

---

### Decision Examples

#### Example 1: High Traditional Confidence

```python
# Input
trad_conf = 0.92
ml_conf = 0.88

# Decision
decision = "Use Traditional"
ml_used = False
rationale = "Traditional confidence sufficient (≥ 0.6)"
```

---

#### Example 2: Low Traditional Confidence

```python
# Input
trad_conf = 0.48
ml_conf = 0.91

# Decision
decision = "Use ML"
ml_used = True
rationale = "Traditional confidence low (< 0.6), ML fallback activated"
```

---

#### Example 3: Borderline Case

```python
# Input
trad_conf = 0.60  # Exactly at threshold
ml_conf = 0.85

# Decision
decision = "Use Traditional"
ml_used = False
rationale = "Traditional confidence at threshold (≥ 0.6)"
```

---

## Traditional Primary Fusion Decision Matrix

| Responses Agree? | Decision | Discrepancy |
|-----------------|----------|-------------|
| Yes | Use Traditional | No |
| No | Use Traditional | Yes (informational_difference) |

**Always Traditional**: Results always use traditional detection.

**Discrepancies**: Only for informational purposes (ML model evaluation).

---

### Decision Examples

#### Example 1: Agreement

```python
# Input
trad_response = ["A"]
ml_response = ["A"]

# Decision
decision = "Use Traditional"
discrepancy = None
rationale = "Agreement - no logging needed"
```

---

#### Example 2: Disagreement (Informational)

```python
# Input
trad_response = ["A"]
ml_response = ["C"]

# Decision
decision = "Use Traditional"
discrepancy = {
    "field_id": "Q7",
    "traditional": ["A"],
    "ml": ["C"],
    "reason": "informational_difference"
}
rationale = "Log ML difference for model evaluation"
```

---

## Discrepancy Flagging Criteria

### When to Flag

**Confidence-Weighted Strategy**:
```python
flag_discrepancy = (
    traditional_confidence > 0.85 and
    ml_confidence > 0.85 and
    not responses_agree(traditional, ml)
)
```

**Traditional Primary Strategy**:
```python
flag_discrepancy = not responses_agree(traditional, ml)
```

**ML Fallback Strategy**:
```python
flag_discrepancy = False  # Never flags
```

---

### Discrepancy Types

#### 1. High Confidence Disagreement

**Trigger**:
```python
both_high_confidence and disagreement
```

**Severity**: High (needs manual review)

**Example**:
```python
{
    "field_id": "Q5",
    "traditional": {"response": ["C"], "confidence": 0.89},
    "ml": {"response": ["D"], "confidence": 0.86},
    "reason": "high_confidence_disagreement",
    "severity": "high"
}
```

**Action**: Manual review required, possible scanning issue or template error.

---

#### 2. Informational Difference

**Trigger**:
```python
traditional_primary_strategy and disagreement
```

**Severity**: Low (informational only)

**Example**:
```python
{
    "field_id": "Q7",
    "traditional": ["A"],
    "ml": ["C"],
    "reason": "informational_difference",
    "severity": "low"
}
```

**Action**: Log for ML model improvement, no review needed.

---

#### 3. Both Low Confidence (Future)

**Trigger**:
```python
traditional_confidence < 0.6 and ml_confidence < 0.6
```

**Severity**: Medium (needs attention)

**Example**:
```python
{
    "field_id": "Q12",
    "traditional": {"response": ["A", "B"], "confidence": 0.45},
    "ml": {"response": ["A", "C"], "confidence": 0.51},
    "reason": "both_low_confidence",
    "needs_manual_review": True,
    "severity": "medium"
}
```

**Action**: Investigate image quality, alignment, or template issues.

---

## Agreement Detection Logic

### Current Implementation

```python
def _responses_agree(trad_response, ml_response) -> bool:
    # Simplified: always returns True
    return True
```

**Limitation**: Cannot detect actual disagreements.

---

### Future Implementation

#### Exact Match Strategy

```python
def _responses_agree_exact(trad_response, ml_response) -> bool:
    """Exact match required."""
    return set(trad_response) == set(ml_response)
```

**Example**:
```python
trad = ["A"]
ml = ["A"]
agree = True  # Exact match

trad = ["A", "B"]
ml = ["A", "C"]
agree = False  # Different
```

---

#### Jaccard Similarity Strategy

```python
def _responses_agree_jaccard(trad_response, ml_response, threshold=0.8) -> bool:
    """Agreement if Jaccard similarity ≥ threshold."""
    trad_set = set(trad_response)
    ml_set = set(ml_response)

    intersection = trad_set & ml_set
    union = trad_set | ml_set

    if len(union) == 0:
        return True  # Both empty

    jaccard = len(intersection) / len(union)
    return jaccard >= threshold
```

**Example**:
```python
trad = ["A", "B"]
ml = ["A", "B"]
jaccard = 2/2 = 1.0 ≥ 0.8 → agree = True

trad = ["A", "B"]
ml = ["A", "C"]
jaccard = 1/3 = 0.33 < 0.8 → agree = False

trad = ["A", "B", "C"]
ml = ["A", "B", "D"]
jaccard = 2/4 = 0.5 < 0.8 → agree = False
```

---

#### Partial Credit Strategy (Multi-Mark)

```python
def _responses_agree_partial(trad_response, ml_response, min_overlap=0.5) -> bool:
    """Agreement if overlap ≥ min_overlap (for multi-mark fields)."""
    trad_set = set(trad_response)
    ml_set = set(ml_response)

    intersection = trad_set & ml_set
    min_size = min(len(trad_set), len(ml_set))

    if min_size == 0:
        return len(trad_set) == len(ml_set)  # Both empty

    overlap = len(intersection) / min_size
    return overlap >= min_overlap
```

**Example**:
```python
trad = ["A", "B", "C"]
ml = ["A", "B"]
overlap = 2/2 = 1.0 ≥ 0.5 → agree = True

trad = ["A"]
ml = ["B", "C"]
overlap = 0/1 = 0.0 < 0.5 → agree = False
```

---

## Strategy Selection Flowchart

```
┌─────────────────────────────────────┐
│ Need to choose fusion strategy?    │
└───────────┬─────────────────────────┘
            │
            ▼
   ┌────────────────────┐
   │ What's priority?   │
   └────┬───────────────┘
        │
        ├─── Accuracy ─────────────────┐
        │                               ▼
        │                    ┌──────────────────────┐
        │                    │ Confidence-Weighted  │
        │                    │ - Best accuracy      │
        │                    │ - Audit trail        │
        │                    │ - Flag disagreements │
        │                    └──────────────────────┘
        │
        ├─── Speed ────────────────────┐
        │                               ▼
        │                    ┌──────────────────────┐
        │                    │ ML Fallback          │
        │                    │ - Fast               │
        │                    │ - ML safety net      │
        │                    │ - Minimal overhead   │
        │                    └──────────────────────┘
        │
        └─── Debugging/Eval ──────────┐
                                       ▼
                            ┌──────────────────────┐
                            │ Traditional Primary  │
                            │ - Deterministic      │
                            │ - ML comparison data │
                            │ - Model evaluation   │
                            └──────────────────────┘
```

---

## Common Decision Scenarios

### Scenario 1: Production Environment (High Accuracy)

**Requirements**:
- Maximum accuracy
- Audit trail for discrepancies
- Trust both traditional and ML

**Decision**:
```python
fusion = DetectionFusion(
    fusion_strategy="confidence_weighted",
    discrepancy_threshold=0.3
)
```

---

### Scenario 2: High-Volume Processing (Speed Priority)

**Requirements**:
- Fast processing
- ML only for edge cases
- Traditional generally reliable

**Decision**:
```python
fusion = DetectionFusion(
    fusion_strategy="ml_fallback"
)
```

---

### Scenario 3: ML Model Evaluation

**Requirements**:
- Compare ML vs traditional
- Collect training data
- Must use traditional results

**Decision**:
```python
fusion = DetectionFusion(
    fusion_strategy="traditional_primary"
)
```

---

### Scenario 4: Untrusted ML Model

**Requirements**:
- ML model still in beta
- Don't want ML to override traditional
- But want to see ML suggestions

**Decision**:
```python
fusion = DetectionFusion(
    fusion_strategy="traditional_primary"
)
# Review discrepancies to evaluate ML accuracy
```

---

### Scenario 5: Critical Application (Zero Tolerance)

**Requirements**:
- Maximum accuracy
- Flag any uncertainty
- Manual review acceptable

**Decision**:
```python
fusion = DetectionFusion(
    fusion_strategy="confidence_weighted",
    discrepancy_threshold=0.1  # Very strict
)

# Future enhancement: Add both_low_confidence flagging
# This would flag any field where both methods have <0.85 confidence
```

---

## Next Steps

1. Read `constraints.md` for performance and accuracy limits
2. Read `integration.md` for ReadOMR pipeline integration
3. Refer to `flows.md` for detailed execution flows
