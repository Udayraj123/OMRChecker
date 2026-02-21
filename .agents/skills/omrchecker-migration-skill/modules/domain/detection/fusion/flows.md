# Detection Fusion - Execution Flows

**Status**: Advanced Detection System
**Python Reference**: `src/processors/detection/fusion/detection_fusion.py`

---

## Overview

This document details the execution flows for Detection Fusion, showing how multiple detection strategies are combined to produce unified, high-confidence OMR results.

---

## Primary Flow: Fuse Detections

### Entry Point

```python
def fuse_detections(
    self,
    traditional_response: dict,
    ml_blocks_response: list[dict],
    field_id_to_interpretation: dict,
    confidence_threshold: float = 0.85,
) -> tuple[dict, list[dict]]:
```

### Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ START: fuse_detections()                                │
│ Inputs:                                                  │
│  - traditional_response (dict, unused currently)        │
│  - ml_blocks_response (list[dict])                      │
│  - field_id_to_interpretation (dict)                    │
│  - confidence_threshold (float, default 0.85)           │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │ Initialize     │
          │ - fused_response = {}    │
          │ - discrepancies = []     │
          └────────┬───────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │ FOR EACH field_id, trad_interp       │
    │ in field_id_to_interpretation        │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │ Find ML Interpretation   │
    │ ml_interp = _find_ml_interpretation() │
    └──────────┬───────────────┘
               │
               ├─── ml_interp is None? ────┐
               │                             │
               ▼ Yes                         ▼ No
    ┌──────────────────┐         ┌─────────────────────────┐
    │ Use Traditional  │         │ Apply Fusion Strategy   │
    │ fused = trad_interp │      │ strategy_method = FUSION_STRATEGIES[strategy] │
    └──────────┬─────────┘        └──────────┬──────────────┘
               │                              │
               │                              ▼
               │                   ┌──────────────────────────┐
               │                   │ Execute Strategy Method  │
               │                   │ - confidence_weighted    │
               │                   │ - ml_fallback            │
               │                   │ - traditional_primary    │
               │                   └──────────┬───────────────┘
               │                              │
               │                              ▼
               │                   ┌──────────────────────────┐
               │                   │ Returns:                 │
               │                   │ (fused, discrepancy)     │
               │                   └──────────┬───────────────┘
               │                              │
               └──────────────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Store Results        │
                   │ fused_response[field_id] = fused │
                   │ if discrepancy:      │
                   │   discrepancies.append(discrepancy) │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Next Field           │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Log Summary          │
                   │ "Fusion complete:    │
                   │  N fields, M discrepancies" │
                   └──────────┬───────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │ RETURN                       │
               │ (fused_response, discrepancies) │
               └──────────────────────────────┘
```

### Step-by-Step Execution

#### Step 1: Initialize Storage

```python
fused_response = {}
discrepancies = []
```

**Purpose**: Create containers for final results and flagged discrepancies.

---

#### Step 2: Iterate Over Fields

```python
for field_id, trad_interp in field_id_to_interpretation.items():
```

**Example**:
```python
field_id_to_interpretation = {
    "Q1": BubblesFieldInterpretation(matched_bubbles=["A"], confidence=0.88),
    "Q2": BubblesFieldInterpretation(matched_bubbles=["B"], confidence=0.92),
    "Name": OCRFieldInterpretation(text="John Doe", confidence=0.95),
}
```

---

#### Step 3: Find ML Interpretation

```python
ml_interp = self._find_ml_interpretation(field_id, ml_blocks_response)
```

**Current Behavior**: Always returns `None` (placeholder implementation)

**Future Behavior**:
```python
# Parse field_id → block_name → find in ml_blocks_response
ml_interp = {
    "detected_bubbles": ["A"],
    "confidence": 0.92,
    "bbox": [100, 200, 50, 20]
}
```

---

#### Step 4: Handle Missing ML Interpretation

```python
if ml_interp is None:
    fused_response[field_id] = trad_interp
    continue  # Skip to next field
```

**Fast Path**: No ML data available → use traditional detection

---

#### Step 5: Apply Fusion Strategy

```python
strategy_method_name = self.FUSION_STRATEGIES.get(self.fusion_strategy)

if strategy_method_name:
    strategy_method = getattr(self, strategy_method_name)
    fused, discrepancy = strategy_method(
        field_id,
        trad_interp,
        ml_interp,
        confidence_threshold,
    )
```

**Strategy Selection**:
- `"confidence_weighted"` → `_confidence_weighted_fusion()`
- `"ml_fallback"` → `_ml_fallback_fusion()`
- `"traditional_primary"` → `_traditional_primary_fusion()`

---

#### Step 6: Handle Unknown Strategy

```python
else:
    fused = trad_interp
    discrepancy = None
    logger.warning(f"Unknown fusion strategy '{self.fusion_strategy}', using traditional")
```

**Fallback**: Default to traditional detection on configuration error.

---

#### Step 7: Store Results

```python
fused_response[field_id] = fused
if discrepancy:
    discrepancies.append(discrepancy)
```

---

#### Step 8: Log Summary

```python
logger.info(
    f"Detection fusion complete: {len(fused_response)} fields, {len(discrepancies)} discrepancies"
)
```

**Example Output**:
```
INFO: Detection fusion complete: 50 fields, 3 discrepancies
```

---

#### Step 9: Return Results

```python
return fused_response, discrepancies
```

---

## Strategy Flows

### Strategy 1: Confidence-Weighted Fusion

```
┌──────────────────────────────────────────┐
│ START: _confidence_weighted_fusion()     │
│ Inputs: field_id, trad_interp, ml_interp, threshold │
└────────────┬─────────────────────────────┘
             │
             ▼
    ┌────────────────────┐
    │ Extract Confidence │
    │ trad_confidence = trad_interp.overall_confidence_score │
    │ ml_confidence = ml_interp.get("confidence") │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Extract Responses  │
    │ trad_response = trad_interp.matched_bubbles │
    │ ml_response = ml_interp.get("detected_bubbles") │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Check Agreement    │
    │ agree = _responses_agree(trad_response, ml_response) │
    └────────┬───────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │ CASE 1: Both High Confidence?        │
    │ trad_conf > threshold AND            │
    │ ml_conf > threshold                  │
    └────┬─────────────────────────────────┘
         │
         ├─── YES ───┬─── Agree? ──────────┐
         │           │                      │
         │           ▼ YES                  ▼ NO
         │  ┌────────────────┐   ┌─────────────────────┐
         │  │ Return         │   │ Create Discrepancy  │
         │  │ (trad_interp,  │   │ reason: "high_confidence_disagreement" │
         │  │  None)         │   │                     │
         │  └────────────────┘   │ Use Higher Conf     │
         │                       │ return (higher_conf_interp, discrepancy) │
         │                       └─────────────────────┘
         │
         ▼ NO
    ┌──────────────────────────────────────┐
    │ CASE 2: Low Trad, High ML?           │
    │ trad_conf < 0.6 AND                  │
    │ ml_conf > threshold                  │
    └────┬─────────────────────────────────┘
         │
         ├─── YES ───┐
         │           ▼
         │  ┌────────────────────────┐
         │  │ Use ML Fallback        │
         │  │ return (_create_ml_interpretation(ml_interp), None) │
         │  └────────────────────────┘
         │
         ▼ NO
    ┌────────────────┐
    │ DEFAULT        │
    │ return (trad_interp, None) │
    └────────────────┘
```

#### Example Scenarios

**Scenario A: High Confidence Agreement**
```python
# Input
trad_interp.overall_confidence_score = 0.88
ml_interp["confidence"] = 0.92
trad_interp.matched_bubbles = ["A"]
ml_interp["detected_bubbles"] = ["A"]

# Output
fused = trad_interp  # Use traditional (more tested)
discrepancy = None
```

**Scenario B: High Confidence Disagreement**
```python
# Input
trad_interp.overall_confidence_score = 0.88
ml_interp["confidence"] = 0.85
trad_interp.matched_bubbles = ["A"]
ml_interp["detected_bubbles"] = ["C"]

# Output
fused = trad_interp  # Higher confidence (0.88 > 0.85)
discrepancy = {
    "field_id": "Q2",
    "traditional": {"response": ["A"], "confidence": 0.88},
    "ml": {"response": ["C"], "confidence": 0.85},
    "reason": "high_confidence_disagreement"
}
```

**Scenario C: Low Traditional, High ML**
```python
# Input
trad_interp.overall_confidence_score = 0.55
ml_interp["confidence"] = 0.92
trad_interp.matched_bubbles = ["B"]
ml_interp["detected_bubbles"] = ["B"]

# Output
fused = _create_ml_interpretation(ml_interp)  # Use ML
discrepancy = None
logger.debug("Using ML for Q1: trad_conf=0.55, ml_conf=0.92")
```

---

### Strategy 2: ML Fallback Fusion

```
┌──────────────────────────────────────────┐
│ START: _ml_fallback_fusion()             │
│ Inputs: field_id, trad_interp, ml_interp │
└────────────┬─────────────────────────────┘
             │
             ▼
    ┌────────────────────┐
    │ Extract Confidence │
    │ trad_confidence = trad_interp.overall_confidence_score │
    └────────┬───────────┘
             │
             ▼
    ┌──────────────────────────┐
    │ Check Traditional Conf   │
    │ trad_confidence >= 0.6?  │
    └────┬─────────────────────┘
         │
         ├─── YES ───┐
         │           ▼
         │  ┌────────────────┐
         │  │ Use Traditional │
         │  │ return (trad_interp, None) │
         │  └────────────────┘
         │
         ▼ NO
    ┌────────────────────────┐
    │ Use ML Fallback        │
    │ return (_create_ml_interpretation(ml_interp), None) │
    └────────────────────────┘
```

#### Example Scenarios

**Scenario A: High Traditional Confidence**
```python
# Input
trad_interp.overall_confidence_score = 0.85

# Output
fused = trad_interp
discrepancy = None
# ML interpretation ignored
```

**Scenario B: Low Traditional Confidence**
```python
# Input
trad_interp.overall_confidence_score = 0.52

# Output
fused = _create_ml_interpretation(ml_interp)
discrepancy = None
# ML used as fallback
```

---

### Strategy 3: Traditional Primary Fusion

```
┌──────────────────────────────────────────┐
│ START: _traditional_primary_fusion()     │
│ Inputs: field_id, trad_interp, ml_interp │
└────────────┬─────────────────────────────┘
             │
             ▼
    ┌────────────────────┐
    │ Extract Responses  │
    │ trad_response = trad_interp.matched_bubbles │
    │ ml_response = ml_interp.get("detected_bubbles") │
    └────────┬───────────┘
             │
             ▼
    ┌──────────────────────────┐
    │ Check Agreement          │
    │ agree = _responses_agree(trad_response, ml_response) │
    └────┬─────────────────────┘
         │
         ├─── Agree? ──────────┐
         │                      │
         ▼ YES                  ▼ NO
    ┌────────────────┐   ┌─────────────────────┐
    │ Return         │   │ Create Discrepancy  │
    │ (trad_interp,  │   │ reason: "informational_difference" │
    │  None)         │   │                     │
    └────────────────┘   │ Still Use Trad      │
                         │ return (trad_interp, discrepancy) │
                         └─────────────────────┘
```

#### Example Scenarios

**Scenario A: Agreement**
```python
# Input
trad_interp.matched_bubbles = ["A"]
ml_interp["detected_bubbles"] = ["A"]

# Output
fused = trad_interp
discrepancy = None
```

**Scenario B: Disagreement (Informational)**
```python
# Input
trad_interp.matched_bubbles = ["A"]
ml_interp["detected_bubbles"] = ["C"]

# Output
fused = trad_interp  # Always use traditional
discrepancy = {
    "field_id": "Q5",
    "traditional": ["A"],
    "ml": ["C"],
    "reason": "informational_difference"
}
# Logged for ML model evaluation
```

---

## Helper Flow: Find ML Interpretation

### Current Implementation

```
┌──────────────────────────────────────────┐
│ START: _find_ml_interpretation()         │
│ Inputs: field_id, ml_blocks_response     │
└────────────┬─────────────────────────────┘
             │
             ▼
    ┌────────────────────┐
    │ Return None        │
    │ # TODO: Implement  │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ END: None          │
    └────────────────────┘
```

### Future Implementation Flow

```
┌──────────────────────────────────────────┐
│ START: _find_ml_interpretation()         │
│ Inputs: field_id, ml_blocks_response     │
└────────────┬─────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │ Parse Field ID                     │
    │ Extract block name from field_id   │
    │ Example: "Q1" → "MCQ_Block_1"      │
    └────────┬───────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │ FOR EACH ml_block in ml_blocks     │
    └────┬───────────────────────────────┘
         │
         ├─── block name matches? ───┐
         │                            │
         ▼ YES                        ▼ NO
    ┌────────────────────┐       Continue to next block
    │ FOR EACH field     │
    │ in ml_block.fields │
    └────┬───────────────┘
         │
         ├─── field_id matches? ─────┐
         │                            │
         ▼ YES                        ▼ NO
    ┌────────────────────┐       Continue to next field
    │ Return ML Interp   │
    │ {                  │
    │   "detected_bubbles": [...],  │
    │   "confidence": 0.92 │
    │ }                  │
    └────────────────────┘

    ┌────────────────────┐
    │ No Match Found     │
    │ Return None        │
    └────────────────────┘
```

---

## Helper Flow: Check Agreement

### Current Implementation

```
┌──────────────────────────────────────────┐
│ START: _responses_agree()                │
│ Inputs: trad_response, ml_response       │
└────────────┬─────────────────────────────┘
             │
             ▼
    ┌────────────────────┐
    │ Return True        │
    │ # Simplified       │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ END: True          │
    └────────────────────┘
```

### Future Implementation Flow

```
┌──────────────────────────────────────────┐
│ START: _responses_agree()                │
│ Inputs: trad_response, ml_response       │
└────────────┬─────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │ Convert to Sets                    │
    │ trad_set = set(trad_response)      │
    │ ml_set = set(ml_response)          │
    └────────┬───────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │ Calculate Overlap                  │
    │ intersection = trad_set & ml_set   │
    │ union = trad_set | ml_set          │
    │ jaccard = len(intersection) / len(union) │
    └────────┬───────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │ Check Threshold                    │
    │ jaccard >= 0.8?                    │
    └────┬───────────────────────────────┘
         │
         ├─── YES ───┐
         │           ▼
         │  ┌────────────────┐
         │  │ Return True    │
         │  └────────────────┘
         │
         ▼ NO
    ┌────────────────┐
    │ Return False   │
    └────────────────┘
```

#### Example Scenarios

**Scenario A: Perfect Agreement**
```python
trad_response = ["A"]
ml_response = ["A"]
# jaccard = 1/1 = 1.0 → True
```

**Scenario B: Partial Agreement (Multi-Mark)**
```python
trad_response = ["A", "B"]
ml_response = ["A", "C"]
# jaccard = 1/3 = 0.33 → False
```

**Scenario C: High Overlap**
```python
trad_response = ["A", "B", "C"]
ml_response = ["A", "B", "D"]
# jaccard = 2/4 = 0.5 → False (below 0.8 threshold)
```

---

## Helper Flow: Create ML Interpretation

### Current Implementation

```
┌──────────────────────────────────────────┐
│ START: _create_ml_interpretation()       │
│ Inputs: ml_interp (dict)                 │
└────────────┬─────────────────────────────┘
             │
             ▼
    ┌────────────────────┐
    │ Return ml_interp   │
    │ # Simplified       │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ END: ml_interp     │
    └────────────────────┘
```

### Future Implementation Flow

```
┌──────────────────────────────────────────┐
│ START: _create_ml_interpretation()       │
│ Inputs: ml_interp (dict)                 │
└────────────┬─────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │ Create FieldInterpretation Object  │
    │ interp = FieldInterpretation(...)  │
    └────────┬───────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │ Map ML Fields to Interp Fields     │
    │ interp.matched_bubbles = ml_interp["detected_bubbles"] │
    │ interp.overall_confidence_score = ml_interp["confidence"] │
    │ interp.is_attempted = len(detected_bubbles) > 0 │
    └────────┬───────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │ Return FieldInterpretation         │
    └────────────────────────────────────┘
```

---

## Error Handling Flows

### Missing ML Interpretation

```
Field Loop
  ├─> ml_interp = _find_ml_interpretation()
  │
  ├─> if ml_interp is None:
  │     fused_response[field_id] = trad_interp
  │     continue  # Skip fusion, use traditional
```

**Result**: Gracefully falls back to traditional detection.

---

### Unknown Fusion Strategy

```
Strategy Selection
  ├─> strategy_method_name = FUSION_STRATEGIES.get(fusion_strategy)
  │
  ├─> if not strategy_method_name:
  │     logger.warning("Unknown fusion strategy, using traditional")
  │     fused = trad_interp
  │     discrepancy = None
```

**Result**: Falls back to traditional detection, logs warning.

---

### Low Confidence in Both Methods

```
Confidence-Weighted Strategy
  ├─> if trad_conf < 0.6 and ml_conf < 0.6:
  │     # Current: Default to traditional
  │     return (trad_interp, None)
  │
  │   # Future: Flag for manual review
  │     return (trad_interp, {
  │       "reason": "both_low_confidence",
  │       "needs_manual_review": True
  │     })
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Field iteration | O(n) | n = number of fields |
| ML interpretation lookup | O(m) | m = number of ML blocks (currently O(1) - returns None) |
| Strategy execution | O(1) | Pure logic, no loops |
| Discrepancy creation | O(1) | Dictionary creation |

**Total**: O(n × m) ≈ O(n) in practice (m is small, ~5-10 blocks)

---

### Space Complexity

| Data Structure | Size | Notes |
|----------------|------|-------|
| fused_response | O(n) | n = number of fields |
| discrepancies | O(d) | d = number of discrepancies (typically <10% of n) |
| Temporary variables | O(1) | Per-field processing |

**Total**: O(n)

---

## Execution Timeline

### Typical 50-Field Template

```
Total Time: ~2-5ms

Breakdown:
  - Initialization: <1ms
  - Field iteration: ~2-4ms
    - ML lookup per field: ~0.01ms (currently instant, returns None)
    - Strategy execution per field: ~0.05ms
    - Result storage per field: ~0.01ms
  - Logging: <1ms
```

### High-Discrepancy Case (10+ discrepancies)

```
Total Time: ~3-7ms

Breakdown:
  - Initialization: <1ms
  - Field iteration: ~3-6ms
    - Discrepancy creation: +0.1ms per discrepancy
  - Logging: <1ms
```

**Conclusion**: Fusion overhead is negligible (<1% of total detection time).

---

## Next Steps

1. Read `decisions.md` for fusion strategy selection criteria
2. Read `constraints.md` for performance and accuracy limits
3. Read `integration.md` for ReadOMR pipeline integration
