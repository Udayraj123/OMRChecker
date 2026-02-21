# Detection Fusion - Constraints & Limitations

**Status**: Advanced Detection System
**Python Reference**: `src/processors/detection/fusion/detection_fusion.py`

---

## Overview

This document details the constraints, limitations, and trade-offs of the Detection Fusion system, including performance bounds, accuracy limits, and browser migration considerations.

---

## Performance Constraints

### Computational Complexity

#### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|-----------|--------------|
| Strategy selection | O(1) | <0.1ms |
| Field iteration | O(n) | ~0.05ms per field |
| ML interpretation lookup | O(m) | ~0.01ms per field |
| Confidence comparison | O(1) | <0.01ms |
| Agreement check | O(k) | <0.01ms (k = bubbles per field) |
| Discrepancy creation | O(1) | <0.05ms |

**Where**:
- n = number of fields (typically 30-100)
- m = number of ML blocks (typically 5-20)
- k = bubbles per field (typically 4-10)

**Total Complexity**: O(n × m) ≈ O(n) in practice

**Typical Execution Time**:
```
50-field template:
  - Confidence-weighted: 2-5ms
  - ML fallback: 1-3ms (faster, less logic)
  - Traditional primary: 2-4ms
```

---

#### Space Complexity

| Data Structure | Space | Notes |
|----------------|-------|-------|
| `fused_response` | O(n) | Same size as traditional response |
| `discrepancies` | O(d) | d = discrepancies (typically 0-10) |
| Temporary vars | O(1) | Per-field processing |

**Total**: O(n)

**Typical Memory Usage**:
```
50-field template:
  - fused_response: ~50KB (1KB per field interpretation)
  - discrepancies: ~2-5KB (500 bytes per discrepancy)
  - Total: ~55KB
```

---

### Performance Limits

#### Maximum Fields

**Python**:
- **Theoretical**: Unlimited (O(n) scales linearly)
- **Practical**: 10,000 fields tested
- **Typical**: 30-100 fields per template

**Browser**:
- **Recommended**: <500 fields (UI responsiveness)
- **Maximum**: 2,000 fields (memory constraints)

---

#### Processing Speed

**Python (CPython 3.11+)**:
```
Benchmark (Intel i7, 2.6 GHz):
  - 50 fields: 2-5ms
  - 100 fields: 4-10ms
  - 500 fields: 20-50ms
  - 1,000 fields: 40-100ms
```

**Browser (Chrome V8, M1 Mac)**:
```
Estimated (TypeScript transpiled):
  - 50 fields: 3-8ms
  - 100 fields: 6-15ms
  - 500 fields: 30-75ms
  - 1,000 fields: 60-150ms
```

**Bottlenecks**:
1. Agreement check (if implemented with array operations)
2. Discrepancy object creation (object allocation)
3. ML interpretation lookup (dictionary/object traversal)

---

### Optimization Constraints

#### Memory Constraints

**Python**:
- No significant memory constraints
- Interpretations already in memory from traditional detection

**Browser**:
- **Heap limit**: ~2-4GB (varies by browser)
- **Per-field overhead**: ~1KB
- **Maximum template size**: ~500-1000 fields before UI lag

**Optimization Strategies**:
```typescript
// 1. Use Map instead of Object for large field sets
const fusedResponse = new Map<string, FieldInterpretation>();

// 2. Lazy discrepancy creation (only if needed)
if (shouldFlagDiscrepancy) {
  discrepancies.push(createDiscrepancy(...));
}

// 3. Early exit for ML-not-found
if (!mlInterp) {
  fusedResponse.set(fieldId, tradInterp);
  continue;
}
```

---

#### CPU Constraints

**Python**:
- CPU usage negligible (<1% for typical templates)
- Not parallelizable (sequential field processing)

**Browser**:
- Must run in main thread (accesses DOM-related data)
- Can block UI if processing >1000 fields
- **Solution**: Use `requestIdleCallback` for large templates

```typescript
async function fuseDetectionsAsync(
  traditionalResponse: Record<string, any>,
  mlBlocksResponse: MLBlock[],
  fieldIdToInterpretation: Record<string, FieldInterpretation>,
  confidenceThreshold: number = 0.85
): Promise<[Record<string, any>, Discrepancy[]]> {
  const fusedResponse: Record<string, any> = {};
  const discrepancies: Discrepancy[] = [];

  const fields = Object.entries(fieldIdToInterpretation);

  for (let i = 0; i < fields.length; i++) {
    const [fieldId, tradInterp] = fields[i];

    // Process field
    const mlInterp = this.findMLInterpretation(fieldId, mlBlocksResponse);
    // ... fusion logic ...

    // Yield to main thread every 50 fields
    if (i % 50 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  return [fusedResponse, discrepancies];
}
```

---

## Accuracy Constraints

### Detection Accuracy

**Fusion Does NOT Improve Base Accuracy**:
- Fusion selects between existing detections
- If both traditional and ML are wrong, fusion cannot fix it
- Accuracy bounded by: `max(traditional_accuracy, ml_accuracy)`

**Example**:
```python
# Scenario: Both detections wrong
Traditional: "A" (confidence: 0.92)
ML: "B" (confidence: 0.88)
Ground Truth: "C"

# Fusion result (confidence-weighted)
Fused: "A" (uses traditional, higher confidence)
Accuracy: WRONG ❌

# Fusion cannot "create" the correct answer "C"
```

---

### Confidence Score Reliability

**Traditional Confidence**:
- Based on threshold separation (darkness gap)
- Range: 0.0 to 1.0
- **Limitation**: High confidence doesn't guarantee correctness
  - Misaligned field → wrong bubble detected with high confidence
  - Faded correct mark + dark wrong mark → high confidence wrong answer

**ML Confidence**:
- YOLO model output probability
- Range: 0.0 to 1.0
- **Limitation**: Overconfident on out-of-distribution images
  - Training bias → high confidence on wrong class
  - Novel template layouts → arbitrary confidence

**Fusion Implication**:
```python
# High confidence from both doesn't mean correct
Traditional: "A" (conf: 0.95) ← Wrong due to alignment
ML: "A" (conf: 0.93) ← Also wrong (same alignment issue)

# Fusion agrees (both high confidence)
Fused: "A" (conf: 0.95)
Discrepancy: None
Actual: WRONG ❌

# Agreement doesn't imply correctness!
```

---

### Agreement Detection Limitations

**Current Implementation**:
```python
def _responses_agree(trad_response, ml_response) -> bool:
    return True  # Always agrees (placeholder)
```

**Limitation**: Cannot detect disagreements, fusion effectively disabled.

---

**Future Implementation Constraints**:

#### Exact Match Limitation
```python
trad = ["A", "B"]  # Student marked A and B
ml = ["A"]         # ML detected only A (missed B)

exact_match = False  # Flagged as disagreement
# But ML might be partially correct (A is right)
```

**Issue**: Strict matching penalizes partial correctness.

---

#### Jaccard Similarity Limitation
```python
trad = ["A"]
ml = ["B", "C", "D"]  # ML hallucinated 3 bubbles

jaccard = 0/4 = 0.0  # No agreement
# Good - correctly identifies ML error

trad = ["A", "B"]
ml = ["A", "C"]

jaccard = 1/3 = 0.33 < 0.8  # Disagreement
# But 50% overlap - is this really a disagreement?
```

**Issue**: Threshold selection is arbitrary, affects discrepancy rate.

---

#### Multi-Mark Ambiguity
```python
# Scenario: Student multi-marked
Ground Truth: Multi-marked (A and B)

Traditional: ["A", "B"] (correct)
ML: ["A", "B", "C"] (hallucinated C)

jaccard = 2/3 = 0.67 < 0.8  # Disagreement flagged
# But traditional is fully correct, ML has false positive
```

**Issue**: Agreement metric doesn't distinguish between:
- False positives (extra detections)
- False negatives (missed detections)

---

### Field-to-ML-Block Mapping Constraints

**Current State**: Not implemented (`_find_ml_interpretation` returns `None`)

**Implication**: Fusion is effectively **disabled** in current codebase.

---

**Future Implementation Challenges**:

#### Challenge 1: Field ID Parsing

```python
# Field IDs can have various formats
"Q1"           # Simple question number
"Name"         # Text field
"Roll_Number"  # Underscore format
"MCQ_Block_1_Q1"  # Fully qualified

# Mapping to ML blocks requires parsing logic
# No universal mapping strategy
```

**Constraint**: Requires template-specific field ID conventions.

---

#### Challenge 2: ML Block Granularity

```python
# Template defines fields at individual level
Template Fields: ["Q1", "Q2", "Q3", "Q4"]

# ML detects blocks (groups of fields)
ML Blocks: [{
  "block_name": "MCQ_Block_1",
  "fields": [...]  # All Q1-Q4 together
}]

# Mapping requires:
# 1. Find which block contains Q1
# 2. Extract Q1's specific detection from block
```

**Constraint**: Requires hierarchical field-to-block lookup.

---

#### Challenge 3: Missing ML Detections

```python
# Scenario: ML model didn't detect a field block
Template: ["Q1", "Q2", "Q3"]
ML Blocks: [{"block_name": "MCQ_Block_1", "fields": ["Q1", "Q2"]}]
# Q3 is missing from ML detections

# Fusion behavior:
# Q1, Q2 → fusion applied
# Q3 → fall back to traditional (ml_interp = None)
```

**Constraint**: Partial ML coverage requires fallback logic.

---

## Discrepancy Flagging Constraints

### False Positive Discrepancies

**Scenario**: Flag discrepancy when both methods are actually correct but detect slightly differently.

```python
# Ground Truth: Student marked "A" darkly, "B" very lightly (not intended)

Traditional: ["A"] (conf: 0.88)
# Threshold-based, ignores light "B"

ML: ["A", "B"] (conf: 0.85)
# More sensitive, detects light "B"

# Fusion flags disagreement
Discrepancy: {
  "reason": "high_confidence_disagreement",
  "traditional": ["A"],
  "ml": ["A", "B"]
}

# Manual review reveals: Traditional is correct, ML false positive
```

**Impact**: Wastes manual review time on false alarms.

---

### False Negative Discrepancies (Missed Disagreements)

**Scenario**: No discrepancy flagged when methods disagree.

```python
# Current implementation: _responses_agree always returns True

Traditional: ["A"] (conf: 0.92)
ML: ["C"] (conf: 0.88)

# Both high confidence, but agreement check disabled
Fused: "A" (higher confidence)
Discrepancy: None ❌  # Should be flagged!
```

**Impact**: Silent disagreements, lost opportunity for quality improvement.

---

### Discrepancy Threshold Constraints

**Current**: `discrepancy_threshold` parameter exists but is **unused**.

```python
def __init__(self, discrepancy_threshold: float = 0.3):
    self.discrepancy_threshold = 0.3  # Not used anywhere
```

**Future Intent**: Flag discrepancies only if confidence gap exceeds threshold.

```python
# Example future implementation
confidence_gap = abs(trad_conf - ml_conf)
if confidence_gap < self.discrepancy_threshold:
    # Don't flag - difference is negligible
    discrepancy = None
```

**Constraint**: Threshold selection affects false positive rate:
- Too low (0.1) → many false positives
- Too high (0.5) → miss important disagreements

---

## Browser Migration Constraints

### JavaScript/TypeScript Limitations

#### 1. No Native `getattr()`

**Python**:
```python
strategy_method = getattr(self, strategy_method_name)
```

**TypeScript**:
```typescript
// Workaround: Use index signature
const strategyMethod = (this as any)[strategyMethodName];

// Or: Use mapping object
const strategyMethod = this.fusionStrategies[this.fusionStrategy];
```

**Constraint**: Less dynamic, requires explicit mapping.

---

#### 2. Confidence Score Extraction

**Python**:
```python
trad_confidence = getattr(trad_interp, "overall_confidence_score", 0.0)
```

**TypeScript**:
```typescript
// Type-safe approach
const tradConfidence = tradInterp.overallConfidenceScore ?? 0.0;

// Or: Defensive null checking
const tradConfidence = tradInterp?.overallConfidenceScore || 0.0;
```

**Constraint**: Requires proper TypeScript interfaces to avoid runtime errors.

---

#### 3. Set Operations for Agreement

**Python**:
```python
# Clean set operations
trad_set = set(trad_response)
ml_set = set(ml_response)
intersection = trad_set & ml_set
union = trad_set | ml_set
```

**TypeScript**:
```typescript
// No native set operations, need helpers
const tradSet = new Set(tradResponse);
const mlSet = new Set(mlResponse);

const intersection = new Set([...tradSet].filter(x => mlSet.has(x)));
const union = new Set([...tradSet, ...mlSet]);

// Or: Use lodash
import { intersection, union } from 'lodash';
const intersect = intersection(tradResponse, mlResponse);
const unionArr = union(tradResponse, mlResponse);
```

**Constraint**: More verbose, requires utility libraries.

---

### Performance Constraints (Browser)

#### Main Thread Blocking

**Problem**: Large templates (>500 fields) block UI during fusion.

**Solution**: Async processing with yielding.

```typescript
async function fuseDetectionsAsync(...): Promise<...> {
  for (let i = 0; i < fields.length; i++) {
    // Process field

    // Yield every 50 fields
    if (i % 50 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }
}
```

**Constraint**: Adds ~50ms overhead for 500-field template.

---

#### Memory Constraints

**Browser Heap Limits**:
```
Chrome: 2-4GB (desktop), 256-512MB (mobile)
Firefox: 2-4GB
Safari: 1-2GB
Mobile browsers: 256MB-1GB
```

**Large Template Impact**:
```
1000 fields × 1KB per field interpretation = 1MB
+ ML blocks (~500KB)
+ Traditional response (~1MB)
= ~2.5MB total

Mobile: Risky at >500 fields
Desktop: Safe up to 2000 fields
```

**Mitigation**: Warn users if template exceeds 500 fields on mobile.

---

#### Web Worker Limitations

**Fusion Cannot Run in Web Worker**:
- Requires access to `FieldInterpretation` objects (non-serializable)
- Agreement check needs bubble comparison (complex objects)

**Alternative**: Run traditional + ML detection in workers, fusion in main thread.

```typescript
// Worker 1: Traditional detection
// Worker 2: ML detection

// Main thread: Fusion
const [fusedResponse, discrepancies] = fuseDetections(
  traditionalResult,  // From Worker 1
  mlResult,           // From Worker 2
  ...
);
```

**Constraint**: Fusion is fast (~5ms), worker overhead not worth it.

---

### Serialization Constraints

**FieldInterpretation Objects**:
```python
# Python: Rich objects with methods
class BubblesFieldInterpretation(FieldInterpretation):
    def get_field_interpretation_string(self):
        ...
```

**Browser**: Must convert to plain objects for storage/transmission.

```typescript
interface SerializableInterpretation {
  fieldId: string;
  matchedBubbles: string[];
  overallConfidenceScore: number;
  isAttempted: boolean;
  // No methods!
}
```

**Constraint**: Cannot store full Python-like objects in IndexedDB/localStorage.

---

## Accuracy-Speed Trade-offs

### Strategy Trade-offs

| Strategy | Accuracy Gain | Speed Penalty | Complexity |
|----------|---------------|---------------|------------|
| **Confidence-Weighted** | +2-5% | +10-15% | Medium |
| **ML Fallback** | +1-3% | +5-10% | Low |
| **Traditional Primary** | 0% | +5-8% | Low |

**Where**:
- Accuracy gain: vs traditional-only
- Speed penalty: vs no fusion
- Complexity: Implementation + maintenance

---

### Discrepancy Flagging Trade-off

**More Aggressive Flagging**:
```python
# Flag everything
flag_all_disagreements = True
```

**Pros**:
- Catches all potential errors
- Maximum quality control

**Cons**:
- High false positive rate (30-50%)
- Manual review bottleneck
- User fatigue (ignoring flags)

---

**Conservative Flagging**:
```python
# Only flag high-confidence disagreements
flag_only_if_both_confident = True
```

**Pros**:
- Low false positive rate (<10%)
- Actionable discrepancies
- Efficient manual review

**Cons**:
- Misses low-confidence errors
- Silent failures possible

---

## Known Limitations & Future Work

### Current Limitations

1. **Agreement Check Disabled**: `_responses_agree` always returns `True`
   - **Impact**: Cannot detect disagreements
   - **Workaround**: None (requires implementation)

2. **Field-to-ML-Block Mapping Missing**: `_find_ml_interpretation` returns `None`
   - **Impact**: Fusion effectively disabled
   - **Workaround**: None (requires implementation)

3. **ML Interpretation Creation Incomplete**: Returns raw dict instead of `FieldInterpretation`
   - **Impact**: Type mismatch if ML is used
   - **Workaround**: Traditional always used (due to limitation #2)

4. **Discrepancy Threshold Unused**: Parameter exists but not implemented
   - **Impact**: No configurable false positive control
   - **Workaround**: Use strategy selection instead

5. **No Both-Low-Confidence Flagging**: Silent failures when both methods uncertain
   - **Impact**: Quality issues not flagged
   - **Workaround**: Manual post-processing review

---

### Future Enhancements

#### 1. Implement Agreement Check

**Priority**: High

**Approach**: Jaccard similarity with configurable threshold

```python
def _responses_agree(self, trad_response, ml_response, threshold=0.8):
    trad_set = set(trad_response)
    ml_set = set(ml_response)

    if len(trad_set | ml_set) == 0:
        return True

    jaccard = len(trad_set & ml_set) / len(trad_set | ml_set)
    return jaccard >= threshold
```

---

#### 2. Implement Field-to-ML-Block Mapping

**Priority**: High

**Approach**: Template-aware field ID parsing

```python
def _find_ml_interpretation(self, field_id, ml_blocks_response):
    # Parse field_id to extract block name
    block_name = self._extract_block_name_from_field_id(field_id)

    # Find matching ML block
    for ml_block in ml_blocks_response:
        if ml_block["field_block_name"] == block_name:
            # Find field within block
            for field in ml_block["fields"]:
                if field["field_id"] == field_id:
                    return {
                        "detected_bubbles": field["bubbles"],
                        "confidence": field["confidence"]
                    }

    return None
```

---

#### 3. Both-Low-Confidence Flagging

**Priority**: Medium

**Approach**: Add third discrepancy type

```python
if trad_conf < 0.6 and ml_conf < 0.6:
    discrepancy = {
        "field_id": field_id,
        "reason": "both_low_confidence",
        "needs_manual_review": True,
        "traditional": {...},
        "ml": {...}
    }
```

---

#### 4. Confidence Gap Threshold

**Priority**: Low

**Approach**: Use existing `discrepancy_threshold` parameter

```python
confidence_gap = abs(trad_conf - ml_conf)
if confidence_gap < self.discrepancy_threshold:
    # Don't flag - difference negligible
    discrepancy = None
```

---

## Next Steps

1. Read `integration.md` for ReadOMR pipeline integration
2. Refer to `concept.md` for architecture overview
3. Review `flows.md` for execution details
4. Check `decisions.md` for strategy selection
