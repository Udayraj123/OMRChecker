# Detection Fusion - Core Concepts

**Status**: Advanced Detection System
**Python Reference**: `src/processors/detection/fusion/detection_fusion.py`
**Browser Compatibility**: TypeScript/JavaScript compatible

---

## Overview

**Detection Fusion** is a hybrid detection system that combines results from multiple detection strategies (traditional threshold-based and ML-based) to produce more accurate and reliable OMR detection results. It uses confidence-weighted voting and discrepancy flagging to merge detections intelligently.

This module enables OMRChecker to leverage the strengths of both approaches:
- **Traditional detection**: Fast, deterministic, well-tested
- **ML detection**: Robust to alignment issues, handles edge cases

---

## What is Detection Fusion?

Detection Fusion is a **meta-detector** that:
- Receives detection results from multiple sources (traditional + ML)
- Compares responses field-by-field
- Applies fusion strategies to select the best result
- Flags discrepancies for manual review
- Returns a unified, high-confidence response

**Key Insight**: Detection Fusion doesn't perform detection itself—it **combines** results from other detectors using configurable strategies.

---

## Why is Detection Fusion Needed?

### Problem: Competing Detection Methods

```python
# Traditional detection might fail on:
- Misaligned sheets (despite alignment)
- Faded marks or poor scan quality
- Unusual bubble shapes

# ML detection might fail on:
- Images outside training distribution
- Novel template layouts
- Edge cases not in training data

# Fusion solves this by combining both:
- Use traditional when both agree (fastest path)
- Use ML when traditional has low confidence
- Flag disagreements for review
```

### Use Case Example

```python
# Scenario: Slightly misaligned bubble field

Traditional Detection:
  Q1: "B" (confidence: 0.55) ← Low confidence due to alignment

ML Detection:
  Q1: "B" (confidence: 0.92) ← ML robust to small shifts

Fusion Result:
  Q1: "B" (confidence: 0.92) ← Uses ML due to higher confidence
  discrepancies: [] ← No flag needed (both agree)

---

# Scenario: Disagreement case

Traditional Detection:
  Q2: "A" (confidence: 0.88) ← High confidence

ML Detection:
  Q2: "C" (confidence: 0.85) ← Also high confidence

Fusion Result:
  Q2: "A" (confidence: 0.88) ← Uses higher confidence
  discrepancies: [{
    field_id: "Q2",
    reason: "high_confidence_disagreement",
    traditional: {response: "A", confidence: 0.88},
    ml: {response: "C", confidence: 0.85}
  }] ← Flagged for manual review
```

---

## Core Architecture

### Class Structure

```python
# src/processors/detection/fusion/detection_fusion.py

class DetectionFusion:
    """Fuses ML-based and traditional bubble detection results."""

    # Strategy pattern mapping
    FUSION_STRATEGIES = {
        "confidence_weighted": "_confidence_weighted_fusion",
        "ml_fallback": "_ml_fallback_fusion",
        "traditional_primary": "_traditional_primary_fusion",
    }

    def __init__(
        self,
        fusion_strategy: str = "confidence_weighted",
        discrepancy_threshold: float = 0.3,
    ):
        self.fusion_strategy = fusion_strategy
        self.discrepancy_threshold = discrepancy_threshold
```

### Primary Method

```python
def fuse_detections(
    self,
    traditional_response: dict,
    ml_blocks_response: list[dict],
    field_id_to_interpretation: dict,
    confidence_threshold: float = 0.85,
) -> tuple[dict, list[dict]]:
    """Main fusion logic.

    Returns:
        (fused_response, discrepancies)
    """
```

---

## Fusion Strategies

Detection Fusion implements **three fusion strategies** using the Strategy pattern:

### 1. Confidence-Weighted Fusion (Default)

**Strategy**: Use the detection with higher confidence, flag significant disagreements.

**Logic**:
```python
if both_high_confidence and agree:
    return traditional  # More tested, faster

if both_high_confidence and disagree:
    flag_for_review()
    return higher_confidence

if traditional_low_confidence and ml_high_confidence:
    return ml  # ML fallback

else:
    return traditional  # Default path
```

**Use Case**: Production environments where accuracy and auditability are critical.

**Configuration**:
```python
fusion = DetectionFusion(
    fusion_strategy="confidence_weighted",
    discrepancy_threshold=0.3  # Min confidence gap to flag
)
```

---

### 2. ML Fallback Fusion

**Strategy**: Use traditional detection by default, only use ML when traditional has low confidence.

**Logic**:
```python
if traditional_confidence >= 0.6:
    return traditional

else:
    return ml  # Fallback to ML
```

**Use Case**: Fast processing with ML safety net for edge cases.

**Configuration**:
```python
fusion = DetectionFusion(
    fusion_strategy="ml_fallback"
)
```

**Advantages**:
- Fastest strategy (ML only runs when needed)
- Minimal computational overhead
- Good balance of speed and accuracy

---

### 3. Traditional Primary Fusion

**Strategy**: Always use traditional detection, but log ML differences for analysis.

**Logic**:
```python
if traditional != ml:
    log_discrepancy(informational=True)

return traditional  # Always traditional
```

**Use Case**:
- Debugging and ML model evaluation
- Collecting training data for ML improvement
- Conservative deployments prioritizing deterministic behavior

**Configuration**:
```python
fusion = DetectionFusion(
    fusion_strategy="traditional_primary"
)
```

**Advantages**:
- Fully deterministic (no ML influence on results)
- Collects ML comparison data for model improvement
- Zero risk of ML false positives

---

## Key Concepts

### 1. Confidence Scores

Each detection method provides confidence scores:

```python
# Traditional detection
traditional_interpretation.overall_confidence_score = 0.88

# ML detection
ml_interpretation = {
    "confidence": 0.92,
    "detected_bubbles": ["B"],
    ...
}
```

**Confidence Calculation** (Traditional):
- Based on threshold separation (larger gap → higher confidence)
- Influenced by multi-marking (multi-marks → lower confidence)
- Range: 0.0 (no confidence) to 1.0 (perfect confidence)

**Confidence Calculation** (ML):
- YOLO model confidence scores
- Based on prediction probability
- Range: 0.0 to 1.0

---

### 2. Agreement Detection

```python
def _responses_agree(trad_response, ml_response) -> bool:
    """Check if traditional and ML responses agree.

    Currently simplified (always returns True).
    Full implementation would compare bubble positions and states.
    """
```

**Future Implementation**:
```python
# Compare detected bubbles
trad_bubbles = set(trad_response)  # {"A", "B"}
ml_bubbles = set(ml_response)      # {"A", "B"}

# Allow small differences (e.g., 1 bubble difference for multi-choice)
agreement_threshold = 0.8
overlap = len(trad_bubbles & ml_bubbles) / max(len(trad_bubbles), len(ml_bubbles))

return overlap >= agreement_threshold
```

---

### 3. Discrepancy Flagging

Discrepancies are flagged when:
1. Both methods have high confidence (>0.85)
2. But they disagree on the response
3. Or confidence gap exceeds threshold

**Discrepancy Structure**:
```python
{
    "field_id": "Q2",
    "traditional": {
        "response": ["A"],
        "confidence": 0.88
    },
    "ml": {
        "response": ["C"],
        "confidence": 0.85
    },
    "reason": "high_confidence_disagreement"
}
```

**Discrepancy Types**:
- `"high_confidence_disagreement"`: Both confident but different
- `"informational_difference"`: Traditional primary mode logging

---

### 4. Field-to-Bubble Mapping (TODO)

**Current State**: Simplified placeholder implementation

```python
def _find_ml_interpretation(field_id, ml_blocks_response) -> dict | None:
    # TODO: Implement proper field-to-bubble mapping
    return None  # Always returns None (fusion skipped)
```

**Future Implementation**:
```python
def _find_ml_interpretation(field_id, ml_blocks_response):
    """Map field_id to ML-detected bubbles.

    Strategy:
    1. Parse field_id to extract block name (e.g., "Q1" → "MCQ_Block_1")
    2. Find matching ML block in ml_blocks_response
    3. Extract bubble detections for that field
    4. Return formatted interpretation
    """
    # Extract block name from field_id
    block_name = self._extract_block_name(field_id)

    # Find ML block
    for ml_block in ml_blocks_response:
        if ml_block["field_block_name"] == block_name:
            # Find bubbles for this field
            for field in ml_block["fields"]:
                if field["field_id"] == field_id:
                    return {
                        "detected_bubbles": field["bubbles"],
                        "confidence": field["confidence"]
                    }

    return None
```

---

### 5. ML Interpretation Object Creation

**Current State**: Simplified placeholder

```python
def _create_ml_interpretation(ml_interp: dict) -> dict:
    # TODO: Create proper FieldInterpretation-compatible object
    return ml_interp
```

**Future Implementation**:
```python
def _create_ml_interpretation(ml_interp: dict) -> FieldInterpretation:
    """Create FieldInterpretation from ML results.

    Converts ML detection format to standard FieldInterpretation
    format compatible with BubblesFieldInterpretation.
    """
    from src.processors.detection.base.interpretation import FieldInterpretation

    # Create mock FieldInterpretation
    interpretation = FieldInterpretation(...)
    interpretation.matched_bubbles = ml_interp["detected_bubbles"]
    interpretation.overall_confidence_score = ml_interp["confidence"]
    interpretation.is_attempted = len(ml_interp["detected_bubbles"]) > 0

    return interpretation
```

---

## Data Flow

### Input

```python
# From traditional detection (ReadOMRProcessor)
traditional_response = {
    # Unused in current implementation (future enhancement)
}

# From ML field block detection
ml_blocks_response = [
    {
        "field_block_name": "MCQ_Block_1",
        "fields": [
            {
                "field_id": "Q1",
                "bubbles": ["A"],
                "confidence": 0.92
            }
        ]
    }
]

# From traditional interpretation
field_id_to_interpretation = {
    "Q1": BubblesFieldInterpretation(
        matched_bubbles=["A"],
        overall_confidence_score=0.88,
        ...
    )
}

# Fusion parameters
confidence_threshold = 0.85  # High confidence threshold
```

### Processing Steps

```
1. For each field in field_id_to_interpretation:
    ├─> Get traditional interpretation
    ├─> Find ML interpretation (via _find_ml_interpretation)
    ├─> If ML not found: use traditional
    ├─> If both found:
    │   ├─> Apply fusion strategy
    │   ├─> Check for discrepancies
    │   └─> Return fused result
    └─> Store in fused_response

2. Return (fused_response, discrepancies)
```

### Output

```python
# Fused response (same format as traditional)
fused_response = {
    "Q1": BubblesFieldInterpretation(...),
    "Q2": BubblesFieldInterpretation(...),
    ...
}

# Discrepancies list
discrepancies = [
    {
        "field_id": "Q2",
        "traditional": {...},
        "ml": {...},
        "reason": "high_confidence_disagreement"
    }
]
```

---

## Configuration Schema

### Fusion Configuration

```python
# In DetectionFusion initialization
fusion = DetectionFusion(
    fusion_strategy="confidence_weighted",  # or "ml_fallback", "traditional_primary"
    discrepancy_threshold=0.3,  # Min confidence gap to flag (unused in current impl)
)
```

### Confidence Thresholds

```python
# High confidence threshold (used in confidence_weighted strategy)
confidence_threshold = 0.85

# Low confidence threshold (used in ml_fallback strategy)
low_confidence_threshold = 0.6
```

### Strategy Selection Guide

| Strategy | Speed | Accuracy | Auditability | Use Case |
|----------|-------|----------|--------------|----------|
| **confidence_weighted** | Medium | Highest | High | Production |
| **ml_fallback** | Fast | High | Medium | High-volume processing |
| **traditional_primary** | Fastest | Medium | Highest | Debugging, ML evaluation |

---

## Edge Cases & Constraints

### 1. ML Interpretation Not Found

```python
if ml_interp is None:
    # No ML detection, use traditional
    fused_response[field_id] = trad_interp
    continue
```

**Cause**: Field not in ML detection results (e.g., ML model didn't detect this field block)

**Resolution**: Falls back to traditional detection

---

### 2. Unknown Fusion Strategy

```python
if strategy_method_name not in FUSION_STRATEGIES:
    logger.warning(f"Unknown strategy '{self.fusion_strategy}', using traditional")
    fused = trad_interp
```

**Cause**: Invalid strategy name in configuration

**Resolution**: Defaults to traditional detection, logs warning

---

### 3. Both Detections Low Confidence

```python
# Scenario: Both traditional and ML have low confidence (<0.6)
# Current behavior: confidence_weighted uses traditional
# Future enhancement: Could flag as "needs_manual_review"
```

---

### 4. Agreement Check Placeholder

**Current State**: `_responses_agree()` always returns `True`

**Impact**:
- No discrepancies flagged for actual disagreements
- Fusion based only on confidence scores

**Future Fix**: Implement actual bubble-level comparison

---

### 5. Field ID to ML Block Mapping Missing

**Current State**: `_find_ml_interpretation()` always returns `None`

**Impact**:
- Fusion effectively disabled (always uses traditional)
- ML detection results not utilized

**Future Fix**: Implement field-to-block mapping logic

---

## Performance Considerations

### Computational Cost

| Operation | Complexity | Time (typical) |
|-----------|-----------|----------------|
| Strategy selection | O(1) | <1ms |
| Field iteration | O(n_fields) | ~1ms per field |
| Confidence comparison | O(1) | <0.1ms |
| ML interpretation lookup | O(n_ml_blocks) | ~1ms per field |
| Discrepancy creation | O(1) | <0.1ms |

**Total**: ~2-5ms per field (negligible overhead)

---

### Memory Usage

```python
# Fused response (same size as traditional)
fused_response: dict[str, FieldInterpretation]  # ~1KB per field

# Discrepancies list
discrepancies: list[dict]  # ~500 bytes per discrepancy

# Total: <100KB for typical 50-field template
```

---

### Optimization Strategies

1. **Early Exit for ML-Not-Found**
```python
if ml_interp is None:
    fused_response[field_id] = trad_interp
    continue  # Skip fusion logic
```

2. **Strategy Pattern (Avoid If-Else Chains)**
```python
# Fast method lookup via dictionary
strategy_method = getattr(self, FUSION_STRATEGIES[self.fusion_strategy])
```

3. **Lazy ML Interpretation Lookup**
```python
# Only look up ML interpretation if fusion strategy needs it
if self.fusion_strategy == "traditional_primary":
    # Skip ML lookup entirely
    return trad_interp
```

---

## Dependencies

### Core Dependencies
- **FieldInterpretation**: Traditional detection interpretation objects
- **BubblesFieldInterpretation**: Bubble-specific interpretation
- **Logger**: Logging fusion decisions

### Data Dependencies
- **field_id_to_interpretation**: Traditional detection results
- **ml_blocks_response**: ML field block detection results
- **confidence_threshold**: Fusion decision threshold

### Future Dependencies
- **Field-to-block mapper**: Map field IDs to ML blocks
- **Interpretation factory**: Create FieldInterpretation from ML results
- **Agreement calculator**: Compare bubble responses

---

## Browser Migration Notes

### TypeScript Implementation

```typescript
interface DetectionFusionConfig {
  fusionStrategy: "confidence_weighted" | "ml_fallback" | "traditional_primary";
  discrepancyThreshold: number;
}

interface Discrepancy {
  fieldId: string;
  traditional: {
    response: string[];
    confidence: number;
  };
  ml: {
    response: string[];
    confidence: number;
  };
  reason: "high_confidence_disagreement" | "informational_difference";
}

class DetectionFusion {
  private fusionStrategies = {
    confidence_weighted: this.confidenceWeightedFusion.bind(this),
    ml_fallback: this.mlFallbackFusion.bind(this),
    traditional_primary: this.traditionalPrimaryFusion.bind(this),
  };

  constructor(
    private fusionStrategy: string = "confidence_weighted",
    private discrepancyThreshold: number = 0.3
  ) {}

  fuseDetections(
    traditionalResponse: Record<string, any>,
    mlBlocksResponse: MLBlock[],
    fieldIdToInterpretation: Record<string, FieldInterpretation>,
    confidenceThreshold: number = 0.85
  ): [Record<string, any>, Discrepancy[]] {
    const fusedResponse: Record<string, any> = {};
    const discrepancies: Discrepancy[] = [];

    for (const [fieldId, tradInterp] of Object.entries(fieldIdToInterpretation)) {
      // Find ML interpretation
      const mlInterp = this.findMLInterpretation(fieldId, mlBlocksResponse);

      if (!mlInterp) {
        fusedResponse[fieldId] = tradInterp;
        continue;
      }

      // Apply fusion strategy
      const strategyMethod = this.fusionStrategies[this.fusionStrategy];
      if (strategyMethod) {
        const [fused, discrepancy] = strategyMethod(
          fieldId,
          tradInterp,
          mlInterp,
          confidenceThreshold
        );
        fusedResponse[fieldId] = fused;
        if (discrepancy) {
          discrepancies.push(discrepancy);
        }
      } else {
        console.warn(`Unknown fusion strategy '${this.fusionStrategy}'`);
        fusedResponse[fieldId] = tradInterp;
      }
    }

    return [fusedResponse, discrepancies];
  }

  private confidenceWeightedFusion(
    fieldId: string,
    tradInterp: FieldInterpretation,
    mlInterp: MLInterpretation,
    confidenceThreshold: number
  ): [any, Discrepancy | null] {
    const tradConfidence = tradInterp.overallConfidenceScore ?? 0.0;
    const mlConfidence = mlInterp.confidence ?? 0.0;

    const tradResponse = tradInterp.matchedBubbles ?? [];
    const mlResponse = mlInterp.detectedBubbles ?? [];

    const agree = this.responsesAgree(tradResponse, mlResponse);

    // Both high confidence
    if (tradConfidence > confidenceThreshold && mlConfidence > confidenceThreshold) {
      if (agree) {
        return [tradInterp, null];
      }

      // Disagreement - flag and use higher confidence
      const discrepancy: Discrepancy = {
        fieldId,
        traditional: { response: tradResponse, confidence: tradConfidence },
        ml: { response: mlResponse, confidence: mlConfidence },
        reason: "high_confidence_disagreement",
      };

      return [tradConfidence > mlConfidence ? tradInterp : this.createMLInterpretation(mlInterp), discrepancy];
    }

    // Low traditional, high ML - use ML
    if (tradConfidence < 0.6 && mlConfidence > confidenceThreshold) {
      return [this.createMLInterpretation(mlInterp), null];
    }

    // Default to traditional
    return [tradInterp, null];
  }

  private mlFallbackFusion(
    fieldId: string,
    tradInterp: FieldInterpretation,
    mlInterp: MLInterpretation,
    confidenceThreshold: number
  ): [any, null] {
    const tradConfidence = tradInterp.overallConfidenceScore ?? 0.0;

    if (tradConfidence < 0.6) {
      return [this.createMLInterpretation(mlInterp), null];
    }
    return [tradInterp, null];
  }

  private traditionalPrimaryFusion(
    fieldId: string,
    tradInterp: FieldInterpretation,
    mlInterp: MLInterpretation
  ): [any, Discrepancy | null] {
    const tradResponse = tradInterp.matchedBubbles ?? [];
    const mlResponse = mlInterp.detectedBubbles ?? [];

    if (!this.responsesAgree(tradResponse, mlResponse)) {
      const discrepancy: Discrepancy = {
        fieldId,
        traditional: { response: tradResponse, confidence: 0 },
        ml: { response: mlResponse, confidence: 0 },
        reason: "informational_difference",
      };
      return [tradInterp, discrepancy];
    }

    return [tradInterp, null];
  }

  private findMLInterpretation(fieldId: string, mlBlocks: MLBlock[]): MLInterpretation | null {
    // TODO: Implement field-to-block mapping
    return null;
  }

  private responsesAgree(tradResponse: string[], mlResponse: string[]): boolean {
    // TODO: Implement actual comparison
    return true;
  }

  private createMLInterpretation(mlInterp: MLInterpretation): any {
    // TODO: Create proper FieldInterpretation object
    return mlInterp;
  }
}
```

### Browser-Specific Considerations

1. **No Performance Concerns**: Fusion is lightweight (pure logic, no CV operations)
2. **JSON Serialization**: All data structures are JSON-serializable
3. **Async Not Required**: Fast enough to run synchronously
4. **Worker Compatible**: Can run in main thread or Web Worker
5. **No External Dependencies**: Pure JavaScript logic

---

## Related Modules

### Detection Systems
- **ReadOMR Processor** (`../concept.md`) - Main detection orchestrator
- **ML Field Block Detector** (`../ml-field-block/`) - ML block detection
- **Bubbles Threshold** (`../bubbles-threshold/`) - Traditional bubble detection
- **Shift Detection** (`../shift-detection/`) - ML-based shift correction

### Supporting Modules
- **Threshold Strategies** (`../../threshold/`) - Confidence calculation
- **FieldInterpretation** (`base/interpretation.py`) - Interpretation data structures
- **ProcessingContext** (`../../processing-context/`) - Data flow container

---

## Next Steps

1. Read `flows.md` for detailed fusion execution flow
2. Read `decisions.md` for fusion strategy selection criteria
3. Read `constraints.md` for performance limits and accuracy trade-offs
4. Read `integration.md` for integration with ReadOMR pipeline
