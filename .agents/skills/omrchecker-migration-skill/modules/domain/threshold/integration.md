# Threshold Strategy Integration

**Module**: Domain - Threshold
**Python Reference**: `src/processors/threshold/*.py`
**Last Updated**: 2026-02-20

---

## Overview

Threshold strategies integrate with the bubble detection pipeline at two key points:
1. **Detection Pass**: Calculate global threshold from all bubbles
2. **Interpretation Pass**: Calculate local threshold per field and classify bubbles

This document describes how threshold strategies connect with other system components.

---

## Integration Points

### 1. Bubble Detection Pipeline

**Location**: `src/processors/detection/bubbles_threshold/`

**Flow**:
```
Detection Pass → Global Threshold → Interpretation Pass → Local Threshold → Bubble Classification
```

**Components**:
- `BubblesThresholdFileRunner`: Orchestrates detection + interpretation
- `BubblesThresholdDetectionPass`: Extracts bubbles, calculates global threshold
- `BubblesThresholdInterpretationPass`: Per-field interpretation using local threshold
- `BubblesFieldInterpretation`: Uses threshold strategies for classification

---

### 2. Configuration System

**Location**: `src/schemas/models/config.py`

**Integration**:
```python
# Tuning config contains threshold parameters
tuning_config.thresholding = {
    "min_jump": 30.0,
    "jump_delta": 20.0,
    "min_gap_two_bubbles": 20.0,
    "min_jump_surplus_for_global_fallback": 10.0,
    "confident_jump_surplus_for_disparity": 15.0,
    "global_threshold_margin": 10.0,
    "global_page_threshold": 127.5
}

# Convert to ThresholdConfig
threshold_config = ThresholdConfig(
    min_jump=tuning_config.thresholding.min_jump,
    jump_delta=tuning_config.thresholding.jump_delta,
    # ... other fields
)
```

**Files**:
- `src/schemas/defaults/tuning_defaults.json`: Default threshold values
- `inputs/<template>/config.json`: User overrides

---

### 3. Detection Results (Typed Models)

**Location**: `src/processors/detection/models/detection_results.py`

**Integration**:
```python
@dataclass
class BubbleFieldDetectionResult:
    """Detection results for a single field."""
    field_label: str
    bubble_means: list[BubbleMean]  # Individual bubble data
    mean_values: list[float]        # Auto-extracted for threshold calc

# Usage in interpretation:
detection_result = file_level_aggregates["bubble_fields"][field_label]
threshold_result = strategy.calculate_threshold(
    detection_result.mean_values,  # Direct access!
    threshold_config
)
```

**Benefit**: Clean separation between detection (find bubbles) and interpretation (classify bubbles).

---

### 4. File-Level Aggregates

**Purpose**: Store global threshold for use across all fields

**Structure**:
```python
file_level_detection_aggregates = {
    "bubble_fields": {
        "Q1": BubbleFieldDetectionResult(...),
        "Q2": BubbleFieldDetectionResult(...),
        # ... all fields
    },
    "all_bubble_means": [45, 50, 55, ...],  # All bubbles in file
}

file_level_interpretation_aggregates = {
    "file_level_fallback_threshold": 96.0,  # Global threshold
    "outlier_deviation_threshold_for_file": 5.0,
}
```

**Flow**:
```
1. Detection Pass:
   - Collect all bubble means
   - Calculate global threshold
   - Store in file_level_interpretation_aggregates

2. Interpretation Pass:
   - For each field:
     - Get field's bubble means
     - Calculate local threshold (using global as fallback)
     - Classify bubbles
```

---

## Component Interactions

### BubblesFieldInterpretation Class

**File**: `src/processors/detection/bubbles_threshold/interpretation.py`

**Key Methods**:

#### run_interpretation()

```python
def run_interpretation(
    self,
    field: Field,
    file_level_detection_aggregates,
    file_level_interpretation_aggregates,
) -> None:
    """Main interpretation flow."""

    # Step 1: Extract detection result
    detection_result = self._extract_detection_result(
        field, file_level_detection_aggregates
    )

    # Step 2: Calculate threshold
    threshold_config = self._create_threshold_config(
        file_level_interpretation_aggregates
    )
    self.threshold_result = self._calculate_threshold(
        detection_result,
        file_level_interpretation_aggregates,
        threshold_config
    )

    # Step 3: Interpret bubbles
    self._interpret_bubbles(detection_result)

    # Step 4: Check multi-marking
    self._check_multi_marking()
```

#### _create_threshold_config()

```python
def _create_threshold_config(
    self, file_level_interpretation_aggregates
) -> ThresholdConfig:
    """Create ThresholdConfig from tuning config."""
    config = self.tuning_config
    return ThresholdConfig(
        min_jump=config.thresholding.min_jump,
        jump_delta=config.thresholding.jump_delta,
        min_gap_two_bubbles=config.thresholding.min_gap_two_bubbles,
        min_jump_surplus_for_global_fallback=config.thresholding.min_jump_surplus_for_global_fallback,
        confident_jump_surplus_for_disparity=config.thresholding.confident_jump_surplus_for_disparity,
        global_threshold_margin=config.thresholding.global_threshold_margin,
        outlier_deviation_threshold=file_level_interpretation_aggregates.get(
            "outlier_deviation_threshold_for_file", 5.0
        ),
        default_threshold=config.thresholding.global_page_threshold,
    )
```

#### _calculate_threshold()

```python
def _calculate_threshold(
    self,
    detection_result: BubbleFieldDetectionResult,
    file_level_interpretation_aggregates,
    config: ThresholdConfig,
) -> ThresholdResult:
    """Calculate threshold using LocalThresholdStrategy.

    This replaces 170+ lines of threshold calculation code!
    """
    # Get global fallback
    global_fallback = file_level_interpretation_aggregates.get(
        "file_level_fallback_threshold",
        config.default_threshold
    )

    # Use local strategy with global fallback
    strategy = LocalThresholdStrategy(global_fallback=global_fallback)

    # Calculate threshold
    return strategy.calculate_threshold(
        detection_result.mean_values,  # List of floats
        config,
    )
```

**Before (Old Code)**:
- 170+ lines of complex threshold logic
- Multiple nested conditionals
- Hard to test and maintain

**After (With Strategies)**:
- 8 lines (instantiate strategy, call calculate)
- Clean separation of concerns
- Easily testable

#### _interpret_bubbles()

```python
def _interpret_bubbles(self, detection_result: BubbleFieldDetectionResult) -> None:
    """Interpret each bubble using calculated threshold."""
    self.bubble_interpretations = [
        BubbleInterpretation(bubble_mean, self.local_threshold_for_field)
        for bubble_mean in detection_result.bubble_means
    ]
```

### BubbleInterpretation Class

**File**: `src/processors/detection/bubbles_threshold/interpretation.py`

**Purpose**: Single bubble's interpretation result

```python
class BubbleInterpretation:
    def __init__(self, bubble_mean, threshold: float) -> None:
        self.bubble_mean = bubble_mean
        self.threshold = threshold
        self.mean_value = bubble_mean.mean_value
        self.is_attempted = bubble_mean.mean_value < threshold  # KEY COMPARISON
        self.bubble_value = bubble_mean.unit_bubble.bubble_value or ""

    def get_value(self) -> str:
        """Get bubble value if marked."""
        return self.bubble_value if self.is_attempted else ""
```

**Key Logic**:
```python
self.is_attempted = bubble_mean.mean_value < threshold
#                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                   This is where threshold is used!
#
# If bubble is DARKER (lower value) than threshold → FILLED
# If bubble is LIGHTER (higher value) than threshold → EMPTY
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│ 1. CONFIGURATION                                             │
└──────────────────────────────────────────────────────────────┘
│
├─► tuning_config.json
│   {
│     "thresholding": {
│       "min_jump": 30.0,
│       "jump_delta": 20.0,
│       ...
│     }
│   }
│
└─► ThresholdConfig
    (Dataclass with defaults + user overrides)

┌──────────────────────────────────────────────────────────────┐
│ 2. DETECTION PASS (File-Level)                               │
└──────────────────────────────────────────────────────────────┘
│
├─► BubblesThresholdDetectionPass
│   │
│   ├─► For each field:
│   │   Extract bubbles → BubbleFieldDetectionResult
│   │
│   └─► Aggregate:
│       all_bubble_means = [45, 50, 55, ..., 145, 150, 155]
│
├─► Calculate Global Threshold
│   GlobalThresholdStrategy().calculate_threshold(
│       all_bubble_means,
│       threshold_config
│   )
│   → global_threshold = 96.0
│
└─► Store in Aggregates
    file_level_interpretation_aggregates = {
        "file_level_fallback_threshold": 96.0
    }

┌──────────────────────────────────────────────────────────────┐
│ 3. INTERPRETATION PASS (Field-Level)                         │
└──────────────────────────────────────────────────────────────┘
│
For each field in template:
│
├─► Get Field's Detection Result
│   detection_result = bubble_fields[field_label]
│   field_bubble_means = [45, 50, 145, 150]
│
├─► Calculate Local Threshold
│   LocalThresholdStrategy(global_fallback=96.0).calculate_threshold(
│       field_bubble_means,
│       threshold_config
│   )
│   → threshold_result = ThresholdResult(
│       threshold_value=97.5,
│       confidence=1.0,
│       method_used="local_max_jump",
│       ...
│   )
│
├─► Classify Each Bubble
│   For each bubble in field:
│       if bubble_mean < 97.5:
│           is_attempted = True  # FILLED
│       else:
│           is_attempted = False # EMPTY
│
├─► Generate Field Value
│   marked_bubbles = ["A", "B"]  # Bubbles 0, 1 were filled
│   field_value = "AB"
│
└─► Return Results
    {
        "field_label": "Q1",
        "field_value": "AB",
        "threshold_used": 97.5,
        "threshold_confidence": 1.0,
        "is_multi_marked": True
    }
```

---

## External Dependencies

### From Threshold Strategies

**Imports**:
```python
from src.processors.threshold.threshold_strategy import ThresholdStrategy
from src.processors.threshold.global_threshold import GlobalThresholdStrategy
from src.processors.threshold.local_threshold import LocalThresholdStrategy
from src.processors.threshold.adaptive_threshold import AdaptiveThresholdStrategy
from src.processors.threshold.threshold_result import ThresholdConfig, ThresholdResult
```

**Usage Locations**:
- `src/processors/detection/bubbles_threshold/interpretation.py`: Uses LocalThresholdStrategy
- `src/processors/detection/bubbles_threshold/interpretation_pass.py`: May use GlobalThresholdStrategy
- Future: Could use AdaptiveThresholdStrategy for best results

### To Threshold Strategies

**No dependencies**: Threshold strategies are self-contained
- Only depend on Python stdlib (abc, dataclasses)
- Local strategy optionally uses numpy (for mean calculation)

---

## Integration with Other Systems

### 1. Logging Integration

**Location**: Threshold calculation emits logs

```python
from src.utils.logger import logger

# In interpretation:
if threshold_result.fallback_used:
    logger.warning(
        f"Threshold fallback used for field {field_label}: "
        f"{threshold_result.method_used}, "
        f"confidence={threshold_result.confidence:.2f}"
    )
```

**Logged Events**:
- Low confidence thresholds
- Fallback usage
- Multi-marking detection (indirectly related)

### 2. Visualization Integration

**Location**: `src/processors/detection/bubbles_threshold/interpretation_drawing.py`

**Integration**:
```python
class BubblesFieldInterpretationDrawing:
    def draw_interpretation(self):
        # Draw threshold line on debug images
        threshold = self.field_interpretation.local_threshold_for_field

        # Color bubbles based on interpretation
        for interp in self.field_interpretation.bubble_interpretations:
            color = GREEN if interp.is_attempted else RED
            # Draw on image...
```

**Visualization Shows**:
- Threshold value used
- Which bubbles are above/below threshold
- Confidence metrics (if enabled)

### 3. Output/Results Integration

**Location**: Final CSV output, JSON results

**Integration**:
```python
# CSV row includes field values
row = {
    "Q1": "AB",  # Derived from threshold-based classification
    "Q2": "C",
    # ...
}

# Optional: Include threshold metadata
if config.outputs.show_confidence_metrics:
    row["Q1_threshold"] = 97.5
    row["Q1_confidence"] = 1.0
    row["Q1_method"] = "local_max_jump"
```

---

## API Surface

### Public Functions

#### create_default_threshold_calculator()

**File**: `src/processors/threshold/adaptive_threshold.py`

```python
def create_default_threshold_calculator(
    global_threshold: float | None = None,
) -> ThresholdStrategy:
    """Factory function to create default threshold calculator.

    Args:
        global_threshold: Optional global threshold for local strategy fallback

    Returns:
        AdaptiveThresholdStrategy combining global and local strategies
    """
    return AdaptiveThresholdStrategy(
        strategies=[
            GlobalThresholdStrategy(),
            LocalThresholdStrategy(global_fallback=global_threshold),
        ],
        weights=[0.4, 0.6],  # Prefer local threshold
    )
```

**Usage**:
```python
# Create calculator with global fallback
calculator = create_default_threshold_calculator(global_threshold=96.0)

# Calculate threshold for field
result = calculator.calculate_threshold(field_bubbles, config)
```

### Public Classes

#### ThresholdStrategy (Abstract)

```python
from src.processors.threshold.threshold_strategy import ThresholdStrategy

class CustomStrategy(ThresholdStrategy):
    def calculate_threshold(self, values, config):
        # Implement custom logic
        pass
```

#### GlobalThresholdStrategy

```python
from src.processors.threshold.global_threshold import GlobalThresholdStrategy

strategy = GlobalThresholdStrategy()
result = strategy.calculate_threshold(all_bubbles, config)
```

#### LocalThresholdStrategy

```python
from src.processors.threshold.local_threshold import LocalThresholdStrategy

strategy = LocalThresholdStrategy(global_fallback=96.0)
result = strategy.calculate_threshold(field_bubbles, config)
```

#### AdaptiveThresholdStrategy

```python
from src.processors.threshold.adaptive_threshold import AdaptiveThresholdStrategy

strategy = AdaptiveThresholdStrategy(
    strategies=[strategy1, strategy2],
    weights=[0.4, 0.6]
)
result = strategy.calculate_threshold(bubbles, config)
```

### Public Data Classes

#### ThresholdConfig

```python
from src.processors.threshold.threshold_result import ThresholdConfig

config = ThresholdConfig(
    min_jump=30.0,
    jump_delta=20.0,
    # ... other parameters (all have defaults)
)
```

#### ThresholdResult

```python
from src.processors.threshold.threshold_result import ThresholdResult

result = ThresholdResult(
    threshold_value=96.5,
    confidence=0.95,
    max_jump=85.0,
    method_used="local_max_jump",
    fallback_used=False,
    metadata={"num_bubbles": 10}
)

# Access fields
print(result.threshold_value)  # 96.5
print(result.confidence)       # 0.95
```

---

## Usage Examples

### Example 1: Simple Local Threshold

```python
from src.processors.threshold.local_threshold import LocalThresholdStrategy
from src.processors.threshold.threshold_result import ThresholdConfig

# Setup
field_bubbles = [45.2, 50.1, 55.8, 145.3, 150.7]
global_fallback = 96.0
config = ThresholdConfig()

# Calculate threshold
strategy = LocalThresholdStrategy(global_fallback=global_fallback)
result = strategy.calculate_threshold(field_bubbles, config)

# Use result
print(f"Threshold: {result.threshold_value:.1f}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Method: {result.method_used}")

# Classify bubbles
for bubble_mean in field_bubbles:
    is_filled = bubble_mean < result.threshold_value
    state = "FILLED" if is_filled else "EMPTY"
    print(f"Bubble {bubble_mean:.1f}: {state}")
```

**Output**:
```
Threshold: 97.6
Confidence: 1.00
Method: local_max_jump
Bubble 45.2: FILLED
Bubble 50.1: FILLED
Bubble 55.8: FILLED
Bubble 145.3: EMPTY
Bubble 150.7: EMPTY
```

### Example 2: Adaptive Strategy

```python
from src.processors.threshold.adaptive_threshold import create_default_threshold_calculator
from src.processors.threshold.threshold_result import ThresholdConfig

# Setup
field_bubbles = [48.0, 52.0, 140.0, 148.0]
global_fallback = 95.0
config = ThresholdConfig()

# Create calculator (combines global + local)
calculator = create_default_threshold_calculator(global_threshold=global_fallback)

# Calculate
result = calculator.calculate_threshold(field_bubbles, config)

# Inspect metadata
print(f"Combined Threshold: {result.threshold_value:.1f}")
for strategy_result in result.metadata["strategy_results"]:
    print(f"  {strategy_result['method']}: "
          f"{strategy_result['threshold']:.1f} "
          f"(conf={strategy_result['confidence']:.2f}, "
          f"weight={strategy_result['weight']})")
```

**Output**:
```
Combined Threshold: 95.2
  global_max_jump: 94.0 (conf=1.00, weight=0.4)
  local_max_jump: 96.0 (conf=1.00, weight=0.6)
```

### Example 3: Integration with Detection Results

```python
from src.processors.threshold.local_threshold import LocalThresholdStrategy
from src.processors.threshold.threshold_result import ThresholdConfig
from src.processors.detection.bubbles_threshold.interpretation import BubbleInterpretation

# Get detection result
detection_result = file_level_aggregates["bubble_fields"]["Q1"]

# Calculate threshold
threshold_config = ThresholdConfig()
global_fallback = file_level_aggregates.get("file_level_fallback_threshold", 127.5)

strategy = LocalThresholdStrategy(global_fallback=global_fallback)
threshold_result = strategy.calculate_threshold(
    detection_result.mean_values,
    threshold_config
)

# Interpret bubbles
bubble_interpretations = [
    BubbleInterpretation(bubble_mean, threshold_result.threshold_value)
    for bubble_mean in detection_result.bubble_means
]

# Get field value
marked_values = [
    interp.bubble_value
    for interp in bubble_interpretations
    if interp.is_attempted
]
field_value = "".join(marked_values)

print(f"Field Value: {field_value}")
print(f"Threshold: {threshold_result.threshold_value:.1f}")
print(f"Confidence: {threshold_result.confidence:.2f}")
```

---

## Browser Migration Notes

### TypeScript Module Structure

```typescript
// threshold/types.ts
export interface ThresholdConfig {
  readonly minJump: number;
  readonly jumpDelta: number;
  // ... other fields
}

export interface ThresholdResult {
  readonly thresholdValue: number;
  readonly confidence: number;
  readonly maxJump: number;
  readonly methodUsed: string;
  readonly fallbackUsed: boolean;
  readonly metadata: Readonly<Record<string, unknown>>;
}

// threshold/strategy.ts
export abstract class ThresholdStrategy {
  abstract calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult;
}

// threshold/global.ts
export class GlobalThresholdStrategy extends ThresholdStrategy {
  calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult {
    // Implementation
  }
}

// threshold/local.ts
export class LocalThresholdStrategy extends ThresholdStrategy {
  constructor(private globalFallback?: number) {
    super();
  }

  calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult {
    // Implementation
  }
}

// threshold/adaptive.ts
export class AdaptiveThresholdStrategy extends ThresholdStrategy {
  constructor(
    private strategies: ThresholdStrategy[],
    private weights: number[]
  ) {
    super();
  }

  calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult {
    // Implementation
  }
}

// threshold/factory.ts
export function createDefaultThresholdCalculator(
  globalThreshold?: number
): ThresholdStrategy {
  return new AdaptiveThresholdStrategy(
    [
      new GlobalThresholdStrategy(),
      new LocalThresholdStrategy(globalThreshold)
    ],
    [0.4, 0.6]
  );
}
```

### Integration in Browser App

```typescript
// detection/bubbleInterpretation.ts
import { LocalThresholdStrategy } from '../threshold/local';
import { ThresholdConfig, ThresholdResult } from '../threshold/types';

export class BubbleFieldInterpretation {
  private thresholdResult?: ThresholdResult;

  calculateThreshold(
    detectionResult: BubbleFieldDetectionResult,
    fileLevelAggregates: FileLevelAggregates,
    config: ThresholdConfig
  ): ThresholdResult {
    const globalFallback = fileLevelAggregates.fileLevelFallbackThreshold
      ?? config.defaultThreshold;

    const strategy = new LocalThresholdStrategy(globalFallback);

    return strategy.calculateThreshold(
      detectionResult.meanValues,
      config
    );
  }

  interpretBubbles(
    detectionResult: BubbleFieldDetectionResult,
    threshold: number
  ): BubbleInterpretation[] {
    return detectionResult.bubbleMeans.map(bubbleMean => ({
      bubbleMean: bubbleMean.meanValue,
      threshold,
      isAttempted: bubbleMean.meanValue < threshold,
      bubbleValue: bubbleMean.unitBubble.bubbleValue ?? ''
    }));
  }
}
```

### Web Worker Usage

```typescript
// workers/thresholdWorker.ts
import { LocalThresholdStrategy } from '../threshold/local';

self.onmessage = (e) => {
  const { bubbleMeans, globalFallback, config } = e.data;

  const strategy = new LocalThresholdStrategy(globalFallback);
  const result = strategy.calculateThreshold(bubbleMeans, config);

  self.postMessage(result);
};

// main.ts
const worker = new Worker('thresholdWorker.js');

worker.postMessage({
  bubbleMeans: fieldBubbles,
  globalFallback: 96.0,
  config: thresholdConfig
});

worker.onmessage = (e) => {
  const thresholdResult = e.data;
  // Use result...
};
```

---

## Testing Integration

### Unit Tests

```python
# tests/processors/threshold/test_integration.py

def test_threshold_integration_with_detection():
    """Test threshold calculation with real detection results."""
    from src.processors.threshold.local_threshold import LocalThresholdStrategy
    from src.processors.threshold.threshold_result import ThresholdConfig
    from src.processors.detection.models.detection_results import BubbleFieldDetectionResult

    # Mock detection result
    detection_result = BubbleFieldDetectionResult(
        field_label="Q1",
        bubble_means=[...],
        mean_values=[45, 50, 145, 150]
    )

    # Calculate threshold
    config = ThresholdConfig()
    strategy = LocalThresholdStrategy(global_fallback=96.0)
    result = strategy.calculate_threshold(detection_result.mean_values, config)

    # Assert
    assert 50 < result.threshold_value < 145
    assert result.confidence > 0.5
```

### Integration Tests

```python
# tests/integration/test_bubble_interpretation.py

def test_full_interpretation_flow():
    """Test complete flow from detection to interpretation."""
    # Load test image
    image = load_test_image("sample_omr.jpg")
    template = load_template("template.json")

    # Run detection
    detection_pass = BubblesThresholdDetectionPass(...)
    detection_results = detection_pass.process_file(image, template)

    # Run interpretation
    interpretation_pass = BubblesThresholdInterpretationPass(...)
    interpretation_results = interpretation_pass.process_file(
        image,
        template,
        detection_results
    )

    # Verify threshold was used
    for field_result in interpretation_results:
        assert field_result.threshold_used > 0
        assert 0 <= field_result.threshold_confidence <= 1
        assert field_result.threshold_method in [
            "global_max_jump",
            "local_max_jump",
            "adaptive_weighted",
            # ... other valid methods
        ]
```

---

## Performance Monitoring

### Metrics to Track

```python
# In interpretation pass:
threshold_calculation_times = []
threshold_confidences = []
fallback_counts = 0

for field in fields:
    start = time.time()
    threshold_result = calculate_threshold(...)
    duration = time.time() - start

    threshold_calculation_times.append(duration)
    threshold_confidences.append(threshold_result.confidence)
    if threshold_result.fallback_used:
        fallback_counts += 1

# Log aggregate metrics
logger.info(f"Threshold calculation avg time: {np.mean(threshold_calculation_times):.3f}s")
logger.info(f"Average confidence: {np.mean(threshold_confidences):.2f}")
logger.info(f"Fallback usage: {fallback_counts}/{len(fields)} fields ({100*fallback_counts/len(fields):.1f}%)")
```

---

## Related Documentation

- **Bubble Detection**: `modules/domain/detection/bubbles-threshold/concept.md`
- **Detection Pass**: `modules/domain/detection/bubbles-threshold/detection-pass/flows.md`
- **Interpretation Pass**: `modules/domain/detection/bubbles-threshold/interpretation-pass/flows.md`
- **Configuration Management**: `modules/foundation/configuration.md` (pending)
- **Error Handling**: `modules/foundation/error-handling.md`
- **Logging**: `modules/foundation/logging.md`

---

## Summary

Threshold strategies integrate with:

1. **Bubble Detection Pipeline**: Core user for classification
2. **Configuration System**: Parameters from tuning config
3. **Detection Results**: Typed models provide clean interface
4. **Aggregates System**: File-level storage for global threshold
5. **Logging System**: Warnings for low confidence/fallback
6. **Visualization**: Debug images show thresholds
7. **Output System**: Optional threshold metadata in results

**Key Benefits**:
- Clean separation of concerns
- Testable in isolation
- Easy to extend with new strategies
- Browser-compatible design (no Python-specific dependencies)
