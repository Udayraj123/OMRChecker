# Bubble Detection - Threshold Strategy

**Status**: Core Detection System
**Python Reference**: `src/processors/detection/bubbles_threshold/`
**Browser Compatibility**: OpenCV.js compatible

---

## Overview

The Bubble Detection Threshold Strategy is the **classical computer vision approach** to detecting filled bubbles in OMR (Optical Mark Recognition) sheets. It uses **intensity-based thresholding** to determine whether a bubble is marked or unmarked by analyzing the mean pixel intensity within each bubble region.

This is the primary detection method in OMRChecker, complemented by optional ML-based detection for edge cases.

---

## Architecture

### Component Hierarchy

```
BubblesThresholdDetectionPass (Detection)
    └─> BubblesFieldDetection
        └─> BubbleMeanValue (per bubble)
            └─> mean_value: float
            └─> unit_bubble: BubblesScanBox
            └─> position: (x, y)

BubblesThresholdInterpretationPass (Interpretation)
    └─> BubblesFieldInterpretation
        └─> threshold strategies (Global/Local)
        └─> BubbleInterpretation (per bubble)
            └─> is_attempted: bool
            └─> bubble_value: str
```

### Key Components

1. **Detection Pass** (`detection_pass.py`)
   - Extracts mean intensity values from each bubble ROI
   - Stores results in `DetectionRepository` using typed models
   - Calculates field-level statistics (std deviation, mean values)

2. **Interpretation Pass** (`interpretation_pass.py`)
   - Calculates thresholds using strategies (Global, Local)
   - Interprets bubble values based on threshold
   - Detects multi-marking and confidence issues

3. **Threshold Strategies** (`src/processors/threshold/`)
   - **GlobalThresholdStrategy**: File-level threshold (all fields)
   - **LocalThresholdStrategy**: Field-level threshold (per field, with fallback)

4. **Detection Models** (`detection_results.py`)
   - `BubbleFieldDetectionResult`: Typed detection result
   - `BubbleMeanValue`: Single bubble measurement
   - Auto-calculated properties: `std_deviation`, `scan_quality`, `jumps`

---

## Detection Flow

### Phase 1: Detection Pass

```python
# For each bubble field in template:
for field in bubble_fields:
    # 1. Extract bubble ROIs from aligned image
    for unit_bubble in field.scan_boxes:
        box_w, box_h = unit_bubble.bubble_dimensions
        x, y = unit_bubble.get_shifted_position()

        # 2. Calculate mean intensity (0-255)
        mean_value = cv2.mean(gray_image[y:y+box_h, x:x+box_w])[0]

        # 3. Store as BubbleMeanValue
        bubble_means.append(BubbleMeanValue(
            mean_value=mean_value,
            unit_bubble=unit_bubble,
            position=(x, y)
        ))

    # 4. Create typed detection result
    result = BubbleFieldDetectionResult(
        field_id=field.id,
        field_label=field.field_label,
        bubble_means=bubble_means
    )

    # 5. Save to repository
    repository.save_bubble_field(field.id, result)
```

### Phase 2: Interpretation Pass

```python
# 1. Calculate Global Threshold (file-level fallback)
all_bubble_means = repository.get_all_bubble_means_for_current_file()
global_threshold, global_max_jump = get_fallback_threshold(all_bubble_means)

# 2. For each field, calculate Local Threshold
for field in bubble_fields:
    detection_result = repository.get_bubble_field(field.field_label)

    # Use LocalThresholdStrategy with global fallback
    strategy = LocalThresholdStrategy(global_fallback=global_threshold)
    threshold_result = strategy.calculate_threshold(
        detection_result.mean_values,
        threshold_config
    )

    # 3. Interpret each bubble
    for bubble_mean in detection_result.bubble_means:
        is_attempted = bubble_mean.mean_value < threshold_result.threshold_value
        bubble_interpretations.append(BubbleInterpretation(
            bubble_mean=bubble_mean,
            threshold=threshold_result.threshold_value
        ))

    # 4. Check for multi-marking
    marked_count = sum(1 for b in bubble_interpretations if b.is_attempted)
    is_multi_marked = marked_count > 1
```

---

## Data Models

### BubbleMeanValue

```python
@dataclass
class BubbleMeanValue:
    mean_value: float        # 0-255 (darker = lower value)
    unit_bubble: BubblesScanBox  # Reference to bubble definition
    position: tuple[int, int]    # (x, y) position in image
```

### BubbleFieldDetectionResult

```python
@dataclass
class BubbleFieldDetectionResult:
    field_id: str
    field_label: str
    bubble_means: list[BubbleMeanValue]

    # Auto-calculated properties:
    @property
    def std_deviation(self) -> float:
        """Standard deviation of bubble means"""

    @property
    def scan_quality(self) -> ScanQuality:
        """EXCELLENT/GOOD/ACCEPTABLE/POOR based on std"""

    @property
    def mean_values(self) -> list[float]:
        """Just the float values"""

    @property
    def sorted_mean_values(self) -> list[float]:
        """Sorted values for threshold calculation"""

    @property
    def jumps(self) -> list[tuple[float, BubbleMeanValue]]:
        """Gaps between consecutive sorted values"""
```

### BubbleInterpretation

```python
class BubbleInterpretation:
    bubble_mean: BubbleMeanValue
    threshold: float
    is_attempted: bool  # mean_value < threshold
    bubble_value: str   # 'A', 'B', '1', etc.

    def get_value(self) -> str:
        return self.bubble_value if self.is_attempted else ""
```

---

## Threshold Strategy Pattern

### ThresholdStrategy Base Class

```python
class ThresholdStrategy(ABC):
    @abstractmethod
    def calculate_threshold(
        self, bubble_mean_values: list[float],
        config: ThresholdConfig
    ) -> ThresholdResult:
        pass
```

### ThresholdResult

```python
@dataclass
class ThresholdResult:
    threshold_value: float      # Calculated threshold
    confidence: float           # 0.0-1.0
    max_jump: float            # Largest gap in sorted values
    method_used: str           # "global_max_jump", "local_max_jump", etc.
    fallback_used: bool        # Whether fallback was used
    metadata: dict             # Additional context
```

### ThresholdConfig

```python
@dataclass
class ThresholdConfig:
    min_jump: float = 30.0
    jump_delta: float = 20.0
    min_gap_two_bubbles: float = 20.0
    min_jump_surplus_for_global_fallback: float = 10.0
    confident_jump_surplus_for_disparity: float = 15.0
    global_threshold_margin: float = 10.0
    outlier_deviation_threshold: float = 5.0
    default_threshold: float = 127.5
```

---

## Key Principles

### 1. Intensity-Based Detection

**Marked bubbles are DARKER (lower mean intensity)**

```
Unmarked bubble: mean ≈ 200-240 (light gray/white)
Marked bubble:   mean ≈ 50-120  (dark gray/black)
```

### 2. Two-Phase Separation

**Detection Phase**:
- Pure computer vision (OpenCV)
- No interpretation logic
- Just extract measurements
- Store in repository

**Interpretation Phase**:
- Calculate thresholds
- Apply thresholds to measurements
- Detect multi-marking
- Generate confidence metrics

### 3. Threshold Hierarchy

```
Default Threshold (127.5)
    └─> Global Threshold (file-level, all bubbles)
        └─> Local Threshold (field-level, with fallback to global)
```

### 4. Confidence-Based System

Every threshold calculation returns confidence (0.0-1.0):
- High confidence: Large gap between marked/unmarked bubbles
- Low confidence: Small gap or ambiguous values
- Fallback: Use global threshold when local confidence is low

---

## Integration with Pipeline

### 1. Detection Pass Registration

```python
# In TemplateFileRunner
detection_passes = {
    FieldType.BUBBLES_THRESHOLD: BubblesThresholdDetectionPass(
        repository=repository,
        tuning_config=tuning_config
    )
}
```

### 2. Interpretation Pass Registration

```python
interpretation_passes = {
    FieldType.BUBBLES_THRESHOLD: BubblesThresholdInterpretationPass(
        repository=repository,
        tuning_config=tuning_config
    )
}
```

### 3. Repository Pattern

```python
class DetectionRepository:
    def save_bubble_field(self, field_id: str, result: BubbleFieldDetectionResult):
        """Store detection result"""

    def get_bubble_field(self, field_id: str) -> BubbleFieldDetectionResult:
        """Retrieve detection result"""

    def get_all_bubble_means_for_current_file(self) -> list[BubbleMeanValue]:
        """Get all bubble means across all fields"""
```

---

## Browser Migration

### OpenCV.js Usage

```typescript
// Detection: Extract bubble mean
const roi = grayImage.roi(new cv.Rect(x, y, width, height));
const mean = cv.mean(roi);
const meanValue = mean[0]; // Intensity value

roi.delete(); // Memory management
```

### Threshold Strategies (Pure JavaScript)

```typescript
class GlobalThresholdStrategy implements ThresholdStrategy {
  calculateThreshold(
    bubbleMeans: number[],
    config: ThresholdConfig
  ): ThresholdResult {
    const sorted = [...bubbleMeans].sort((a, b) => a - b);

    // Find max jump
    let maxJump = config.minJump;
    let threshold = config.defaultThreshold;

    for (let i = 1; i < sorted.length - 1; i++) {
      const jump = sorted[i + 1] - sorted[i - 1];
      if (jump > maxJump) {
        maxJump = jump;
        threshold = sorted[i - 1] + jump / 2;
      }
    }

    return {
      thresholdValue: threshold,
      confidence: Math.min(1.0, maxJump / (config.minJump * 3)),
      maxJump,
      methodUsed: "global_max_jump",
      fallbackUsed: maxJump < config.minJump
    };
  }
}
```

### TypeScript Models

```typescript
interface BubbleMeanValue {
  meanValue: number;
  unitBubble: BubblesScanBox;
  position: [number, number];
}

interface BubbleFieldDetectionResult {
  fieldId: string;
  fieldLabel: string;
  bubbleMeans: BubbleMeanValue[];

  // Computed properties
  get stdDeviation(): number;
  get scanQuality(): ScanQuality;
  get meanValues(): number[];
  get sortedMeanValues(): number[];
  get jumps(): Array<[number, BubbleMeanValue]>;
}

enum ScanQuality {
  EXCELLENT = "excellent", // std > 50
  GOOD = "good",           // std > 30
  ACCEPTABLE = "acceptable", // std > 15
  POOR = "poor"            // std <= 15
}
```

---

## Performance Characteristics

### Time Complexity

- **Detection**: O(n × w × h) where n = number of bubbles, w×h = bubble dimensions
- **Global Threshold**: O(n log n) for sorting all bubble means
- **Local Threshold**: O(m log m) where m = bubbles per field (typically 4-10)

### Space Complexity

- **Detection Results**: O(n) - one BubbleMeanValue per bubble
- **Repository**: O(f × b) where f = fields, b = bubbles per field

### Browser Considerations

```typescript
// Memory management for OpenCV.js
function detectBubbleField(field: Field, grayImage: cv.Mat): BubbleFieldDetectionResult {
  const bubbleMeans: BubbleMeanValue[] = [];

  for (const bubble of field.scanBoxes) {
    const roi = grayImage.roi(new cv.Rect(bubble.x, bubble.y, bubble.w, bubble.h));
    const mean = cv.mean(roi);

    bubbleMeans.push({
      meanValue: mean[0],
      unitBubble: bubble,
      position: [bubble.x, bubble.y]
    });

    roi.delete(); // CRITICAL: Free memory
  }

  return {
    fieldId: field.id,
    fieldLabel: field.fieldLabel,
    bubbleMeans
  };
}
```

---

## Advantages

1. **Fast**: Simple intensity calculations, no ML inference
2. **Reliable**: Works for clean scans with good contrast
3. **Explainable**: Clear logic (darker = marked)
4. **Browser-friendly**: Minimal dependencies (just OpenCV.js)
5. **No training needed**: No ML models to train or deploy

---

## Limitations

1. **Requires good contrast**: Fails on poor scans or faded marks
2. **Sensitive to lighting**: Uneven illumination can cause false positives/negatives
3. **No context awareness**: Treats each bubble independently
4. **Threshold tuning**: May need adjustment for different scan qualities

---

## Related Components

- **Detection Pass**: `/bubbles-threshold/detection-pass/` - ROI extraction logic
- **Interpretation Pass**: `/bubbles-threshold/interpretation-pass/` - Threshold application
- **Threshold Strategies**: `/threshold/` - Global and Local strategies
- **ReadOMR Processor**: Main orchestrator that calls detection/interpretation
- **ML Bubble Detector**: ML-based fallback for low-confidence cases

---

## Next Steps

See related documentation:
- `flows.md` - Complete detection and interpretation flows
- `decisions.md` - Threshold strategy selection logic
- `constraints.md` - Performance, accuracy, edge cases
- `integration.md` - Integration with ReadOMR and pipeline
