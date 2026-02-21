# Bubble Detection Pass - Flow Documentation

**Status**: Core Detection Component
**Python Reference**: `src/processors/detection/bubbles_threshold/detection_pass.py`, `src/processors/detection/bubbles_threshold/detection.py`
**Browser Compatibility**: OpenCV.js compatible

---

## Overview

The **Detection Pass** is the first phase of bubble detection that extracts raw intensity measurements from bubble regions. It performs pure computer vision operations without any interpretation logic, measuring pixel intensities within each bubble's Region of Interest (ROI).

This phase is separated from the Interpretation Pass to maintain a clean separation of concerns:
- **Detection Pass**: Measures bubble intensities (computer vision)
- **Interpretation Pass**: Decides which bubbles are marked (thresholding logic)

---

## Architecture

### Component Hierarchy

```
BubblesThresholdDetectionPass
  └─> FieldTypeDetectionPass (base class)
      └─> FilePassAggregates (aggregate management)

BubblesFieldDetection
  └─> FieldDetection (base class)
      └─> run_detection() → BubbleFieldDetectionResult
```

### Key Classes

1. **BubblesThresholdDetectionPass** (`detection_pass.py`)
   - Orchestrates detection for all bubble fields in a file
   - Manages file-level and directory-level aggregates
   - Integrates with `DetectionRepository` for storage

2. **BubblesFieldDetection** (`detection.py`)
   - Performs detection for a single field
   - Extracts mean intensity from each bubble ROI
   - Returns typed `BubbleFieldDetectionResult`

3. **BubbleFieldDetectionResult** (`detection_results.py`)
   - Typed model replacing dictionary-based results
   - Auto-calculated properties (std, quality, jumps)
   - Immutable detection record

---

## Detection Flow

### Phase 1: Initialization

```python
# Called by TemplateFileRunner
detection_pass = BubblesThresholdDetectionPass(
    repository=repository,
    tuning_config=tuning_config,
    field_detection_type=FieldType.BUBBLES_THRESHOLD
)

# Initialize directory-level aggregates (once per directory)
detection_pass.initialize_directory_level_aggregates(directory_path)

# Initialize file-level aggregates (once per file)
detection_pass.initialize_file_level_aggregates(file_path)
```

**Directory-Level Aggregates:**
```python
{
    "file_wise_thresholds": NumberAggregate()  # Track threshold stats
}
```

**File-Level Aggregates:**
```python
{
    "global_max_jump": None,              # Maximum jump across all fields
    "all_field_bubble_means": [],         # All bubble means in file
    "all_field_bubble_means_std": []      # Std deviation per field
}
```

---

### Phase 2: Field-Level Detection

```python
# For each bubble field in the template:
field_detection = detection_pass.run_field_level_detection(
    field=field,
    gray_image=gray_image,
    colored_image=colored_image  # Not used for bubble detection
)
```

**Step-by-step:**

1. **Initialize Field Aggregates**
   ```python
   detection_pass.initialize_field_level_aggregates(field)
   ```

2. **Create Field Detection Instance**
   ```python
   field_detection = BubblesFieldDetection(field, gray_image, colored_image)
   ```

3. **Run Detection**
   ```python
   field_detection.run_detection(field, gray_image, colored_image)
   ```

4. **Update Aggregates**
   ```python
   detection_pass.update_aggregates_on_processed_field_detection(field, field_detection)
   ```

---

### Phase 3: Bubble ROI Extraction

**Core Algorithm** (in `BubblesFieldDetection.run_detection()`):

```python
def run_detection(self, field: Field, gray_image, _colored_image) -> None:
    bubble_means = []

    # For each bubble in the field
    for unit_bubble in field.scan_boxes:
        # 1. Extract mean intensity value
        bubble_mean_value = self.read_bubble_mean_value(unit_bubble, gray_image)
        bubble_means.append(bubble_mean_value)

    # 2. Create typed result with auto-calculated properties
    self.result = BubbleFieldDetectionResult(
        field_id=field.id,
        field_label=field.field_label,
        bubble_means=bubble_means
    )
```

**Single Bubble Measurement:**

```python
@staticmethod
def read_bubble_mean_value(
    unit_bubble: BubblesScanBox,
    gray_image
) -> BubbleMeanValue:
    # 1. Get bubble dimensions and position
    box_w, box_h = unit_bubble.bubble_dimensions
    x, y = unit_bubble.get_shifted_position()  # Accounts for alignment shifts

    # 2. Extract ROI rectangle [y_start, y_end, x_start, x_end]
    rect = [y, y + box_h, x, x + box_w]

    # 3. Calculate mean pixel intensity (0-255)
    mean_value = cv2.mean(gray_image[rect[0]:rect[1], rect[2]:rect[3]], None)[0]

    # 4. Return typed result
    return BubbleMeanValue(
        mean_value=mean_value,
        unit_bubble=unit_bubble,
        position=(x, y)
    )
```

**Key Points:**
- Uses `get_shifted_position()` to account for alignment corrections
- Extracts ROI using NumPy array slicing: `gray_image[y:y+h, x:x+w]`
- `cv2.mean()` returns tuple; we use `[0]` for single-channel grayscale
- **Lower values = darker pixels = marked bubbles**

---

### Phase 4: Repository Storage

```python
# Save detection result to repository
if field_detection.result is None:
    raise ValueError(f"field_detection.result is None for field {field.id}")

self.repository.save_bubble_field(field.id, field_detection.result)
```

**Repository Structure:**

```python
class DetectionRepository:
    def save_bubble_field(self, field_id: str, result: BubbleFieldDetectionResult):
        """Store detection result for a field"""
        current_file = self._get_current_file()
        current_file.bubble_fields[field_id] = result

    def get_bubble_field(self, field_id: str) -> BubbleFieldDetectionResult:
        """Retrieve detection result"""
        current_file = self._get_current_file()
        return current_file.bubble_fields[field_id]
```

---

### Phase 5: Aggregate Updates

**Field-Level Aggregates:**

```python
def update_field_level_aggregates_on_processed_field_detection(
    self, field: Field, field_detection: BubblesFieldDetection
) -> None:
    # Extract auto-calculated properties
    field_bubble_means = field_detection.result.bubble_means
    std_deviation = field_detection.result.std_deviation

    self.insert_field_level_aggregates({
        "field_bubble_means": field_bubble_means,
        "field_bubble_means_std": std_deviation
    })
```

**File-Level Aggregates:**

```python
def update_file_level_aggregates_on_processed_field_detection(
    self, field, field_detection, field_level_aggregates
) -> None:
    # Update field count
    file_level_aggregates["fields_count"].push("processed")

    # Collect all bubble means for global threshold calculation
    field_bubble_means = field_level_aggregates["field_bubble_means"]
    field_bubble_means_std = field_level_aggregates["field_bubble_means_std"]

    file_level_aggregates["all_field_bubble_means"].extend(field_bubble_means)
    file_level_aggregates["all_field_bubble_means_std"].append(field_bubble_means_std)
```

---

## Data Models

### BubbleMeanValue

**Purpose**: Represents a single bubble measurement

```python
@dataclass
class BubbleMeanValue:
    mean_value: float              # 0-255 (darker = lower value)
    unit_bubble: BubblesScanBox    # Reference to bubble definition
    position: tuple[int, int]      # (x, y) position in image

    def __lt__(self, other: "BubbleMeanValue") -> bool:
        """Enable sorting by mean value"""
        return self.mean_value < other.mean_value
```

**Example:**
```python
BubbleMeanValue(
    mean_value=78.5,  # Dark (marked)
    unit_bubble=BubblesScanBox(...),
    position=(120, 450)
)
```

---

### BubbleFieldDetectionResult

**Purpose**: Complete detection result for a field with auto-calculated properties

```python
@dataclass
class BubbleFieldDetectionResult:
    field_id: str                      # "field_block_1::q1"
    field_label: str                   # "q1"
    bubble_means: list[BubbleMeanValue]
    timestamp: datetime                # Auto-generated

    # Auto-calculated properties:

    @property
    def std_deviation(self) -> float:
        """Standard deviation of bubble means"""
        values = [bm.mean_value for bm in self.bubble_means]
        return float(np.std(values))

    @property
    def scan_quality(self) -> ScanQuality:
        """Quality based on std deviation"""
        std = self.std_deviation
        if std > 50:   return ScanQuality.EXCELLENT
        if std > 30:   return ScanQuality.GOOD
        if std > 15:   return ScanQuality.ACCEPTABLE
        return ScanQuality.POOR

    @property
    def mean_values(self) -> list[float]:
        """Just the float values"""
        return [bm.mean_value for bm in self.bubble_means]

    @property
    def sorted_mean_values(self) -> list[float]:
        """Sorted values for threshold calculation"""
        return [bm.mean_value for bm in sorted(self.bubble_means)]

    @property
    def jumps(self) -> list[tuple[float, BubbleMeanValue]]:
        """Gaps between consecutive sorted values"""
        sorted_means = sorted(self.bubble_means)
        return [
            (round(sorted_means[i].mean_value - sorted_means[i-1].mean_value, 2),
             sorted_means[i-1])
            for i in range(1, len(sorted_means))
        ]
```

**Example:**
```python
result = BubbleFieldDetectionResult(
    field_id="field_block_1::q1",
    field_label="q1",
    bubble_means=[
        BubbleMeanValue(78.5, ...),   # Marked
        BubbleMeanValue(210.2, ...),  # Unmarked
        BubbleMeanValue(215.8, ...),  # Unmarked
        BubbleMeanValue(208.1, ...)   # Unmarked
    ]
)

# Auto-calculated:
result.std_deviation  # ~65.3 (high variance = good)
result.scan_quality   # ScanQuality.EXCELLENT
result.sorted_mean_values  # [78.5, 208.1, 210.2, 215.8]
result.jumps  # [(129.6, first_bubble), (2.1, second), (5.6, third)]
result.max_jump  # 129.6
```

---

## Scan Quality Assessment

**Automatic Quality Detection:**

```python
enum ScanQuality:
    EXCELLENT   # std > 50  → Clear marked/unmarked distinction
    GOOD        # std > 30  → Acceptable distinction
    ACCEPTABLE  # std > 15  → Marginal, may need review
    POOR        # std ≤ 15  → Likely errors, all similar values
```

**Quality Implications:**

| Quality | Std Dev | Typical Scenario | Threshold Strategy |
|---------|---------|------------------|-------------------|
| EXCELLENT | > 50 | 1 marked, rest unmarked | Local threshold reliable |
| GOOD | 30-50 | Clear marks, good contrast | Local threshold works |
| ACCEPTABLE | 15-30 | Faded marks or uneven scan | May need global fallback |
| POOR | < 15 | All unmarked or all marked | Use global threshold |

---

## Shifted Position Calculation

**Why Shifts Matter:**

After alignment, the actual bubble positions may differ from template positions. The detection pass uses shifted positions to account for:
1. **Alignment corrections** (SIFT/template matching)
2. **Field block shifts** (ML-based shift detection)
3. **Preprocessing transformations** (rotation, cropping)

**Calculation:**

```python
class BubblesScanBox(ScanBox):
    def get_shifted_position(self) -> tuple[int, int]:
        """Get position accounting for all shifts"""
        # Base position from template
        x, y = self.x, self.y

        # Add field-level shifts
        x += self.shifts[0]
        y += self.shifts[1]

        # Add field block shifts (propagated from parent)
        x += self.field.field_block.shifts[0]
        y += self.field.field_block.shifts[1]

        return (round(x), round(y))
```

**Shift Sources:**

```
Template Position (from JSON)
  → + Preprocessing shift (rotation, crop)
  → + Alignment shift (SIFT/template matching)
  → + Field Block shift (ML detection)
  → = Final Shifted Position
```

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                           │
├─────────────────────────────────────────────────────────────┤
│ - Initialize directory aggregates (once per directory)      │
│ - Initialize file aggregates (once per file)                │
│ - Create DetectionRepository                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. FIELD ITERATION (for each bubble field)                  │
├─────────────────────────────────────────────────────────────┤
│ - Initialize field aggregates                               │
│ - Create BubblesFieldDetection instance                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. BUBBLE MEASUREMENT (for each bubble in field)            │
├─────────────────────────────────────────────────────────────┤
│ a. Get shifted position (template + alignment + ML shifts)  │
│ b. Extract ROI: gray_image[y:y+h, x:x+w]                   │
│ c. Calculate mean: cv2.mean(roi)[0]                        │
│ d. Create BubbleMeanValue(mean, bubble, position)          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. RESULT CREATION                                          │
├─────────────────────────────────────────────────────────────┤
│ - Create BubbleFieldDetectionResult                         │
│ - Auto-calculate: std_deviation, scan_quality, jumps       │
│ - Validate result is not None                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. REPOSITORY STORAGE                                       │
├─────────────────────────────────────────────────────────────┤
│ - Save to repository.save_bubble_field(field_id, result)   │
│ - Typed storage for interpretation pass access             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. AGGREGATE UPDATES                                        │
├─────────────────────────────────────────────────────────────┤
│ Field-level:                                                │
│   - field_bubble_means, field_bubble_means_std             │
│ File-level:                                                 │
│   - all_field_bubble_means (for global threshold)          │
│   - all_field_bubble_means_std                             │
│ Directory-level:                                            │
│   - file_wise_thresholds statistics                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Browser Implementation

### OpenCV.js ROI Extraction

```typescript
function readBubbleMeanValue(
  unitBubble: BubblesScanBox,
  grayImage: cv.Mat
): BubbleMeanValue {
  const [boxW, boxH] = unitBubble.bubbleDimensions;
  const [x, y] = unitBubble.getShiftedPosition();

  // Extract ROI
  const roi = grayImage.roi(new cv.Rect(x, y, boxW, boxH));

  // Calculate mean
  const mean = cv.mean(roi);
  const meanValue = mean[0];  // First channel (grayscale)

  // CRITICAL: Free memory
  roi.delete();

  return {
    meanValue,
    unitBubble,
    position: [x, y]
  };
}
```

### TypeScript Models

```typescript
interface BubbleMeanValue {
  meanValue: number;              // 0-255
  unitBubble: BubblesScanBox;
  position: [number, number];
}

interface BubbleFieldDetectionResult {
  fieldId: string;
  fieldLabel: string;
  bubbleMeans: BubbleMeanValue[];
  timestamp: Date;

  // Computed properties
  get stdDeviation(): number;
  get scanQuality(): ScanQuality;
  get meanValues(): number[];
  get sortedMeanValues(): number[];
  get jumps(): Array<[number, BubbleMeanValue]>;
  get maxJump(): number;
}

enum ScanQuality {
  EXCELLENT = "excellent",
  GOOD = "good",
  ACCEPTABLE = "acceptable",
  POOR = "poor"
}
```

### TypeScript Implementation

```typescript
class BubblesFieldDetection {
  result?: BubbleFieldDetectionResult;

  runDetection(field: Field, grayImage: cv.Mat): void {
    const bubbleMeans: BubbleMeanValue[] = [];

    // Extract each bubble
    for (const unitBubble of field.scanBoxes) {
      const bubbleMean = this.readBubbleMeanValue(unitBubble, grayImage);
      bubbleMeans.push(bubbleMean);
    }

    // Create result with computed properties
    this.result = new BubbleFieldDetectionResultImpl(
      field.id,
      field.fieldLabel,
      bubbleMeans
    );
  }

  private readBubbleMeanValue(
    unitBubble: BubblesScanBox,
    grayImage: cv.Mat
  ): BubbleMeanValue {
    const [boxW, boxH] = unitBubble.bubbleDimensions;
    const [x, y] = unitBubble.getShiftedPosition();

    const roi = grayImage.roi(new cv.Rect(x, y, boxW, boxH));
    const mean = cv.mean(roi);
    const meanValue = mean[0];

    roi.delete();  // Memory management

    return { meanValue, unitBubble, position: [x, y] };
  }
}
```

### Computed Properties Implementation

```typescript
class BubbleFieldDetectionResultImpl implements BubbleFieldDetectionResult {
  constructor(
    public fieldId: string,
    public fieldLabel: string,
    public bubbleMeans: BubbleMeanValue[],
    public timestamp: Date = new Date()
  ) {}

  get stdDeviation(): number {
    if (this.bubbleMeans.length === 0) return 0;

    const values = this.bubbleMeans.map(bm => bm.meanValue);
    const mean = values.reduce((a, b) => a + b) / values.length;
    const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;

    return Math.sqrt(variance);
  }

  get scanQuality(): ScanQuality {
    const std = this.stdDeviation;
    if (std > 50) return ScanQuality.EXCELLENT;
    if (std > 30) return ScanQuality.GOOD;
    if (std > 15) return ScanQuality.ACCEPTABLE;
    return ScanQuality.POOR;
  }

  get sortedMeanValues(): number[] {
    return [...this.bubbleMeans]
      .sort((a, b) => a.meanValue - b.meanValue)
      .map(bm => bm.meanValue);
  }

  get jumps(): Array<[number, BubbleMeanValue]> {
    const sorted = [...this.bubbleMeans].sort((a, b) => a.meanValue - b.meanValue);

    return sorted.slice(1).map((bubble, i) => {
      const jump = bubble.meanValue - sorted[i].meanValue;
      return [Math.round(jump * 100) / 100, sorted[i]];
    });
  }
}
```

---

## Integration Points

### Called By

- **TemplateFileRunner**: Main orchestrator for file processing
- **ProcessingPipeline**: Part of detection stage

### Calls

- **DetectionRepository**: Storage and retrieval of results
- **cv2.mean()**: OpenCV function for mean calculation
- **BubblesScanBox.get_shifted_position()**: Position calculation

### Produces

- **BubbleFieldDetectionResult**: Typed detection results
- **File-level aggregates**: For global threshold calculation
- **Repository entries**: For interpretation pass consumption

---

## Performance Characteristics

### Time Complexity

- **Per bubble**: O(w × h) where w×h = bubble dimensions
- **Per field**: O(n × w × h) where n = number of bubbles
- **Per file**: O(f × n × w × h) where f = number of fields

**Typical Values:**
- Bubble dimensions: 30×30 pixels
- Bubbles per field: 4-10
- Fields per file: 10-50
- **Total operations per file**: ~4,000 - 450,000 pixel operations

### Space Complexity

- **Per field**: O(n) where n = number of bubbles
- **Per file**: O(f × n) where f = fields
- **Typical**: ~200-500 BubbleMeanValue objects per file

### Browser Optimization

```typescript
// Memory-efficient batch processing
function detectAllFields(
  fields: Field[],
  grayImage: cv.Mat
): Map<string, BubbleFieldDetectionResult> {
  const results = new Map();

  for (const field of fields) {
    const detection = new BubblesFieldDetection();
    detection.runDetection(field, grayImage);
    results.set(field.id, detection.result!);

    // No need to delete grayImage here (shared across fields)
  }

  return results;
}
```

---

## Error Handling

### Common Errors

1. **None Result Error**
   ```python
   if field_detection.result is None:
       raise ValueError(f"field_detection.result is None for field {field.id}")
   ```

2. **Invalid ROI**
   - Bubble position outside image bounds
   - Negative dimensions
   - **Solution**: Validate shifted positions before extraction

3. **Empty Field**
   - Field has no scan boxes
   - **Solution**: Skip field or use default result

### Browser Error Handling

```typescript
function readBubbleMeanValue(
  unitBubble: BubblesScanBox,
  grayImage: cv.Mat
): BubbleMeanValue | null {
  try {
    const [boxW, boxH] = unitBubble.bubbleDimensions;
    const [x, y] = unitBubble.getShiftedPosition();

    // Validate bounds
    if (x < 0 || y < 0 || x + boxW > grayImage.cols || y + boxH > grayImage.rows) {
      console.warn(`Bubble ROI out of bounds: (${x},${y}) ${boxW}x${boxH}`);
      return null;
    }

    const roi = grayImage.roi(new cv.Rect(x, y, boxW, boxH));
    const mean = cv.mean(roi);
    const meanValue = mean[0];

    roi.delete();

    return { meanValue, unitBubble, position: [x, y] };
  } catch (error) {
    console.error(`Error reading bubble mean: ${error}`);
    return null;
  }
}
```

---

## Related Documentation

- **Parent**: `/bubbles-threshold/concept.md` - Overall bubble detection architecture
- **Next Step**: `/bubbles-threshold/interpretation-pass/flows.md` - Threshold application
- **Models**: `src/processors/detection/models/detection_results.py` - Data models
- **Repository**: `src/processors/repositories/detection_repository.py` - Storage
- **Integration**: `/detection/concept.md` - Main detection orchestrator

---

## Key Takeaways

1. **Pure Computer Vision**: No interpretation logic, just measurements
2. **Typed Models**: Strongly-typed results with auto-calculated properties
3. **Repository Pattern**: Decoupled storage for interpretation pass
4. **Shift-Aware**: Uses shifted positions from alignment/ML detection
5. **Quality Assessment**: Automatic scan quality calculation
6. **Browser-Ready**: Direct OpenCV.js mapping with memory management
7. **Performance**: O(n × w × h) per file, highly parallelizable
