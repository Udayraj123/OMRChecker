# Bubble Detection - Integration with ReadOMR

**Status**: Integration Documentation
**Python Reference**: `src/processors/detection/processor.py`, `src/processors/detection/template_file_runner.py`

---

## Overview

This document covers how the **Bubble Detection Threshold Strategy** integrates with the ReadOMR processor and the broader OMRChecker pipeline.

---

## Integration Architecture

```
ProcessingPipeline
  └─> ReadOMRProcessor
      └─> TemplateFileRunner
          ├─> DetectionRepository (shared state)
          ├─> BubblesThresholdDetectionPass
          │   └─> BubblesFieldDetection (per field)
          │       └─> read_bubble_mean_value (per bubble)
          └─> BubblesThresholdInterpretationPass
              └─> BubblesFieldInterpretation (per field)
                  ├─> GlobalThresholdStrategy (file-level)
                  └─> LocalThresholdStrategy (field-level)
```

---

## Integration Point 1: ReadOMRProcessor

### Processor Initialization

```python
class ReadOMRProcessor(Processor):
    def __init__(self, template, ml_model_path: str | Path | None = None):
        self.template = template
        self.tuning_config = template.tuning_config

        # Create TemplateFileRunner (contains detection/interpretation logic)
        self.template_file_runner = TemplateFileRunner(template)

        # Optional ML fallback
        self.ml_detector = None
        self.hybrid_strategy = None
        if ml_model_path:
            self._initialize_ml_fallback(ml_model_path)
```

### Process Method

```python
def process(self, context: ProcessingContext) -> ProcessingContext:
    """Execute OMR detection and interpretation."""

    # 1. Extract inputs from context
    template = context.template
    file_path = context.file_path
    input_gray_image = context.gray_image
    colored_image = context.colored_image

    # 2. Resize to template dimensions
    gray_image, colored_image = ImageUtils.resize_to_dimensions(
        template.template_dimensions,
        input_gray_image,
        colored_image
    )

    # 3. Normalize images
    gray_image, colored_image = ImageUtils.normalize(gray_image, colored_image)

    # 4. Run detection and interpretation via TemplateFileRunner
    raw_omr_response = self.template_file_runner.read_omr_and_update_metrics(
        file_path, gray_image, colored_image
    )

    # 5. Get concatenated response (handles custom labels)
    concatenated_omr_response = template.get_concatenated_omr_response(
        raw_omr_response
    )

    # 6. Extract interpretation metrics
    directory_level_interpretation_aggregates = (
        self.template_file_runner.get_directory_level_interpretation_aggregates()
    )

    template_file_level_interpretation_aggregates = (
        directory_level_interpretation_aggregates["file_wise_aggregates"][file_path]
    )

    is_multi_marked = template_file_level_interpretation_aggregates[
        "read_response_flags"
    ]["is_multi_marked"]

    field_id_to_interpretation = template_file_level_interpretation_aggregates[
        "field_id_to_interpretation"
    ]

    # 7. Update context with results
    context.omr_response = concatenated_omr_response
    context.is_multi_marked = is_multi_marked
    context.field_id_to_interpretation = field_id_to_interpretation
    context.gray_image = gray_image
    context.colored_image = colored_image

    # 8. Store raw response and aggregates in metadata
    context.metadata["raw_omr_response"] = raw_omr_response
    context.metadata["directory_level_interpretation_aggregates"] = (
        directory_level_interpretation_aggregates
    )

    # 9. Check for low-confidence fields and use ML fallback if needed
    if self.hybrid_strategy and self.hybrid_strategy.should_use_ml_fallback(context):
        logger.info(f"Using ML fallback for low-confidence fields")
        self.ml_detector.enable_for_low_confidence()
        context = self.ml_detector.process(context)
        self.ml_detector.disable()
        self.hybrid_strategy.stats["ml_fallback_used"] += 1

    return context
```

---

## Integration Point 2: TemplateFileRunner

### Initialization

```python
class TemplateFileRunner:
    def __init__(self, template):
        self.template = template
        self.tuning_config = template.tuning_config

        # Initialize DetectionRepository (shared state)
        self.repository = DetectionRepository()

        # Initialize detection passes (one per field type)
        self.detection_passes = {
            FieldType.BUBBLES_THRESHOLD: BubblesThresholdDetectionPass(
                repository=self.repository,
                tuning_config=self.tuning_config
            ),
            FieldType.OCR: OCRDetectionPass(...),
            FieldType.BARCODE_QR: BarcodeDetectionPass(...),
        }

        # Initialize interpretation passes
        self.interpretation_passes = {
            FieldType.BUBBLES_THRESHOLD: BubblesThresholdInterpretationPass(
                repository=self.repository,
                tuning_config=self.tuning_config
            ),
            FieldType.OCR: OCRInterpretationPass(...),
            FieldType.BARCODE_QR: BarcodeInterpretationPass(...),
        }
```

### Main Entry Point

```python
def read_omr_and_update_metrics(
    self, file_path: str, gray_image, colored_image
) -> dict:
    """Main entry point for OMR detection and interpretation."""

    # 1. Initialize repository for this file
    self.repository.new_file(file_path)

    # 2. Initialize file-level aggregates for all passes
    for detection_pass in self.detection_passes.values():
        detection_pass.initialize_file_level_aggregates(file_path)

    # 3. Run DETECTION pass for all fields
    for field_block in self.template.field_blocks:
        for field in field_block.fields:
            detection_pass = self.detection_passes[field.field_type]
            detection_pass.process_field(field, gray_image, colored_image)

    # 4. Initialize file-level aggregates for interpretation
    for interpretation_pass in self.interpretation_passes.values():
        interpretation_pass.initialize_file_level_aggregates(file_path)

    # 5. Run INTERPRETATION pass for all fields
    for field_block in self.template.field_blocks:
        for field in field_block.fields:
            interpretation_pass = self.interpretation_passes[field.field_type]
            interpretation_pass.process_field(field, file_level_aggregates)

    # 6. Collect and return results
    return self._collect_omr_response(file_path)
```

---

## Integration Point 3: DetectionRepository

### Repository Pattern

The `DetectionRepository` acts as a **centralized store** for detection results, decoupling detection from interpretation.

```python
class DetectionRepository:
    """Centralized repository for typed detection results."""

    def __init__(self):
        self.current_file_path: str | None = None
        self.file_results: dict[str, FileDetectionResults] = {}

    def new_file(self, file_path: str):
        """Initialize repository for a new file."""
        self.current_file_path = file_path
        self.file_results[file_path] = FileDetectionResults(file_path=file_path)

    def save_bubble_field(self, field_id: str, result: BubbleFieldDetectionResult):
        """Save bubble field detection result."""
        if self.current_file_path is None:
            raise ValueError("No current file set in repository")

        file_results = self.file_results[self.current_file_path]
        file_results.bubble_fields[field_id] = result

    def get_bubble_field(self, field_id: str) -> BubbleFieldDetectionResult | None:
        """Retrieve bubble field detection result."""
        if self.current_file_path is None:
            return None

        file_results = self.file_results.get(self.current_file_path)
        if file_results is None:
            return None

        return file_results.bubble_fields.get(field_id)

    def get_all_bubble_means_for_current_file(self) -> list[BubbleMeanValue]:
        """Get all bubble means across all fields in current file."""
        if self.current_file_path is None:
            return []

        file_results = self.file_results.get(self.current_file_path)
        if file_results is None:
            return []

        return file_results.all_bubble_means

    def get_all_bubble_fields_for_current_file(self) -> dict[str, BubbleFieldDetectionResult]:
        """Get all bubble field results for current file."""
        if self.current_file_path is None:
            return {}

        file_results = self.file_results.get(self.current_file_path)
        if file_results is None:
            return {}

        return file_results.bubble_fields
```

### Data Flow

```
Detection Pass:
  Field 1 → BubbleFieldDetectionResult → Repository.save_bubble_field()
  Field 2 → BubbleFieldDetectionResult → Repository.save_bubble_field()
  ...
  Field N → BubbleFieldDetectionResult → Repository.save_bubble_field()

Interpretation Pass:
  Repository.get_all_bubble_means_for_current_file()
    ├─> Calculate Global Threshold
    └─> Store in file_level_aggregates

  For each field:
    Repository.get_bubble_field(field_id)
      ├─> Calculate Local Threshold
      └─> Interpret bubbles
```

---

## Integration Point 4: Field Type Dispatch

### Field Type Enum

```python
class FieldType(str, Enum):
    BUBBLES_THRESHOLD = "BUBBLES_THRESHOLD"
    OCR = "OCR"
    BARCODE_QR = "BARCODE_QR"
```

### Field Type Routing

```python
# In TemplateFileRunner.read_omr_and_update_metrics()

for field_block in self.template.field_blocks:
    for field in field_block.fields:
        # Route to correct detection pass based on field type
        detection_pass = self.detection_passes[field.field_type]
        detection_pass.process_field(field, gray_image, colored_image)

        # Later, route to correct interpretation pass
        interpretation_pass = self.interpretation_passes[field.field_type]
        interpretation_pass.process_field(field, file_level_aggregates)
```

### Example Field Definitions

```json
{
  "fieldBlocks": [
    {
      "name": "MCQ_Block_1",
      "fieldType": "BUBBLES_THRESHOLD",
      "origin": [100, 200],
      "fieldLabels": ["q1", "q2", "q3"],
      "bubblesGap": 30
    },
    {
      "name": "StudentID_Block",
      "fieldType": "OCR",
      "origin": [500, 100]
    },
    {
      "name": "Barcode_Block",
      "fieldType": "BARCODE_QR",
      "origin": [700, 100]
    }
  ]
}
```

---

## Integration Point 5: ProcessingContext

### Context Flow

```python
# Input context (from preprocessing/alignment)
context = ProcessingContext(
    template=template,
    file_path="inputs/sheet1.jpg",
    gray_image=gray_image,
    colored_image=colored_image,
    # ... other fields
)

# After ReadOMRProcessor.process()
context.omr_response = {
    "q1": "A",
    "q2": "C",
    "q3": "B",
    "StudentID": "12345",
    "Barcode": "ABC-123"
}
context.is_multi_marked = False
context.field_id_to_interpretation = {
    "q1": { "interpretation": "A", "local_threshold": 118.5, ... },
    "q2": { "interpretation": "C", "local_threshold": 121.3, ... },
    "q3": { "interpretation": "B", "local_threshold": 119.8, ... },
}
context.metadata = {
    "raw_omr_response": { ... },
    "directory_level_interpretation_aggregates": { ... }
}
```

### Context Fields Used

| Field | Usage | Set By |
|-------|-------|--------|
| `template` | Field definitions, config | Pipeline |
| `file_path` | Current file path | Pipeline |
| `gray_image` | Grayscale image for detection | ReadOMR |
| `colored_image` | Colored image for visualization | ReadOMR |
| `omr_response` | Final interpreted values | ReadOMR |
| `is_multi_marked` | Multi-mark flag | ReadOMR |
| `field_id_to_interpretation` | Per-field details | ReadOMR |
| `metadata` | Raw results, aggregates | ReadOMR |

---

## Integration Point 6: ML Fallback Integration

### Hybrid Detection Strategy

```python
class HybridDetectionStrategy:
    def __init__(self, ml_detector, confidence_threshold=0.75):
        self.ml_detector = ml_detector
        self.confidence_threshold = confidence_threshold
        self.stats = {
            "total_fields": 0,
            "high_confidence_fields": 0,
            "low_confidence_fields": 0,
            "ml_fallback_used": 0
        }

    def should_use_ml_fallback(self, context: ProcessingContext) -> bool:
        """Determine if ML fallback should be used."""

        # Check overall confidence
        overall_confidence = context.metadata.get("overall_confidence_score", 1.0)
        if overall_confidence < self.confidence_threshold:
            return True

        # Check for multi-marking
        if context.is_multi_marked:
            return True

        # Check for threshold disparity
        aggregates = context.metadata.get(
            "directory_level_interpretation_aggregates", {}
        )
        file_aggregates = aggregates.get("file_wise_aggregates", {}).get(
            context.file_path, {}
        )
        confidence_metrics = file_aggregates.get("field_wise_confidence_metrics", {})

        for field_id, metrics in confidence_metrics.items():
            disparity_bubbles = metrics.get("bubbles_in_doubt", {}).get(
                "by_disparity", []
            )
            if len(disparity_bubbles) > 0:
                return True

        return False
```

### Integration in ReadOMRProcessor

```python
def process(self, context: ProcessingContext) -> ProcessingContext:
    # ... normal detection/interpretation ...

    # Check for low-confidence fields and use ML fallback if needed
    if self.hybrid_strategy and self.hybrid_strategy.should_use_ml_fallback(context):
        logger.info("Using ML fallback for low-confidence fields")
        self.ml_detector.enable_for_low_confidence()
        context = self.ml_detector.process(context)
        self.ml_detector.disable()
        self.hybrid_strategy.stats["ml_fallback_used"] += 1

    return context
```

---

## Integration Point 7: Visualization & Debug Output

### Debug Image Generation

```python
# In BubblesFieldInterpretationDrawing

def draw(self, colored_image, field):
    """Draw bubble interpretations on colored image."""

    for bubble_interp in self.field_interpretation.bubble_interpretations:
        bubble = bubble_interp.item_reference
        x, y = bubble.get_shifted_position()
        w, h = bubble.bubble_dimensions

        # Color based on interpretation
        if bubble_interp.is_attempted:
            color = (0, 255, 0)  # Green for marked
        else:
            color = (255, 0, 0)  # Blue for unmarked

        # Draw rectangle
        cv2.rectangle(colored_image, (x, y), (x+w, y+h), color, 2)

        # Draw mean value
        cv2.putText(
            colored_image,
            f"{round(bubble_interp.mean_value, 1)}",
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

    # Draw threshold line (optional)
    threshold = self.field_interpretation.local_threshold_for_field
    cv2.putText(
        colored_image,
        f"THR: {round(threshold, 1)}",
        (field.origin[0], field.origin[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )
```

### Save Image Ops

```python
# In Template.save_image_ops

template.save_image_ops.append_save_image(
    "Bubble Detection",
    range(3, 7),  # Debug levels 3-6
    gray_image,
    colored_image_with_bubbles
)
```

---

## Integration Point 8: CSV Export

### OMR Response to CSV

```python
# In CSVWriter

def write_response(self, file_path: str, omr_response: dict):
    """Write OMR response to CSV."""

    row = {
        "file_path": file_path,
        **omr_response,  # Spread all field responses
    }

    # Add multi-mark flag
    if is_multi_marked:
        row["_multi_marked"] = True

    self.csv_writer.writerow(row)
```

### Example CSV Output

```csv
file_path,q1,q2,q3,q4,q5,StudentID,_multi_marked
inputs/sheet1.jpg,A,C,B,D,A,12345,False
inputs/sheet2.jpg,B,D,C,A,B,67890,False
inputs/sheet3.jpg,AB,C,B,D,A,11111,True
```

---

## Integration Point 9: Evaluation System

### Answer Key Matching

```python
# In EvaluationProcessor

def process(self, context: ProcessingContext) -> ProcessingContext:
    """Evaluate OMR response against answer key."""

    omr_response = context.omr_response
    answer_key = self.evaluation_config.answer_key

    score = 0
    for field_id, detected_value in omr_response.items():
        correct_answer = answer_key.get(field_id)

        if detected_value == correct_answer:
            score += self.marking_scheme.correct_marks
        elif detected_value == "":
            score += self.marking_scheme.unmarked_marks
        else:
            score += self.marking_scheme.incorrect_marks

    context.score = score
    context.metadata["evaluation"] = {
        "score": score,
        "total": len(answer_key) * self.marking_scheme.correct_marks
    }

    return context
```

---

## Integration Point 10: Browser Migration

### Web Worker Integration

```typescript
// Main thread
class BubbleDetectionController {
  private worker: Worker;

  constructor() {
    this.worker = new Worker('/workers/bubble-detection.worker.js');
  }

  async detectBubbles(
    fields: Field[],
    grayImageData: ImageData
  ): Promise<BubbleFieldDetectionResult[]> {
    return new Promise((resolve, reject) => {
      this.worker.postMessage({
        type: 'DETECT_BUBBLES',
        fields,
        grayImageData
      });

      this.worker.onmessage = (event) => {
        if (event.data.type === 'DETECTION_COMPLETE') {
          resolve(event.data.results);
        }
      };

      this.worker.onerror = reject;
    });
  }
}
```

### Worker Implementation

```typescript
// bubble-detection.worker.ts
importScripts('opencv.js');

self.onmessage = async (event) => {
  const { type, fields, grayImageData } = event.data;

  if (type === 'DETECT_BUBBLES') {
    // Convert ImageData to cv.Mat
    const grayMat = cv.matFromImageData(grayImageData);

    // Detect bubbles
    const results: BubbleFieldDetectionResult[] = [];
    for (const field of fields) {
      const result = detectBubbleField(field, grayMat);
      results.push(result);
    }

    // Cleanup
    grayMat.delete();

    // Send results back
    self.postMessage({
      type: 'DETECTION_COMPLETE',
      results
    });
  }
};

function detectBubbleField(
  field: Field,
  grayImage: cv.Mat
): BubbleFieldDetectionResult {
  // Same logic as Python
  // ...
}
```

### IndexedDB Storage

```typescript
// Store detection results for offline use
class DetectionCache {
  private db: IDBDatabase;

  async saveDetectionResults(
    fileId: string,
    results: FileDetectionResults
  ): Promise<void> {
    const tx = this.db.transaction(['detections'], 'readwrite');
    const store = tx.objectStore('detections');
    await store.put({ fileId, results, timestamp: Date.now() });
  }

  async getDetectionResults(
    fileId: string
  ): Promise<FileDetectionResults | null> {
    const tx = this.db.transaction(['detections'], 'readonly');
    const store = tx.objectStore('detections');
    const result = await store.get(fileId);
    return result?.results || null;
  }
}
```

---

## Complete Integration Example

### Python

```python
# Full pipeline integration
def process_omr_sheet(file_path: str, template: Template):
    # 1. Create processing context
    context = ProcessingContext(
        template=template,
        file_path=file_path,
        gray_image=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE),
        colored_image=cv2.imread(file_path)
    )

    # 2. Preprocessing
    preprocessor = PreprocessingCoordinator(template)
    context = preprocessor.process(context)

    # 3. Alignment
    aligner = AlignmentProcessor(template)
    context = aligner.process(context)

    # 4. OMR Detection (includes bubble detection)
    detector = ReadOMRProcessor(template)
    context = detector.process(context)

    # 5. Evaluation (optional)
    if template.evaluation_config:
        evaluator = EvaluationProcessor(template.evaluation_config)
        context = evaluator.process(context)

    # 6. Return results
    return context.omr_response, context.score
```

### TypeScript

```typescript
async function processOMRSheet(
  file: File,
  template: Template
): Promise<{ omrResponse: Record<string, string>; score?: number }> {
  // 1. Read file as image
  const image = await loadImage(file);
  const grayImage = convertToGrayscale(image);

  // 2. Preprocessing
  const preprocessor = new PreprocessingCoordinator(template);
  const { grayImage: processedGray, coloredImage: processedColor } =
    await preprocessor.process(grayImage, image);

  // 3. Alignment
  const aligner = new AlignmentProcessor(template);
  const { grayImage: alignedGray, coloredImage: alignedColor } =
    await aligner.process(processedGray, processedColor);

  // 4. OMR Detection (Web Worker)
  const detector = new BubbleDetectionController();
  const detectionResults = await detector.detectBubbles(
    template.fields,
    alignedGray
  );

  // 5. Interpretation (main thread)
  const interpreter = new BubbleInterpretationEngine();
  const omrResponse = interpreter.interpret(detectionResults, template.config);

  // 6. Evaluation (optional)
  let score: number | undefined;
  if (template.evaluationConfig) {
    const evaluator = new EvaluationEngine(template.evaluationConfig);
    score = evaluator.evaluate(omrResponse);
  }

  return { omrResponse, score };
}
```

---

## Related Documentation

- `concept.md` - Architecture and data models
- `flows.md` - Complete detection and interpretation flows
- `decisions.md` - Threshold strategy selection logic
- `constraints.md` - Performance and accuracy constraints
