# ReadOMR Processor - Execution Flow

## High-Level Flow

```
Pipeline
  ↓
ReadOMRProcessor.process(context)
  ↓
Check for shift_detection (early exit if present)
  ↓
Resize images to template dimensions
  ↓
Normalize images (contrast/brightness)
  ↓
TemplateFileRunner.read_omr_and_update_metrics()
  ├─→ Detection Pass (extract raw data)
  └─→ Interpretation Pass (convert to answers)
  ↓
Get concatenated response (apply custom labels)
  ↓
Extract interpretation metrics
  ↓
Optional: ML Fallback (if low confidence)
  ↓
Update context with results
  ↓
Return updated context
```

## Detailed Step-by-Step Flow

### Step 1: Processor Entry Point

**File**: `src/processors/detection/processor.py`

```python
def process(self, context: ProcessingContext) -> ProcessingContext:
    """Execute OMR detection and interpretation."""

    logger.debug("Starting ReadOMR processor")

    # 1.1: Check if ShiftDetectionProcessor already ran
    shift_detection_meta = context.metadata.get("shift_detection")

    if shift_detection_meta:
        # Shifts already applied and validated
        logger.debug("Using shift-validated results from ShiftDetectionProcessor")
        return context  # Early exit

    # 1.2: Extract data from context
    template = context.template
    file_path = context.file_path
    input_gray_image = context.gray_image
    colored_image = context.colored_image
```

**Key Points**:
- Early exit optimization: Skip if ShiftDetectionProcessor already ran
- Extracts all needed data from ProcessingContext
- Logs processor start for debugging

---

### Step 2: Image Resizing

```python
    # 2.1: Resize to template dimensions
    gray_image, colored_image = ImageUtils.resize_to_dimensions(
        template.template_dimensions,
        input_gray_image,
        colored_image
    )

    # 2.2: Save resized image (for debug output)
    template.save_image_ops.append_save_image(
        "Resized Image",
        range(3, 7),  # Debug levels 3-6
        gray_image,
        colored_image
    )
```

**Why Resize?**
- Template coordinates assume specific dimensions
- Scans may come from different DPI settings
- Creates consistent coordinate system for detection

**Example**:
```
Input image: 2480 × 3508 (300 DPI scan)
Template dimensions: 1240 × 1754 (150 DPI template)
Resized to: 1240 × 1754 (matches template)
```

---

### Step 3: Image Normalization

```python
    # 3.1: Normalize images
    gray_image, colored_image = ImageUtils.normalize(gray_image, colored_image)
```

**Normalization Process**:
```python
# Inside ImageUtils.normalize():
def normalize(gray_image, colored_image):
    # Adjust contrast and brightness
    # Apply histogram equalization (optional)
    # Standardize intensity range [0, 255]
    return normalized_gray, normalized_colored
```

**Why Normalize?**
- Reduces impact of lighting variations
- Improves threshold accuracy for bubble detection
- Consistent detection across different scanners

**Before/After Example**:
```
Before: mean=120, std=40 (low contrast)
After:  mean=128, std=60 (enhanced contrast)
```

---

### Step 4: Run Detection and Interpretation

**File**: `src/processors/detection/template_file_runner.py`

```python
    # 4.1: Run TemplateFileRunner
    raw_omr_response = self.template_file_runner.read_omr_and_update_metrics(
        file_path, gray_image, colored_image
    )
```

#### Step 4.1: Detection Pass

**Sub-Flow**:
```python
def read_omr_and_update_metrics(self, file_path, gray_image, colored_image):
    # 4.1.1: Run detection pass
    self.run_file_level_detection(file_path, gray_image, colored_image)

    # 4.1.2: Run interpretation pass
    return self.run_file_level_interpretation(file_path, gray_image, colored_image)
```

**Detection Pass Detail**:
```python
def run_file_level_detection(self, file_path, gray_image, colored_image):
    # A. Initialize file-level aggregates
    self.initialize_file_level_detection_aggregates(file_path)
    self.repository.initialize_file(file_path)

    # B. Process each field
    for field in self.all_fields:
        self.run_field_level_detection(field, gray_image, colored_image)

    # C. Update aggregates after all fields processed
    self.update_detection_aggregates_on_processed_file(file_path)
    self.repository.finalize_file()
```

**Per-Field Detection**:
```python
def run_field_level_detection(self, field, gray_image, colored_image):
    # 1. Get field-type-specific runner
    field_type = field.field_detection_type  # BUBBLES_THRESHOLD, OCR, BARCODE_QR
    field_runner = self.field_detection_type_file_runners[field_type]

    # 2. Run field-level detection (delegates to specialized runner)
    field_detection = field_runner.run_field_level_detection(
        field, gray_image, colored_image
    )

    # 3. Store detection in template-level detection pass
    self.detection_pass.run_field_level_detection(field, field_detection)
```

**Field-Type Dispatch Table**:
```python
field_detection_type_to_runner = {
    FieldDetectionType.BUBBLES_THRESHOLD: BubblesThresholdFileRunner,
    FieldDetectionType.OCR: OCRFileRunner,
    FieldDetectionType.BARCODE_QR: BarcodeFileRunner,
}
```

**Example Detection Flow for Bubble Field**:
```python
# BubblesThresholdFileRunner.run_field_level_detection()
1. Extract field bounding box from template
2. Crop ROI (region of interest) from gray_image
3. For each bubble in field:
   - Extract scan_box region
   - Calculate mean darkness (pixel intensity)
   - Store in field_detection metrics
4. Return field_detection object
```

#### Step 4.2: Interpretation Pass

```python
def run_file_level_interpretation(self, file_path, _gray_image, _colored_image):
    # A. Initialize interpretation aggregates
    self.initialize_file_level_interpretation_aggregates(file_path)

    # B. Build OMR response
    current_omr_response = {}

    # C. Process each field
    for field in self.all_fields:
        self.run_field_level_interpretation(field, current_omr_response)

    # D. Update aggregates
    self.update_interpretation_aggregates_on_processed_file(file_path)

    return current_omr_response
```

**Per-Field Interpretation**:
```python
def run_field_level_interpretation(self, field, current_omr_response):
    # 1. Get field-type runner
    field_runner = self.field_detection_type_file_runners[field.field_detection_type]

    # 2. Get file-level detection aggregates (from detection pass)
    file_level_aggregates = self.detection_pass.get_file_level_aggregates()

    # 3. Run interpretation (applies threshold logic)
    field_interpretation = field_runner.interpretation_pass.run_field_level_interpretation(
        field, file_level_aggregates
    )

    # 4. Store in template-level interpretation pass
    self.interpretation_pass.run_field_level_interpretation(
        field, field_interpretation, ..., current_omr_response
    )

    # 5. Add to response
    current_omr_response[field.field_label] = (
        field_interpretation.get_field_interpretation_string()
    )
```

**Example Interpretation for Bubble Field**:
```python
# Input (from detection pass):
field_detection = {
    "Q1": {
        "A": 0.85,  # Mean darkness
        "B": 0.12,
        "C": 0.09,
        "D": 0.11
    }
}

# Interpretation logic:
1. Calculate global threshold (e.g., 0.50)
2. Find darkest bubble (A = 0.85)
3. Check if above threshold → YES
4. Check for multi-marks (other bubbles > threshold) → NO
5. Return answer: "A"

# Output:
current_omr_response["Q1"] = "A"
```

---

### Step 5: Get Concatenated Response

```python
    # 5.1: Apply custom output labels
    concatenated_omr_response = template.get_concatenated_omr_response(
        raw_omr_response
    )
```

**Custom Label Mapping**:
```python
# Template config:
{
  "fieldBlocks": {
    "Q_BLOCK": {
      "fields": [
        {
          "field_label": "Q1",
          "outputFormat": "question_1_answer"  # Custom label
        }
      ]
    }
  }
}

# Transform:
raw_omr_response = {"Q1": "A", "Q2": "B"}
concatenated_omr_response = {"question_1_answer": "A", "Q2": "B"}
```

---

### Step 6: Extract Interpretation Metrics

```python
    # 6.1: Get directory-level aggregates
    directory_level_aggregates = (
        self.template_file_runner.get_directory_level_interpretation_aggregates()
    )

    # 6.2: Get file-level aggregates
    file_level_aggregates = (
        directory_level_aggregates["file_wise_aggregates"][file_path]
    )

    # 6.3: Extract specific metrics
    is_multi_marked = file_level_aggregates["read_response_flags"]["is_multi_marked"]
    field_id_to_interpretation = file_level_aggregates["field_id_to_interpretation"]
```

**Aggregates Structure**:
```python
directory_level_aggregates = {
    "file_wise_aggregates": {
        "path/to/file.jpg": {
            "read_response_flags": {
                "is_multi_marked": False,  # Any multi-marked fields?
                "has_errors": False
            },
            "field_id_to_interpretation": {
                "Q1": {
                    "field_label": "Q1",
                    "marked_answer": "A",
                    "field_level_confidence_metrics": {
                        "overall_confidence_score": 0.95
                    }
                }
            }
        }
    }
}
```

---

### Step 7: Update Context

```python
    # 7.1: Store core results
    context.omr_response = concatenated_omr_response
    context.is_multi_marked = is_multi_marked
    context.field_id_to_interpretation = field_id_to_interpretation

    # 7.2: Update images (normalized versions)
    context.gray_image = gray_image
    context.colored_image = colored_image

    # 7.3: Store metadata
    context.metadata["raw_omr_response"] = raw_omr_response
    context.metadata["directory_level_interpretation_aggregates"] = (
        directory_level_aggregates
    )
```

---

### Step 8: Optional ML Fallback

```python
    # 8.1: Check if ML fallback should be used
    if self.hybrid_strategy and self.hybrid_strategy.should_use_ml_fallback(context):
        logger.info(f"Using ML fallback for low-confidence fields")

        # 8.2: Enable ML detector
        self.ml_detector.enable_for_low_confidence()

        # 8.3: Run ML detection
        context = self.ml_detector.process(context)

        # 8.4: Disable ML detector
        self.ml_detector.disable()

        # 8.5: Update statistics
        self.hybrid_strategy.stats["ml_fallback_used"] += 1
```

**ML Fallback Decision Logic**:
```python
def should_use_ml_fallback(self, context):
    # Check each field's confidence
    for field_id, interpretation in context.field_id_to_interpretation.items():
        confidence_metrics = interpretation.get("field_level_confidence_metrics", {})
        confidence_score = confidence_metrics.get("overall_confidence_score", 1.0)

        if confidence_score < self.confidence_threshold:  # Default: 0.75
            logger.debug(f"Low confidence: {field_id} = {confidence_score:.3f}")
            return True  # Trigger ML fallback

    return False  # All fields high confidence
```

**ML Detection Process**:
```python
def process(self, context):
    if not self.enabled or self.model is None:
        return context  # Skip if disabled or no model

    # Run YOLO inference
    results = self.model.predict(
        context.gray_image,
        conf=self.confidence_threshold,
        verbose=False
    )

    # Extract detections
    ml_detections = self._extract_detections(results)

    # Store in metadata
    context.metadata["ml_detections"] = ml_detections
    context.metadata["ml_fallback_used"] = True

    return context
```

**ML Detection Result**:
```python
ml_detections = [
    {
        "bbox": {
            "x_center": 150.5,
            "y_center": 200.3,
            "width": 20.0,
            "height": 20.0
        },
        "class_id": 1,
        "class_name": "bubble_filled",
        "confidence": 0.92
    },
    # ... more detections
]
```

---

### Step 9: Finish Processing Directory

**Called After All Files Processed**:
```python
def finish_processing_directory(self):
    """Finalize directory-level processing."""

    # 9.1: Get aggregated results from TemplateFileRunner
    results = self.template_file_runner.finish_processing_directory()

    # 9.2: Log ML fallback statistics (if enabled)
    if self.hybrid_strategy:
        stats = self.hybrid_strategy.get_statistics()
        logger.info("=" * 60)
        logger.info("ML Fallback Statistics")
        logger.info("=" * 60)
        logger.info(f"Total fields processed: {stats['total_fields']}")
        logger.info(f"High confidence fields: {stats['high_confidence_fields']}")
        logger.info(f"Low confidence fields: {stats['low_confidence_fields']}")
        logger.info(f"ML fallback used: {stats['ml_fallback_used']} times")
        logger.info("=" * 60)

    return results
```

---

## Data Structures

### ProcessingContext (Input/Output)
```python
@dataclass
class ProcessingContext:
    # Input
    file_path: str
    gray_image: MatLike
    colored_image: MatLike
    template: Template

    # Output (populated by ReadOMR)
    omr_response: dict[str, str]              # {"Q1": "A", "Q2": "B"}
    is_multi_marked: bool                     # True if multi-marked
    field_id_to_interpretation: dict          # Detailed interpretation
    metadata: dict                            # raw_omr_response, ml_detections
```

### Raw OMR Response
```python
raw_omr_response = {
    "Q1": "A",
    "Q2": "B",
    "Q3": "",      # Empty if no answer detected
    "Name": "JOHN DOE",
    "ID": "12345",
    "QR_CODE": "https://example.com/student/12345"
}
```

### Field Interpretation
```python
field_interpretation = {
    "field_label": "Q1",
    "field_id": "Q1",
    "marked_answer": "A",
    "field_level_confidence_metrics": {
        "overall_confidence_score": 0.95,
        "threshold_used": 0.50,
        "darkest_bubble_value": 0.85,
        "second_darkest_value": 0.12
    },
    "raw_bubble_values": {
        "A": 0.85,
        "B": 0.12,
        "C": 0.09,
        "D": 0.11
    }
}
```

### ML Detection Result
```python
ml_detection = {
    "bbox": {
        "x_center": float,
        "y_center": float,
        "width": float,
        "height": float,
        "x1": float,  # Top-left corner
        "y1": float,
        "x2": float,  # Bottom-right corner
        "y2": float
    },
    "class_id": int,              # 0: empty, 1: filled, 2: partial
    "class_name": str,            # "bubble_empty", "bubble_filled", "bubble_partial"
    "confidence": float           # 0.0 - 1.0
}
```

---

## Alternative Flows

### Flow 1: With ShiftDetectionProcessor (Early Exit)

```
Pipeline
  ↓
ShiftDetectionProcessor.process(context)
  ├─→ Runs ML-based shift detection
  ├─→ Applies shifts to field positions
  ├─→ Runs detection + interpretation
  └─→ Stores results in context.metadata["shift_detection"]
  ↓
ReadOMRProcessor.process(context)
  ├─→ Checks context.metadata.get("shift_detection")
  └─→ Returns context immediately (EARLY EXIT)
```

**Code**:
```python
shift_detection_meta = context.metadata.get("shift_detection")
if shift_detection_meta:
    logger.debug("Using shift-validated results")
    return context  # Skip ReadOMR processing
```

### Flow 2: Without ML Fallback

```
ReadOMRProcessor.process(context)
  ↓
Resize + Normalize
  ↓
Detection + Interpretation
  ↓
Update Context
  ↓
Return (NO ML FALLBACK)
```

**When This Happens**:
- `ml_model_path` not provided in initialization
- All fields have high confidence (>= 0.75)
- ML model failed to load

### Flow 3: With ML Fallback

```
ReadOMRProcessor.process(context)
  ↓
Traditional Detection (fast)
  ↓
Check Confidence Scores
  ↓
IF any field < 0.75 confidence:
  ↓
  ML Detection (slow)
  ↓
  Merge Results
  ↓
Update Context
  ↓
Return
```

---

## Performance Characteristics

### Time Complexity

| Stage | Complexity | Typical Time |
|-------|-----------|--------------|
| Resize | O(w × h) | 10ms |
| Normalize | O(w × h) | 5ms |
| Detection Pass | O(n_fields × field_size) | 50-200ms |
| Interpretation Pass | O(n_fields × bubbles) | 10-50ms |
| ML Fallback | O(YOLO_inference) | 200-500ms |

### Memory Complexity

```python
# Image copies
gray_image: w × h × 1 byte        # ~4 MB for 2000×3000
colored_image: w × h × 3 bytes    # ~12 MB

# Detection aggregates
field_detections: O(n_fields × n_options)  # ~1 KB per field
interpretation: O(n_fields)                 # ~500 bytes per field

# ML detections (if used)
ml_detections: O(n_bubbles × detection_size)  # ~100 bytes per bubble
```

### Optimization Points

1. **Early Exit**: Skip processing if ShiftDetectionProcessor ran
2. **Lazy ML Loading**: Model loaded only if needed
3. **Repository Pattern**: Typed storage avoids repeated lookups
4. **Single Image Resize**: Reused for all field detections

---

## Error Handling

### No Template Fields
```python
if len(self.all_fields) == 0:
    logger.warning("No fields defined in template")
    return {}  # Empty response
```

### ML Model Load Failure
```python
try:
    self.model = YOLO(str(self.model_path))
except Exception as e:
    logger.warning(f"Failed to load ML model: {e}")
    self.model = None  # Disable ML fallback
```

### OCR/Barcode Library Missing
```python
try:
    from easyocr import Reader
except ImportError:
    logger.error("EasyOCR not installed")
    # OCR fields will return empty strings
```

### Invalid Image Dimensions
```python
# ImageUtils.resize_to_dimensions handles gracefully
if image is None or image.size == 0:
    logger.error("Invalid image")
    return empty_context
```

---

## Browser Migration Notes

### Async Processing Required

```javascript
async function process(context) {
  // All steps must be async for UI responsiveness

  // Step 1: Resize (can use OffscreenCanvas)
  const resized = await resizeImages(context);

  // Step 2: Normalize
  const normalized = await normalizeImages(resized);

  // Step 3: Detection (may take 1-2 seconds)
  const response = await runDetection(normalized);

  // Step 4: ML Fallback (MUST use Web Worker)
  if (needsML(response)) {
    const mlResults = await runMLInWorker(normalized);
  }

  return context;
}
```

### Web Worker for ML Inference

```javascript
// main.js
const worker = new Worker('ml-worker.js');
worker.postMessage({ image: imageData, modelPath: 'model.json' });
worker.onmessage = (e) => {
  const mlDetections = e.data.detections;
  mergeMLResults(mlDetections);
};

// ml-worker.js
importScripts('tfjs.min.js');
self.onmessage = async (e) => {
  const { image, modelPath } = e.data;
  const model = await tf.loadGraphModel(modelPath);
  const predictions = await model.predict(imageTensor);
  self.postMessage({ detections: predictions });
};
```

### Progress Reporting

```javascript
async function processOMR(context, onProgress) {
  onProgress({ stage: 'resize', percent: 10 });
  await resizeImages(context);

  onProgress({ stage: 'normalize', percent: 20 });
  await normalizeImages(context);

  onProgress({ stage: 'detection', percent: 40 });
  await runDetection(context);

  onProgress({ stage: 'ml_fallback', percent: 80 });
  if (needsML) await runML(context);

  onProgress({ stage: 'complete', percent: 100 });
}
```

---

## Testing Considerations

### Unit Tests

```python
def test_readomr_basic_processing():
    context = create_test_context()
    processor = ReadOMRProcessor(template)

    result = processor.process(context)

    assert result.omr_response == {"Q1": "A", "Q2": "B"}
    assert result.is_multi_marked == False

def test_readomr_shift_detection_early_exit():
    context = create_test_context()
    context.metadata["shift_detection"] = True

    processor = ReadOMRProcessor(template)
    result = processor.process(context)

    # Should return immediately without processing
    assert result == context

def test_readomr_ml_fallback():
    context = create_test_context_with_low_confidence()
    processor = ReadOMRProcessor(template, ml_model_path="test_model.pt")

    result = processor.process(context)

    assert "ml_detections" in result.metadata
    assert result.metadata["ml_fallback_used"] == True
```

### Integration Tests

```python
def test_full_pipeline_with_readomr():
    pipeline = ProcessingPipeline(template)
    context = ProcessingContext(file_path, gray, colored, template)

    result = pipeline.process_file(file_path, gray, colored)

    assert result.omr_response is not None
    assert "Q1" in result.omr_response
```

---

## Related Flows

- **Bubble Detection** (`bubbles-threshold/flows.md`) - Threshold-based detection
- **OCR Detection** (`ocr/flows.md`) - Text recognition flow
- **Barcode Detection** (`barcode/flows.md`) - QR/barcode decoding
- **ML Bubble Detection** (`ml-bubble/flows.md`) - YOLO-based fallback
- **Shift Detection** (`shift-detection/flows.md`) - ML shift correction
