# ReadOMR Processor - Pipeline Integration

## Overview

The **ReadOMR Processor** integrates into the OMRChecker processing pipeline as the **detection and interpretation stage**. It receives preprocessed and aligned images from upstream processors, runs multi-format field detection, and produces structured OMR response data for downstream consumers (evaluation, CSV export).

## Pipeline Architecture

### Complete Pipeline Flow

```
Entry Point (main.py / entry.py)
  ↓
Template Loading
  ├─→ Load template.json
  ├─→ Parse field definitions
  ├─→ Initialize TemplateFileRunner
  └─→ Create ProcessingPipeline
  ↓
ProcessingPipeline
  ├─→ Stage 1: PreprocessingCoordinator
  │   ├─→ AutoRotate
  │   ├─→ CropOnMarkers
  │   ├─→ CropPage
  │   ├─→ Filters (Blur, Contrast, Levels)
  │   └─→ Warping
  ↓
  ├─→ Stage 2: AlignmentProcessor
  │   ├─→ SIFT feature matching
  │   ├─→ K-nearest interpolation
  │   └─→ Apply shifts to scan_box.shifts
  ↓
  ├─→ Stage 3 (Optional): MLFieldBlockDetector
  │   └─→ Detect field blocks using YOLO
  ↓
  ├─→ Stage 4 (Optional): ShiftDetectionProcessor
  │   ├─→ ML-based shift prediction
  │   ├─→ Apply shifts
  │   └─→ Run detection (if enabled)
  ↓
  ├─→ Stage 5: ReadOMRProcessor ◄── THIS MODULE
  │   ├─→ Check for shift_detection (early exit)
  │   ├─→ Resize + normalize images
  │   ├─→ Run detection pass (bubbles, OCR, barcode)
  │   ├─→ Run interpretation pass (threshold logic)
  │   ├─→ Optional: ML fallback (YOLO bubbles)
  │   └─→ Store omr_response in context
  ↓
  ├─→ Stage 6 (Optional): TrainingDataCollector
  │   └─→ Export training data for ML models
  ↓
  └─→ Stage 7 (Optional): EvaluationProcessor
      ├─→ Load answer keys
      ├─→ Compare omr_response to answers
      └─→ Calculate score
  ↓
CSV Export / Visualization
  ├─→ Write results to CSV
  └─→ Generate HTML reports
```

---

## Integration Points

### 1. Input: ProcessingContext from Pipeline

**Source**: `src/processors/pipeline.py`

```python
class ProcessingPipeline:
    def process_file(self, file_path, gray_image, colored_image):
        # Create initial context
        context = ProcessingContext(
            file_path=file_path,
            gray_image=gray_image,
            colored_image=colored_image,
            template=self.template,
        )

        # Execute processors in sequence
        for processor in self.processors:
            context = processor.process(context)  # ← ReadOMR called here

        return context
```

**What ReadOMR Receives**:
```python
context.gray_image           # Preprocessed + aligned grayscale
context.colored_image        # Preprocessed + aligned color
context.template             # Template with shifts applied
context.file_path            # Current file being processed
context.metadata             # May contain shift_detection info
```

---

### 2. Output: Updated ProcessingContext

**What ReadOMR Produces**:
```python
# Core results
context.omr_response: dict[str, str]
# Example: {"Q1": "A", "Q2": "B", "Name": "JOHN DOE", "ID": "12345"}

context.is_multi_marked: bool
# True if any field had multiple marks above threshold

context.field_id_to_interpretation: dict[str, Any]
# Detailed per-field interpretation (confidence, raw values)

# Updated images
context.gray_image              # Normalized version (replaces original)
context.colored_image           # Normalized version

# Metadata
context.metadata["raw_omr_response"]  # Before custom labels
context.metadata["directory_level_interpretation_aggregates"]  # All files
context.metadata["ml_detections"]  # If ML fallback used
context.metadata["ml_fallback_used"]  # Boolean flag
```

---

### 3. Integration with AlignmentProcessor

**Dependency**: ReadOMR depends on AlignmentProcessor running first

```python
# AlignmentProcessor updates:
for field_block in template.field_blocks:
    for field in field_block.fields:
        for scan_box in field.scan_boxes:
            scan_box.shifts = [dx, dy]  # Applied by alignment

# ReadOMR uses shifts during detection:
def detect_bubble(scan_box, gray_image):
    position = scan_box.position + scan_box.shifts  # Use shifted position
    roi = extract_roi(gray_image, position)
    return calculate_darkness(roi)
```

**Code Reference**:
- Alignment: `src/processors/alignment/processor.py`
- Shift application: `src/processors/alignment/k_nearest_interpolation.py`

---

### 4. Integration with ShiftDetectionProcessor

**Alternative Flow**: ShiftDetectionProcessor can replace ReadOMR

```python
# Pipeline without ShiftDetectionProcessor:
pipeline.processors = [
    PreprocessingCoordinator(template),
    AlignmentProcessor(template),
    ReadOMRProcessor(template),  # Runs detection
]

# Pipeline with ShiftDetectionProcessor:
pipeline.processors = [
    PreprocessingCoordinator(template),
    AlignmentProcessor(template),
    ShiftDetectionProcessor(template, shift_config),  # Runs detection + shifts
    ReadOMRProcessor(template),  # Early exit (skips detection)
]
```

**Early Exit Logic**:
```python
# In ReadOMRProcessor.process():
shift_detection_meta = context.metadata.get("shift_detection")
if shift_detection_meta:
    # ShiftDetectionProcessor already populated:
    # - context.omr_response
    # - context.field_id_to_interpretation
    # - context.is_multi_marked
    return context  # Skip ReadOMR detection
```

**Code Reference**: `src/processors/detection/processor.py:110-117`

---

### 5. Integration with EvaluationProcessor

**Downstream Consumer**: EvaluationProcessor uses `context.omr_response`

```python
# ReadOMR output:
context.omr_response = {"Q1": "A", "Q2": "B", "Q3": "C"}

# EvaluationProcessor input:
class EvaluationProcessor(Processor):
    def process(self, context):
        omr_response = context.omr_response  # Uses ReadOMR output

        # Load answer keys
        answer_keys = {"Q1": "A", "Q2": "C", "Q3": "C"}

        # Compare
        score = calculate_score(omr_response, answer_keys)

        # Update context
        context.score = score
        context.evaluation_meta = {...}

        return context
```

**Code Reference**: `src/processors/evaluation/processor.py`

---

### 6. Integration with CSV Export

**Final Output**: CSV writer uses `context.omr_response`

```python
# After pipeline completes:
def export_to_csv(context):
    row = {
        "file_name": context.file_path,
        **context.omr_response,  # Spread OMR response
        "score": context.score,
        "is_multi_marked": context.is_multi_marked
    }

    csv_writer.writerow(row)
```

**Example CSV**:
```csv
file_name,Q1,Q2,Q3,Name,ID,score,is_multi_marked
sample_001.jpg,A,B,C,JOHN DOE,12345,85.0,False
sample_002.jpg,A,A,C,JANE SMITH,67890,100.0,False
```

**Code Reference**: `src/utils/csv.py`

---

## Template Integration

### Template Loading

**Entry Point**: Template loaded before pipeline creation

```python
# src/entry.py or main.py
from src.processors.template import TemplateProcessor

# Load template
template_processor = TemplateProcessor(template_path)
template = template_processor.load_template()

# Create pipeline with template
pipeline = ProcessingPipeline(template, args)

# Pipeline creates ReadOMRProcessor:
# ReadOMRProcessor.__init__(template) ← Template passed here
```

---

### Template Configuration

**ReadOMR Uses Template For**:

1. **Field Definitions**
```python
template.all_fields  # List of all fields (bubbles, OCR, barcode)
```

2. **Dimensions**
```python
template.template_dimensions  # [width, height] for resizing
```

3. **Tuning Configuration**
```python
template.tuning_config.ml.confidence_threshold  # ML fallback threshold
template.tuning_config.outputs.show_image_level  # Debug visualization
```

4. **Save Image Operations**
```python
template.save_image_ops.append_save_image(
    "Resized Image", range(3, 7), gray_image, colored_image
)
```

5. **Custom Output Labels**
```python
template.get_concatenated_omr_response(raw_omr_response)
# Applies outputFormat from template
```

---

### TemplateFileRunner Integration

**Created by ReadOMR**:
```python
class ReadOMRProcessor(Processor):
    def __init__(self, template, ml_model_path=None):
        self.template = template
        self.template_file_runner = TemplateFileRunner(template)  # Delegate
```

**TemplateFileRunner Responsibilities**:
- Maintain field-type-specific runners (bubbles, OCR, barcode)
- Orchestrate detection and interpretation passes
- Manage directory-level and file-level aggregates
- Use DetectionRepository for typed storage

**Code Reference**: `src/processors/detection/template_file_runner.py`

---

## ML Integration

### ML Model Initialization

**Optional ML Fallback**:
```python
# Pipeline initialization:
ml_model_path = args.get("ml_model_path")  # From CLI args

pipeline = ProcessingPipeline(template, args)
# Pipeline creates ReadOMRProcessor with ML:
ReadOMRProcessor(template, ml_model_path=ml_model_path)
```

**Lazy Loading**:
```python
def __init__(self, template, ml_model_path=None):
    self.ml_detector = None
    self.hybrid_strategy = None

    if ml_model_path:
        self._initialize_ml_fallback(ml_model_path)  # Load model
```

---

### ML Fallback Integration

**Hybrid Detection Strategy**:
```python
# After traditional detection:
if self.hybrid_strategy and self.hybrid_strategy.should_use_ml_fallback(context):
    # Enable ML detector
    self.ml_detector.enable_for_low_confidence()

    # Run ML detection
    context = self.ml_detector.process(context)

    # Disable ML detector
    self.ml_detector.disable()

    # Update stats
    self.hybrid_strategy.stats["ml_fallback_used"] += 1
```

**ML Results Storage**:
```python
context.metadata["ml_detections"] = [
    {"bbox": {...}, "class_name": "bubble_filled", "confidence": 0.92},
    # ... more detections
]
context.metadata["ml_fallback_used"] = True
```

---

## Error Handling Integration

### Pipeline-Level Error Handling

**Pipeline Catches Processor Errors**:
```python
# src/processors/pipeline.py
for processor in self.processors:
    processor_name = processor.get_name()
    try:
        context = processor.process(context)
    except Exception as e:
        logger.error(f"Processor {processor_name} failed: {e}")
        # Pipeline decides how to handle:
        # - Continue to next processor
        # - Skip file
        # - Abort batch
```

---

### ReadOMR Error Handling

**Graceful Degradation**:
```python
# ML model load failure:
try:
    self.ml_detector = MLBubbleDetector(ml_model_path)
except Exception as e:
    logger.warning(f"Failed to initialize ML fallback: {e}")
    self.ml_detector = None  # Disable ML, continue with traditional

# OCR library missing:
try:
    from easyocr import Reader
except ImportError:
    logger.warning("EasyOCR not installed, OCR fields will return empty")
    # Continue processing, OCR fields return ""
```

**No Hard Failures**:
- Missing dependencies → feature disabled, log warning
- Low-quality image → low confidence result, no error
- Template misconfiguration → incorrect results, no error

**Design Philosophy**: Better to return low-quality results than crash

---

## Batch Processing Integration

### Directory-Level Processing

**Pipeline Processes Multiple Files**:
```python
# src/entry.py
def process_directory(input_dir, template, args):
    pipeline = ProcessingPipeline(template, args)

    # Discover all images
    image_files = discover_images(input_dir)

    # Process each file
    results = []
    for file_path in image_files:
        gray, colored = load_images(file_path)
        context = pipeline.process_file(file_path, gray, colored)
        results.append(context)

    # Finish processing (directory-level summary)
    for processor in pipeline.processors:
        if hasattr(processor, "finish_processing_directory"):
            processor.finish_processing_directory()

    return results
```

---

### ReadOMR Finish Hook

**Directory-Level Summary**:
```python
def finish_processing_directory(self):
    # Get results from TemplateFileRunner
    results = self.template_file_runner.finish_processing_directory()

    # Log ML statistics
    if self.hybrid_strategy:
        stats = self.hybrid_strategy.get_statistics()
        logger.info("ML Fallback Statistics:")
        logger.info(f"  Total fields: {stats['total_fields']}")
        logger.info(f"  High confidence: {stats['high_confidence_fields']}")
        logger.info(f"  Low confidence: {stats['low_confidence_fields']}")
        logger.info(f"  ML fallback used: {stats['ml_fallback_used']} times")

    return results
```

**Example Output**:
```
============================================================
ML Fallback Statistics
============================================================
Total fields processed: 500
High confidence fields: 475
Low confidence fields: 25
ML fallback used: 25 times
============================================================
```

---

## Parallel Processing Integration

### ThreadPoolExecutor Integration

**Parallel File Processing**:
```python
from concurrent.futures import ThreadPoolExecutor

def process_directory_parallel(input_dir, template, args):
    pipeline = ProcessingPipeline(template, args)
    image_files = discover_images(input_dir)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file_path in image_files:
            future = executor.submit(process_single_file, pipeline, file_path)
            futures.append(future)

        results = [f.result() for f in futures]

    return results
```

**Thread Safety**:
- **TemplateFileRunner**: Thread-safe (one per template, reused)
- **DetectionRepository**: Thread-safe (per-file storage)
- **ML Model**: Not thread-safe (use locks or per-thread model)

---

## Configuration Integration

### CLI Arguments

**ReadOMR Configuration from CLI**:
```python
# main.py or entry.py
args = {
    "ml_model_path": "models/bubble_detector.pt",  # ML fallback model
    "collect_training_data": False,                # Training data collection
    "enable_shift_detection": False,               # Use ShiftDetectionProcessor
}

pipeline = ProcessingPipeline(template, args)
# Pipeline passes ml_model_path to ReadOMRProcessor
```

---

### Template Configuration

**Tuning Config Integration**:
```python
# template.json
{
  "tuningConfig": {
    "ml": {
      "confidence_threshold": 0.75,
      "enable_ml_fallback": true
    },
    "outputs": {
      "show_image_level": 0  # Debug visualization level
    }
  }
}

# ReadOMR uses:
confidence_threshold = template.tuning_config.ml.confidence_threshold
```

---

## Visualization Integration

### Debug Image Output

**Save Debug Images**:
```python
# ReadOMR saves resized image:
template.save_image_ops.append_save_image(
    "Resized Image",
    range(3, 7),  # Show at debug levels 3-6
    gray_image,
    colored_image
)

# Field-type runners save detection visualizations:
# - Bubble detection with circles drawn
# - OCR with bounding boxes
# - Barcode with decoded text overlay
```

**Output Structure**:
```
outputs/
  sample_001/
    3_Resized Image.jpg
    4_Bubble Detection.jpg
    5_OCR Detection.jpg
    6_Barcode Detection.jpg
```

---

## Testing Integration

### Unit Testing

**Isolated ReadOMR Testing**:
```python
def test_readomr_basic():
    template = create_test_template()
    processor = ReadOMRProcessor(template)

    context = ProcessingContext(
        file_path="test.jpg",
        gray_image=test_gray,
        colored_image=test_colored,
        template=template
    )

    result = processor.process(context)

    assert result.omr_response == {"Q1": "A", "Q2": "B"}
    assert result.is_multi_marked == False
```

---

### Integration Testing

**Full Pipeline Testing**:
```python
def test_pipeline_with_readomr():
    template = load_template("test_template.json")
    pipeline = ProcessingPipeline(template)

    gray, colored = load_test_images()
    context = pipeline.process_file("test.jpg", gray, colored)

    # Verify ReadOMR output
    assert "Q1" in context.omr_response
    assert context.field_id_to_interpretation is not None
```

---

## Browser Integration Patterns

### Async Processing

**Browser Pipeline**:
```javascript
class ProcessingPipeline {
  async processFile(filePath, grayImage, coloredImage) {
    let context = new ProcessingContext(filePath, grayImage, coloredImage, this.template);

    // Execute processors sequentially (async)
    for (const processor of this.processors) {
      context = await processor.process(context);  // Await each processor
    }

    return context;
  }
}
```

---

### Web Worker Integration

**Offload ReadOMR to Worker**:
```javascript
// main.js
const readOMRWorker = new Worker('readomr-worker.js');

async function processImage(filePath, gray, colored) {
  // Send to worker
  readOMRWorker.postMessage({
    filePath, gray, colored, template
  });

  // Receive result
  return new Promise((resolve) => {
    readOMRWorker.onmessage = (e) => {
      const context = e.data.context;
      resolve(context);
    };
  });
}
```

---

### Progress Reporting

**Real-Time Progress Updates**:
```javascript
class ReadOMRProcessor {
  async process(context, onProgress) {
    onProgress({ stage: 'resize', percent: 10 });
    await this.resizeImages(context);

    onProgress({ stage: 'detection', percent: 40 });
    await this.runDetection(context);

    onProgress({ stage: 'interpretation', percent: 70 });
    await this.runInterpretation(context);

    onProgress({ stage: 'ml_fallback', percent: 90 });
    if (this.needsML(context)) {
      await this.runMLFallback(context);
    }

    onProgress({ stage: 'complete', percent: 100 });
    return context;
  }
}
```

---

## Summary of Integration Points

| Integration Point | Type | Direction | Data |
|-------------------|------|-----------|------|
| **ProcessingPipeline** | Input | Pipeline → ReadOMR | ProcessingContext (gray/colored/template) |
| **AlignmentProcessor** | Dependency | Alignment → ReadOMR | scan_box.shifts applied |
| **ShiftDetectionProcessor** | Alternative | ShiftDetection → ReadOMR | Early exit if present |
| **EvaluationProcessor** | Output | ReadOMR → Evaluation | context.omr_response |
| **CSV Export** | Output | ReadOMR → CSV | context.omr_response |
| **Template** | Configuration | Template → ReadOMR | Field defs, dimensions, config |
| **ML Models** | Optional | ReadOMR ↔ ML | Fallback detection |
| **Debug Visualization** | Output | ReadOMR → Files | Debug images |

ReadOMR integrates seamlessly into the pipeline by following the **Processor** interface, accepting **ProcessingContext**, and returning updated **ProcessingContext** for downstream consumers.
