# Main Pipeline Flow - Concept

**Module**: Domain / Pipeline
**Python Reference**: `src/processors/pipeline.py`, `src/entry.py`
**Last Updated**: 2026-02-20

---

## Overview

The ProcessingPipeline orchestrates the entire OMR processing workflow from input image to CSV output. It coordinates preprocessing, alignment, detection, and optional evaluation through a unified Processor interface.

**Key Responsibilities**:
1. **Processor Orchestration**: Execute processors in sequence
2. **Context Management**: Pass ProcessingContext through pipeline
3. **Error Handling**: Gracefully handle processor failures
4. **Extensibility**: Easy to add new processors
5. **Type Safety**: Unified interface for all processors

---

## Pipeline Architecture

### ProcessingPipeline Class

**Code Reference**: `src/processors/pipeline.py:11-158`

**Purpose**: Simplified pipeline using unified Processor interface

**Benefits**:
- All processors use same interface
- Easy to test each processor independently
- Simple to extend with new processors
- Type-safe ProcessingContext
- Consistent error handling

**Initialization**:
```python
pipeline = ProcessingPipeline(template, args)
```

**Processors** (in order):
1. **PreprocessingCoordinator**: Image transformation (rotation, cropping, filtering)
2. **AlignmentProcessor**: Feature-based alignment with reference
3. **(Optional) MLFieldBlockDetector**: ML-based field block detection
4. **(Optional) ShiftDetectionProcessor**: ML-based shift correction
5. **ReadOMRProcessor**: Bubble/barcode/OCR detection and interpretation
6. **(Optional) TrainingDataCollector**: Collect labeled data for ML training

---

## Pipeline Execution Flow

### process_file() Method

**Code Reference**: `src/processors/pipeline.py:117-157`

```python
def process_file(
    self,
    file_path: Path | str,
    gray_image: MatLike,
    colored_image: MatLike,
) -> ProcessingContext:
    """Process a single OMR file through all processors."""

    # 1. Create initial context
    context = ProcessingContext(
        file_path=file_path,
        gray_image=gray_image,
        colored_image=colored_image,
        template=self.template,
    )

    # 2. Execute each processor in sequence
    for processor in self.processors:
        processor_name = processor.get_name()
        logger.debug(f"Executing processor: {processor_name}")

        try:
            context = processor.process(context)
        except Exception as e:
            logger.error(f"Error in {processor_name}: {e}")
            raise

    # 3. Return final context with all results
    return context
```

---

## Pipeline Stages (Detailed)

### Stage 1: Preprocessing

**Processor**: PreprocessingCoordinator
**Code Reference**: `src/processors/image/coordinator.py`

**Inputs**:
- Raw grayscale and colored images
- Template with preprocessors

**Operations**:
- Resize to processing dimensions
- Execute preprocessors in sequence:
  - AutoRotate: Detect and fix rotation
  - CropOnMarkers: Crop based on marker dots/lines
  - CropPage: Detect page contours and crop
  - GaussianBlur: Reduce noise
  - Contrast: Adjust contrast
  - Levels: Adjust brightness
  - FeatureBasedAlignment: SIFT alignment (optional)

**Outputs**:
- Preprocessed images
- Updated template with field block shifts

**Example**:
```python
# Before preprocessing
gray_image: 2000x1500 (full scan)

# After preprocessing
gray_image: 900x650 (cropped, rotated, filtered)
template.field_blocks[0].shifts: [10, -5] (shifted by cropping)
```

---

### Stage 2: Alignment

**Processor**: AlignmentProcessor
**Code Reference**: `src/processors/alignment/processor.py`

**Inputs**:
- Preprocessed images
- Alignment configuration (reference image, margins)

**Operations**:
- Load reference image (if configured)
- Detect features (SIFT keypoints)
- Match features between reference and input
- Estimate homography transformation
- Warp image to align with reference

**Outputs**:
- Aligned images
- Alignment success flag in metadata

**Example**:
```python
# Before alignment
image: misaligned by 5px horizontally, 3px vertically

# After alignment
image: aligned to reference
context.metadata["alignment_successful"]: True
```

**Skip Condition**: If no reference image configured, alignment is skipped

---

### Stage 3: ML Field Block Detection (Optional)

**Processor**: MLFieldBlockDetector
**Code Reference**: `src/processors/detection/ml_field_block_detector.py`

**Enabled When**: `use_field_block_detection=True` and `field_block_model_path` provided

**Inputs**:
- Aligned image
- Field block detection model

**Operations**:
- Run YOLO model to detect field blocks
- Filter detections by confidence threshold
- Update template field block positions based on detections

**Outputs**:
- Updated field block positions
- Detection confidence scores

**Use Case**: When OMR sheets have significant position variations

---

### Stage 4: Shift Detection (Optional)

**Processor**: ShiftDetectionProcessor
**Code Reference**: `src/processors/detection/shift_detection_processor.py`

**Enabled When**: `shift_config.enabled=True` or `enable_shift_detection=True`

**Inputs**:
- Field block detections
- Shift detection configuration

**Operations**:
- Predict field-level shifts using ML model
- Apply shifts to field positions

**Outputs**:
- Updated field positions with fine-grained shifts

**Use Case**: Correct small misalignments within field blocks

---

### Stage 5: Detection & Interpretation

**Processor**: ReadOMRProcessor
**Code Reference**: `src/processors/detection/processor.py`

**Inputs**:
- Aligned and shifted images
- Template with field definitions

**Operations**:
- For each field in template:
  - Extract field region (ROI)
  - Detect based on field type:
    - **Bubbles**: Threshold-based or ML-based bubble detection
    - **Barcode**: PyZbar barcode decoding
    - **OCR**: EasyOCR or Tesseract text recognition
  - Interpret detections (bubble darkness, barcode value, OCR text)
- Aggregate results across all fields
- Calculate multi-mark flags

**Outputs**:
- `omr_response`: Detected field values
- `is_multi_marked`: Multi-mark flag
- `field_id_to_interpretation`: Bubble-level interpretations

**Example**:
```python
context.omr_response: {
    "Roll": "12345",
    "q1": "A",
    "q2": "B",
    "q3": "",  # Empty (not marked)
    "q4": "C"
}
context.is_multi_marked: False
context.field_id_to_interpretation: {
    "q1": [
        {"bubble": "A", "darkness": 0.85, "marked": True},
        {"bubble": "B", "darkness": 0.12, "marked": False},
        ...
    ],
    ...
}
```

---

### Stage 6: Training Data Collection (Optional)

**Processor**: TrainingDataCollector
**Code Reference**: `src/processors/training/data_collector.py`

**Enabled When**: `collect_training_data=True`

**Inputs**:
- Detection results with confidence scores
- Confidence threshold

**Operations**:
- Filter high-confidence detections
- Export bubble images with labels for training

**Outputs**:
- Training data saved to disk
- YOLO format annotations

**Use Case**: Collect labeled data to improve ML models

**Browser Migration**: SKIP (training is server-side only)

---

## End-to-End Flow (From File to CSV)

**Code Reference**: `src/entry.py:process_directory_wise()`

### 1. Directory Discovery

```python
# Find all template.json files
template_paths = find_all_templates(input_dir)

# For each template:
for template_path in template_paths:
    # Load template
    template = Template(template_path, tuning_config, args)

    # Find OMR images
    omr_files = find_omr_images(template_path.parent)

    # Setup output directories
    output_dir = get_output_dir(template_path)
    template.reset_and_setup_for_directory(output_dir)
```

---

### 2. Parallel Processing

```python
# Process files in parallel
with ThreadPoolExecutor(max_workers=tuning_config.processing.max_workers) as executor:
    futures = [
        executor.submit(process_single_file, file_info, template, tuning_config)
        for file_info in omr_files
    ]

    for future in as_completed(futures):
        result = future.result()
        results.append(result)
```

---

### 3. Single File Processing

**Code Reference**: `src/entry.py:process_single_file()`

```python
def process_single_file(file_info, template, tuning_config):
    file_path = file_info["path"]

    # 1. Read image
    gray_image, colored_image = ImageUtils.read_image_util(file_path, tuning_config)

    # 2. Process through pipeline
    context = template.process_file(file_path, gray_image, colored_image)

    # 3. Extract results
    omr_response = context.omr_response
    is_multi_marked = context.is_multi_marked
    score = context.score

    # 4. Format output
    output_row = [
        file_path.name,
        *[omr_response.get(field, "") for field in template.output_columns],
        score
    ]

    # 5. Save to CSV
    template.append_result(output_row)

    # 6. Handle multi-marked files
    if is_multi_marked:
        template.save_multi_marked_file(file_path, omr_response)

    return {"status": "success", "file_path": file_path}
```

---

### 4. CSV Output

**Code Reference**: `src/utils/csv.py:thread_safe_csv_append()`

**Output Format**:
```csv
filename,Roll,q1,q2,q3,q4,q5,score
sheet1.jpg,12345,A,B,C,D,A,4.5
sheet2.jpg,12346,B,A,D,C,B,3.0
sheet3.jpg,12347,C,C,A,A,D,5.0
```

**Thread Safety**: Uses file lock for concurrent writes

---

## Error Handling & Recovery

### Error Categories

**1. Marker Detection Errors**:
- Markers not found
- Insufficient markers
- Poor image quality

**Handling**: Move file to `errors/` directory, log in errors.csv

---

**2. Processing Errors**:
- Alignment failure
- Detection failure
- OCR error

**Handling**: Log error, use fallback strategy if available

---

**3. Multi-Marked Files**:
- Multiple bubbles marked in same field

**Handling**: Move file to `multi_marked/` directory, log in multi_marked.csv

---

### Graceful Degradation

```python
try:
    # Attempt ML bubble detection
    detections = ml_bubble_detector.detect(image)
except Exception as e:
    logger.warning(f"ML detection failed: {e}")
    if tuning_config.ml.bubble_detection.fallback_to_threshold:
        # Fallback to traditional threshold-based detection
        detections = threshold_detector.detect(image)
```

---

## Browser Migration Notes

### TypeScript Pipeline Class

```typescript
class ProcessingPipeline {
    private processors: Processor[] = [];

    constructor(template: Template, args?: Record<string, any>) {
        this.processors = [
            new PreprocessingCoordinator(template),
            new AlignmentProcessor(template),
            new ReadOMRProcessor(template)
        ];

        // Add optional processors based on config
        if (args?.mlModelPath) {
            this.processors.push(new MLBubbleDetector(args.mlModelPath));
        }
    }

    async processFile(
        filePath: string,
        grayImage: cv.Mat,
        coloredImage: cv.Mat
    ): Promise<ProcessingContext> {
        // Create initial context
        let context: ProcessingContext = {
            filePath,
            grayImage,
            coloredImage,
            template: this.template,
            omrResponse: {},
            isMultiMarked: false,
            fieldIdToInterpretation: {},
            score: 0,
            metadata: {}
        };

        // Execute processors sequentially
        for (const processor of this.processors) {
            const name = processor.getName();
            console.log(`Executing processor: ${name}`);

            try {
                context = await processor.process(context);
            } catch (error) {
                console.error(`Error in ${name}:`, error);
                throw error;
            }
        }

        return context;
    }
}
```

### Browser-Specific Adaptations

**1. Web Workers for Parallel Processing**:
```typescript
// Instead of ThreadPoolExecutor, use Web Workers
const workers = Array.from({ length: 4 }, () => new Worker('omr-worker.js'));

const results = await Promise.all(
    files.map((file, i) => {
        const worker = workers[i % workers.length];
        return processInWorker(worker, file);
    })
);
```

**2. File API for Image Loading**:
```typescript
async function loadImage(file: File): Promise<{ gray: cv.Mat; colored: cv.Mat }> {
    const img = await createImageBitmap(file);
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0);

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const mat = cv.matFromImageData(imageData);
    const gray = new cv.Mat();
    cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);

    return { gray, colored: mat };
}
```

**3. Download Results Instead of CSV Write**:
```typescript
function downloadResults(results: any[]): void {
    const csv = resultsToCSV(results);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'results.csv';
    a.click();
}
```

---

## Summary

**Pipeline**: Orchestrates OMR processing from image to CSV
**Stages**: Preprocessing → Alignment → Detection → Evaluation
**Architecture**: Unified Processor interface, type-safe context flow
**Error Handling**: Graceful degradation, error/multi-mark file separation
**Extensibility**: Easy to add new processors

**Browser Migration**:
- Use async/await for processor execution
- Replace ThreadPoolExecutor with Web Workers
- Use File API for image loading
- Download results instead of file system writes
- Maintain same processor sequence and context flow

**Key Takeaway**: The pipeline provides a clean, extensible architecture for OMR processing. Browser version should maintain same processor sequence and interface, but adapt I/O operations for browser environment.
