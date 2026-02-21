# Initial Codebase Scan Report

**Generated**: 2026-02-20
**Python Version**: 3.11+
**Total LOC**: ~39,000
**Application Files**: 172 Python files
**Test Files**: 68 Python files

---

## External Dependencies

### Core Dependencies
| Package | Version | Purpose | Browser Equivalent |
|---------|---------|---------|-------------------|
| **opencv-python** | 4.12.0.88 | Computer vision operations | OpenCV.js (WebAssembly) |
| **opencv-contrib-python** | 4.12.0.88 | Additional CV algorithms (SIFT, etc.) | OpenCV.js (limited support) |
| **numpy** | 2.2.6 | Array operations, numerical computing | TypedArrays, custom implementations |
| **pyzbar** | 0.1.9 | Barcode detection/decoding | @zxing/library, jsQR |
| **pandas** | 2.3.1 | CSV operations, data manipulation | PapaParse, custom CSV writer |
| **rich** | 13.8.0 | Terminal UI, colored output | N/A (browser UI framework) |
| **matplotlib** | 3.10.5 | Plotting, color utilities | Chart.js, D3.js (if needed) |
| **scipy** | 1.15.1 | Scientific computing, spatial algorithms | Custom implementations |
| **shapely** | 2.1.1 | Geometric operations | Turf.js or custom geometry utils |
| **jsonschema** | 4.25.0 | JSON schema validation | Zod, Ajv, JSON Schema validator |
| **dotmap** | 1.3.30 | Dot notation access to dicts | Lodash _.get() or custom |
| **deepmerge** | 2.0 | Deep merge dictionaries | Lodash _.merge() |
| **screeninfo** | 0.8.1 | Screen dimensions (for interactive mode) | window.screen API |

### Optional Dependencies
| Package | Version | Purpose | Browser Equivalent |
|---------|---------|---------|-------------------|
| **easyocr** | 1.7.2+ | OCR (neural network-based) | Tesseract.js |
| **ultralytics** | 8.0.0+ | YOLO ML models | TensorFlow.js, ONNX Runtime Web |
| **torch** | 2.0.0+ | PyTorch (ML backend) | TensorFlow.js (different framework) |
| **torchvision** | 0.15.0+ | Computer vision models | TensorFlow.js models |

---

## Discovered Entities

### Core Domain Entities (15)

1. **Template** (`src/processors/template/template.py`)
   - Central entity representing an OMR template
   - Contains field blocks, pre-processors, layout
   - Manages pipeline execution
   - References: 17+ files

2. **FieldBlock** (`src/processors/layout/field_block/base.py`)
   - Container for related fields
   - Defines bubble areas, dimensions, layout strategy
   - Types: QType (integer, multiple choice, custom)
   - References: 12+ files

3. **Field** (`src/processors/layout/field/`)
   - Individual detection units (bubble, barcode, OCR)
   - Types: BubbleField, BarcodeField, OCRField
   - Contains position, dimensions, metadata
   - References: 15+ files

4. **ProcessingContext** (`src/processors/base.py`)
   - State container passed through pipeline
   - Contains images, template, responses, metadata
   - Aggregates file-level metrics
   - References: 25+ files

5. **Config** (`src/schemas/models/config.py`)
   - Tuning configuration (thresholding, outputs, processing)
   - Hierarchical: defaults → global → local
   - Pydantic-validated
   - References: 30+ files

6. **EvaluationConfig** (`src/processors/evaluation/evaluation_config.py`)
   - Answer keys, marking schemes
   - Question set selection
   - Score calculation rules
   - References: 8+ files

7. **ThresholdStrategy** (`src/processors/threshold/threshold_strategy.py`)
   - Base class for thresholding algorithms
   - Implementations: Global, Adaptive, Local
   - References: 10+ files

8. **DetectionResult** (`src/processors/detection/models/detection_results.py`)
   - Results from detection passes
   - Types: BubbleFieldDetectionResult, OCRFieldDetectionResult, BarcodeFieldDetectionResult
   - Contains confidence, values, metadata
   - References: 20+ files

9. **ProcessingPipeline** (`src/processors/pipeline.py`)
   - Orchestrates processor execution
   - Processors: Preprocessing → Alignment → Detection
   - Extensible with add_processor()
   - References: 5+ files

10. **Processor (Base)** (`src/processors/base.py`)
    - Unified interface for all processors
    - process(context) → context pattern
    - References: 30+ files

11. **WorkflowSession** (`src/processors/visualization/workflow_session.py`)
    - Tracks workflow execution for visualization
    - Captures images, states, transitions
    - Exports to HTML reports
    - References: 4+ files

12. **TemplateLayout** (`src/processors/layout/template_layout.py`)
    - Handles template layout parsing
    - Field block placement, validation
    - References: 8+ files

13. **SaveImageOps** (`src/processors/template/template.py`)
    - Manages debug image saving
    - Image stacking for visualization
    - Level-based filtering
    - References: 10+ files

14. **DetectionRepository** (`src/processors/repositories/detection_repository.py`)
    - Stores and aggregates detection results
    - File-level and template-level aggregates
    - References: 6+ files

15. **MLBubbleDetector** (`src/processors/detection/ml_detector.py`)
    - Machine learning-based bubble detection
    - YOLO model integration
    - Fallback to traditional detection
    - References: 5+ files

---

## Discovered Flows (20 Major Flows)

### 1. Main Processing Flow
**Entry**: `main.py` → `src/entry.py:entry_point()`
**Steps**:
1. Parse CLI arguments
2. Validate paths
3. Traverse directories recursively
4. For each directory with images:
   - Load/inherit template + config + evaluation
   - Process files (sequential or parallel)
   - Export results to CSV
5. Print statistics

**Code References**:
- `main.py:177-215`
- `src/entry.py:29-228`

---

### 2. Single File Processing Flow
**Entry**: `src/entry.py:process_single_file()`
**Steps**:
1. Read image (gray + colored)
2. Create ProcessingContext
3. Run pipeline: template.process_file()
   - Preprocessing
   - Alignment
   - Detection
4. Extract OMR response
5. Evaluate (if evaluation config exists)
6. Draw output image
7. Save results (CSV, images, metrics)
8. Handle errors/multi-marks

**Code References**: `src/entry.py:277-530`

---

### 3. Pipeline Processing Flow
**Entry**: `src/processors/pipeline.py:process_file()`
**Steps**:
1. Create ProcessingContext
2. Execute processors sequentially:
   - PreprocessingCoordinator
   - AlignmentProcessor
   - (Optional) MLFieldBlockDetector
   - (Optional) ShiftDetectionProcessor
   - ReadOMRProcessor
   - (Optional) TrainingDataCollector
3. Return updated context

**Code References**: `src/processors/pipeline.py:117-156`

---

### 4. Preprocessing Flow
**Entry**: `src/processors/image/coordinator.py:PreprocessingCoordinator.process()`
**Steps**:
1. Load template pre-processors from config
2. For each pre-processor:
   - AutoRotate (if configured)
   - CropOnMarkers/CropPage (if configured)
   - Filters: GaussianBlur, Contrast, Levels (if configured)
3. Return preprocessed images

**Preprocessors**:
- AutoRotate: Detect and correct rotation
- CropOnMarkers: Marker-based cropping (dots, lines, custom)
- CropPage: Page detection and perspective correction
- GaussianBlur: Noise reduction
- Contrast: Enhance contrast
- Levels: Adjust brightness levels
- MedianBlur: Additional noise reduction
- FeatureBasedAlignment: Feature matching (SIFT)

**Code References**: `src/processors/image/coordinator.py`

---

### 5. Bubble Detection Flow (Threshold-Based)
**Entry**: `src/processors/detection/bubbles_threshold/file_runner.py:BubblesThresholdFileRunner.run()`
**Steps**:
1. **Detection Pass**:
   - For each field block
   - Extract bubble ROIs
   - Detect contours
   - Filter valid bubbles
2. **Interpretation Pass**:
   - Apply threshold strategy
   - Calculate bubble darkness
   - Mark bubbles above threshold
   - Detect multi-marks
   - Generate field values
3. **Drawing** (if enabled):
   - Draw bubble outlines
   - Show darkness values
   - Highlight marked bubbles

**Code References**:
- Detection: `src/processors/detection/bubbles_threshold/detection_pass.py`
- Interpretation: `src/processors/detection/bubbles_threshold/interpretation_pass.py`
- Drawing: `src/processors/detection/bubbles_threshold/interpretation_drawing.py`

---

### 6. Barcode Detection Flow
**Entry**: `src/processors/detection/barcode/file_runner.py:BarcodeFileRunner.run()`
**Steps**:
1. **Detection Pass**:
   - Extract barcode ROI
   - Use PyZbar to decode
   - Extract barcode type, value
2. **Interpretation Pass**:
   - Validate barcode format
   - Generate field value
3. **Drawing** (if enabled):
   - Draw barcode bounding box
   - Show decoded value

**Code References**: `src/processors/detection/barcode/`

---

### 7. OCR Detection Flow
**Entry**: `src/processors/detection/ocr/file_runner.py:OCRFileRunner.run()`
**Steps**:
1. **Detection Pass**:
   - Extract OCR ROI
   - Use EasyOCR or Tesseract
   - Extract text with confidence
2. **Interpretation Pass**:
   - Filter by confidence threshold
   - Clean/normalize text
   - Generate field value
3. **Drawing** (if enabled):
   - Draw OCR bounding box
   - Show recognized text

**Code References**: `src/processors/detection/ocr/`

---

### 8. Alignment Flow (SIFT)
**Entry**: `src/processors/alignment/sift_matcher.py`
**Steps**:
1. Detect SIFT keypoints in input image
2. Detect SIFT keypoints in template image
3. Match keypoints using FLANN matcher
4. Filter good matches (Lowe's ratio test)
5. Compute homography matrix (RANSAC)
6. Warp input image to align with template

**Code References**: `src/processors/alignment/sift_matcher.py`

---

### 9. Evaluation Flow
**Entry**: `src/processors/evaluation/evaluation_meta.py:evaluate_concatenated_response()`
**Steps**:
1. Load answer key
2. For each field:
   - Compare response to answer key
   - Apply marking scheme (correct/wrong/unattempted)
   - Calculate marks
3. Aggregate score across sections
4. Generate evaluation metadata (question-wise breakdown)
5. Optionally export explanation CSV

**Code References**: `src/processors/evaluation/evaluation_meta.py:10-150`

---

### 10. Template Loading Flow
**Entry**: `src/processors/template/template.py:Template.__init__()`
**Steps**:
1. Load template.json
2. Validate schema (Pydantic)
3. Parse field blocks
4. Parse pre-processors
5. Initialize pipeline
6. Setup output directories
7. Initialize drawing utils

**Code References**: `src/processors/template/template.py:17-150`

---

### 11. Auto-Rotation Flow
**Entry**: `src/processors/image/AutoRotate.py:AutoRotate.apply_filter()`
**Steps**:
1. Detect page orientation
2. Try rotations: 0°, 90°, 180°, 270°
3. For each rotation:
   - Detect alignment features
   - Score alignment quality
4. Select best rotation
5. Rotate image

**Code References**: `src/processors/image/AutoRotate.py`

---

### 12. Marker Detection Flow
**Entry**: `src/processors/image/crop_on_patches/marker_detection.py`
**Steps**:
1. Detect marker type (dots, lines, custom)
2. **For dots**:
   - Detect circular markers
   - Find 4 corner dots
   - Compute transformation
3. **For lines**:
   - Detect line markers
   - Find intersections
   - Compute crop region
4. **For custom**:
   - Use custom marker detection
5. Warp/crop image

**Code References**: `src/processors/image/crop_on_patches/`

---

### 13. Parallel Processing Flow
**Entry**: `src/entry.py:process_directory_files()`
**Steps**:
1. Prepare file tasks (file_path, template, config, etc.)
2. Check if parallel processing enabled
3. **If parallel** (max_workers > 1):
   - Create ThreadPoolExecutor
   - Submit all tasks
   - Collect results as completed
   - Sort results by file counter
   - Write CSVs in order
4. **If sequential** (max_workers = 1):
   - Process files one by one
   - Write CSVs immediately

**Code References**: `src/entry.py:532-642`

---

### 14. CSV Export Flow
**Entry**: `src/utils/csv.py:thread_safe_csv_append()`
**Steps**:
1. Acquire CSV write lock
2. Open CSV file in append mode
3. Write row
4. Release lock

**Output Files**:
- Results.csv: Successfully processed
- Errors.csv: Error files
- MultiMarked.csv: Multi-marked files

**Code References**: `src/utils/csv.py`

---

### 15. Debug Image Saving Flow
**Entry**: `src/processors/template/template.py:SaveImageOps.save_image_stacks()`
**Steps**:
1. Collect all saved images for file
2. Filter by debug level
3. Stack images (N per row)
4. Save composite image
5. Clear saved images

**Code References**: `src/processors/template/template.py`

---

### 16. ML Bubble Detection Flow
**Entry**: `src/processors/detection/ml_detector.py:MLBubbleDetector.process()`
**Steps**:
1. Load YOLO model
2. Run inference on image
3. Extract bubble bounding boxes
4. Filter by confidence threshold
5. **Fallback**: If low confidence, use traditional detection
6. Return detection results

**Code References**: `src/processors/detection/ml_detector.py`

---

### 17. Shift Detection Flow
**Entry**: `src/processors/detection/shift_detection_processor.py:ShiftDetectionProcessor.process()`
**Steps**:
1. Detect field shifts using ML
2. Predict shift direction and magnitude
3. Adjust field block positions
4. Update template layout
5. Continue with detection

**Code References**: `src/processors/detection/shift_detection_processor.py`

---

### 18. File Organization Flow
**Entry**: `src/processors/organization/processor.py:FileOrganizerProcessor.process()`
**Steps**:
1. Load file grouping config
2. Evaluate dynamic patterns (field values, scores)
3. Determine target directory
4. Move/copy file to target
5. Update output paths

**Code References**: `src/processors/organization/processor.py`

---

### 19. Workflow Visualization Flow
**Entry**: `src/processors/visualization/workflow_tracker.py:track_workflow()`
**Steps**:
1. Create WorkflowSession
2. For each processing step:
   - Capture image state
   - Record processor name
   - Store metadata
3. Export to HTML report
4. Generate interactive workflow graph

**Code References**: `src/processors/visualization/workflow_tracker.py`

---

### 20. Template Layout Drawing Flow
**Entry**: `src/processors/layout/template_drawing.py:TemplateDrawing.draw_template_layout()`
**Steps**:
1. For each field block:
   - Draw field outlines
   - Draw bubble labels
   - Show detected values
2. Draw evaluation results (if available):
   - Correct (green)
   - Wrong (red)
   - Unattempted (gray)
3. Draw scores, legends

**Code References**: `src/processors/layout/template_drawing.py`

---

## Integration Points

### File Formats

**Input**:
- Images: PNG, JPG, JPEG
- Template: template.json (JSON schema-validated)
- Config: config.json (JSON schema-validated)
- Evaluation: evaluation.json (JSON schema-validated)

**Output**:
- Results.csv: CSV with detected responses + scores
- Errors.csv: CSV with error files
- MultiMarked.csv: CSV with multi-marked files
- CheckedOMRs/: PNG images with marked bubbles
- ImageAnalysis/: Debug image stacks
- metrics.json: Processing metrics (if enabled)
- explanation_*.csv: Per-file scoring explanations

### External Libraries Used

**OpenCV Operations** (100+ calls):
- Image I/O: imread, imwrite, cvtColor
- Thresholding: threshold, adaptiveThreshold
- Morphology: dilate, erode, morphologyEx
- Contours: findContours, drawContours, boundingRect
- Transformations: warpPerspective, getAffineTransform, getPerspectiveTransform
- Feature detection: SIFT, ORB (opencv-contrib)
- Matching: FlannBasedMatcher, BFMatcher
- Geometry: minAreaRect, convexHull, approxPolyDP
- Filters: GaussianBlur, medianBlur, bilateralFilter
- Drawing: rectangle, circle, putText, line

**NumPy Operations** (50+ patterns):
- Array creation: np.zeros, np.ones, np.array
- Array manipulation: reshape, transpose, concatenate
- Statistical: mean, std, percentile, sum
- Mathematical: sqrt, abs, round, clip
- Boolean indexing: array[condition]
- Broadcasting: element-wise operations

**PyZbar**:
- decode(): Decode barcodes from image

**EasyOCR/Tesseract**:
- readtext(): Extract text from image ROI
- Image to text with confidence scores

**Pandas**:
- DataFrame creation and manipulation
- CSV reading/writing

**Rich**:
- Console output with styling
- Tables for config summary
- Progress bars (if needed)

### System Integration

**File System**:
- Directory traversal (recursive)
- File movement (errors, multi-marked)
- Path handling (pathlib)
- Temp file creation

**Concurrency**:
- ThreadPoolExecutor for parallel processing
- Thread-safe CSV writes (Lock)
- Shared state management

**CLI**:
- argparse for argument parsing
- Interactive mode (setLayout)
- Debug mode (verbose logging)

---

## Architecture Patterns

### 1. Strategy Pattern
**Usage**: Thresholding strategies
**Classes**: ThresholdStrategy, GlobalThresholdStrategy, AdaptiveThresholdStrategy, LocalThresholdStrategy
**References**: `src/processors/threshold/`

### 2. Template Method Pattern
**Usage**: File-level runners
**Classes**: FileLevelRunner, FieldTypeFileLevelRunner
**References**: `src/processors/detection/base/file_runner.py`

### 3. Pipeline Pattern
**Usage**: Processing pipeline
**Classes**: ProcessingPipeline, Processor
**References**: `src/processors/pipeline.py`, `src/processors/base.py`

### 4. Repository Pattern
**Usage**: Detection result storage
**Classes**: DetectionRepository
**References**: `src/processors/repositories/detection_repository.py`

### 5. Builder Pattern
**Usage**: Template construction
**Classes**: Template
**References**: `src/processors/template/template.py`

### 6. Observer Pattern
**Usage**: Workflow tracking
**Classes**: WorkflowTracker, WorkflowSession
**References**: `src/processors/visualization/`

### 7. Factory Pattern (Implicit)
**Usage**: Processor creation
**Location**: ProcessingPipeline creates processors based on config
**References**: `src/processors/pipeline.py`

---

## Known Edge Cases

### Image Quality
1. **Low resolution** (< 640x480): May fail marker detection
2. **Poor lighting**: Requires adaptive thresholding
3. **Xeroxed/degraded**: Needs blur + contrast adjustment
4. **Skewed/rotated**: Auto-rotation handles up to 360°
5. **Perspective distortion**: 4-point warping corrects

### Bubble Detection
6. **Stray marks**: Threshold tuning required
7. **Partial fills**: Configurable fill threshold
8. **Multi-marks**: Detected and flagged
9. **No marks**: Recorded as empty
10. **Overlapping bubbles**: Contour separation logic

### Alignment
11. **No features detected**: Fallback to no alignment
12. **Insufficient matches**: Requires minimum matches for homography
13. **Template mismatch**: Warning logged, alignment skipped

### Template Configuration
14. **Invalid JSON**: Schema validation error
15. **Overlapping fields**: Validation warning
16. **Out-of-bounds fields**: Validation error
17. **Missing required fields**: Schema validation error

### Evaluation
18. **Multi-correct answers**: Supported via config
19. **Negative marking**: Supported via marking scheme
20. **Partial credit**: Supported via marking scheme
21. **Section-wise scoring**: Supported via section config

---

## Key Metrics

### Performance
- **Processing Speed**: ~200 OMRs/minute (4 workers, medium complexity)
- **Accuracy**: ~100% on scanned, ~90% on mobile
- **Memory Usage**: ~100-500 MB per batch (varies by image count/size)

### Code Metrics
- **Cyclomatic Complexity**: High in detection/alignment modules
- **Test Coverage**: ~70% (estimated from test file count)
- **Module Cohesion**: High (processors are well-separated)
- **Coupling**: Medium (context object shared across processors)

---

## SKIP Items (Browser Migration)

### Features Not Needed in Browser

1. **CLI Interface** (`src/cli/`, `main.py`)
   - Reason: Browser has UI controls
   - Alternative: Form inputs, buttons, file upload

2. **Directory Crawling** (`src/entry.py:process_directory_wise`)
   - Reason: Browser can't access file system arbitrarily
   - Alternative: File input element with multiple selection

3. **File Movement** (Error/MultiMarked directories)
   - Reason: Browser can't move files on disk
   - Alternative: Download separate CSVs, mark in results

4. **ML Model Training** (`src/processors/training/`, `src/training/`)
   - Reason: Training is server-side, not browser
   - Alternative: Use pre-trained models only

5. **Parallel Threading** (ThreadPoolExecutor)
   - Reason: Browser has Web Workers instead
   - Alternative: Spawn Web Workers per file

6. **Rich Terminal UI** (`rich` library)
   - Reason: Browser has HTML/CSS/JS for UI
   - Alternative: Progress bars, tables in HTML

7. **Matplotlib/SciPy** (Heavy scientific libraries)
   - Reason: Large bundle size, not browser-friendly
   - Alternative: Custom implementations for needed functions

8. **Screen Dimension Detection** (`screeninfo`)
   - Reason: Browser has window.screen API
   - Alternative: window.screen.width/height

---

## Summary

**Entities**: 15 core domain entities
**Flows**: 20 major processing flows
**Dependencies**: 13 core, 4 optional (ML/OCR)
**Processors**: 30+ processor classes
**Patterns**: 7 architectural patterns identified
**Edge Cases**: 21+ documented edge cases
**SKIP Items**: 8 features not needed in browser

**Next Phase**: Begin Phase 1 - Foundation Patterns documentation

---

**Task 0.5 Status**: ✅ Completed
