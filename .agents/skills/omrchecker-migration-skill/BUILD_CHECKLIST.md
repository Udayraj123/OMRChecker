# BUILD CHECKLIST: OMRChecker Migration Documentation

**Status**: ✅ COMPLETE - All documentation extracted and validated
**Created**: 2026-02-20
**Completed**: 2026-02-21
**Total Tasks**: 88 (all completed)

---

## Progress Summary

| Category | Completed | Total | % |
|----------|-----------|-------|---|
| Foundation | 5 | 5 | 100% |
| Domain | 55 | 55 | 100% |
| Technical | 10 | 10 | 100% |
| Integration | 5 | 5 | 100% |
| Migration Context | 5 | 5 | 100% |
| Documentation Setup | 5 | 5 | 100% |
| Finalization | 3 | 3 | 100% |
| **TOTAL** | **88** | **88** | **100%** |

---

## Phase 0: Documentation Setup

### Task 0.1: Create SKILL.md
**Status**: ✅ completed
**Module**: core/
**Summary**: Created orchestrator file with load rules, progressive disclosure patterns, and skill metadata

### Task 0.2: Create core/boundaries.md
**Status**: ✅ completed
**Module**: core/
**Summary**: Documented system boundaries, what OMRChecker does/doesn't do, traffic patterns, constraints

### Task 0.3: Create core/quick-ref.md
**Status**: ✅ completed
**Module**: core/
**Summary**: Created quick reference with 10 common operations, code references, browser migration tips

### Task 0.4: Set up directory structure
**Status**: ✅ completed
**Module**: root
**Summary**: Created complete module directory structure (foundation, domain, technical, integration, migration)

### Task 0.5: Initial codebase scan report
**Status**: ✅ completed
**Module**: root
**Summary**: Documented 15 entities, 20 major flows, 13 core dependencies, 30+ processors, 21+ edge cases in CODEBASE_SCAN_REPORT.md

---

## Phase 1: Foundation Patterns

### Task 1.1: Document error handling
**Status**: ✅ completed
**Module**: modules/foundation/
**File**: error-handling.md
**Python Reference**: src/exceptions.py
**Summary**: Documented 29 exception types, 7 categories, hierarchical error handling, context-rich errors, browser migration patterns

### Task 1.2: Document logging patterns
**Status**: ✅ completed
**Module**: modules/foundation/
**File**: logging.md
**Python Reference**: src/utils/logger.py
**Summary**: Documented 5 log levels, Rich integration, thread-safe logging, dynamic log level control, browser console mapping

### Task 1.3: Document testing patterns
**Status**: ✅ completed
**Module**: modules/foundation/
**File**: testing.md
**Python Reference**: src/tests/conftest.py, pytest.ini
**Summary**: Documented pytest config, image snapshot testing, 5+ fixtures, parallel execution, Jest/Vitest migration

### Task 1.4: Document validation patterns
**Status**: ✅ completed
**Module**: modules/foundation/
**File**: validation.md
**Python Reference**: src/schemas/, src/utils/validations.py
**Summary**: Documented JSON Schema validation, Pydantic models, error display with Rich tables, Zod/Ajv browser migration

### Task 1.5: Document configuration management
**Status**: ✅ completed
**Module**: modules/foundation/
**File**: configuration.md
**Python Reference**: src/schemas/defaults/, src/utils/parsing.py
**Summary**: Documented hierarchical config (defaults → global → local), deep merging, 4 config sections, localStorage browser migration

---

## Phase 2: Domain - Core Entities

### Task 2.1: Template Entity
**Status**: ✅ completed
**Module**: modules/domain/template/
**Files**: concept.md
**Python Reference**: src/processors/template/template.py, src/schemas/models/template.py
**Summary**: Documented Template class architecture, TemplateConfig dataclass, template lifecycle (init → setup → process → export), template JSON structure (minimal & full examples), template mutations and copy strategy, browser migration with TypeScript interfaces and Zod validation

### Task 2.2: FieldBlock Entity
**Status**: ✅ completed
**Module**: modules/domain/field-block/
**Files**: concept.md
**Python Reference**: src/processors/layout/field_block/base.py
**Summary**: Documented FieldBlock class, 3 field detection types (BUBBLES_THRESHOLD, OCR, BARCODE_QR), field generation from labels, shift management for preprocessing, bubble field types (built-in and custom), field label parsing patterns, browser migration with TypeScript and Zod validation

### Task 2.3: Field Entity
**Status**: ✅ completed
**Module**: modules/domain/field/
**Files**: concept.md
**Python Reference**: src/processors/layout/field/base.py, bubble_field.py, ocr_field.py, barcode_field.py
**Summary**: Documented Field base class and 3 concrete types (BubbleField, OCRField, BarcodeField), ScanBox entity (smallest unit), hierarchy (Template → FieldBlock → Field → ScanBox), shift propagation and shifted position calculation, field ID format, browser migration with TypeScript abstract classes and factory pattern

### Task 2.4: ProcessingContext Entity
**Status**: ✅ completed
**Module**: modules/domain/processing-context/
**Files**: concept.md
**Python Reference**: src/processors/base.py:11-43
**Summary**: Documented ProcessingContext dataclass architecture, 4 field categories (Input, Detection Results, Evaluation, Metadata), context flow through pipeline stages (Preprocessing → Alignment → Detection → Evaluation), usage patterns for processors, browser migration with TypeScript interface and optional immutable updates

### Task 2.5: Config Entity
**Status**: ✅ completed
**Module**: modules/domain/config/
**Files**: concept.md
**Python Reference**: src/schemas/models/config.py, src/schemas/defaults/config.py
**Summary**: Documented TuningConfig dataclass with 4 sections (Thresholds, Outputs, Processing, ML), config hierarchy (Default → Global → Local), config usage patterns, default config values, config file examples (minimal, development, production), browser migration with TypeScript interfaces, Zod validation, and browser-specific adaptations (Web Workers, downloads, UI display)

---

## Phase 3: Domain - Processing Flows

### Task 3.1: Main Pipeline Flow
**Status**: ✅ completed
**Module**: modules/domain/pipeline/
**Files**: concept.md
**Python Reference**: src/processors/pipeline.py, src/entry.py
**Summary**: Documented ProcessingPipeline class, unified Processor interface, 6 pipeline stages (Preprocessing, Alignment, ML Field Block Detection, Shift Detection, Detection/Interpretation, Training Data Collection), end-to-end flow from file to CSV (discovery → parallel processing → single file → CSV output), error handling & recovery (3 categories: marker errors, processing errors, multi-marked files), browser migration with async/await, Web Workers, File API, and download instead of file writes

### Task 3.2: Preprocessing Flow
**Status**: ✅ completed
**Module**: modules/domain/preprocessing/
**Files**: concept.md
**Python Reference**: src/processors/image/coordinator.py, src/processors/image/base.py
**Summary**: Documented PreprocessingCoordinator orchestration, preprocessing flow (template copy → initial resize → preprocessor execution → final resize), ImageTemplatePreprocessor base class, preprocessor types (geometric transformations, image enhancement, advanced transformations), visualization debug mode, browser migration with TypeScript and OpenCV.js

### Task 3.3: AutoRotate Processor
**Status**: ✅ completed
**Module**: modules/domain/preprocessing/auto-rotate/
**Files**: flows.md
**Python Reference**: src/processors/image/AutoRotate.py
**Summary**: Documented AutoRotate configuration (referenceImage, markerDimensions, threshold), rotation detection flow (try all 4 rotations → template matching → threshold check → apply rotation), template matching details (normalized cross-correlation with TM_CCOEFF_NORMED), edge cases (missing reference, low match score, similar scores), browser migration with OpenCV.js cv.matchTemplate and cv.rotate

### Task 3.4: CropOnMarkers Processor
**Status**: ✅ completed
**Module**: modules/domain/preprocessing/crop-on-markers/
**Files**: flows.md
**Python Reference**: src/processors/image/CropOnMarkers.py, src/processors/image/crop_on_patches/
**Summary**: Documented 4 marker types (FOUR_DOTS, TWO_DOTS_ONE_LINE, TWO_LINES, CUSTOM), FOUR_DOTS flow (marker detection → corner ordering → perspective transform → template shift update), detection algorithms (SIMPLE and HOUGH_CIRCLES), edge cases (marker not detected, ambiguous positions, large perspective distortion), browser migration with OpenCV.js contour detection and perspective transform

### Task 3.5: CropPage Processor
**Status**: ✅ completed
**Module**: modules/domain/preprocessing/crop-page/
**Files**: flows.md
**Python Reference**: src/processors/image/CropPage.py, src/processors/image/page_detection.py
**Summary**: Documented CropPage configuration, 4-step flow (edge detection → contour detection → corner detection → perspective transform), algorithm (Canny edges, morphological operations, largest contour selection, polygon approximation), browser implementation with OpenCV.js Canny and findContours

### Task 3.6: Image Filters (Blur, Contrast, Levels)
**Status**: ✅ completed
**Module**: modules/domain/preprocessing/filters/
**Files**: flows.md
**Python Reference**: src/processors/image/GaussianBlur.py, src/processors/image/Contrast.py, src/processors/image/Levels.py
**Summary**: Documented 3 filters: GaussianBlur (noise reduction with kSize and sigmaX), Contrast (adjust contrast/brightness with alpha and beta), Levels (histogram adjustment with low/high/gamma), configuration examples, algorithms, browser implementation with OpenCV.js filter functions

### Task 3.7: Warp Strategies
**Status**: ✅ completed
**Module**: modules/domain/preprocessing/warping/
**Files**: flows.md
**Python Reference**: src/processors/image/warp_strategies.py, src/processors/image/WarpOnPointsCommon.py
**Summary**: Documented 4 warp strategies (Perspective 4-point, Affine 3-point, Piecewise Affine with Delaunay triangulation, Homography feature-based), algorithms with code examples, configuration format, edge cases (collinear points, extreme perspective, out-of-bounds coordinates), browser support via OpenCV.js getPerspectiveTransform and warpPerspective

### Task 3.8: Alignment Flow
**Status**: ✅ completed
**Module**: modules/domain/alignment/
**Files**: concept.md, flows.md, decisions.md, constraints.md, integration.md
**Python Reference**: src/processors/alignment/processor.py
**Summary**: Documented 4 alignment methods (SIFT K-Nearest active), per-field-block alignment, coordinate warping strategy, MIN_MATCH_COUNT=10, browser fallbacks (ORB/AKAZE)

### Task 3.9: SIFT Alignment
**Status**: ✅ completed
**Module**: modules/domain/alignment/sift/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/alignment/sift_matcher.py
**Summary**: SIFT singleton pattern, FLANN KD-tree matching (k=2), Lowe's ratio test (simplified to 1.0), MIN_MATCH_COUNT=10, RANSAC homography, 128-dim descriptors, browser fallbacks to ORB/AKAZE

### Task 3.10: Phase Correlation Alignment
**Status**: ✅ completed
**Module**: modules/domain/alignment/phase-correlation/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/alignment/phase_correlation.py
**Summary**: FFT-based translation detection (commented/unused), rigid shift only, O(n log n) complexity, faster than SIFT but limited to translation, browser FFT via fft.js/ml-fft

### Task 3.11: Template Alignment
**Status**: ✅ completed
**Module**: modules/domain/alignment/template-matching/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/alignment/template_alignment.py
**Summary**: Documented normalized cross-correlation (cv2.TM_CCOEFF_NORMED), multi-scale template matching, AutoRotate rotation detection, custom marker detection, comparison with SIFT (single position vs multiple correspondences), performance characteristics (10-30ms Python, 30-80ms browser), 8 edge cases (scale/rotation sensitivity, multiple matches, border effects), browser support (excellent with OpenCV.js), memory management patterns

### Task 3.12: Piecewise Affine Alignment
**Status**: ✅ completed
**Module**: modules/domain/alignment/piecewise-affine/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/alignment/piecewise_affine_delaunay.py
**Summary**: Documented Delaunay triangulation algorithm (cv2.Subdiv2D, ~2n-5 triangles for n points), piecewise affine transformation (per-triangle 2x3 affine matrix with cv2.getAffineTransform), 9-step flow (parse pairs → build Delaunay → filter triangles → map correspondences → per-triangle warp with cv2.warpAffine → triangle masking with cv2.fillConvexPoly → blend), warp_triangle_inplace() implementation (bounding box crop, origin shift, affine transform, INTER_LINEAR interpolation, BORDER_REFLECT_101, anti-aliased triangle masking), replace_triangle_inplace() blending (white/black triangle masks, element-wise multiplication), current status (unused due to overlapping field blocks limitation), performance O(n log n + N×area) where N=triangles (100-500ms Python, 300-1000ms browser), constraints (minimum 3 points, strict rectangle bounds, collinearity check, 12 edge cases including empty pairs, all triangles filtered, extreme distortions, coordinate rounding), browser migration with OpenCV.js Subdiv2D/getAffineTransform/warpAffine, memory management (must delete cv.Mat), comparison with K-Nearest (higher accuracy, slower, no overlap support vs lower accuracy, faster, overlap support)

### Task 3.13: K-Nearest Interpolation
**Status**: ✅ completed
**Module**: modules/domain/alignment/k-nearest/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/alignment/k_nearest_interpolation.py
**Summary**: Documented K-nearest interpolation algorithm (K=4 default), coordinate-based alignment using SIFT displacement pairs, find_k_nearest_anchors() with Euclidean distance sorting, shift_by_field_blocks() strategy (single shift per field block), field-level and bubble-level alternatives (commented), step-by-step flow from SIFT matches to scan_box.shifts, edge cases (zero pairs, K>anchors, truncation vs rounding), browser migration with JavaScript arrays/TensorFlow.js, performance O(m log m) for K-nearest search

---

## Phase 4: Domain - Detection Systems

### Task 4.1: ReadOMR Processor
**Status**: ✅ completed
**Module**: modules/domain/detection/
**Files**: concept.md, flows.md, decisions.md, constraints.md, integration.md
**Python Reference**: src/processors/detection/processor.py
**Summary**: Documented ReadOMRProcessor orchestrator architecture with TemplateFileRunner delegation, two-pass processing (detection → interpretation), multi-format field dispatch (BUBBLES_THRESHOLD, OCR, BARCODE_QR via strategy pattern), ML fallback system (HybridDetectionStrategy with confidence-based triggering at 0.75 threshold), ShiftDetectionProcessor integration (early exit optimization), image normalization flow (resize to template dimensions → normalize intensity), custom output labels (concatenated response), DetectionRepository pattern for typed storage, directory-level summary with ML statistics, performance characteristics (150ms/image traditional, 350ms with ML fallback), constraints (95-99% accuracy, 50MB memory/image, dependency availability, pipeline ordering), browser migration with async processing, Web Workers for ML, TensorFlow.js/ONNX Runtime for YOLO, Tesseract.js for OCR, @zxing/library for barcodes, progressive result streaming, IndexedDB model caching

### Task 4.2: Bubble Detection - Threshold Strategy
**Status**: ✅ completed
**Module**: modules/domain/detection/bubbles-threshold/
**Files**: concept.md, flows.md, decisions.md, constraints.md, integration.md
**Python Reference**: src/processors/detection/bubbles_threshold/
**Summary**: Documented classical threshold-based bubble detection system with 2-phase architecture (detection + interpretation), threshold strategies (Global/Local with confidence-based fallback), typed detection models (BubbleFieldDetectionResult, BubbleMeanValue, ThresholdResult), complete edge case handling (empty fields, single/two bubbles, all marked/unmarked, Xeroxed OMRs, ROI out-of-bounds), multi-marking detection, confidence scoring (5 factors: threshold/margin/scan quality/multi-mark/disparity), ML fallback integration, DetectionRepository pattern, performance characteristics (O(m×w×h) detection, O(m log m) thresholding), browser migration with OpenCV.js and Web Workers, memory management patterns, configuration tuning for different scan qualities

### Task 4.3: Bubble Detection - Detection Pass
**Status**: ✅ completed
**Module**: modules/domain/detection/bubbles-threshold/detection-pass/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/detection/bubbles_threshold/detection_pass.py
**Summary**: Documented detection pass architecture with BubblesThresholdDetectionPass orchestrator and BubblesFieldDetection per-field processor. Covers ROI extraction using cv2.mean() for bubble intensity measurement, typed BubbleFieldDetectionResult with auto-calculated properties (std_deviation, scan_quality, jumps), repository pattern for storage, shifted position calculation, 5-phase detection flow (initialization → field detection → ROI extraction → repository storage → aggregate updates). Constraints include OpenCV.js memory management patterns (explicit roi.delete()), 10 edge cases (empty bubbles, out-of-bounds ROI, all-same-intensity, single bubble), browser-specific limits (Safari WASM, mobile memory, Firefox GC), performance optimization (batch ROI extraction, Web Worker pool, property caching), validation checklist. Complete browser migration with TypeScript models, memory-safe implementations, quality assessment (EXCELLENT/GOOD/ACCEPTABLE/POOR based on std dev).

### Task 4.4: Bubble Detection - Interpretation Pass
**Status**: ✅ completed
**Module**: modules/domain/detection/bubbles-threshold/interpretation-pass/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/detection/bubbles_threshold/interpretation_pass.py
**Summary**: Documented complete interpretation pass flow with file-level orchestration (BubblesThresholdInterpretationPass) and field-level interpretation (BubblesFieldInterpretation). Covered 7-step interpretation flow: extract detection result → create threshold config → calculate threshold (LocalThresholdStrategy with global fallback) → interpret bubbles (is_attempted = mean_value < threshold) → check multi-marking → generate interpretation string → calculate confidence metrics. Documented BubbleInterpretation model, threshold calculation logic (single bubble/two bubbles/3+ bubbles cases), multi-marking detection, all-bubbles-marked edge case, confidence score calculation with 5 factors (threshold/margin/scan quality/penalties), disparity analysis. Covered performance (O(B log B) per field, ~0.1ms typical, ~64 KB memory per file), 12 edge cases (single bubble, two bubbles, all same intensity, no marks, all marks, multi-marks, disparity, missing results, xeroxed OMR, low scan quality, empty results, outliers), browser compatibility (Chrome/Firefox/Safari 90+, Web Workers, typed arrays), validation tests, error handling patterns. Browser migration with TypeScript implementation, LocalThresholdStrategy class, confidence metrics calculation, async/await support, memory management for batch processing (100 files safe limit).

### Task 4.5: Bubble Detection - Statistics
**Status**: ✅ completed
**Module**: modules/domain/detection/bubbles-threshold/stats/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/detection/bubbles_threshold/stats.py, src/processors/detection/models/detection_results.py
**Summary**: Documented MeanValueItem generic wrapper, BubbleMeanValue dataclass, BubbleFieldDetectionResult with 9 auto-calculated statistical properties (std_deviation, scan_quality with 4 levels, is_reliable, sorted access, jump detection, min/max tracking), complete flow diagrams for all statistical calculations, edge cases (empty/single/all-same/outliers), performance characteristics (O(n log n) sorting, O(n) std), browser migration with manual std calculation, TypeScript getters, property caching patterns, full validation and concurrency constraints

### Task 4.6: Bubble Detection - Drawing
**Status**: ✅ completed
**Module**: modules/domain/detection/bubbles-threshold/drawing/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/detection/bubbles_threshold/interpretation_drawing.py
**Summary**: Documented BubblesFieldInterpretationDrawing class with two drawing modes (with/without verdicts). Covered visual feedback flow including verdict colors (correct/incorrect/neutral/bonus), verdict symbols (+/-/o/*), answer group edge indicators (4 groups max), and DrawingUtils integration. Browser migration includes Canvas 2D API rendering, color space conversion (BGR→RGB), High DPI support, OffscreenCanvas optimization, and grayscale pattern alternatives. Documented constraints: performance targets (<100ms for 400 bubbles), answer group limit (4 edges), text overflow handling, font rendering differences, and memory management strategies

### Task 4.7: Bubble Interpretation Logic
**Status**: ✅ completed
**Module**: modules/domain/detection/bubbles-threshold/interpretation/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/detection/bubbles_threshold/bubble_interpretation.py, interpretation.py
**Summary**: Documented bubble interpretation algorithm with darkness-based detection (mean < threshold), BubbleInterpretation class, field interpretation flow, confidence metrics, 10 edge cases, performance, browser compatibility

### Task 4.8: Barcode Detection
**Status**: ✅ completed
**Module**: modules/domain/detection/barcode/
**Files**: concept.md, flows.md, decisions.md, constraints.md, integration.md
**Python Reference**: src/processors/detection/barcode/
**Summary**: Documented complete barcode detection system with PyZBar as primary library (singleton pattern, lazy loading from pyzbar.pyzbar.decode, C library wrapper), BarcodeDetection/BarcodeFieldDetection/BarcodeFieldDetectionResult architecture, scan-zone-based barcode extraction with absolute coordinate transformation, two-pass processing (detection → interpretation), PyZBar format support (QR Code/CODE-128/EAN-13/Data Matrix/PDF417), text post-processing (ASCII filtering, whitespace normalization, optional charset/maxLength), confidence scoring (default 0.8 threshold, quality 0-100 scale), typed detection models with DetectionRepository pattern, BarcodeInterpretation with concatenation and multi-marking detection, drawing system with bounding boxes and interpreted text labels, evaluation integration with color-coded verdict feedback, 8 key decisions (PyZBar selection, single barcode per field default, hardcoded 0.8 threshold with future TuningConfig, absolute coordinates, configurable empty value, full text processing, auto-detect all formats, flag multi-marking + concatenate), performance characteristics (15-30ms QR Code, 10-20ms CODE-128, ~90KB memory per field), edge cases (no barcode, multiple barcodes, low confidence, rotation handling, partial barcode with QR error correction, non-ASCII characters with ASCII-only filter, very long data, format ambiguity with quality-based selection), browser migration to @zxing/library (BrowserMultiFormatReader, 50-100ms decode time, no quality score workaround with binary success/failure, async-only API requiring Promise-based flow, Web Worker offloading to prevent main thread blocking, Canvas/ImageData conversion overhead, format hints for optimization, 2-3x slower than PyZBar with mitigation strategies), integration with ReadOMR processor via TemplateFileRunner, BarcodeFileRunner orchestrator with detection/interpretation passes, template system with BARCODE_QR field type registration, BarcodeField with single BarcodeScanBox, response generation with custom label concatenation, evaluation system for barcode verification, CSV output with barcode columns, visualization drawing on marked images, file organization by barcode value with regex matching

### Task 4.9: OCR Detection
**Status**: ✅ completed
**Module**: modules/domain/detection/ocr/
**Files**: concept.md, flows.md, decisions.md, constraints.md, integration.md
**Python Reference**: src/processors/detection/ocr/
**Summary**: Documented complete OCR detection system with EasyOCR as primary engine (singleton pattern, GPU acceleration, lazy loading), OCRDetection/OCRFieldDetection/OCRFieldDetectionResult architecture, zone-based text extraction with absolute coordinate transformation, two-pass processing (detection → interpretation), text post-processing (non-ASCII cleanup, whitespace normalization, charset filtering), confidence scoring (default 0.8 threshold), typed detection models with DetectionRepository pattern, OCRInterpretation with concatenation logic, drawing system with bounding boxes and text labels, evaluation integration with color-coded feedback, 8 key decisions (EasyOCR selection, singleton reader, single detection per field, absolute coordinates, 0.8 threshold, typed results, always-enabled post-processing, repository pattern), performance characteristics (200-500ms GPU, 1-2s CPU per field, ~500MB memory), edge cases (empty field, text outside zone, multiple words, tilted text, low contrast, noise, Unicode, very long text), browser migration to Tesseract.js (3-5x slower, 1-2s per field, ~150MB memory, Web Worker pool, model caching, PSM/OEM configuration, progressive optimization), integration with ReadOMR processor, template system, interpretation pipeline, evaluation system, CSV/image output generation, multi-format detection coexistence with bubbles/barcodes

### Task 4.10: ML Bubble Detector
**Status**: ✅ completed
**Module**: modules/domain/detection/ml-bubble/
**Files**: concept.md, flows.md, decisions.md, constraints.md, integration.md
**Python Reference**: src/processors/detection/ml_bubble_detector.py
**Summary**: Documented ML-based bubble detection with YOLOv8 integration as Stage 2 of hierarchical detection. Covered YOLO-based architecture (Processor interface, CLASS_NAMES dict for 3 classes, lazy model loading, confidence filtering), 3 bubble detection classes (bubble_empty, bubble_filled, bubble_partial with conservative interpretation treating partial as empty), hierarchical 2-stage detection pipeline (Stage 1 ML Field Block Detector provides bounding boxes → Stage 2 ML Bubble Detector processes crops → coordinate mapping to full image), data flow (ml_detected_blocks input → per-block ml_bubbles/ml_bubbles_count output), YOLO integration (ultralytics library, model.predict() with conf threshold, YOLOv8n nano variant ~6MB), coordinate mapping algorithm (crop coords + block offset → full image coords with bounds checking), hybrid fallback detection strategy (HybridDetectionStrategy with confidence-based triggering, ReadOMR integration, enable/disable pattern, statistics tracking), 6 flows (hierarchical detection, image cropping with NumPy views, YOLO inference with NMS, coordinate transformation, hybrid fallback, browser migration), YOLO inference details (640px input size, preprocessing/post-processing, confidence filtering at 0.7 default), memory management (NumPy views not copies, crop validation, YOLO automatic cleanup), browser migration with TensorFlow.js (YOLO→ONNX→TF→TFJS conversion pipeline, custom NMS implementation, WebGL/WASM backends, singleton pattern for WebGL context limits, tensor disposal for memory), ONNX Runtime Web (direct YOLO export, better performance 20-30% faster, smaller models, NCHW layout conversion, output parsing), 7 key decisions (hybrid strategy for confidence-based fallback, hierarchical 2-stage for better accuracy/performance, YOLOv8 for best balance, 3-class granularity for actionable info, xyxy coordinate format for direct cropping, 0.7 confidence threshold for F1 score balance, ONNX Runtime Web primary with TensorFlow.js fallback), performance constraints (Python CPU 50-100ms/block, GPU 10-20ms, browser TensorFlow.js WebGL 30-50ms/WASM 200-300ms, ONNX WebGL 20-40ms/WASM 100-200ms, memory ~6MB model/~50MB Python RAM/~100MB browser), browser limitations (no direct YOLO support, manual NMS, memory leaks prevention, WebGL context limits, mobile RAM 1-2GB, thermal throttling), training constraints (minimum 1300+ samples, recommended 5000+, GPU 8GB+ VRAM, 100-200 epochs, augmentation for robustness, mAP@0.5 > 0.85 target), deployment constraints (singleton model instances, IndexedDB caching, optional ML dependencies with graceful degradation, version compatibility), edge cases (no field blocks → skip gracefully, model not found → disable detector, empty crops → validate and skip, coordinate issues → bounds clamping, CUDA OOM → fallback to CPU, overlapping bubbles → built-in NMS), integration with MLFieldBlockDetector (consumes Stage 1 bounding boxes), ReadOMR (hybrid fallback for low confidence < 0.75), ShiftDetectionProcessor (uses bubble positions for shift calculation), pipeline registration (optional stages via config), DetectionFusion (confidence-weighted merging), browser pipeline with async/await and Web Workers, testing strategy (unit tests for coordinate mapping, integration tests for full pipeline), configuration schema (MLConfig with enable flags, model paths, confidence thresholds), model quantization for browser (FP16 ~3MB, INT8 ~1.5MB), caching strategy with IndexedDB 7-day expiration, mobile browser optimizations (INT8 quantization, sequential processing, cooldown periods for thermal throttling)

### Task 4.11: ML Field Block Detector
**Status**: ✅ completed
**Module**: modules/domain/detection/ml-field-block/
**Files**: concept.md, flows.md, decisions.md, constraints.md, integration.md
**Python Reference**: src/processors/detection/ml_field_block_detector.py
**Summary**: Documented YOLO-based field block detection (Stage 1 hierarchical detection), STN preprocessing architecture (affine vs translation-only variants), confidence thresholds (default 0.7), bounding box parsing (xyxy → origin+dimensions), alignment adjustment algorithm (200px distance threshold, greedy nearest neighbor matching), 3 field block classes (MCQ/OCR/Barcode), performance characteristics (~100-150ms inference), browser migration with ONNX Runtime Web/TensorFlow.js, model quantization strategies, Web Worker offloading patterns, memory management (800MB GPU/1-4GB browser limits), edge cases (zero detections, template-ML mismatch, out-of-bounds coords), integration with shift detection and bubble detection processors

### Task 4.12: Shift Detection
**Status**: ✅ completed
**Module**: modules/domain/detection/shift-detection/
**Files**: concept.md, flows.md, decisions.md, constraints.md, integration.md
**Python Reference**: src/processors/detection/shift_detection_processor.py
**Summary**: Documented ML-based shift detection and validation system with dual detection strategy (shifted vs baseline comparison), shift validation algorithm (Euclidean distance magnitude check against configurable per-block or global margins), confidence adjustment system (linear interpolation 10-50% reduction based on mismatch severity), 7-step main flow (preconditions → validation → dual detection → comparison → confidence adjustment → context update), shift validation with per-block override margins (global 50px default, customizable per block), dual detection rationale (validates ML shifts actually improve accuracy, prevents silent failures), result comparison at bubble-level and field-level with severity calculation (mismatched_bubbles/total_bubbles), confidence reduction formula (min + severity × (max - min)), ShiftDetectionConfig with enabled flag, global/per-block max shift pixels, confidence reduction min/max bounds, integration at pipeline stage 4 (after MLFieldBlockDetector, before Detection/Interpretation), metadata flow (ml_block_alignments input → shift_detection output with applied shifts, comparison data, confidence adjustments), TemplateFileRunner reuse for detection execution, template mutation strategy (apply shifts → detect → reset shifts → baseline detect → compare), 7 key design decisions (dual detection for validation, linear confidence reduction for proportionality, per-block margins for flexibility, use shifted results with adjusted confidence, Euclidean distance for shift magnitude, explicit reset to prevent contamination, enable/disable flag for configurability), performance constraints (2× detection time ~500ms total, ~200KB peak memory for dual results during comparison), accuracy constraints (50px default max shift based on bubble spacing, 10-50% confidence reduction range), browser migration with TypeScript processor implementation, Web Worker parallel detection, performance.now() timing, IndexedDB metadata storage, Math.sqrt for magnitude calculation, Promise.all for parallel dual detection, graceful degradation on errors, memory management (null references after comparison), testing strategy (unit tests for validation/comparison/confidence, integration tests for full pipeline, performance tests for timing/memory)

### Task 4.13: Detection Fusion
**Status**: ✅ completed
**Module**: modules/domain/detection/fusion/
**Files**: concept.md, flows.md, decisions.md, constraints.md, integration.md
**Python Reference**: src/processors/detection/fusion/detection_fusion.py
**Summary**: Documented detection fusion architecture with 3 fusion strategies (confidence-weighted, ml_fallback, traditional_primary), confidence-based decision logic, discrepancy flagging system, field-to-ML-block mapping patterns, agreement detection algorithms, integration with ReadOMR/ShiftDetection processors, browser migration with TypeScript implementation, performance characteristics (O(n) complexity, ~2-5ms for 50 fields), accuracy constraints and trade-offs

### Task 4.14: STN Module (Spatial Transformer Network)
**Status**: ✅ completed
**Module**: modules/domain/detection/stn/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/detection/models/stn_module.py, src/processors/detection/models/stn_utils.py
**Summary**: Documented Spatial Transformer Network architecture with 4 model variants (SpatialTransformerNetwork, STNWithRegularization, TranslationOnlySTN, TranslationOnlySTNWithRegularization), 3-stage pipeline (localization CNN → grid generator → bilinear sampler), affine transformation flow (6 params for full affine, 2 params for translation-only), initialization to identity transform, forward pass with ~10K parameters, inference flow with model loading/saving utilities, visualization system, performance characteristics (35-80ms CPU, 8-18ms GPU), browser migration via PyTorch→ONNX→TensorFlow.js conversion, WebGL GPU acceleration (5-15ms), constraints including fixed input dimensions, affine transformation limits (cannot model curved/bent pages), memory requirements (~6.5MB per 640×640 image), regularization weight tuning (0.05-0.2 for stability), edge cases (identity transforms, extreme misalignment, blank images, out-of-bounds translations), browser-specific limitations (WASM heap limits, WebGL texture size, model loading time), validation checks for input/output shapes and transformation validity

---

## Phase 5: Domain - Thresholding Strategies

### Task 5.1: Threshold Strategy Pattern
**Status**: ✅ completed
**Module**: modules/domain/threshold/
**Files**: concept.md, flows.md, decisions.md, constraints.md, integration.md
**Python Reference**: src/processors/threshold/threshold_strategy.py
**Summary**: Documented strategy pattern with ThresholdStrategy base class, ThresholdConfig/ThresholdResult dataclasses, confidence-based system, 3 concrete strategies (Global, Local, Adaptive), integration with bubble detection pipeline, browser migration patterns

### Task 5.2: Global Threshold
**Status**: ✅ completed
**Module**: modules/domain/threshold/global/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/threshold/global_threshold.py
**Summary**: Documented global threshold algorithm with looseness parameter, max jump calculation, 7-step flow (validate → sort → initialize → find max jump → calculate confidence → check fallback → return), edge cases (empty/single/two bubbles, all same value, outliers), performance O(n log n), browser implementation with JavaScript sorting and Math operations

### Task 5.3: Adaptive Threshold
**Status**: ✅ completed
**Module**: modules/domain/threshold/adaptive/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/threshold/adaptive_threshold.py
**Summary**: Documented adaptive threshold meta-strategy that combines Global + Local strategies using confidence-weighted averaging (weights [0.4, 0.6]). Detailed weighted averaging algorithm with 7-step flow, 4 real-world examples (both confident, global confident, local confident, zero confidence), factory pattern usage, integration with bubble detection pipeline, mathematical constraints (bounds, division-by-zero prevention), performance O(k × m log m), browser migration with TypeScript interfaces, thread safety for Web Workers, 9 edge cases. Note: NOT OpenCV's cv2.adaptiveThreshold (image-level), but strategy composition pattern for bubble detection threshold calculation.

### Task 5.4: Local Threshold
**Status**: ✅ completed
**Module**: modules/domain/threshold/local/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/threshold/local_threshold.py
**Summary**: Documented local threshold algorithm with global fallback strategy, 8-step flow (initialize fallback → handle <2 bubbles → sort → handle 2 bubbles with gap check → handle 3+ bubbles with max jump → check confidence → calculate result → return), 5 distinct method types and confidence levels (0.0, 0.3, 0.4, 0.7, 0.8-1.0), special two-bubble handling (min_gap_two_bubbles=20.0), confident_jump threshold (min_jump + surplus = 40.0), 7 edge cases (empty/single/two bubbles small gap/two bubbles large gap/low confidence/all same/one filled), performance O(m log m) where m=2-20 bubbles per field, browser implementation with JavaScript sorting and manual mean calculation for two-bubble case, field-level independence constraint, immutable global_fallback for Web Worker safety

---

## Phase 6: Domain - Evaluation System

### Task 6.1: Evaluation Config
**Status**: ✅ completed
**Module**: modules/domain/evaluation/
**Files**: concept.md, flows.md, decisions.md, constraints.md, integration.md
**Python Reference**: src/processors/evaluation/evaluation_config.py
**Summary**: Documented comprehensive evaluation system architecture with EvaluationConfig and EvaluationConfigForSet classes. Covered two-level configuration (top-level routing + per-set scoring), three answer sources (local JSON, CSV, image-based), three answer types (standard, multiple-correct, multiple-correct-weighted), marking schemes (DEFAULT + custom sections + bonus + streak), conditional sets with regex-based routing (first-match-wins, parent-child merging), question-to-answer matching via AnswerMatcher, score calculation flow (prepare response → iterate questions → match → update score → metadata), evaluation metadata (QuestionMeta with verdict/delta/score, EvaluationMeta with aggregates), visual outputs (draw score, answers summary, question verdicts with symbols/colors), explanation table with conditional columns, format string validation, CSV answer key parsing (auto-detect answer type from column format), image-based answer key generation (process through template, validate non-empty, extract answers), parent-child question/answer merging (inherit + override + append), section marking schemes (disjoint questions, bonus detection, streak handling), verdict determination (standard/multiple-correct/weighted logic), delta calculation with streak bonuses (verdict-level vs section-level), explanation table export to CSV, 17 design decisions (two-level architecture, answer types, conditional sets, merge semantics, marking isolation, streak types, bonus detection, CSV format, etc.), 25+ edge cases (empty keys, all unmarked, negative scores, match failures, duplicates, overflows, etc.), performance benchmarks (100 questions <10ms, 1000 questions <50ms), browser migration with Zod validation, native RegExp, papaparse CSV, Blob downloads, Canvas rendering, and fetch/File API

### Task 6.2: Section Marking Scheme
**Status**: ✅ completed
**Module**: modules/domain/evaluation/section-marking/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/evaluation/section_marking_scheme.py
**Summary**: Documented SectionMarkingScheme class for per-section scoring rules. Covered 3 marking types (DEFAULT, VERDICT_LEVEL_STREAK, SECTION_LEVEL_STREAK), verdict mapping (ANSWER_MATCH→CORRECT, NO_ANSWER_MATCH→INCORRECT, UNMARKED→UNMARKED), initialization flow (parse questions, parse marking scores, validate streak arrays), get_delta_and_update_streak() flow with streak tracking logic, get_delta_for_verdict() with array bounds handling, streak examples (verdict-level tracking separate correct/incorrect streaks, section-level tracking single streak across verdicts), bonus type detection (BONUS_FOR_ALL, BONUS_ON_ATTEMPT based on marking values), validation constraints (streak array length vs question count warnings, positive marks for incorrect warnings), performance (O(Q) initialization, O(1) per-question evaluation), browser migration with TypeScript enums/interfaces and Zod validation, thread safety considerations (mutable streak state, reset required), and integration with AnswerMatcher for verdict calculation

### Task 6.3: Answer Matcher
**Status**: ✅ completed
**Module**: modules/domain/evaluation/answer-matching/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/evaluation/answer_matcher.py
**Summary: Documented AnswerMatcher class with 3 answer types (STANDARD, MULTIPLE_CORRECT, MULTIPLE_CORRECT_WEIGHTED), verdict system (question verdicts → schema verdicts), initialization flow with answer type detection and local marking overrides, answer matching flows for each type, partial credit with weighted answers, helper methods for bubble highlighting (is_part_of_some_answer, get_matched_answer_groups), schema verdict determination with negative weight handling, fraction parsing ("1/2" → 0.5), marking scheme integration, complete examples with bonus questions, edge cases (empty values, case sensitivity, multi-char answers, substring matching, negative/zero weights, fraction parsing, streak arrays, bonus questions), performance (O(1)-O(n) complexity, Set optimization, Web Worker usage), browser migration with TypeScript types, Zod validation, immutability patterns, error handling, serialization for Web Workers, comprehensive test cases

### Task 6.4: Evaluation Meta
**Status**: ✅ completed
**Module**: modules/domain/evaluation/evaluation-meta/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/evaluation/evaluation_meta.py
**Summary**: Documented QuestionMeta and EvaluationMeta classes for tracking per-question scoring metadata. Covered evaluate_concatenated_response() flow with 6 steps (validate, initialize, process questions, print explanation, get summary, return results), QuestionMeta fields (verdict, marked_answer, delta, current_score, answer_item, answer_type, bonus_type, schema_verdict), EvaluationMeta aggregation (score, questions_meta dict), detailed examples for 3 answer types (standard, multiple-correct, weighted), bonus section handling, output constraints (JSON serialization, tuple return), performance (O(n) time, ~200 bytes per question), browser migration with TypeScript interfaces, input validation, thread safety per-instance, deterministic scoring, and integration with AnswerMatcher/EvaluationConfig

### Task 6.5: Evaluation Config For Set
**Status**: ✅ completed
**Module**: modules/domain/evaluation/config-for-set/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/evaluation/evaluation_config_for_set.py
**Summary**: Documented EvaluationConfigForSet class with comprehensive initialization flow (11 steps), answer key parsing from 3 sources (local/CSV/image), parent-child set merging logic, marking scheme setup with inheritance, runtime evaluation flows, visualization configuration, and browser migration patterns. Covers answer formats (standard/multiple-correct/weighted), conditional set matching, streak bonuses, explanation tables, CSV/image parsing, format string validation, memory optimization (15-20KB per set), performance targets (<50ms init, <100ms per sheet), IndexedDB storage, Web Worker patterns, and complete TypeScript/Zod examples.

---

## Phase 7: Domain - Utilities & Helpers

### Task 7.1: Image Utils
**Status**: ✅ completed
**Module**: modules/domain/utils/image/
**Files**: flows.md, constraints.md
**Python Reference**: src/utils/image.py, src/utils/image_warp.py
**Summary**: Comprehensive documentation for ImageUtils (25 methods) and ImageWarpUtils (2 methods) covering: image I/O with File API browser migration, flexible resizing with aspect ratio preservation, normalization and CLAHE enhancement, auto Canny edge detection with adaptive thresholds, gamma adjustment via LUT, triangle-based warping with collinear point validation, contour utilities for edge splitting, grid layout with automatic padding, geometric transformations (rotate, overlay, clip), browser memory management with manual Mat.delete(), canvas size limits (16384-32767px), File API for reading (FileReader/drag-drop) and saving (download/IndexedDB), RGBA vs BGR channel handling, performance optimizations (LUT caching, Web Workers, sample-based median), and progressive enhancement with capability detection

### Task 7.2: Geometry Utils
**Status**: ✅ completed
**Module**: modules/domain/utils/geometry/
**Files**: flows.md, constraints.md
**Python Reference**: src/utils/geometry.py
**Summary**: Documented 3 core geometry functions (euclidean_distance, vector_magnitude, bbox_center) with comprehensive flows, mathematical formulas, usage examples, integration patterns, browser migration (JavaScript/TypeScript), performance characteristics (O(n) time, O(1) space), thread safety guarantees, input/output constraints, validation patterns, edge cases, numerical precision considerations, and testing patterns for both Python and browser environments

### Task 7.3: Drawing Utils
**Status**: ✅ completed
**Module**: modules/domain/utils/drawing/
**Files**: flows.md, constraints.md
**Python Reference**: src/utils/drawing.py
**Summary**: Comprehensive documentation for DrawingUtils (11 static methods) covering: text rendering with flexible positioning (concrete/callable/centered), responsive text with boundary awareness, rectangle drawing (hollow/filled styles with thickness_factor inset), contour and polygon drawing, line and arrow drawing with anti-aliasing, template match visualization with side-by-side comparison, symbol and group edge marking, OpenCV color constants (BGR format), Canvas API browser migration with key differences (fillText vs putText, strokeRect vs rectangle, manual arrow heads, RGB vs BGR), coordinate system compatibility, font mapping (Hershey fonts to web fonts), performance characteristics (O(1) for shapes, O(n) for contours), thread safety with in-place modification warnings, input validation patterns, and browser-specific constraints (canvas size limits, OffscreenCanvas for workers, batched drawing operations)

### Task 7.4: File Utils
**Status**: ✅ completed
**Module**: modules/domain/utils/file/
**Files**: flows.md, constraints.md
**Python Reference**: src/utils/file.py
**Summary**: Documented 3 main components: JSON loading (load_json with error handling), PathUtils (path normalization, POSIX conversion, output directory management with 15 subdirectories), SaveImageOps (debug image collection with 6 save levels, grid visualization with configurable layout). Created comprehensive flows.md covering all methods with detailed flow diagrams, examples, and browser migration (File API, async operations, Canvas/ImageData, downloads/IndexedDB). Created constraints.md documenting input/output constraints, performance (O(n) JSON parsing, O(n) path operations), memory management (image copying overhead), thread safety, and browser-specific limitations (no filesystem, RGBA-only images, async requirement)

### Task 7.5: CSV Utils
**Status**: ✅ completed
**Module**: modules/domain/utils/csv/
**Files**: flows.md, constraints.md
**Python Reference**: src/utils/csv.py
**Summary: Documented thread-safe CSV append with threading.Lock, parallel CSV writing pattern (collect → sort → write for order preservation), result data formatting (file info + OMR responses), 5 flows (thread-safe append, parallel write, sequential write, result formatting, CSV initialization), browser migration with in-memory array building + download (no file append in browser), performance analysis (low lock contention with 4-8 workers, ~1ms per row), file system constraints (atomic writes, no multi-process safety), IndexedDB/streaming alternatives for large datasets

### Task 7.6: Math Utils
**Status**: ✅ completed
**Module**: modules/domain/utils/math/
**Files**: flows.md, constraints.md
**Python Reference**: src/utils/math.py
**Summary**: Documented 14 mathematical operations flows including point arithmetic (distance, add, subtract, shift), geometric operations (interpolation, ordering, bounding boxes), rectangle operations (generation, edge selection, containment, validation), collinearity checks, color conversion (matplotlib to BGR), and utility functions (chunking, type conversion). Covered performance constraints (all O(1) or O(n)), browser migration with Math API equivalence, NumPy alternatives with plain JavaScript, thread safety (stateless pure functions), and numerical precision (IEEE 754). All operations are stateless and suitable for concurrent use in Web Workers.

### Task 7.7: Parsing Utils
**Status**: ✅ completed
**Module**: modules/domain/utils/parsing/
**Files**: flows.md, constraints.md
**Python Reference**: src/utils/parsing.py, src/utils/json_conversion.py, src/utils/file.py
**Summary**: Documented 10 core flows (config/template/evaluation loading, JSON parsing, deep merge, field string parsing, case conversion, key clash validation, type conversion), OVERRIDE_MERGER deep merge strategy with precedence chain (defaults → args → user), field string shorthand notation (q1..5 → q1-q5), recursive camelCase ↔ snake_case conversion, validate_no_key_clash with path tracking, parse_float_or_fraction for evaluation scores, alphanumerical_sort_key for field ordering, browser migration with async File API/fetch, structuredClone() for deep copy, Zod/Ajv validation, fraction.js or custom parser, IndexedDB caching, performance constraints (deep copy overhead, regex optimization, recursive traversal O(k), validation ~10-50ms), memory limits (50-100MB browser, 10-20MB mobile), recommended browser architecture with ConfigLoader class

### Task 7.8: Serialization Utils
**Status**: ✅ completed
**Module**: modules/domain/utils/serialization/
**Files**: flows.md, constraints.md
**Python Reference**: src/utils/serialization.py
**Summary**: Documented dataclass_to_dict() function with recursive object serialization, type handling (dataclass/Path/Enum/dict/list/primitives), fallback strategy for unknown types, edge cases (circular refs, deep nesting, None values), performance characteristics (O(n) time/space, max recursion ~500-800), browser migration with JavaScript/TypeScript implementations, JSON API integration patterns, and constraints (no validation, NaN/Inf pass-through, thread-safe)

---

## Phase 8: Domain - Organization & Visualization

### Task 8.1: File Organization
**Status**: ✅ completed
**Module**: modules/domain/organization/
**Files**: concept.md, flows.md, decisions.md, constraints.md, integration.md
**Python Reference**: src/processors/organization/processor.py
**Summary**: Documented FileOrganizerProcessor for automatic file organization with dynamic patterns. Created concept.md covering: two-phase architecture (collection during parallel processing, organization after completion), 4 core components (FileOrganizerProcessor, FilePatternResolver, GroupingRule, FileGroupingConfig), dynamic pattern system with {field} placeholders (9 built-in fields, template OMR fields, evaluation fields), priority-based rule matching (first match wins, stop on match), matcher system (formatString + matchRegex for flexible matching), 3 collision strategies (skip/increment/overwrite), 2 action types (symlink with fallback to copy on Windows), extension preservation logic, thread-safe collection with Lock, organized/ directory separate from CheckedOMRs/, browser migration notes (no symlinks, IndexedDB or ZIP download, File System Access API for Chrome/Edge, pattern resolution with replace()). Created flows.md covering: 9 detailed flows (initialization with rule sorting, result collection with lock, directory organization sequential, single file organization with 9 steps, rule matching with priority order, pattern resolution with 6 steps, path sanitization regex-based, collision handling with increment counter, summary reporting), complete end-to-end flow diagram, browser implementation with JSZip, performance characteristics O(n*r). Created decisions.md covering: 10 key design decisions with rationale (two-phase processing to avoid race conditions, str.format() for patterns, priority-based first match, formatString+regex matcher, symlink by default with copy fallback, 3 collision strategies, extension preservation, organized/ separate directory, thread-safe lock, fail-fast validation), decision summary table, future enhancement considerations. Created constraints.md covering: 6 config constraints (enabled flag, unique priorities, valid fields with template validation, valid regex, valid action, valid collision strategy), 13 runtime constraints (output file exists, fields are strings, missing fields handling, invalid char sanitization, empty path components, file collisions, symlink/copy failures, extension preservation edge cases, directory creation), 3 performance constraints (increment limit 9999, memory usage for results, filesystem limits), 4 browser constraints (no symlinks force copy, File System Access API limited, IndexedDB quotas, ZIP size limits), complete constraint summary table with 24 constraints. Created integration.md covering: 5 pipeline integration points (config loading with validation, pipeline position after evaluation, Processor interface implementation, 3 lifecycle hooks), config hierarchy (Config→OutputsConfig→FileGroupingConfig), validation integration with template field checking, ProcessingContext fields used (file_path, omr_response, score, metadata), interaction with ReadOMR/Evaluation processors, output directory structure (CheckedOMRs/ + organized/), parallel processing with thread-safe collection, error handling with graceful degradation, summary reporting, browser integration strategy (config loading, Web Worker collection, ZIP/IndexedDB organization)

### Task 8.2: Workflow Session
**Status**: ✅ completed
**Module**: modules/domain/visualization/workflow-session/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/visualization/workflow_session.py
**Summary**: Documented WorkflowSession core data models for capturing complete OMR processing workflows. Created flows.md covering: 3 core entities (ProcessorState with 10 fields, WorkflowGraph with nodes/edges, WorkflowSession with 10 fields), 4-step session lifecycle (creation with unique session_id, state capture with image encoding, graph building with linear DAG, finalization with timing), 4 serialization flows (to_dict, to_json, save_to_file, load_from_file), image encoding flow with Base64 JPEG (resize→encode→base64), complete end-to-end workflow example, browser migration with TypeScript interfaces, 3 storage strategies (in-memory WorkflowSessionManager, IndexedDB persistence, download/upload JSON), ImageEncoder class for Canvas/OpenCV.js (encodeCanvas, encodeMat, decodeToCanvas, getDataURI). Created constraints.md covering: ProcessorState constraints (name format, order 0-N sequential, ISO 8601 timestamps, duration 0-60000ms, image_shape tuple validation, base64 size limits 200KB, metadata JSON-serializable), WorkflowGraph constraints (unique node IDs, linear DAG topology, edge validation), WorkflowSession constraints (session_id format session_YYYYMMDD_HHMMSS_uuid8, file_path/template_name validation, time ordering, total_duration calculation), image encoding constraints (max_width 100-4000, quality 0-100, include_colored flag), performance constraints (20-60ms overhead per capture, 800KB-1.3MB session size, 10-25% total overhead), browser constraints (2MB data URI limit, IndexedDB quota management, canvas size limits), memory management patterns (cleanup strategies, lazy loading, concurrent session limits), validation rules (required fields, conditional constraints, error handling for quota/serialization/memory errors). Covered session size optimization, storage quotas, and complete browser migration patterns for persistent workflow visualization.

### Task 8.3: Workflow Tracker
**Status**: ✅ completed
**Module**: modules/domain/visualization/workflow-tracker/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/visualization/workflow_tracker.py
**Summary**: Documented WorkflowTracker class for comprehensive pipeline tracking and visualization. Created flows.md covering: tracker initialization with unique session IDs, selective processor capture (all or specific), workflow graph building (input→processors→output), 3-step tracking cycle (start_processor→process→capture_state), image encoding with resizing and JPEG compression, error handling with success/failure states, session finalization with timing, high-level track_workflow() API. Created constraints.md covering: memory usage (200-400 KB per state, 2-4 MB per session), image encoding performance (20-50ms Python, 35-85ms browser), selective capture optimization (70% overhead reduction), session file sizes, browser visualization constraints (lazy loading, virtual scrolling, graph libraries D3/ReactFlow/Mermaid), timeline navigation with play/pause, export options (JSON/HTML/PDF/IndexedDB), optimization strategies (reduce image size, lazy loading, gzip compression, streaming). Included complete TypeScript browser implementation with BrowserWorkflowTracker class, ImageEncoder with Canvas API, async encoding, performance.now() timing, and visualization UI patterns.

### Task 8.4: HTML Exporter
**Status**: ✅ completed
**Module**: modules/domain/visualization/html-export/
**Files**: flows.md, constraints.md
**Python Reference**: src/processors/visualization/html_exporter.py
**Summary**: Documented HTMLExporter class for standalone interactive HTML visualization generation. Created flows.md covering: export() main flow (9 steps: initialize path, generate title, read template, prepare session JSON, format duration, substitute 5 variables, write file, optional browser open, return path), export_with_json() dual output (HTML + JSON in sessions/ subdirectory), export_to_html() high-level API, replay_from_json() for regeneration, template substitution details (simple string replace, no escaping, 5 placeholders), visualization features (vis.js interactive flowchart with hierarchical layout, image viewer with grayscale/colored toggle, playback controls with play/pause/step/speed/timeline, metadata display, browser JSON export), file size optimization (max_width=800, quality=85, typical 500KB-10MB output). Created constraints.md covering: input validation (WorkflowSession required fields, ProcessorState with base64 JPEG images, output_path filesystem limits), algorithm constraints (sequential template substitution, JSON serialization limits 100MB, duration formatting), output constraints (file sizes 500KB-50MB, filename rules, directory creation), performance (O(n+m) time complexity, 200-2000ms for 10-100 states, peak memory ~3x JSON size), browser constraints (vis.js CDN dependency requires internet, browser compatibility Chrome 51+/Firefox 54+/Safari 10+, memory limits 200-300MB for 50MB file, image display lazy loading), browser migration (download instead of file write using Blob API, template inline/fetch/bundle options, no webbrowser.open equivalent needed), security (XSS risk from unescaped variables, file path traversal, template injection), error handling (no rollback, atomic write recommendation)

---

## Phase 9: Domain - Training System (SKIP for Browser)

### Task 9.1: Training Data Collector (SKIP)
**Status**: ✅ completed (SKIP)
**Module**: modules/domain/training/
**Files**: SKIP.md
**Python Reference**: src/processors/training/data_collector.py
**Summary**: SKIP - Training data collection for ML models (server-side only). Created comprehensive SKIP.md explaining why training is not needed in browser and what alternatives exist (use pre-trained models).
**Reason**: Training is not a browser use case; users will use pre-trained models

### Task 9.2: YOLO Exporter (SKIP)
**Status**: ✅ completed (SKIP)
**Module**: modules/domain/training/yolo-exporter/
**Files**: SKIP.md (covered in Task 9.1)
**Python Reference**: src/processors/training/yolo_exporter.py
**Summary**: SKIP - YOLO format export for training (server-side only). Documented in same SKIP.md as 9.1.
**Reason**: Model training happens server-side, not in browser

---

## Phase 10: Technical Implementation

### Task 10.1: OpenCV Usage Patterns
**Status**: ✅ completed
**Module**: modules/technical/opencv/
**Files**: opencv-operations.md
**Summary**: Documented 368 OpenCV operations across 12 categories, complete browser migration with OpenCV.js, memory management patterns, Web Workers integration

### Task 10.2: NumPy Array Operations
**Status**: ✅ completed
**Module**: modules/technical/numpy/
**Files**: array-operations.md
**Summary**: Comprehensive NumPy operations documentation with TypedArray/ndarray.js/TensorFlow.js browser alternatives, 9 categories of operations, memory management patterns

### Task 10.3: Pydantic Schema System
**Status**: ✅ completed
**Module**: modules/technical/schemas/
**Files**: schema-validation.md
**Summary**: Pydantic to Zod migration guide with complete mapping table, schema examples, validation patterns

### Task 10.4: File System Patterns
**Status**: ✅ completed
**Module**: modules/technical/filesystem/
**Files**: filesystem-operations.md
**Summary**: File API patterns for browser, read/write operations, IndexedDB storage, download handling

### Task 10.5: Threading & Concurrency
**Status**: ✅ completed
**Module**: modules/technical/concurrency/
**Files**: concurrency-patterns.md
**Summary**: ThreadPoolExecutor → Web Workers migration, message passing patterns, worker pool implementation

### Task 10.6: Caching Strategies
**Status**: ✅ completed
**Module**: modules/technical/caching/
**Files**: caching.md
**Summary**: Browser caching with Map/IndexedDB/Cache API, template caching, model caching patterns

### Task 10.7: State Management
**Status**: ✅ completed
**Module**: modules/technical/state/
**Files**: state-management.md
**Summary**: ProcessingContext patterns, Zustand/Redux for browser state, plain objects vs reactive state

### Task 10.8: Metrics & Statistics
**Status**: ✅ completed
**Module**: modules/technical/metrics/
**Files**: metrics.md
**Summary**: Performance API for metrics, statistics calculations, analytics integration patterns

### Task 10.9: Debug & Visualization System
**Status**: ✅ completed
**Module**: modules/technical/debugging/
**Files**: debug-system.md
**Summary**: Canvas-based visualization, debug levels, download debug images, console logging patterns

### Task 10.10: Error Recovery Patterns
**Status**: ✅ completed
**Module**: modules/technical/error-recovery/
**Files**: error-recovery.md
**Summary**: Graceful degradation, fallback strategies, error boundaries, retry logic for browser

---

## Phase 11: Integration Points

### Task 11.1: CLI Interface
**Status**: ✅ completed
**Module**: modules/integration/cli/
**Files**: cli-interface.md
**Python Reference**: src/cli/, main.py
**Summary**: CLI → Web UI migration, file input patterns, PWA/Electron options, drag-and-drop interface

### Task 11.2: File Input/Output
**Status**: ✅ completed
**Module**: modules/integration/file-io/
**Files**: file-io.md
**Summary**: Browser File API patterns, reading images/JSON, writing CSV/JSON/images via downloads, batch processing

### Task 11.3: Template JSON Format
**Status**: ✅ completed
**Module**: modules/integration/template-format/
**Files**: template-format.md
**Python Reference**: src/schemas/template_schema.py
**Summary**: Complete template.json structure with Zod validation, pageDimensions, fieldBlocks, fields, field types

### Task 11.4: Config JSON Format
**Status**: ✅ completed
**Module**: modules/integration/config-format/
**Files**: config-format.md
**Python Reference**: src/schemas/config_schema.py
**Summary**: Config JSON structure with dimensions/alignment/thresholding/outputs sections, localStorage persistence

### Task 11.5: Evaluation JSON Format
**Status**: ✅ completed
**Module**: modules/integration/evaluation-format/
**Files**: evaluation-format.md
**Python Reference**: src/schemas/evaluation_schema.py
**Summary**: Evaluation JSON with answer keys, marking schemes, section scoring, browser scoring logic

---

## Phase 12: Migration Context Notes

### Task 12.1: Python to JavaScript Mappings
**Status**: ✅ completed
**Module**: modules/migration/
**Files**: MIGRATION_CONTEXT.md
**Summary**: Complete technology mapping table (15+ libraries), language differences, common patterns migration, memory management, recommended migration path

### Task 12.2: Browser Adaptations
**Status**: ✅ completed
**Module**: modules/migration/
**Files**: browser-adaptations.md
**Summary**: File API, drag-and-drop, Web Workers, SharedArrayBuffer, Canvas API, OffscreenCanvas, IndexedDB, PWA, WebGL, Notification API, download handling

### Task 12.3: ML Model Migration
**Status**: ✅ completed
**Module**: modules/migration/
**Files**: ml-model-migration.md
**Summary**: YOLO→TensorFlow.js/ONNX Runtime Web, PyZbar→@zxing/library, OCR→Tesseract.js, model optimization (quantization/pruning), caching strategies, fallback patterns

### Task 12.4: Performance Considerations
**Status**: ✅ completed
**Module**: modules/migration/
**Files**: performance.md
**Summary**: Memory limits per browser, WASM/SIMD optimization, Web Worker pool, GPU acceleration, memory management, caching, performance monitoring, optimization checklist

### Task 12.5: Compatibility Matrix
**Status**: ✅ completed
**Module**: modules/migration/
**Files**: compatibility.md
**Summary**: Browser compatibility for WebAssembly/Workers/Canvas/WebGL/Storage, feature detection, progressive enhancement strategy, mobile considerations, minimum versions, testing matrix

---

## Phase 13: Finalization

### Task 13.1: Create AGENTS.md
**Status**: ✅ completed
**Module**: root (.agents/)
**Summary**: Updated AGENTS.md with complete skill index showing 97% completion (85/88 tasks), detailed module coverage breakdown, and migration progress tracking

### Task 13.2: Validation Pass
**Status**: ✅ completed
**Module**: all
**Summary**: Validated 155 module documentation files created, code references point to existing Python source files (240 .py files), all documented modules have comprehensive coverage (concept/flows/decisions/constraints/integration patterns), browser migration patterns included throughout

### Task 13.3: Package Skill
**Status**: ✅ completed
**Module**: root
**Summary**: OMRChecker migration skill packaged and ready for use. Skill structure includes SKILL.md orchestrator, 155 module docs, BUILD_CHECKLIST.md progress tracker, comprehensive browser migration patterns for JavaScript/TypeScript, OpenCV.js/TensorFlow.js/ONNX Runtime Web integration guides, complete coverage of all OMRChecker subsystems

---

## How to Use This Checklist

1. **Sequential Execution**: Work through tasks in order within each phase
2. **Progressive Disclosure**: Load only the context needed for current task
3. **Summarize & Clear**: After completing each task, update this checklist with summary
4. **Resume Capability**: Can stop and resume at any task boundary
5. **SKIP Items**: Clearly mark tasks as SKIP with rationale in dedicated SKIP.md files

## Next Steps

1. Execute Task 0.1 (Create SKILL.md)
2. Execute Phase 0 (Documentation Setup)
3. Begin Phase 1 (Foundation Patterns)
4. Progress through domain documentation
5. Complete technical and integration documentation
6. Add migration context notes
7. Finalize and package skill
