# OMRChecker.js Implementation Checklist

Track your progress as you implement the TypeScript port manually.

**Reference**: `.agents/migration-toolkit/MANUAL_IMPLEMENTATION_GUIDE.md`

---

## Phase 1: Foundation ☐

- [ ] **Error Classes** (already done in `src/types/errors.ts`)
- [ ] **Logger** (`src/utils/logger/logger.ts`)
  - [ ] Logger class with 5 log levels
  - [ ] Dynamic log level control
  - [ ] Console API integration
- [ ] **Configuration** (`src/core/config.ts`)
  - [ ] Config class
  - [ ] Deep merge utility
  - [ ] Load from JSON
  - [ ] Hierarchical config (defaults → global → local)
- [ ] **Validation** (`src/utils/validation/`)
  - [ ] Zod schemas for Template
  - [ ] Zod schemas for Config
  - [ ] Zod schemas for Evaluation
  - [ ] Validation error handling
- [ ] **Test**: `npm run type-check` passes

---

## Phase 2: Core Entities ☐

- [ ] **Template** (`src/core/template.ts`)
  - [ ] Template class
  - [ ] Load from JSON
  - [ ] Validate structure
  - [ ] Parse field blocks
  - [ ] Pre-processor list
- [ ] **FieldBlock** (`src/core/field-block.ts`)
  - [ ] FieldBlock class
  - [ ] 3 detection types: BUBBLES_THRESHOLD, BARCODE, OCR
  - [ ] Field generation logic
- [ ] **Field** (`src/core/field.ts`)
  - [ ] Field class
  - [ ] ScanBox entity
  - [ ] Field processing methods
- [ ] **Config Models** (`src/core/config-models.ts`)
  - [ ] ThresholdingConfig
  - [ ] OutputsConfig
  - [ ] ProcessingConfig
  - [ ] MLConfig
- [ ] **Pipeline** (`src/processors/pipeline.ts`)
  - [ ] Pipeline coordinator
  - [ ] 6 stages: preprocessing → alignment → threshold → detection → evaluation → output
- [ ] **Test**: Load template.json, parse, validate

---

## Phase 3: Image Processing ☐

### 3.1 OpenCV.js Setup ☐
- [ ] **OpenCV Loader** (`src/utils/opencv-loader.ts`)
  - [ ] Async loading
  - [ ] WASM initialization
  - [ ] Global cv object check

### 3.2 Preprocessing ☐
- [ ] **Auto-Rotate** (`src/processors/preprocessing/auto-rotate.ts`)
  - [ ] 4 rotation detection methods
  - [ ] OpenCV.js rotation functions
  - [ ] Memory management (.delete() calls)
- [ ] **Crop Page** (`src/processors/preprocessing/crop-page.ts`)
  - [ ] Page boundary detection
  - [ ] Cropping logic
- [ ] **Crop on Markers** (`src/processors/preprocessing/crop-on-markers.ts`)
  - [ ] 4 marker types detection
  - [ ] Perspective transform
- [ ] **Filters** (`src/processors/preprocessing/filters.ts`)
  - [ ] Blur filter
  - [ ] Contrast filter
  - [ ] Levels filter
- [ ] **Warping** (`src/processors/preprocessing/warping.ts`)
  - [ ] 4 warp strategies
  - [ ] Homography calculation

### 3.3 Alignment ☐
- [ ] **ORB/SIFT** (`src/processors/alignment/orb.ts`)
  - [ ] Feature detection (use ORB, not SIFT)
  - [ ] Feature matching
  - [ ] Homography estimation
  - [ ] Perspective transform
- [ ] **Phase Correlation** (`src/processors/alignment/phase-correlation.ts`)
  - [ ] FFT-based alignment
  - [ ] Translation calculation
- [ ] **K-Nearest** (`src/processors/alignment/k-nearest.ts`)
  - [ ] Interpolation logic

### 3.4 Threshold ☐
- [ ] **Global Threshold** (`src/processors/threshold/global-threshold.ts`)
  - [ ] 7-step algorithm
  - [ ] Otsu's method
- [ ] **Local Threshold** (`src/processors/threshold/local-threshold.ts`)
  - [ ] 8-step adaptive threshold
  - [ ] Fallback to global
- [ ] **Strategy Pattern** (`src/processors/threshold/strategy.ts`)
  - [ ] Abstract strategy interface
  - [ ] Strategy selection

### 3.5 Detection ☐
- [ ] **Bubble Detection** (`src/processors/detection/bubbles-threshold.ts`)
  - [ ] Contour detection
  - [ ] Bubble filtering
  - [ ] Answer interpretation
- [ ] **Barcode** (`src/processors/detection/barcode.ts`)
  - [ ] Dummy placeholder (optional: implement with @zxing/library)
- [ ] **OCR** (`src/processors/detection/ocr.ts`)
  - [ ] Dummy placeholder (optional: implement with Tesseract.js)
- [ ] **Test**: Process sample OMR image end-to-end

---

## Phase 4: Evaluation & Results ☐

- [ ] **Evaluation Config** (`src/processors/evaluation/evaluation-config.ts`)
  - [ ] Load evaluation.json
  - [ ] Parse marking schemes
  - [ ] Question mappings
- [ ] **Marking Scheme** (`src/processors/evaluation/marking-scheme.ts`)
  - [ ] Scoring logic (correct/incorrect/unmarked)
  - [ ] Multi-marking handling
- [ ] **Grader** (`src/processors/evaluation/grader.ts`)
  - [ ] Calculate scores
  - [ ] Apply marking schemes
  - [ ] Generate results
- [ ] **CSV Export** (`src/utils/export.ts`)
  - [ ] Results to CSV conversion
  - [ ] Browser download
- [ ] **Test**: Full pipeline with evaluation

---

## Phase 5: File Handling & UI ☐

- [ ] **File Input** (`src/utils/file-handler.ts`)
  - [ ] Load image from File API
  - [ ] Convert to cv.Mat
  - [ ] Error handling
- [ ] **Canvas Debug** (`src/utils/debug-canvas.ts`)
  - [ ] Draw image on canvas
  - [ ] Draw bounding boxes
  - [ ] Draw detected bubbles
  - [ ] Show alignment markers
- [ ] **Progress UI** (optional)
  - [ ] Progress bar for batch processing
  - [ ] Status messages
- [ ] **Test**: Upload image, process, download results

---

## Memory Management Checklist ☐

For every function that creates Mat objects:
- [ ] Add `try/finally` blocks
- [ ] Call `.delete()` on all temporary Mat objects
- [ ] Clone return values if needed
- [ ] Test with multiple images to check for leaks

---

## Testing Checklist ☐

- [ ] **Type Checking**: `npm run type-check` passes
- [ ] **Unit Tests**: Implement test bodies in `tests/`
- [ ] **Integration Tests**: Full pipeline tests
- [ ] **Browser Testing**:
  - [ ] Chrome
  - [ ] Firefox
  - [ ] Safari
  - [ ] Edge
- [ ] **Performance Testing**:
  - [ ] Small images (< 1MB)
  - [ ] Medium images (1-5MB)
  - [ ] Large images (5-10MB)
  - [ ] Batch processing (10+ images)

---

## Final Checklist ☐

- [ ] All TypeScript compiles without errors
- [ ] All tests pass
- [ ] Memory leaks fixed
- [ ] Works in all major browsers
- [ ] Documentation updated
- [ ] Example templates included
- [ ] README with usage instructions
- [ ] Performance is acceptable (< 5s per image)

---

## Optional Enhancements ☐

- [ ] **Web Workers** for parallel processing
- [ ] **ML Models**:
  - [ ] YOLO for field detection (TensorFlow.js)
  - [ ] OCR (Tesseract.js)
  - [ ] Barcode (@zxing/library)
- [ ] **UI Framework**: React/Vue/Svelte interface
- [ ] **Progressive Web App** (PWA)
- [ ] **Offline Mode** with service workers
- [ ] **Real-time Preview** while uploading

---

## Progress Tracker

**Started**: _______________
**Target Completion**: _______________
**Actual Completion**: _______________

**Daily Log**:
- Day 1: _______________________________
- Day 2: _______________________________
- Day 3: _______________________________
- Day 4: _______________________________
- Day 5: _______________________________
- Day 6: _______________________________
- Day 7: _______________________________

---

**When all core items are checked, you have a fully functional browser-based OMR checker!**
