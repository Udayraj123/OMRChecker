# Migration Context: OMRChecker Python → JavaScript

## Migration Discovery

**Created:** 2026-02-20

### Purpose
Document the complete Python OMRChecker codebase for zero-edge-case-loss migration to JavaScript (browser/client-side).

### Migration Goal
- **Primary Driver**: Browser/Web compatibility
- **Target Runtime**: Browser (client-side JavaScript)
  - Canvas API for image processing
  - WebAssembly (OpenCV.js) for computer vision operations
  - No server-side Node.js dependencies
- **Feature Parity**: Must maintain 100% parity with Python version
- **Legacy Code Handling**: Document deprecated features as SKIP items

### Legacy Codebase Overview

**Language**: Python 3.11+
**Total Lines**: ~39,000 LOC (application code only)
**Primary Dependencies**:
- OpenCV (cv2) - image processing and computer vision
- NumPy - numerical operations
- Rich - terminal UI
- Pydantic - schema validation
- PyZbar - barcode detection
- EasyOCR/Tesseract - OCR capabilities

**Repository Structure**:
```
src/
├── cli/                    # Command-line interface
├── constants/              # Application constants
├── processors/             # Core processing modules
│   ├── alignment/          # Image alignment (SIFT, phase correlation)
│   ├── detection/          # Bubble, barcode, OCR detection
│   ├── evaluation/         # Scoring and answer matching
│   ├── image/              # Image preprocessing
│   ├── organization/       # File organization
│   ├── template/           # Template management
│   ├── threshold/          # Thresholding strategies
│   ├── training/           # ML model training
│   └── visualization/      # Workflow tracking
├── schemas/                # Pydantic schemas for validation
├── utils/                  # Utility modules
└── tests/                  # Test suite

main.py                     # Entry point
```

### Key Architecture Patterns

1. **Pipeline Architecture**: ProcessingPipeline chains processors
   - PreprocessingCoordinator → AlignmentProcessor → ReadOMRProcessor
   - Each processor follows unified Processor interface
   - ProcessingContext passed between processors

2. **Template-Driven**: JSON template defines OMR layout
   - Field blocks contain fields (bubbles, barcodes, OCR)
   - Pre-processors for image transformation
   - Evaluation config for scoring

3. **Multi-Format Detection**:
   - Bubble detection (threshold-based + ML)
   - Barcode detection (PyZbar)
   - OCR detection (EasyOCR/Tesseract)
   - ML-based field block detection (YOLO)

4. **Parallel Processing**: ThreadPoolExecutor for batch processing
   - Thread-safe CSV writes
   - Configurable worker count

5. **Rich Visualization**:
   - Step-by-step debug images
   - Colored output support
   - Interactive template layout mode

### Browser Migration Challenges

**Python → JavaScript Considerations**:
1. **OpenCV → OpenCV.js**:
   - Most CV operations available in WebAssembly version
   - May need custom implementations for advanced features

2. **File System Access**:
   - Browser File API instead of pathlib
   - LocalStorage/IndexedDB for caching
   - File input elements for upload

3. **Multi-threading**:
   - Web Workers for parallel processing
   - SharedArrayBuffer for image data

4. **ML Models**:
   - YOLO → TensorFlow.js or ONNX Runtime Web
   - PyZbar → Third-party WebAssembly barcode libs
   - OCR → Tesseract.js

5. **Schema Validation**:
   - Pydantic → Zod or JSON Schema validation

### What to Document

**MUST DOCUMENT**:
- All bubble detection algorithms and thresholding strategies
- Image alignment techniques (SIFT, phase correlation, affine)
- Template layout parsing and field block detection
- Evaluation logic and scoring algorithms
- Edge case handling (rotated images, low quality, xerox)
- Configuration schemas and defaults
- All flows from image input to CSV output

**SKIP (Document as SKIP)**:
- CLI-specific features (argparse, terminal UI)
- File system crawling logic (replaced by browser file input)
- Python-specific patterns (decorators, metaclasses)
- Training/ML model generation (out of scope for browser)
- Parallel threading (replaced by Web Workers)

### Success Criteria

Migration documentation is complete when:
1. All bubble detection flows documented (100+ edge cases)
2. All image preprocessing steps captured
3. All alignment strategies explained
4. Template schema fully documented
5. Evaluation logic completely mapped
6. Browser adaptation notes provided for all features
7. Zero ambiguity in algorithm implementation
8. SKIP items clearly marked with rationale
