# OMRChecker Boundaries

**Document Type**: System Boundaries
**Last Updated**: 2026-02-20

---

## What OMRChecker DOES

### Core Capabilities

1. **OMR Sheet Processing**
   - Detects and interprets marked bubbles on standardized forms
   - Handles multiple bubble types (circular, oval, rectangular)
   - Supports scanned documents (640x480+ resolution)
   - Supports mobile phone images (various angles, lighting conditions)

2. **Multi-Format Detection**
   - **Bubble Detection**: Classical threshold-based + ML-based
   - **Barcode Detection**: 1D and 2D barcodes (QR, Code128, etc.)
   - **OCR Detection**: Text recognition for labels and IDs

3. **Image Preprocessing**
   - Automatic rotation correction
   - Perspective correction (4-point warping)
   - Marker-based cropping (dots, lines, custom markers)
   - Page detection and alignment
   - Image filtering (blur, contrast, levels)

4. **Template-Driven Layout**
   - JSON-based template definition
   - Field blocks containing multiple fields
   - Configurable bubble areas, dimensions
   - Pre-processor pipeline configuration
   - Exclude files from processing

5. **Evaluation & Scoring**
   - Answer key-based evaluation
   - Multiple marking schemes (per section, weighted)
   - Partial credit support
   - Multiple correct answers per question
   - Scoring explanations and breakdowns

6. **Batch Processing**
   - Parallel processing (ThreadPoolExecutor)
   - Directory traversal (recursive)
   - Multi-image processing
   - Configurable worker threads

7. **Output Formats**
   - CSV results (responses + scores)
   - Visual debug images (step-by-step)
   - Colored output overlays
   - Error/multi-marked file tracking
   - HTML reports (workflow visualization)

8. **Alignment Strategies**
   - SIFT feature-based alignment
   - Phase correlation (translation detection)
   - Template matching
   - Piecewise affine (Delaunay triangulation)

9. **Edge Case Handling**
   - Rotated images (any angle)
   - Low-quality xeroxed sheets
   - Mobile images (perspective distortion)
   - Poor lighting conditions
   - Multi-marked bubbles
   - Skewed/warped sheets

---

## What OMRChecker DOES NOT DO

### Out of Scope

1. **Document Types**
   - ❌ Free-form handwriting recognition (not OMR)
   - ❌ Essay/paragraph grading (not bubble-based)
   - ❌ Signature verification
   - ❌ Non-OMR document processing

2. **Recognition Capabilities**
   - ❌ General object detection (only bubbles, barcodes, OCR fields)
   - ❌ Face recognition
   - ❌ Fingerprint scanning
   - ❌ Complex diagram interpretation

3. **Image Sources**
   - ❌ Video stream processing (only static images)
   - ❌ Real-time camera feed processing
   - ❌ PDF parsing (requires pre-conversion to images)

4. **Storage & Persistence**
   - ❌ Database integration (only CSV output)
   - ❌ Cloud storage integration
   - ❌ Long-term result storage
   - ❌ User authentication/authorization

5. **Advanced Features**
   - ❌ Web interface (CLI only in Python version)
   - ❌ REST API (standalone application)
   - ❌ Real-time collaboration
   - ❌ Online template editor

6. **ML Model Management**
   - ❌ Automatic model training (requires manual training setup)
   - ❌ Model versioning
   - ❌ A/B testing of models
   - ❌ Auto-retraining pipelines

---

## Browser Version Boundaries

### Additional Scope for JavaScript Version

**INCLUDES**:
- Browser-based image upload (File API)
- Client-side image processing (OpenCV.js, Canvas API)
- Local storage for templates and results
- Web Workers for parallel processing
- Offline-first capability
- Interactive UI for template setup

**EXCLUDES (Same as Python)**:
- Server-side processing (fully client-side)
- Database persistence
- Multi-user collaboration
- Cloud synchronization

### Platform-Specific Boundaries

**Browser Limitations**:
- File system access limited to user-selected files
- Memory constraints (heap size limits)
- Processing speed (interpreted JS vs native Python/C++)
- WebAssembly module size limits

**Python-Only Features (SKIP for Browser)**:
- CLI argument parsing (replaced by UI controls)
- Directory crawling (replaced by file selection dialog)
- ML model training (use pre-trained models only)
- System-level file operations (moving files to error/multi-marked dirs)

---

## Traffic Coverage

### Typical Usage Patterns

1. **Exam Grading (Primary Use Case - 70%)**
   - Educational institutions
   - Multiple choice exams
   - 50-500 sheets per batch
   - High accuracy requirements

2. **Survey Processing (20%)**
   - Questionnaires
   - Feedback forms
   - Opinion polls
   - 10-1000 sheets per batch

3. **Attendance Tracking (5%)**
   - Attendance sheets
   - Simple present/absent marking
   - 20-100 sheets per batch

4. **Custom Applications (5%)**
   - Voting forms
   - Registration sheets
   - Data collection forms
   - Variable batch sizes

### Input Quality Distribution

- **Scanned documents**: 60% (high quality, 300+ DPI)
- **Mobile photos**: 35% (variable quality, 720p-4K)
- **Xeroxed sheets**: 5% (degraded quality, noise)

### Template Complexity

- **Simple templates** (1-50 bubbles): 40%
- **Medium templates** (51-200 bubbles): 45%
- **Complex templates** (201-500 bubbles): 13%
- **Very complex templates** (500+ bubbles): 2%

---

## System Constraints

### Hard Constraints

1. **Image Resolution**: Minimum 640x480 pixels
2. **Image Format**: PNG, JPG, JPEG only
3. **Template Format**: JSON only (schema-validated)
4. **Bubble Shape**: Must be detectable contours (circles, ovals, rectangles)
5. **Marker Types**: Dots, lines, or custom markers (for CropOnMarkers)

### Soft Constraints

1. **Processing Speed**: ~200 OMRs/minute (varies by complexity, hardware)
2. **Parallel Workers**: 1-8 threads (configurable)
3. **Image Size**: Recommended < 10MB per image
4. **Batch Size**: Recommended < 1000 images per run

---

## Known Limitations

### Python Version

1. **Memory Usage**: High memory for large batches (images loaded into RAM)
2. **Error Recovery**: Basic error handling (fails on critical errors)
3. **Template Validation**: Runtime validation only (no IDE autocomplete)
4. **Debugging**: Visual debug outputs only (no interactive debugger)

### Browser Version (Expected)

1. **Browser Heap Limits**: ~2-4GB depending on browser
2. **Single-Threaded Main**: UI can freeze during heavy processing
3. **File Access**: Limited to user-selected files only
4. **Model Size**: Large ML models may exceed practical download sizes

---

## Future Scope (NOT in Current Version)

Features that may be added but are NOT currently supported:

- Real-time video processing
- Adaptive learning (model improves over time)
- Multi-page PDF support (native, without pre-conversion)
- Cloud-based template marketplace
- Collaborative template editing
- Advanced analytics dashboard
- Integration with Learning Management Systems (LMS)
- Mobile native apps (React Native, Flutter)

---

**Note**: This boundaries document describes the current state of the Python OMRChecker and expected boundaries for the JavaScript browser version. Migration should preserve all DOES capabilities while explicitly marking DOES NOT DO items as out of scope.
