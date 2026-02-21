# Manual Implementation Guide

You have all scaffolding ready. Here's how to implement the TypeScript port manually using the migration documentation.

## Quick Reference

**Migration Docs**: `.agents/skills/omrchecker-migration-skill/`
**Python Source**: `src/`
**TypeScript Target**: `omrchecker-js/src/`
**OpenCV Patterns**: `.agents/migration-toolkit/opencv-js-patterns.md`

---

## Implementation Order (Recommended)

Follow this order to build incrementally and test as you go:

### Phase 1: Foundation (Day 1 - 4-6 hours)

These are utilities with minimal dependencies.

#### 1.1 Error Classes (30 min)
Already done! Check `omrchecker-js/src/types/errors.ts`

#### 1.2 Logger (1 hour)
**Python**: `src/utils/logger.py`
**TypeScript**: `omrchecker-js/src/utils/logger/logger.ts`
**Reference**: `.agents/skills/omrchecker-migration-skill/modules/foundation/logging.md`

```typescript
// Key changes:
// - Python logging → console API
// - Rich terminal → console.log with styles
// - Dynamic log levels preserved

export class Logger {
  private showLogsByType: Record<string, boolean> = {
    debug: false,
    info: true,
    warning: true,
    error: true,
    critical: true,
  };

  debug(...msg: any[]) {
    if (this.showLogsByType.debug) {
      console.debug(...msg);
    }
  }

  // ... other levels
}

export const logger = new Logger();
```

#### 1.3 Configuration (1 hour)
**Python**: `src/utils/parsing.py`, `src/schemas/defaults/config.py`
**TypeScript**: `omrchecker-js/src/core/config.ts`
**Reference**: `.agents/skills/omrchecker-migration-skill/modules/foundation/configuration.md`

```typescript
// Key changes:
// - Pydantic dataclass → TypeScript interface (already in src/types/core.ts)
// - deepmerge for hierarchical config
// - localStorage for browser persistence

import { merge } from 'lodash-es';

export class Config {
  static loadFromJSON(jsonPath: string): Config {
    // Fetch JSON, merge with defaults
    const defaults = CONFIG_DEFAULTS;
    const loaded = await fetch(jsonPath).then(r => r.json());
    const merged = merge({}, defaults, loaded);
    return new Config(merged);
  }
}
```

#### 1.4 Validation (1-2 hours)
**Python**: `src/utils/validations.py`
**TypeScript**: `omrchecker-js/src/utils/validation/`
**Reference**: `.agents/skills/omrchecker-migration-skill/modules/foundation/validation.md`

```typescript
// Key changes:
// - JSON Schema + jsonschema → Zod schemas
// - Rich table errors → console.table()

import { z } from 'zod';

export const TemplateSchema = z.object({
  templateDimensions: z.tuple([z.number(), z.number()]),
  bubbleDimensions: z.tuple([z.number(), z.number()]),
  fieldBlocks: z.record(z.object({ /* ... */ })),
  // ...
});

export function validateTemplate(json: unknown) {
  try {
    return TemplateSchema.parse(json);
  } catch (e) {
    if (e instanceof z.ZodError) {
      console.table(e.errors);
      throw new TemplateValidationError('Invalid template', e.errors);
    }
    throw e;
  }
}
```

**Test**: `npm run type-check` should pass for foundation files

---

### Phase 2: Core Entities (Day 2 - 6-8 hours)

#### 2.1 Template (2-3 hours)
**Python**: `src/core/template/template.py`
**TypeScript**: `omrchecker-js/src/core/template.ts`
**Reference**: `.agents/skills/omrchecker-migration-skill/modules/domain/template/concept.md`

Key class with template loading, validation, field block management.

```typescript
export class Template {
  path: string;
  fieldBlocks: Record<string, FieldBlock>;
  preProcessors: string[];
  tuningConfig: Config;

  constructor(jsonPath: string, config?: Config) {
    // Load JSON, validate, parse field blocks
  }

  static async fromJSON(jsonPath: string): Promise<Template> {
    const json = await fetch(jsonPath).then(r => r.json());
    validateTemplate(json);
    return new Template(json);
  }
}
```

#### 2.2 Field & FieldBlock (2 hours)
**Python**: `src/core/template/field.py`, `src/core/template/field_block.py`
**TypeScript**: `omrchecker-js/src/core/field.ts`, `omrchecker-js/src/core/field-block.ts`
**Reference**:
- `.agents/skills/omrchecker-migration-skill/modules/domain/field/concept.md`
- `.agents/skills/omrchecker-migration-skill/modules/domain/field-block/concept.md`

```typescript
export class FieldBlock {
  name: string;
  fieldDetectionType: 'BUBBLES_THRESHOLD' | 'BARCODE' | 'OCR';
  bubbleValues: string[];

  generateFields(): Field[] {
    // Generate Field instances based on detection type
  }
}

export class Field {
  fieldLabel: string;
  scanBox: ScanBox;
  fieldType: string;

  // Processing methods
}
```

#### 2.3 Config Classes (1-2 hours)
**Python**: `src/schemas/models/config.py`
**TypeScript**: `omrchecker-js/src/core/config-models.ts`
**Reference**: `.agents/skills/omrchecker-migration-skill/modules/domain/config/concept.md`

ThresholdingConfig, OutputsConfig, ProcessingConfig, MLConfig classes.

#### 2.4 Pipeline Coordinator (1 hour)
**Python**: `src/processors/pipeline.py`
**TypeScript**: `omrchecker-js/src/processors/pipeline.ts`
**Reference**: `.agents/skills/omrchecker-migration-skill/modules/domain/pipeline/concept.md`

**Test**: Load a template JSON, parse it, validate structure

---

### Phase 3: Image Processing (Day 3-5 - 16-24 hours)

This is the bulk of the work. OpenCV.js operations are nearly identical to Python!

#### 3.1 OpenCV.js Setup (1 hour)
**TypeScript**: `omrchecker-js/src/utils/opencv-loader.ts`

```typescript
export async function loadOpenCV(): Promise<void> {
  return new Promise((resolve) => {
    if (window.cv && cv.Mat) {
      resolve();
      return;
    }

    window.Module = {
      onRuntimeInitialized: resolve
    };

    const script = document.createElement('script');
    script.src = '/opencv.js'; // From node_modules or CDN
    document.head.appendChild(script);
  });
}
```

#### 3.2 Preprocessing (4-6 hours)
**Python**: `src/processors/image/`
**TypeScript**: `omrchecker-js/src/processors/preprocessing/`
**Reference**: `.agents/skills/omrchecker-migration-skill/modules/domain/preprocessing/`

**Files to implement** (in order):
1. `auto-rotate.ts` - Rotation detection (4 methods)
2. `crop-page.ts` - Page detection
3. `crop-on-markers.ts` - Marker-based cropping (4 marker types)
4. `filters.ts` - Blur, Contrast, Levels filters
5. `warping.ts` - 4 warp strategies

**OpenCV Pattern Example**:
```typescript
// Python
def autoRotate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # ... detect rotation
    rotated = cv2.rotate(image, angle)
    return rotated

// TypeScript (nearly identical!)
function autoRotate(image: cv.Mat): cv.Mat {
  const gray = new cv.Mat();
  const edges = new cv.Mat();

  try {
    cv.cvtColor(image, gray, cv.COLOR_BGR2GRAY);
    cv.Canny(gray, edges, 50, 150);
    // ... detect rotation (same logic!)
    const rotated = new cv.Mat();
    cv.rotate(image, rotated, angle);
    return rotated.clone();
  } finally {
    gray.delete();
    edges.delete();
  }
}
```

#### 3.3 Alignment (4-6 hours)
**Python**: `src/processors/alignment/`
**TypeScript**: `omrchecker-js/src/processors/alignment/`
**Reference**: `.agents/skills/omrchecker-migration-skill/modules/domain/alignment/`

**Files**:
1. `sift.ts` - SIFT feature matching (use ORB in browser - SIFT is patented)
2. `phase-correlation.ts` - FFT-based alignment
3. `k-nearest.ts` - K-Nearest interpolation

**Note**: OpenCV.js doesn't have SIFT (patented). Use ORB instead:
```typescript
// Python SIFT
sift = cv2.SIFT_create()

// Browser ORB (free alternative)
const orb = new cv.ORB();
```

#### 3.4 Threshold Strategies (3-4 hours)
**Python**: `src/processors/threshold/`
**TypeScript**: `omrchecker-js/src/processors/threshold/`
**Reference**: `.agents/skills/omrchecker-migration-skill/modules/domain/threshold/`

**Files**:
1. `global-threshold.ts` - Global thresholding (7 steps)
2. `local-threshold.ts` - Local adaptive (8 steps)
3. `strategy-pattern.ts` - Strategy abstraction

#### 3.5 Detection (4-6 hours)
**Python**: `src/processors/detection/`
**TypeScript**: `omrchecker-js/src/processors/detection/`
**Reference**: `.agents/skills/omrchecker-migration-skill/modules/domain/detection/` (to be found in your docs)

**Files**:
1. `bubbles-threshold.ts` - Bubble detection via thresholding
2. `barcode-detection.ts` - Barcode (dummy placeholder for now)
3. `ocr-detection.ts` - OCR (dummy placeholder for now)

**ML Model Placeholders**:
```typescript
// Barcode detection (skip for now)
export function detectBarcode(image: cv.Mat): string | null {
  // TODO: Implement with @zxing/library (optional)
  logger.warning('Barcode detection not implemented - returning null');
  return null;
}

// OCR (skip for now)
export function extractText(image: cv.Mat): string {
  // TODO: Implement with Tesseract.js (optional)
  logger.warning('OCR not implemented - returning empty string');
  return '';
}
```

**Test**: Process a sample OMR image through preprocessing → alignment → threshold → detection

---

### Phase 4: Evaluation & Results (Day 6 - 4-6 hours)

#### 4.1 Evaluation (3-4 hours)
**Python**: `src/processors/evaluation/`
**TypeScript**: `omrchecker-js/src/processors/evaluation/`
**Reference**: `.agents/skills/omrchecker-migration-skill/modules/domain/evaluation/` (in your docs)

**Files**:
1. `evaluation-config.ts` - Load and parse evaluation.json
2. `marking-scheme.ts` - Scoring logic
3. `grader.ts` - Calculate scores

#### 4.2 Results Export (1-2 hours)
**Python**: `src/utils/file.py` (CSV export)
**TypeScript**: `omrchecker-js/src/utils/export.ts`

```typescript
// Browser: Download as CSV
export function downloadResults(results: Result[], filename: string) {
  const csv = resultsToCSV(results);
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();

  URL.revokeObjectURL(url);
}
```

**Test**: Full pipeline - image → detection → evaluation → results

---

### Phase 5: File Handling & UI (Optional - Day 7)

#### 5.1 File Input (2-3 hours)
**TypeScript**: `omrchecker-js/src/utils/file-handler.ts`

```typescript
export async function loadImageFromFile(file: File): Promise<cv.Mat> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const mat = cv.imread(img);
      resolve(mat);
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}
```

#### 5.2 Canvas Debug UI (2-4 hours)
**Python**: Rich terminal visualization
**TypeScript**: `omrchecker-js/src/utils/debug-canvas.ts`

```typescript
export function drawDebugOutput(
  canvas: HTMLCanvasElement,
  image: cv.Mat,
  detections: Detection[]
) {
  cv.imshow(canvas, image);
  const ctx = canvas.getContext('2d')!;

  // Draw bounding boxes
  detections.forEach(det => {
    ctx.strokeStyle = 'red';
    ctx.strokeRect(det.x, det.y, det.width, det.height);
  });
}
```

---

## Key Migration Patterns

### 1. Memory Management (Critical!)

Always use try/finally for Mat objects:

```typescript
function processImage(img: cv.Mat): cv.Mat {
  const gray = new cv.Mat();
  const result = new cv.Mat();

  try {
    cv.cvtColor(img, gray, cv.COLOR_BGR2GRAY);
    // ... processing ...
    return result.clone();
  } finally {
    gray.delete();
    // Don't delete result - caller owns it
  }
}
```

### 2. OpenCV Operations

98% identical! Just:
- `cv2.` → `cv.`
- Return values → output parameters
- Add `.delete()` calls

```python
# Python
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# JavaScript
const thresh = new cv.Mat();
cv.threshold(gray, thresh, 127, 255, cv.THRESH_BINARY);
// Note: threshold value available but usually ignored
```

### 3. NumPy Arrays → TypedArrays

```python
# Python
arr = np.array([1, 2, 3], dtype=np.uint8)

# JavaScript
const arr = new Uint8Array([1, 2, 3]);
```

### 4. Async Operations

```typescript
// Load template async
const template = await Template.fromJSON('/templates/template.json');

// Process image async (if using Web Workers)
const result = await processImageAsync(image, template);
```

---

## Testing Strategy

### Test Incrementally

After each phase, verify:

```typescript
// Phase 1 test
import { logger } from './utils/logger/logger';
logger.info('Logger works!');

// Phase 2 test
const template = await Template.fromJSON('/template.json');
console.log('Template loaded:', template.fieldBlocks);

// Phase 3 test
const img = await loadImageFromFile(file);
const preprocessed = preprocessImage(img, template);
cv.imshow('canvas', preprocessed);

// Phase 4 test
const results = evaluateResponse(detectedAnswers, evaluationConfig);
console.log('Score:', results.score);
```

### Run Type Checker

```bash
cd omrchecker-js
npm run type-check
```

### Run Tests

```bash
npm test
```

---

## Common Issues & Solutions

### Issue: Memory Leaks

**Symptom**: Browser slows down after processing multiple images
**Solution**: Add `.delete()` calls for all Mat objects

```typescript
// Before (leaks memory)
function process(img: cv.Mat) {
  const gray = new cv.Mat();
  cv.cvtColor(img, gray, cv.COLOR_BGR2GRAY);
  return gray; // Caller must delete!
}

// After (safe)
function process(img: cv.Mat) {
  const gray = new cv.Mat();
  try {
    cv.cvtColor(img, gray, cv.COLOR_BGR2GRAY);
    return gray.clone();
  } finally {
    gray.delete();
  }
}
```

### Issue: OpenCV.js Not Loaded

**Symptom**: `cv is not defined`
**Solution**: Call `loadOpenCV()` before any cv operations

```typescript
// In main app initialization
await loadOpenCV();
console.log('OpenCV.js ready!');
```

### Issue: CORS Errors Loading Templates

**Symptom**: Can't fetch template.json
**Solution**: Run a local dev server

```bash
npm run dev  # Vite dev server handles CORS
```

### Issue: Large Images Crash Browser

**Symptom**: Out of memory on 10MB+ images
**Solution**: Downscale before processing

```typescript
function downscaleIfNeeded(img: cv.Mat, maxDimension = 2000): cv.Mat {
  const maxDim = Math.max(img.rows, img.cols);
  if (maxDim > maxDimension) {
    const scale = maxDimension / maxDim;
    const resized = new cv.Mat();
    cv.resize(img, resized, new cv.Size(0, 0), scale, scale);
    return resized;
  }
  return img.clone();
}
```

---

## Resources

### Migration Documentation
- **Core**: `.agents/skills/omrchecker-migration-skill/core/`
- **Foundation**: `.agents/skills/omrchecker-migration-skill/modules/foundation/`
- **Domain**: `.agents/skills/omrchecker-migration-skill/modules/domain/`
- **Technical**: `.agents/skills/omrchecker-migration-skill/modules/technical/`
- **Migration Context**: `.agents/skills/omrchecker-migration-skill/modules/migration/`

### OpenCV.js
- **Patterns**: `.agents/migration-toolkit/opencv-js-patterns.md`
- **Official Docs**: https://docs.opencv.org/4.x/d5/d10/tutorial_js_root.html
- **API Reference**: https://docs.opencv.org/4.x/d0/d84/group__core__basic.html

### Python Source
- **Code**: `src/`
- **Tests**: `src/tests/`

---

## Estimated Timeline

| Phase | Hours | Days |
|-------|-------|------|
| Foundation | 4-6 | 1 |
| Core Entities | 6-8 | 1 |
| Image Processing | 16-24 | 2-3 |
| Evaluation | 4-6 | 1 |
| File Handling & UI | 4-6 | 1 |
| **Total** | **34-50 hrs** | **6-7 days** |

**Note**: This assumes ~6-8 hours/day. If full-time (8-10 hrs/day), can finish in ~5 days.

---

## When You're Done

You'll have:
- ✅ Fully functional browser-based OMR checker
- ✅ Template loading and validation
- ✅ Complete image processing pipeline
- ✅ Bubble detection
- ✅ Evaluation and grading
- ✅ Results export (CSV download)
- ✅ Debug visualization (canvas)
- ⏭️ ML models (optional - can add later)
- ⏭️ Web Workers (optional - can add later)

**It's a complete port, not a partial implementation!**

Good luck! Refer to the migration documentation as you implement each module.
