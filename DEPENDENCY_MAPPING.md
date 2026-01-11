# OMRChecker Python to JavaScript Dependency Mapping

## Overview

This document provides a comprehensive mapping of Python dependencies used in OMRChecker to their JavaScript/TypeScript equivalents for the web port.

## Core Dependencies Mapping

### Image Processing & Computer Vision

| Python Package | Version | JavaScript Equivalent | NPM Package | Status | Notes |
|----------------|---------|----------------------|-------------|--------|-------|
| `opencv-python` | 4.12.0.88 | opencv.js | `@techstark/opencv-js` or CDN | ✅ Available | Official WASM build, full OpenCV 4.x API |
| `opencv-contrib-python` | 4.12.0.88 | opencv.js (with contrib) | Custom build needed | ⚠️ Partial | Need custom build with xfeatures2d for SIFT |
| `numpy` | 2.2.6 | TypedArrays + cv.Mat | Built-in + opencv.js | ✅ Available | Use Uint8Array, Float32Array, cv.Mat objects |
| `scipy` | 1.15.1 | Math.js + custom | `mathjs` | ⚠️ Partial | Most needed functions available in opencv.js |

### Data Processing & Utilities

| Python Package | Version | JavaScript Equivalent | NPM Package | Status | Notes |
|----------------|---------|----------------------|-------------|--------|-------|
| `pandas` | 2.3.1 | Custom CSV builder | Native JS | ✅ Available | Simple CSV generation, no need for full dataframe |
| `jsonschema` | 4.25.0 | AJV | `ajv@8.12.0` | ✅ Available | Faster and more popular in JS ecosystem |
| `dotmap` | 1.3.30 | Plain objects | TypeScript interfaces | ✅ Available | Use native objects with type safety |
| `deepmerge` | 2.0 | deepmerge | `deepmerge@4.3.1` | ✅ Available | Direct equivalent |

### Barcode & OCR

| Python Package | Version | JavaScript Equivalent | NPM Package | Status | Notes |
|----------------|---------|----------------------|-------------|--------|-------|
| `pyzbar` | 0.1.9 | ZXing | `@zxing/library@0.20.0` | ✅ Available | TypeScript-native barcode/QR decoder |
| `easyocr` | >=1.7.2 | Tesseract.js | `tesseract.js@5.0.4` | 🔄 Future | OCR support planned for future |

### Geometry & Math

| Python Package | Version | JavaScript Equivalent | NPM Package | Status | Notes |
|----------------|---------|----------------------|-------------|--------|-------|
| `shapely` | 2.1.1 | JSTS | `jsts@2.11.0` | ✅ Available | JTS Topology Suite port |
| - | - | Turf.js | `@turf/turf@6.5.0` | ✅ Alternative | Better for GIS operations |
| `matplotlib` | 3.10.5 | Chart.js | `chart.js@4.4.1` | ⚠️ Optional | Only for visualization, not core |

### Machine Learning (Future Support)

| Python Package | Version | JavaScript Equivalent | NPM Package | Status | Notes |
|----------------|---------|----------------------|-------------|--------|-------|
| `ultralytics` | >=8.0.0 | ONNX Runtime | `onnxruntime-web@1.17.0` | 🔄 Future | Export YOLO to ONNX format |
| - | - | TensorFlow.js | `@tensorflow/tfjs@4.17.0` | 🔄 Future | Alternative for ML models |
| `torch` | >=2.0.0 | ONNX Runtime | `onnxruntime-web@1.17.0` | 🔄 Future | Export PyTorch models to ONNX |
| `torchvision` | >=0.15.0 | Custom transforms | opencv.js + custom | 🔄 Future | Implement transforms in JS |

### UI & Interaction (Not Needed for Web)

| Python Package | Version | JavaScript Equivalent | NPM Package | Status | Notes |
|----------------|---------|----------------------|-------------|--------|-------|
| `rich` | 13.8.0 | Console styling | `chalk@5.3.0` (Node) | ❌ N/A | Use browser console with CSS |
| `screeninfo` | 0.8.1 | window.screen API | Built-in | ❌ N/A | Browser native API |

## OpenCV Function Mapping

### Image I/O

| Python (cv2) | JavaScript (cv) | Notes |
|--------------|-----------------|-------|
| `cv2.imread(path, mode)` | `cv.imread(imageElement)` | Load from `<img>` or Canvas |
| `cv2.imwrite(path, img)` | `cv.imshow(canvas, mat)` | Write to Canvas, then save |
| `cv2.cvtColor(img, CODE)` | `cv.cvtColor(src, dst, code)` | Same API |

### Image Processing

| Python (cv2) | JavaScript (cv) | Notes |
|--------------|-----------------|-------|
| `cv2.GaussianBlur(img, ksize, sigma)` | `cv.GaussianBlur(src, dst, ksize, sigma)` | Direct equivalent |
| `cv2.threshold(img, thresh, maxval, type)` | `cv.threshold(src, dst, thresh, maxval, type)` | Direct equivalent |
| `cv2.Canny(img, t1, t2)` | `cv.Canny(src, dst, t1, t2)` | Direct equivalent |
| `cv2.findContours()` | `cv.findContours()` | Direct equivalent |
| `cv2.approxPolyDP()` | `cv.approxPolyDP()` | Direct equivalent |
| `cv2.createCLAHE()` | `cv.createCLAHE()` | Direct equivalent |
| `cv2.warpPerspective()` | `cv.warpPerspective()` | Direct equivalent |
| `cv2.resize()` | `cv.resize()` | Direct equivalent |
| `cv2.mean()` | `cv.mean()` | Direct equivalent |

### Feature Detection

| Python (cv2) | JavaScript (cv) | Notes |
|--------------|-----------------|-------|
| `cv2.SIFT_create()` | Needs custom build | Not in default opencv.js build |
| `cv2.BFMatcher()` | `cv.BFMatcher()` | Available in opencv.js |
| `cv2.FlannBasedMatcher()` | May not be available | Check opencv.js build |

### Drawing

| Python (cv2) | JavaScript (cv) | Notes |
|--------------|-----------------|-------|
| `cv2.rectangle()` | `cv.rectangle()` | Direct equivalent |
| `cv2.circle()` | `cv.circle()` | Direct equivalent |
| `cv2.line()` | `cv.line()` | Direct equivalent |
| `cv2.putText()` | `cv.putText()` | Direct equivalent |

## NumPy to TypedArray Mapping

| NumPy Type | TypedArray | Notes |
|------------|------------|-------|
| `np.uint8` | `Uint8Array` | Most common for images |
| `np.float32` | `Float32Array` | For normalized values |
| `np.float64` | `Float64Array` | High precision |
| `np.int32` | `Int32Array` | Integer operations |
| `np.ndarray` | `cv.Mat` | OpenCV's matrix type |

### Common Operations

| NumPy | JavaScript | Notes |
|-------|------------|-------|
| `arr.shape` | `mat.rows, mat.cols, mat.channels()` | Get dimensions |
| `arr.dtype` | `mat.type()` | Get data type |
| `arr[y, x]` | `mat.data[y * mat.cols + x]` | Indexing |
| `np.mean(arr)` | `cv.mean(mat)` | Mean value |
| `np.std(arr)` | Calculate manually | Standard deviation |
| `np.where()` | Custom implementation | Conditional selection |

## Build Tools & Dev Dependencies

| Python | JavaScript | NPM Package | Notes |
|--------|------------|-------------|-------|
| `uv` | pnpm | `pnpm@8.15.0` | Fast package manager |
| `pytest` | Vitest | `vitest@1.2.0` | Fast test runner with Vite |
| `ruff` | ESLint + Prettier | `eslint@8.56.0`, `prettier@3.2.4` | Code quality |
| `pyright` | TypeScript | `typescript@5.3.3` | Type checking built-in |
| `pre-commit` | Husky + lint-staged | `husky@9.0.0`, `lint-staged@15.2.0` | Git hooks |

## Memory Management Differences

### Python (Automatic)
```python
# Automatic garbage collection
image = cv2.imread('file.jpg')
result = cv2.GaussianBlur(image, (5, 5), 0)
# Memory freed automatically
```

### JavaScript (Manual for cv.Mat)
```typescript
// Manual memory management required
const mat = cv.imread(imgElement);
const result = new cv.Mat();
try {
    cv.GaussianBlur(mat, result, new cv.Size(5, 5), 0);
    // Use result
} finally {
    // Must manually delete to avoid memory leaks
    mat.delete();
    result.delete();
}
```

## Package Versions for package.json

```json
{
  "dependencies": {
    "@zxing/library": "^0.20.0",
    "ajv": "^8.12.0",
    "deepmerge": "^4.3.1",
    "jsts": "^2.11.0"
  },
  "devDependencies": {
    "@techstark/opencv-js": "^4.9.0",
    "@types/node": "^20.11.0",
    "@typescript-eslint/eslint-plugin": "^6.19.0",
    "@typescript-eslint/parser": "^6.19.0",
    "eslint": "^8.56.0",
    "husky": "^9.0.0",
    "lint-staged": "^15.2.0",
    "prettier": "^3.2.4",
    "typescript": "^5.3.3",
    "vite": "^5.0.11",
    "vitest": "^1.2.0"
  }
}
```

## Browser Compatibility

### Minimum Requirements
- **Chrome/Edge**: 90+ (WASM SIMD support)
- **Firefox**: 89+ (WASM SIMD support)
- **Safari**: 14.1+ (WASM support, limited SIMD)

### Feature Detection
```typescript
// Check for required features
const hasWASM = typeof WebAssembly !== 'undefined';
const hasWorkers = typeof Worker !== 'undefined';
const hasSharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';
```

## Performance Considerations

| Aspect | Python | JavaScript (Browser) | Notes |
|--------|--------|----------------------|-------|
| Startup Time | ~1-2s | ~3-5s | opencv.js WASM loading |
| Processing Speed | 1x baseline | 0.7-0.9x | Slight overhead from WASM |
| Memory Usage | Lower | Higher | WASM heap allocation |
| Parallel Processing | ThreadPoolExecutor | Web Workers | Similar capabilities |

## Missing Features & Workarounds

### Not Available in opencv.js
1. **SIFT/SURF** (patented): Use ORB instead or custom build
2. **Some contrib modules**: Build custom opencv.js
3. **File system access**: Use File API + IndexedDB

### Custom Implementations Needed
1. **Template directory traversal**: Use File API with directory picker
2. **CSV file writing**: String builder + Blob download
3. **Progress visualization**: Custom HTML/CSS instead of rich

## Status Legend
- ✅ **Available**: Direct equivalent exists, ready to use
- ⚠️ **Partial**: Some functionality available, may need workarounds
- 🔄 **Future**: Planned for future implementation
- ❌ **N/A**: Not applicable in browser environment

## Python to TypeScript File Structure Mapping

This section provides 1:1 correspondence between Python and TypeScript files in the codebase.

### Core Module Structure

| Python File | TypeScript File | Phase | Status | Notes |
|------------|----------------|-------|--------|-------|
| `src/core/omr_processor.py` | `packages/core/src/core/OMRProcessor.ts` | 1 | ⚠️ Partial | Main OMR processor class |
| `src/core/types.py` | `packages/core/src/core/types.ts` | 1 | ⚠️ Partial | Core type definitions |

### Processors Module Structure

| Python File | TypeScript File | Phase | Status | Notes |
|------------|----------------|-------|--------|-------|
| `src/processors/base.py` | `packages/core/src/processors/base.ts` | 1 | ⚠️ Partial | Base Processor class & ProcessingContext |
| `src/processors/pipeline.py` | `packages/core/src/processors/Pipeline.ts` | 1 | ⚠️ Partial | Processing pipeline orchestrator |
| `src/processors/image/coordinator.py` | `packages/core/src/processors/image/coordinator.ts` | 1 | 🔄 Future | Preprocessing coordinator |
| `src/processors/image/AutoRotate.py` | `packages/core/src/processors/image/AutoRotate.ts` | 1 | 🔄 Future | Auto-rotation processor |
| `src/processors/image/Contrast.py` | `packages/core/src/processors/image/Contrast.ts` | 1 | 🔄 Future | Contrast adjustment |
| `src/processors/image/GaussianBlur.py` | `packages/core/src/processors/image/GaussianBlur.ts` | 1 | 🔄 Future | Gaussian blur filter |
| `src/processors/image/MedianBlur.py` | `packages/core/src/processors/image/MedianBlur.ts` | 1 | 🔄 Future | Median blur filter |
| `src/processors/image/Levels.py` | `packages/core/src/processors/image/Levels.ts` | 1 | 🔄 Future | Levels adjustment |
| `src/processors/image/CropOnMarkers.py` | `packages/core/src/processors/image/CropOnMarkers.ts` | 1 | 🔄 Future | Marker-based cropping |
| `src/processors/image/CropPage.py` | `packages/core/src/processors/image/CropPage.ts` | 1 | 🔄 Future | Page cropping |
| `src/processors/alignment/processor.py` | `packages/core/src/processors/alignment/AlignmentProcessor.ts` | 1 | 🔄 Future | Template alignment |
| `src/processors/alignment/template_alignment.py` | `packages/core/src/processors/alignment/templateAlignment.ts` | 1 | 🔄 Future | Template alignment logic |
| `src/processors/detection/processor.py` | `packages/core/src/processors/detection/DetectionProcessor.ts` | 1 | 🔄 Future | OMR detection processor |
| `src/processors/evaluation/processor.py` | `packages/core/src/processors/evaluation/EvaluationProcessor.ts` | 1 | 🔄 Future | Answer evaluation |

### Utils Module Structure

| Python File | TypeScript File | Phase | Status | Notes |
|------------|----------------|-------|--------|-------|
| `src/utils/image.py` | `packages/core/src/utils/image.ts` | 1 | 🔄 Future | Image utility functions |
| `src/utils/geometry.py` | `packages/core/src/utils/geometry.ts` | 1 | 🔄 Future | Geometry utilities |
| `src/utils/math.py` | `packages/core/src/utils/math.ts` | 1 | 🔄 Future | Math utilities |
| `src/utils/logger.py` | `packages/core/src/utils/logger.ts` | 1 | 🔄 Future | Logging utilities |
| `src/utils/drawing.py` | `packages/core/src/utils/drawing.ts` | 2 | 🔄 Future | Drawing utilities |
| `src/utils/csv.py` | `packages/core/src/utils/csv.ts` | 1 | 🔄 Future | CSV export utilities |
| `src/utils/file.py` | `packages/core/src/utils/file.ts` | 1 | 🔄 Future | File handling (browser adapted) |

### Schemas Module Structure

| Python File | TypeScript File | Phase | Status | Notes |
|------------|----------------|-------|--------|-------|
| `src/schemas/config_schema.py` | `packages/core/src/schemas/configSchema.ts` | 1 | 🔄 Future | Config schema validation |
| `src/schemas/template_schema.py` | `packages/core/src/schemas/templateSchema.ts` | 1 | 🔄 Future | Template schema validation |
| `src/schemas/evaluation_schema.py` | `packages/core/src/schemas/evaluationSchema.ts` | 1 | 🔄 Future | Evaluation schema |

### Advanced Features (Phase 2+)

| Python File | TypeScript File | Phase | Status | Notes |
|------------|----------------|-------|--------|-------|
| `src/processors/detection/ml_detector.py` | `packages/core/src/processors/detection/mlDetector.ts` | Future | 🔄 Future | ML-based bubble detection |
| `src/processors/detection/barcode/*.py` | `packages/core/src/processors/detection/barcode/*.ts` | 2 | 🔄 Future | Barcode detection |
| `src/processors/detection/ocr/*.py` | `packages/core/src/processors/detection/ocr/*.ts` | Future | 🔄 Future | OCR support |
| `src/processors/visualization/*.py` | `packages/core/src/processors/visualization/*.ts` | 2 | 🔄 Future | Visualization tools |

### Phase Definitions

- **Phase 1**: Core pipeline + basic processors (alignment, detection, evaluation)
- **Phase 2**: Advanced processors + visualization + barcode
- **Future**: ML models, OCR, advanced features

## ES6/TypeScript Module System Mapping

### Python Import Patterns → ES6 Import/Export

| Python Pattern | TypeScript/ES6 Equivalent | Notes |
|---------------|---------------------------|-------|
| `from module import Class` | `import { Class } from './module'` | Named import |
| `from module import *` | `import * as module from './module'` | Namespace import (avoid if possible) |
| `import module` | `import * as module from './module'` | Import entire module |
| `from .relative import Class` | `import { Class } from './relative'` | Relative imports (`.py` → `.ts`/`.js`) |
| `from ..parent import Class` | `import { Class } from '../parent'` | Parent directory import |

### Python Export Patterns → ES6 Export

| Python Pattern | TypeScript/ES6 Equivalent | Notes |
|---------------|---------------------------|-------|
| Class definition in file | `export class ClassName { }` | Named export |
| Function definition | `export function funcName() { }` | Named export |
| `__all__ = [...]` | Multiple `export` statements | Define public API |
| Module-level variable | `export const VAR_NAME = ...` | Constant export |

### Module Resolution

**Python** uses relative/absolute imports based on `sys.path`:
```python
from src.core.omr_processor import OMRProcessor
from ..utils.image import ImageUtils
```

**TypeScript** uses ES6 module resolution:
```typescript
import { OMRProcessor } from '@/core/OMRProcessor';  // Path alias
import { ImageUtils } from '../utils/image';         // Relative path
```

**Recommendation**: Configure TypeScript path aliases in `tsconfig.json`:
```json
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./src/*"],
      "@core/*": ["./src/core/*"],
      "@processors/*": ["./src/processors/*"],
      "@utils/*": ["./src/utils/*"]
    }
  }
}
```

## Class and Function Naming Conventions

### Naming Convention Translation

| Python Convention | TypeScript Convention | Example |
|------------------|----------------------|---------|
| `snake_case` (functions/variables) | `camelCase` | `apply_on_image` → `applyOnImage` |
| `PascalCase` (classes) | `PascalCase` | `OMRProcessor` → `OMRProcessor` |
| `UPPER_SNAKE_CASE` (constants) | `UPPER_SNAKE_CASE` or `camelCase` | `DEFAULT_THRESHOLD` → `DEFAULT_THRESHOLD` |
| `_private_method` (convention) | `private methodName()` or `#methodName` | `_apply_filter` → `private applyFilter()` |
| `__dunder__` (special methods) | Standard methods | `__init__` → `constructor` |

### Common Method Name Mappings

| Python Method | TypeScript Method | Notes |
|--------------|------------------|-------|
| `__init__(self, ...)` | `constructor(...)` | Class constructor |
| `__str__(self)` | `toString()` | String representation |
| `__repr__(self)` | `toString()` or custom | Debug representation |
| `apply_on_image(image)` | `applyOnImage(image)` | snake_case → camelCase |
| `get_name()` | `getName()` | Getter method |
| `is_valid()` | `isValid()` | Boolean check |
| `has_errors()` | `hasErrors()` | Boolean check |

### Property Access Patterns

**Python** uses `@property` decorator:
```python
class Processor:
    @property
    def name(self) -> str:
        return self._name
```

**TypeScript** uses getter/setter or public fields:
```typescript
class Processor {
    private _name: string;

    get name(): string {
        return this._name;
    }

    // Or simply:
    // public readonly name: string;
}
```

## Common Python to TypeScript Pattern Translations

### List Comprehensions

**Python:**
```python
squared = [x ** 2 for x in numbers]
filtered = [x for x in numbers if x > 0]
```

**TypeScript:**
```typescript
const squared = numbers.map(x => x ** 2);
const filtered = numbers.filter(x => x > 0);
```

### Dictionary Comprehensions

**Python:**
```python
dict_comp = {k: v * 2 for k, v in items.items()}
```

**TypeScript:**
```typescript
const dictComp = Object.fromEntries(
    Object.entries(items).map(([k, v]) => [k, v * 2])
);
```

### Context Managers

**Python:**
```python
with open(file_path, 'r') as f:
    content = f.read()
```

**TypeScript** (no direct equivalent, use try-finally):
```typescript
const resource = await openResource(filePath);
try {
    const content = await resource.read();
} finally {
    await resource.close();
}
```

### Decorators

**Python:**
```python
@staticmethod
def utility_function():
    pass

@property
def value(self):
    return self._value
```

**TypeScript:**
```typescript
static utilityFunction(): void {
    // ...
}

get value(): ValueType {
    return this._value;
}

// Or with decorators (experimental):
@readonly
public value: ValueType;
```

### Default Parameters with None

**Python:**
```python
def process(image, threshold=None):
    if threshold is None:
        threshold = 128
```

**TypeScript:**
```typescript
function process(
    image: cv.Mat,
    threshold?: number  // Optional parameter
): void {
    const thresh = threshold ?? 128;  // Nullish coalescing
}
```

### Type Annotations

**Python:**
```python
def process_image(
    image: MatLike,
    threshold: int = 128
) -> MatLike:
    pass
```

**TypeScript:**
```typescript
function processImage(
    image: cv.Mat,
    threshold: number = 128
): cv.Mat {
    // ...
}
```

### Dataclasses to Interfaces/Types

**Python:**
```python
from dataclasses import dataclass

@dataclass
class ProcessingContext:
    file_path: str
    gray_image: MatLike
    colored_image: MatLike
```

**TypeScript:**
```typescript
interface ProcessingContext {
    filePath: string;
    grayImage: cv.Mat;
    coloredImage: cv.Mat;
}

// Or as a class:
class ProcessingContext {
    constructor(
        public filePath: string,
        public grayImage: cv.Mat,
        public coloredImage: cv.Mat
    ) {}
}
```

### Enums

**Python:**
```python
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    COMPLETE = "complete"
```

**TypeScript:**
```typescript
enum Status {
    Pending = "pending",
    Complete = "complete"
}

// Or as const object (preferred):
const Status = {
    Pending: "pending",
    Complete: "complete"
} as const;

type Status = typeof Status[keyof typeof Status];
```

## Development Tools & Workflow

### Updated Build Tools Mapping

| Tool Type | Python | TypeScript/JavaScript | Notes |
|-----------|--------|----------------------|-------|
| Package Manager | `uv` | `pnpm` | Fast, efficient, monorepo support |
| Test Runner | `pytest` | `vitest` | Fast, Vite-powered testing |
| Linter | `ruff` | `eslint` + `@typescript-eslint` | Code quality enforcement |
| Formatter | `ruff format` | `prettier` | Code formatting |
| Type Checker | `pyright` | `tsc` (TypeScript compiler) | Built into TypeScript |
| Git Hooks | `pre-commit` | `husky` + `lint-staged` | Pre-commit validation |
| Build Tool | N/A | `vite` | Fast bundler for development & production |
| E2E Testing | `playwright` (Python) | `playwright` (@playwright/test) | Same tool, different bindings |

### Package.json for Monorepo (pnpm)

```json
{
  "name": "omrchecker-monorepo",
  "private": true,
  "workspaces": [
    "packages/*",
    "change-propagation-tool"
  ],
  "scripts": {
    "dev": "pnpm --filter @omrchecker/demo dev",
    "build": "pnpm -r build",
    "test": "pnpm -r test",
    "lint": "pnpm -r lint",
    "format": "prettier --write \"**/*.{ts,tsx,js,jsx,json,md}\"",
    "typecheck": "pnpm -r typecheck",
    "change-tool": "pnpm --filter change-propagation-tool dev"
  },
  "devDependencies": {
    "@typescript-eslint/eslint-plugin": "^6.19.0",
    "@typescript-eslint/parser": "^6.19.0",
    "eslint": "^8.56.0",
    "husky": "^9.0.0",
    "lint-staged": "^15.2.0",
    "prettier": "^3.2.4",
    "typescript": "^5.3.3"
  }
}
```

### pnpm-workspace.yaml

```yaml
packages:
  - 'packages/*'
  - 'change-propagation-tool'
```

### Core Package Configuration

**packages/core/package.json:**
```json
{
  "name": "@omrchecker/core",
  "version": "1.0.0",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "types": "./dist/index.d.ts"
    }
  },
  "scripts": {
    "dev": "vite build --watch",
    "build": "tsc && vite build",
    "test": "vitest",
    "lint": "eslint src --ext .ts,.tsx",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@techstark/opencv-js": "^4.9.0",
    "@zxing/library": "^0.20.0",
    "ajv": "^8.12.0",
    "deepmerge": "^4.3.1",
    "jsts": "^2.11.0"
  },
  "devDependencies": {
    "@types/node": "^20.11.0",
    "vite": "^5.0.11",
    "vitest": "^1.2.0"
  }
}
```

### React Demo Package Configuration

**packages/demo/package.json:**
```json
{
  "name": "@omrchecker/demo",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint src --ext .ts,.tsx"
  },
  "dependencies": {
    "@omrchecker/core": "workspace:*",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.1",
    "vite": "^5.0.11"
  }
}
```

### Playwright E2E Package Configuration

**packages/e2e/package.json:**
```json
{
  "name": "@omrchecker/e2e",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "test": "playwright test",
    "test:ui": "playwright test --ui",
    "test:debug": "playwright test --debug"
  },
  "devDependencies": {
    "@playwright/test": "^1.41.0",
    "@types/node": "^20.11.0"
  }
}
```

## Change Propagation Workflow

This section describes the workflow for keeping Python and TypeScript codebases synchronized.

### Overview

When Python code changes, the corresponding TypeScript code must be updated to maintain feature parity. This workflow is enforced through:

1. **Automated Detection**: Pre-commit hooks detect Python changes
2. **Visual Feedback**: Rich terminal output shows what needs updating
3. **Interactive Tools**: Web UI guides developers through updates
4. **Validation**: Automated checks ensure synchronization

### Step-by-Step Workflow

#### 1. Developer Makes Python Changes

Developer works on Python code normally:
```bash
# Edit Python file
vim src/processors/image/AutoRotate.py

# Stage changes
git add src/processors/image/AutoRotate.py

# Attempt commit
git commit -m "Add rotation angle parameter to AutoRotate"
```

#### 2. Pre-commit Hook Detects Changes

The `validate-code-correspondence` hook analyzes the commit:

```
🔍 Analyzing Python changes...
📊 Python AST parsing complete
📋 Checking FILE_MAPPING.json for correspondences...

⚠️  SYNC CHECK FAILED

┌──────────────────────────────────────────────────────────────┐
│ File                         │ Status  │ Priority │ Action   │
├──────────────────────────────┼─────────┼──────────┼──────────┤
│ src/processors/image/        │ ❌ OUT  │ HIGH     │ Required │
│   AutoRotate.py              │ OF SYNC │          │          │
│                              │         │          │          │
│ TS File:                     │         │          │          │
│   packages/core/src/         │         │          │          │
│   processors/image/          │         │          │          │
│   AutoRotate.ts              │         │          │          │
└──────────────────────────────────────────────────────────────┘

📝 Detected changes:
  • Method 'apply_on_image' modified (line 45-67)
  • New parameter added: rotation_angle: int = 0
  • Return type unchanged: MatLike

🎯 Required TypeScript updates:
  1. Add parameter to applyOnImage() method
  2. Update parameter type: rotationAngle: number = 0
  3. Update method implementation if needed

💡 Quick actions:
  1. Launch change propagation tool: pnpm run change-tool
  2. Or manually update: code packages/core/src/processors/image/AutoRotate.ts

  Bypass validation (not recommended): git commit --no-verify
```

#### 3. Launch Change Propagation Tool

Developer chooses to use the interactive tool:

```bash
pnpm run change-tool
```

Web UI opens at `http://localhost:5174` showing:
- **Dashboard**: Card for AutoRotate.py with "Out of Sync" status
- **Details**: Click card → opens side-by-side diff view
- **Python (left)**: Highlighted changes in green
- **TypeScript (right)**: Current code with suggested changes in ghost text

#### 4. Review and Apply Suggestions

In the UI:
1. **Review Python changes** highlighted in green
2. **See suggested TypeScript code** generated automatically
3. **Edit in Monaco Editor** with live TypeScript validation
4. **Apply suggestions** or make manual adjustments
5. **Save and validate** → TypeScript compiler checks for errors
6. **Mark as synced** → Updates FILE_MAPPING.json

#### 5. Retry Commit

After marking as synced, retry commit:

```bash
git commit -m "Add rotation angle parameter to AutoRotate"

🔍 Analyzing Python changes...
✅ All Python changes have corresponding TypeScript updates
✅ FILE_MAPPING.json is up to date
✅ Commit allowed

[main abc1234] Add rotation angle parameter to AutoRotate
 3 files changed, 15 insertions(+), 3 deletions(-)
```

### Automated Suggestion Generation

The system can automatically generate TypeScript suggestions for common patterns:

**Python change detected:**
```python
def apply_on_image(
    self,
    image: MatLike,
    template: Template,
    rotation_angle: int = 0  # ← NEW
) -> MatLike:
    """Apply rotation to the image.

    Args:
        image: Input image
        template: Template configuration
        rotation_angle: Rotation angle in degrees

    Returns:
        Rotated image
    """
    # Implementation...
```

**Auto-generated TypeScript suggestion:**
```typescript
applyOnImage(
    image: cv.Mat,
    template: Template,
    rotationAngle: number = 0  // ← SUGGESTED
): cv.Mat {
    /**
     * Apply rotation to the image.
     *
     * @param image - Input image
     * @param template - Template configuration
     * @param rotationAngle - Rotation angle in degrees
     * @returns Rotated image
     */
    // Implementation...
}
```

### Command Line Tools

**Check sync status:**
```bash
pnpm sync-tool status

Sync Status Report
==================
Total mapped files: 47
✅ In sync: 42 (89%)
⚠️  Needs review: 3 (6%)
❌ Out of sync: 2 (4%)

Out of sync files:
  1. src/processors/image/AutoRotate.py → AutoRotate.ts
  2. src/utils/geometry.py → geometry.ts
```

**Generate suggestions for a file:**
```bash
pnpm sync-tool suggest src/processors/image/AutoRotate.py

Analyzing src/processors/image/AutoRotate.py...

Suggested TypeScript changes for AutoRotate.ts:

1. Method applyOnImage (line 45):
   + Add parameter: rotationAngle: number = 0

2. Import updates:
   + No new imports needed

3. Type mappings:
   - int → number
   - MatLike → cv.Mat

Full suggested code written to: .sync-suggestions/AutoRotate.ts.patch
Apply with: git apply .sync-suggestions/AutoRotate.ts.patch
```

### CI/CD Integration

**GitHub Actions workflow:**
```yaml
name: TypeScript Sync Check

on: [pull_request]

jobs:
  check-sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v2

      - name: Check Python-TypeScript sync
        run: |
          python scripts/detect_python_changes.py --ci-mode

      - name: Comment on PR if out of sync
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '⚠️ Python changes detected without corresponding TypeScript updates. Please sync the codebases.'
            })
```

### Best Practices

1. **Make atomic changes**: Change one feature at a time in both codebases
2. **Sync frequently**: Don't let Python changes accumulate without TS updates
3. **Use the tools**: The change propagation tool saves time and reduces errors
4. **Review suggestions**: Auto-generated code is a starting point, always review
5. **Test both**: Run tests for both Python and TypeScript after changes
6. **Document differences**: If implementations must differ, document why in FILE_MAPPING.json

### Bypassing Validation (Emergency Only)

In rare cases where you need to commit Python changes without immediate TS updates:

```bash
# Add a tracking issue to FILE_MAPPING.json first
# Then bypass with:
git commit --no-verify -m "Python changes - TS update tracked in #123"
```

**Important**: Create a GitHub issue tracking the needed TypeScript update before bypassing!

