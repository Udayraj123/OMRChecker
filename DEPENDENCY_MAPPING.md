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

