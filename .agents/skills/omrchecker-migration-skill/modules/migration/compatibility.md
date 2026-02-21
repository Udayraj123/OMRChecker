# Browser Compatibility Matrix

**Module**: modules/migration/
**Created**: 2026-02-20

---

## Core Technologies Compatibility

### WebAssembly

| Browser | Version | SIMD Support | Threads Support |
|---------|---------|--------------|-----------------|
| Chrome | 57+ | 91+ | 74+ (with headers) |
| Firefox | 52+ | 89+ | 79+ (with headers) |
| Safari | 11+ | 16.4+ | ❌ |
| Edge | 16+ | 91+ | 79+ (with headers) |
| Mobile Safari | 11+ | 16.4+ | ❌ |
| Mobile Chrome | 57+ | 91+ | 74+ (with headers) |

**Notes**: SIMD provides 2-3x performance boost for OpenCV.js

### Web Workers

| Browser | Version | Support |
|---------|---------|---------|
| Chrome | All modern | ✅ |
| Firefox | All modern | ✅ |
| Safari | All modern | ✅ |
| Edge | All modern | ✅ |
| Mobile | All modern | ✅ |

**Universal support** - safe to use

### SharedArrayBuffer

| Browser | Version | Requirements |
|---------|---------|--------------|
| Chrome | 68+ | COOP + COEP headers |
| Firefox | 79+ | COOP + COEP headers |
| Safari | 15.2+ | COOP + COEP headers |
| Edge | 79+ | COOP + COEP headers |

**Required Headers**:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

### OffscreenCanvas

| Browser | Version | Support |
|---------|---------|---------|
| Chrome | 69+ | ✅ |
| Firefox | 105+ | ✅ |
| Safari | ❌ | ❌ |
| Edge | 79+ | ✅ |

**Fallback needed for Safari**

---

## Canvas API

### 2D Context

**Universal support** across all modern browsers

### WebGL

| Browser | Version | WebGL 1 | WebGL 2 |
|---------|---------|---------|---------|
| Chrome | 56+ | ✅ | ✅ |
| Firefox | 51+ | ✅ | ✅ |
| Safari | 15+ | ✅ | ✅ |
| Edge | 79+ | ✅ | ✅ |

**Detection**:
```javascript
function detectWebGL() {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
  return !!gl;
}
```

---

## File API

### FileReader

**Universal support** - safe to use

### Drag and Drop

**Universal support** with minor differences:

```javascript
// Compatible drag/drop handler
function setupDragDrop(element, handler) {
  element.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
  });

  element.addEventListener('drop', async (e) => {
    e.preventDefault();
    e.stopPropagation();

    const files = Array.from(e.dataTransfer.files);
    await handler(files);
  });
}
```

---

## Storage APIs

### LocalStorage

**Universal support** - 5-10 MB limit

### IndexedDB

| Browser | Version | Support | Size Limit |
|---------|---------|---------|------------|
| Chrome | All | ✅ | 60% of disk |
| Firefox | All | ✅ | 50% of disk |
| Safari | 10+ | ✅ | ~1 GB |
| Edge | All | ✅ | 60% of disk |

**Wrapper recommended**:
```javascript
import { openDB } from 'idb'; // idb library for better API
```

### Cache API

| Browser | Version | Support |
|---------|---------|---------|
| Chrome | 40+ | ✅ |
| Firefox | 39+ | ✅ |
| Safari | 11.1+ | ✅ |
| Edge | 17+ | ✅ |

---

## ML Libraries Compatibility

### TensorFlow.js

| Backend | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| WebGL | ✅ | ✅ | ✅ | ✅ |
| WASM | ✅ | ✅ | ✅ | ✅ |
| CPU | ✅ | ✅ | ✅ | ✅ |

**Recommended**:
```javascript
import * as tf from '@tensorflow/tfjs';
await tf.setBackend('webgl'); // Best performance
```

### ONNX Runtime Web

| Browser | Version | Support |
|---------|---------|---------|
| Chrome | 80+ | ✅ |
| Firefox | 80+ | ✅ |
| Safari | 13+ | ✅ |
| Edge | 80+ | ✅ |

### Tesseract.js

**Universal support** - Pure JavaScript + WebAssembly

---

## OpenCV.js Compatibility

| Browser | Version | Support | Notes |
|---------|---------|---------|-------|
| Chrome | 57+ | ✅ | Full support, SIMD 91+ |
| Firefox | 52+ | ✅ | Full support, SIMD 89+ |
| Safari | 11+ | ✅ | No SIMD until 16.4 |
| Edge | 16+ | ✅ | Full support |
| Mobile Chrome | 74+ | ✅ | May be slow on older devices |
| Mobile Safari | 11+ | ⚠️ | Memory constraints |

**Mobile Considerations**:
- Downscale images aggressively
- Use lower processing width
- Disable heavy features (ML models)

---

## Feature Detection

```javascript
const capabilities = {
  webAssembly: (() => {
    try {
      return typeof WebAssembly === 'object' &&
        typeof WebAssembly.instantiate === 'function';
    } catch (e) {
      return false;
    }
  })(),

  wasmSIMD: (() => {
    try {
      return WebAssembly.validate(new Uint8Array([
        0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123,
        3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
      ]));
    } catch (e) {
      return false;
    }
  })(),

  webGL: (() => {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    return !!gl;
  })(),

  webWorkers: typeof Worker !== 'undefined',

  sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',

  offscreenCanvas: typeof OffscreenCanvas !== 'undefined',

  indexedDB: 'indexedDB' in window
};

console.log('Browser Capabilities:', capabilities);
```

---

## Progressive Enhancement Strategy

### Level 1: Basic (All Browsers)

- File upload
- Canvas rendering
- Classical threshold detection
- CSV export

### Level 2: Enhanced (Modern Browsers)

- Web Workers for parallel processing
- IndexedDB for caching
- OpenCV.js (basic operations)

### Level 3: Advanced (Latest Browsers)

- SIMD-enabled OpenCV.js
- ML model inference
- WebGL acceleration
- SharedArrayBuffer

**Implementation**:
```javascript
function selectFeatureLevel() {
  if (capabilities.wasmSIMD && capabilities.webGL) {
    return 'advanced';
  } else if (capabilities.webWorkers && capabilities.indexedDB) {
    return 'enhanced';
  } else {
    return 'basic';
  }
}

const featureLevel = selectFeatureLevel();
console.log(`Running in ${featureLevel} mode`);
```

---

## Mobile Browser Considerations

### Memory Constraints

| Device Type | Typical Limit | Recommendation |
|-------------|---------------|----------------|
| High-end phone | 1-2 GB | Process up to 2000x3000 images |
| Mid-range phone | 512 MB - 1 GB | Process up to 1200x1600 images |
| Low-end phone | 256-512 MB | Process up to 800x1200 images |
| Tablet | 1-4 GB | Similar to high-end phone |

### Touch Interface

```javascript
// Mobile-friendly file input
<input type="file" accept="image/*" capture="camera" multiple>
```

### Viewport Meta

```html
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
```

---

## Minimum Browser Versions

**Recommended Minimum**:
- Chrome 91+ (for SIMD)
- Firefox 89+ (for SIMD)
- Safari 15.2+ (for SharedArrayBuffer)
- Edge 91+ (for SIMD)

**Absolute Minimum** (degraded experience):
- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 16+

---

## Testing Matrix

### Desktop Testing

- [ ] Chrome latest (Windows/Mac/Linux)
- [ ] Firefox latest (Windows/Mac/Linux)
- [ ] Safari latest (Mac only)
- [ ] Edge latest (Windows)

### Mobile Testing

- [ ] Chrome on Android 10+
- [ ] Safari on iOS 14+
- [ ] Firefox on Android 10+

### Feature-Specific Testing

- [ ] SIMD build performance
- [ ] SharedArrayBuffer with headers
- [ ] OffscreenCanvas in workers
- [ ] IndexedDB quota limits

---

## Polyfills & Fallbacks

### For older browsers:

```javascript
// Promise polyfill
import 'core-js/features/promise';

// Fetch polyfill
import 'whatwg-fetch';

// Typed Array polyfills
import 'core-js/features/typed-array';
```

### Feature fallbacks:

```javascript
// OffscreenCanvas fallback
function createCanvas(width, height) {
  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(width, height);
  } else {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }
}
```

---

## Summary

**Universal Support**:
- WebAssembly (basic)
- Web Workers
- Canvas 2D
- File API
- IndexedDB

**Require Feature Detection**:
- SIMD
- SharedArrayBuffer
- OffscreenCanvas
- WebGL

**Progressive Enhancement**:
- Basic: Classical detection (all browsers)
- Enhanced: OpenCV.js + workers (modern browsers)
- Advanced: SIMD + GPU + ML (latest browsers)

**Critical for OMRChecker**:
- WebAssembly: Core requirement
- Web Workers: Highly recommended
- Canvas API: Core requirement
- File API: Core requirement
- IndexedDB: Recommended for caching
