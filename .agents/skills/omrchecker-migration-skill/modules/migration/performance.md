# Performance Considerations for Browser

**Module**: modules/migration/
**Created**: 2026-02-20

---

## Browser Performance Constraints

### Memory Limits

| Browser | Typical Heap Limit | ArrayBuffer Limit |
|---------|-------------------|-------------------|
| Chrome Desktop | 2-4 GB | ~2 GB |
| Firefox Desktop | 2-4 GB | ~2 GB |
| Safari Desktop | 2-4 GB | ~2 GB |
| Chrome Mobile | 512 MB - 1 GB | ~500 MB |
| Safari iOS | 256 MB - 512 MB | ~200 MB |

**Implications**:
- Process images in batches
- Dispose of Mat objects immediately
- Use Web Workers to separate heap
- Downscale large images before processing

### Processing Large Images

```javascript
// Bad: Load full resolution
const image = await loadImage(file); // 4000x3000 = 12 MP

// Good: Downscale first
async function loadAndResize(file, maxDimension = 1500) {
  const img = await loadImage(file);
  const scale = Math.min(1, maxDimension / Math.max(img.width, img.height));

  if (scale < 1) {
    const canvas = document.createElement('canvas');
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  return img;
}
```

---

## WebAssembly Optimization

### OpenCV.js Performance

**Use SIMD build when available**:
```html
<script>
  // Detect SIMD support
  const simdSupport = (() => {
    try {
      return typeof WebAssembly.validate === 'function' &&
        WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]));
    } catch (e) {
      return false;
    }
  })();

  // Load appropriate build
  const opencvUrl = simdSupport
    ? '/opencv-simd.js'
    : '/opencv.js';
</script>
```

**Benchmark**: SIMD build can be 2-3x faster for image operations

---

## Web Worker Strategy

### Dedicated Worker Pool

```javascript
class WorkerPool {
  constructor(workerUrl, poolSize = navigator.hardwareConcurrency || 4) {
    this.workers = [];
    this.queue = [];

    for (let i = 0; i < poolSize; i++) {
      const worker = new Worker(workerUrl);
      worker.busy = false;

      worker.onmessage = (e) => {
        worker.busy = false;
        worker.resolver(e.data);
        this.processQueue();
      };

      this.workers.push(worker);
    }
  }

  async execute(data) {
    return new Promise((resolve) => {
      this.queue.push({ data, resolve });
      this.processQueue();
    });
  }

  processQueue() {
    if (this.queue.length === 0) return;

    const worker = this.workers.find(w => !w.busy);
    if (!worker) return;

    const { data, resolve } = this.queue.shift();
    worker.busy = true;
    worker.resolver = resolve;
    worker.postMessage(data);
  }
}

// Usage
const pool = new WorkerPool('omr-worker.js', 4);

async function processBatch(images) {
  const promises = images.map(img =>
    pool.execute({ type: 'process', image: img })
  );

  return Promise.all(promises);
}
```

---

## GPU Acceleration

### WebGL for Image Processing

```javascript
// Custom WebGL shader for grayscale conversion (faster than Canvas)
function createGrayscaleShader(gl) {
  const fragmentShader = `
    precision mediump float;
    varying vec2 vTexCoord;
    uniform sampler2D uImage;

    void main() {
      vec4 color = texture2D(uImage, vTexCoord);
      float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
      gl_FragColor = vec4(vec3(gray), 1.0);
    }
  `;

  return compileShader(gl, fragmentShader);
}

// 5-10x faster than CPU grayscale conversion
```

### TensorFlow.js WebGL Backend

```javascript
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';

// Configure WebGL for performance
await tf.setBackend('webgl');
tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
tf.env().set('WEBGL_PACK', true);

// Batch operations for GPU efficiency
const batched = tf.stack(imageTensors); // Batch 8 images
const results = await model.predict(batched);
```

---

## Memory Management

### OpenCV.js Memory Leaks

```javascript
// Memory leak detector
let matCount = 0;

cv.Mat = new Proxy(cv.Mat, {
  construct(target, args) {
    matCount++;
    console.log(`Mat created. Total: ${matCount}`);
    return new target(...args);
  }
});

// Override delete to track
const originalDelete = cv.Mat.prototype.delete;
cv.Mat.prototype.delete = function() {
  matCount--;
  console.log(`Mat deleted. Total: ${matCount}`);
  return originalDelete.call(this);
};
```

### Auto-cleanup Pattern

```javascript
class MatManager {
  constructor() {
    this.mats = new Set();
  }

  create(...args) {
    const mat = new cv.Mat(...args);
    this.mats.add(mat);
    return mat;
  }

  cleanup() {
    this.mats.forEach(mat => {
      if (!mat.isDeleted()) {
        mat.delete();
      }
    });
    this.mats.clear();
  }
}

// Usage
async function processImage(src) {
  const manager = new MatManager();

  try {
    const gray = manager.create();
    const blurred = manager.create();

    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);

    return blurred.clone(); // Return copy
  } finally {
    manager.cleanup(); // Auto-cleanup
  }
}
```

---

## Caching Strategies

### Template Caching

```javascript
const templateCache = new Map();

async function getTemplate(id) {
  if (templateCache.has(id)) {
    return templateCache.get(id);
  }

  const template = await loadTemplate(id);

  // Cache parsed template
  templateCache.set(id, template);

  return template;
}
```

### Service Worker Caching

```javascript
// Cache static assets and models
const CACHE_NAME = 'omr-v1';
const STATIC_ASSETS = [
  '/opencv.js',
  '/models/yolo.json',
  '/models/group1-shard1of1.bin'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache =>
      cache.addAll(STATIC_ASSETS)
    )
  );
});
```

---

## Performance Monitoring

### Performance API

```javascript
function measureOperation(name, fn) {
  return async (...args) => {
    performance.mark(`${name}-start`);

    const result = await fn(...args);

    performance.mark(`${name}-end`);
    performance.measure(name, `${name}-start`, `${name}-end`);

    const measure = performance.getEntriesByName(name)[0];
    console.log(`${name} took ${measure.duration.toFixed(2)}ms`);

    return result;
  };
}

// Usage
const processImage = measureOperation('processImage', async (image) => {
  // ... processing
});
```

### Memory Monitoring

```javascript
// Chrome-only
if (performance.memory) {
  setInterval(() => {
    const used = (performance.memory.usedJSHeapSize / 1048576).toFixed(2);
    const limit = (performance.memory.jsHeapSizeLimit / 1048576).toFixed(2);
    console.log(`Memory: ${used} MB / ${limit} MB`);
  }, 5000);
}
```

---

## Optimization Checklist

### Image Processing
- [ ] Downscale large images before processing
- [ ] Use WebGL for heavy pixel operations
- [ ] Batch similar operations together
- [ ] Reuse buffers when possible

### OpenCV.js
- [ ] Always delete Mat objects
- [ ] Use SIMD build when available
- [ ] Warm up operations on startup
- [ ] Process in Web Worker

### ML Models
- [ ] Quantize models (4x smaller)
- [ ] Use WebGL backend
- [ ] Batch predictions
- [ ] Cache models in IndexedDB

### General
- [ ] Use Web Workers for parallelism
- [ ] Implement lazy loading
- [ ] Monitor memory usage
- [ ] Provide progress feedback

---

## Performance Targets

| Operation | Target Time | Notes |
|-----------|-------------|-------|
| Load OpenCV.js | < 2s | First time, then cached |
| Load template | < 100ms | From IndexedDB |
| Process single image | < 3s | 1500x2000 image |
| ML bubble detection | < 1s | With warm model |
| Classical detection | < 500ms | Threshold-based |
| Export CSV | < 100ms | 100 images |

---

## Browser-Specific Optimizations

### Chrome
- Use SharedArrayBuffer when possible
- Enable WebAssembly SIMD
- Use OffscreenCanvas in workers

### Firefox
- Prefer typed arrays over regular arrays
- Use transferable objects
- Monitor memory actively

### Safari
- Be conservative with memory
- Test on iOS devices (more constrained)
- Provide simpler fallbacks

---

## Summary

**Key Strategies**:
1. **Memory**: Downscale images, cleanup Mat objects, use workers
2. **Speed**: WebGL/GPU, SIMD, batch operations
3. **Caching**: Templates, models, static assets
4. **Monitoring**: Performance API, memory tracking

**Critical Optimizations**:
- Use Web Workers for all heavy processing
- Always cleanup OpenCV.js Mat objects
- Downscale images to reasonable size
- Cache everything that doesn't change

**Performance Formula**: Good UX = Fast loading + Responsive UI + Efficient processing
