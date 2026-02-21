# Browser Adaptations for OMRChecker

**Module**: modules/migration/
**Created**: 2026-02-20

---

## File API Usage

### Reading Images

```javascript
async function loadImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = e.target.result;
    };

    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
```

### Drag and Drop

```html
<div id="dropZone" ondrop="handleDrop(event)" ondragover="event.preventDefault()">
  Drop images here
</div>
```

```javascript
function handleDrop(event) {
  event.preventDefault();
  const files = Array.from(event.dataTransfer.files)
    .filter(f => f.type.startsWith('image/'));

  processFiles(files);
}
```

---

## Web Workers for Heavy Processing

### Main Thread

```javascript
const worker = new Worker('omr-worker.js');

// Send image for processing
worker.postMessage({
  imageData: canvas.toDataURL(),
  template: template
});

// Receive results
worker.onmessage = (e) => {
  const { results, debugImage } = e.data;
  displayResults(results);
};
```

### Worker Thread

```javascript
// omr-worker.js
importScripts('opencv.js');

self.onmessage = async (e) => {
  const { imageData, template } = e.data;

  // Load image
  const img = await loadImageFromDataURL(imageData);

  // Process
  const results = processOMRSheet(img, template);

  // Send back
  self.postMessage({ results });
};
```

### Transferable Objects (Zero-Copy)

```javascript
// Transfer ArrayBuffer ownership
const buffer = imageData.data.buffer;
worker.postMessage({ buffer }, [buffer]);
// buffer is now neutered (unusable in main thread)
```

---

## SharedArrayBuffer for Shared Memory

**Requirements**: COOP/COEP headers

```html
<!-- index.html headers -->
<meta http-equiv="Cross-Origin-Opener-Policy" content="same-origin">
<meta http-equiv="Cross-Origin-Embedder-Policy" content="require-corp">
```

**Usage**:
```javascript
// Main thread
const sab = new SharedArrayBuffer(width * height * 4);
const imageData = new Uint8ClampedArray(sab);

worker.postMessage({ sab, width, height });

// Worker thread
self.onmessage = (e) => {
  const { sab, width, height } = e.data;
  const imageData = new Uint8ClampedArray(sab);

  // Both threads can access same memory
  processInPlace(imageData, width, height);
};
```

---

## Canvas API Patterns

### Drawing Debug Overlays

```javascript
function drawBubbleOverlay(canvas, bubbles, interpretations) {
  const ctx = canvas.getContext('2d');

  bubbles.forEach((bubble, idx) => {
    const interpretation = interpretations[idx];

    // Draw rectangle
    ctx.strokeStyle = interpretation.isAttempted ? '#00ff00' : '#ff0000';
    ctx.lineWidth = 2;
    ctx.strokeRect(bubble.x, bubble.y, bubble.width, bubble.height);

    // Draw label
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(bubble.x, bubble.y - 20, 40, 20);
    ctx.fillStyle = '#000000';
    ctx.font = '12px Arial';
    ctx.fillText(bubble.value, bubble.x + 5, bubble.y - 5);
  });
}
```

### OffscreenCanvas (Performance)

```javascript
// Main thread
const offscreen = canvas.transferControlToOffscreen();
worker.postMessage({ canvas: offscreen }, [offscreen]);

// Worker thread
self.onmessage = (e) => {
  const canvas = e.data.canvas;
  const ctx = canvas.getContext('2d');

  // Draw operations in worker
  ctx.fillStyle = 'red';
  ctx.fillRect(0, 0, 100, 100);
};
```

---

## IndexedDB for Persistent Storage

### Setup

```javascript
import { openDB } from 'idb';

const db = await openDB('omr-database', 1, {
  upgrade(db) {
    db.createObjectStore('templates', { keyPath: 'id' });
    db.createObjectStore('results', { keyPath: 'id', autoIncrement: true });
    db.createObjectStore('cache', { keyPath: 'key' });
  }
});
```

### Store Template

```javascript
async function saveTemplate(template) {
  const db = await getDB();
  await db.put('templates', {
    id: template.id,
    name: template.name,
    data: template,
    timestamp: Date.now()
  });
}
```

### Retrieve Template

```javascript
async function loadTemplate(id) {
  const db = await getDB();
  const stored = await db.get('templates', id);
  return stored?.data;
}
```

### Cache Models

```javascript
async function cacheModel(modelUrl, modelData) {
  const db = await getDB();
  await db.put('cache', {
    key: modelUrl,
    data: modelData,
    timestamp: Date.now()
  });
}
```

---

## LocalStorage for Simple Config

```javascript
// Save config
function saveConfig(config) {
  localStorage.setItem('omr-config', JSON.stringify(config));
}

// Load config
function loadConfig() {
  const stored = localStorage.getItem('omr-config');
  if (!stored) return getDefaultConfig();

  try {
    return JSON.parse(stored);
  } catch (e) {
    console.error('Invalid config in localStorage', e);
    return getDefaultConfig();
  }
}

// Clear config
function clearConfig() {
  localStorage.removeItem('omr-config');
}
```

---

## Progressive Web App (PWA)

### Manifest

```json
{
  "name": "OMRChecker Web",
  "short_name": "OMRChecker",
  "description": "Optical Mark Recognition in your browser",
  "start_url": "/",
  "display": "standalone",
  "theme_color": "#2196f3",
  "background_color": "#ffffff",
  "icons": [
    {
      "src": "/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/icon-512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
```

### Service Worker

```javascript
// service-worker.js
const CACHE_NAME = 'omr-v1';
const ASSETS = [
  '/',
  '/index.html',
  '/styles.css',
  '/app.js',
  '/opencv.js'
];

// Install
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache =>
      cache.addAll(ASSETS)
    )
  );
});

// Fetch (offline support)
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then(response =>
      response || fetch(event.request)
    )
  );
});
```

---

## WebGL for GPU Acceleration

### Custom Shaders for Image Processing

```javascript
// Vertex shader
const vertexShaderSource = `
  attribute vec2 position;
  attribute vec2 texCoord;
  varying vec2 vTexCoord;

  void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    vTexCoord = texCoord;
  }
`;

// Fragment shader (grayscale conversion)
const fragmentShaderSource = `
  precision mediump float;
  varying vec2 vTexCoord;
  uniform sampler2D uImage;

  void main() {
    vec4 color = texture2D(uImage, vTexCoord);
    float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    gl_FragColor = vec4(vec3(gray), color.a);
  }
`;
```

---

## Notification API

```javascript
// Request permission
await Notification.requestPermission();

// Show notification when processing complete
function notifyComplete(results) {
  if (Notification.permission === 'granted') {
    new Notification('OMR Processing Complete', {
      body: `Processed ${results.length} images`,
      icon: '/icon-192.png'
    });
  }
}
```

---

## Download Handling

### Single File

```javascript
function downloadFile(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
```

### Multiple Files (ZIP)

```javascript
import JSZip from 'jszip';

async function downloadResults(results) {
  const zip = new JSZip();

  // Add CSV
  zip.file('results.csv', generateCSV(results));

  // Add images
  results.forEach((result, idx) => {
    zip.file(`result_${idx}.jpg`, result.imageBlob);
  });

  const blob = await zip.generateAsync({ type: 'blob' });
  downloadFile(blob, 'omr-results.zip');
}
```

---

## Summary

**Key Browser APIs**:
1. **File API**: Reading user files
2. **Canvas API**: Image display and manipulation
3. **Web Workers**: Parallel processing
4. **IndexedDB**: Persistent storage
5. **LocalStorage**: Simple config storage
6. **Service Workers**: Offline support
7. **WebGL**: GPU acceleration
8. **Notifications**: User feedback

**Best Practices**:
- Use Web Workers for heavy processing
- Cache models and templates
- Provide offline support via service workers
- Use SharedArrayBuffer when available (with proper headers)
- Always handle memory cleanup for OpenCV.js
- Implement progress feedback for long operations
