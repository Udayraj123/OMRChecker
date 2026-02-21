# Python to JavaScript Migration Context

**Module**: modules/migration/
**Created**: 2026-02-20

---

## Technology Mapping Table

| Python Package | Browser Alternative | Notes |
|----------------|---------------------|-------|
| **OpenCV (cv2)** | OpenCV.js | WebAssembly, same API mostly |
| **NumPy** | TypedArrays + ndarray.js | Manual memory management |
| **Pydantic** | Zod | TypeScript schema validation |
| **PIL/Pillow** | Canvas API | Browser native image handling |
| **PyZbar** | @zxing/library | WebAssembly barcode scanner |
| **EasyOCR** | Tesseract.js | JavaScript OCR engine |
| **Tesseract** | Tesseract.js | Direct port |
| **YOLO (Ultralytics)** | TensorFlow.js | Convert models to tfjs format |
| **PyTorch** | TensorFlow.js / ONNX Runtime Web | Model conversion required |
| **ThreadPoolExecutor** | Web Workers | Message passing vs shared memory |
| **argparse** | Commander.js / Electron | CLI for Node.js/Electron apps |
| **pathlib** | File API | Browser file handling |
| **json** | JSON (native) | Built-in JavaScript |
| **csv** | PapaParse | CSV parsing library |
| **Rich** | blessed / chalk | Terminal styling (Node.js) |
| **pytest** | Jest / Vitest | Testing frameworks |

---

## Language Differences

### 1. Type System

**Python**:
```python
def process_image(image: np.ndarray, threshold: int = 200) -> dict:
    return {"success": True}
```

**TypeScript**:
```typescript
function processImage(
  image: Uint8Array,
  threshold: number = 200
): { success: boolean } {
  return { success: true };
}
```

### 2. Array Indexing

**Python** (NumPy):
```python
pixel = image[y, x]
roi = image[y1:y2, x1:x2]
```

**JavaScript**:
```javascript
const pixel = image[y * width + x];
const roi = extractROI(image, width, x1, y1, x2, y2);
```

### 3. Async/Await

**Python**:
```python
async def process():
    result = await load_model()
```

**JavaScript** (same syntax):
```javascript
async function process() {
  const result = await loadModel();
}
```

### 4. Classes

**Python**:
```python
class Processor:
    def __init__(self, config):
        self.config = config

    def process(self, image):
        return image
```

**TypeScript**:
```typescript
class Processor {
  config: Config;

  constructor(config: Config) {
    this.config = config;
  }

  process(image: Uint8Array): Uint8Array {
    return image;
  }
}
```

---

## Common Patterns Migration

### Pattern 1: File Processing Loop

**Python**:
```python
from pathlib import Path

for image_path in Path("inputs").glob("*.jpg"):
    result = process_image(image_path)
    save_result(result)
```

**Browser**:
```javascript
// User selects files
const fileInput = document.getElementById('input');

for (const file of fileInput.files) {
  const result = await processImage(file);
  saveResult(result);
}
```

### Pattern 2: Configuration Loading

**Python**:
```python
with open("config.json") as f:
    config = json.load(f)
```

**Browser**:
```javascript
// From file
const configFile = await selectFile();
const configText = await configFile.text();
const config = JSON.parse(configText);

// From localStorage
const config = JSON.parse(localStorage.getItem('config') || '{}');
```

### Pattern 3: Error Handling

**Python**:
```python
try:
    result = process_image(image)
except ImageProcessingError as e:
    logger.error(f"Failed: {e}")
```

**JavaScript**:
```javascript
try {
  const result = processImage(image);
} catch (error) {
  if (error instanceof ImageProcessingError) {
    console.error(`Failed: ${error.message}`);
  }
}
```

---

## Memory Management

### Python (Automatic)

```python
# Garbage collector handles cleanup
image = cv2.imread("test.jpg")
processed = cv2.GaussianBlur(image, (5, 5), 0)
# Memory automatically freed
```

### JavaScript (Manual for OpenCV.js)

```javascript
// Must manually delete Mat objects
const src = cv.imread(canvas);
const dst = new cv.Mat();

try {
  cv.GaussianBlur(src, dst, new cv.Size(5, 5), 0);
  // Use dst
} finally {
  src.delete();
  dst.delete();
}
```

---

## Performance Considerations

### Python Strengths
- Native NumPy operations (C/Fortran backend)
- Mature libraries
- Server-side resources

### Browser Strengths
- WebAssembly performance (near-native)
- GPU acceleration (WebGL, WebGPU)
- Parallel processing (Web Workers)
- No installation required

### Browser Limitations
- Memory constraints (heap size limits)
- No direct file system access
- Async-heavy operations
- Network latency for model loading

---

## Recommended Migration Path

1. **Start with Core Logic**: Port template parsing, field detection logic
2. **Add OpenCV.js**: Replace cv2 calls with OpenCV.js equivalents
3. **Handle File I/O**: Use File API, downloads instead of file writes
4. **Web Workers**: Move heavy processing to workers
5. **State Management**: Use Zustand/Redux for UI state
6. **Testing**: Port pytest tests to Jest/Vitest
7. **Optimization**: Profile and optimize bottlenecks
8. **Progressive Enhancement**: Start with basic features, add ML later

---

## See Also

- `browser-adaptations.md` - Browser-specific patterns
- `ml-model-migration.md` - ML model conversion
- `performance.md` - Performance optimization
- `compatibility.md` - Browser compatibility matrix
