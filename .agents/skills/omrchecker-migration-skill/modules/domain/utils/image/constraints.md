# Image Utils Constraints

## Performance Constraints

### Memory Management

**Python Behavior**:
- NumPy arrays share memory when possible (views)
- `image.copy()` creates deep copy
- OpenCV operations may modify in-place or return new array
- Garbage collection handles cleanup automatically

**Browser Constraints**:
```typescript
// CRITICAL: Manual memory management required
const image = cv.imread(file);
const processed = new cv.Mat();

// Use the image...
cv.cvtColor(image, processed, cv.COLOR_BGR2GRAY);

// MUST delete when done
image.delete();
processed.delete();
```

**Memory Leak Prevention**:
```typescript
class ImageProcessor {
  private mats: cv.Mat[] = [];

  createMat(...args: any[]): cv.Mat {
    const mat = new cv.Mat(...args);
    this.mats.push(mat);
    return mat;
  }

  cleanup(): void {
    this.mats.forEach(mat => mat.delete());
    this.mats = [];
  }
}

// Usage
const processor = new ImageProcessor();
try {
  const result = processor.process(image);
  // Use result...
} finally {
  processor.cleanup();
}
```

### Image Size Limits

**Python**:
- Limited by available RAM (typically 4-16GB)
- Can process 10MP+ images easily
- Batch processing supported

**Browser**:
- **Chrome/Edge**: ~2GB per tab (may vary)
- **Firefox**: ~1.5GB per tab
- **Safari**: ~1GB per tab
- **Mobile**: 512MB - 1GB per tab

**Recommendations**:
```typescript
// Check image size before processing
function validateImageSize(width: number, height: number): void {
  const pixelCount = width * height;
  const maxPixels = 4096 * 4096; // 16MP limit for safety

  if (pixelCount > maxPixels) {
    throw new Error(
      `Image too large: ${width}x${height} (${pixelCount} pixels). ` +
      `Maximum: ${maxPixels} pixels. Consider downsampling.`
    );
  }
}

// Auto-resize large images
function ensureSafeImageSize(mat: cv.Mat, maxDim = 4096): cv.Mat {
  const [h, w] = [mat.rows, mat.cols];

  if (w > maxDim || h > maxDim) {
    const scale = maxDim / Math.max(w, h);
    const newSize = new cv.Size(
      Math.floor(w * scale),
      Math.floor(h * scale)
    );

    const resized = new cv.Mat();
    cv.resize(mat, resized, newSize);
    mat.delete();
    return resized;
  }

  return mat;
}
```

### Resize Performance

**Python Performance**:
- `cv2.resize()` is highly optimized (SIMD, multithreading)
- Can resize 4K image in 10-50ms

**Browser Performance**:
```typescript
// OpenCV.js resize performance
// 1920x1080 → 640x480: ~20-50ms (depends on browser)
// 3840x2160 → 1920x1080: ~80-200ms

// Optimization: Use Web Workers for large images
async function resizeInWorker(
  imageData: ImageData,
  width: number,
  height: number
): Promise<ImageData> {
  const worker = new Worker('/workers/image-processor.js');

  return new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      resolve(e.data.result);
      worker.terminate();
    };

    worker.postMessage({
      type: 'resize',
      imageData,
      width,
      height
    });
  });
}
```

### Warping Performance

**Triangle Warping Complexity**:
- Python: O(n) where n = number of triangles
- Each triangle: `getAffineTransform` + `warpAffine` + mask operations
- Typical use: 50-200 triangles for page warping

**Browser Optimization**:
```typescript
// Use OffscreenCanvas in Worker for better performance
// worker.js
self.onmessage = (e) => {
  const { triangles, image } = e.data;

  // Load OpenCV.js in worker
  cv['onRuntimeInitialized'] = () => {
    const mat = cv.matFromImageData(image);
    const result = new cv.Mat();

    // Process all triangles
    for (const triangle of triangles) {
      warpTriangleInplace(mat, result, triangle.source, triangle.warped);
    }

    const resultData = new ImageData(
      new Uint8ClampedArray(result.data),
      result.cols,
      result.rows
    );

    mat.delete();
    result.delete();

    self.postMessage({ result: resultData });
  };
};
```

---

## File API Browser Migration

### Reading Images from File

**Python Approach**:
```python
# Read from disk
image = ImageUtils.load_image(Path("input.jpg"))
```

**Browser Approach**:
```typescript
// Method 1: File input
async function loadImageFromFile(file: File): Promise<cv.Mat> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;

        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const mat = cv.matFromImageData(imageData);

        resolve(mat);
      };
      img.src = e.target!.result as string;
    };

    reader.readAsDataURL(file);
  });
}

// Method 2: Drag and drop
<div
  id="dropzone"
  onDrop={(e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    const imageFiles = files.filter(f => f.type.startsWith('image/'));

    imageFiles.forEach(async (file) => {
      const mat = await loadImageFromFile(file);
      // Process mat...
      mat.delete();
    });
  }}
  onDragOver={(e) => e.preventDefault()}
>
  Drop images here
</div>
```

### Saving Images to File

**Python Approach**:
```python
# Write to disk
ImageUtils.save_img("output.png", image)
```

**Browser Approach**:
```typescript
// Download as file (browser can't write to arbitrary paths)
function downloadImage(mat: cv.Mat, filename: string): void {
  // Convert Mat to canvas
  const canvas = document.createElement('canvas');
  cv.imshow(canvas, mat);

  // Trigger download
  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob!);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
}

// Alternative: Save to IndexedDB for later
async function saveImageToIndexedDB(
  mat: cv.Mat,
  key: string
): Promise<void> {
  const canvas = document.createElement('canvas');
  cv.imshow(canvas, mat);

  const blob = await new Promise<Blob>((resolve) => {
    canvas.toBlob((b) => resolve(b!), 'image/png');
  });

  // Store in IndexedDB
  const db = await openDB('omr-images', 1, {
    upgrade(db) {
      db.createObjectStore('images');
    }
  });

  await db.put('images', blob, key);
}

// Retrieve later
async function loadImageFromIndexedDB(key: string): Promise<cv.Mat> {
  const db = await openDB('omr-images', 1);
  const blob = await db.get('images', key);

  const img = new Image();
  img.src = URL.createObjectURL(blob);
  await new Promise((resolve) => { img.onload = resolve; });

  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return cv.matFromImageData(imageData);
}
```

### Batch Processing

**Python Approach**:
```python
# Process directory of images
for image_path in Path("inputs").glob("*.jpg"):
    image = ImageUtils.load_image(image_path)
    # Process...
```

**Browser Approach**:
```typescript
// Multiple file input
<input
  type="file"
  multiple
  accept="image/*"
  onChange={async (e) => {
    const files = Array.from(e.target.files);

    // Process with progress tracking
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const mat = await loadImageFromFile(file);

      // Process mat...
      const result = processOMR(mat);

      // Download result
      downloadImage(result, `result_${file.name}`);

      // Update progress
      updateProgress((i + 1) / files.length);

      // Cleanup
      mat.delete();
      result.delete();
    }
  }}
/>

// Or use FileSystemAccess API (modern browsers)
async function processDirectory(): Promise<void> {
  const dirHandle = await window.showDirectoryPicker();

  for await (const entry of dirHandle.values()) {
    if (entry.kind === 'file' && entry.name.match(/\.(jpg|png|jpeg)$/i)) {
      const file = await entry.getFile();
      const mat = await loadImageFromFile(file);
      // Process...
      mat.delete();
    }
  }
}
```

---

## Format Conversion Constraints

### Color Space Conversions

**Python**:
```python
# BGR ↔ Grayscale
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# BGR ↔ RGB
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
```

**Browser**:
```typescript
// OpenCV.js uses RGBA (not BGR) from canvas
const gray = new cv.Mat();
cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);

const colored = new cv.Mat();
cv.cvtColor(gray, colored, cv.COLOR_GRAY2RGBA);

// Canvas ImageData is always RGBA
// No need for RGB ↔ BGR conversion
```

### Image Channels

**Python Behavior**:
- Grayscale: `shape = (H, W)`, 2D array
- Color: `shape = (H, W, 3)`, 3D array (BGR)
- Alpha: `shape = (H, W, 4)`, 3D array (BGRA)

**Browser Behavior**:
```typescript
// OpenCV.js Mat properties
interface Mat {
  rows: number;      // Height
  cols: number;      // Width
  channels(): number; // 1 (gray), 3 (RGB), 4 (RGBA)
  type(): number;    // cv.CV_8UC1, cv.CV_8UC3, cv.CV_8UC4
}

// Check channels
function getChannels(mat: cv.Mat): number {
  return mat.channels();
}

// Ensure grayscale
function ensureGrayscale(mat: cv.Mat): cv.Mat {
  if (mat.channels() === 1) {
    return mat;
  }

  const gray = new cv.Mat();
  const code = mat.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY;
  cv.cvtColor(mat, gray, code);
  mat.delete();
  return gray;
}
```

---

## Edge Detection Constraints

### Auto Canny Performance

**Python**:
```python
# Fast: np.median() is optimized
v = np.median(image)
edges = cv2.Canny(image, lower, upper)
```

**Browser**:
```typescript
// Median calculation is slow for large images
// Optimization: Use approximate median
function fastMedian(mat: cv.Mat): number {
  // Sample pixels instead of computing exact median
  const sampleSize = Math.min(10000, mat.rows * mat.cols);
  const step = Math.floor(mat.rows * mat.cols / sampleSize);

  const samples: number[] = [];
  const data = mat.data;

  for (let i = 0; i < data.length; i += step) {
    samples.push(data[i]);
  }

  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)];
}

// Or use histogram-based median (faster for 8-bit images)
function histogramMedian(mat: cv.Mat): number {
  const hist = new cv.Mat();
  cv.calcHist(
    new cv.MatVector([mat]),
    [0],
    new cv.Mat(),
    hist,
    [256],
    [0, 256]
  );

  const totalPixels = mat.rows * mat.cols;
  let sum = 0;

  for (let i = 0; i < 256; i++) {
    sum += hist.data32F[i];
    if (sum >= totalPixels / 2) {
      hist.delete();
      return i;
    }
  }

  hist.delete();
  return 128; // Fallback
}
```

---

## Warping Constraints

### Collinear Point Detection

**Python**:
```python
# Uses MathUtils.check_collinear_points()
if MathUtils.check_collinear_points(*triangle):
    logger.critical("Collinear points detected")
    return  # Skip warp
```

**Browser**:
```typescript
function checkCollinearPoints(p1: Point, p2: Point, p3: Point): boolean {
  // Cross product method
  // If cross product is 0, points are collinear
  const crossProduct = (p2.x - p1.x) * (p3.y - p1.y) -
                       (p2.y - p1.y) * (p3.x - p1.x);

  // Allow small tolerance for floating-point errors
  const tolerance = 1e-6;
  return Math.abs(crossProduct) < tolerance;
}
```

### Affine Transform Limitations

**Key Constraint**: Affine transform requires exactly 3 non-collinear points

**Error Cases**:
```typescript
// Invalid: Collinear points
const invalid = [
  [0, 0],
  [100, 100],  // On line from [0,0] to [200,200]
  [200, 200]
];
// cv.getAffineTransform() will fail or produce undefined results

// Valid: Non-collinear points
const valid = [
  [0, 0],
  [100, 0],
  [0, 100]
];
```

**Validation**:
```typescript
function validateAffinePoints(points: Point[]): void {
  if (points.length !== 3) {
    throw new Error('Affine transform requires exactly 3 points');
  }

  if (checkCollinearPoints(points[0], points[1], points[2])) {
    throw new Error('Points must not be collinear');
  }
}
```

---

## Grid Layout Constraints

### Stack Size Limits

**Python**:
```python
# Can stack many images (limited by RAM)
grid = ImageUtils.get_vstack_image_grid([
    [img1, img2, ..., img20],  # 20 images in a row
    # ... many rows
])
```

**Browser**:
```typescript
// Canvas size limits
const MAX_CANVAS_DIMENSION = {
  Chrome: 32767,
  Firefox: 32767,
  Safari: 16384,  // Smallest limit
  Edge: 32767
};

// Validate before stacking
function validateGridSize(images: cv.Mat[][]): void {
  let totalWidth = 0;
  let totalHeight = 0;

  for (const row of images) {
    const rowHeight = Math.max(...row.map(img => img.rows));
    totalHeight += rowHeight;

    const rowWidth = row.reduce((sum, img) => sum + img.cols, 0);
    totalWidth = Math.max(totalWidth, rowWidth);
  }

  const maxDim = 16384; // Use Safari's limit for compatibility

  if (totalWidth > maxDim || totalHeight > maxDim) {
    throw new Error(
      `Grid too large: ${totalWidth}x${totalHeight}. ` +
      `Maximum: ${maxDim}x${maxDim}`
    );
  }
}
```

### Memory Considerations for Grids

**Issue**: Stacking many debug images can cause OOM

**Solution**:
```typescript
// Downscale debug images before stacking
function createDebugGrid(images: cv.Mat[][], maxDim = 2048): cv.Mat {
  // Calculate scale factor
  let maxWidth = 0;
  let maxHeight = 0;

  for (const row of images) {
    maxHeight += Math.max(...row.map(img => img.rows));
    maxWidth = Math.max(maxWidth, row.reduce((sum, img) => sum + img.cols, 0));
  }

  const scale = Math.min(1, maxDim / Math.max(maxWidth, maxHeight));

  // Resize all images
  const scaledImages = images.map(row =>
    row.map(img => {
      if (scale >= 1) return img;

      const scaled = new cv.Mat();
      cv.resize(img, scaled, new cv.Size(
        Math.floor(img.cols * scale),
        Math.floor(img.rows * scale)
      ));
      return scaled;
    })
  );

  // Create grid
  const grid = createVStackImageGrid(scaledImages);

  // Cleanup scaled images if they're different from originals
  if (scale < 1) {
    scaledImages.forEach(row => row.forEach(img => img.delete()));
  }

  return grid;
}
```

---

## Rotation Constraints

### keep_original_shape Behavior

**Python**:
```python
# Without keep_original_shape (default)
rotated = ImageUtils.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# If input is 1920x1080, output is 1080x1920

# With keep_original_shape
rotated = ImageUtils.rotate(image, cv2.ROTATE_90_CLOCKWISE, keep_original_shape=True)
# Output remains 1920x1080 (cropped/padded as needed)
```

**Browser Implications**:
```typescript
// Without preserve: Dimensions swap
function rotate90CW(mat: cv.Mat): cv.Mat {
  const rotated = new cv.Mat();
  cv.rotate(mat, rotated, cv.ROTATE_90_CLOCKWISE);
  // Input: 1920x1080 → Output: 1080x1920
  return rotated;
}

// With preserve: May lose content at edges
function rotate90CWPreserve(mat: cv.Mat): cv.Mat {
  const rotated = rotate90CW(mat);

  const resized = new cv.Mat();
  cv.resize(rotated, resized, new cv.Size(mat.cols, mat.rows));
  rotated.delete();

  // Warning: This will distort the image!
  // Original: 1920x1080 → Rotated: 1080x1920 → Forced back: 1920x1080
  return resized;
}
```

**Recommendation**: Avoid `keep_original_shape` in browser to prevent unexpected distortion

---

## Gamma Adjustment Constraints

### Lookup Table Performance

**Python**:
```python
# LUT creation is fast
table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)])
adjusted = cv2.LUT(image, table)
```

**Browser**:
```typescript
// Cache LUT for repeated use
class GammaAdjuster {
  private lutCache = new Map<number, cv.Mat>();

  adjustGamma(image: cv.Mat, gamma: number): cv.Mat {
    let lut = this.lutCache.get(gamma);

    if (!lut) {
      const invGamma = 1.0 / gamma;
      const table = new Uint8Array(256);

      for (let i = 0; i < 256; i++) {
        table[i] = Math.floor(Math.pow(i / 255.0, invGamma) * 255);
      }

      lut = cv.matFromArray(1, 256, cv.CV_8UC1, Array.from(table));
      this.lutCache.set(gamma, lut);
    }

    const adjusted = new cv.Mat();
    cv.LUT(image, lut, adjusted);
    return adjusted;
  }

  cleanup(): void {
    this.lutCache.forEach(lut => lut.delete());
    this.lutCache.clear();
  }
}
```

---

## Normalization Constraints

### Edge Case: No Contrast

**Python**:
```python
# Returns original if max == min
if image.max() == image.min():
    return image
```

**Browser**:
```typescript
// minMaxLoc is expensive for large images
// Optimization: Sample-based check
function hasContrast(mat: cv.Mat): boolean {
  const minMax = cv.minMaxLoc(mat);
  return minMax.maxVal !== minMax.minVal;
}

// For very large images, sample approach:
function hasContrastFast(mat: cv.Mat): boolean {
  const data = mat.data;
  const sampleSize = Math.min(1000, data.length);
  const step = Math.floor(data.length / sampleSize);

  let min = 255, max = 0;

  for (let i = 0; i < data.length; i += step) {
    min = Math.min(min, data[i]);
    max = Math.max(max, data[i]);

    if (max > min) return true; // Early exit
  }

  return false;
}
```

---

## Contour Utilities Constraints

### grab_contours Compatibility

**Python Issue**: OpenCV version inconsistency

**Browser**: OpenCV.js is consistent (always returns `{ contours, hierarchy }`)

```typescript
// No version checking needed in browser
function grabContours(result: any): cv.MatVector {
  return result.contours; // Always same structure
}
```

### Shapely Dependency

**Python**:
```python
from shapely import LineString, Point

# Use for geometric calculations
distance = Point(p).distance(LineString(edge))
```

**Browser Alternative**:
```typescript
// Implement basic geometric functions
function pointToLineDistance(
  point: Point,
  lineStart: Point,
  lineEnd: Point
): number {
  // Use perpendicular distance formula
  const dx = lineEnd.x - lineStart.x;
  const dy = lineEnd.y - lineStart.y;
  const lineLength = Math.sqrt(dx * dx + dy * dy);

  if (lineLength === 0) {
    // Line is a point
    return Math.sqrt(
      (point.x - lineStart.x) ** 2 +
      (point.y - lineStart.y) ** 2
    );
  }

  const numerator = Math.abs(
    dy * point.x - dx * point.y +
    lineEnd.x * lineStart.y - lineEnd.y * lineStart.x
  );

  return numerator / lineLength;
}

// Or use lightweight library like @turf/turf
import { point, lineString, pointToLineDistance } from '@turf/turf';

const distance = pointToLineDistance(
  point([p.x, p.y]),
  lineString([[start.x, start.y], [end.x, end.y]])
);
```

---

## CLAHE Constraints

### Global Instance

**Python**:
```python
# Single global instance
CLAHE_HELPER = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))

# Use anywhere
enhanced = CLAHE_HELPER.apply(image)
```

**Browser**:
```typescript
// Create and cache CLAHE instance
class ImageEnhancer {
  private static clahe: cv.CLAHE | null = null;

  static getCLAHE(): cv.CLAHE {
    if (!this.clahe) {
      this.clahe = new cv.CLAHE();
      this.clahe.setClipLimit(5.0);
      this.clahe.setTilesGridSize(new cv.Size(8, 8));
    }
    return this.clahe;
  }

  static enhance(image: cv.Mat): cv.Mat {
    const enhanced = new cv.Mat();
    this.getCLAHE().apply(image, enhanced);
    return enhanced;
  }

  static cleanup(): void {
    if (this.clahe) {
      this.clahe.delete();
      this.clahe = null;
    }
  }
}
```

---

## Progressive Enhancement Strategy

### Feature Detection

```typescript
class ImageUtilsCapabilities {
  static hasOpenCV(): boolean {
    return typeof cv !== 'undefined' && cv.Mat !== undefined;
  }

  static hasWebAssembly(): boolean {
    return typeof WebAssembly !== 'undefined';
  }

  static hasWorkers(): boolean {
    return typeof Worker !== 'undefined';
  }

  static hasOffscreenCanvas(): boolean {
    return typeof OffscreenCanvas !== 'undefined';
  }

  static getMaxImageSize(): number {
    // Conservative estimate based on browser
    const ua = navigator.userAgent;
    if (ua.includes('Safari') && !ua.includes('Chrome')) {
      return 16384;
    }
    return 32767;
  }
}

// Use capabilities
if (!ImageUtilsCapabilities.hasOpenCV()) {
  throw new Error('OpenCV.js not loaded');
}

if (ImageUtilsCapabilities.hasWorkers()) {
  // Use worker-based processing
} else {
  // Fallback to main thread
}
```

### Fallback Strategies

```typescript
class ImageUtils {
  static async loadImage(file: File): Promise<cv.Mat> {
    // Try OpenCV.js first
    if (ImageUtilsCapabilities.hasOpenCV()) {
      return this.loadImageWithOpenCV(file);
    }

    // Fallback to canvas
    return this.loadImageWithCanvas(file);
  }

  private static async loadImageWithCanvas(file: File): Promise<cv.Mat> {
    // Minimal processing without OpenCV
    const img = await this.fileToImage(file);
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;

    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0);

    // Return as ImageData (can't do OpenCV operations)
    return ctx.getImageData(0, 0, canvas.width, canvas.height) as any;
  }
}
```

---

## Best Practices Summary

1. **Memory Management**: Always delete cv.Mat objects when done
2. **Size Validation**: Check image dimensions before processing
3. **Worker Offloading**: Use Web Workers for intensive operations
4. **Cache LUTs**: Reuse lookup tables for gamma/normalization
5. **Progressive Enhancement**: Detect capabilities and fallback gracefully
6. **File API**: Use FileReader for input, download/IndexedDB for output
7. **Canvas Limits**: Respect browser-specific canvas size limits
8. **Channel Handling**: Remember browser uses RGBA, not BGR
9. **Error Handling**: Validate affine transform points (non-collinear)
10. **Performance Monitoring**: Track memory usage and processing time

## Migration Checklist

- [ ] Replace `Path` with `File` API
- [ ] Add `.delete()` calls for all cv.Mat objects
- [ ] Validate image sizes before processing
- [ ] Implement worker-based processing for large operations
- [ ] Cache reusable objects (CLAHE, LUTs)
- [ ] Handle RGBA ↔ Grayscale conversions
- [ ] Replace file writes with downloads/IndexedDB
- [ ] Add canvas size validation
- [ ] Implement progress tracking for batch operations
- [ ] Add error boundaries for OOM handling
