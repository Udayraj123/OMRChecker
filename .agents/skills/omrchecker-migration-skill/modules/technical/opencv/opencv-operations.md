# OpenCV Usage Patterns in OMRChecker

**Module**: modules/technical/opencv/
**Status**: Complete
**Created**: 2026-02-20

---

## Overview

OMRChecker extensively uses OpenCV (cv2) for image processing operations. This document catalogs all OpenCV functions used, their parameters, edge cases, and provides browser migration guidance using OpenCV.js.

**Key Files**:
- `src/utils/image.py` - Main image utilities wrapper
- `src/utils/drawing.py` - Drawing utilities
- `src/processors/image/` - Image processing processors
- `src/processors/alignment/` - Alignment processors

---

## OpenCV.js Migration Strategy

### Loading OpenCV.js

```javascript
// Method 1: NPM Package
import cv from '@techstark/opencv-js';

// Method 2: CDN (fallback)
<script async src="https://docs.opencv.org/4.x/opencv.js"></script>

// Method 3: Self-hosted
<script async src="/static/opencv.js" onload="onOpenCvReady()"></script>
```

### Key Differences from Python OpenCV

| Python OpenCV | OpenCV.js | Notes |
|--------------|-----------|-------|
| `cv2.imread()` | `cv.imread(canvas)` | Use Canvas API or ImageData |
| NumPy arrays | `cv.Mat` objects | Must manage memory manually |
| `image.shape` | `mat.rows`, `mat.cols`, `mat.channels()` | Different shape access |
| Auto garbage collection | Manual `.delete()` | **Critical**: Prevent memory leaks |
| `cv2.IMREAD_GRAYSCALE` | `cv.IMREAD_GRAYSCALE` | Same constants available |

---

## Category 1: Image I/O

### 1.1 Reading Images

**Python**:
```python
# src/utils/image.py:25-50
image = cv2.imread(str(file_path), mode)

# Modes:
# - cv2.IMREAD_GRAYSCALE (0)
# - cv2.IMREAD_COLOR (1)
# - cv2.IMREAD_UNCHANGED (-1)
```

**Browser (OpenCV.js)**:
```javascript
// Use Canvas API
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
const img = new Image();

img.onload = function() {
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  // Load into OpenCV Mat
  const mat = cv.imread(canvas);

  // Convert to grayscale
  const gray = new cv.Mat();
  cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);

  // Clean up
  mat.delete();
};

img.src = fileURL;
```

**Browser (File API)**:
```javascript
async function loadImageFromFile(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const mat = cv.imread(canvas);
        resolve(mat);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}
```

**Edge Cases**:
- File not found → Returns `None` in Python, throws error in browser
- Invalid image format → Returns `None` in Python
- Large images → Memory constraints in browser (use Web Workers)

### 1.2 Writing Images

**Python**:
```python
# src/utils/image.py:65-66
cv2.imwrite(path, final_marked)
```

**Browser (OpenCV.js)**:
```javascript
// Write to Canvas, then download
function saveImage(mat, filename) {
  const canvas = document.createElement('canvas');
  cv.imshow(canvas, mat);

  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/jpeg', 0.95);
}
```

**Browser (Direct Canvas)**:
```javascript
// Display in existing canvas
cv.imshow('outputCanvas', mat);
```

---

## Category 2: Color Conversion

### 2.1 BGR to Grayscale

**Python**:
```python
# src/utils/image.py:57
gray_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
```

**Browser (OpenCV.js)**:
```javascript
const gray = new cv.Mat();
cv.cvtColor(colored, gray, cv.COLOR_RGBA2GRAY);
// Note: Browser uses RGBA, not BGR
```

**Migration Note**: Browser images are RGBA (not BGR like OpenCV), so use `COLOR_RGBA2GRAY` instead of `COLOR_BGR2GRAY`.

---

## Category 3: Geometric Transformations

### 3.1 Resize

**Python**:
```python
# src/utils/image.py:103
cv2.resize(image, (int(u_width), int(u_height)))
```

**Browser (OpenCV.js)**:
```javascript
const resized = new cv.Mat();
const dsize = new cv.Size(width, height);
cv.resize(src, resized, dsize, 0, 0, cv.INTER_LINEAR);
dsize.delete();
```

**Interpolation Methods**:
- `cv.INTER_LINEAR` - Default, good for most cases
- `cv.INTER_CUBIC` - Slower, higher quality
- `cv.INTER_AREA` - Best for shrinking
- `cv.INTER_NEAREST` - Fastest, lowest quality

### 3.2 Rotation

**Python**:
```python
# src/utils/image.py:447-452
cv2.rotate(image, rotation)

# Rotation constants:
# - cv2.ROTATE_90_CLOCKWISE
# - cv2.ROTATE_180
# - cv2.ROTATE_90_COUNTERCLOCKWISE
```

**Browser (OpenCV.js)**:
```javascript
const rotated = new cv.Mat();
cv.rotate(src, rotated, cv.ROTATE_90_CLOCKWISE);
```

### 3.3 Warp Perspective

**Python**:
```python
# src/utils/image_warp.py
M = cv2.getPerspectiveTransform(src_points, dst_points)
warped = cv2.warpPerspective(image, M, (width, height))
```

**Browser (OpenCV.js)**:
```javascript
// Create transformation matrix
const srcPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
  x1, y1, x2, y2, x3, y3, x4, y4
]);
const dstPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
  0, 0, width, 0, width, height, 0, height
]);

const M = cv.getPerspectiveTransform(srcPoints, dstPoints);
const warped = new cv.Mat();
const dsize = new cv.Size(width, height);

cv.warpPerspective(src, warped, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT);

// Clean up
srcPoints.delete();
dstPoints.delete();
M.delete();
dsize.delete();
```

**Edge Cases**:
- Invalid transformation matrix → Returns distorted image
- Out of bounds coordinates → Use border modes

### 3.4 Warp Affine

**Python**:
```python
M = cv2.getAffineTransform(src_points, dst_points)
warped = cv2.warpAffine(image, M, (width, height))
```

**Browser (OpenCV.js)**:
```javascript
const srcPoints = cv.matFromArray(3, 1, cv.CV_32FC2, [x1, y1, x2, y2, x3, y3]);
const dstPoints = cv.matFromArray(3, 1, cv.CV_32FC2, [0, 0, w, 0, 0, h]);

const M = cv.getAffineTransform(srcPoints, dstPoints);
const warped = new cv.Mat();
const dsize = new cv.Size(width, height);

cv.warpAffine(src, warped, M, dsize);

srcPoints.delete();
dstPoints.delete();
M.delete();
dsize.delete();
```

---

## Category 4: Image Filtering

### 4.1 Gaussian Blur

**Python**:
```python
# src/processors/image/GaussianBlur.py
cv2.GaussianBlur(image, (kernel_width, kernel_height), 0)
```

**Browser (OpenCV.js)**:
```javascript
const blurred = new cv.Mat();
const ksize = new cv.Size(kernelWidth, kernelHeight);
cv.GaussianBlur(src, blurred, ksize, 0, 0, cv.BORDER_DEFAULT);
ksize.delete();
```

**Common Kernel Sizes**: (3, 3), (5, 5), (7, 7), (9, 9)
**Note**: Kernel sizes must be odd numbers

### 4.2 Median Blur

**Python**:
```python
# src/processors/image/MedianBlur.py
cv2.medianBlur(image, kernel_size)
```

**Browser (OpenCV.js)**:
```javascript
const blurred = new cv.Mat();
cv.medianBlur(src, blurred, kernelSize);
```

**Edge Cases**:
- Kernel size must be odd
- Effective for salt-and-pepper noise
- Slower than Gaussian blur

### 4.3 Canny Edge Detection

**Python**:
```python
# src/utils/image.py:190-197
v = np.median(image)
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
cv2.Canny(image, lower, upper)
```

**Browser (OpenCV.js)**:
```javascript
// Auto Canny implementation
function autoCanny(src, sigma = 0.93) {
  // Calculate median (requires conversion to array)
  const data = src.data;
  const sorted = Array.from(data).sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];

  const lower = Math.max(0, (1.0 - sigma) * median);
  const upper = Math.min(255, (1.0 + sigma) * median);

  const edges = new cv.Mat();
  cv.Canny(src, edges, lower, upper);
  return edges;
}
```

---

## Category 5: Image Enhancement

### 5.1 Normalize

**Python**:
```python
# src/utils/image.py:174-178
cv2.normalize(image, None, alpha, beta, norm_type)

# Norm types:
# - cv2.NORM_MINMAX (default)
# - cv2.NORM_L1
# - cv2.NORM_L2
```

**Browser (OpenCV.js)**:
```javascript
const normalized = new cv.Mat();
cv.normalize(src, normalized, alpha, beta, cv.NORM_MINMAX, cv.CV_8U);
```

### 5.2 CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Python**:
```python
# src/utils/image.py:18
CLAHE_HELPER = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
enhanced = CLAHE_HELPER.apply(image)
```

**Browser (OpenCV.js)**:
```javascript
const clahe = new cv.CLAHE(5.0, new cv.Size(8, 8));
const enhanced = new cv.Mat();
clahe.apply(src, enhanced);
clahe.delete();
```

### 5.3 Gamma Correction

**Python**:
```python
# src/utils/image.py:202-211
inv_gamma = 1.0 / gamma
table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
cv2.LUT(image, table)
```

**Browser (OpenCV.js)**:
```javascript
function adjustGamma(src, gamma) {
  const invGamma = 1.0 / gamma;
  const table = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    table[i] = Math.pow(i / 255.0, invGamma) * 255;
  }

  const lut = cv.matFromArray(256, 1, cv.CV_8UC1, Array.from(table));
  const result = new cv.Mat();
  cv.LUT(src, lut, result);
  lut.delete();
  return result;
}
```

---

## Category 6: Contour Detection

### 6.1 Find Contours

**Python**:
```python
# src/utils/image.py:139-171
cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Handle different OpenCV versions
if len(cnts) == 2:
    cnts = cnts[0]  # OpenCV v2.4, v4-beta, v4-official
elif len(cnts) == 3:
    cnts = cnts[1]  # OpenCV v3, v4-pre, v4-alpha
```

**Browser (OpenCV.js)**:
```javascript
const contours = new cv.MatVector();
const hierarchy = new cv.Mat();

cv.findContours(
  src,
  contours,
  hierarchy,
  cv.RETR_EXTERNAL,
  cv.CHAIN_APPROX_SIMPLE
);

// Access contours
for (let i = 0; i < contours.size(); i++) {
  const contour = contours.get(i);
  // Process contour
  contour.delete();
}

contours.delete();
hierarchy.delete();
```

**Retrieval Modes**:
- `cv.RETR_EXTERNAL` - Only external contours
- `cv.RETR_LIST` - All contours, no hierarchy
- `cv.RETR_TREE` - Full hierarchy
- `cv.RETR_CCOMP` - Two-level hierarchy

**Approximation Methods**:
- `cv.CHAIN_APPROX_NONE` - All contour points
- `cv.CHAIN_APPROX_SIMPLE` - Compressed representation

### 6.2 Draw Contours

**Python**:
```python
# src/utils/drawing.py:44-62
cv2.drawContours(
    image,
    [np.intp(contour)],
    contourIdx=-1,
    color=color,
    thickness=thickness,
)
```

**Browser (OpenCV.js)**:
```javascript
const color = new cv.Scalar(0, 255, 0); // Green
cv.drawContours(image, contours, contourIdx, color, thickness, cv.LINE_8);
color.delete();
```

### 6.3 Contour Properties

**Python**:
```python
# Arc length
cv2.arcLength(contour, closed=True)

# Contour area
cv2.contourArea(contour)

# Bounding rectangle
x, y, w, h = cv2.boundingRect(contour)

# Approximate polygon
epsilon = 0.1 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)
```

**Browser (OpenCV.js)**:
```javascript
// Arc length
const arcLen = cv.arcLength(contour, true);

// Contour area
const area = cv.contourArea(contour);

// Bounding rectangle
const rect = cv.boundingRect(contour);
// rect = {x, y, width, height}

// Approximate polygon
const epsilon = 0.1 * cv.arcLength(contour, true);
const approx = new cv.Mat();
cv.approxPolyDP(contour, approx, epsilon, true);
approx.delete();
```

---

## Category 7: Drawing Operations

### 7.1 Rectangle

**Python**:
```python
# src/utils/drawing.py:35-41
cv2.rectangle(
    image,
    [int(position[0]), int(position[1])],
    [int(position_diagonal[0]), int(position_diagonal[1])],
    color,
    border,
)
```

**Browser (OpenCV.js)**:
```javascript
const pt1 = new cv.Point(x, y);
const pt2 = new cv.Point(x + w, y + h);
const color = new cv.Scalar(255, 0, 0); // Blue

cv.rectangle(image, pt1, pt2, color, thickness);

pt1.delete();
pt2.delete();
color.delete();
```

### 7.2 Line

**Python**:
```python
# src/utils/drawing.py:216-217
cv2.line(image, start, end, color, thickness)
```

**Browser (OpenCV.js)**:
```javascript
const pt1 = new cv.Point(startX, startY);
const pt2 = new cv.Point(endX, endY);
const color = new cv.Scalar(0, 0, 0);

cv.line(image, pt1, pt2, color, thickness, cv.LINE_AA);

pt1.delete();
pt2.delete();
color.delete();
```

### 7.3 Arrow

**Python**:
```python
# src/utils/drawing.py:130-138
cv2.arrowedLine(
    image,
    start_point,
    end_point,
    color,
    thickness,
    line_type,
    tipLength=tip_length,
)
```

**Browser (OpenCV.js)**:
```javascript
const pt1 = new cv.Point(startX, startY);
const pt2 = new cv.Point(endX, endY);
const color = new cv.Scalar(0, 255, 0);

cv.arrowedLine(image, pt1, pt2, color, thickness, cv.LINE_AA, 0, tipLength);

pt1.delete();
pt2.delete();
color.delete();
```

### 7.4 Text

**Python**:
```python
# src/utils/drawing.py:192-201
cv2.putText(
    image,
    text_value,
    position,
    font_face,
    text_size,
    color,
    thickness,
    lineType=line_type,
)
```

**Browser (OpenCV.js)**:
```javascript
const point = new cv.Point(x, y);
const color = new cv.Scalar(0, 0, 0);

cv.putText(
  image,
  text,
  point,
  cv.FONT_HERSHEY_SIMPLEX,
  fontSize,
  color,
  thickness,
  cv.LINE_AA
);

point.delete();
color.delete();
```

**Font Faces**:
- `cv.FONT_HERSHEY_SIMPLEX`
- `cv.FONT_HERSHEY_PLAIN`
- `cv.FONT_HERSHEY_DUPLEX`
- `cv.FONT_HERSHEY_COMPLEX`

---

## Category 8: Feature Detection (SIFT)

### 8.1 SIFT Features

**Python**:
```python
# src/processors/alignment/sift_matcher.py
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image, None)
```

**Browser (OpenCV.js)**:
```javascript
// SIFT is patented and may not be available in all OpenCV.js builds
// Use ORB as alternative

const orb = new cv.ORB();
const keypoints = new cv.KeyPointVector();
const descriptors = new cv.Mat();

orb.detectAndCompute(image, new cv.Mat(), keypoints, descriptors);

// Clean up
orb.delete();
keypoints.delete();
descriptors.delete();
```

**Alternative: ORB (Oriented FAST and Rotated BRIEF)**:
```javascript
const orb = new cv.ORB(500); // max 500 features
const kp1 = new cv.KeyPointVector();
const kp2 = new cv.KeyPointVector();
const desc1 = new cv.Mat();
const desc2 = new cv.Mat();

orb.detectAndCompute(img1, new cv.Mat(), kp1, desc1);
orb.detectAndCompute(img2, new cv.Mat(), kp2, desc2);

// Match features
const matcher = new cv.BFMatcher(cv.NORM_HAMMING, true);
const matches = new cv.DMatchVector();
matcher.match(desc1, desc2, matches);

// Clean up
orb.delete();
kp1.delete();
kp2.delete();
desc1.delete();
desc2.delete();
matcher.delete();
matches.delete();
```

---

## Category 9: Morphological Operations

### 9.1 Erosion & Dilation

**Python**:
```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded = cv2.erode(image, kernel, iterations=1)
dilated = cv2.dilate(image, kernel, iterations=1)
```

**Browser (OpenCV.js)**:
```javascript
const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
const eroded = new cv.Mat();
const dilated = new cv.Mat();
const anchor = new cv.Point(-1, -1);

cv.erode(src, eroded, kernel, anchor, 1, cv.BORDER_CONSTANT);
cv.dilate(src, dilated, kernel, anchor, 1, cv.BORDER_CONSTANT);

kernel.delete();
anchor.delete();
```

### 9.2 Morphological Transformations

**Python**:
```python
# Opening (erosion followed by dilation)
cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Closing (dilation followed by erosion)
cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
```

**Browser (OpenCV.js)**:
```javascript
const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
const result = new cv.Mat();

cv.morphologyEx(src, result, cv.MORPH_OPEN, kernel);
// or
cv.morphologyEx(src, result, cv.MORPH_CLOSE, kernel);

kernel.delete();
```

---

## Category 10: Border Operations

### 10.1 Copy Make Border (Padding)

**Python**:
```python
# src/utils/image.py:393-414
cv2.copyMakeBorder(
    image,
    top,
    bottom,
    left,
    right,
    cv2.BORDER_CONSTANT,
    value,
)
```

**Browser (OpenCV.js)**:
```javascript
const padded = new cv.Mat();
const color = new cv.Scalar(255, 255, 255); // White

cv.copyMakeBorder(
  src,
  padded,
  top,
  bottom,
  left,
  right,
  cv.BORDER_CONSTANT,
  color
);

color.delete();
```

**Border Types**:
- `cv.BORDER_CONSTANT` - Solid color
- `cv.BORDER_REPLICATE` - Repeat edge pixels
- `cv.BORDER_REFLECT` - Reflect border
- `cv.BORDER_WRAP` - Wrap around

---

## Category 11: Thresholding

### 11.1 Binary Threshold

**Python**:
```python
_, binary = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
```

**Browser (OpenCV.js)**:
```javascript
const binary = new cv.Mat();
cv.threshold(src, binary, threshValue, 255, cv.THRESH_BINARY);
```

### 11.2 Adaptive Threshold

**Python**:
```python
adaptive = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize,
    C
)
```

**Browser (OpenCV.js)**:
```javascript
const adaptive = new cv.Mat();
cv.adaptiveThreshold(
  src,
  adaptive,
  255,
  cv.ADAPTIVE_THRESH_GAUSSIAN_C,
  cv.THRESH_BINARY,
  blockSize,
  C
);
```

**Adaptive Methods**:
- `cv.ADAPTIVE_THRESH_MEAN_C` - Mean of neighborhood
- `cv.ADAPTIVE_THRESH_GAUSSIAN_C` - Gaussian weighted sum

---

## Category 12: Image Arithmetic

### 12.1 Add Weighted (Overlay)

**Python**:
```python
# src/utils/image.py:455-465
cv2.addWeighted(
    overlay,
    transparency,
    image2,
    1 - transparency,
    0,
    overlay,
)
```

**Browser (OpenCV.js)**:
```javascript
const overlay = new cv.Mat();
cv.addWeighted(img1, alpha, img2, beta, gamma, overlay);
```

### 12.2 Bitwise Operations

**Python**:
```python
# AND
result = cv2.bitwise_and(img1, img2)

# OR
result = cv2.bitwise_or(img1, img2)

# NOT
result = cv2.bitwise_not(img)
```

**Browser (OpenCV.js)**:
```javascript
const result = new cv.Mat();
cv.bitwise_and(img1, img2, result);
// or
cv.bitwise_or(img1, img2, result);
// or
cv.bitwise_not(img, result);
```

---

## Memory Management in OpenCV.js

### Critical Pattern

**Always delete Mat objects to prevent memory leaks**:

```javascript
function processImage(src) {
  const gray = new cv.Mat();
  const blurred = new cv.Mat();
  const edges = new cv.Mat();

  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
    cv.Canny(blurred, edges, 50, 150);

    return edges.clone(); // Return a clone
  } finally {
    // Clean up
    gray.delete();
    blurred.delete();
    edges.delete();
  }
}
```

### Helper Function

```javascript
class MatPool {
  constructor() {
    this.mats = [];
  }

  create(...args) {
    const mat = new cv.Mat(...args);
    this.mats.push(mat);
    return mat;
  }

  cleanup() {
    this.mats.forEach(mat => mat.delete());
    this.mats = [];
  }
}

// Usage
const pool = new MatPool();
try {
  const gray = pool.create();
  const blurred = pool.create();
  // ... processing
} finally {
  pool.cleanup();
}
```

---

## Performance Considerations

### 1. Web Workers for Heavy Processing

```javascript
// main.js
const worker = new Worker('opencv-worker.js');

worker.postMessage({
  type: 'process',
  imageData: imageData,
  operation: 'blur'
});

worker.onmessage = (e) => {
  const result = e.data.result;
  displayResult(result);
};
```

```javascript
// opencv-worker.js
importScripts('opencv.js');

cv['onRuntimeInitialized'] = () => {
  self.postMessage({ type: 'ready' });
};

self.onmessage = (e) => {
  const { type, imageData, operation } = e.data;

  if (type === 'process') {
    const src = cv.matFromImageData(imageData);
    const result = new cv.Mat();

    // Process based on operation
    if (operation === 'blur') {
      cv.GaussianBlur(src, result, new cv.Size(5, 5), 0);
    }

    // Convert back to ImageData
    const resultImageData = new ImageData(
      new Uint8ClampedArray(result.data),
      result.cols,
      result.rows
    );

    self.postMessage({ type: 'result', result: resultImageData });

    src.delete();
    result.delete();
  }
};
```

### 2. SharedArrayBuffer for Zero-Copy

```javascript
// Enable SharedArrayBuffer (requires COOP/COEP headers)
const sab = new SharedArrayBuffer(width * height * 4);
const imageData = new ImageData(new Uint8ClampedArray(sab), width, height);

// Share with worker
worker.postMessage({ sab, width, height });
```

### 3. OffscreenCanvas (Modern Browsers)

```javascript
const offscreen = canvas.transferControlToOffscreen();
worker.postMessage({ canvas: offscreen }, [offscreen]);
```

---

## Common Pitfalls

### 1. Color Space Differences

**Problem**: Python OpenCV uses BGR, browser uses RGBA
```javascript
// Wrong
cv.cvtColor(src, dst, cv.COLOR_BGR2GRAY);

// Correct
cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
```

### 2. Forgetting to Delete Objects

**Problem**: Memory leaks
```javascript
// Wrong
const mat = new cv.Mat();
cv.cvtColor(src, mat, cv.COLOR_RGBA2GRAY);
// mat never deleted!

// Correct
const mat = new cv.Mat();
try {
  cv.cvtColor(src, mat, cv.COLOR_RGBA2GRAY);
  // use mat
} finally {
  mat.delete();
}
```

### 3. Assuming Synchronous Operation

**Problem**: OpenCV.js loading is asynchronous
```javascript
// Wrong
import cv from 'opencv.js';
const mat = new cv.Mat(); // Error: cv not ready

// Correct
cv['onRuntimeInitialized'] = () => {
  const mat = new cv.Mat(); // Now safe
};
```

---

## Browser Compatibility

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| Basic OpenCV.js | ✅ | ✅ | ✅ | ✅ |
| WASM | ✅ | ✅ | ✅ (14+) | ✅ |
| Web Workers | ✅ | ✅ | ✅ | ✅ |
| SharedArrayBuffer | ✅ (COOP/COEP) | ✅ (COOP/COEP) | ⚠️ (limited) | ✅ (COOP/COEP) |
| OffscreenCanvas | ✅ | ✅ | ❌ | ✅ |

---

## Summary

**Total OpenCV Operations**: 368 usages across 49 Python files

**Key Categories**:
1. Image I/O (read, write)
2. Color conversion (BGR↔Gray, RGBA↔Gray)
3. Geometric transformations (resize, rotate, warp)
4. Filtering (blur, edge detection)
5. Enhancement (normalize, CLAHE, gamma)
6. Contours (find, draw, properties)
7. Drawing (rectangle, line, text)
8. Features (SIFT/ORB)
9. Morphology (erosion, dilation)
10. Borders (padding)
11. Thresholding (binary, adaptive)
12. Arithmetic (overlay, bitwise)

**Migration Strategy**:
- Use OpenCV.js for most operations
- Manage memory manually (always `.delete()`)
- Use Web Workers for heavy processing
- Handle color space differences (BGR→RGBA)
- Provide fallbacks for unsupported features (SIFT→ORB)
