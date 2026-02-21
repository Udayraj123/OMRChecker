# NumPy Array Operations in OMRChecker

**Module**: modules/technical/numpy/
**Status**: Complete
**Created**: 2026-02-20

---

## Overview

OMRChecker uses NumPy extensively for array operations, particularly for image data manipulation, coordinate transformations, and mathematical computations. This document catalogs all NumPy patterns used and provides browser migration guidance using TypedArrays and ndarray libraries.

---

## Browser Migration Strategy

### Option 1: TypedArrays (Native)

For simple operations, use native JavaScript TypedArrays:
- `Uint8Array` - 8-bit unsigned integers (image data)
- `Float32Array` - 32-bit floats (coordinates, transformations)
- `Int32Array` - 32-bit signed integers

### Option 2: ndarray.js

For complex operations:
```bash
npm install ndarray ndarray-ops
```

### Option 3: TensorFlow.js

For mathematical operations:
```bash
npm install @tensorflow/tfjs
```

---

## Category 1: Array Creation

### 1.1 Zeros

**Python**:
```python
import numpy as np
zeros = np.zeros((height, width), np.uint8)
zeros_3d = np.zeros((height, width, 3), np.uint8)
```

**Browser (TypedArray)**:
```javascript
// 2D grayscale
const zeros = new Uint8Array(height * width);

// 3D color (interleaved)
const zeros3d = new Uint8Array(height * width * 3);
```

**Browser (ndarray)**:
```javascript
import ndarray from 'ndarray';
import zeros from 'ndarray-scratch';

const zeroArray = zeros([height, width], 'uint8');
```

### 1.2 Ones

**Python**:
```python
ones = np.ones((height, width), np.uint8)
white_image = 255 * np.ones((height, width), np.uint8)
```

**Browser (TypedArray)**:
```javascript
const ones = new Uint8Array(height * width).fill(1);
const whiteImage = new Uint8Array(height * width).fill(255);
```

### 1.3 From Array

**Python**:
```python
# src/utils/image.py:126-134
arr = np.array([
    [0, 0],
    [max_width - 1, 0],
    [max_width - 1, max_height - 1],
    [0, max_height - 1],
], dtype="float32")
```

**Browser**:
```javascript
const arr = new Float32Array([
  0, 0,
  maxWidth - 1, 0,
  maxWidth - 1, maxHeight - 1,
  0, maxHeight - 1
]);

// Or as 2D representation
const points = [
  [0, 0],
  [maxWidth - 1, 0],
  [maxWidth - 1, maxHeight - 1],
  [0, maxHeight - 1]
];
```

### 1.4 Arange

**Python**:
```python
# src/utils/image.py:206
table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
```

**Browser**:
```javascript
const table = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
  table[i] = Math.pow(i / 255.0, invGamma) * 255;
}

// Or using Array methods
const table = Array.from({ length: 256 }, (_, i) =>
  Math.pow(i / 255.0, invGamma) * 255
);
```

---

## Category 2: Array Reshaping

### 2.1 Reshape

**Python**:
```python
# Flatten 2D to 1D
flat = image.reshape(-1)

# Reshape to specific shape
reshaped = arr.reshape(height, width, channels)
```

**Browser (Manual)**:
```javascript
// TypedArrays are already flat
const flat = imageData; // Already 1D

// Access as 2D
function get2D(arr, width, row, col) {
  return arr[row * width + col];
}

function set2D(arr, width, row, col, value) {
  arr[row * width + col] = value;
}
```

**Browser (ndarray)**:
```javascript
import ndarray from 'ndarray';

const flat = ndarray(data, [height * width]);
const reshaped = ndarray(data, [height, width, channels]);
```

### 2.2 Transpose

**Python**:
```python
transposed = arr.T
```

**Browser (ndarray)**:
```javascript
import { transpose } from 'ndarray-ops';

const transposed = ndarray(new Float32Array(rows * cols), [cols, rows]);
transpose(transposed, original);
```

---

## Category 3: Array Indexing & Slicing

### 3.1 Basic Indexing

**Python**:
```python
# Get pixel value
value = image[y, x]

# Set pixel value
image[y, x] = 255

# Get row
row = image[y, :]

# Get column
col = image[:, x]
```

**Browser**:
```javascript
// Get pixel (grayscale)
function getPixel(image, width, y, x) {
  return image[y * width + x];
}

// Set pixel (grayscale)
function setPixel(image, width, y, x, value) {
  image[y * width + x] = value;
}

// Get row
function getRow(image, width, height, y) {
  return image.slice(y * width, (y + 1) * width);
}

// Get column
function getColumn(image, width, height, x) {
  const col = new Uint8Array(height);
  for (let y = 0; y < height; y++) {
    col[y] = image[y * width + x];
  }
  return col;
}
```

### 3.2 ROI (Region of Interest)

**Python**:
```python
roi = image[y1:y2, x1:x2]
```

**Browser**:
```javascript
function extractROI(image, width, x1, y1, x2, y2) {
  const roiWidth = x2 - x1;
  const roiHeight = y2 - y1;
  const roi = new Uint8Array(roiWidth * roiHeight);

  for (let y = 0; y < roiHeight; y++) {
    for (let x = 0; x < roiWidth; x++) {
      const srcIdx = (y1 + y) * width + (x1 + x);
      const dstIdx = y * roiWidth + x;
      roi[dstIdx] = image[srcIdx];
    }
  }

  return roi;
}
```

---

## Category 4: Mathematical Operations

### 4.1 Element-wise Operations

**Python**:
```python
# Addition
result = arr1 + arr2

# Subtraction
result = arr1 - arr2

# Multiplication
result = arr * scalar

# Division
result = arr / scalar
```

**Browser (TypedArray)**:
```javascript
// Addition
const result = new Uint8Array(arr1.length);
for (let i = 0; i < arr1.length; i++) {
  result[i] = arr1[i] + arr2[i];
}

// Scalar multiplication
const result = arr.map(v => v * scalar);
```

**Browser (ndarray-ops)**:
```javascript
import { add, sub, muls, divs } from 'ndarray-ops';

add(result, arr1, arr2);
sub(result, arr1, arr2);
muls(result, arr, scalar);
divs(result, arr, scalar);
```

### 4.2 Statistical Operations

**Python**:
```python
# src/utils/image.py:192
median = np.median(image)
mean = np.mean(image)
std = np.std(image)
minimum = np.min(image)
maximum = np.max(image)
```

**Browser**:
```javascript
// Median
function median(arr) {
  const sorted = Array.from(arr).sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Mean
function mean(arr) {
  return arr.reduce((sum, val) => sum + val, 0) / arr.length;
}

// Standard deviation
function std(arr) {
  const m = mean(arr);
  const variance = arr.reduce((sum, val) => sum + (val - m) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}

// Min/Max
const minimum = Math.min(...arr);
const maximum = Math.max(...arr);
```

### 4.3 Matrix Operations

**Python**:
```python
# Dot product
result = np.dot(matrix1, matrix2)

# Matrix multiplication
result = matrix1 @ matrix2
```

**Browser (Simple)**:
```javascript
function matrixMultiply(a, b, rowsA, colsA, colsB) {
  const result = new Float32Array(rowsA * colsB);

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        sum += a[i * colsA + k] * b[k * colsB + j];
      }
      result[i * colsB + j] = sum;
    }
  }

  return result;
}
```

**Browser (TensorFlow.js)**:
```javascript
import * as tf from '@tensorflow/tfjs';

const a = tf.tensor2d([[1, 2], [3, 4]]);
const b = tf.tensor2d([[5, 6], [7, 8]]);
const result = tf.matMul(a, b);
```

---

## Category 5: Array Manipulation

### 5.1 Concatenation

**Python**:
```python
# Horizontal stack
hstack = np.hstack([img1, img2])

# Vertical stack
vstack = np.vstack([img1, img2])
```

**Browser**:
```javascript
// Horizontal stack (same height)
function hstack(img1, img2, width1, width2, height) {
  const newWidth = width1 + width2;
  const result = new Uint8Array(newWidth * height);

  for (let y = 0; y < height; y++) {
    // Copy from img1
    for (let x = 0; x < width1; x++) {
      result[y * newWidth + x] = img1[y * width1 + x];
    }
    // Copy from img2
    for (let x = 0; x < width2; x++) {
      result[y * newWidth + width1 + x] = img2[y * width2 + x];
    }
  }

  return result;
}

// Vertical stack (same width)
function vstack(img1, img2, width, height1, height2) {
  const result = new Uint8Array(width * (height1 + height2));

  // Copy img1
  result.set(img1, 0);
  // Copy img2
  result.set(img2, width * height1);

  return result;
}
```

### 5.2 Flip/Reverse

**Python**:
```python
# Flip vertically
flipped = np.flip(image, axis=0)

# Flip horizontally
flipped = np.flip(image, axis=1)
```

**Browser**:
```javascript
// Flip vertically
function flipVertical(image, width, height) {
  const flipped = new Uint8Array(width * height);

  for (let y = 0; y < height; y++) {
    const srcRow = (height - 1 - y) * width;
    const dstRow = y * width;
    flipped.set(image.slice(srcRow, srcRow + width), dstRow);
  }

  return flipped;
}

// Flip horizontally
function flipHorizontal(image, width, height) {
  const flipped = new Uint8Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const srcIdx = y * width + (width - 1 - x);
      const dstIdx = y * width + x;
      flipped[dstIdx] = image[srcIdx];
    }
  }

  return flipped;
}
```

---

## Category 6: Boolean Operations

### 6.1 Comparison

**Python**:
```python
mask = image > threshold
mask = (image >= min_val) & (image <= max_val)
```

**Browser**:
```javascript
// Greater than
const mask = new Uint8Array(image.length);
for (let i = 0; i < image.length; i++) {
  mask[i] = image[i] > threshold ? 1 : 0;
}

// Range check
const mask = new Uint8Array(image.length);
for (let i = 0; i < image.length; i++) {
  mask[i] = (image[i] >= minVal && image[i] <= maxVal) ? 1 : 0;
}
```

### 6.2 Where (Conditional Selection)

**Python**:
```python
result = np.where(condition, value_if_true, value_if_false)
```

**Browser**:
```javascript
function where(condition, valueIfTrue, valueIfFalse) {
  const result = new Uint8Array(condition.length);

  for (let i = 0; i < condition.length; i++) {
    result[i] = condition[i] ? valueIfTrue : valueIfFalse;
  }

  return result;
}
```

---

## Category 7: Data Type Conversion

### 7.1 astype

**Python**:
```python
float_arr = int_arr.astype("float32")
int_arr = float_arr.astype("uint8")
```

**Browser**:
```javascript
// uint8 to float32
const float32Arr = new Float32Array(uint8Arr);

// float32 to uint8 (with clamping)
const uint8Arr = new Uint8Array(float32Arr.length);
for (let i = 0; i < float32Arr.length; i++) {
  uint8Arr[i] = Math.max(0, Math.min(255, Math.round(float32Arr[i])));
}
```

---

## Category 8: Image-Specific Operations

### 8.1 White Image Creation

**Python**:
```python
# src/utils/image.py:426-430
white_image = 255 * np.ones(
    (padding_height * 2 + input_height, padding_width * 2 + input_width),
    np.uint8,
)
```

**Browser**:
```javascript
function createWhiteImage(width, height) {
  const size = width * height;
  const image = new Uint8Array(size);
  image.fill(255);
  return image;
}
```

### 8.2 Padding from Center

**Python**:
```python
# src/utils/image.py:417-432
white_image = 255 * np.ones((h_padded, w_padded), np.uint8)
white_image[y1:y2, x1:x2] = image
```

**Browser**:
```javascript
function padFromCenter(image, width, height, padWidth, padHeight) {
  const newWidth = width + 2 * padWidth;
  const newHeight = height + 2 * padHeight;
  const padded = new Uint8Array(newWidth * newHeight);
  padded.fill(255); // White background

  // Copy original image to center
  for (let y = 0; y < height; y++) {
    const srcRow = y * width;
    const dstRow = (y + padHeight) * newWidth + padWidth;
    padded.set(image.slice(srcRow, srcRow + width), dstRow);
  }

  return { image: padded, width: newWidth, height: newHeight };
}
```

---

## Category 9: Coordinate Operations

### 9.1 Point Arrays

**Python**:
```python
# src/processors/alignment/
points = np.float32([[x1, y1], [x2, y2], [x3, y3]])
```

**Browser**:
```javascript
// Flat representation
const points = new Float32Array([x1, y1, x2, y2, x3, y3]);

// Array of arrays
const points = [
  [x1, y1],
  [x2, y2],
  [x3, y3]
];

// Helper class
class Point {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  toArray() {
    return [this.x, this.y];
  }
}
```

### 9.2 Integer Conversion

**Python**:
```python
# src/utils/drawing.py:58
np.intp(contour)
```

**Browser**:
```javascript
// Convert float coordinates to integers
function toIntPoints(points) {
  return points.map(p => [Math.round(p[0]), Math.round(p[1])]);
}

// Or for flat array
const intPoints = new Int32Array(floatPoints.length);
for (let i = 0; i < floatPoints.length; i++) {
  intPoints[i] = Math.round(floatPoints[i]);
}
```

---

## Memory Efficiency Tips

### 1. Reuse Buffers

**Instead of**:
```javascript
for (let i = 0; i < iterations; i++) {
  const temp = new Uint8Array(size); // Allocates every iteration
  // process temp
}
```

**Do**:
```javascript
const temp = new Uint8Array(size); // Allocate once
for (let i = 0; i < iterations; i++) {
  temp.fill(0); // Reuse buffer
  // process temp
}
```

### 2. SharedArrayBuffer for Web Workers

```javascript
// Main thread
const sab = new SharedArrayBuffer(width * height);
const imageData = new Uint8Array(sab);

worker.postMessage({ buffer: sab, width, height });

// Worker thread
self.onmessage = (e) => {
  const { buffer, width, height } = e.data;
  const imageData = new Uint8Array(buffer);
  // Process in-place, no copying needed
};
```

### 3. Transferable Objects

```javascript
// Transfer ownership (zero-copy)
const buffer = new Uint8Array(largeSize);
worker.postMessage({ buffer: buffer.buffer }, [buffer.buffer]);
// buffer is now unusable in main thread
```

---

## Performance Comparison

| Operation | NumPy (Python) | TypedArray (JS) | ndarray.js | TensorFlow.js |
|-----------|----------------|-----------------|------------|---------------|
| Create array | Fast | Fast | Fast | Medium |
| Element access | Fast | Fast | Medium | Slow |
| Math operations | Very Fast | Medium | Fast | Very Fast (GPU) |
| Memory usage | Efficient | Efficient | Medium | High |
| Setup complexity | Low | Low | Medium | High |

**Recommendations**:
- Simple operations: Use native TypedArrays
- Complex math: Use TensorFlow.js (GPU acceleration)
- NumPy-like API: Use ndarray.js
- Large datasets: Consider Web Workers + SharedArrayBuffer

---

## Common Patterns

### Pattern 1: Image Processing Loop

**Python**:
```python
for y in range(height):
    for x in range(width):
        image[y, x] = process(image[y, x])
```

**Browser**:
```javascript
for (let y = 0; y < height; y++) {
  for (let x = 0; x < width; x++) {
    const idx = y * width + x;
    image[idx] = process(image[idx]);
  }
}
```

### Pattern 2: Kernel Convolution

**Python**:
```python
for y in range(1, height - 1):
    for x in range(1, width - 1):
        result[y, x] = (
            image[y-1, x-1] * kernel[0, 0] +
            image[y-1, x] * kernel[0, 1] +
            # ... etc
        )
```

**Browser**:
```javascript
for (let y = 1; y < height - 1; y++) {
  for (let x = 1; x < width - 1; x++) {
    const idx = y * width + x;
    result[idx] = (
      image[(y-1) * width + (x-1)] * kernel[0] +
      image[(y-1) * width + x] * kernel[1] +
      // ... etc
    );
  }
}
```

---

## Summary

**Key Takeaways**:
1. TypedArrays are the browser equivalent of NumPy arrays
2. Manual indexing replaces NumPy's multi-dimensional syntax
3. Use helper functions for common operations
4. Consider ndarray.js for NumPy-like API
5. Use TensorFlow.js for GPU acceleration
6. Always profile before optimizing
7. Web Workers for heavy computations

**Migration Priority**:
- High: Array creation, indexing, basic math
- Medium: Statistical operations, reshaping
- Low: Advanced linear algebra (use TensorFlow.js)
