# Template Matching Alignment - Constraints & Edge Cases

## Core Constraints

### 1. Template Size Constraints

**Minimum Size**: Template must be smaller than search image

```python
# REQUIRED:
template.shape[0] < image.shape[0]  # Height
template.shape[1] < image.shape[1]  # Width

# Example FAIL:
image.shape = (100, 100)
template.shape = (120, 80)  # Template height > image height
# cv2.matchTemplate() raises error
```

**Maximum Size**: No hard limit, but performance degrades

```python
# Practical limits:
# Small template: 10×10 to 50×50 → Fast
# Medium template: 50×50 to 200×200 → Moderate
# Large template: 200×200 to 500×500 → Slow
# Very large: > 500×500 → Very slow (use downsampling)
```

**Result Matrix Dimensions**:
```python
result.shape = (H - h + 1, W - w + 1)

# Example:
image.shape = (1200, 800)
template.shape = (50, 50)
result.shape = (1151, 751)  # 864,401 values

# Edge case: Template almost as large as image
image.shape = (100, 100)
template.shape = (95, 95)
result.shape = (6, 6)  # Only 36 positions to test
```

---

### 2. Match Score Range

**TM_CCOEFF_NORMED Range**: -1.0 to +1.0

```python
# Interpretation:
+1.0:  Perfect positive match (ideal)
+0.9:  Excellent match (very reliable)
+0.7:  Good match (usually reliable)
+0.5:  Weak match (borderline)
+0.3:  Poor match (unreliable)
 0.0:  No correlation
-1.0:  Perfect inverse match (opposite)
```

**Typical Thresholds**:
```python
# AutoRotate default (not specified):
threshold = None  # Accept any rotation with highest score

# Recommended thresholds:
threshold = 0.6   # Conservative (high quality required)
threshold = 0.5   # Moderate (balanced)
threshold = 0.4   # Permissive (allow poor matches)
```

**Passthrough Mode**:
```python
threshold = {
    "value": 0.6,
    "passthrough": True  # Log warning but don't fail
}

# If best_val = 0.55 < 0.6:
# → Warning logged, processing continues
# → Risk: May use incorrect rotation
```

---

### 3. Normalization Constraints

**Input Image Type**: Must be 8-bit (CV_8U) or float

```python
# VALID:
image.dtype = np.uint8    # 0-255
image.dtype = np.float32  # Normalized to 0-1

# INVALID:
image.dtype = np.int16    # Not supported
image.dtype = np.uint16   # Not supported (must convert to 8-bit)

# Conversion:
image_8bit = (image_16bit / 256).astype(np.uint8)
```

**Normalization Method**:
```python
# TM_CCOEFF_NORMED is brightness-invariant:
image1 = template * 1.0        # Original
image2 = template * 1.5 + 50   # Brighter + offset

# Both should match well:
score1 = cv2.matchTemplate(image1, template, cv2.TM_CCOEFF_NORMED)
score2 = cv2.matchTemplate(image2, template, cv2.TM_CCOEFF_NORMED)
# score1 ≈ score2 (normalized method compensates)
```

---

### 4. Scale Sensitivity

**Critical Limitation**: Template matching is NOT scale-invariant

```python
# Template at 100% scale:
template.shape = (50, 50)
match_score = 0.95  # Excellent

# Same marker in image, but 110% scale (slightly larger):
actual_marker_size = (55, 55)
match_score = 0.65  # Drops significantly!

# 150% scale:
actual_marker_size = (75, 75)
match_score = 0.15  # Fails completely
```

**Solution**: Multi-scale matching

```python
# Test multiple scales: 85%, 92%, 100%, 107%, 115%
scale_range = (85, 115)
scale_steps = 5

# Example results:
scales = [85, 92, 100, 107, 115]
scores = [0.72, 0.81, 0.95, 0.88, 0.74]
                    ↑
            Best match at 100%
```

**Scale Range Trade-offs**:
```python
# Narrow range (faster):
scale_range = (95, 105)  # Only ±5%
scale_steps = 3          # 95%, 100%, 105%
# Risk: Miss match if actual scale is 110%

# Wide range (slower, more robust):
scale_range = (70, 130)  # ±30%
scale_steps = 13         # Every 5%
# Safer but 4× slower
```

---

### 5. Rotation Sensitivity

**Critical Limitation**: Template matching is NOT rotation-invariant

```python
# No rotation:
match_score = 0.95  # Excellent

# 5° rotation:
match_score = 0.75  # Drops

# 15° rotation:
match_score = 0.40  # Poor

# 45° rotation:
match_score = 0.10  # Fails

# 90° rotation:
match_score = 0.05  # Complete failure (unless marker is symmetric)
```

**Solution**: Test discrete rotations (AutoRotate approach)

```python
rotations = [
    None,                           # 0°
    cv2.ROTATE_90_CLOCKWISE,        # 90°
    cv2.ROTATE_180,                 # 180°
    cv2.ROTATE_90_COUNTERCLOCKWISE, # 270°
]

# Test each, keep best
for rotation in rotations:
    rotated = cv2.rotate(image, rotation) if rotation else image
    score = matchTemplate(rotated, template)
    # Track maximum

# Limitation: Only works for 90° increments
# Cannot detect arbitrary angles like 37°
```

---

### 6. Computational Performance Limits

**Time Complexity**:

```python
# FFT-based (OpenCV default for medium/large templates):
O((W × H) × log(W × H))

# Sliding window (small templates):
O(W × H × w × h)
```

**Benchmark** (approximate, varies by hardware):

```python
# Template size: 50×50
# Image size: 1200×800

Time per matchTemplate call:
Python (OpenCV): ~10-30ms
Browser (OpenCV.js): ~30-80ms

# Multi-scale (5 scales):
Python: ~50-150ms
Browser: ~150-400ms

# 4 rotations × 5 scales = 20 calls:
Python: ~200-600ms
Browser: ~600-1600ms (1-2 seconds)
```

**Memory Peak**:
```python
# Result matrix:
result.shape = (H - h + 1, W - w + 1)
result_memory = result.size × 4 bytes (float32)

# Example:
result.shape = (1151, 751)
result_memory = 864,401 × 4 = 3.46 MB

# Multi-scale with 5 scales:
# 5 result matrices in memory (if not freed)
peak_memory = 5 × 3.46 MB = ~17 MB

# Browser: Must explicitly delete cv.Mat to free
```

---

### 7. Border Effects

**Edge Clipping**: Template cannot match at image borders

```python
# Searchable region:
valid_x = 0 to (W - w)
valid_y = 0 to (H - h)

# Example:
image.shape = (100, 100)
template.shape = (20, 20)

# If marker is at image corner (0, 0):
# Template can only match from position (0, 0) to (80, 80)
# Marker at (90, 90) → CANNOT be detected (too close to edge)
```

**Solution**: Add padding

```python
# Add border before matching:
padded_image = cv2.copyMakeBorder(
    image,
    top=template.shape[0],
    bottom=template.shape[0],
    left=template.shape[1],
    right=template.shape[1],
    borderType=cv2.BORDER_REPLICATE
)

# Now full image is searchable
# Adjust detected positions by subtracting padding
```

---

### 8. Multiple Matches (Ambiguity)

**Problem**: Multiple identical/similar markers in image

```python
# Image with 4 corner markers (all identical):
result = cv2.matchTemplate(image, marker_template, cv2.TM_CCOEFF_NORMED)

# minMaxLoc returns ONLY one match (the best):
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# max_loc points to one of the 4 markers
# Other 3 markers ignored!
```

**Solution**: Threshold-based multi-match detection

```python
# Find all matches above threshold:
threshold = 0.8
locations = np.where(result >= threshold)

match_positions = list(zip(*locations[::-1]))  # (x, y) pairs

# Example:
match_positions = [
    (10, 10),    # Top-left marker
    (10, 990),   # Bottom-left marker
    (790, 10),   # Top-right marker
    (790, 990),  # Bottom-right marker
]

# Now we have all 4 markers
```

**Non-Maximum Suppression** (avoid overlapping detections):

```python
def non_max_suppression(locations, template_shape, threshold=0.5):
    """Remove overlapping matches."""
    w, h = template_shape
    final_matches = []

    for loc in locations:
        # Check if overlaps with existing matches
        overlap = False
        for existing in final_matches:
            if (abs(loc[0] - existing[0]) < w * threshold and
                abs(loc[1] - existing[1]) < h * threshold):
                overlap = True
                break

        if not overlap:
            final_matches.append(loc)

    return final_matches
```

---

## Edge Cases

### 1. Blank/Uniform Images

**Scenario**: Template or image is solid color

```python
# Blank template:
template = np.ones((50, 50), dtype=np.uint8) * 255  # All white

# Matching against any image:
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# Result:
# All positions have similar low correlation
# max_val ≈ 0.0 to 0.2 (no distinctive features)
# Unreliable match
```

**Detection**:
```python
# Check template variance:
template_variance = np.var(template)
if template_variance < 10:  # Threshold
    logger.warning("Template is too uniform, matching unreliable")
```

---

### 2. Template Larger Than Image

**Error**:
```python
image.shape = (50, 50)
template.shape = (100, 100)

cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
# OpenCV error: "Template must be smaller than image"
```

**Validation**:
```python
def validate_template_size(image, template):
    if (template.shape[0] >= image.shape[0] or
        template.shape[1] >= image.shape[1]):
        raise ValueError(
            f"Template {template.shape} must be smaller than image {image.shape}"
        )
```

---

### 3. Single-Pixel Result

**Scenario**: Template almost fills entire image

```python
image.shape = (100, 100)
template.shape = (100, 100)  # Exact match

result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
result.shape = (1, 1)  # Single value

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
max_loc = (0, 0)  # Only one position possible
```

**Interpretation**: Not an error, but limited usefulness

---

### 4. Negative Correlation (Inverse Match)

**Scenario**: Template is brightness-inverted version of pattern

```python
# Template: Dark marker on white background
template = np.array([
    [255, 255, 255],
    [255,   0, 255],
    [255, 255, 255],
], dtype=np.uint8)

# Image: Light marker on dark background
image_patch = np.array([
    [0, 0, 0],
    [0, 255, 0],
    [0, 0, 0],
], dtype=np.uint8)

# Match score:
# TM_CCOEFF_NORMED → -1.0 (perfect inverse)
```

**Handling**:
```python
# Use absolute value if inverse matches are valid:
abs_result = np.abs(result)
max_val = np.max(abs_result)

# Or use different method:
# TM_CCORR_NORMED → Always positive
```

---

### 5. Threshold at Boundary

**Scenario**: best_val exactly equals threshold

```python
best_val = 0.6000
threshold = 0.6

# Current code:
if threshold > best_val:  # 0.6 > 0.6 → False
    # Pass (edge case: equality accepted)

# Safer comparison:
if threshold > best_val + 1e-6:  # Add epsilon
```

---

### 6. Marker Preprocessing Side Effects

**Edge Enhancement Can Fail**:

```python
# Original marker (already edge-enhanced):
marker = load_marker()  # Has strong edges

# Apply erode-subtract:
marker_enhanced = marker - cv2.erode(marker, kernel, iterations=5)

# Result: Over-enhanced (noise amplified)
# Match score may actually decrease
```

**Solution**: Make enhancement optional

```python
apply_erode_subtract = self.options.get("apply_erode_subtract", False)
# Only enable if marker is low-contrast
```

---

### 7. Rotation with Non-Square Images

**Issue**: Rotating non-square images changes dimensions

```python
image.shape = (1200, 800)  # Landscape

# Rotate 90°:
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
rotated.shape = (800, 1200)  # Portrait (dimensions swapped!)

# Affects subsequent processing if code assumes fixed dimensions
```

**Solution**: `keep_original_shape=True`

```python
# OMRChecker approach:
rotated = ImageUtils.rotate(image, rotation, keep_original_shape=True)
# Pads image to keep original bounding box
```

---

### 8. Scale Step Selection

**Issue**: Too few steps miss optimal scale

```python
# Actual marker scale: 103%
# Tested scales: [85, 100, 115]  # 3 steps, 15% increments
# Best tested: 100% → score 0.75
# Missed optimal: 103% → would score 0.95

# More steps:
# Tested scales: [85, 92, 100, 107, 115]  # 5 steps
# Now 107% tested → score 0.88 (better, but not optimal)
```

**Trade-off**:
```python
# Fewer steps (3-5): Faster, may miss peak
# More steps (10-20): Slower, more accurate
# Adaptive: Test coarse, then refine around peak
```

---

## Browser-Specific Constraints

### 1. Memory Management (Critical)

**JavaScript GC Doesn't Free cv.Mat**:

```javascript
// BAD: Memory leak
for (let i = 0; i < 100; i++) {
  const result = new cv.Mat();
  cv.matchTemplate(image, template, result, cv.TM_CCOEFF_NORMED);
  // result not deleted → 100× result matrices in memory!
}

// GOOD: Proper cleanup
for (let i = 0; i < 100; i++) {
  const result = new cv.Mat();
  try {
    cv.matchTemplate(image, template, result, cv.TM_CCOEFF_NORMED);
    processResult(result);
  } finally {
    result.delete();  // CRITICAL
  }
}
```

---

### 2. Performance Constraints

**Browser is ~2-4× slower than Python**:

```javascript
// Python: 10-30ms per call
// Browser: 30-80ms per call

// Multi-scale (5 scales) + 4 rotations = 20 calls:
// Python: ~200-600ms
// Browser: ~600-1600ms (may feel sluggish)
```

**Optimization**: Use Web Worker

```javascript
// Offload to worker:
const worker = new Worker('template-matcher-worker.js');
worker.postMessage({ imageData, templateData, rotations, scales });

worker.onmessage = (event) => {
  const { bestRotation, bestScore } = event.data;
  // Process result on main thread
};
```

---

### 3. ImageData Conversion Overhead

**cv.Mat ↔ ImageData conversion is slow**:

```javascript
// Converting canvas ImageData to cv.Mat:
const imageData = ctx.getImageData(0, 0, width, height);
const mat = cv.matFromImageData(imageData);  // ~10-50ms

// Converting back:
cv.imshow('canvas', mat);  // ~10-50ms

// Adds 20-100ms overhead per operation
```

**Mitigation**: Minimize conversions, keep cv.Mat in memory

---

### 4. Heap Size Limits

**Browser Wasm Heap**: 1-4 GB (varies by browser)

```javascript
// Large image: 4000×3000 pixels
// Gray: 12 MB
// Result matrix: ~12 MB
// Multi-scale (5 scales): 5× templates + 5× results = ~60 MB
// 4 rotations: 4× images = ~48 MB

// Total: ~120 MB per image (manageable)

// Batch of 10 images: ~1.2 GB (approaching limit!)
```

**Solution**: Process images sequentially, free memory between

---

## Validation Checklist

### Before Template Matching

- [ ] Template width < image width
- [ ] Template height < image height
- [ ] Template variance > threshold (not uniform)
- [ ] Images are 8-bit grayscale or float
- [ ] Template dimensions > minimum (e.g., 5×5)

### After Template Matching

- [ ] Result matrix has expected shape
- [ ] max_val is finite (not NaN, not Inf)
- [ ] max_val meets threshold (if required)
- [ ] max_loc is within valid range
- [ ] Memory freed (browser: result.delete())

### Multi-Scale Specific

- [ ] Scale range reasonable (e.g., 70-130%)
- [ ] Enough scale steps (at least 3-5)
- [ ] Scaled templates not larger than image
- [ ] Best scale is reasonable (not at extreme edge)

---

## Comparison with Other Methods

| Constraint | Template Matching | SIFT | Phase Correlation |
|------------|-------------------|------|-------------------|
| **Scale Invariance** | No (requires multi-scale) | Yes | No |
| **Rotation Invariance** | No (requires testing) | Yes | No |
| **Match Type** | Single position | Multiple points | Single shift |
| **Performance** | Fast (FFT) | Moderate | Very fast (FFT) |
| **Memory** | Low-moderate | High (descriptors) | Low |
| **Robustness** | Template-dependent | High | Moderate |
| **Browser Support** | Excellent | Limited (ORB fallback) | Good |

---

## Summary Table

| Constraint | Value/Range | Impact |
|------------|-------------|--------|
| **Template Size** | < Image size | Hard limit (error if violated) |
| **Min Template Size** | ~5×5 pixels | Below this: unreliable |
| **Max Template Size** | ~500×500 | Above this: very slow |
| **Match Score Range** | -1.0 to +1.0 | Normalized correlation |
| **Typical Threshold** | 0.5 - 0.7 | Scene-dependent |
| **Scale Sensitivity** | ±5% → noticeable | Multi-scale required |
| **Rotation Sensitivity** | ±5° → noticeable | Test rotations required |
| **Time (Python)** | 10-30ms | Per matchTemplate call |
| **Time (Browser)** | 30-80ms | 2-4× slower than Python |
| **Memory** | ~3-10 MB | Per result matrix |

---

## Related Constraints

- **AutoRotate Constraints** (`modules/domain/preprocessing/auto-rotate/flows.md`)
- **Alignment Constraints** (`../constraints.md`)
- **OpenCV Constraints** (`modules/technical/opencv/opencv-operations.md`)
- **Browser Performance** (`modules/migration/performance.md`)
