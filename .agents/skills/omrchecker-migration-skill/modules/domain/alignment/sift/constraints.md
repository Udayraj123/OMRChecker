# SIFT Alignment - Constraints & Edge Cases

## Core Constraints

### 1. SIFT Patent Status (Historical)

**Expired**: March 2020

**Impact**:
- Pre-2020: SIFT in `opencv-contrib` (non-free)
- Post-2020: SIFT in main `opencv` package
- Browser: May or may not be included in OpenCV.js builds

**Browser Fallback**:
```javascript
if (typeof cv.SIFT_create === 'undefined') {
  // Use ORB or AKAZE
  detector = new cv.ORB_create(500);
}
```

---

### 2. MIN_MATCH_COUNT = 10

**Hard Limit**: Requires at least 10 good matches

**Rationale**:
- Homography needs minimum 4 points
- RANSAC needs redundancy for robustness
- 10 provides good balance

**Failure Modes**:
```python
# Case 1: Blank/solid field block
features detected = 0
good matches = 0 < 10  → FAIL

# Case 2: Low-quality scan
features detected = 50
good matches (after filters) = 7 < 10  → FAIL

# Result: alignment skipped, shifts = [0, 0]
```

---

### 3. Max Displacement Filter

**Purpose**: Reject matches with excessive displacement

**Example**:
```python
max_displacement = 30
match: source=(100,200), dest=(150,250)
displacement = √(50² + 50²) ≈ 70.7
70.7 > 30  → REJECT
```

**Edge Cases**:
- **Too small** (e.g., 5): Rejects valid matches in misaligned scans
- **Too large** (e.g., 200): Accepts false matches
- **Recommended**: 10-50 pixels for flatbed, 50-100 for mobile

---

### 4. Descriptor Dimension = 128

**Fixed**: SIFT descriptors always 128-dimensional float vectors

**Memory**:
```python
200 keypoints × 128 floats × 4 bytes = 100 KB per image
```

**Comparison**:
- **SIFT**: 128 floats (512 bytes/descriptor)
- **ORB**: 256 bits (32 bytes/descriptor) - 16× smaller!
- **AKAZE**: 486 bits (61 bytes/descriptor)

---

### 5. FLANN Index Parameters

**Constraint**: Tree-based index optimized for float descriptors

```python
index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
search_params = {"checks": 50}
```

**Trade-offs**:
- `trees=5`: More trees = better accuracy, slower build
- `checks=50`: More checks = better accuracy, slower search

**Browser**: FLANN may not work reliably → use BFMatcher

---

### 6. Feature Detection Variability

**Non-Deterministic**: SIFT may detect different features on repeat runs

**Causes**:
- Floating-point precision
- Image noise
- Minor implementation differences

**Impact**:
```python
# Same image, two runs:
Run 1: 237 features detected
Run 2: 239 features detected  ← Slightly different

# Not a bug, expected behavior
```

**Mitigation**: Use multiple runs and average results (not currently done)

---

### 7. Homography Matrix Constraints

**Requires**: Minimum 4 point correspondences

**Degenerate Cases**:
```python
# All points collinear:
points = [[0,0], [1,1], [2,2], [3,3]]
# Homography: ILL-CONDITIONED

# All points identical:
points = [[100,200], [100,200], [100,200], [100,200]]
# Homography: UNDEFINED

# cv2.findHomography() may return None or identity
```

**RANSAC Protection**: Tries multiple sample sets, implicitly handles some degenerate cases

---

### 8. Displacement Pair Order

**Critical**: Order matters for warping direction

```python
# CORRECT (current):
displacement_pairs.append([destination_point, source_point])
# destination = where it IS (scanned)
# source = where it SHOULD BE (template)

# WRONG:
displacement_pairs.append([source_point, destination_point])
# Inverts warping direction!
```

---

### 9. Thread Safety

**Issue**: Singleton SIFT/FLANN shared across threads

**Python GIL**: Protects for now, but not guaranteed

**Risk**: ThreadPoolExecutor in batch processing

**Fix** (not implemented):
```python
import threading

class SiftMatcher:
    _local = threading.local()

    @staticmethod
    def get_sift():
        if not hasattr(SiftMatcher._local, 'sift'):
            SiftMatcher._local.sift = cv2.SIFT_create()
        return SiftMatcher._local.sift
```

---

### 10. Memory Limits

**Browser Constraint**: Heap size limited (1-4 GB)

**Large Image**:
```python
# 7000×10000 px image
SIFT features: 2000-5000
Descriptors: 5000 × 128 × 4 = 2.5 MB per image
Two images: 5 MB

# Manageable, but adds up across field blocks
```

**Mitigation**: Downsample large images before SIFT

---

## Edge Cases

### 1. No Features Detected

**Cause**: Blank/solid color region

**Handling**:
```python
source_features, des1 = sift.detectAndCompute(blank_image, None)
# source_features = []
# des1 = None or empty array

matches = flann.knnMatch(des1, des2, k=2)
# matches = []

len(good) = 0 < MIN_MATCH_COUNT
# logger.critical("Not enough matches...")
# Return empty displacement_pairs → shifts=[0,0]
```

---

### 2. Fewer than 10 Matches

**Scenario**: Low-texture region, poor scan quality

**Result**:
```python
len(good) = 7
logger.critical("Not enough matches are found - 7/10")
matches_mask = None
# But still return 7 displacement_pairs
# k_nearest_interpolation will use them (risky)
```

**Better Handling** (not implemented):
```python
if len(good) < MIN_MATCH_COUNT:
    return []  # Don't return unreliable pairs
```

---

### 3. All Matches Filtered Out

**Cause**: max_displacement too strict

**Example**:
```python
max_displacement = 10
Actual shift: [25, 30]  ← Exceeds limit

# All matches rejected:
for m, n in matches:
    if distance(src, dst) <= 10:  # Never true
        good.append(m)

len(good) = 0
```

---

### 4. Scale/Rotation Invariance Limits

**SIFT is NOT fully invariant**:

**Scale**: Works well for 0.5× to 2× scale changes
```python
# Beyond this:
template_dpi = 300
scan_dpi = 72  # 0.24× scale
# SIFT may fail to match
```

**Rotation**: Works well for < 45° rotations
```python
# Beyond this:
rotation = 90°
# Many features may not match
```

**Pre-processing helps**: AutoRotate, resize to consistent dimensions

---

### 5. Lowe's Ratio Test Edge Cases

**Simplified Test**: `m.distance < n.distance`

**Problem**: Nearly identical bubbles
```python
# Two empty bubbles look identical:
m.distance = 85.3  # Match to bubble A
n.distance = 85.7  # Match to bubble B (very close!)

85.3 < 85.7  → TRUE (accept)

# But this is ambiguous! Could match either bubble.
# max_displacement filter helps disambiguate.
```

---

### 6. RANSAC Outlier Rejection

**RANSAC threshold**: `max_displacement`

**Edge Case**: All matches are outliers
```python
homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 30)
# mask = [[0], [0], [0], ...] ← All zeros

# All matches rejected as outliers
# But displacement_pairs still returned
```

---

### 7. Visualization Buffer Overflow

**Issue**: `cv2.drawMatches()` can crash with too many matches

**Limit**:
```python
if len(good) > 1000:
    # Displaying 1000+ lines may crash or be slow
    good = good[:500]  # Only visualize subset
```

**Not currently implemented**, but rare (usually < 200 matches)

---

### 8. Keypoint Location Precision

**Sub-pixel Precision**: `keypoint.pt = (152.37, 87.93)`

**Rounding**:
```python
# When converting to displacement:
source_point = (152.37, 87.93)
destination_point = (157.41, 85.88)

# Displacement:
dx = 157.41 - 152.37 = 5.04
dy = 85.88 - 87.93 = -2.05

# Later converted to int:
shifts = [int(5.04), int(-2.05)] = [5, -2]
# Loses sub-pixel precision (acceptable for OMR)
```

---

## Browser-Specific Constraints

### 1. SIFT Unavailability

**Issue**: OpenCV.js build may not include SIFT

**Detection**:
```javascript
if (typeof cv.SIFT_create === 'undefined') {
  console.error('SIFT not available');
  // Fallback to ORB/AKAZE
}
```

### 2. FLANN Unreliability

**Issue**: FLANN in OpenCV.js less stable than Python

**Solution**: Use BFMatcher (brute force)
```javascript
const bf = new cv.BFMatcher(cv.NORM_L2, true);
const matches = bf.match(desc1, desc2);
```

**Trade-off**: Slower (O(n²) vs O(n log n)), but more reliable

### 3. Memory Leaks

**Critical**: Must manually delete cv.Mat objects

```javascript
const keypoints = new cv.KeyPointVector();
const descriptors = new cv.Mat();

try {
  sift.detectAndCompute(image, mask, keypoints, descriptors);
  // Use descriptors
} finally {
  keypoints.delete();  // MUST delete
  descriptors.delete(); // MUST delete
}
```

**Failure to delete**: Memory accumulates → tab crash

### 4. Performance

**SIFT is slow in browser**: 50-200ms per image (vs 10-50ms in Python)

**Mitigation**:
- Use Web Worker (offload from main thread)
- Downsample images
- Consider ORB (faster, 10-50ms)

---

## Validation Checks

### Before SIFT Matching

1. ✓ Images are grayscale (8-bit, single channel)
2. ✓ Images not empty (width > 0, height > 0)
3. ✓ max_displacement reasonable (10-200 pixels)
4. ✓ SIFT detector initialized (not None)

### After SIFT Matching

1. ✓ Check `len(good) >= MIN_MATCH_COUNT`
2. ✓ Verify displacement_pairs not empty
3. ✓ All displacements finite (not NaN, not Inf)
4. ✓ Homography matrix valid (not None, not all zeros)

---

## Summary Table

| Constraint | Value | Rationale |
|------------|-------|-----------|
| MIN_MATCH_COUNT | 10 | Homography robustness |
| Descriptor dimension | 128 floats | SIFT standard |
| FLANN trees | 5 | Accuracy/speed trade-off |
| FLANN checks | 50 | Search quality |
| Lowe's ratio | 1.0 (simplified) | More matches, filtered by displacement |
| Max displacement | 10-200px | Scene-dependent |
| RANSAC threshold | max_displacement | Geometric consistency |

---

## Related Constraints

- **Alignment Constraints** (`../constraints.md`)
- **K-Nearest Constraints** (`../k-nearest/constraints.md`)
- **OpenCV Constraints** (`modules/technical/opencv/opencv-operations.md`)
