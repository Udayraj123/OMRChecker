# Alignment Flow - Constraints & Edge Cases

## System Constraints

### 1. SIFT Feature Limit

**Constraint**: SIFT detector may find 0 to 10,000+ features per image

**Implications**:
```python
# Feature-poor regions (blank areas, solid colors):
source_features = []  # No features detected
# Result: No matches → alignment fails gracefully

# Feature-rich regions (text, complex patterns):
source_features = [...]  # 1000+ features
# Result: Slow matching, but accurate
```

**Handling**:
```python
# MIN_MATCH_COUNT = 10
if len(good_matches) < MIN_MATCH_COUNT:
    logger.critical(f"Not enough matches: {len(good)}/{MIN_MATCH_COUNT}")
    # Skip alignment for this field block
    # scan_box.shifts remain [0, 0]
```

**Edge Case**: Completely blank field block
```python
# Example: Empty answer sheet section
# Features detected: 0
# Matches: 0
# Result: No alignment, proceed with template coordinates
```

---

### 2. Max Displacement Constraint

**Constraint**: Shifts limited by `max_displacement` parameter

**Configuration**:
```json
{
  "alignment": {
    "max_displacement": 30  // Maximum pixels in any direction
  }
}
```

**Purpose**:
1. **Filter outliers**: Reject matches too far apart
2. **Performance**: Limit search space
3. **Sanity check**: Prevent absurd displacements

**Filtering Logic**:
```python
# sift_matcher.py:68-71
if (m.distance < n.distance and
    MathUtils.distance(source_point, dest_point) <= max_displacement):
    good.append(m)
```

**Example**:
```python
# max_displacement = 30 pixels
match1: source=[100, 200], dest=[105, 202]
  displacement = √((105-100)² + (202-200)²) = √29 ≈ 5.4 pixels ✓ ACCEPT

match2: source=[100, 200], dest=[150, 250]
  displacement = √((150-100)² + (250-200)²) = √5000 ≈ 70.7 pixels ✗ REJECT
```

**Edge Cases**:

#### Case 1: max_displacement = 0
```python
# template_alignment.py:54-56
if max_displacement == 0:
    continue  # Skip alignment for this field block
```
**Use Case**: Known perfect alignment (e.g., digital template)

#### Case 2: max_displacement too small
```python
# Actual shift: [35, 40] pixels
# max_displacement: 30 pixels
# Result: All matches rejected → alignment fails
# Fallback: Use template coordinates (may be inaccurate)
```

#### Case 3: max_displacement too large
```python
# max_displacement: 200 pixels
# Many false matches accepted
# RANSAC handles some outliers, but quality degrades
```

**Recommended Values**:
- **Flatbed scanner**: 10-30 pixels
- **Mobile camera**: 50-100 pixels
- **Low-quality scan**: 100-200 pixels

---

### 3. Image Dimension Constraints

**Constraint**: Alignment images must match template dimensions

**Normalization**:
```python
# template_alignment.py:28-39
(gray_image, colored_image, gray_alignment_image, colored_alignment_image) = (
    ImageUtils.resize_to_dimensions(
        template.template_dimensions,  # e.g., [1200, 1600]
        gray_image,
        colored_image,
        gray_alignment_image,
        colored_alignment_image,
    )
)
```

**Why Resize?**
1. Different scan DPI (200, 300, 600 DPI)
2. Template created at different resolution
3. Coordinate system consistency

**Edge Case**: Extreme aspect ratio mismatch
```python
# Template: [1200, 1600]  (aspect ratio 3:4)
# Scan: [800, 2000]       (aspect ratio 2:5)
# Result: Distorted resize → poor alignment

# Prevention: Pre-processors should crop to correct aspect ratio BEFORE alignment
```

**Browser Constraint**: Canvas size limits
```javascript
// Most browsers limit canvas to ~16,000 x 16,000 pixels
// For large scans (e.g., A3 at 600 DPI ≈ 7000x10000):
// May need to downsample before alignment
const maxDimension = 4000;
if (width > maxDimension || height > maxDimension) {
  const scale = maxDimension / Math.max(width, height);
  width *= scale;
  height *= scale;
}
```

---

### 4. Zone Boundary Constraints

**Constraint**: Alignment zones may extend beyond image bounds

**Calculation**:
```python
# template_alignment.py:59-68
zone_start = [
    int(bounding_box_origin[0] - margins["left"]),
    int(bounding_box_origin[1] - margins["top"]),
]
zone_end = [
    int(bounding_box_origin[0] + margins["right"] + bounding_box_dimensions[0]),
    int(bounding_box_origin[1] + margins["bottom"] + bounding_box_dimensions[1]),
]
```

**Edge Cases**:

#### Case 1: Negative zone_start
```python
# Field block at top-left corner: origin=[10, 10]
# Margins: {left: 50, top: 50, ...}
# zone_start = [10 - 50, 10 - 50] = [-40, -40]  ← NEGATIVE!

# NumPy slicing:
image[-40:..., -40:...]  # Wraps around (unexpected!)

# Better: Explicit clipping
zone_start = [max(0, x) for x in zone_start]
```

**Current Behavior**: Relies on NumPy's slicing
```python
# NumPy automatically clips:
block_image = image[zone_start[1]:zone_end[1], zone_start[0]:zone_end[0]]
# If zone_start is negative, starts from 0
# If zone_end exceeds shape, ends at shape boundary
```

#### Case 2: zone_end exceeds image dimensions
```python
# Field block near bottom-right
# zone_end = [1250, 1650]
# Image shape = [1200, 1600]
# Result: zone_end clipped to [1200, 1600]
# ROI smaller than expected, fewer features
```

#### Case 3: Invalid zone (negative dimensions)
```python
# Malformed template:
zone_start = [500, 600]
zone_end = [400, 700]  # end.x < start.x
# Result: Empty ROI
block_image.shape = (100, 0)  # Zero width!
# SIFT on empty image: 0 features → alignment fails
```

---

### 5. Memory Constraints

**Constraint**: Processing multiple large images can exhaust memory

**Memory Usage Per Field Block**:
```python
# Typical field block alignment:
# 1. ROI extraction (2 images):
#    - gray_block: 400x300 px × 1 byte = 120 KB
#    - alignment_block: 400x300 px × 1 byte = 120 KB
# 2. SIFT features:
#    - ~200 keypoints × 128-dim descriptors × 4 bytes = 100 KB
# 3. Displacement pairs: ~50 pairs × 16 bytes = 0.8 KB
# Total per block: ~350 KB

# Template with 20 field blocks:
# Peak memory: 20 × 350 KB = 7 MB (manageable)

# Template with 100 field blocks:
# Peak memory: 100 × 350 KB = 35 MB (still OK)
```

**Edge Case**: Very large images
```python
# A3 page at 600 DPI: ~7000 x 10000 px
# Full image: 70 MB (grayscale)
# Field block 1000x1000 with margins: 1 MB per ROI
# 50 field blocks: 50 MB additional memory

# Risk: Memory exhaustion on low-end machines
```

**Mitigation**:
1. Process field blocks sequentially (current approach)
2. Release ROI references after processing
3. Downsample for alignment (accuracy trade-off)

**Browser Constraint**:
```javascript
// Browser memory limits vary (typically 1-4 GB)
// For large scans, consider:
// 1. Offload to Web Worker (separate heap)
// 2. Process in chunks
// 3. Use OffscreenCanvas for image ops
```

---

### 6. SIFT Patent Constraints (Expired)

**Historical Constraint**: SIFT was patented until 2020

**Current Status**: Patent-free since March 2020

**OpenCV Impact**:
- Pre-2020: SIFT in `opencv-contrib` (non-free)
- Post-2020: SIFT in `opencv` (main)

**Browser Consideration**:
```javascript
// OpenCV.js builds may or may not include SIFT
// Depends on build configuration

// Check availability:
if (typeof cv.SIFT_create === 'undefined') {
  // Fallback to ORB or AKAZE
  console.warn('SIFT not available, using ORB');
  detector = new cv.ORB_create();
}
```

---

### 7. Homography Constraints

**Constraint**: Homography requires at least 4 point correspondences

**RANSAC Requirements**:
```python
# cv2.findHomography() needs:
# - Minimum: 4 matches
# - Recommended: 10+ matches (for RANSAC robustness)

# sift_matcher.py:78-81
if len(good) > MIN_MATCH_COUNT:  # MIN_MATCH_COUNT = 10
    homography_matrix, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, max_displacement
    )
```

**Edge Case**: Exactly 4 matches
```python
# 4 matches: Homography computable, but unstable
# Single outlier → completely wrong homography
# RANSAC helps, but with only 4 points, limited samples
```

**Degenerate Cases**:
```python
# All points collinear (on same line):
points = [[100, 200], [150, 300], [200, 400], [250, 500]]
# Homography: ILL-CONDITIONED (matrix near-singular)
# Result: Unreliable transformation

# All points identical:
points = [[100, 200], [100, 200], [100, 200], [100, 200]]
# Homography: UNDEFINED
# cv2.findHomography() may return None or identity matrix
```

**Handling**:
```python
# Current: No explicit check for degenerate cases
# RANSAC implicitly handles by trying multiple sample sets
# If all sets degenerate, homography quality degrades
# But displacement_pairs still used for k-nearest averaging
```

---

### 8. K-Nearest Constraints

**Constraint**: K must not exceed number of displacement pairs

**Code**: `k_nearest_interpolation.py:12-25`
```python
def find_k_nearest_anchors(origin, anchors_with_displacements, k):
    sorted_by_distance = sorted([...])
    return sorted_by_distance[:k]  # May return fewer than k if len < k
```

**Edge Cases**:

#### Case 1: Fewer displacement pairs than K
```python
# K = 4 (default)
# displacement_pairs = [[p1, d1], [p2, d2], [p3, d3]]  # Only 3 pairs

# Result:
nearest_anchors = sorted_by_distance[:4]  # Returns only 3
average_shifts = mean([[d1], [d2], [d3]])  # Average of 3, not 4
# Still works, just less stable
```

#### Case 2: Zero displacement pairs
```python
# SIFT failed, displacement_pairs = []
# find_k_nearest_anchors() returns []
# np.average([]) → RuntimeWarning + NaN

# Current: Not explicitly handled
# Result: shifts = [NaN, NaN] → DETECTION FAILS

# Fix needed:
if len(displacement_pairs) == 0:
    logger.warning("No displacement pairs, skipping alignment")
    return [0, 0]  # Use template coordinates
```

#### Case 3: Single displacement pair
```python
# displacement_pairs = [[p1, d1]]
# nearest_anchors = [[p1, d1]]
# average_shifts = d1  # No averaging, just use the one displacement
# Risky: Single match could be outlier
```

**Robustness Check**:
```python
# Recommended:
MIN_DISPLACEMENT_PAIRS = 5

if len(displacement_pairs) < MIN_DISPLACEMENT_PAIRS:
    logger.warning(f"Too few displacement pairs: {len(displacement_pairs)}")
    return [0, 0]
```

---

### 9. Coordinate System Constraints

**Constraint**: All coordinates relative to specific image spaces

**Coordinate Spaces**:

1. **Template Space**: Original template.json coordinates
2. **Resized Space**: After `resize_to_dimensions`
3. **Block Space**: ROI coordinates (relative to zone_start)
4. **Shifted Space**: After applying scan_box.shifts

**Conversions**:
```python
# Template → Block space:
block_image_shifts = subtract_points(
    [margins["left"], margins["top"]],
    field_block.origin
)

# Block → Template space:
template_position = add_points(
    block_position,
    subtract_points(field_block.origin, [margins["left"], margins["top"]])
)

# Shifted position:
final_position = add_points(template_position, scan_box.shifts)
```

**Edge Case**: Coordinate overflow
```python
# scan_box.shifts = [30000, 40000]  # Absurdly large
# final_position = [100 + 30000, 200 + 40000] = [30100, 40200]
# Outside image bounds!

# Detection will fail (accessing out-of-bounds pixel)
# Should validate: final_position within image bounds
```

---

### 10. Thread Safety Constraints

**Constraint**: Singleton SIFT/FLANN shared across threads

**Current**:
```python
# sift_matcher.py:18-21
class SiftMatcher:
    sift = None  # Class-level (shared)
    flann = None  # Class-level (shared)
```

**Python GIL**: Global Interpreter Lock protects for now

**Risk**: ThreadPoolExecutor in batch processing
```python
# If multiple threads call SiftMatcher.get_matches() simultaneously:
# Race condition on:
# - sift.detectAndCompute()
# - flann.knnMatch()

# OpenCV methods are NOT thread-safe
```

**Mitigation Options**:

1. **Thread-local SIFT** (recommended):
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

2. **Lock** (slower):
```python
import threading

sift_lock = threading.Lock()

def get_matches(...):
    with sift_lock:
        features1, desc1 = SiftMatcher.sift.detectAndCompute(...)
```

3. **Process pool** (instead of thread pool):
```python
# Each process has separate SIFT instance
# No shared memory → no race condition
```

**Browser**: No threading (Web Workers have separate contexts)

---

### 11. Numerical Precision Constraints

**Constraint**: Float arithmetic precision errors

**Example**:
```python
# Average of shifts:
shifts = [[-3, 2], [-2, 1], [-3, 2], [-4, 3]]
average = np.average(shifts, axis=0)
# Result: [-3.0, 2.0] (exact)

# But with float imprecision:
shifts = [[-3.1, 2.2], [-2.9, 1.8], [-3.2, 2.1]]
average = np.average(shifts, axis=0)
# Result: [-3.066666..., 2.033333...]
# Convert to int: [-3, 2]
```

**Rounding**:
```python
# k_nearest_interpolation.py:121-124
average_shifts = np.average([...], axis=0).astype(np.int32)
# np.int32 truncates: 2.9 → 2 (not rounded!)

# Better:
average_shifts = np.round(np.average([...], axis=0)).astype(np.int32)
# np.round: 2.9 → 3, 2.1 → 2
```

**Impact**: Off-by-one pixel errors in bubble detection

---

### 12. Browser-Specific Constraints

#### Constraint 1: SIFT Unavailability
```javascript
// OpenCV.js may not include SIFT
// Must check at runtime:
if (typeof cv.SIFT_create === 'undefined') {
  // Fallback to ORB/AKAZE
}
```

#### Constraint 2: Memory Limits
```javascript
// Browser tabs typically have 1-4 GB limit
// Large images may cause:
// - "Out of memory" error
// - Tab crash
// - Slow garbage collection

// Mitigation:
// - Downsample large images
// - Process in Web Worker
// - Manually call cv.Mat.delete() after use
```

#### Constraint 3: Single-threaded (except Web Workers)
```javascript
// Main thread blocking:
for (const fieldBlock of fieldBlocks) {
  alignFieldBlock(fieldBlock);  // Blocks UI!
}

// Solution: Async + yield
async function alignFieldBlocks(fieldBlocks) {
  for (const fieldBlock of fieldBlocks) {
    await alignFieldBlock(fieldBlock);
    await new Promise(resolve => setTimeout(resolve, 0));  // Yield
  }
}
```

#### Constraint 4: No Direct File Access
```javascript
// Can't load alignment image from file path
// Must use:
// - FileReader API
// - Fetch API
// - Base64 embedded in template JSON
```

---

## Edge Case Summary Table

| Edge Case | Current Behavior | Recommended Handling |
|-----------|------------------|----------------------|
| 0 SIFT features | Alignment fails, shifts=[0,0] | ✓ Graceful fallback |
| < 10 matches | Alignment fails, shifts=[0,0] | ✓ Logged as critical |
| max_displacement=0 | Skip alignment | ✓ Correct |
| Negative zone_start | NumPy clips to 0 | ⚠️ Could log warning |
| zone > image bounds | NumPy clips to bounds | ✓ Handled |
| Empty ROI | 0 features → fail | ✓ Graceful |
| Aspect ratio mismatch | Distorted resize | ⚠️ Pre-processor should fix |
| < K displacement pairs | Use fewer than K | ⚠️ Could validate |
| 0 displacement pairs | NaN shifts (bug) | ❌ Needs fix |
| Thread race condition | Undefined (GIL protects) | ⚠️ Use thread-local |
| Float precision | Truncation errors | ⚠️ Use rounding |

---

## Validation Checklist

Before running alignment, validate:

1. ✓ Alignment image exists in template
2. ✓ Image dimensions compatible (aspect ratio similar)
3. ✓ max_displacement reasonable (10-200 pixels)
4. ✓ Margins don't cause all zones to be out-of-bounds
5. ✓ Field blocks have non-zero dimensions
6. ✓ SIFT available (or fallback configured)

After alignment, validate:

1. ✓ All shifts finite (not NaN, not Inf)
2. ✓ Shifts within reasonable range (< max_displacement × 2)
3. ✓ Sufficient matches found (> MIN_MATCH_COUNT)
4. ✓ Shifted coordinates within image bounds

---

## Related Constraints

- **Template Constraints**: See `modules/domain/template/constraints.md`
- **Image Processing Constraints**: See `modules/technical/opencv/opencv-operations.md`
- **Memory Constraints**: See `modules/migration/performance.md`
