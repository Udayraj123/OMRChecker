# Alignment Flow - Design Decisions

## Key Design Decisions

### Decision 1: Warp Coordinates vs. Warp Image

**Current Choice**: **Warp Coordinates** (Method 3: K-Nearest Interpolation)

#### Options Considered

| Approach | How It Works | Pros | Cons |
|----------|--------------|------|------|
| **Warp Image** | Transform scanned image to match template | • Visually intuitive<br>• Standard CV approach | • Cannot handle overlapping field blocks<br>• Computationally expensive<br>• Image quality degradation |
| **Warp Coordinates** | Adjust bubble positions to match scan | • Supports overlapping blocks<br>• Faster (no image transformation)<br>• Preserves original image quality | • Less intuitive<br>• Requires careful coordinate tracking |

#### Why Warp Coordinates?

```python
# Problem with Image Warping (Method 2):
gray_image[zone_start[1]:zone_end[1], zone_start[0]:zone_end[0]] = warped_block_image

# If field blocks overlap, which warped image wins?
# Field Block A: zone [100:300, 100:300]
# Field Block B: zone [200:400, 200:400]
# Overlap region: [200:300, 200:300] - CONFLICT!
```

```python
# Solution with Coordinate Warping (Method 3):
for field_block in template.field_blocks:
    for field in field_block.fields:
        for scan_box in field.scan_boxes:
            scan_box.shifts = average_shifts

# Each bubble tracks its OWN shift
# Overlapping field blocks: no problem!
# Detection uses: bubble_position + scan_box.shifts
```

**Code Reference**:
- Method 2 (commented): `template_alignment.py:95-114`
- Method 3 (active): `template_alignment.py:118-125`

**Trade-off**: Slightly more complex coordinate tracking for better flexibility

---

### Decision 2: Per-Field-Block vs. Global Alignment

**Current Choice**: **Per-Field-Block Alignment**

#### Rationale

**Why Not Global?**
```python
# Problem: Different parts of page may shift differently
# Example: Page curl causes non-uniform distortion
#
#  ┌─────────┐
#  │ Block A │  shift = [-2, 3]
#  ├─────────┤
#  │ Block B │  shift = [-4, 5]  ← Different!
#  └─────────┘
#
# Global average = [-3, 4] → Both blocks misaligned!
```

**Per-Field-Block Benefits**:
- Handles local distortions
- Page warping, bending, scanner artifacts
- Independent shifts per block

**Code**:
```python
# Each field block gets its own alignment
for field_block in template.field_blocks:
    average_shifts = apply_k_nearest_interpolation_inplace(...)
    # Shifts specific to THIS field block
```

**Trade-off**: More computation, but significantly more accurate

---

### Decision 3: SIFT vs. Other Feature Detectors

**Current Choice**: **SIFT (Scale-Invariant Feature Transform)**

#### Feature Detector Comparison

| Method | Invariance | Speed | Accuracy | Patent Status |
|--------|------------|-------|----------|---------------|
| **SIFT** | Scale, rotation | Slow | Excellent | Expired (2020) |
| **SURF** | Scale, rotation | Medium | Very good | Patented |
| **ORB** | Rotation | Fast | Good | Free |
| **AKAZE** | Scale, rotation | Medium | Very good | Free |

#### Why SIFT?

1. **Scale Invariance**: Handles DPI differences
2. **Rotation Tolerance**: Small angle misalignments
3. **Robust Matching**: Works with low-quality scans
4. **Patent-Free** (since 2020)

**Code**: `sift_matcher.py:24`
```python
SiftMatcher.sift = cv2.SIFT_create()
```

#### Browser Consideration

**Problem**: OpenCV.js may not include SIFT (build-dependent)

**Browser Fallback Strategy**:
```javascript
// Recommended fallback chain
if (cv.SIFT_create) {
  detector = new cv.SIFT_create();
} else if (cv.AKAZE_create) {
  detector = new cv.AKAZE_create();  // Good alternative
} else if (cv.ORB_create) {
  detector = new cv.ORB_create(500);  // Fast, less accurate
} else {
  // Ultimate fallback: Phase Correlation (no features)
  usePhaseCorrelation();
}
```

**Trade-off**: SIFT best for Python, need fallback for browser

---

### Decision 4: FLANN vs. Brute Force Matching

**Current Choice**: **FLANN (Fast Library for Approximate Nearest Neighbors)**

#### Matcher Comparison

| Matcher | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| **FLANN** | O(k log n) | Approximate | Large feature sets (>100) |
| **BFMatcher** | O(n²) | Exact | Small feature sets (<100) |

**Why FLANN?**
- Field blocks typically have 100-500 features
- Speed critical for batch processing
- Approximation error negligible for alignment

**Code**: `sift_matcher.py:26-30`
```python
index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
search_params = {"checks": 50}
SiftMatcher.flann = cv2.FlannBasedMatcher(index_params, search_params)
```

#### Browser Consideration

**Problem**: OpenCV.js FLANN support is limited

**Browser Alternative**:
```javascript
// Use BFMatcher instead
const bf = new cv.BFMatcher(cv.NORM_L2, true);
const matches = bf.match(desc1, desc2);

// For ORB (binary descriptors):
const bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
```

**Trade-off**: Slower in browser, but more compatible

---

### Decision 5: K=4 for K-Nearest Anchors

**Current Choice**: **K = 4 neighbors**

#### Why 4?

```python
def apply_k_nearest_interpolation_inplace(..., k=4):
    nearest_anchors = find_k_nearest_anchors(field_block_position, anchors, k)
```

**Reasoning**:
1. **Stability**: Average of 4 reduces outlier impact
2. **Local Context**: Captures nearby displacement trend
3. **Performance**: Small enough for fast computation
4. **Geometric**: Forms a quadrilateral (4 corners)

**Comparison**:

| K | Pros | Cons |
|---|------|------|
| 1 | Fastest | Highly sensitive to outliers |
| 2-3 | Fast, local | May miss displacement trend |
| **4** | **Balanced** | **Good trade-off** |
| 5-10 | More stable | May over-smooth local variations |

**Example**:
```
Field block at [200, 300]
Nearest 4 anchors:
  A1: distance=2.2, displacement=[-3, 2]
  A2: distance=2.8, displacement=[-2, 1]
  A3: distance=3.6, displacement=[-3, 2]
  A4: distance=8.6, displacement=[-4, 3]

Average = [(-3-2-3-4)/4, (2+1+2+3)/4] = [-3.0, 2.0]

If K=1: displacement = [-3, 2]  (could be outlier)
If K=4: displacement = [-3.0, 2.0]  (more stable)
```

**Trade-off**: Could be tunable parameter in future

---

### Decision 6: Lowe's Ratio Test for Match Filtering

**Current Choice**: **m.distance < n.distance** (Lowe's Ratio Test)

**Code**: `sift_matcher.py:59-76`
```python
for m, n in matches:  # m = best match, n = second-best match
    if m.distance < n.distance:  # Lowe's test
        good.append(m)
```

#### Standard Lowe's Ratio Test
```python
# Original formulation:
if m.distance < 0.7 * n.distance:
    good.append(m)
```

#### Why Simplified Version?

**OMRChecker uses**: `m.distance < n.distance` (ratio = 1.0)

**Rationale**:
- Additional filter: `max_displacement` constraint
- OMR sheets have repetitive patterns (bubbles look similar)
- Stricter ratio (0.7) rejects too many valid matches
- Combined with max_displacement provides enough filtering

**Code**:
```python
if (m.distance < n.distance and
    MathUtils.distance(source_point, dest_point) <= max_displacement):
    good.append(m)
    displacement_pairs.append([dest_point, source_point])
```

**Trade-off**: More matches (good for sparse features), but requires distance constraint

---

### Decision 7: MIN_MATCH_COUNT = 10

**Current Choice**: **Minimum 10 matches required**

**Code**: `sift_matcher.py:14, 78`
```python
MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    # Proceed with alignment
else:
    logger.critical(f"Not enough matches: {len(good)}/{MIN_MATCH_COUNT}")
```

#### Why 10?

| Count | Reliability | Coverage |
|-------|-------------|----------|
| 4 | Minimum for homography | Too few (unreliable) |
| **10** | **Good balance** | **Typical for OMR** |
| 20+ | Very robust | May fail on small blocks |

**Failure Scenario**:
```python
# If fewer than 10 matches:
# 1. Field block has very few features (solid color, blank area)
# 2. Scan quality too poor
# 3. Template-scan mismatch

# Result: shifts remain [0, 0], detection proceeds without alignment
# Better to skip alignment than use unreliable shifts
```

**Trade-off**: Conservative threshold, better false negative than false positive

---

### Decision 8: Margins for Context

**Current Choice**: **Configurable margins per field block**

**Default**: No margins (0, 0, 0, 0)
**Recommended**: 20-50 pixels per side

#### Why Margins?

```
Without Margins:                With Margins (50px):
┌─────────────┐                ╔═══════════════════╗
│             │                ║   ┌─────────────┐ ║
│  Field      │                ║   │             │ ║
│  Block      │                ║   │  Field      │ ║
│             │                ║   │  Block      │ ║
│             │                ║   │             │ ║
└─────────────┘                ║   └─────────────┘ ║
                               ╚═══════════════════╝

Features: Few                  Features: Many more!
SIFT may fail                  SIFT has context
```

**Code**: `template_alignment.py:49-52`
```python
margins = field_block_alignment.get("margins", template_margins)

zone_start = [
    int(bounding_box_origin[0] - margins["left"]),
    int(bounding_box_origin[1] - margins["top"]),
]
```

**Trade-off**:
- **Larger margins**: More features, better alignment, slower
- **Smaller margins**: Faster, may fail on feature-poor blocks

**Recommendation**: Start with 50px, tune per template

---

### Decision 9: Homography Estimation with RANSAC

**Current Choice**: **RANSAC for outlier rejection**

**Code**: `sift_matcher.py:91-93`
```python
homography_matrix, mask = cv2.findHomography(
    src_pts, dst_pts, cv2.RANSAC, max_displacement
)
matches_mask = mask.ravel().tolist()
```

#### Why RANSAC?

| Method | Outlier Handling | Speed | Accuracy |
|--------|------------------|-------|----------|
| **None** | No | Fast | Poor (sensitive to outliers) |
| **LMEDS** | Yes (50% threshold) | Medium | Good |
| **RANSAC** | Yes (adaptive) | Medium | Excellent |

**RANSAC Benefits**:
- Robust to 20-30% outlier matches
- Adaptive threshold based on `max_displacement`
- Returns inlier mask for visualization

**Example**:
```python
# 20 matches total:
# 15 correct (inliers)
# 5 incorrect (outliers - e.g., similar-looking bubbles)

# RANSAC:
# 1. Randomly sample 4 matches
# 2. Compute homography
# 3. Test all matches against homography
# 4. Keep if error < max_displacement
# 5. Repeat, keep best homography

# Result: mask = [1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,0,1]
#                 (15 inliers, 5 outliers rejected)
```

**Trade-off**: Slightly slower, but essential for reliable alignment

---

### Decision 10: Zone Clipping Strategy

**Current Choice**: **Implicit clipping via NumPy slicing**

**Code**: `template_alignment.py:77-84`
```python
block_gray_image = gray_image[zone_start[1]:zone_end[1], zone_start[0]:zone_end[0]]
# NumPy handles out-of-bounds gracefully
```

#### Behavior

```python
# Image shape: [800, 600]
# Zone: [750:850, 500:650]  ← Extends beyond image

# NumPy automatically clips:
# actual_zone = [750:800, 500:600]

# Result: Smaller ROI than expected, but no error
```

**Alternative Considered**: Explicit clipping
```python
# NOT used:
zone_start = [
    max(0, zone_start[0]),
    max(0, zone_start[1])
]
zone_end = [
    min(image.shape[1], zone_end[0]),
    min(image.shape[0], zone_end[1])
]
```

**Why Implicit?**
- Cleaner code
- NumPy is well-tested
- Rare edge case (margins usually don't exceed image)

**Trade-off**: Less explicit, but simpler

---

### Decision 11: Singleton SIFT/FLANN Initialization

**Current Choice**: **Module-level singleton**

**Code**: `sift_matcher.py:18-31, 146`
```python
class SiftMatcher:
    sift = None
    flann = None

    def singleton_init() -> None:
        SiftMatcher.sift = cv2.SIFT_create()
        SiftMatcher.flann = cv2.FlannBasedMatcher(...)

# Called once at import
SiftMatcher.singleton_init()
```

#### Why Singleton?

**Performance**:
```python
# Without singleton (per field block):
# Image with 10 field blocks:
# 10 × cv2.SIFT_create() = overhead
# 10 × FlannBasedMatcher() = overhead

# With singleton:
# 1 × cv2.SIFT_create() = shared across all blocks
# 1 × FlannBasedMatcher() = shared across all blocks
```

**Memory**:
- SIFT detector includes internal state
- FLANN includes index structures
- Reusing saves ~10-50 MB per field block

**Trade-off**: Thread-safety concerns (but Python GIL protects for now)

---

## Rejected Alternatives

### 1. Template Matching (Cross-Correlation)

**Why Rejected**: Cannot handle rotation or scale variance

**Code Reference**: `template_alignment.py` (separate file exists but not used)

### 2. Feature-less Alignment (Edge Detection)

**Why Rejected**: OMR sheets have similar edges, hard to distinguish

### 3. Homography-Based Image Warping

**Why Rejected**: Cannot handle overlapping field blocks

**Code**: Commented in `template_alignment.py:95-114`

### 4. Per-Bubble Alignment

**Why Rejected**: Too computationally expensive

**Code**: Commented in `k_nearest_interpolation.py:163-181`
```python
# def shift_by_scan_boxes(...):  # COMMENTED OUT
#     for scan_box in field.scan_boxes:
#         # Individual alignment per bubble - TOO SLOW
```

---

## Future Considerations

### 1. Adaptive K Value
```python
# Current: k = 4 (fixed)
# Future: Adapt based on feature density
k = min(4, len(displacement_pairs) // 2)
```

### 2. Multi-Level Alignment
```python
# Current: Single-pass per field block
# Future: Coarse-to-fine (global → block → field → bubble)
```

### 3. Machine Learning-Based Alignment
```python
# Use CNN to predict displacement directly
# Faster, no feature detection needed
```

### 4. Optical Flow
```python
# Dense optical flow for pixel-level alignment
# Better for severe distortions
```

---

## Browser-Specific Decisions

### Use ORB Instead of SIFT
```javascript
// SIFT may not be available
const orb = new cv.ORB_create(500);  // 500 features
```

### Use BFMatcher Instead of FLANN
```javascript
// FLANN less reliable in OpenCV.js
const bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
```

### Web Worker for Alignment
```javascript
// Offload heavy computation
const worker = new Worker('alignment-worker.js');
```

### Progressive Alignment
```javascript
// Align and display incrementally
for (const fieldBlock of fieldBlocks) {
  await alignFieldBlock(fieldBlock);
  updateUI();  // Show progress
}
```

---

## Summary of Key Trade-offs

| Decision | Choice | Trade-off |
|----------|--------|-----------|
| Warp what? | Coordinates | Flexibility vs. Complexity |
| Granularity | Per-field-block | Accuracy vs. Speed |
| Feature detector | SIFT | Quality vs. Browser compatibility |
| Matcher | FLANN | Speed vs. Accuracy |
| K neighbors | 4 | Stability vs. Locality |
| Min matches | 10 | Reliability vs. Coverage |
| Margins | Configurable | Context vs. Performance |
| Outlier rejection | RANSAC | Robustness vs. Speed |
| Initialization | Singleton | Performance vs. Thread-safety |

All decisions optimize for **accuracy and flexibility** while maintaining **acceptable performance** for batch OMR processing.
