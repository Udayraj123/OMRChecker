# SIFT Alignment - Execution Flow

## Overview

**SIFT (Scale-Invariant Feature Transform)** is the primary feature detection method used in OMRChecker's alignment system. It detects keypoints in images that are invariant to scale, rotation, and illumination changes.

**File**: `src/processors/alignment/sift_matcher.py`

---

## Initialization Flow

### Singleton Pattern

```python
# Module-level initialization (runs at import time)
class SiftMatcher:
    sift = None
    flann = None

    def singleton_init() -> None:
        # Step 1: Create SIFT detector
        SiftMatcher.sift = cv2.SIFT_create()

        # Step 2: Configure FLANN matcher
        index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
        search_params = {"checks": 50}

        # Step 3: Initialize FLANN matcher
        SiftMatcher.flann = cv2.FlannBasedMatcher(index_params, search_params)

# Called once at module load
SiftMatcher.singleton_init()
```

**Why Singleton?**
- SIFT detector creation is expensive (~10-50ms)
- FLANN index structure reused across all matching operations
- Shared across all field blocks in batch processing

**FLANN Parameters**:
- `FLANN_INDEX_KDTREE = 0`: K-D tree algorithm for fast nearest neighbor search
- `trees = 5`: Number of randomized k-d trees (more = more accurate, slower)
- `checks = 50`: Maximum leaf nodes to visit during search (trade-off: speed vs accuracy)

---

## Feature Detection & Matching Flow

### High-Level Steps

```
1. Detect SIFT features in both images
   ↓
2. Match features using FLANN
   ↓
3. Filter matches (Lowe's ratio + displacement constraint)
   ↓
4. Return displacement pairs
```

---

### Step 1: Feature Detection

**Code**: `sift_matcher.py:41-42`

```python
def get_matches(field_block_name, gray_image, alignment_image, max_displacement, config):
    # Detect keypoints and compute descriptors
    source_features, des1 = SiftMatcher.sift.detectAndCompute(alignment_image, None)
    destination_features, des2 = SiftMatcher.sift.detectAndCompute(gray_image, None)
```

#### What SIFT Detects

**Input**: Grayscale image (8-bit, single channel)

**Output**:
1. **Keypoints** (`source_features`, `destination_features`):
   - Location: `(x, y)` pixel coordinates
   - Scale: Pyramid level where detected
   - Orientation: Dominant gradient direction
   - Response: Strength of feature

2. **Descriptors** (`des1`, `des2`):
   - 128-dimensional vector per keypoint
   - Histogram of gradient orientations
   - Invariant to scale, rotation, partial illumination

#### Example

```python
# Image: 400x300 pixels
source_features, des1 = sift.detectAndCompute(alignment_image, None)

# Typical output:
len(source_features) = 237  # 237 keypoints detected
des1.shape = (237, 128)     # 237 descriptors, 128 dimensions each

# Keypoint structure:
keypoint = source_features[0]
keypoint.pt = (152.3, 87.9)      # Sub-pixel location
keypoint.size = 15.2             # Scale (diameter)
keypoint.angle = 47.3            # Orientation (degrees)
keypoint.response = 0.089        # Strength
```

#### Feature Distribution

```
Blank regions:       0-10 features
Simple patterns:     50-200 features
Complex patterns:    200-1000+ features
Text-heavy regions:  500-2000 features
```

---

### Step 2: FLANN Matching

**Code**: `sift_matcher.py:52`

```python
matches = SiftMatcher.flann.knnMatch(des1, des2, k=2)
```

#### KNN Matching (k=2)

**Input**:
- `des1`: Descriptors from alignment image (query)
- `des2`: Descriptors from scanned image (train)
- `k=2`: Find 2 nearest neighbors for each query descriptor

**Output**: List of match pairs
```python
matches = [
    [DMatch(best_match), DMatch(second_best_match)],
    [DMatch(best_match), DMatch(second_best_match)],
    # ... one pair per query descriptor
]

# DMatch structure:
match.queryIdx = 0        # Index in des1 (alignment image)
match.trainIdx = 15       # Index in des2 (scanned image)
match.distance = 87.3     # L2 distance between descriptors (lower = better)
```

#### FLANN Algorithm

```
1. Build k-d tree index from des2 (training descriptors)
   - Tree depth ≈ log₂(n) where n = number of descriptors
   - 5 randomized trees for robustness

2. For each descriptor in des1:
   a. Traverse k-d tree to find approximate nearest neighbor
   b. Check up to 50 leaf nodes (search_params.checks)
   c. Return top 2 matches (k=2)

3. Time complexity: O(m × log n) where m = query count, n = train count
   - Much faster than brute force O(m × n)
```

---

### Step 3: Match Filtering

**Code**: `sift_matcher.py:55-76`

```python
good = []
displacement_pairs = []

for m, n in matches:
    # Extract feature point locations
    source_feature_point = source_features[m.queryIdx].pt
    destination_feature_point = destination_features[m.trainIdx].pt

    # Filter 1: Lowe's ratio test (simplified)
    # Filter 2: Max displacement constraint
    if (m.distance < n.distance and
        MathUtils.distance(source_feature_point, destination_feature_point) <= max_displacement):
        good.append(m)

        # Store displacement pair (reversed for warping direction)
        displacement_pairs.append([destination_feature_point, source_feature_point])
```

#### Filter 1: Lowe's Ratio Test

**Standard Formulation**:
```python
if m.distance < 0.7 * n.distance:
    # Accept match
```

**OMRChecker Simplification**:
```python
if m.distance < n.distance:
    # Accept match (ratio = 1.0)
```

**Rationale**:
- OMR sheets have repetitive patterns (bubbles look similar)
- Stricter ratio (0.7) rejects too many valid matches
- Additional `max_displacement` filter compensates

**Example**:
```python
# Match for keypoint A:
m.distance = 85.3  # Best match distance
n.distance = 92.1  # Second-best match distance

# Standard test (ratio=0.7):
85.3 < 0.7 × 92.1 = 64.47  → FALSE (reject)

# OMRChecker test (ratio=1.0):
85.3 < 92.1  → TRUE (accept, then check displacement)
```

#### Filter 2: Max Displacement Constraint

```python
# Euclidean distance between matched points
displacement = MathUtils.distance(source_point, destination_point)
displacement = √((x₂ - x₁)² + (y₂ - y₁)²)

if displacement <= max_displacement:
    # Accept
```

**Example**:
```python
# max_displacement = 30 pixels
source_point = (100, 200)       # From alignment image
destination_point = (105, 198)  # From scanned image

displacement = √((105-100)² + (198-200)²)
             = √(25 + 4)
             = √29 ≈ 5.4 pixels

5.4 <= 30  → TRUE (accept)
```

#### Displacement Pair Structure

```python
# Reversed order for warping direction:
displacement_pairs.append([destination_point, source_point])

# Interpretation:
# - destination_point: Where feature IS in scanned image
# - source_point: Where feature SHOULD BE (template)
# - Displacement: How to move from scanned → template
```

---

### Step 4: Verify Minimum Matches

**Code**: `sift_matcher.py:78-117`

```python
MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    # Sufficient matches → Proceed with homography
    src_pts = np.float32([source_features[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([destination_features[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Estimate homography with RANSAC
    homography_matrix, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, max_displacement
    )
    matches_mask = mask.ravel().tolist()

else:
    # Insufficient matches → Alignment fails
    logger.critical(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
    matches_mask = None
```

#### Homography Estimation

**Purpose**: Validate matches using geometric consistency

**RANSAC Algorithm**:
```
1. Randomly sample 4 matches
2. Compute homography from these 4 point pairs
3. Count inliers (matches consistent with homography)
4. Repeat N iterations (typically 100-1000)
5. Keep homography with most inliers
6. Refine using all inliers
```

**Output**:
```python
homography_matrix = [
    [a, b, c],
    [d, e, f],
    [g, h, 1]
]  # 3x3 matrix

# Transforms point from alignment → scanned:
[x', y', w'] = homography_matrix × [x, y, 1]
x_final = x' / w'
y_final = y' / w'
```

**Mask** (inlier/outlier labels):
```python
mask = [
    [1],  # Inlier (match 0 consistent with homography)
    [1],  # Inlier
    [0],  # Outlier (rejected by RANSAC)
    [1],  # Inlier
    # ...
]

matches_mask = [1, 1, 0, 1, ...]  # Flattened
```

---

### Step 5: Optional Visualization

**Code**: `sift_matcher.py:119-140`

```python
if config.outputs.show_image_level >= 6:
    draw_params = {
        "matchColor": (0, 255, 0),        # Green for matches
        "singlePointColor": None,
        "matchesMask": matches_mask,      # Only draw inliers
        "flags": 2,
    }

    display_feature_matches = cv2.drawMatches(
        alignment_image,
        source_features,
        gray_image,
        destination_features,
        good,
        None,
        **draw_params,
    )

    InteractionUtils.show(f"Matches for {field_block_name}", display_feature_matches, 0)
```

**Output**:
```
┌──────────────┬──────────────┐
│ Alignment    │ Scanned      │
│ Image        │ Image        │
│              │              │
│  ●────────────────→●         │  Green line = good match
│    ●──────────────→●         │
│      ●──X  (outlier, red)   │
└──────────────┴──────────────┘
```

---

### Step 6: Return Displacement Pairs

```python
return displacement_pairs  # List of [[dest_pt, src_pt], ...]
```

**Used by**: `k_nearest_interpolation.py` to compute shifts

---

## Alternative Flow: SIFT + Piecewise Affine (Commented)

**File**: `sift_matcher.py:149-207`

This method warps the IMAGE instead of coordinates.

### Flow

```
1. get_matches() → displacement_pairs
   ↓
2. Add corner points (for complete triangulation)
   ↓
3. apply_piecewise_affine() → warped_block_image
   ↓
4. Replace ROI in original image
```

### Code (Commented in template_alignment.py:95-114)

```python
# Method 2: SIFT + Piecewise Affine
# warped_block_image, warped_colored_image = apply_sift_shifts(
#     field_block_name,
#     block_gray_image,
#     block_colored_image,
#     block_gray_alignment_image,
#     max_displacement,
#     margins,
#     bounding_box_dimensions,
# )

# # Set warped field block back into original image
# gray_image[zone_start[1]:zone_end[1], zone_start[0]:zone_end[0]] = warped_block_image
```

### Why Commented?

**Problem**: Cannot handle overlapping field blocks
- If two field blocks overlap, which warped image wins?
- Overwrites cause conflicts

**Current Solution**: Warp coordinates instead (Method 3)

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| SIFT Detection | O(n × log n) | 10-50ms per image |
| FLANN Matching | O(m × log n) | 5-20ms |
| Filtering | O(m) | < 1ms |
| Homography (RANSAC) | O(k × n) | 5-10ms |
| **Total per field block** | - | **20-80ms** |

Where:
- n = number of keypoints (50-500)
- m = number of matches (10-200)
- k = RANSAC iterations (100-1000)

### Memory Usage

```python
# Per field block:
# Keypoints: 200 × 32 bytes = 6.4 KB
# Descriptors: 200 × 128 × 4 bytes = 100 KB
# Matches: 100 × 24 bytes = 2.4 KB
# Total: ~110 KB

# Template with 20 field blocks:
# Peak: 20 × 110 KB = 2.2 MB
```

---

## Edge Cases

### 1. No Features Detected

```python
source_features, des1 = sift.detectAndCompute(blank_image, None)
# source_features = []  # Empty list
# des1 = None or empty array

# flann.knnMatch(empty_des, des2, k=2)
# → Returns empty matches
# → len(good) = 0 < MIN_MATCH_COUNT
# → Alignment fails gracefully
```

### 2. Insufficient Matches

```python
len(good) = 7  # Less than MIN_MATCH_COUNT=10
# logger.critical("Not enough matches...")
# matches_mask = None
# displacement_pairs still returned (7 pairs)
# k_nearest_interpolation will use them (risky but allowed)
```

### 3. All Matches Rejected by Displacement Filter

```python
# Scenario: Image completely misaligned (> max_displacement everywhere)
for m, n in matches:
    if distance(src_pt, dst_pt) <= max_displacement:
        # Never true!

# Result: good = []
# Same as Case 2
```

### 4. Homography Fails (Degenerate Points)

```python
# All matched points collinear:
src_pts = [[100,200], [150,300], [200,400], [250,500]]

homography_matrix, mask = cv2.findHomography(...)
# homography_matrix may be None or ill-conditioned
# mask all zeros (all outliers)

# But displacement_pairs still valid for k-nearest averaging
```

---

## Browser Migration Notes

### SIFT Availability

**Problem**: OpenCV.js may not include SIFT (build-dependent)

**Detection**:
```javascript
if (typeof cv.SIFT_create === 'undefined') {
  console.error('SIFT not available in this OpenCV.js build');
}
```

### Alternative: ORB

```javascript
// ORB (Oriented FAST and Rotated BRIEF)
const orb = new cv.ORB_create(500);  // Max 500 features
const keypoints = new cv.KeyPointVector();
const descriptors = new cv.Mat();

orb.detectAndCompute(image, new cv.Mat(), keypoints, descriptors);

// Use BFMatcher (FLANN less reliable in browser)
const bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
const matches = bf.match(desc1, desc2);
```

**Differences**:
- ORB: Binary descriptors (256 bits), HAMMING distance
- SIFT: Float descriptors (128 × 32-bit), L2 distance
- ORB: Faster but less accurate

### Alternative: AKAZE

```javascript
// AKAZE (Accelerated-KAZE)
const akaze = new cv.AKAZE_create();
akaze.detectAndCompute(image, new cv.Mat(), keypoints, descriptors);

// Use BFMatcher
const bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
const matches = bf.match(desc1, desc2);
```

**Advantage**: Good accuracy, patent-free, often available in OpenCV.js

### Memory Management

```javascript
// CRITICAL: Explicitly delete cv.Mat objects
const keypoints = new cv.KeyPointVector();
const descriptors = new cv.Mat();

try {
  sift.detectAndCompute(image, new cv.Mat(), keypoints, descriptors);
  // ... use descriptors
} finally {
  keypoints.delete();
  descriptors.delete();
}
```

### Async Processing

```javascript
// Offload to Web Worker
const worker = new Worker('sift-worker.js');

worker.postMessage({
  type: 'detect',
  imageData: imageData,
});

worker.onmessage = (event) => {
  const { keypoints, descriptors } = event.data;
  // Continue processing
};
```

---

## Related Flows

- **K-Nearest Interpolation** (`../k-nearest/flows.md`) - Uses displacement pairs
- **Piecewise Affine** (`../piecewise-affine/flows.md`) - Alternative warping method
- **Phase Correlation** (`../phase-correlation/flows.md`) - Feature-less alternative
- **Alignment Main Flow** (`../flows.md`) - Overall alignment orchestration

---

## Testing Considerations

### Unit Tests

```python
def test_sift_matching():
    # Perfect alignment (no shift)
    displacement_pairs = SiftMatcher.get_matches(
        "test_block", image, image, max_displacement=30, config
    )
    # All displacements should be near [0, 0]
    assert all(distance(dest, src) < 2 for dest, src in displacement_pairs)

def test_sift_with_shift():
    # Shifted image
    shifted_image = shift_image(image, dx=10, dy=5)
    displacement_pairs = SiftMatcher.get_matches(
        "test_block", shifted_image, image, max_displacement=30, config
    )
    # Displacements should be near [10, 5]
    avg_displacement = np.mean([
        [dest[0]-src[0], dest[1]-src[1]]
        for dest, src in displacement_pairs
    ], axis=0)
    assert abs(avg_displacement[0] - 10) < 2
    assert abs(avg_displacement[1] - 5) < 2
```

### Edge Case Tests

1. Blank image (no features)
2. Insufficient matches (< MIN_MATCH_COUNT)
3. Extreme displacement (> max_displacement)
4. Rotation (SIFT should handle)
5. Scale change (SIFT should handle)
