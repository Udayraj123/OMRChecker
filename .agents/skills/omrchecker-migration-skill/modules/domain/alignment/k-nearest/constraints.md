# K-Nearest Interpolation - Constraints & Edge Cases

## Core Constraints

### 1. K Value (Number of Nearest Neighbors)

**Default**: K = 4

**Constraint**: K must be positive integer

**Rationale**:
- Too small (K=1): Unstable, sensitive to outliers
- Too large (K=10+): Over-smoothing, ignores local variations
- K=4: Good balance for typical OMR sheets

**Trade-offs**:

```python
# K = 1 (single nearest neighbor):
# - Fast computation
# - Very sensitive to noise/outliers
# - Sharp discontinuities between regions

# K = 4 (default):
# - Balanced smoothing
# - Robust to single outliers
# - Local variations preserved

# K = 10+:
# - Heavy smoothing
# - Very robust to noise
# - May miss local distortions
```

**Edge Case**: K larger than available anchors
```python
k = 4
displacement_pairs = [pair1, pair2]  # Only 2 pairs

# sorted_by_distance[:4] returns only 2
# Result: Uses all available (no error, just warning-worthy)
```

**Recommendation**:
- Flatbed scanners: K = 3-5
- Mobile photos: K = 5-8 (more distortion)
- Low-quality scans: K = 6-10 (more noise)

---

### 2. Minimum Displacement Pairs

**Current**: No explicit minimum (relies on SIFT MIN_MATCH_COUNT=10)

**Issue**: SIFT may return < 10 matches and log critical, but still pass pairs

**Problem Scenario**:
```python
# SIFT returns 7 matches (below MIN_MATCH_COUNT=10)
# logger.critical("Not enough matches are found - 7/10")
# But still returns 7 displacement_pairs

# K-nearest uses these 7 pairs:
k = 4
nearest_anchors = find_k_nearest_anchors(origin, 7_pairs, k)
# Returns 4 anchors (seems OK)

# BUT: Those 7 matches may be unreliable!
# SIFT's homography couldn't be computed confidently
```

**Recommended Fix**:
```python
MIN_DISPLACEMENT_PAIRS = 5

def apply_k_nearest_interpolation_inplace(...):
    displacement_pairs = SiftMatcher.get_matches(...)

    if len(displacement_pairs) < MIN_DISPLACEMENT_PAIRS:
        logger.warning(f"Insufficient displacement pairs: {len(displacement_pairs)}")
        # Return early without modifying shifts
        return [0, 0]

    # Continue with k-nearest...
```

**Failure Mode**: Zero displacement pairs
```python
displacement_pairs = []  # SIFT completely failed

# anchors_with_displacements = []
# find_k_nearest_anchors() returns []
# np.average([]) → RuntimeWarning + [nan, nan]
# scan_box.shifts = [nan, nan]
# Detection fails catastrophically!
```

---

### 3. Coordinate System Constraints

**Constraint**: All coordinates must be in consistent coordinate space

**Coordinate Spaces**:

1. **Template Space**: Original template.json coordinates
2. **ROI Space**: Relative to extracted zone
3. **Shifted Space**: After applying shifts

**Conversions**:
```python
# Template → ROI (block_image_shifts):
block_image_shifts = subtract_points(
    [margins["left"], margins["top"]],
    field_block.origin
)

# ROI → Template (reverse):
template_coords = add_points(roi_coords, field_block.origin) - [margins.left, margins.top]
```

**Edge Case**: Coordinate overflow
```python
# Displacement calculation:
anchor_point = [10000, 15000]  # Far from typical range
displaced_point = [9950, 14980]
displacement = [50, 20]  # Still valid

# But if used for shift:
scan_box.position = [100, 200]
shifted_position = [100 + 50, 200 + 20] = [150, 220]
# Within bounds ✓

# Extreme case:
displacement = [10000, 15000]  # Absurdly large
shifted_position = [10100, 15200]
# Outside image bounds → detection fails
```

**Validation** (not currently done):
```python
MAX_REASONABLE_SHIFT = 500  # pixels

if abs(average_shifts[0]) > MAX_REASONABLE_SHIFT or \
   abs(average_shifts[1]) > MAX_REASONABLE_SHIFT:
    logger.error(f"Unreasonable shift detected: {average_shifts}")
    return [0, 0]
```

---

### 4. Floating Point Precision

**Constraint**: Sub-pixel precision lost during integer conversion

**Issue**: Truncation instead of rounding

```python
displacements = [
    [5, 2],
    [5, 3],
    [6, 4],
    [5, -2],
]

average = np.average(displacements, axis=0)
       = [5.25, 1.75]

# Current implementation:
average_shifts = average.astype(np.int32)
              = [5, 1]  # TRUNCATES

# Better approach:
average_shifts = np.round(average).astype(np.int32)
              = [5, 2]  # ROUNDS
```

**Impact**:
```python
# Scenario: True average shift is 5.75 pixels
# Truncation: 5 pixels → 0.75 pixel error
# Over 10 bubbles: Up to 7-8 pixel cumulative error

# For typical bubbles (20-30px diameter):
# 0.75 pixel error ≈ 3-4% error
# Usually acceptable, but rounding would be better
```

**Edge Case**: Negative averages
```python
displacements = [[-2, 3], [-3, 2], [-2, 4], [-3, 1]]
average = np.average(displacements, axis=0)
       = [-2.5, 2.5]

# Truncation:
.astype(np.int32) → [-2, 2]

# Rounding:
np.round() → [-2, 2]  # Same for this case

# BUT for [-2.8, 2.2]:
# Truncation: [-2, 2]
# Rounding: [-3, 2]  # More accurate
```

**NumPy Behavior**:
```python
np.int32(5.9) → 5    # Truncates toward zero
np.int32(-5.9) → -5  # Truncates toward zero

np.round(5.9) → 6.0  # Rounds to nearest even (banker's rounding)
np.round(-5.9) → -6.0
```

---

### 5. Distance Calculation Constraints

**Constraint**: Euclidean distance only (no Manhattan or other metrics)

**Current**:
```python
distance = math.hypot(dx, dy)  # √(dx² + dy²)
```

**Alternative Metrics** (not used):

#### Manhattan Distance
```python
distance = abs(dx) + abs(dy)
# Faster (no sqrt), but different topology
```

#### Chebyshev Distance
```python
distance = max(abs(dx), abs(dy))
# Even faster, but very different selection
```

**Why Euclidean?**
- Natural distance metric for 2D plane
- Isotropic (same in all directions)
- Standard for K-nearest neighbors

**Performance**:
```python
# For 50 anchors:
# Euclidean: 50 × sqrt = ~50µs
# Manhattan: 50 × (2 abs + 1 add) = ~10µs
# Negligible difference for OMR use case
```

---

### 6. Sorting Stability

**Constraint**: When multiple anchors have same distance, order is implementation-dependent

**Python's `sorted()` is STABLE**:
```python
# If two anchors have identical distance:
distance_list = [
    [5.0, [[100, 200], [3, 2]]],
    [5.0, [[105, 195], [4, 1]]],  # Same distance!
    [3.2, [[110, 198], [2, 3]]],
]

sorted_by_distance = sorted(distance_list, key=lambda x: x[0])
# Preserves original order for ties:
# [3.2, ...], [5.0, [[100, 200], ...]], [5.0, [[105, 195], ...]]
```

**JavaScript's `.sort()` is NOT STABLE (before ES2019)**:
```javascript
// Same distance, unpredictable order:
distance_list.sort((a, b) => a[0] - b[0]);
// May swap order of tied elements
```

**Solution for JavaScript**:
```javascript
// ES2019+: Array.sort() is now stable
// OR add secondary sort key:
distance_list.sort((a, b) => {
  if (a[0] !== b[0]) return a[0] - b[0];
  return a[1][0][0] - b[1][0][0];  // Use anchor x-coord as tiebreaker
});
```

---

### 7. Memory Constraints

**Constraint**: Must store distance-anchor pairs for all anchors before sorting

**Memory Usage**:
```python
# Per field block:
# Input: 50 displacement pairs
# Distance list: 50 × (8 bytes float + 32 bytes anchor data) = 2 KB
# Sorted list: 2 KB (same data, reordered)
# Nearest K: 4 × 32 bytes = 128 bytes
# Total: ~5 KB per field block

# Template with 100 field blocks:
# Peak: 500 KB (manageable)
```

**Edge Case**: Very large templates
```python
# 500 field blocks × 200 anchors each:
# 500 × 200 × 40 bytes = 4 MB
# Still acceptable
```

**Browser Constraint**:
- Heap limits vary (1-4 GB)
- For large templates, consider streaming/chunking

**Optimization** (not implemented):
```python
# Don't build full distance list, use heap:
import heapq

def find_k_nearest_anchors_optimized(origin, anchors, k):
    # O(n log k) instead of O(n log n)
    heap = []
    for anchor_point, displacement in anchors:
        dist = MathUtils.distance(origin, anchor_point)
        heapq.heappush(heap, (dist, [anchor_point, displacement]))

    return [heapq.heappop(heap)[1] for _ in range(min(k, len(heap)))]
```

**Benefit**: Slightly faster for large anchor sets (50+ anchors)

---

### 8. Granularity Constraint

**Constraint**: All bubbles in field block get SAME shift

**Current Approach**: Field block-level shifts
```python
# Single shift for entire field block:
average_shifts = [5, 2]

for field in field_block.fields:
    for scan_box in field.scan_boxes:
        scan_box.shifts = average_shifts  # All identical
```

**Assumption**: Rigid transformation within field block
- No rotation
- No scaling
- No local warping

**When Valid**:
- Flatbed scans (minimal distortion)
- Small field blocks (< 500px × 500px)
- Pre-processed images (already aligned roughly)

**When Breaks Down**:
- Large field blocks (> 1000px × 1000px)
- Mobile photos with lens distortion
- Partial page scans with extreme perspective

**Alternative Granularities** (commented in code):

#### Option 1: Field-Level (Lines 134-160)
```python
# Separate shift per field (not field block):
for field in field_block.fields:
    field_center = compute_center(field.scan_boxes)
    nearest = find_k_nearest_anchors(field_center, anchors, k)
    field_shifts = average(nearest)

    for scan_box in field.scan_boxes:
        scan_box.shifts = field_shifts
```

**Trade-off**: More localized, slower

#### Option 2: Bubble-Level (Lines 163-181)
```python
# Separate shift per bubble:
for scan_box in all_scan_boxes:
    nearest = find_k_nearest_anchors(scan_box.position, anchors, k)
    scan_box.shifts = average(nearest)
```

**Trade-off**: Most precise, much slower, noisier

**Why Field Block-Level Chosen?**
- Speed: One K-nearest query per field block
- Stability: Averaging over larger area reduces noise
- Sufficient for most OMR use cases

---

### 9. Empty Field Block Constraint

**Constraint**: Field blocks must have at least one field with scan boxes

**Edge Case**: Empty field block
```python
field_block.fields = []  # No fields

# shift_by_field_blocks() computes:
field_block_position = add_points(field_block.origin, block_image_shifts)
# Uses origin, not bubble positions (OK)

# But if applying shifts:
for field in field_block.fields:  # Empty loop
    for scan_box in field.scan_boxes:
        scan_box.shifts = average_shifts
# No-op (shifts not applied to anything)
```

**Impact**: Harmless (no shifts applied, but no bubbles to detect anyway)

---

### 10. Centered Mode Constraint

**Constraint**: `centered=False` is hardcoded in call

**Code**:
```python
average_shifts = shift_by_field_blocks(
    field_block, block_image_shifts, anchors_with_displacements, k
)
# No centered parameter passed → defaults to False
```

**Centered Mode** (unused):
```python
if centered:
    # Compute geometric center of all bubbles:
    field_block_position = np.average([
        np.average([
            scan_box.get_shifted_position(block_image_shifts)
            for scan_box in field.scan_boxes
        ], axis=0)
        for field in field_block.fields
    ], axis=0).astype(np.int32)
else:
    # Use top-left corner:
    field_block_position = add_points(field_block.origin, block_image_shifts)
```

**Why Centered=False?**
- Simpler (no nested averaging)
- Faster (no need to iterate over all bubbles)
- Sufficient (field block corner is close enough to center for K-nearest purposes)

**When Centered Might Help**:
- Very large field blocks where corner is far from actual bubble distribution
- Irregularly shaped field blocks

**Performance**:
```python
# centered=False: O(1)
# centered=True: O(b) where b = number of bubbles

# For 100 bubbles:
# centered=True adds ~0.1-0.5ms (negligible)
```

---

## Edge Cases

### 1. Zero Displacement Pairs

**Scenario**: SIFT matching completely failed

**Current Behavior**:
```python
displacement_pairs = []

anchors_with_displacements = []

nearest_anchors = find_k_nearest_anchors(origin, [], k)
# Returns []

average_shifts = np.average([], axis=0)
# RuntimeWarning: Mean of empty slice
# Result: nan

scan_box.shifts = [nan, nan]
# Detection will fail!
```

**Fix Needed**:
```python
if len(displacement_pairs) == 0:
    logger.warning("No displacement pairs, skipping alignment")
    return [0, 0]  # Use template coordinates
```

---

### 2. Single Displacement Pair

**Scenario**: Only 1 SIFT match

```python
displacement_pairs = [[[100, 200], [95, 198]]]

anchors_with_displacements = [[[100, 200], [5, 2]]]

k = 4
nearest_anchors = find_k_nearest_anchors(origin, anchors, k)
# Returns only 1 anchor (not 4)

average_shifts = np.average([[5, 2]], axis=0)
             = [5, 2]
# No averaging, just uses the single displacement
```

**Risk**: That one match could be an outlier

**Better Handling**:
```python
if len(displacement_pairs) < 3:
    logger.warning("Too few matches for reliable k-nearest")
    return [0, 0]
```

---

### 3. K Larger than Available Anchors

**Scenario**: K=4, but only 2 displacement pairs

```python
k = 4
displacement_pairs = [pair1, pair2]

anchors_with_displacements = [anchor1, anchor2]

sorted_by_distance = [
    [dist1, anchor1],
    [dist2, anchor2],
]

nearest_anchors = sorted_by_distance[:4]
# Python slice returns all available (2 items, not 4)

average_shifts = np.average([disp1, disp2], axis=0)
# Uses 2 instead of 4
```

**Impact**: Less stable averaging, but doesn't crash

**Better Handling**:
```python
if len(displacement_pairs) < k:
    logger.warning(f"Only {len(displacement_pairs)} anchors available (K={k})")
    # Could reduce K dynamically, or warn user
```

---

### 4. All Anchors Equidistant

**Scenario**: Perfectly symmetric anchor placement

```python
origin = [200, 200]

anchors = [
    [[200, 100], [5, 2]],  # North
    [[300, 200], [6, 3]],  # East
    [[200, 300], [4, 1]],  # South
    [[100, 200], [5, 2]],  # West
]

# All at distance 100 from origin:
distances = [100, 100, 100, 100]

# Sorted order depends on stable sort:
# Python: Preserves original order
# JavaScript (pre-ES2019): Unpredictable
```

**Impact**: Minor (order doesn't affect average much if all selected)

---

### 5. Negative Displacements

**Scenario**: Features moved left/up in scan

```python
# Feature moved LEFT by 5, UP by 3:
anchor_point = [100, 200]     # In scan
displaced_point = [105, 203]  # In template

displacement = subtract_points(anchor_point, displaced_point)
            = [100 - 105, 200 - 203]
            = [-5, -3]

# Negative displacements are VALID
# Mean shifts bubbles LEFT and UP
```

**Impact**: No issue (negative shifts handled correctly)

---

### 6. Extreme Outlier Displacement

**Scenario**: One anchor has absurd displacement

```python
anchors = [
    [[100, 200], [5, 2]],
    [[150, 250], [6, 3]],
    [[200, 300], [5, 2]],
    [[125, 225], [500, 400]],  # OUTLIER!
]

# If K=4 and origin near [125, 225]:
nearest_anchors = all 4

average_shifts = np.average([[5, 2], [6, 3], [5, 2], [500, 400]], axis=0)
              = [129, 101.75]
              → [129, 101]  # Absurdly large!
```

**Problem**: K-nearest doesn't filter outliers (relies on SIFT filtering)

**Mitigation**: SIFT's `max_displacement` filter should prevent this
```python
# SIFT filtering (in sift_matcher.py):
if distance(source, dest) <= max_displacement:
    # Accept
# So displacement = 500 would be rejected if max_displacement=30
```

**But**: K-nearest could add additional outlier detection
```python
# After averaging, sanity check:
if np.linalg.norm(average_shifts) > max_displacement * 2:
    logger.error(f"Suspiciously large shift: {average_shifts}")
    return [0, 0]
```

---

### 7. Truncation vs Rounding Error Accumulation

**Scenario**: Multiple field blocks with truncation errors

```python
# Field block 1: True average = [5.8, 2.7]
# Truncated: [5, 2] → 0.8 and 0.7 pixel errors

# Field block 2: True average = [6.2, 3.9]
# Truncated: [6, 3] → 0.2 and 0.9 pixel errors

# Over 10 field blocks:
# Cumulative error: Up to 8-9 pixels in worst case
# Not systematic (doesn't accumulate in same direction)
```

**Impact**: Minor for isolated field blocks, but could affect relative alignment

---

### 8. Field Block at Image Boundary

**Scenario**: Field block near edge, anchors only on one side

```python
# Field block at top-left corner:
field_block_position = [50, 50]

# Anchors (SIFT features only found in bottom-right):
anchors = [
    [[80, 80], [5, 2]],
    [[90, 90], [6, 3]],
    [[100, 100], [5, 2]],
    [[110, 110], [6, 3]],
]

# All nearest anchors are from one direction
# Bias in shift estimation (no anchors from other sides to balance)
```

**Impact**: Shift may be biased, but SIFT should find features throughout ROI

---

## Browser-Specific Constraints

### 1. JavaScript Number Precision

**Constraint**: JavaScript uses double-precision floats (64-bit) for all numbers

**Python**:
```python
np.int32(5.9) → 5  # Explicit 32-bit integer
```

**JavaScript**:
```javascript
Math.floor(5.9) → 5  // Number type (64-bit float internally)
Math.trunc(5.9) → 5  // Same, explicit truncation

// No native 32-bit integer type (except typed arrays)
const shifts = new Int32Array([5, 2]);  // 32-bit integers
```

**Impact**: Usually none (coordinate values fit in int32 range)

---

### 2. Array Performance

**Constraint**: JavaScript arrays less efficient than NumPy

**Python**:
```python
# NumPy vectorized:
average_shifts = np.average(displacements, axis=0).astype(np.int32)
# Fast (compiled C code)
```

**JavaScript**:
```javascript
// Manual loop:
const avgX = displacements.reduce((sum, d) => sum + d[0], 0) / displacements.length;
const avgY = displacements.reduce((sum, d) => sum + d[1], 0) / displacements.length;

// OR use TensorFlow.js:
const tensor = tf.tensor2d(displacements);
const avg = tensor.mean(0);
const shifts = await avg.array();
```

**Performance**: JavaScript ~2-5x slower for K-nearest (but still < 1ms)

---

### 3. Sorting Stability

**Constraint**: JavaScript `.sort()` stability varies

**ES2019+**: Stable sort guaranteed

**Pre-ES2019**: Unstable (V8, SpiderMonkey varied)

**Solution**: Use modern browsers or add tiebreaker
```javascript
// Stable tiebreaker:
distance_list.sort((a, b) => {
  if (a[0] !== b[0]) return a[0] - b[0];
  // Tiebreaker: use original index or anchor coordinates
  return a[1][0][0] - b[1][0][0];
});
```

---

### 4. Memory Management

**Constraint**: No automatic garbage collection for large arrays in tight loops

**Issue**: Creating many temporary arrays
```javascript
// Bad (creates many temporaries):
for (const field_block of field_blocks) {
  const sorted = distance_list.sort(...);  // New array
  const nearest = sorted.slice(0, k);      // Another new array
  // Old arrays not freed immediately
}
```

**Better**: Reuse arrays or rely on modern GC
```javascript
// Reuse array:
const distance_list = new Array(anchors.length);
for (let i = 0; i < anchors.length; i++) {
  distance_list[i] = [distance(origin, anchors[i][0]), anchors[i]];
}
distance_list.sort((a, b) => a[0] - b[0]);
```

---

### 5. Web Worker Communication Overhead

**Constraint**: Transferring data to/from Web Worker has cost

**Scenario**: Offload K-nearest to worker
```javascript
// Serialize and send:
worker.postMessage({
  anchors: anchors_with_displacements,  // Array of arrays
  origin: field_block_position,
  k: 4,
});
// Copy cost: ~1-5ms for 50 anchors
```

**Trade-off**: Communication overhead vs computation time
- K-nearest computation: < 1ms
- Worker communication: 1-5ms
- **Not worth offloading for single field block**

**Better**: Batch multiple field blocks
```javascript
// Send all field blocks at once:
worker.postMessage({
  field_blocks: field_blocks,  // 20 field blocks
  anchors: anchors_per_block,
});
// Amortize communication cost
```

---

## Validation Checklist

### Before K-Nearest Interpolation

1. ✓ Check `len(displacement_pairs) >= MIN_DISPLACEMENT_PAIRS`
2. ✓ Verify `k > 0`
3. ✓ Ensure `field_block.origin` is valid
4. ✓ Validate `block_image_shifts` are finite

### After K-Nearest Interpolation

1. ✓ Check `average_shifts` is finite (not NaN, not Inf)
2. ✓ Verify `|average_shifts| < max_displacement × 2` (sanity check)
3. ✓ Ensure all `scan_box.shifts` updated correctly
4. ✓ Validate shifted positions within image bounds

---

## Summary Table

| Constraint | Value | Rationale |
|------------|-------|-----------|
| K (default) | 4 | Balance: stability vs locality |
| K (min recommended) | 3 | Minimum for stable averaging |
| K (max recommended) | 10 | Avoid over-smoothing |
| Min displacement pairs | 5+ | Robustness (not enforced) |
| Max reasonable shift | 500px | Sanity check (not enforced) |
| Distance metric | Euclidean | Standard for K-NN |
| Granularity | Field block | Speed vs precision |
| Conversion | Truncation | Should use rounding |
| Centered mode | False | Faster, sufficient |

---

## Recommended Improvements

### 1. Add Minimum Pairs Check
```python
MIN_DISPLACEMENT_PAIRS = 5

if len(displacement_pairs) < MIN_DISPLACEMENT_PAIRS:
    logger.warning(f"Too few displacement pairs: {len(displacement_pairs)}")
    return [0, 0]
```

### 2. Use Rounding Instead of Truncation
```python
average_shifts = np.round(np.average(..., axis=0)).astype(np.int32)
```

### 3. Add Sanity Check
```python
if np.linalg.norm(average_shifts) > max_displacement * 2:
    logger.error(f"Unreasonable shift: {average_shifts}")
    return [0, 0]
```

### 4. Use Heap for Large Anchor Sets
```python
import heapq

def find_k_nearest_anchors_optimized(origin, anchors, k):
    heap = []
    for anchor_point, displacement in anchors:
        dist = MathUtils.distance(origin, anchor_point)
        if len(heap) < k:
            heapq.heappush(heap, (-dist, [anchor_point, displacement]))
        elif -dist > heap[0][0]:
            heapq.heapreplace(heap, (-dist, [anchor_point, displacement]))

    return [item[1] for item in heap]
```

---

## Related Constraints

- **Alignment Constraints** (`../constraints.md`)
- **SIFT Constraints** (`../sift/constraints.md`)
- **Math Utils** (`modules/domain/utils/math/`)
- **NumPy Operations** (`modules/technical/numpy/array-operations.md`)
