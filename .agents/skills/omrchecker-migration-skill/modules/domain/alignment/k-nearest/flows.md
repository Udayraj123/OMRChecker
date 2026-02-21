# K-Nearest Interpolation - Execution Flow

## Overview

**K-Nearest Interpolation** is the active coordinate-based alignment method in OMRChecker. Instead of warping the image itself, it adjusts bubble coordinates by averaging displacement vectors from the K nearest anchor points detected by SIFT.

**File**: `src/processors/alignment/k_nearest_interpolation.py`

**Key Advantage**: Supports overlapping field blocks (unlike image warping methods)

---

## High-Level Flow

```
SIFT Matcher
  ↓
Get displacement pairs [[dest, src], ...]
  ↓
Convert to anchor-displacement format [[anchor, displacement], ...]
  ↓
For each field block:
  ↓
  Find K nearest anchors to field block center
  ↓
  Average their displacements
  ↓
  Apply average shift to all scan boxes in field block
  ↓
Updated scan_box.shifts
```

---

## Entry Point

### apply_k_nearest_interpolation_inplace()

**File**: `k_nearest_interpolation.py:28-91`

```python
def apply_k_nearest_interpolation_inplace(
    field_block,
    block_gray_image,
    block_gray_alignment_image,
    max_displacement,
    margins,
    config,
    k=4,
) -> None:
    """Apply k-nearest interpolation to align field block coordinates."""
```

**Parameters**:
- `field_block`: Field block entity to align
- `block_gray_image`: Cropped ROI from scanned image
- `block_gray_alignment_image`: Cropped ROI from template alignment image
- `max_displacement`: Maximum allowed pixel shift (filters SIFT matches)
- `margins`: Zone padding around field block
- `config`: Tuning config (for visualization debug levels)
- `k`: Number of nearest neighbors to average (default: 4)

**Returns**: `average_shifts` (for optional use by caller)

**Side Effect**: Updates `scan_box.shifts` for all scan boxes in field block (in-place)

---

## Step-by-Step Flow

### Step 1: Get SIFT Displacement Pairs

**Code**: `k_nearest_interpolation.py:38-45`

```python
field_block_name = field_block.name
displacement_pairs = SiftMatcher.get_matches(
    field_block_name,
    block_gray_image,
    block_gray_alignment_image,
    max_displacement,
    config,
)
```

**Output**: List of displacement pairs from SIFT matching
```python
displacement_pairs = [
    [dest_point, source_point],
    [dest_point, source_point],
    # ... typically 10-100 pairs
]
# dest_point: [x, y] where feature is in SCANNED image
# source_point: [x, y] where feature is in TEMPLATE image
```

**Example**:
```python
displacement_pairs = [
    [[105, 198], [100, 200]],  # Feature moved right by 5, up by 2
    [[215, 303], [210, 300]],  # Feature moved right by 5, down by 3
    [[320, 405], [315, 400]],  # Feature moved right by 5, down by 5
    # ...
]
```

**See Also**: `sift/flows.md` for SIFT matching details

---

### Step 2: Convert to Anchor-Displacement Format

**Code**: `k_nearest_interpolation.py:46-49`

```python
anchors_with_displacements = [
    [anchor_point, MathUtils.subtract_points(anchor_point, displaced_point)]
    for anchor_point, displaced_point in displacement_pairs
]
```

**Transformation**:
```python
# Input (displacement pair):
dest_point = [105, 198]     # Where feature IS in scan
source_point = [100, 200]   # Where feature SHOULD BE in template

# Output (anchor-displacement):
anchor_point = dest_point = [105, 198]
displacement = subtract_points(anchor_point, displaced_point)
            = subtract_points([105, 198], [100, 200])
            = [105 - 100, 198 - 200]
            = [5, -2]

# Result: [[105, 198], [5, -2]]
```

**Why This Format?**
- `anchor_point`: Known reference position in scanned image
- `displacement`: How much to shift coordinates to align with template
- Positive displacement = feature moved right/down in scan
- Negative displacement = feature moved left/up in scan

**MathUtils.subtract_points()**:
```python
# From src/utils/math.py:29-33
def subtract_points(point, new_origin):
    return [
        point[0] - new_origin[0],
        point[1] - new_origin[1],
    ]
```

**Full Example**:
```python
displacement_pairs = [
    [[105, 198], [100, 200]],
    [[215, 303], [210, 300]],
    [[320, 405], [315, 400]],
]

anchors_with_displacements = [
    [[105, 198], [5, -2]],   # 105-100=5, 198-200=-2
    [[215, 303], [5, 3]],    # 215-210=5, 303-300=3
    [[320, 405], [5, 5]],    # 320-315=5, 405-400=5
]
```

---

### Step 3: Compute Block Image Shifts

**Code**: `k_nearest_interpolation.py:50-52`

```python
block_image_shifts = MathUtils.subtract_points(
    [margins["left"], margins["top"]], field_block.origin
)
```

**Purpose**: Convert from template coordinate space to ROI coordinate space

**Calculation**:
```python
# Field block in template:
field_block.origin = [100, 200]
margins = {"left": 50, "top": 50, "right": 50, "bottom": 50}

# Zone extraction (from template_alignment.py):
zone_start = [origin.x - margins.left, origin.y - margins.top]
          = [100 - 50, 200 - 50]
          = [50, 150]

# Block image shifts (coordinate transformation):
block_image_shifts = [margins.left - origin.x, margins.top - origin.y]
                   = [50 - 100, 50 - 200]
                   = [-50, -150]

# This converts template coords → ROI coords
```

**Why Needed?**
- ROI was extracted starting at `zone_start`
- Field block coordinates are relative to full template
- Need to shift to ROI-relative coordinates for distance calculations

---

### Step 4: Optional Pre-Visualization

**Code**: `k_nearest_interpolation.py:53-62`

```python
if config.outputs.show_image_level >= 2:
    block_gray_image_before = block_gray_image.copy()

    # Shift the coordinates to the field block's origin to draw on the cropped block
    old_shifts = field_block.shifts.copy()
    field_block.shifts = block_image_shifts
    field_block.drawing.draw_field_block(
        block_gray_image_before, shifted=True, thickness=2
    )
    field_block.shifts = old_shifts
```

**Purpose**: Visualize field block BEFORE alignment for comparison

**Debug Level**: `show_image_level >= 2`

---

### Step 5: Shift Bubble Coordinates (Main Logic)

**Code**: `k_nearest_interpolation.py:64-67`

```python
average_shifts = shift_by_field_blocks(
    field_block, block_image_shifts, anchors_with_displacements, k
)
```

**This is the core alignment logic.**

---

## Core Algorithm: shift_by_field_blocks()

**Code**: `k_nearest_interpolation.py:93-131`

```python
def shift_by_field_blocks(
    field_block, block_image_shifts, anchors_with_displacements, k, centered=False
):
    """Compute and apply shifts to all scan boxes in field block."""
```

### Step 5.1: Determine Query Position

**Code**: `k_nearest_interpolation.py:96-114`

```python
field_block_position = (
    np.average(
        [
            # field center
            np.average(
                [
                    scan_box.get_shifted_position(block_image_shifts)
                    for scan_box in field.scan_boxes
                ],
                axis=0,
            )
            for field in field_block.fields
        ],
        axis=0,
    ).astype(np.int32)
    if centered
    else MathUtils.add_points(field_block.origin, block_image_shifts)
)
```

**Two Modes**:

#### Mode 1: centered=False (DEFAULT)
```python
field_block_position = add_points(field_block.origin, block_image_shifts)
```

**Example**:
```python
field_block.origin = [100, 200]
block_image_shifts = [-50, -150]
field_block_position = [100 + (-50), 200 + (-150)]
                     = [50, 50]
# Uses field block's top-left corner
```

#### Mode 2: centered=True (COMMENTED)
```python
# Compute geometric center of all bubbles:
1. For each field in field_block:
   - Get all scan_box positions (shifted to ROI space)
   - Average → field center
2. Average all field centers → field_block center
```

**Why Field Block Origin (centered=False)?**
- Simpler, faster computation
- Field blocks typically uniform, so corner ≈ center for nearest neighbor purposes
- Reduces complexity

---

### Step 5.2: Find K Nearest Anchors

**Code**: `k_nearest_interpolation.py:116-118`

```python
nearest_anchors = find_k_nearest_anchors(
    field_block_position, anchors_with_displacements, k
)
```

**Calls helper function** (detailed below)

---

## Helper Function: find_k_nearest_anchors()

**Code**: `k_nearest_interpolation.py:12-25`

```python
def find_k_nearest_anchors(origin, anchors_with_displacements, k) -> list[Any]:
    # TODO: any further optimization needed?
    sorted_by_distance = sorted(
        [
            [MathUtils.distance(origin, anchor_point), [anchor_point, displacement]]
            for anchor_point, displacement in anchors_with_displacements
        ],
        key=operator.itemgetter(0),
    )

    return [
        anchor_with_displacement
        for _distance, anchor_with_displacement in sorted_by_distance[:k]
    ]
```

### Algorithm

#### Step A: Compute Distances

```python
# For each anchor:
[distance_to_origin, [anchor_point, displacement]]
```

**Distance Calculation**:
```python
# MathUtils.distance() from src/utils/math.py:14-15
def distance(point1, point2):
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

# Euclidean distance:
distance = √((x₂ - x₁)² + (y₂ - y₁)²)
```

**Example**:
```python
origin = [200, 300]  # Field block position

anchors_with_displacements = [
    [[105, 198], [5, -2]],
    [[215, 303], [5, 3]],
    [[320, 405], [5, 5]],
    [[198, 302], [4, 2]],
    [[250, 350], [6, 4]],
]

# Compute distances:
distance_list = [
    [√((200-105)² + (300-198)²), [[105, 198], [5, -2]]],    # √14029 ≈ 118.4
    [√((200-215)² + (300-303)²), [[215, 303], [5, 3]]],     # √234 ≈ 15.3
    [√((200-320)² + (300-405)²), [[320, 405], [5, 5]]],     # √25425 ≈ 159.5
    [√((200-198)² + (300-302)²), [[198, 302], [4, 2]]],     # √8 ≈ 2.8
    [√((200-250)² + (300-350)²), [[250, 350], [6, 4]]],     # √5000 ≈ 70.7
]
```

#### Step B: Sort by Distance

```python
sorted_by_distance = sorted(distance_list, key=operator.itemgetter(0))
```

**Sorted Result**:
```python
sorted_by_distance = [
    [2.8,   [[198, 302], [4, 2]]],     # Closest
    [15.3,  [[215, 303], [5, 3]]],     # 2nd closest
    [70.7,  [[250, 350], [6, 4]]],     # 3rd closest
    [118.4, [[105, 198], [5, -2]]],    # 4th closest
    [159.5, [[320, 405], [5, 5]]],     # 5th closest
]
```

#### Step C: Select Top K

```python
# k = 4 (default)
nearest_anchors = [anchor_data for _dist, anchor_data in sorted_by_distance[:4]]
```

**Result**:
```python
nearest_anchors = [
    [[198, 302], [4, 2]],
    [[215, 303], [5, 3]],
    [[250, 350], [6, 4]],
    [[105, 198], [5, -2]],
]
# Returns 4 nearest anchors (ignoring distances)
```

**Time Complexity**: O(m log m) where m = number of anchors (typically 10-100)

**Space Complexity**: O(m) for sorted list

---

### Step 5.3: Average Displacements

**Code**: `k_nearest_interpolation.py:120-124`

```python
# Method 1: Get average displacement
average_shifts = np.average(
    [displacement for _anchor_point, displacement in nearest_anchors],
    axis=0,
).astype(np.int32)
```

**Computation**:
```python
nearest_anchors = [
    [[198, 302], [4, 2]],
    [[215, 303], [5, 3]],
    [[250, 350], [6, 4]],
    [[105, 198], [5, -2]],
]

# Extract displacements:
displacements = [
    [4, 2],
    [5, 3],
    [6, 4],
    [5, -2],
]

# NumPy average:
average_shifts = np.average(displacements, axis=0)
              = [(4+5+6+5)/4, (2+3+4-2)/4]
              = [20/4, 7/4]
              = [5.0, 1.75]

# Convert to int32:
average_shifts = [5, 1]  # Note: truncates, not rounds!
```

**Important**: `.astype(np.int32)` truncates (1.75 → 1), doesn't round (1.75 → 2)

**Better Approach** (not currently used):
```python
average_shifts = np.round(np.average(..., axis=0)).astype(np.int32)
# Would give [5, 2] instead of [5, 1]
```

---

### Step 5.4: Apply Shifts to All Scan Boxes

**Code**: `k_nearest_interpolation.py:126-130`

```python
# Shift all bubbles
for field in field_block.fields:
    for scan_box in field.scan_boxes:
        scan_box.shifts = average_shifts
```

**Effect**: ALL scan boxes in the field block get the SAME shift

**Example**:
```python
# Field block has 2 fields:
# Field 1: 4 scan boxes (bubbles A, B, C, D)
# Field 2: 5 scan boxes (bubbles 1, 2, 3, 4, 5)

average_shifts = [5, 1]

# After assignment:
scan_box_A.shifts = [5, 1]
scan_box_B.shifts = [5, 1]
scan_box_C.shifts = [5, 1]
scan_box_D.shifts = [5, 1]
scan_box_1.shifts = [5, 1]
scan_box_2.shifts = [5, 1]
# ... etc (all 9 bubbles get [5, 1])
```

**Why Same Shift for All Bubbles?**
- Assumes rigid (translation-only) displacement within field block
- Simplification: no rotation, no scaling, no local warping
- Trade-off: Speed vs accuracy (good enough for most OMR sheets)

---

### Step 5.5: Return Average Shifts

**Code**: `k_nearest_interpolation.py:131`

```python
return average_shifts
```

**Used by**: Caller can optionally use for logging/visualization

---

## Step 6: Optional Post-Visualization

**Code**: `k_nearest_interpolation.py:71-85`

```python
if config.outputs.show_image_level >= 6:
    block_gray_image_after = block_gray_image.copy()
    old_shifts = field_block.shifts.copy()
    field_block.shifts = block_image_shifts
    field_block.drawing.draw_field_block(
        block_gray_image_after, shifted=True, thickness=2
    )
    field_block.shifts = old_shifts

    InteractionUtils.show(
        f"Field Block shifts: {average_shifts}",
        ImageUtils.get_padded_hstack(
            [block_gray_image_before, block_gray_image_after]
        ),
    )
```

**Output**: Side-by-side comparison
```
┌───────────────┬───────────────┐
│ BEFORE        │ AFTER         │
│ Alignment     │ Alignment     │
│               │               │
│  ┌─────┐      │  ┌─────┐      │
│  │ □ □ │      │  │ □ □ │      │ (shifted boxes)
│  │ □ □ │      │  │ □ □ │      │
│  └─────┘      │  └─────┘      │
└───────────────┴───────────────┘
Title: "Field Block shifts: [5, 1]"
```

**Debug Level**: `show_image_level >= 6`

---

## Step 7: Return Average Shifts

**Code**: `k_nearest_interpolation.py:90`

```python
return average_shifts
```

**Caller**: `template_alignment.py` (currently ignores return value)

---

## Alternative Methods (Commented Code)

### Method 1: shift_by_fields() (Lines 134-160)

**Approach**: Compute separate shifts for each FIELD (instead of field block)

**Flow**:
```python
for field in field_block.fields:
    # Compute field center
    field_center = average([scan_box.position for scan_box in field.scan_boxes])

    # Find K nearest to field center
    nearest_anchors = find_k_nearest_anchors(field_center, anchors, k)

    # Average displacements
    average_shifts = mean([disp for _, disp in nearest_anchors])

    # Apply to field's scan boxes only
    for scan_box in field.scan_boxes:
        scan_box.shifts = average_shifts
```

**Advantage**: More localized alignment (field-level granularity)

**Disadvantage**: Slower (more K-nearest queries)

**Why Commented?**: Field block-level is sufficient for most cases

---

### Method 2: shift_by_scan_boxes() (Lines 163-181)

**Approach**: Compute separate shifts for each SCAN BOX (bubble)

**Flow**:
```python
for field in field_block.fields:
    for scan_box in field.scan_boxes:
        # Find K nearest to this specific bubble
        nearest_anchors = find_k_nearest_anchors(scan_box.position, anchors, k)

        # Average displacements
        average_shifts = mean([disp for _, disp in nearest_anchors])

        # Apply to this bubble only
        scan_box.shifts = average_shifts
```

**Advantage**: Most precise alignment (per-bubble granularity)

**Disadvantage**: Much slower (one K-nearest query per bubble)

**Why Commented?**:
- Overkill for OMR (bubbles close together move similarly)
- Performance cost too high (100+ bubbles = 100+ K-nearest queries)
- Noise amplification (individual bubbles get noisy shifts)

---

## Data Structures Summary

### Displacement Pair (from SIFT)
```python
[
    [destination_point, source_point],
    # destination: where feature IS in scanned image
    # source: where feature SHOULD BE in template
]
```

### Anchor with Displacement
```python
[
    [anchor_point, displacement],
    # anchor: known position in scanned image
    # displacement: how much to shift to align
]
```

### Scan Box Shifts
```python
scan_box.shifts = [dx, dy]  # Applied during bubble detection
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| SIFT Matching | O(n log n) | 20-80ms |
| Convert to anchors | O(m) | < 1ms |
| Find K-nearest | O(m log m) | < 1ms |
| Average displacements | O(k) | < 1ms |
| Apply shifts | O(b) | < 1ms |
| **Total per field block** | - | **20-80ms** |

Where:
- n = number of keypoints (50-500)
- m = number of displacement pairs (10-100)
- k = number of nearest neighbors (default 4)
- b = number of bubbles in field block (10-100)

**Bottleneck**: SIFT matching (95%+ of time)

---

### Memory Usage

```python
# Per field block:
# Displacement pairs: 50 × 32 bytes = 1.6 KB
# Anchors: 50 × 32 bytes = 1.6 KB
# Distance list: 50 × 40 bytes = 2.0 KB
# Nearest anchors: 4 × 32 bytes = 128 bytes
# Total: ~5 KB

# Template with 20 field blocks:
# Total: 20 × 5 KB = 100 KB (negligible)
```

**Note**: SIFT matching uses ~100 KB per field block (see `sift/flows.md`)

---

## Browser Migration Notes

### Critical Changes

#### 1. NumPy Array Operations

**Python**:
```python
average_shifts = np.average(displacements, axis=0).astype(np.int32)
```

**JavaScript**:
```javascript
// Option 1: Manual calculation
const avgX = displacements.reduce((sum, d) => sum + d[0], 0) / displacements.length;
const avgY = displacements.reduce((sum, d) => sum + d[1], 0) / displacements.length;
const average_shifts = [Math.floor(avgX), Math.floor(avgY)];

// Option 2: Use ndarray.js or TensorFlow.js
const tf = require('@tensorflow/tfjs');
const tensor = tf.tensor2d(displacements);
const avg = tensor.mean(0);  // axis=0
const average_shifts = await avg.array();  // Convert to JS array
avg.dispose();
tensor.dispose();
```

#### 2. Math Utilities

**Python**:
```python
distance = MathUtils.distance(point1, point2)
```

**JavaScript**:
```javascript
function distance(point1, point2) {
  const dx = point1[0] - point2[0];
  const dy = point1[1] - point2[1];
  return Math.hypot(dx, dy);  // Math.hypot available in ES6
}
```

#### 3. Sorting

**Python**:
```python
sorted_by_distance = sorted(distance_list, key=operator.itemgetter(0))
```

**JavaScript**:
```javascript
const sorted_by_distance = distance_list.sort((a, b) => a[0] - b[0]);
```

#### 4. List Slicing

**Python**:
```python
nearest_anchors = sorted_by_distance[:k]
```

**JavaScript**:
```javascript
const nearest_anchors = sorted_by_distance.slice(0, k);
```

#### 5. Type Casting

**Python**:
```python
.astype(np.int32)
```

**JavaScript**:
```javascript
Math.floor(value)  // Truncates like astype(int32)
// OR
Math.round(value)  // Better: rounds to nearest
```

---

### Web Worker Pattern

**Main Thread**:
```javascript
const worker = new Worker('k-nearest-worker.js');

worker.postMessage({
  type: 'align_field_block',
  anchors_with_displacements: anchorsData,
  field_block_position: [200, 300],
  k: 4,
});

worker.onmessage = (event) => {
  const { average_shifts } = event.data;
  // Apply shifts to scan boxes
  applyShiftsToScanBoxes(field_block, average_shifts);
};
```

**Worker (k-nearest-worker.js)**:
```javascript
self.onmessage = (event) => {
  const { anchors_with_displacements, field_block_position, k } = event.data;

  // Run find_k_nearest_anchors in worker
  const nearest = findKNearestAnchors(field_block_position, anchors_with_displacements, k);

  // Average displacements
  const average_shifts = computeAverage(nearest);

  self.postMessage({ average_shifts });
};
```

**Benefit**: Offloads sorting/computation from main thread (prevents UI freezing)

---

## Edge Cases

### 1. Fewer Displacement Pairs than K

**Scenario**: K=4, but only 3 SIFT matches

```python
k = 4
displacement_pairs = [[...], [...], [...]]  # Only 3 pairs

# sorted_by_distance[:4] returns only 3 items
nearest_anchors = sorted_by_distance[:3]

# np.average still works:
average_shifts = np.average([d1, d2, d3], axis=0)
# Result: Average of 3 instead of 4 (less stable, but acceptable)
```

### 2. Zero Displacement Pairs

**Scenario**: SIFT matching failed completely

```python
displacement_pairs = []

# anchors_with_displacements = []
# nearest_anchors = []
# np.average([]) → RuntimeWarning: Mean of empty slice
# Result: [nan, nan]

# PROBLEM: Shifts become NaN → detection fails!
```

**Fix Needed** (not currently implemented):
```python
if len(displacement_pairs) == 0:
    logger.warning("No displacement pairs, skipping alignment")
    return [0, 0]
```

### 3. K > Number of Anchors

**Same as Case 1**: Returns all available anchors (fewer than K)

### 4. Single Displacement Pair

**Scenario**: Only 1 SIFT match (very risky)

```python
nearest_anchors = [[[198, 302], [4, 2]]]
average_shifts = [4, 2]  # No averaging, just use the one displacement

# Risk: That one match could be an outlier!
```

**Better Handling** (not implemented):
```python
MIN_DISPLACEMENT_PAIRS = 5

if len(displacement_pairs) < MIN_DISPLACEMENT_PAIRS:
    logger.warning(f"Too few displacement pairs: {len(displacement_pairs)}")
    return [0, 0]
```

### 5. Identical Anchor Positions

**Scenario**: Multiple SIFT features at same location

```python
anchors = [
    [[100, 200], [5, 2]],
    [[100, 200], [6, 3]],  # Same position!
]

# Distances both 0 (or very close)
# Both selected in K-nearest
# Average: [5.5, 2.5] → truncated to [5, 2]
# Valid behavior
```

### 6. Floating Point Precision

**Issue**: Truncation instead of rounding

```python
average_shifts = np.average([[5, 2], [5, 3], [6, 4], [5, -2]], axis=0)
              = [5.25, 1.75]

# Current:
.astype(np.int32) → [5, 1]  # Truncates

# Better:
np.round().astype(np.int32) → [5, 2]  # Rounds
```

**Impact**: Off-by-one pixel errors in bubble alignment

---

## Testing Considerations

### Unit Tests

```python
def test_k_nearest_basic():
    anchors = [
        [[100, 100], [5, 2]],
        [[200, 200], [6, 3]],
        [[150, 150], [5, 2]],
        [[300, 300], [7, 4]],
    ]
    origin = [150, 150]
    k = 2

    nearest = find_k_nearest_anchors(origin, anchors, k)

    # Should return the 2 closest: [150,150] and [200,200] or [100,100]
    assert len(nearest) == 2

def test_average_shifts():
    nearest = [
        [[100, 100], [5, 2]],
        [[200, 200], [5, 3]],
    ]

    # Extract displacements and average
    displacements = [d for _, d in nearest]
    avg = np.average(displacements, axis=0).astype(np.int32)

    assert avg[0] == 5  # (5+5)/2 = 5
    assert avg[1] == 2  # (2+3)/2 = 2.5 → truncated to 2
```

### Edge Case Tests

1. Zero displacement pairs
2. Fewer pairs than K
3. K = 1 (single nearest)
4. All anchors equidistant
5. Very large K (K > number of anchors)
6. Negative displacements

---

## Related Flows

- **SIFT Matching** (`../sift/flows.md`) - Generates displacement pairs
- **Alignment Main Flow** (`../flows.md`) - Orchestrates k-nearest interpolation
- **Piecewise Affine** (`../piecewise-affine/flows.md`) - Alternative image warping method
- **Template Alignment** (`../flows.md`) - Overall alignment context
