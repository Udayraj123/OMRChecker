# Alignment Flow - Execution Flow

## High-Level Flow

```
Pipeline
  ↓
AlignmentProcessor.process(context)
  ↓
apply_template_alignment(gray_image, colored_image, template, config)
  ↓
For each field_block in template:
  ↓
  apply_k_nearest_interpolation_inplace(...)
  ↓
  Updates scan_box.shifts for all bubbles
  ↓
Return updated template with alignment shifts
```

## Detailed Step-by-Step Flow

### Step 1: Processor Entry Point

**File**: `src/processors/alignment/processor.py`

```python
def process(self, context: ProcessingContext) -> ProcessingContext:
    """Execute alignment on the images."""

    gray_image = context.gray_image
    colored_image = context.colored_image
    template = context.template

    # Guard clause: Skip if no alignment image
    if gray_image is not None and "gray_alignment_image" in template.alignment:
        gray_image, colored_image, template = apply_template_alignment(
            gray_image, colored_image, template, self.tuning_config
        )

        # Update context with aligned images and template
        context.gray_image = gray_image
        context.colored_image = colored_image
        context.template = template

    return context
```

**Key Points**:
- Checks for `gray_alignment_image` existence
- Passes `tuning_config` for visualization settings
- Updates template in-place with shifts

---

### Step 2: Template Alignment Setup

**File**: `src/processors/alignment/template_alignment.py`

```python
def apply_template_alignment(gray_image, colored_image, template, config):
    # 2.1: Extract configuration
    template_margins = template.alignment["margins"]
    template_max_displacement = template.alignment["max_displacement"]

    # 2.2: Get pre-loaded alignment images
    gray_alignment_image = template.alignment["gray_alignment_image"]
    colored_alignment_image = template.alignment["colored_alignment_image"]

    # 2.3: Resize all images to template dimensions
    (
        gray_image,
        colored_image,
        gray_alignment_image,
        colored_alignment_image,
    ) = ImageUtils.resize_to_dimensions(
        template.template_dimensions,
        gray_image,
        colored_image,
        gray_alignment_image,
        colored_alignment_image,
    )
```

**Why Resize?**
- Ensures scanned image matches template dimensions
- Normalizes scale differences from different DPI settings
- Creates copies (needed for safe modification)

---

### Step 3: Per-Field-Block Alignment Loop

```python
for field_block in template.field_blocks:
    # 3.1: Extract field block properties
    field_block_name = field_block.name
    field_block.shifts = [0, 0]  # Reset shifts
    bounding_box_origin = field_block.bounding_box_origin
    bounding_box_dimensions = field_block.bounding_box_dimensions
    field_block_alignment = field_block.alignment

    # 3.2: Get alignment config (with fallback to template-level)
    margins = field_block_alignment.get("margins", template_margins)
    max_displacement = field_block_alignment.get(
        "max_displacement", template_max_displacement
    )

    # 3.3: EDGE CASE - Skip if max_displacement is 0
    if max_displacement == 0:
        continue

    # 3.4: Compute alignment zone (with margins)
    zone_start = [
        int(bounding_box_origin[0] - margins["left"]),
        int(bounding_box_origin[1] - margins["top"]),
    ]
    zone_end = [
        int(bounding_box_origin[0] + margins["right"] + bounding_box_dimensions[0]),
        int(bounding_box_origin[1] + margins["bottom"] + bounding_box_dimensions[1]),
    ]

    # 3.5: Extract ROIs (Region of Interest)
    block_gray_image = gray_image[
        zone_start[1] : zone_end[1],
        zone_start[0] : zone_end[0]
    ]
    block_gray_alignment_image = gray_alignment_image[
        zone_start[1] : zone_end[1],
        zone_start[0] : zone_end[0]
    ]
```

**Zone Calculation**:
```
zone_start = [origin.x - margin_left, origin.y - margin_top]
zone_end = [origin.x + margin_right + width, origin.y + margin_bottom + height]

Example:
  origin = [100, 200]
  dimensions = [300, 400]
  margins = {top: 50, bottom: 50, left: 50, right: 50}

  zone_start = [100 - 50, 200 - 50] = [50, 150]
  zone_end = [100 + 50 + 300, 200 + 50 + 400] = [450, 650]
```

---

### Step 4: Apply K-Nearest Interpolation (CURRENT METHOD)

**File**: `src/processors/alignment/k_nearest_interpolation.py`

```python
average_shifts = apply_k_nearest_interpolation_inplace(
    field_block,
    block_gray_image,
    block_gray_alignment_image,
    max_displacement,
    margins,
    config,
)
```

**Sub-Flow**:

#### 4.1: Get SIFT Displacement Pairs

```python
def apply_k_nearest_interpolation_inplace(
    field_block, block_gray_image, block_gray_alignment_image,
    max_displacement, margins, config, k=4
):
    # 4.1.1: Get SIFT matches
    displacement_pairs = SiftMatcher.get_matches(
        field_block_name,
        block_gray_image,
        block_gray_alignment_image,
        max_displacement,
        config,
    )
    # Returns: [[dest_point, source_point], ...]
    # dest = where feature is in SCANNED image
    # source = where feature is in TEMPLATE image
```

**SIFT Matching Process** (see `sift_matcher.py`):

```python
# 4.1.2: SIFT feature detection
source_features, des1 = sift.detectAndCompute(alignment_image, None)
destination_features, des2 = sift.detectAndCompute(gray_image, None)

# 4.1.3: FLANN-based matching
matches = flann.knnMatch(des1, des2, k=2)

# 4.1.4: Lowe's ratio test + displacement filter
good = []
displacement_pairs = []

for m, n in matches:
    source_point = source_features[m.queryIdx].pt
    dest_point = destination_features[m.trainIdx].pt

    # Filter criteria:
    # 1. Lowe's ratio test: m.distance < n.distance
    # 2. Max displacement constraint
    if (m.distance < n.distance and
        distance(source_point, dest_point) <= max_displacement):
        good.append(m)
        displacement_pairs.append([dest_point, source_point])

# 4.1.5: Verify minimum matches
if len(good) < MIN_MATCH_COUNT:  # MIN_MATCH_COUNT = 10
    logger.critical("Not enough matches found")
    # Alignment FAILS - shifts remain [0, 0]
```

**Why Reverse Order?**
- Displacement pair: `[destination_point, source_point]`
- Destination (scanned image) → Source (template)
- We want to warp FROM template TO scanned position

#### 4.2: Convert to Anchor Displacements

```python
anchors_with_displacements = [
    [anchor_point, subtract_points(anchor_point, displaced_point)]
    for anchor_point, displaced_point in displacement_pairs
]
# anchor_point = destination (where it is NOW in scan)
# displacement = how much it moved from template
```

**Example**:
```python
template_point = [100, 200]
scanned_point = [105, 198]
displacement = [100 - 105, 200 - 198] = [-5, 2]
# Bubble moved RIGHT by 5, UP by 2
```

#### 4.3: Compute Block Image Shifts

```python
block_image_shifts = subtract_points(
    [margins["left"], margins["top"]],
    field_block.origin
)
# Converts from full image coordinates to block ROI coordinates
```

#### 4.4: Shift by Field Blocks (Default Strategy)

```python
def shift_by_field_blocks(
    field_block, block_image_shifts, anchors_with_displacements, k=4
):
    # 4.4.1: Get field block center position
    field_block_position = add_points(
        field_block.origin,
        block_image_shifts
    )

    # 4.4.2: Find K nearest anchor points
    nearest_anchors = find_k_nearest_anchors(
        field_block_position,
        anchors_with_displacements,
        k
    )

    # 4.4.3: Average the displacements
    average_shifts = np.average(
        [displacement for _anchor, displacement in nearest_anchors],
        axis=0,
    ).astype(np.int32)

    # 4.4.4: Apply shifts to ALL scan boxes in field block
    for field in field_block.fields:
        for scan_box in field.scan_boxes:
            scan_box.shifts = average_shifts

    return average_shifts
```

**K-Nearest Anchor Selection**:

```python
def find_k_nearest_anchors(origin, anchors_with_displacements, k):
    # Sort by Euclidean distance
    sorted_by_distance = sorted(
        [
            [distance(origin, anchor_point), [anchor_point, displacement]]
            for anchor_point, displacement in anchors_with_displacements
        ],
        key=lambda x: x[0],  # Sort by distance
    )

    # Return top K
    return [anchor_data for _dist, anchor_data in sorted_by_distance[:k]]
```

**Example**:
```
Field block center: [200, 300]
Anchors:
  A1: position=[195, 298], displacement=[-3, 2]   distance=√13 ≈ 3.6
  A2: position=[210, 305], displacement=[-4, 3]   distance=√74 ≈ 8.6
  A3: position=[198, 302], displacement=[-2, 1]   distance=√8 ≈ 2.8
  A4: position=[250, 320], displacement=[-5, 4]   distance=√650 ≈ 25.5
  A5: position=[202, 299], displacement=[-3, 2]   distance=√5 ≈ 2.2

K=4 nearest: [A5, A3, A1, A2]
Average displacement = mean([-3,2], [-2,1], [-3,2], [-4,3])
                     = [(-3-2-3-4)/4, (2+1+2+3)/4]
                     = [-3, 2]
```

---

### Step 5: Optional Visualization (Debug Mode)

```python
if config.outputs.show_image_level >= 2:
    # Draw field blocks BEFORE alignment
    block_gray_image_before = block_gray_image.copy()
    field_block.shifts = block_image_shifts
    field_block.drawing.draw_field_block(
        block_gray_image_before, shifted=True, thickness=2
    )

if config.outputs.show_image_level >= 6:
    # Draw field blocks AFTER alignment
    block_gray_image_after = block_gray_image.copy()
    field_block.drawing.draw_field_block(
        block_gray_image_after, shifted=True, thickness=2
    )

    InteractionUtils.show(
        f"Field Block shifts: {average_shifts}",
        ImageUtils.get_padded_hstack([
            block_gray_image_before,
            block_gray_image_after
        ]),
    )
```

---

### Step 6: Return Updated Template

```python
return gray_image, colored_image, template
# Template now contains updated scan_box.shifts for all field blocks
```

---

## Alternative Methods (Commented in Code)

### Method 1: Phase Correlation (Lines 88-91)

```python
# warped_block_image = apply_phase_correlation_shifts(
#     field_block, block_gray_alignment_image, block_gray_image
# )
```

**Approach**: FFT-based translation detection
**Limitation**: Only detects rigid translation (no rotation/warping)

### Method 2: SIFT + Piecewise Affine (Lines 95-114)

```python
# warped_block_image, warped_colored_image = apply_sift_shifts(...)
# gray_image[zone_start[1]:zone_end[1], zone_start[0]:zone_end[0]] = warped_block_image
```

**Approach**: Warp the IMAGE using Delaunay triangulation
**Limitation**: Cannot handle overlapping field blocks

### Method 3: K-Nearest Interpolation (Lines 118-125) **ACTIVE**

```python
average_shifts = apply_k_nearest_interpolation_inplace(...)
```

**Approach**: Warp the COORDINATES (bubble positions)
**Advantage**: Supports overlapping field blocks, faster

### Method 4: Field-Level Warping (Lines 128-129) **TODO**

```python
# TODO: figure out how to apply detection on these copies to support overlapping field blocks!
```

**Planned**: Per-field image transformation

---

## Data Structures

### Displacement Pair
```python
[
    [destination_point, source_point],  # Format from SIFT matcher
    ...
]
# destination_point: [x, y] in SCANNED image
# source_point: [x, y] in TEMPLATE image
```

### Anchor with Displacement
```python
[
    [anchor_point, displacement],
    ...
]
# anchor_point: [x, y] position in image
# displacement: [dx, dy] shift from template
```

### Scan Box Shifts
```python
scan_box.shifts = [dx, dy]  # Applied during detection
```

---

## Performance Characteristics

### Time Complexity
- **SIFT Detection**: O(n) where n = number of keypoints
- **FLANN Matching**: O(k log n) approximate
- **K-Nearest Search**: O(m × k) where m = number of bubbles
- **Per Field Block**: O(SIFT + FLANN + K-nearest)

### Memory Complexity
- **SIFT Features**: ~100-500 features per field block
- **Displacement Pairs**: Typically 10-100 pairs
- **Image ROIs**: Only extract zones, not full image copies

### Optimization Points
1. **Singleton SIFT/FLANN**: Initialized once at module load
2. **Zone Extraction**: Only process relevant regions
3. **Skip Zero Displacement**: Early exit for `max_displacement=0`
4. **K-Nearest Caching**: Could cache for similar templates

---

## Error Handling

### No Alignment Image
```python
if "gray_alignment_image" not in template.alignment:
    return context  # Skip alignment
```

### Insufficient Matches
```python
if len(good_matches) < MIN_MATCH_COUNT:
    logger.critical("Not enough matches found")
    # Shifts remain [0, 0], detection proceeds without alignment
```

### Out-of-Bounds Zone
```python
# NumPy handles negative indices and out-of-bounds gracefully
block_image = image[zone_start[1]:zone_end[1], zone_start[0]:zone_end[0]]
# May return smaller ROI if zone extends beyond image
```

---

## Browser Migration Notes

### Critical Changes for Browser

1. **SIFT Alternative**:
```javascript
// Replace SIFT with ORB or AKAZE
const orb = new cv.ORB_create(500);
const [keypoints, descriptors] = orb.detectAndCompute(image, new cv.Mat());
```

2. **FLANN Alternative**:
```javascript
// Use BFMatcher instead
const bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
const matches = bf.match(desc1, desc2);
```

3. **NumPy Arrays**:
```javascript
// Use cv.Mat operations
const avgX = shifts.reduce((sum, s) => sum + s[0], 0) / shifts.length;
const avgY = shifts.reduce((sum, s) => sum + s[1], 0) / shifts.length;
```

4. **Async Processing**:
```javascript
// Offload to Web Worker for large images
const worker = new Worker('alignment-worker.js');
worker.postMessage({ image, template });
```

---

## Related Flows

- **SIFT Matching** (`sift/flows.md`) - Feature detection details
- **Phase Correlation** (`phase-correlation/flows.md`) - FFT-based alignment
- **K-Nearest Interpolation** (`k-nearest/flows.md`) - Coordinate shifting
- **Piecewise Affine** (`piecewise-affine/flows.md`) - Image warping

---

## Testing Considerations

### Test Cases
1. **Perfect Alignment**: Shifts should be [0, 0]
2. **Small Translation**: Shifts should match scan offset
3. **Insufficient Features**: Should gracefully fail with [0, 0]
4. **Max Displacement = 0**: Should skip alignment
5. **No Alignment Image**: Should skip processor
6. **Per-Field-Block Config**: Should override template config

### Edge Cases
- Empty ROI (zone outside image bounds)
- Very small field blocks (< 50x50 pixels)
- Overlapping field blocks (future support)
- Multiple pages with different shifts
