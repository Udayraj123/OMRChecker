# Piecewise Affine Alignment - Execution Flow

## Overview

**Piecewise Affine Transformation** applies localized warping to images by dividing them into triangles and applying separate affine transformations to each triangle. This enables non-rigid alignment that can handle complex distortions like paper warping, perspective variations, and localized stretching.

**File**: `src/processors/alignment/piecewise_affine_delaunay.py`

**Status**: Implementation exists but currently unused in main pipeline (commented out)

**Algorithm**: Delaunay Triangulation + Per-Triangle Affine Warping

---

## Key Concepts

### What is Piecewise Affine Transformation?

**Problem**: Global transformations (perspective, affine) assume uniform distortion
- Real-world scans have NON-UNIFORM distortions (paper bend, camera angle variations)
- A single homography can't capture localized warping

**Solution**: Divide image into triangles, warp each independently
```
Global Affine:      [Single 2x3 matrix for entire image]
Piecewise Affine:   [N × 2x3 matrices, one per triangle]
```

### Why Delaunay Triangulation?

**Delaunay Property**: Maximizes minimum angle of all triangles
- Avoids "sliver" triangles (long and thin)
- Produces more stable affine transformations
- Well-conditioned matrices for cv2.getAffineTransform()

**Alternative**: Grid-based triangulation
- Problem: May create degenerate triangles near feature points
- Delaunay adapts to feature distribution

---

## High-Level Flow

```
1. Parse Displacement Pairs (filter within bounds)
   ↓
2. Build Delaunay Triangulation (destination points)
   ↓
3. Filter Valid Triangles (inside warped_rectangle)
   ↓
4. Map to Source Triangles (using point correspondence)
   ↓
5. For each triangle pair:
   a. Compute affine transform matrix
   b. Warp triangle from source → destination
   c. Blend into output image
   ↓
6. Return warped images
```

---

## Detailed Step-by-Step Flow

### Step 0: Function Signature

**Code**: `piecewise_affine_delaunay.py:42-44`

```python
def apply_piecewise_affine(
    gray_block_image,
    colored_block_image,
    displacement_pairs,
    warped_rectangle,
    config
):
    """
    Apply piecewise affine transformation using Delaunay triangulation.

    Args:
        gray_block_image: Source grayscale image (to be warped)
        colored_block_image: Source colored image (optional, for visualization)
        displacement_pairs: List of [[source_point, destination_point], ...]
        warped_rectangle: (x, y, width, height) bounding rectangle
        config: TuningConfig for visualization settings

    Returns:
        warped_block_image, warped_colored_image
    """
```

**Displacement Pair Format**:
```python
displacement_pairs = [
    [[source_x, source_y], [dest_x, dest_y]],
    # source = where feature is in ORIGINAL image
    # dest = where feature SHOULD BE (template)
]
```

---

### Step 1: Parse and Filter Displacement Pairs

**Code**: `piecewise_affine_delaunay.py:48-59`

```python
parsed_displacement_pairs = [
    [list(map(round, source_point)), list(map(round, destination_point))]
    for [source_point, destination_point] in displacement_pairs
    if MathUtils.rectangle_contains(source_point, warped_rectangle)
    and MathUtils.rectangle_contains(destination_point, warped_rectangle)
]

logger.info(
    f"displacement_pairs: {len(displacement_pairs)} -> {len(parsed_displacement_pairs)}"
)

if len(parsed_displacement_pairs) == 0:
    logger.warning(
        f"Invalid displacement points, no point-pair found in the given rectangle: {warped_rectangle}"
    )
    return gray_block_image, colored_block_image
```

#### Why Round?
- Delaunay triangulation uses integer coordinates
- Sub-pixel precision lost but acceptable for OMR alignment

#### Why Filter?
- Points outside `warped_rectangle` cause triangulation errors
- Ensures all triangles are within bounds

**Example**:
```python
warped_rectangle = (0, 0, 500, 400)  # (x, y, w, h)
displacement_pairs = [
    [[100, 150], [105, 148]],  # INSIDE → KEEP
    [[250, 200], [252, 198]],  # INSIDE → KEEP
    [[-10, 50], [-8, 52]],     # source OUTSIDE → REJECT
    [[100, 500], [102, 498]],  # dest OUTSIDE → REJECT
]
# Result: 2 valid pairs
```

---

### Step 2: Build Delaunay Triangulation

**Code**: `piecewise_affine_delaunay.py:75-83`

```python
# Create Delaunay subdivider
destination_subdiv = cv2.Subdiv2D(warped_rectangle)

# Bulk insert all destination points
destination_subdiv.insert(
    [
        (int(destination_point[0]), int(destination_point[1]))
        for [_source_point, destination_point] in parsed_displacement_pairs
    ]
)
```

#### What is cv2.Subdiv2D?

OpenCV's Delaunay triangulation and Voronoi diagram builder.

**Input**:
- Rectangle bounds: `(x, y, width, height)`
- Points: `[(x1, y1), (x2, y2), ...]`

**Output**:
- Delaunay triangles: List of triangles covering all points
- Voronoi facets: Dual graph (used for visualization)

**Algorithm** (Bowyer-Watson):
```
1. Start with super-triangle containing all points
2. For each point:
   a. Find all triangles whose circumcircle contains the point
   b. Remove these triangles
   c. Retriangulate the "hole" with new point
3. Remove super-triangle vertices
```

**Time Complexity**: O(n log n) average, O(n²) worst case

---

### Step 3: Optional Voronoi Visualization

**Code**: `piecewise_affine_delaunay.py:82-83` + `draw_voronoi` function

```python
initial_voronoi_image = draw_voronoi(warped_block_image, destination_subdiv)
InteractionUtils.show("initial_voronoi_image", initial_voronoi_image, 0)
```

**What is Voronoi Diagram?**
- Partitions space into regions
- Each region = all points closest to one seed point
- Dual of Delaunay triangulation

**Visualization** (if `show_image_level >= 5`):
```
┌────────────────────────────────────┐
│  ╱╲    Voronoi cells (colored)    │
│ ╱  ╲   Each cell corresponds to   │
│╱____╲  one feature point          │
│╲    ╱  Boundaries = equidistant   │
│ ╲  ╱   from neighboring points    │
│  ╲╱                                │
└────────────────────────────────────┘
```

**Purpose**: Debug feature distribution and triangulation quality

---

### Step 4: Extract Delaunay Triangles

**Code**: `piecewise_affine_delaunay.py:85-107`

```python
# Get all triangles (includes some outside bounds)
destination_delaunay_triangles_list = [
    [(round(triangle[2 * i]), round(triangle[2 * i + 1])) for i in range(3)]
    for triangle in destination_subdiv.getTriangleList()
]

# Filter triangles completely inside warped_rectangle
destination_delaunay_triangles = [
    triangle
    for triangle in destination_delaunay_triangles_list
    if all(
        MathUtils.rectangle_contains(point, warped_rectangle) for point in triangle
    )
]

logger.info(
    f"destination_delaunay_triangles: {len(destination_delaunay_triangles_list)} -> {len(destination_delaunay_triangles)} inside rectangle"
)

if len(destination_delaunay_triangles) == 0:
    logger.warning("No valid triangles found")
    return warped_block_image, warped_colored_image
```

#### Triangle Format

`getTriangleList()` returns flat arrays:
```python
# OpenCV output:
triangle = [x1, y1, x2, y2, x3, y3]  # 6 floats

# Parsed to:
triangle = [(x1, y1), (x2, y2), (x3, y3)]  # 3 points
```

#### Why Filter Again?

Delaunay may create triangles with vertices outside bounds:
- Super-triangle vertices
- Edge triangles extending beyond rectangle

**Example**:
```python
warped_rectangle = (0, 0, 500, 400)
all_triangles = [
    [(100, 150), (200, 180), (150, 250)],  # INSIDE → KEEP
    [(450, 380), (490, 390), (510, 420)],  # OUTSIDE (510 > 500) → REJECT
    [(10, 10), (50, 10), (30, 30)],        # INSIDE → KEEP
]
# Result: 2 valid triangles
```

---

### Step 5: Map to Source Triangles

**Code**: `piecewise_affine_delaunay.py:109-122`

```python
# Create reverse mapping: destination → source
destination_to_source_point_map = {
    tuple(destination_point): source_point
    for [source_point, destination_point] in parsed_displacement_pairs
}

# Get corresponding source triangles
source_delaunay_triangles = [
    list(map(destination_to_source_point_map.get, destination_triangle))
    for destination_triangle in destination_delaunay_triangles
]
```

#### Point Correspondence

**Delaunay Property**: Same topological structure
- If points A, B, C form a triangle in destination
- Then corresponding A', B', C' form a triangle in source

**Example**:
```python
# Displacement pairs:
pairs = [
    [[100, 150], [105, 148]],  # P1: source=(100,150), dest=(105,148)
    [[200, 180], [202, 178]],  # P2: source=(200,180), dest=(202,178)
    [[150, 250], [152, 248]],  # P3: source=(150,250), dest=(152,248)
]

# Destination triangle:
dest_tri = [(105, 148), (202, 178), (152, 248)]  # P1, P2, P3

# Lookup source triangle:
src_tri = [(100, 150), (200, 180), (150, 250)]  # P1', P2', P3'
```

---

### Step 6: Per-Triangle Affine Warping Loop

**Code**: `piecewise_affine_delaunay.py:126-146`

```python
for source_points, destination_points in zip(
    source_delaunay_triangles, destination_delaunay_triangles, strict=False
):
    # Optional: Debug visualization (show_image_level >= 5)
    if config.outputs.show_image_level >= 5:
        gray_block_image_before = cv2.cvtColor(gray_block_image, cv2.COLOR_GRAY2BGR)
        warped_block_image_before = cv2.cvtColor(warped_block_image, cv2.COLOR_GRAY2BGR)

    # CORE: Warp this triangle
    ImageWarpUtils.warp_triangle_inplace(
        gray_block_image,
        warped_block_image,
        source_points,
        destination_points,
        config,
    )

    # Optional: Visualize warping step (show_image_level >= 5)
    # [Visualization code omitted for brevity - see lines 153-191]
```

**Loop Iteration**:
```
Triangle 1: src=[(100,150), (200,180), (150,250)]
            dst=[(105,148), (202,178), (152,248)]
            → Warp triangle 1

Triangle 2: src=[(200,180), (300,200), (250,280)]
            dst=[(202,178), (302,198), (252,278)]
            → Warp triangle 2

...repeat for all N triangles
```

---

### Step 7: Triangle Warping (Core Algorithm)

**File**: `src/utils/image_warp.py:16-140`

```python
@staticmethod
def warp_triangle_inplace(
    image, warped_image, source_triangle, warped_triangle, show_image_level=False
):
    # 7.1: Check for collinear points (degenerate triangles)
    if MathUtils.check_collinear_points(*source_triangle):
        logger.critical("Found collinear points. Skipping warp step.")
        return
    if MathUtils.check_collinear_points(*warped_triangle):
        logger.critical("Found collinear points. Skipping warp step.")
        return

    # 7.2: Compute bounding boxes
    (
        (source_tl, _source_tr, source_br, _source_bl),
        _source_box_dimensions,
    ) = MathUtils.get_bounding_box_of_points(source_triangle)

    (
        (warped_tl, _warped_tr, warped_br, _warped_bl),
        warped_box_dimensions,
    ) = MathUtils.get_bounding_box_of_points(warped_triangle)

    # 7.3: Shift triangles to origin (relative to bounding box)
    source_shifted_triangle = MathUtils.shift_points_to_origin(
        source_tl, source_triangle
    )
    warped_shifted_triangle = MathUtils.shift_points_to_origin(
        warped_tl, warped_triangle
    )

    # 7.4: Compute affine transform matrix
    triangle_affine_matrix = cv2.getAffineTransform(
        np.float32(source_shifted_triangle),
        np.float32(warped_shifted_triangle),
    )

    # 7.5: Crop source image to bounding box
    source_triangle_box = image[
        source_tl[1] : source_br[1], source_tl[0] : source_br[0]
    ]

    # 7.6: Apply affine warp
    warped_triangle_box = cv2.warpAffine(
        source_triangle_box,
        triangle_affine_matrix,
        warped_box_dimensions,
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # 7.7: Replace triangle region in output image
    ImageWarpUtils.replace_triangle_inplace(
        warped_image,
        warped_shifted_triangle,
        warped_triangle_box,
        (warped_tl, warped_br),
        warped_box_dimensions,
    )
```

#### Step 7.1: Collinearity Check

**Why?** Collinear points don't form a valid triangle
- `cv2.getAffineTransform()` requires 3 non-collinear points
- Would produce singular matrix (non-invertible)

**Check**:
```python
def check_collinear_points(point1, point2, point3):
    [x1, y1], [x2, y2], [x3, y3] = point1, point2, point3
    return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)
    # Cross product = 0 → collinear
```

**Example**:
```python
# Collinear (all on line y=x):
[(0,0), (1,1), (2,2)]  → SKIP

# Non-collinear:
[(0,0), (1,0), (0,1)]  → PROCESS
```

#### Step 7.2-7.3: Bounding Box & Origin Shift

**Purpose**: Work with smaller image regions for efficiency

```python
# Original triangle (in image coordinates):
source_triangle = [(100, 150), (200, 180), (150, 250)]

# Bounding box:
source_tl = (100, 150)  # Top-left
source_br = (200, 250)  # Bottom-right
box_dimensions = (100, 100)  # width=200-100, height=250-150

# Shifted triangle (relative to bounding box):
source_shifted_triangle = [(0, 0), (100, 30), (50, 100)]
# Point (100, 150) → (0, 0)
# Point (200, 180) → (100, 30)
# Point (150, 250) → (50, 100)
```

**Benefit**: Transform operates on small ROI, not full image

#### Step 7.4: Affine Transform Matrix

**cv2.getAffineTransform(src_triangle, dst_triangle)**:

Given 3 point correspondences, finds 2x3 affine matrix `M`:
```
[x']   [a  b  c]   [x]
[y'] = [d  e  f] × [y]
                   [1]

x' = ax + by + c
y' = dx + ey + f
```

**Example**:
```python
src = [(0, 0), (100, 0), (0, 100)]
dst = [(5, 3), (103, 2), (2, 98)]

M = cv2.getAffineTransform(np.float32(src), np.float32(dst))
# M ≈ [[1.02, -0.01, 5],
#      [0.01,  0.98, 3]]
# Represents: slight scale, small rotation, translation
```

#### Step 7.5-7.6: Crop & Warp

**Crop**:
```python
source_triangle_box = image[
    source_tl[1] : source_br[1],  # Rows (y-axis)
    source_tl[0] : source_br[0]   # Cols (x-axis)
]
# Extract small ROI containing triangle
```

**Warp**:
```python
warped_triangle_box = cv2.warpAffine(
    source_triangle_box,         # Input image (small ROI)
    triangle_affine_matrix,      # 2x3 affine matrix
    warped_box_dimensions,       # Output size (w, h)
    flags=cv2.INTER_LINEAR,      # Bilinear interpolation
    borderMode=cv2.BORDER_REFLECT_101  # Border handling
)
```

**Interpolation**:
- `INTER_LINEAR`: Bilinear (fast, smooth)
- `INTER_CUBIC`: Bicubic (slower, smoother)
- `INTER_NEAREST`: Nearest neighbor (fastest, blocky)

**Border Mode**:
- `BORDER_REFLECT_101`: Mirror at borders (avoids black edges)

#### Step 7.7: Triangle Blending

**File**: `src/utils/image_warp.py:141-219`

```python
@staticmethod
def replace_triangle_inplace(
    source_image,
    shifted_triangle,
    warped_triangle_box,
    warped_tl_br,
    warped_box_dimensions,
):
    tl, br = warped_tl_br
    dest_w, dest_h = warped_box_dimensions

    # Determine grayscale vs color
    num_channels = 1 if len(warped_triangle_box.shape) == 2 else warped_triangle_box.shape[2]

    if num_channels == 3:
        # Create color mask
        white_triangle = np.zeros((dest_h, dest_w, 3), dtype=np.float32)
        cv2.fillConvexPoly(
            white_triangle,
            np.int32(shifted_triangle),
            (1.0, 1.0, 1.0),  # White
            cv2.LINE_AA,
        )
    else:
        # Create grayscale mask
        white_triangle = np.zeros((dest_h, dest_w), dtype=np.float32)
        cv2.fillConvexPoly(
            white_triangle,
            np.int32(shifted_triangle),
            1.0,  # White
            cv2.LINE_AA,
        )

    # Invert mask
    black_triangle = 1.0 - white_triangle

    # Extract triangle-only from warped image
    triangle_from_warped_image = (warped_triangle_box * white_triangle).astype(np.uint8)

    # Extract non-triangle from destination
    source_triangle_box = source_image[tl[1] : br[1], tl[0] : br[0]]
    background_from_source_image = (source_triangle_box * black_triangle).astype(np.uint8)

    # Blend: background + warped triangle
    source_image[tl[1] : br[1], tl[0] : br[0]] = (
        background_from_source_image + triangle_from_warped_image
    )
```

**Masking Process**:

```
Step 1: Create white triangle mask
┌─────────────┐
│             │  white_triangle:
│   ╱╲        │  1.0 inside triangle
│  ╱  ╲       │  0.0 outside
│ ╱____╲      │
│             │
└─────────────┘

Step 2: Invert to get black triangle
┌─────────────┐
│▓▓▓▓▓▓▓▓▓▓▓▓▓│  black_triangle:
│▓▓▓╱╲▓▓▓▓▓▓▓▓│  0.0 inside triangle
│▓▓╱  ╲▓▓▓▓▓▓▓│  1.0 outside
│▓╱____╲▓▓▓▓▓▓│
│▓▓▓▓▓▓▓▓▓▓▓▓▓│
└─────────────┘

Step 3: Extract warped triangle
warped_box * white_mask = triangle only

Step 4: Extract background
source_box * black_mask = background only

Step 5: Blend
result = background + triangle
```

**Anti-Aliasing**:
- `cv2.LINE_AA`: Anti-aliased edges
- Smooth blending at triangle boundaries

---

### Step 8: Optional Debug Visualization

**Code**: `piecewise_affine_delaunay.py:153-191`

If `config.outputs.show_image_level >= 5`:

```python
# Draw source triangle (red) and destination triangle (green)
DrawingUtils.draw_polygon(gray_block_image_before, source_points, color=CLR_DARK_RED)
DrawingUtils.draw_polygon(gray_block_image_before, destination_points, color=CLR_DARK_GREEN)

# Show before/after comparison
overlay = ImageUtils.overlay_image(
    gray_block_image_before, warped_block_image_after
)

InteractionUtils.show(
    f"warped_block_image-{destination_points}",
    ImageUtils.get_padded_hstack([
        gray_block_image_before,
        warped_block_image_before,
        warped_block_image_after,
        overlay,
    ]),
    0,
)
```

**Visualization**:
```
┌───────────┬───────────┬───────────┬───────────┐
│ Source    │ Dest      │ Warped    │ Overlay   │
│ (before)  │ (before)  │ (after)   │ (blend)   │
│ + Red Tri │ + Blue Tri│ + Green   │           │
└───────────┴───────────┴───────────┴───────────┘
```

---

### Step 9: Return Warped Images

**Code**: `piecewise_affine_delaunay.py:193`

```python
return warped_block_image, warped_colored_image
```

**Output**:
- `warped_block_image`: Grayscale image with all triangles warped
- `warped_colored_image`: Colored version (currently not fully implemented - see line 148-151 TODO)

---

## Why Currently Unused?

**Problem**: Cannot handle overlapping field blocks

**Scenario**:
```
Field Block A: [x=100, y=100, w=200, h=300]
Field Block B: [x=150, y=150, w=200, h=300]  ← Overlaps with A

# Method 2 (Piecewise Affine - Image Warping):
warped_A = apply_piecewise_affine(block_A, ...)
warped_B = apply_piecewise_affine(block_B, ...)
gray_image[100:400, 100:300] = warped_A  # Write A
gray_image[150:450, 150:350] = warped_B  # OVERWRITES overlap region!
# Result: Conflict, data loss
```

**Current Solution**: Method 3 (K-Nearest Interpolation)
- Warps COORDINATES instead of images
- No conflict when field blocks overlap

**Future TODO** (line 128-129):
```python
# TODO: modify this loop to support 4-point transforms too!
# TODO: figure out how to apply detection on these copies to support overlapping field blocks!
```

---

## Data Structures

### Displacement Pairs
```python
[
    [[source_x, source_y], [dest_x, dest_y]],
    ...
]
# source: Position in ORIGINAL scanned image
# dest: Position in TEMPLATE (where it should be)
```

### Warped Rectangle
```python
(x, y, width, height)  # 4-tuple
# Example: (0, 0, 500, 400)
```

### Delaunay Triangles
```python
[
    [(x1, y1), (x2, y2), (x3, y3)],  # Triangle 1
    [(x4, y4), (x5, y5), (x6, y6)],  # Triangle 2
    ...
]
```

### Affine Transform Matrix
```python
np.array([
    [a, b, c],
    [d, e, f]
], dtype=np.float32)  # 2x3 matrix
```

---

## Performance Characteristics

### Time Complexity
| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Delaunay triangulation | O(n log n) | 5-20ms for 50-200 points |
| Per-triangle warp | O(area) | 1-5ms per triangle |
| Total (N triangles) | O(n log n + N × area) | **50-500ms** per field block |

Where:
- n = number of feature points (10-200)
- N = number of triangles (typically 2n - 5 for Delaunay)
- area = bounding box size of triangle

### Memory Complexity
```python
# Per field block:
# Parsed pairs: 50 × 2 × 2 × 8 bytes = 1.6 KB (float64)
# Triangle list: 100 × 3 × 2 × 4 bytes = 2.4 KB (int32)
# Affine matrices: 100 × 2 × 3 × 4 bytes = 2.4 KB (float32)
# Warped image: 500 × 400 × 1 byte = 200 KB (grayscale)
# Total: ~210 KB per field block
```

### Comparison with K-Nearest Interpolation

| Metric | Piecewise Affine | K-Nearest Interpolation |
|--------|------------------|-------------------------|
| **Speed** | Slower (100-500ms) | Faster (20-50ms) |
| **Accuracy** | Higher (localized warping) | Lower (averaging) |
| **Memory** | Higher (warped images) | Lower (shift vectors only) |
| **Overlap Support** | No (image conflicts) | Yes (coordinate-based) |
| **Use Case** | High distortion, non-overlap | Moderate distortion, overlaps allowed |

---

## Edge Cases

### 1. Collinear Points

**Cause**: Three feature points on a line
```python
triangle = [(100, 100), (150, 150), (200, 200)]  # All on y=x line
```

**Handling**:
```python
if MathUtils.check_collinear_points(*triangle):
    logger.critical("Skipping collinear triangle")
    return  # Skip this triangle, continue with others
```

**Impact**: One triangle skipped, rest of warping continues

---

### 2. Zero Displacement Pairs

**Cause**: No valid features found in rectangle
```python
parsed_displacement_pairs = []  # Empty after filtering
```

**Handling**:
```python
if len(parsed_displacement_pairs) == 0:
    logger.warning("No valid displacement points")
    return gray_block_image, colored_block_image  # Return original
```

**Impact**: No warping applied, equivalent to no alignment

---

### 3. All Triangles Outside Bounds

**Cause**: Delaunay creates only edge triangles
```python
destination_delaunay_triangles = []  # All filtered out
```

**Handling**:
```python
if len(destination_delaunay_triangles) == 0:
    logger.warning("No valid triangles found")
    return warped_block_image, warped_colored_image
```

**Impact**: No warping applied

---

### 4. Degenerate Triangles (Very Small Area)

**Problem**: Tiny triangles cause numerical instability
```python
triangle = [(100, 100), (100.1, 100), (100, 100.1)]  # Area ≈ 0.005
```

**Not Explicitly Handled**: May cause poor affine transforms
**Mitigation**: Delaunay naturally avoids slivers (maximizes minimum angle)

---

### 5. Very Large Affine Distortions

**Problem**: Extreme scaling or shearing
```python
src_triangle = [(0, 0), (10, 0), (0, 10)]
dst_triangle = [(0, 0), (100, 0), (0, 10)]  # 10× stretch in x
```

**Handling**: cv2.warpAffine handles gracefully
- May introduce visual artifacts (pixelation)
- Interpolation smooths results

---

### 6. Coordinate Rounding Errors

**Cause**: Rounding floats to integers for Delaunay
```python
source_point = [152.7, 198.3]
rounded = [153, 198]  # 0.7, 0.3 precision lost
```

**Impact**: Minor shift (< 1 pixel), acceptable for OMR

---

### 7. Triangle Mask Aliasing

**Problem**: Hard edges at triangle boundaries
```python
# Without LINE_AA:
cv2.fillConvexPoly(mask, triangle, 1.0)  # Jagged edges
```

**Handling**:
```python
cv2.fillConvexPoly(mask, triangle, 1.0, cv2.LINE_AA)  # Anti-aliased
```

**Result**: Smooth blending, no visible seams

---

## Browser Migration Notes

### 1. Delaunay Triangulation

**OpenCV.js Support**: ✅ `cv.Subdiv2D` available

```javascript
// Create subdivision
const rect = new cv.Rect(0, 0, 500, 400);
const subdiv = new cv.Subdiv2D(rect);

// Insert points
const points = new cv.Point2fVector();
for (const [dx, dy] of destinationPoints) {
  points.push_back(new cv.Point2f(dx, dy));
}
subdiv.insert(points);

// Get triangles
const triangleList = new cv.Mat();
subdiv.getTriangleList(triangleList);

// Cleanup
points.delete();
triangleList.delete();
subdiv.delete();
```

**Memory Management**: ⚠️ CRITICAL - Must delete all objects

---

### 2. Affine Transform

**OpenCV.js Support**: ✅ `cv.getAffineTransform`, `cv.warpAffine`

```javascript
// Get transform matrix
const srcMat = cv.matFromArray(3, 1, cv.CV_32FC2, [
  x1, y1, x2, y2, x3, y3
]);
const dstMat = cv.matFromArray(3, 1, cv.CV_32FC2, [
  x1p, y1p, x2p, y2p, x3p, y3p
]);
const M = cv.getAffineTransform(srcMat, dstMat);

// Warp image
const warped = new cv.Mat();
cv.warpAffine(
  src,
  warped,
  M,
  new cv.Size(width, height),
  cv.INTER_LINEAR,
  cv.BORDER_REFLECT_101
);

// Cleanup
srcMat.delete();
dstMat.delete();
M.delete();
warped.delete();
```

---

### 3. Triangle Masking

**OpenCV.js Support**: ✅ `cv.fillConvexPoly`

```javascript
// Create mask
const mask = new cv.Mat.zeros(height, width, cv.CV_32FC1);
const trianglePts = cv.matFromArray(3, 1, cv.CV_32SC2, [
  x1, y1, x2, y2, x3, y3
]);

cv.fillConvexPoly(
  mask,
  trianglePts,
  new cv.Scalar(1.0),
  cv.LINE_AA,
  0
);

// Cleanup
trianglePts.delete();
mask.delete();
```

---

### 4. Performance Optimization

**Web Workers**: Offload heavy computation
```javascript
// main.js
const worker = new Worker('piecewise-affine-worker.js');
worker.postMessage({
  imageData: imageData,
  displacementPairs: pairs,
  rectangle: rect
});

worker.onmessage = (e) => {
  const warpedImage = e.data.warpedImage;
  // Use result
};
```

**OffscreenCanvas**: Render in worker
```javascript
// worker.js
const offscreen = new OffscreenCanvas(width, height);
const ctx = offscreen.getContext('2d');
// Perform warping
postMessage({ warpedImage: ctx.getImageData(0, 0, width, height) });
```

---

### 5. Memory Management Pattern

```javascript
function applyPiecewiseAffine(image, pairs, rect) {
  const resources = [];

  try {
    const subdiv = new cv.Subdiv2D(rect);
    resources.push(subdiv);

    // ... processing

    return warpedImage;
  } finally {
    // Cleanup all resources
    resources.forEach(r => r.delete());
  }
}
```

---

## Related Flows

- **SIFT Matching** (`../sift/flows.md`) - Provides displacement pairs
- **K-Nearest Interpolation** (`../k-nearest/flows.md`) - Alternative coordinate warping
- **Alignment Main Flow** (`../flows.md`) - Overall alignment orchestration
- **Image Warp Utils** (`../../utils/image/flows.md`) - Triangle warping implementation

---

## Testing Considerations

### Unit Tests

```python
def test_piecewise_affine_perfect_alignment():
    # Same image → no warping needed
    displacement_pairs = [
        [[100, 150], [100, 150]],  # Zero displacement
        [[200, 180], [200, 180]],
        [[150, 250], [150, 250]],
    ]
    warped = apply_piecewise_affine(image, image, displacement_pairs, rect, config)
    assert np.allclose(warped, image)  # Should be identical

def test_piecewise_affine_small_shift():
    # Uniform shift [5, -3]
    displacement_pairs = [
        [[100, 150], [105, 147]],
        [[200, 180], [205, 177]],
        [[150, 250], [155, 247]],
    ]
    warped = apply_piecewise_affine(image, None, displacement_pairs, rect, config)
    # Check warped image shifted correctly (approximately)

def test_collinear_points():
    # All points on a line
    displacement_pairs = [
        [[0, 0], [0, 0]],
        [[50, 50], [50, 50]],
        [[100, 100], [100, 100]],
    ]
    # Should handle gracefully (skip triangles or return original)
    warped = apply_piecewise_affine(image, None, displacement_pairs, rect, config)
    assert warped is not None

def test_empty_displacement_pairs():
    displacement_pairs = []
    warped = apply_piecewise_affine(image, None, displacement_pairs, rect, config)
    assert np.array_equal(warped, image)  # Return original
```

### Integration Tests

1. **Real scan with distortion**: Apply to warped image, verify alignment
2. **Multiple field blocks**: Ensure no conflicts (note: current limitation)
3. **Visualization**: Check debug images generated correctly
4. **Performance**: Measure time for various point counts (10, 50, 200)

---

## Summary

**Piecewise Affine Alignment** provides high-quality localized warping through Delaunay triangulation and per-triangle affine transforms. While currently unused due to overlapping field block conflicts, it remains a powerful technique for scenarios requiring precise non-rigid alignment.

**Key Takeaways**:
- ✅ Handles complex distortions (bending, warping)
- ✅ Delaunay ensures stable triangulation
- ✅ Per-triangle transforms preserve local geometry
- ❌ Cannot handle overlapping field blocks (image conflict)
- ⚠️ Slower than K-Nearest (100-500ms vs 20-50ms)
- ✅ Browser-compatible via OpenCV.js
