# Piecewise Affine Alignment - Constraints & Edge Cases

## Core Constraints

### 1. Minimum Point Requirement

**Hard Limit**: Requires at least 3 displacement pairs

**Rationale**:
- Delaunay triangulation needs ≥ 3 points
- Affine transform requires exactly 3 non-collinear points
- Fewer points → no triangulation possible

**Failure Mode**:
```python
# Case 1: Empty displacement pairs
parsed_displacement_pairs = []
# Result: Return original image, no warping

# Case 2: Only 2 points
parsed_displacement_pairs = [
    [[100, 150], [105, 148]],
    [[200, 180], [202, 178]],
]
# Delaunay may create degenerate triangulation
# May fail or create super-triangle only
```

**Recommended Minimum**: 10-20 points for robust triangulation

---

### 2. Rectangle Bounds Constraint

**Hard Requirement**: All points must be within `warped_rectangle`

**Format**:
```python
warped_rectangle = (x, y, width, height)
# Example: (0, 0, 500, 400)
# Valid range: x ∈ [0, 500], y ∈ [0, 400]
```

**Filtering**:
```python
if not (
    MathUtils.rectangle_contains(source_point, warped_rectangle)
    and MathUtils.rectangle_contains(destination_point, warped_rectangle)
):
    # REJECT this displacement pair
```

**Why Strict?**
- OpenCV Subdiv2D crashes on out-of-bounds points
- Triangle masking fails if vertices outside image
- Prevents buffer overflow in image operations

**Edge Case**:
```python
# Point exactly on boundary:
point = [500, 400]  # Right-bottom corner of (0, 0, 500, 400)
# Behavior depends on rectangle_contains() implementation
# Current: Uses `<=` so boundary points INCLUDED
```

---

### 3. Non-Collinear Points Constraint

**Requirement**: Every triangle must have 3 non-collinear vertices

**Check**:
```python
def check_collinear_points(p1, p2, p3):
    [x1, y1], [x2, y2], [x3, y3] = p1, p2, p3
    # Cross product == 0 → collinear
    return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)
```

**Failure Scenario**:
```python
# Three points on a line:
triangle = [(100, 100), (150, 150), (200, 200)]  # All on y=x

# cv2.getAffineTransform() will:
# - Produce singular matrix (determinant = 0)
# - Raise cv2.error or return invalid matrix
```

**Handling**:
```python
if MathUtils.check_collinear_points(*source_triangle):
    logger.critical("Skipping warp for collinear triangle")
    return  # Skip this triangle, continue with others
```

**Prevention**: Delaunay naturally avoids collinear triangles (maximizes minimum angle)

---

### 4. Triangle Count Constraint

**Delaunay Property**: For n points, creates approximately 2n - 5 triangles

**Example**:
```python
n = 50 points → ~95 triangles
n = 100 points → ~195 triangles
n = 200 points → ~395 triangles
```

**Performance Impact**:
- Each triangle: 1-5ms to warp
- 100 triangles: 100-500ms total
- Too many triangles → slow processing

**Practical Limit**: 50-200 points (100-400 triangles) for interactive performance

---

### 5. Affine Transform Constraints

**Input**: 3 non-collinear point pairs
**Output**: 2×3 affine matrix

```python
M = cv2.getAffineTransform(src_pts, dst_pts)
# src_pts, dst_pts: shape (3, 2) float32
```

**Mathematical Constraints**:
- Preserves parallelism (parallel lines → parallel)
- Preserves ratios along lines
- Does NOT preserve angles or lengths (unless rotation+translation only)

**Ill-Conditioned Cases**:
```python
# Near-collinear points (small triangle area):
src = [(0, 0), (100, 0), (0.1, 0.1)]  # Area ≈ 5
# May produce numerically unstable transform

# Extreme aspect ratio:
src = [(0, 0), (1000, 0), (0, 1)]  # Very elongated
# May amplify numerical errors
```

**OpenCV Handling**: Usually robust, but may produce poor results for degenerate cases

---

### 6. Interpolation Constraints

**cv2.warpAffine() Modes**:

| Mode | Quality | Speed | Constraints |
|------|---------|-------|-------------|
| `INTER_NEAREST` | Blocky | Fastest | No floating-point ops |
| `INTER_LINEAR` | Smooth | Fast | Default choice |
| `INTER_CUBIC` | Smoothest | Slower | May overshoot (ringing) |
| `INTER_LANCZOS4` | Best | Slowest | Expensive, rare use |

**Current Choice**: `INTER_LINEAR` (good balance)

**Constraint**: Interpolation cannot recover lost information
- Downsampling → data loss
- Upsampling → blurriness

---

### 7. Border Mode Constraints

**cv2.warpAffine() Border Modes**:

```python
# Current:
borderMode=cv2.BORDER_REFLECT_101

# Alternatives:
cv2.BORDER_CONSTANT      # Fill with black (0)
cv2.BORDER_REPLICATE     # Repeat edge pixels
cv2.BORDER_REFLECT       # Mirror (abcdcba)
cv2.BORDER_REFLECT_101   # Mirror (abcddcba) - smoother
cv2.BORDER_WRAP          # Wrap around (abcdabcd)
```

**Why REFLECT_101?**
- Avoids black borders (CONSTANT)
- Smoother than REPLICATE
- Natural appearance for document edges

**Constraint**: Border only affects pixels transformed OUTSIDE source bounds

---

### 8. Overlapping Field Blocks (CRITICAL LIMITATION)

**Problem**: Image-based warping conflicts with overlaps

**Scenario**:
```python
# Template with overlapping field blocks:
field_block_A = {
    "origin": [100, 100],
    "dimensions": [200, 300]
}
field_block_B = {
    "origin": [150, 150],  # Overlaps with A
    "dimensions": [200, 300]
}

# Processing:
warped_A = apply_piecewise_affine(block_A, ...)
warped_B = apply_piecewise_affine(block_B, ...)

# Placement:
gray_image[100:400, 100:300] = warped_A  # Write A
gray_image[150:450, 150:350] = warped_B  # OVERWRITES overlap!

# Result: Region [150:400, 150:300] has data from B only
# Data from A is LOST in overlap zone
```

**Why This Matters**:
- OMR templates often have overlapping field blocks
- Example: Roll number field overlaps with name field
- Cannot safely use image warping

**Current Solution**: Use K-Nearest Interpolation (coordinate warping)
- Shifts bubble positions, not images
- No conflicts when field blocks overlap

**Future TODO** (line 128-129 in code):
```python
# TODO: figure out how to apply detection on these copies to
# support overlapping field blocks!
```

**Possible Solutions**:
1. **Blending**: Weighted average in overlap zones (complex)
2. **Priority**: Higher-priority field block wins (loses data)
3. **Coordinate Warping**: Current approach (K-Nearest)
4. **Per-Field Warping**: Warp individual fields, not blocks (expensive)

---

### 9. Memory Constraints

**Per Field Block Memory Usage**:

```python
# Image size: 500×400 pixels (typical field block)
# Grayscale: 500 × 400 × 1 byte = 200 KB
# Colored: 500 × 400 × 3 bytes = 600 KB
# Warped copy: +200 KB (grayscale) or +600 KB (colored)

# Triangulation data:
# - Displacement pairs: 100 × 2 × 2 × 8 bytes = 3.2 KB
# - Triangle list: 200 × 3 × 2 × 4 bytes = 4.8 KB
# - Affine matrices: 200 × 2 × 3 × 4 bytes = 4.8 KB
# Total: ~13 KB (negligible)

# Peak memory per field block: ~1.6 MB (if colored)
# Template with 20 field blocks: 20 × 1.6 MB = 32 MB
```

**Browser Constraint**: Heap size limited (1-4 GB)
- 100 field blocks at 1.6 MB each = 160 MB (acceptable)
- Large templates (100+ blocks) may hit limits

**Mitigation**:
- Process field blocks sequentially (not all at once)
- Use grayscale only (saves 3× memory)
- Downsample large images before warping

---

### 10. Coordinate Precision Constraints

**Float → Int Rounding**:

```python
# SIFT features (sub-pixel precision):
source_point = [152.73, 198.41]

# Delaunay triangulation (integer coordinates):
rounded_point = [153, 198]

# Precision loss: [0.73, 0.41] pixels
```

**Impact**:
- Triangle vertices shifted by up to 0.5 pixels
- Affine transform slightly different
- **Acceptable** for OMR (bubbles are 10-30 pixels wide)

**Accumulation**: Errors don't accumulate (each triangle independent)

---

### 11. Voronoi Visualization Constraint

**cv2.Subdiv2D.getVoronoiFacetList()**:

**Known Issue**: May fail on degenerate point sets
```python
(facets, centers) = subdiv.getVoronoiFacetList([])
# May return empty facets for some configurations
```

**Not Critical**: Visualization only, doesn't affect warping

**Workaround**: Check `len(facets) > 0` before drawing

---

### 12. Triangle Size Constraints

**Very Large Triangles**: Slow warping
```python
# Triangle covering entire image:
triangle = [(0, 0), (2000, 0), (0, 1500)]
bounding_box_area = 2000 × 1500 = 3,000,000 pixels

# Warp time: O(area) = ~30-50ms for 3M pixels
```

**Very Small Triangles**: Numerical instability
```python
# Tiny triangle:
triangle = [(100.0, 100.0), (100.1, 100.0), (100.0, 100.1)]
area = 0.005 pixels²

# After rounding to int:
triangle_int = [(100, 100), (100, 100), (100, 100)]  # COLLAPSED!
# Becomes collinear, skipped
```

**Delaunay Benefit**: Balances triangle sizes (maximizes minimum angle)

---

## Edge Cases

### 1. Empty Displacement Pairs

**Cause**: All points filtered out (outside bounds)

```python
displacement_pairs = [
    [[-10, 50], [-8, 52]],     # source outside
    [[100, 500], [102, 498]],  # dest outside
]
# After filtering:
parsed_displacement_pairs = []
```

**Handling**:
```python
if len(parsed_displacement_pairs) == 0:
    logger.warning("No valid displacement points")
    return gray_block_image, colored_block_image
```

**Result**: Original image returned unchanged

---

### 2. Insufficient Points (< 3)

**Cause**: Only 1-2 valid points

```python
parsed_displacement_pairs = [
    [[100, 150], [105, 148]],
    [[200, 180], [202, 178]],
]  # Only 2 points
```

**Delaunay Behavior**:
- 1 point: No triangles created
- 2 points: Only super-triangle edges (filtered out)
- 3+ points: Valid triangulation

**Handling**: Filtered triangles will be empty
```python
if len(destination_delaunay_triangles) == 0:
    return warped_block_image, warped_colored_image
```

---

### 3. All Triangles Filtered Out

**Cause**: Delaunay creates triangles extending outside bounds

```python
# Points near edge:
points = [(5, 5), (495, 5), (250, 200)]
warped_rectangle = (0, 0, 500, 400)

# Delaunay may create edge triangles with vertices at:
# (-50, -50), (550, -50), etc. (super-triangle remnants)

# Filtering:
destination_delaunay_triangles = []  # All rejected
```

**Handling**: Same as insufficient points (return original)

---

### 4. Collinear Points Detected

**Cause**: SIFT features along a line (e.g., ruler edge)

```python
# Features along top edge:
displacement_pairs = [
    [[100, 50], [105, 48]],
    [[200, 50], [205, 48]],
    [[300, 50], [305, 48]],
]
# All y ≈ 50 → nearly collinear
```

**Detection**:
```python
if MathUtils.check_collinear_points(*source_triangle):
    logger.critical("Skipping collinear triangle")
    return  # Skip this triangle
```

**Impact**: Some triangles skipped, rest processed normally

---

### 5. Extreme Affine Distortions

**Scenario**: Large perspective change or paper fold

```python
# Source (flat scan):
src_triangle = [(100, 100), (200, 100), (150, 200)]

# Destination (warped template):
dst_triangle = [(100, 100), (300, 80), (140, 250)]
# Stretched horizontally, compressed vertically
```

**cv2.warpAffine() Behavior**:
- Applies transformation faithfully
- May produce pixelation (downsampling) or blurriness (upsampling)
- Visual quality degraded but mathematically correct

**Mitigation**: Pre-processing (CropOnMarkers, AutoRotate) reduces extreme distortions

---

### 6. Boundary Triangles (Partial Outside)

**Problem**: Triangle vertices inside bounds, but bounding box extends outside

```python
triangle = [(10, 10), (490, 20), (250, 390)]
bounding_box = [(10, 10), (490, 390)]  # INSIDE

# But after warping:
warped_triangle = [(5, 5), (495, 15), (250, 385)]
warped_bounding_box = [(5, 5), (495, 385)]  # OUTSIDE (x=495 < 500 OK, but x=5 < 0 NO)
```

**Current Filtering**: Checks vertices only, not bounding box
```python
if all(MathUtils.rectangle_contains(point, warped_rectangle) for point in triangle):
    # Accept (checks vertices, not bounding box)
```

**Rare Issue**: May cause index out-of-bounds in edge cases

**Safer Check** (not implemented):
```python
bounding_box, _ = MathUtils.get_bounding_box_of_points(triangle)
if all(MathUtils.rectangle_contains(point, warped_rectangle) for point in bounding_box):
    # Accept
```

---

### 7. Mask Aliasing at Triangle Boundaries

**Problem**: Adjacent triangles may have gaps or overlaps at boundaries

**Cause**: Floating-point rounding in cv2.fillConvexPoly()

**Example**:
```
Triangle A edge: [(100.4, 50.6), (150.7, 80.2)]
Triangle B edge: [(150.7, 80.2), (200.3, 60.1)]

# After rounding:
A edge: [(100, 51), (151, 80)]
B edge: [(151, 80), (200, 60)]

# Possible gap or overlap at (151, 80)
```

**Mitigation**: `cv2.LINE_AA` (anti-aliasing)
- Smooths edges with fractional coverage
- Reduces visible seams

**Impact**: Negligible for OMR (grayscale documents, small triangles)

---

### 8. Warped Rectangle Mismatch

**Problem**: `warped_triangle_box.shape != expected_shape`

**Cause**: cv2.warpAffine() output size mismatch

```python
# Expected:
warped_box_dimensions = (100, 150)  # (width, height)

# cv2.warpAffine() returns:
warped_triangle_box.shape = (150, 100)  # (height, width) - OpenCV format!

# Check:
if warped_triangle_box.shape != tuple(reversed(warped_box_dimensions)):
    raise ImageProcessingError("Shape mismatch")
```

**Note**: OpenCV uses (height, width), NumPy uses (rows, cols)
**Handling**: Explicit error raised (lines 82-90 in `image_warp.py`)

---

### 9. Colored Image Handling

**Current Status**: Partial implementation

```python
# Colored image warping (lines 148-151):
# TODO: modify warped_colored_image as well
# ImageWarpUtils.warp_triangle_inplace(
#     colored_image, warped_colored_image, source_points, destination_points
# )
```

**Issue**: Only grayscale warping fully implemented
**Impact**: Colored outputs may not reflect warping (visualization only)

**Future Fix**: Uncomment and test colored image warping

---

### 10. Zero-Area Triangles

**Cause**: Duplicate points or near-duplicates

```python
# SIFT detects same feature twice:
displacement_pairs = [
    [[100, 150], [105, 148]],
    [[100.1, 150.1], [105.1, 148.1]],  # Very close
]

# After rounding:
parsed_pairs = [
    [[100, 150], [105, 148]],
    [[100, 150], [105, 148]],  # DUPLICATE
]

# Delaunay may create zero-area triangle:
triangle = [(105, 148), (105, 148), (202, 178)]  # First two points identical
```

**Detection**: Collinearity check catches this
```python
check_collinear_points((105,148), (105,148), (202,178))  → True
```

**Handling**: Skip triangle (logged as critical)

---

## Performance Edge Cases

### 1. Very High Point Density

**Scenario**: 500+ displacement pairs in small region

```python
# 500 points in 300×400 region:
# Triangle count: ~995 triangles
# Warp time: 995 × 2ms = ~2 seconds (very slow)
```

**Mitigation**:
- Limit SIFT features (currently unlimited)
- Downsample point cloud (not implemented)
- Use grid-based triangulation (fewer triangles)

---

### 2. Very Large Images

**Scenario**: 7000×10000 pixel scans

```python
# Field block: 2000×3000 pixels
# Warped image memory: 2000 × 3000 × 1 byte = 6 MB
# Per-triangle warp time: 10-20ms (large bounding boxes)
# 200 triangles: 2-4 seconds
```

**Browser Constraint**: May timeout or run out of memory

**Mitigation**:
- Downsample before alignment
- Process in tiles (not implemented)
- Use Web Workers (parallel processing)

---

### 3. Degenerate Geometries in Real Scans

**Examples**:
- Folded paper: Creates sharp creases (high distortion)
- Perspective skew: Parallel lines converge
- Lens distortion: Barrel/pincushion effect

**Piecewise Affine Limitation**: Affine transforms can't handle:
- Rotation (can approximate with small triangles)
- Perspective (parallel lines converge)
- Lens distortion (curved lines)

**Better Suited For**:
- Local stretching/compression
- Paper warping (smooth deformations)
- Non-uniform shifts

---

## Browser-Specific Constraints

### 1. OpenCV.js Memory Management

**CRITICAL**: Must manually delete all cv.Mat objects

```javascript
// WRONG:
const subdiv = new cv.Subdiv2D(rect);
subdiv.insert(points);
// Memory leak! subdiv not deleted

// CORRECT:
const subdiv = new cv.Subdiv2D(rect);
try {
  subdiv.insert(points);
  // ... use subdiv
} finally {
  subdiv.delete();  // MUST DELETE
}
```

**Failure**: Memory accumulates → tab crashes after 10-50 images

---

### 2. Performance Degradation

**Browser vs Python**:

| Operation | Python (OpenCV) | Browser (OpenCV.js) |
|-----------|-----------------|---------------------|
| Delaunay triangulation | 5-20ms | 20-80ms |
| Per-triangle warp | 1-5ms | 3-10ms |
| Total (100 triangles) | 100-500ms | 300-1000ms |

**3-5× slower** in browser (JavaScript, no SIMD optimization)

**Mitigation**:
- Web Workers (parallel processing)
- Reduce point count (fewer triangles)
- Downsample images

---

### 3. Float32Array Constraints

**JavaScript Typed Arrays**: Limited precision

```javascript
// Float32Array (32-bit floats):
const points = new Float32Array([152.73, 198.41]);
// Precision: ~7 decimal digits

// vs Python (64-bit floats):
points = [152.73, 198.41]  # Precision: ~15 decimal digits
```

**Impact**: Minor rounding errors (< 0.001 pixels)
**Acceptable**: For OMR alignment

---

### 4. Web Worker Data Transfer

**Limitation**: Can't transfer cv.Mat directly

```javascript
// WRONG:
worker.postMessage({ mat: cvMat });  // cv.Mat not transferable

// CORRECT:
const data = cvMat.data;  // Uint8Array
worker.postMessage({ data: data }, [data.buffer]);  // Transfer buffer
```

**Reconstruction**:
```javascript
// In worker:
const mat = cv.matFromArray(rows, cols, type, data);
```

**Cost**: Serialization overhead (50-200ms for large images)

---

### 5. Heap Size Limits

**Browser Memory Limits**:
- Chrome: 1-2 GB per tab (depends on system RAM)
- Firefox: 2-4 GB per tab
- Safari: 1-2 GB per tab

**Risk**: Processing 50+ field blocks with piecewise affine
- 50 × 2 MB = 100 MB (peak usage)
- Close to limit if template has 100+ blocks

**Mitigation**: Process sequentially, free memory between blocks

---

## Validation Checks

### Before Piecewise Affine

1. ✓ Displacement pairs not empty
2. ✓ All points within warped_rectangle bounds
3. ✓ warped_rectangle has positive width and height
4. ✓ Images not None, correct dimensions
5. ✓ At least 3 displacement pairs (for triangulation)

### During Processing

1. ✓ Check collinearity for each triangle
2. ✓ Validate triangle count > 0
3. ✓ Verify affine matrix not None
4. ✓ Check warped image shape matches expected dimensions

### After Warping

1. ✓ Warped image not None
2. ✓ Warped image shape correct
3. ✓ No NaN or Inf values in warped image
4. ✓ Pixel values in valid range [0, 255]

---

## Comparison with K-Nearest Interpolation

| Constraint | Piecewise Affine | K-Nearest Interpolation |
|------------|------------------|-------------------------|
| **Minimum points** | 3 (hard limit) | 1 (soft limit, K=4 recommended) |
| **Overlapping blocks** | ❌ Not supported (image conflict) | ✅ Supported (coordinate-based) |
| **Memory usage** | High (warped images) | Low (shift vectors only) |
| **Speed** | Slow (100-500ms) | Fast (20-50ms) |
| **Accuracy** | High (localized warping) | Medium (averaging) |
| **Degenerate cases** | Must skip collinear triangles | Handles gracefully |
| **Browser performance** | 3-5× slower than Python | 2-3× slower than Python |

---

## Summary Table

| Constraint | Value | Rationale |
|------------|-------|-----------|
| **Minimum points** | 3 (hard), 10+ (recommended) | Delaunay requirement |
| **Rectangle bounds** | Strict (all points inside) | Prevent crashes, buffer overflow |
| **Collinearity** | Skip invalid triangles | Avoid singular matrices |
| **Triangle count** | ~2n - 5 (n = points) | Delaunay property |
| **Interpolation** | INTER_LINEAR | Speed/quality balance |
| **Border mode** | BORDER_REFLECT_101 | Natural appearance |
| **Overlapping blocks** | ❌ Not supported | Current limitation |
| **Memory per block** | ~1.6 MB (colored) | Image + overhead |
| **Browser heap** | 1-4 GB per tab | Platform limit |
| **Performance** | 100-500ms (Python), 300-1000ms (browser) | Per field block |

---

## Related Constraints

- **Alignment Constraints** (`../constraints.md`) - Overall alignment system limits
- **SIFT Constraints** (`../sift/constraints.md`) - Feature detection limits (provides displacement pairs)
- **K-Nearest Constraints** (`../k-nearest/constraints.md`) - Alternative approach
- **OpenCV Constraints** (`../../../technical/opencv/opencv-operations.md`) - Platform-specific limitations

---

## Recommendations

### When to Use Piecewise Affine

✅ **Good for**:
- High-quality scans with complex distortions
- Paper warping, bending, or folding
- Non-overlapping field blocks
- Offline processing (batch jobs)
- High accuracy requirements

❌ **Avoid for**:
- Overlapping field blocks (use K-Nearest)
- Real-time processing (too slow)
- Low-quality scans (unreliable features)
- Simple translations (use Phase Correlation)

### Best Practices

1. **Pre-filter displacement pairs**: Remove outliers before triangulation
2. **Validate point count**: Ensure ≥ 10 points for robustness
3. **Monitor triangle count**: Limit to 100-200 for performance
4. **Handle collinearity**: Gracefully skip degenerate triangles
5. **Memory management**: Delete cv.Mat objects in browser
6. **Async processing**: Use Web Workers for heavy computation
7. **Downsample large images**: Process at 300 DPI, not 600 DPI
8. **Visualize triangulation**: Debug with Voronoi diagrams
9. **Profile performance**: Measure time per triangle, optimize hot paths
10. **Fallback strategy**: If piecewise affine fails, fall back to K-Nearest

---

## Future Improvements

### Planned (from TODO comments)

1. **Support 4-point transforms** (line 129):
   - Extend to quadrilaterals (perspective per quad)
   - More accurate for rectangular regions

2. **Colored image warping** (line 148-151):
   - Implement full RGB/BGR warping
   - Match grayscale warping quality

3. **Overlapping field block support** (line 128-129):
   - Per-field warping instead of per-block
   - Blending strategy for overlap zones
   - Coordinate-based detection on warped copies

### Potential Enhancements

1. **Adaptive triangulation**:
   - Grid-based in low-feature areas
   - Delaunay in high-feature areas

2. **Multi-resolution warping**:
   - Warp at low resolution (fast)
   - Refine at high resolution (accurate)

3. **GPU acceleration**:
   - WebGL shaders for triangle warping
   - 10-100× speedup potential

4. **Automatic point filtering**:
   - Detect and remove outliers (RANSAC-like)
   - Improve triangulation quality

5. **Caching**:
   - Cache triangulation for similar templates
   - Reuse affine matrices when possible

6. **Progressive rendering**:
   - Show partial results as triangles complete
   - Better UX for slow processing
