# Image Utils Flows

## Overview

The Image Utils module (`src/utils/image.py` and `src/utils/image_warp.py`) provides comprehensive image processing utilities built on OpenCV, including reading, saving, resizing, warping, normalization, and various transformations.

## Core Components

### ImageUtils Class
Static-only utility class providing:
- Image loading and saving
- Resizing and resolution handling
- Normalization and enhancement
- Geometric transformations
- Contour operations
- Grid layout utilities
- Format conversions

### ImageWarpUtils Class
Specialized utilities for:
- Triangle-based warping
- Piecewise affine transformations
- In-place image warping

---

## 1. Image Reading Flow

### 1.1 Basic Image Loading

**Function**: `load_image(file_path, mode)`

```python
# Python
image = ImageUtils.load_image(file_path, cv2.IMREAD_GRAYSCALE)
```

**Flow**:
1. Convert Path to string
2. Call `cv2.imread(str(file_path), mode)`
3. Validate image is not None
4. Return loaded image as MatLike (numpy array)

**Modes**:
- `cv2.IMREAD_GRAYSCALE` - Load as grayscale (default)
- `cv2.IMREAD_COLOR` - Load as BGR color
- `cv2.IMREAD_UNCHANGED` - Load with alpha channel if present

**Error Handling**:
- Raises `ImageReadError` if OpenCV returns None
- Error message includes mode name for debugging

```typescript
// Browser equivalent
async function loadImage(file: File, mode: 'grayscale' | 'color' = 'grayscale'): Promise<cv.Mat> {
  const image = await cv.imread(file);

  if (!image || image.empty()) {
    throw new ImageReadError(file.name, `Failed to load image in ${mode} mode`);
  }

  if (mode === 'grayscale' && image.channels() > 1) {
    const gray = new cv.Mat();
    cv.cvtColor(image, gray, cv.COLOR_RGBA2GRAY);
    image.delete();
    return gray;
  }

  return image;
}
```

### 1.2 Smart Image Reading

**Function**: `read_image_util(file_path, tuning_config)`

```python
# Python
gray_image, colored_image = ImageUtils.read_image_util(file_path, tuning_config)
```

**Flow**:
1. Print file checksum (MD5) for debugging
2. Check if `tuning_config.outputs.colored_outputs_enabled`
3. **If colored outputs enabled**:
   - Load image as BGR color
   - Convert BGR to grayscale using `cv2.cvtColor`
   - Return both grayscale and colored versions
4. **If colored outputs disabled**:
   - Load only grayscale image
   - Return grayscale and None for colored

**Purpose**: Memory optimization - only load color image if needed for debug visualization

```typescript
// Browser equivalent
async function readImageUtil(
  file: File,
  tuningConfig: TuningConfig
): Promise<{ gray: cv.Mat; colored: cv.Mat | null }> {
  printFileChecksum(file, 'md5');

  if (tuningConfig.outputs.coloredOutputsEnabled) {
    const colored = await loadImage(file, 'color');
    const gray = new cv.Mat();
    cv.cvtColor(colored, gray, cv.COLOR_BGR2GRAY);
    return { gray, colored };
  } else {
    const gray = await loadImage(file, 'grayscale');
    return { gray, colored: null };
  }
}
```

---

## 2. Image Saving Flow

### 2.1 Basic Save

**Function**: `save_img(path, final_marked)`

```python
# Python
ImageUtils.save_img(path, final_marked)
```

**Flow**:
1. Call `cv2.imwrite(path, final_marked)`
2. OpenCV automatically handles format based on extension

**Supported Formats**: `.jpg`, `.png`, `.bmp`, `.tiff`, etc.

```typescript
// Browser equivalent
function saveImg(filename: string, image: cv.Mat): void {
  // Encode to desired format
  const encoded = cv.imencode('.png', image);

  // Create blob and download
  const blob = new Blob([encoded], { type: 'image/png' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();

  URL.revokeObjectURL(url);
  encoded.delete();
}
```

### 2.2 Save with Logging

**Function**: `save_marked_image(save_marked_dir, file_id, final_marked)`

```python
# Python
ImageUtils.save_marked_image(save_marked_dir, file_id, final_marked)
```

**Flow**:
1. Construct path: `save_marked_dir.joinpath(file_id)`
2. Convert to string
3. Log: "Saving Image to '{path}'"
4. Call `save_img(path, final_marked)`

```typescript
// Browser equivalent
function saveMarkedImage(
  saveMarkedDir: string,
  fileId: string,
  finalMarked: cv.Mat
): void {
  const imagePath = `${saveMarkedDir}/${fileId}`;
  logger.info(`Saving Image to '${imagePath}'`);
  saveImg(imagePath, finalMarked);
}
```

---

## 3. Resizing Flow

### 3.1 Resize Architecture

Four resize methods with different input types:
- `resize_to_shape(image_shape, *images)` - Takes `(h, w)` tuple
- `resize_to_dimensions(image_dimensions, *images)` - Takes `(w, h)` tuple
- `resize_multiple(images, u_width, u_height)` - Universal method
- `resize_single(image, u_width, u_height)` - Single image

### 3.2 Resize to Shape

**Function**: `resize_to_shape(image_shape, *images)`

```python
# Python
resized = ImageUtils.resize_to_shape((480, 640), image1, image2)
```

**Flow**:
1. Extract `h, w = image_shape`
2. Call `resize_multiple(images, w, h)`
3. Return resized images

### 3.3 Resize to Dimensions

**Function**: `resize_to_dimensions(image_dimensions, *images)`

```python
# Python
resized = ImageUtils.resize_to_dimensions((640, 480), image1, image2)
```

**Flow**:
1. Extract `w, h = image_dimensions`
2. Call `resize_multiple(images, w, h)`
3. Return resized images

**Note**: Dimensions are (width, height), shape is (height, width)

### 3.4 Resize Multiple Images

**Function**: `resize_multiple(images, u_width, u_height)`

```python
# Python
resized = ImageUtils.resize_multiple([img1, img2], u_width=800, u_height=600)
```

**Flow**:
1. If only one image, call `resize_single` and return directly
2. If multiple images, apply `resize_single` to each
3. Return list of resized images

### 3.5 Resize Single Image

**Function**: `resize_single(image, u_width, u_height)`

**Flow**:
```
1. Check if image is None → return None
2. Get current dimensions: h, w = image.shape[:2]
3. Calculate missing dimension:
   - If u_height is None: u_height = int(h * u_width / w)
   - If u_width is None: u_width = int(w * u_height / h)
4. Check if resize needed: u_height == h AND u_width == w
   - If no change needed → return original image
5. Call cv2.resize(image, (int(u_width), int(u_height)))
6. Return resized image
```

**Aspect Ratio Preservation**:
- If only width provided, height is calculated to maintain aspect ratio
- If only height provided, width is calculated to maintain aspect ratio
- If both provided, aspect ratio may change

```python
# Examples
# Maintain aspect ratio (width only)
resized = ImageUtils.resize_single(image, u_width=800)  # height auto-calculated

# Maintain aspect ratio (height only)
resized = ImageUtils.resize_single(image, u_height=600)  # width auto-calculated

# Force dimensions (may distort)
resized = ImageUtils.resize_single(image, u_width=800, u_height=600)
```

```typescript
// Browser equivalent
function resizeSingle(
  image: cv.Mat | null,
  uWidth?: number,
  uHeight?: number
): cv.Mat | null {
  if (!image) return null;

  const [h, w] = [image.rows, image.cols];

  // Calculate missing dimension
  if (uHeight === undefined) {
    uHeight = Math.floor(h * uWidth! / w);
  }
  if (uWidth === undefined) {
    uWidth = Math.floor(w * uHeight! / h);
  }

  // No resize needed
  if (uHeight === h && uWidth === w) {
    return image;
  }

  const resized = new cv.Mat();
  cv.resize(image, resized, new cv.Size(uWidth, uHeight));
  return resized;
}
```

---

## 4. Normalization Flow

### 4.1 Normalize Single

**Function**: `normalize_single(image, alpha, beta, norm_type)`

```python
# Python
normalized = ImageUtils.normalize_single(image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
```

**Flow**:
```
1. Check if image is None → return image
2. Check if image.max() == image.min() → return image (no contrast)
3. Call cv2.normalize(image, None, alpha, beta, norm_type)
4. Return normalized image
```

**Normalization Types**:
- `cv2.NORM_MINMAX` - Scale to [alpha, beta] range (default)
- `cv2.NORM_L1` - L1 normalization
- `cv2.NORM_L2` - L2 normalization

**Use Cases**:
- Enhance contrast before processing
- Standardize intensity ranges
- Prepare for machine learning models

```typescript
// Browser equivalent
function normalizeSingle(
  image: cv.Mat | null,
  alpha = 0,
  beta = 255,
  normType = cv.NORM_MINMAX
): cv.Mat | null {
  if (!image) return image;

  const minMax = cv.minMaxLoc(image);
  if (minMax.maxVal === minMax.minVal) {
    return image; // No contrast
  }

  const normalized = new cv.Mat();
  cv.normalize(image, normalized, alpha, beta, normType);
  return normalized;
}
```

### 4.2 Normalize Multiple

**Function**: `normalize(*images, alpha, beta, norm_type)`

```python
# Python
img1_norm, img2_norm = ImageUtils.normalize(img1, img2, alpha=0, beta=255)
```

**Flow**:
1. If single image, call `normalize_single`
2. If multiple images, apply `normalize_single` to each
3. Return normalized images

---

## 5. Edge Detection Flow

### 5.1 Auto Canny

**Function**: `auto_canny(image, sigma)`

```python
# Python
edges = ImageUtils.auto_canny(image, sigma=0.93)
```

**Flow**:
```
1. Compute median pixel intensity: v = np.median(image)
2. Calculate lower threshold: lower = max(0, (1.0 - sigma) * v)
3. Calculate upper threshold: upper = min(255, (1.0 + sigma) * v)
4. Apply Canny: cv2.Canny(image, lower, upper)
5. Return edge image
```

**Sigma Parameter**:
- Default: 0.93
- Higher sigma → wider threshold range → more edges detected
- Lower sigma → narrower range → fewer edges detected

**Automatic Thresholding**:
- Adapts to image intensity distribution
- No manual threshold tuning needed

```typescript
// Browser equivalent
function autoCanny(image: cv.Mat, sigma = 0.93): cv.Mat {
  // Compute median (OpenCV.js doesn't have median, use approximation)
  const flattened = image.data;
  const sorted = Array.from(flattened).sort((a, b) => a - b);
  const v = sorted[Math.floor(sorted.length / 2)];

  const lower = Math.max(0, Math.floor((1.0 - sigma) * v));
  const upper = Math.min(255, Math.floor((1.0 + sigma) * v));

  const edges = new cv.Mat();
  cv.Canny(image, edges, lower, upper);
  return edges;
}
```

---

## 6. Gamma Adjustment Flow

### 6.1 Adjust Gamma

**Function**: `adjust_gamma(image, gamma)`

```python
# Python
adjusted = ImageUtils.adjust_gamma(image, gamma=1.5)
```

**Flow**:
```
1. Calculate inverse gamma: inv_gamma = 1.0 / gamma
2. Build lookup table:
   - For each intensity i in [0, 255]:
     - Normalize: i / 255.0
     - Apply power: (i / 255.0) ** inv_gamma
     - Scale back: * 255
3. Apply lookup table: cv2.LUT(image, table)
4. Return adjusted image
```

**Gamma Values**:
- `gamma < 1.0` → Brighten dark regions (expand shadows)
- `gamma = 1.0` → No change
- `gamma > 1.0` → Darken bright regions (compress highlights)

**Use Cases**:
- Correct exposure
- Enhance visibility in shadows
- Prepare for threshold-based detection

```typescript
// Browser equivalent
function adjustGamma(image: cv.Mat, gamma = 1.0): cv.Mat {
  const invGamma = 1.0 / gamma;

  // Build lookup table
  const table = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    table[i] = Math.floor(Math.pow(i / 255.0, invGamma) * 255);
  }

  // Apply LUT
  const adjusted = new cv.Mat();
  const lut = cv.matFromArray(1, 256, cv.CV_8UC1, Array.from(table));
  cv.LUT(image, lut, adjusted);
  lut.delete();

  return adjusted;
}
```

---

## 7. Warping Utilities

### 7.1 Get Cropped Warped Rectangle Points

**Function**: `get_cropped_warped_rectangle_points(ordered_page_corners)`

```python
# Python
warped_points, (max_width, max_height) = ImageUtils.get_cropped_warped_rectangle_points(corners)
```

**Flow**:
```
1. Unpack corners: (tl, tr, br, bl) = ordered_page_corners
2. Calculate edge lengths:
   - length_t = distance(tr, tl)  # top edge
   - length_b = distance(br, bl)  # bottom edge
   - length_r = distance(tr, br)  # right edge
   - length_l = distance(tl, bl)  # left edge
3. Calculate output dimensions:
   - max_width = max(length_t, length_b)
   - max_height = max(length_r, length_l)
4. Create destination points:
   - [0, 0]                        # top-left
   - [max_width - 1, 0]            # top-right
   - [max_width - 1, max_height - 1]  # bottom-right
   - [0, max_height - 1]           # bottom-left
5. Return (warped_points, warped_box_dimensions)
```

**Purpose**: Calculate ideal output dimensions for perspective transform to maintain realistic aspect ratio

**Note**: Comment indicates this is less critical since images are resized anyway

### 7.2 Warp Triangle In-Place

**Function**: `warp_triangle_inplace(image, warped_image, source_triangle, warped_triangle, show_image_level)`

**Purpose**: Apply piecewise affine transformation using triangular patches

**Flow**:
```
1. Validate triangles (check for collinear points)
   - If collinear in source → skip and log critical warning
   - If collinear in warped → skip and log critical warning

2. Calculate bounding boxes:
   - source_box = get_bounding_box_of_points(source_triangle)
   - warped_box = get_bounding_box_of_points(warped_triangle)

3. Shift triangles to origin:
   - source_shifted = shift_points_to_origin(source_tl, source_triangle)
   - warped_shifted = shift_points_to_origin(warped_tl, warped_triangle)

4. Compute affine transform matrix:
   - triangle_affine_matrix = cv2.getAffineTransform(
       np.float32(source_shifted),
       np.float32(warped_shifted)
     )

5. Crop source image:
   - source_triangle_box = image[source_tl[1]:source_br[1], source_tl[0]:source_br[0]]

6. Apply warp:
   - warped_triangle_box = cv2.warpAffine(
       source_triangle_box,
       triangle_affine_matrix,
       warped_box_dimensions,
       flags=cv2.INTER_LINEAR,
       borderMode=cv2.BORDER_REFLECT_101
     )

7. Validate warped dimensions match expected dimensions

8. Replace triangle in output image (in-place):
   - Call replace_triangle_inplace()

9. If show_image_level >= 5:
   - Display debug visualization
```

**Collinear Point Handling**:
- Three points in a line cannot define a triangle
- Affine transform requires non-collinear points
- Critical error logged, transformation skipped

```typescript
// Browser equivalent
function warpTriangleInplace(
  image: cv.Mat,
  warpedImage: cv.Mat,
  sourceTriangle: Point[],
  warpedTriangle: Point[],
  showImageLevel = 0
): void {
  // Validate non-collinear
  if (checkCollinearPoints(...sourceTriangle)) {
    logger.critical(`Collinear source points: ${sourceTriangle}`);
    return;
  }
  if (checkCollinearPoints(...warpedTriangle)) {
    logger.critical(`Collinear warped points: ${warpedTriangle}`);
    return;
  }

  // Get bounding boxes
  const [sourceBounds, _sourceDims] = getBoundingBoxOfPoints(sourceTriangle);
  const [warpedBounds, warpedDims] = getBoundingBoxOfPoints(warpedTriangle);

  // Shift to origin
  const sourceShifted = shiftPointsToOrigin(sourceBounds.tl, sourceTriangle);
  const warpedShifted = shiftPointsToOrigin(warpedBounds.tl, warpedTriangle);

  // Get affine matrix
  const srcMat = cv.matFromArray(3, 2, cv.CV_32F, sourceShifted.flat());
  const dstMat = cv.matFromArray(3, 2, cv.CV_32F, warpedShifted.flat());
  const affineMatrix = cv.getAffineTransform(srcMat, dstMat);

  // Crop and warp
  const sourceBox = image.roi(new cv.Rect(
    sourceBounds.tl.x, sourceBounds.tl.y,
    sourceBounds.br.x - sourceBounds.tl.x,
    sourceBounds.br.y - sourceBounds.tl.y
  ));

  const warpedBox = new cv.Mat();
  cv.warpAffine(
    sourceBox,
    warpedBox,
    affineMatrix,
    new cv.Size(warpedDims[0], warpedDims[1]),
    cv.INTER_LINEAR,
    cv.BORDER_REFLECT_101
  );

  // Replace in output
  replaceTriangleInplace(warpedImage, warpedShifted, warpedBox, warpedBounds, warpedDims);

  // Cleanup
  srcMat.delete();
  dstMat.delete();
  affineMatrix.delete();
  sourceBox.delete();
  warpedBox.delete();
}
```

### 7.3 Replace Triangle In-Place

**Function**: `replace_triangle_inplace(source_image, shifted_triangle, warped_triangle_box, warped_tl_br, warped_box_dimensions)`

**Purpose**: Blend warped triangle into destination image using masks

**Flow**:
```
1. Extract dimensions: tl, br = warped_tl_br; dest_w, dest_h = warped_box_dimensions
2. Determine number of channels (grayscale or colored)
3. Create white triangle mask:
   - Create zero array of size (dest_h, dest_w, channels)
   - Fill shifted_triangle with 1.0 using cv2.fillConvexPoly
4. Create black triangle mask:
   - black_triangle = (1.0, 1.0, 1.0) - white_triangle  (or 1.0 - white_triangle for grayscale)
5. Extract triangle region from warped image:
   - triangle_from_warped_image = warped_triangle_box * white_triangle
6. Extract background region from destination:
   - source_triangle_box = source_image[tl[1]:br[1], tl[0]:br[0]]
   - background_from_source_image = source_triangle_box * black_triangle
7. Composite the triangle:
   - source_image[tl[1]:br[1], tl[0]:br[0]] = background + triangle
8. Return debug images: (background, triangle_from_source, triangle_from_warped)
```

**Mask Strategy**:
- White mask (1.0) selects triangle region
- Black mask (0.0) selects everything outside triangle
- Multiply to extract regions
- Add to composite final image

---

## 8. Contour Utilities

### 8.1 Grab Contours

**Function**: `grab_contours(cnts)`

**Purpose**: Handle OpenCV version differences in findContours return signature

```python
# Python
contours = ImageUtils.grab_contours(cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
```

**Flow**:
```
1. Check length of contours tuple:
   - len(cnts) == 2 → OpenCV v2.4, v4-beta, v4-official
     - Return cnts[0]
   - len(cnts) == 3 → OpenCV v3, v4-pre, v4-alpha
     - Return cnts[1]
   - Otherwise → Raise ImageProcessingError
2. Return actual contours array
```

**OpenCV Version Compatibility**:
- OpenCV 2.4: `findContours()` returns `(contours, hierarchy)`
- OpenCV 3.x: `findContours()` returns `(image, contours, hierarchy)`
- OpenCV 4.x: Returns to `(contours, hierarchy)`

```typescript
// Browser equivalent (OpenCV.js is consistent)
function grabContours(findContoursResult: any): cv.MatVector {
  // OpenCV.js always returns { contours, hierarchy }
  return findContoursResult.contours;
}
```

### 8.2 Get Control Destination Points from Contour

**Function**: `get_control_destination_points_from_contour(source_contour, warped_line, max_points)`

**Purpose**: Map contour points to straightened line for piecewise warping

**Flow**:
```
1. Validate max_points >= 2
2. Extract line endpoints: start, end = warped_line
3. Calculate total contour arc length:
   - Iterate through contour points
   - Sum distances between consecutive points
4. Initialize with first point:
   - control_points = [source_contour[0]]
   - warped_points = [start]
5. Iterate through remaining contour points:
   - Calculate edge_length to previous point
   - Accumulate current_arc_length += edge_length
   - Calculate length_ratio = current_arc_length / contour_length
   - Find corresponding point on warped_line:
     - warped_point = get_point_on_line_by_ratio(warped_line, length_ratio)
   - Append to control_points and warped_points
6. Validate:
   - Check len(warped_points) <= max_points
   - Check last warped_point is close to end (within 2% error)
7. Return (control_points, warped_points)
```

**Use Case**: Straighten curved contours (e.g., fixing warped page edges)

**Arc Length Mapping**:
- Each point on source contour is mapped to warped line
- Position on line is proportional to arc length along contour
- Preserves relative spacing of points

### 8.3 Split Patch Contour on Corners

**Function**: `split_patch_contour_on_corners(patch_corners, source_contour)`

**Purpose**: Divide contour into 4 edge segments based on corner points

**Flow**:
```
1. Order corners: ordered_patch_corners = order_four_points(patch_corners)
2. Create LineString objects for each edge:
   - TOP: [corner[0], corner[1]]
   - RIGHT: [corner[1], corner[2]]
   - BOTTOM: [corner[2], corner[3]]
   - LEFT: [corner[3], corner[0]]
3. Initialize edge_contours_map: {TOP: [], RIGHT: [], BOTTOM: [], LEFT: []}
4. For each boundary_point in source_contour:
   - Calculate distance to each edge line
   - Find nearest edge (min distance)
   - Append point to that edge's contour list
5. For each edge:
   - Check if any points were assigned (warn if empty)
   - Reverse order if needed (ensure clockwise)
   - Insert start corner at beginning
   - Append end corner at end
6. Return (ordered_patch_corners, edge_contours_map)
```

**Use Case**: Prepare contours for edge-wise warping (e.g., straightening page edges)

**Edge Assignment**:
- Each contour point is assigned to nearest edge
- Uses Shapely's LineString.distance() for geometric distance
- Warns if edge has no assigned points (potential issue)

---

## 9. Image Layout & Padding

### 9.1 Grid Layout

**Function**: `get_vstack_image_grid(debug_vstack)`

**Purpose**: Stack multiple rows of images into a grid

```python
# Python
grid = ImageUtils.get_vstack_image_grid([
    [img1, img2, img3],  # row 1
    [img4, img5],        # row 2
    [img6, img7, img8, img9]  # row 3
])
```

**Flow**:
```
1. For each row in debug_vstack:
   - Call get_padded_hstack(row) to create horizontal stack
2. Call get_padded_vstack(padded_rows) to stack rows vertically
3. Return final grid image
```

### 9.2 Horizontal Stack with Padding

**Function**: `get_padded_hstack(hstack)`

```python
# Python
row_image = ImageUtils.get_padded_hstack([img1, img2, img3])
```

**Flow**:
```
1. Find maximum height: max_height = max(image.shape[0] for image in hstack)
2. Pad each image to max_height:
   - Call pad_image_to_height(image, max_height)
3. Stack horizontally: np.hstack(padded_images)
4. Return stacked image
```

### 9.3 Vertical Stack with Padding

**Function**: `get_padded_vstack(vstack)`

```python
# Python
grid_image = ImageUtils.get_padded_vstack([row1, row2, row3])
```

**Flow**:
```
1. Find maximum width: max_width = max(image.shape[1] for image in vstack)
2. Pad each image to max_width:
   - Call pad_image_to_width(image, max_width)
3. Stack vertically: np.vstack(padded_images)
4. Return stacked image
```

### 9.4 Pad to Height

**Function**: `pad_image_to_height(image, max_height, value)`

```python
# Python
padded = ImageUtils.pad_image_to_height(image, 600, value=CLR_WHITE)
```

**Flow**:
```
1. Calculate padding needed: padding = max_height - image.shape[0]
2. Call cv2.copyMakeBorder:
   - top=0, bottom=padding, left=0, right=0
   - borderType=cv2.BORDER_CONSTANT
   - value=CLR_WHITE (default)
3. Return padded image
```

**Purpose**: Pad bottom of image to reach target height

### 9.5 Pad to Width

**Function**: `pad_image_to_width(image, max_width, value)`

```python
# Python
padded = ImageUtils.pad_image_to_width(image, 800, value=CLR_WHITE)
```

**Flow**:
```
1. Calculate padding needed: padding = max_width - image.shape[1]
2. Call cv2.copyMakeBorder:
   - top=0, bottom=0, left=0, right=padding
   - borderType=cv2.BORDER_CONSTANT
   - value=CLR_WHITE (default)
3. Return padded image
```

**Purpose**: Pad right side of image to reach target width

### 9.6 Pad from Center

**Function**: `pad_image_from_center(image, padding_width, padding_height, value)`

```python
# Python
padded_img, pad_range = ImageUtils.pad_image_from_center(
    image,
    padding_width=50,
    padding_height=30,
    value=255
)
```

**Flow**:
```
1. Get input dimensions: input_height, input_width = image.shape[:2]
2. Calculate pad_range:
   - [padding_height, padding_height + input_height, padding_width, padding_width + input_width]
3. Create white canvas:
   - size = (padding_height * 2 + input_height, padding_width * 2 + input_width)
   - white_image = value * np.ones(size, np.uint8)
4. Place original image in center:
   - white_image[pad_range[0]:pad_range[1], pad_range[2]:pad_range[3]] = image
5. Return (white_image, pad_range)
```

**Purpose**: Add equal padding on all sides (useful for edge detection preprocessing)

**Return Values**:
- `white_image`: Padded result
- `pad_range`: Region where original image was placed

```typescript
// Browser equivalent
function padImageFromCenter(
  image: cv.Mat,
  paddingWidth: number,
  paddingHeight = 0,
  value = 255
): { padded: cv.Mat; padRange: number[] } {
  const [h, w] = [image.rows, image.cols];
  const padRange = [
    paddingHeight,
    paddingHeight + h,
    paddingWidth,
    paddingWidth + w
  ];

  // Create white canvas
  const padded = new cv.Mat(
    paddingHeight * 2 + h,
    paddingWidth * 2 + w,
    image.type(),
    new cv.Scalar(value)
  );

  // Copy original to center
  const roi = padded.roi(new cv.Rect(
    paddingWidth, paddingHeight, w, h
  ));
  image.copyTo(roi);
  roi.delete();

  return { padded, padRange };
}
```

---

## 10. Geometric Utilities

### 10.1 Clip Zone to Image Bounds

**Function**: `clip_zone_to_image_bounds(rectangle, image)`

```python
# Python
clipped_rect = ImageUtils.clip_zone_to_image_bounds(
    [[x1, y1], [x2, y2]],
    image
)
```

**Flow**:
```
1. Get image dimensions: h, w = image.shape[:2]
2. Unpack rectangle: zone_start, zone_end = rectangle
3. Clip to top-left (0, 0):
   - zone_start = [max(0, zone_start[0]), max(0, zone_start[1])]
   - zone_end = [max(0, zone_end[0]), max(0, zone_end[1])]
4. Clip to bottom-right (w, h):
   - zone_start = [min(w, zone_start[0]), min(h, zone_start[1])]
   - zone_end = [min(w, zone_end[0]), min(h, zone_end[1])]
5. Return [zone_start, zone_end]
```

**Purpose**: Ensure rectangle coordinates stay within image bounds (prevent out-of-bounds access)

### 10.2 Rotate Image

**Function**: `rotate(image, rotation, keep_original_shape)`

```python
# Python
# Standard rotation (image size may change)
rotated = ImageUtils.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Rotation preserving original size
rotated = ImageUtils.rotate(image, cv2.ROTATE_90_CLOCKWISE, keep_original_shape=True)
```

**Flow**:
```
1. If keep_original_shape:
   - Store original shape: image_shape = image.shape[0:2]
   - Apply rotation: image = cv2.rotate(image, rotation)
   - Resize back to original: resize_to_shape(image_shape, image)
2. Else:
   - Apply rotation: cv2.rotate(image, rotation)
3. Return rotated image
```

**Rotation Values**:
- `cv2.ROTATE_90_CLOCKWISE`
- `cv2.ROTATE_180`
- `cv2.ROTATE_90_COUNTERCLOCKWISE`

**Keep Original Shape**:
- When `True`: Output maintains input dimensions (may crop/pad)
- When `False`: Output dimensions adjust to rotated size

```typescript
// Browser equivalent
function rotate(
  image: cv.Mat,
  rotation: number,
  keepOriginalShape = false
): cv.Mat {
  const rotated = new cv.Mat();
  cv.rotate(image, rotated, rotation);

  if (keepOriginalShape) {
    const resized = new cv.Mat();
    cv.resize(rotated, resized, new cv.Size(image.cols, image.rows));
    rotated.delete();
    return resized;
  }

  return rotated;
}
```

### 10.3 Overlay Image

**Function**: `overlay_image(image1, image2, transparency)`

```python
# Python
overlay = ImageUtils.overlay_image(image1, image2, transparency=0.5)
```

**Flow**:
```
1. Create copy: overlay = image1.copy()
2. Blend images: cv2.addWeighted(
     overlay,              # src1
     transparency,         # alpha
     image2,              # src2
     1 - transparency,    # beta
     0,                   # gamma
     overlay              # dst
   )
3. Return overlay
```

**Transparency**:
- 0.0 → Only image2 visible
- 0.5 → Equal blend (default)
- 1.0 → Only image1 visible

**Formula**: `output = image1 * transparency + image2 * (1 - transparency)`

```typescript
// Browser equivalent
function overlayImage(
  image1: cv.Mat,
  image2: cv.Mat,
  transparency = 0.5
): cv.Mat {
  const overlay = image1.clone();
  cv.addWeighted(
    overlay,
    transparency,
    image2,
    1 - transparency,
    0,
    overlay
  );
  return overlay;
}
```

---

## 11. CLAHE Enhancement

**Global Helper**: `CLAHE_HELPER = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))`

**Purpose**: Contrast Limited Adaptive Histogram Equalization - improve local contrast

**Configuration**:
- `clipLimit=5.0` - Threshold for contrast limiting (prevents over-enhancement)
- `tileGridSize=(8, 8)` - Size of grid for local histogram equalization

**Usage** (not a method, but available for import):
```python
# Python
from src.utils.image import CLAHE_HELPER
enhanced = CLAHE_HELPER.apply(image)
```

**Use Cases**:
- Enhance bubble visibility in low-contrast regions
- Improve OCR accuracy
- Prepare images for threshold-based detection

```typescript
// Browser equivalent
const claheHelper = new cv.CLAHE();
claheHelper.setClipLimit(5.0);
claheHelper.setTilesGridSize(new cv.Size(8, 8));

// Usage
const enhanced = new cv.Mat();
claheHelper.apply(image, enhanced);
```

---

## Summary of Key Flows

1. **Image Reading**: Load with error handling, optional color/grayscale modes
2. **Image Saving**: Write with format auto-detection from extension
3. **Resizing**: Flexible API with aspect ratio preservation
4. **Normalization**: Contrast enhancement with min-max scaling
5. **Edge Detection**: Auto Canny with adaptive thresholds
6. **Gamma Adjustment**: Exposure correction via lookup table
7. **Warping**: Triangle-based piecewise affine transformation
8. **Contour Utilities**: Version-safe contour extraction, edge splitting
9. **Layout**: Grid creation with automatic padding
10. **Geometric**: Clipping, rotation, overlay blending
11. **Enhancement**: CLAHE for local contrast improvement

## Browser Migration Strategy

See `constraints.md` for detailed browser migration patterns, performance considerations, and File API integration.
