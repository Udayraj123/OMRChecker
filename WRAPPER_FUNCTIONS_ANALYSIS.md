# Wrapper Functions Analysis

Similar to `create_structuring_element` / `createStructuringElement`, here are other utility functions that are thin wrappers (5-10 lines) around existing library functions.

## Quick Summary

**Found 14 wrapper functions** that are simple wrappers around OpenCV/numpy/math functions:

- **9 OpenCV wrappers:** morphology, normalization, edge detection, padding, rotation, stacking
- **5 Math/Geometry wrappers:** distance, point operations, type conversion

**Common patterns:**
1. Parameter mapping (string → library constants)
2. Simple validation (None checks, edge cases)
3. Auto-parameter calculation
4. Tuple unpacking for coordinates
5. Fixed-parameter wrappers

---

## Image Processing Wrappers (OpenCV)

### 1. `create_structuring_element` (dot_line_detection.py:367-390)
**Wraps:** `cv2.getStructuringElement`
**Lines:** ~24 lines (with docstring)
**Logic:** Maps string shape names to OpenCV constants
```python
def create_structuring_element(shape: str, size: Tuple[int, int]) -> np.ndarray:
    shape_map = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}
    if shape not in shape_map:
        raise ValueError(...)
    return cv2.getStructuringElement(shape_map[shape], size)
```

### 2. `normalize_single` (image.py:174-178)
**Wraps:** `cv2.normalize`
**Lines:** 5 lines
**Logic:** Simple None check and edge case handling
```python
def normalize_single(image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX):
    if image is None or image.max() == image.min():
        return image
    return cv2.normalize(image, None, alpha, beta, norm_type)
```

### 3. `auto_canny` (image.py:190-197)
**Wraps:** `cv2.Canny`
**Lines:** 8 lines
**Logic:** Automatic threshold calculation from median
```python
def auto_canny(image, sigma=0.93):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)
```

### 4. `adjust_gamma` (image.py:202-211)
**Wraps:** `cv2.LUT`
**Lines:** 10 lines
**Logic:** Builds lookup table for gamma correction
```python
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
```

### 5. `pad_image_to_height` (image.py:392-401)
**Wraps:** `cv2.copyMakeBorder`
**Lines:** 10 lines
**Logic:** Calculates padding and calls copyMakeBorder with fixed parameters
```python
def pad_image_to_height(image, max_height, value=CLR_WHITE):
    return cv2.copyMakeBorder(
        image, 0, max_height - image.shape[0], 0, 0,
        cv2.BORDER_CONSTANT, value
    )
```

### 6. `pad_image_to_width` (image.py:404-414)
**Wraps:** `cv2.copyMakeBorder`
**Lines:** 11 lines
**Logic:** Similar to pad_image_to_height but for width
```python
def pad_image_to_width(image, max_width, value=CLR_WHITE):
    return cv2.copyMakeBorder(
        image, 0, 0, 0, max_width - image.shape[1],
        cv2.BORDER_CONSTANT, value
    )
```

### 7. `rotate` (image.py:447-452)
**Wraps:** `cv2.rotate`
**Lines:** 6 lines
**Logic:** Optional shape preservation
```python
def rotate(image, rotation, keep_original_shape=False):
    if keep_original_shape:
        image_shape = image.shape[0:2]
        image = cv2.rotate(image, rotation)
        return ImageUtils.resize_to_shape(image_shape, image)
    return cv2.rotate(image, rotation)
```

### 8. `get_padded_hstack` (image.py:374-380)
**Wraps:** `np.hstack`
**Lines:** 7 lines
**Logic:** Pads images to same height before stacking
```python
def get_padded_hstack(hstack):
    max_height = max(image.shape[0] for image in hstack)
    padded_hstack = [ImageUtils.pad_image_to_height(image, max_height) for image in hstack]
    return np.hstack(padded_hstack)
```

### 9. `get_padded_vstack` (image.py:383-389)
**Wraps:** `np.vstack`
**Lines:** 7 lines
**Logic:** Pads images to same width before stacking
```python
def get_padded_vstack(vstack):
    max_width = max(image.shape[1] for image in vstack)
    padded_vstack = [ImageUtils.pad_image_to_width(image, max_width) for image in vstack]
    return np.vstack(padded_vstack)
```

## Math/Geometry Wrappers

### 10. `distance` (math.py:14-15)
**Wraps:** `math.hypot`
**Lines:** 2 lines
**Logic:** Tuple unpacking for point coordinates
```python
def distance(point1, point2):
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])
```

### 11. `add_points` (math.py:22-26)
**Wraps:** Simple addition
**Lines:** 5 lines
**Logic:** Tuple unpacking and addition
```python
def add_points(new_origin, point):
    return [
        new_origin[0] + point[0],
        new_origin[1] + point[1],
    ]
```

### 12. `subtract_points` (math.py:29-33)
**Wraps:** Simple subtraction
**Lines:** 5 lines
**Logic:** Tuple unpacking and subtraction
```python
def subtract_points(point, new_origin):
    return [
        point[0] - new_origin[0],
        point[1] - new_origin[1],
    ]
```

### 13. `get_tuple_points` (math.py:68-69)
**Wraps:** List comprehension
**Lines:** 2 lines
**Logic:** Converts points to tuples
```python
def get_tuple_points(points):
    return [(int(point[0]), int(point[1])) for point in points]
```

### 14. `get_point_on_line_by_ratio` (math.py:42-47)
**Wraps:** Linear interpolation
**Lines:** 6 lines
**Logic:** Simple linear interpolation calculation
```python
def get_point_on_line_by_ratio(edge_line, length_ratio):
    start, end = edge_line
    return [
        start[0] + (end[0] - start[0]) * length_ratio,
        start[1] + (end[1] - start[1]) * length_ratio,
    ]
```

## Summary

**Total found:** 14 wrapper functions

**Categories:**
- **OpenCV wrappers:** 9 functions (morphology, normalization, edge detection, padding, rotation, stacking)
- **Math/Geometry wrappers:** 5 functions (distance, point operations, type conversion)

**Common patterns:**
1. **Parameter mapping:** String/enum to library constants (`create_structuring_element`)
2. **Simple validation:** None checks or edge cases (`normalize_single`)
3. **Parameter calculation:** Auto-compute parameters from input (`auto_canny`, `adjust_gamma`)
4. **Tuple unpacking:** Extract coordinates from point tuples (`distance`, `add_points`, `subtract_points`)
5. **Fixed parameter wrappers:** Call library function with fixed parameters (`pad_image_to_height`, `pad_image_to_width`)

**Recommendation:** Consider inlining these simple wrappers or consolidating them into more substantial utility functions that provide real value beyond just parameter mapping.

