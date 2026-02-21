# Warp Strategies - Flows

**Module**: Domain / Preprocessing / Warping
**Python Reference**: `src/processors/image/warp_strategies.py`, `src/processors/image/WarpOnPointsCommon.py`
**Last Updated**: 2026-02-20

---

## Overview

Warp strategies perform perspective transformation to correct geometric distortion in scanned OMR sheets.

**Use Cases**:
- Fix perspective distortion from camera scans
- Align sheets based on reference points
- Correct non-planar scanning

---

## Warp Strategies

### 1. Perspective Transform (4-point)

**Algorithm**: `cv2.getPerspectiveTransform(src_points, dst_points)`

**Input**: 4 source points (corners on scanned image)
**Output**: Homography matrix for perspective correction

**Code**:
```python
M = cv2.getPerspectiveTransform(src_points, dst_points)
warped = cv2.warpPerspective(image, M, (width, height))
```

**Browser**:
```typescript
const srcPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [...]);
const dstPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [...]);
const M = cv.getPerspectiveTransform(srcPoints, dstPoints);
const warped = new cv.Mat();
cv.warpPerspective(image, warped, M, new cv.Size(width, height));
```

---

### 2. Affine Transform (3-point)

**Algorithm**: `cv2.getAffineTransform(src_points, dst_points)`

**Input**: 3 source points
**Output**: Affine transformation matrix

**Code**:
```python
M = cv2.getAffineTransform(src_points, dst_points)
warped = cv2.warpAffine(image, M, (width, height))
```

**Use Case**: Simpler than perspective, handles rotation + scaling + translation

---

### 3. Piecewise Affine (Delaunay Triangulation)

**Purpose**: Correct complex distortions with multiple control points

**Algorithm**:
1. Perform Delaunay triangulation on source points
2. For each triangle, compute affine transform
3. Apply transforms piecewise

**Code Reference**: `src/processors/alignment/piecewise_affine_delaunay.py`

**Use Case**: Non-planar scans with localized distortions

**Browser**: Requires custom implementation (no built-in OpenCV.js support)

---

### 4. Homography (Feature-based)

**Purpose**: Calculate transformation from matched feature points

**Algorithm**:
1. Detect features (SIFT, ORB)
2. Match features between source and reference
3. Estimate homography with RANSAC

**Code**:
```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
warped = cv2.warpPerspective(image, H, (width, height))
```

**Browser**:
```typescript
// Note: SIFT not available in OpenCV.js, use ORB
const orb = new cv.ORB();
const keypoints1 = new cv.KeyPointVector();
const descriptors1 = new cv.Mat();
orb.detectAndCompute(image1, mask, keypoints1, descriptors1);

// Match and find homography
const H = cv.findHomography(srcPts, dstPts, cv.RANSAC, 5.0);
const warped = new cv.Mat();
cv.warpPerspective(image, warped, H, new cv.Size(width, height));
```

---

## Configuration Example

```json
{
  "name": "WarpOnPoints",
  "options": {
    "warpMethod": "PERSPECTIVE_4",
    "sourcePoints": [[100, 100], [500, 100], [500, 400], [100, 400]],
    "destinationPoints": [[0, 0], [600, 0], [600, 500], [0, 500]]
  }
}
```

---

## Edge Cases

### 1. Collinear Points

**Issue**: Source or destination points are collinear
**Result**: Singular matrix, transformation fails
**Solution**: Ensure points form valid quadrilateral

### 2. Extreme Perspective

**Issue**: Source points define very skewed quadrilateral
**Result**: Warped image heavily distorted
**Solution**: Check point positions before transformation

### 3. Out-of-bounds Coordinates

**Issue**: Warped coordinates fall outside destination image
**Result**: Clipped or cropped output
**Solution**: Choose destination points to encompass full warped region

---

## Summary

**Warp Strategies**: Correct geometric distortion
**Types**: Perspective (4-point), Affine (3-point), Piecewise Affine, Homography
**Browser**: OpenCV.js supports perspective and affine transforms
**Use Case**: Fix camera distortion, align sheets
