# CropPage Processor - Flows

**Module**: Domain / Preprocessing / CropPage
**Python Reference**: `src/processors/image/CropPage.py`, `src/processors/image/page_detection.py`
**Last Updated**: 2026-02-20

---

## Overview

CropPage automatically detects the OMR sheet boundaries within a scanned image and crops to the page contour, removing background.

**Use Case**: Remove scanner bed background, extract only the OMR sheet

**Algorithm**: Contour detection + perspective correction

---

## Configuration

```json
{
  "name": "CropPage",
  "options": {
    "morphKernel": [10, 10],
    "dilate": 1,
    "erode": 1,
    "threshold": 200,
    "minAreaRatio": 0.5
  }
}
```

---

## Flow

### 1. Edge Detection
- Apply Canny edge detection
- Morphological operations (dilate/erode) to connect edges

### 2. Contour Detection
- Find all contours
- Filter by area (minAreaRatio * image_area)
- Select largest rectangular contour

### 3. Corner Detection
- Approximate contour to polygon
- Extract 4 corners

### 4. Perspective Transform
- Map corners to standard rectangle
- Warp image to remove perspective distortion

---

## Browser Implementation

```typescript
class CropPage extends ImageTemplatePreprocessor {
    getName(): string { return 'CropPage'; }

    async applyFilter(image: cv.Mat, coloredImage: cv.Mat, template: Template, filePath: string) {
        const edges = new cv.Mat();
        cv.Canny(image, edges, 50, 150);

        // Morphological operations
        const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(10, 10));
        cv.dilate(edges, edges, kernel);
        cv.erode(edges, edges, kernel);

        // Find contours
        const contours = new cv.MatVector();
        const hierarchy = new cv.Mat();
        cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        // Find largest contour
        let maxArea = 0;
        let maxContourIdx = -1;
        for (let i = 0; i < contours.size(); i++) {
            const area = cv.contourArea(contours.get(i));
            if (area > maxArea) {
                maxArea = area;
                maxContourIdx = i;
            }
        }

        if (maxContourIdx === -1) {
            throw new Error('No page contour detected');
        }

        // Approximate to rectangle
        const contour = contours.get(maxContourIdx);
        const approx = new cv.Mat();
        cv.approxPolyDP(contour, approx, 0.02 * cv.arcLength(contour, true), true);

        if (approx.rows !== 4) {
            throw new Error('Page contour is not rectangular');
        }

        // Extract 4 corners and apply perspective transform
        // ... (similar to CropOnMarkers)

        return { grayImage: cropped, coloredImage: croppedColored, template };
    }
}
```

---

## Summary

**CropPage**: Auto-detect page boundaries, remove background
**Algorithm**: Edge detection → Contours → Perspective transform
**Browser**: Use OpenCV.js Canny and findContours
