# CropOnMarkers Processor - Flows

**Module**: Domain / Preprocessing / CropOnMarkers
**Python Reference**: `src/processors/image/CropOnMarkers.py`, `src/processors/image/crop_on_patches/`
**Last Updated**: 2026-02-20

---

## Overview

CropOnMarkers detects alignment markers (dots, lines, or custom patterns) on OMR sheets and crops the image to a standard region based on marker positions.

**Use Case**: Normalize different scan positions/sizes to consistent template coordinates

**Marker Types**:
1. **FOUR_DOTS**: Four corner dots
2. **TWO_DOTS_ONE_LINE**: Two dots + one line
3. **TWO_LINES**: Two perpendicular lines
4. **CUSTOM**: User-defined marker pattern

---

## Configuration

```json
{
  "name": "CropOnMarkers",
  "options": {
    "type": "FOUR_DOTS",
    "referenceImage": "omr_marker.jpg",
    "dotMinRadius": 5,
    "dotMaxRadius": 30,
    "detectMethod": "SIMPLE",
    "margins": [50, 50, 50, 50],
    "morphKernel": [5, 5]
  }
}
```

**Options**:
- `type`: Marker type (FOUR_DOTS, TWO_DOTS_ONE_LINE, TWO_LINES, CUSTOM)
- `referenceImage`: Reference marker image
- `dotMinRadius`: Minimum dot radius (pixels)
- `dotMaxRadius`: Maximum dot radius (pixels)
- `detectMethod`: Detection method (SIMPLE, HOUGH_CIRCLES)
- `margins`: Crop margins [top, bottom, left, right]
- `morphKernel`: Morphological operation kernel size

---

## Flow: FOUR_DOTS

**Code Reference**: `src/processors/image/crop_on_patches/dot_line_detection.py`

### Step 1: Marker Detection

```python
# Detect four corner dots
dots = detect_dots(image, dot_min_radius, dot_max_radius, method='SIMPLE')

if len(dots) != 4:
    raise MarkerDetectionError(
        f"Expected 4 markers, found {len(dots)}",
        file_path=file_path
    )
```

**Algorithm** (SIMPLE):
1. Threshold image
2. Find contours
3. Filter by circularity and size
4. Select 4 largest circular contours

**Algorithm** (HOUGH_CIRCLES):
1. Gaussian blur
2. Hough circle transform
3. Filter by radius range

---

### Step 2: Corner Ordering

```python
# Order dots: top-left, top-right, bottom-right, bottom-left
ordered_dots = order_four_points(dots)
```

**Ordering Logic**:
1. Find centroid of all dots
2. Classify each dot by quadrant (TL, TR, BR, BL)
3. Return in standard order

---

### Step 3: Perspective Transform

```python
# Define source points (detected dots)
src_points = np.array(ordered_dots, dtype=np.float32)

# Define destination points (template coordinates)
dst_points = np.array([
    [margins[2], margins[0]],                    # Top-left
    [template_width - margins[3], margins[0]],   # Top-right
    [template_width - margins[3], template_height - margins[1]],  # Bottom-right
    [margins[2], template_height - margins[1]]   # Bottom-left
], dtype=np.float32)

# Calculate homography
M = cv2.getPerspectiveTransform(src_points, dst_points)

# Warp image
cropped_image = cv2.warpPerspective(image, M, (template_width, template_height))
```

---

### Step 4: Update Template Shifts

```python
# Calculate shifts from original to cropped coordinates
shifts = calculate_shifts(src_points, dst_points, margins)

# Update template field blocks
for field_block in template.field_blocks:
    field_block.shifts = shifts
```

---

## Flow: TWO_DOTS_ONE_LINE

**Use Case**: OMR sheets with 2 dots + 1 reference line

### Detection Steps

1. **Detect Dots**: Find 2 circular markers
2. **Detect Line**: Find 1 straight line using Hough line transform
3. **Calculate Transform**: Use dots + line to define coordinate system
4. **Warp Image**: Apply perspective transform

**Advantage**: More robust than 4 dots (fewer markers to detect)

---

## Flow: TWO_LINES

**Use Case**: OMR sheets with 2 perpendicular reference lines

### Detection Steps

1. **Detect Lines**: Find 2 perpendicular lines
2. **Find Intersection**: Calculate line intersection point
3. **Calculate Rotation**: Determine rotation from line angles
4. **Crop & Rotate**: Apply rotation and crop

**Advantage**: Very precise alignment (lines define exact axes)

---

## Flow: CUSTOM

**Use Case**: Non-standard marker patterns

### Configuration

```json
{
  "type": "CUSTOM",
  "options": {
    "customMarkers": [
      {"type": "dot", "position": [100, 100]},
      {"type": "line", "position": [200, 200], "angle": 0},
      {"type": "cross", "position": [300, 300]}
    ]
  }
}
```

### Detection Steps

1. **Load Custom Markers**: Parse marker definitions
2. **Detect Each Marker**: Use appropriate detector for each type
3. **Match to Template**: Map detected markers to template positions
4. **Calculate Transform**: Compute transformation matrix

---

## Edge Cases

### 1. Marker Not Detected

**Error**: `MarkerDetectionError`
**Message**: "Expected 4 markers, found {n}"

**Common Causes**:
- Poor image quality
- Markers obscured or damaged
- Wrong marker type configured
- Incorrect detection parameters

**Solution**: Adjust `dotMinRadius`, `dotMaxRadius`, or use different marker type

---

### 2. Ambiguous Marker Positions

**Scenario**: More than 4 circular objects detected

**Behavior**: Select 4 largest circles

**Risk**: May select wrong objects as markers

**Solution**: Tune `dotMinRadius` and `dotMaxRadius` more precisely

---

### 3. Perspective Distortion Too Large

**Scenario**: Markers detected but image heavily skewed

**Behavior**: Perspective transform may fail or produce distorted image

**Solution**: Improve scanning process or use more markers

---

## Browser Migration

```typescript
class CropOnMarkers extends ImageTemplatePreprocessor {
    private markerType: string;
    private options: any;

    getName(): string {
        return 'CropOnMarkers';
    }

    async applyFilter(
        image: cv.Mat,
        coloredImage: cv.Mat,
        template: Template,
        filePath: string
    ): Promise<{ grayImage: cv.Mat; coloredImage: cv.Mat; template: Template }> {
        let markers: any[];

        switch (this.markerType) {
            case 'FOUR_DOTS':
                markers = this.detectFourDots(image);
                break;
            case 'TWO_DOTS_ONE_LINE':
                markers = this.detectTwoDotsOneLine(image);
                break;
            case 'TWO_LINES':
                markers = this.detectTwoLines(image);
                break;
            case 'CUSTOM':
                markers = this.detectCustomMarkers(image);
                break;
            default:
                throw new Error(`Unknown marker type: ${this.markerType}`);
        }

        if (!markers || markers.length === 0) {
            throw new MarkerDetectionError(`Markers not detected for type ${this.markerType}`);
        }

        // Calculate perspective transform
        const { croppedGray, croppedColored, shifts } = this.applyPerspectiveTransform(
            image,
            coloredImage,
            markers,
            template
        );

        // Update template shifts
        template.templateLayout.fieldBlocks.forEach(fb => {
            fb.shifts = shifts;
        });

        return {
            grayImage: croppedGray,
            coloredImage: croppedColored,
            template
        };
    }

    private detectFourDots(image: cv.Mat): any[] {
        // Threshold
        const binary = new cv.Mat();
        cv.threshold(image, binary, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);

        // Find contours
        const contours = new cv.MatVector();
        const hierarchy = new cv.Mat();
        cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        // Filter circular contours
        const dots: any[] = [];
        for (let i = 0; i < contours.size(); i++) {
            const contour = contours.get(i);
            const area = cv.contourArea(contour);
            const perimeter = cv.arcLength(contour, true);
            const circularity = (4 * Math.PI * area) / (perimeter * perimeter);

            if (circularity > 0.7) {
                const moments = cv.moments(contour);
                const cx = moments.m10 / moments.m00;
                const cy = moments.m01 / moments.m00;
                dots.push({ x: cx, y: cy, area });
            }
        }

        // Sort by area and take 4 largest
        dots.sort((a, b) => b.area - a.area);
        const fourDots = dots.slice(0, 4);

        if (fourDots.length !== 4) {
            throw new MarkerDetectionError(`Expected 4 dots, found ${fourDots.length}`);
        }

        // Order dots (TL, TR, BR, BL)
        return this.orderFourPoints(fourDots);
    }

    private orderFourPoints(points: any[]): any[] {
        // Calculate centroid
        const cx = points.reduce((sum, p) => sum + p.x, 0) / points.length;
        const cy = points.reduce((sum, p) => sum + p.y, 0) / points.length;

        // Classify by quadrant
        const tl = points.find(p => p.x < cx && p.y < cy);
        const tr = points.find(p => p.x >= cx && p.y < cy);
        const br = points.find(p => p.x >= cx && p.y >= cy);
        const bl = points.find(p => p.x < cx && p.y >= cy);

        return [tl, tr, br, bl];
    }

    private applyPerspectiveTransform(
        image: cv.Mat,
        coloredImage: cv.Mat,
        markers: any[],
        template: Template
    ): { croppedGray: cv.Mat; croppedColored: cv.Mat; shifts: [number, number] } {
        // Source points (detected markers)
        const srcPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
            markers[0].x, markers[0].y,
            markers[1].x, markers[1].y,
            markers[2].x, markers[2].y,
            markers[3].x, markers[3].y
        ]);

        // Destination points (template coordinates)
        const margins = this.options.margins || [0, 0, 0, 0];
        const [width, height] = template.templateDimensions;
        const dstPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
            margins[2], margins[0],
            width - margins[3], margins[0],
            width - margins[3], height - margins[1],
            margins[2], height - margins[1]
        ]);

        // Calculate homography
        const M = cv.getPerspectiveTransform(srcPoints, dstPoints);

        // Warp images
        const croppedGray = new cv.Mat();
        const croppedColored = new cv.Mat();
        const dsize = new cv.Size(width, height);

        cv.warpPerspective(image, croppedGray, M, dsize);
        cv.warpPerspective(coloredImage, croppedColored, M, dsize);

        // Calculate shifts (for now, assume [0, 0] after crop)
        const shifts: [number, number] = [0, 0];

        // Cleanup
        srcPoints.delete();
        dstPoints.delete();
        M.delete();

        return { croppedGray, croppedColored, shifts };
    }
}
```

---

## Summary

**CropOnMarkers**: Detect markers and crop to standard coordinates
**Marker Types**: FOUR_DOTS, TWO_DOTS_ONE_LINE, TWO_LINES, CUSTOM
**Algorithm**: Detect markers → Order points → Perspective transform
**Template Update**: Calculate and apply field block shifts
**Browser**: Use OpenCV.js for contour detection and perspective transform

**Key Takeaway**: CropOnMarkers normalizes scan positions to template coordinates. Essential for handling varied scan positions.
