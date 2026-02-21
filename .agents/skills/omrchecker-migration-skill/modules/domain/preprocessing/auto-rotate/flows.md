# AutoRotate Processor - Flows

**Module**: Domain / Preprocessing / AutoRotate
**Python Reference**: `src/processors/image/AutoRotate.py`
**Last Updated**: 2026-02-20

---

## Overview

AutoRotate automatically detects and corrects rotation of OMR sheets (0°, 90°, 180°, 270°) using template matching with a reference marker image.

**Use Case**: Handle scanned sheets that may be rotated (upside down, sideways)

---

## Configuration

### Template JSON

```json
{
  "preProcessors": [
    {
      "name": "AutoRotate",
      "options": {
        "referenceImage": "marker.jpg",
        "markerDimensions": [100, 100],
        "threshold": {
          "value": 0.7,
          "passthrough": false
        }
      }
    }
  ]
}
```

**Options**:
- `referenceImage` (required): Path to reference marker image (relative to template directory)
- `markerDimensions` (optional): Resize marker to these dimensions for matching
- `threshold` (optional): Minimum match score
  - `value`: Minimum correlation coefficient (0.0-1.0)
  - `passthrough`: Continue even if below threshold (True/False)

---

## Initialization

**Code Reference**: `src/processors/image/AutoRotate.py:20-36`

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Load reference image
    path = self.get_relative_path(self.options["reference_image"])
    if not path.exists():
        raise ImageReadError(f"Reference image for AutoRotate not found at {path}")

    self.reference_image = ImageUtils.load_image(path, cv2.IMREAD_GRAYSCALE)

    # Optional: resize reference marker
    self.marker_dimensions = self.options.get("marker_dimensions", None)
    if self.marker_dimensions:
        self.resized_reference = ImageUtils.resize_to_dimensions(
            self.marker_dimensions, self.reference_image
        )
    else:
        self.resized_reference = self.reference_image

    self.threshold = self.options.get("threshold", None)
```

---

## Rotation Detection Flow

**Code Reference**: `src/processors/image/AutoRotate.py:38-113`

### Step 1: Try All Rotations

```python
rotations = [
    None,                           # 0° (no rotation)
    cv2.ROTATE_90_CLOCKWISE,        # 90° CW
    cv2.ROTATE_180,                 # 180°
    cv2.ROTATE_90_COUNTERCLOCKWISE  # 270° CW (90° CCW)
]

best_val, best_rotation = -1, None

for rotation in rotations:
    rotated_img = image if rotation is None else ImageUtils.rotate(image, rotation)

    # Template matching
    res = cv2.matchTemplate(rotated_img, self.resized_reference, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(res)

    if max_val > best_val:
        best_val = max_val
        best_rotation = rotation
```

**Algorithm**: Template matching with normalized cross-correlation
**Output**: Best rotation with highest correlation score

---

### Step 2: Threshold Check (Optional)

```python
if self.threshold is not None and self.threshold["value"] > best_val:
    if self.threshold["passthrough"]:
        logger.warning("Autorotate score below threshold. Continuing due to passthrough.")
    else:
        logger.error("Autorotate score below threshold.")
        raise ImageProcessingError("Autorotate score below threshold")
```

**Behavior**:
- If threshold configured and score too low:
  - `passthrough=True`: Log warning, continue
  - `passthrough=False`: Raise error, stop processing

---

### Step 3: Apply Rotation

```python
logger.info(f"AutoRotate Applied with rotation {best_rotation} and value {best_val}")

if best_rotation is None:
    return image, colored_image, template  # No rotation needed

# Apply rotation
image = ImageUtils.rotate(image, best_rotation, keep_original_shape=True)
if tuning_config.outputs.colored_outputs_enabled:
    colored_image = ImageUtils.rotate(colored_image, best_rotation, keep_original_shape=True)

return image, colored_image, template
```

---

## Template Matching Details

### Normalized Cross-Correlation

**Method**: `cv2.TM_CCOEFF_NORMED`

**Formula**:
```
R(x,y) = Σ(T'(x',y') · I'(x+x',y+y')) / sqrt(Σ(T'²) · Σ(I'²))
```

Where:
- T' = Template - mean(Template)
- I' = Image region - mean(Image region)

**Range**: -1.0 to 1.0
- 1.0: Perfect match
- 0.0: No correlation
- -1.0: Inverse correlation

---

### Rotation Mapping

**OpenCV Constants**:
- `None`: No rotation (0°)
- `cv2.ROTATE_90_CLOCKWISE`: Rotate 90° clockwise
- `cv2.ROTATE_180`: Rotate 180°
- `cv2.ROTATE_90_COUNTERCLOCKWISE`: Rotate 270° clockwise (90° counter-clockwise)

---

## Edge Cases

### 1. Reference Image Not Found

**Error**: `ImageReadError`
**Message**: "Reference image for AutoRotate not found at {path}"
**Solution**: Check `referenceImage` path in template.json

---

### 2. Low Match Score

**Scenario**: Best match score below threshold

**Behavior**:
- `passthrough=True`: Log warning, use best rotation anyway
- `passthrough=False`: Raise `ImageProcessingError`, stop processing

**Common Causes**:
- Reference marker doesn't match input
- Poor image quality
- Wrong reference image

---

### 3. Multiple Similar Scores

**Scenario**: Two rotations have very similar scores (e.g., 0.85 vs 0.86)

**Behavior**: Use rotation with highest score (no tie-breaking)

**Risk**: May select wrong rotation if scores are very close

---

## Debugging

### Visualization (showImageLevel >= 5)

**Code Reference**: `src/processors/image/AutoRotate.py:59-67`

```python
if config.outputs.show_image_level >= 5:
    hstack = ImageUtils.get_padded_hstack([rotated_img, self.resized_reference])
    InteractionUtils.show("Template matching result", res, 0)
    InteractionUtils.show(
        f"Template Matching for rotation: {rotation_name} ({max_val:.2f})",
        hstack
    )
```

**Shows**:
- Template matching heatmap
- Side-by-side comparison (rotated image vs reference)
- Match score for each rotation

---

## Browser Migration

### TypeScript Implementation

```typescript
class AutoRotate extends ImageTemplatePreprocessor {
    private referenceImage: cv.Mat;
    private resizedReference: cv.Mat;
    private threshold: { value: number; passthrough: boolean } | null;

    constructor(options: any, relativeDir: string, tuningConfig: TuningConfig) {
        super(options, relativeDir, tuningConfig);
        this.threshold = options.threshold || null;
    }

    async initialize(): Promise<void> {
        // Load reference image
        const path = this.getRelativePath(this.options.referenceImage);
        const imageElement = await loadImage(path);
        this.referenceImage = cv.imread(imageElement);

        // Convert to grayscale
        cv.cvtColor(this.referenceImage, this.referenceImage, cv.COLOR_RGBA2GRAY);

        // Resize if needed
        if (this.options.markerDimensions) {
            this.resizedReference = new cv.Mat();
            const size = new cv.Size(
                this.options.markerDimensions[0],
                this.options.markerDimensions[1]
            );
            cv.resize(this.referenceImage, this.resizedReference, size);
        } else {
            this.resizedReference = this.referenceImage.clone();
        }
    }

    getName(): string {
        return 'AutoRotate';
    }

    async applyFilter(
        image: cv.Mat,
        coloredImage: cv.Mat,
        template: Template,
        filePath: string
    ): Promise<{ grayImage: cv.Mat; coloredImage: cv.Mat; template: Template }> {
        const rotations = [
            { angle: null, code: null },
            { angle: 90, code: cv.ROTATE_90_CLOCKWISE },
            { angle: 180, code: cv.ROTATE_180 },
            { angle: 270, code: cv.ROTATE_90_COUNTERCLOCKWISE }
        ];

        let bestVal = -1;
        let bestRotation: any = null;

        for (const rotation of rotations) {
            let rotatedImg: cv.Mat;

            if (rotation.code === null) {
                rotatedImg = image;
            } else {
                rotatedImg = new cv.Mat();
                cv.rotate(image, rotatedImg, rotation.code);
            }

            // Template matching
            const result = new cv.Mat();
            const mask = new cv.Mat();
            cv.matchTemplate(rotatedImg, this.resizedReference, result, cv.TM_CCOEFF_NORMED, mask);

            const minMax = cv.minMaxLoc(result);
            const maxVal = minMax.maxVal;

            if (maxVal > bestVal) {
                bestVal = maxVal;
                bestRotation = rotation;
            }

            // Cleanup
            result.delete();
            mask.delete();
            if (rotation.code !== null) {
                rotatedImg.delete();
            }
        }

        // Check threshold
        if (this.threshold && this.threshold.value > bestVal) {
            if (this.threshold.passthrough) {
                console.warn('AutoRotate score below threshold. Continuing due to passthrough.');
            } else {
                throw new Error(`AutoRotate score below threshold: ${bestVal} < ${this.threshold.value}`);
            }
        }

        console.log(`AutoRotate: Best rotation ${bestRotation.angle}° with score ${bestVal}`);

        // Apply best rotation
        let finalGray = image;
        let finalColored = coloredImage;

        if (bestRotation.code !== null) {
            finalGray = new cv.Mat();
            cv.rotate(image, finalGray, bestRotation.code);

            if (this.tuningConfig.outputs.coloredOutputsEnabled) {
                finalColored = new cv.Mat();
                cv.rotate(coloredImage, finalColored, bestRotation.code);
            }
        }

        return { grayImage: finalGray, coloredImage: finalColored, template };
    }

    excludeFiles(): string[] {
        return [this.getRelativePath(this.options.referenceImage)];
    }
}
```

---

## Summary

**AutoRotate**: Detect and fix sheet rotation (0°, 90°, 180°, 270°)
**Algorithm**: Template matching with normalized cross-correlation
**Threshold**: Optional minimum match score
**Edge Cases**: Missing reference, low scores, similar scores
**Browser**: Use OpenCV.js cv.matchTemplate and cv.rotate

**Key Takeaway**: AutoRotate handles rotated scans automatically. Browser version uses same algorithm with OpenCV.js.
