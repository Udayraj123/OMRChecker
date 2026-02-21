# Preprocessing Flow - Concept

**Module**: Domain / Preprocessing
**Python Reference**: `src/processors/image/coordinator.py`, `src/processors/image/base.py`
**Last Updated**: 2026-02-20

---

## Overview

The PreprocessingCoordinator orchestrates all image preprocessing steps before detection. It executes preprocessors in sequence to transform raw OMR scans into clean, standardized images ready for bubble detection.

**Key Responsibilities**:
1. **Image Resizing**: Resize to processing dimensions
2. **Preprocessor Execution**: Run preprocessors in configured order
3. **Template Mutation**: Track field block shifts from preprocessing
4. **Visualization**: Show before/after diffs when configured
5. **State Management**: Create template copy for mutation

---

## Preprocessing Architecture

### PreprocessingCoordinator

**Code Reference**: `src/processors/image/coordinator.py:9-115`

**Purpose**: Coordinates all image preprocessing steps in sequence

**Not an individual preprocessor**: It orchestrates preprocessors defined in template

---

## Preprocessing Flow

### 1. Template Copy

**Code Reference**: `src/processors/image/coordinator.py:48-50`

```python
# Get a copy of the template layout for mutation
next_template_layout = context.template.template_layout.get_copy_for_shifting()
next_template_layout.reset_all_shifts()
```

**Why Copy?**:
- Preprocessors mutate template (shifts, transformations)
- Each file may have different shifts
- Original template preserved for next file

**What's Copied**:
- Shallow copy of template layout
- Deep copy of field blocks (only mutable parts)

---

### 2. Initial Resize

**Code Reference**: `src/processors/image/coordinator.py:57-63`

```python
# Resize to conform to common preprocessor input requirements
gray_image = ImageUtils.resize_to_shape(
    next_template_layout.processing_image_shape, gray_image
)
if tuning_config.outputs.colored_outputs_enabled:
    colored_image = ImageUtils.resize_to_shape(
        next_template_layout.processing_image_shape, colored_image
    )
```

**Purpose**: Standardize image dimensions for consistent processing

**Processing Dimensions**: From template.json `processingImageShape` (e.g., [900, 650])

---

### 3. Preprocessor Execution

**Code Reference**: `src/processors/image/coordinator.py:72-99`

```python
# Run preprocessors in sequence using their process() method
for pre_processor in next_template_layout.pre_processors:
    pre_processor_name = pre_processor.get_name()

    # Show Before Preview (if configured)
    if show_preprocessors_diff.get(pre_processor_name, False):
        InteractionUtils.show(f"Before {pre_processor_name}", image)

    # Process using unified interface
    context = pre_processor.process(context)

    # Show After Preview (if configured)
    if show_preprocessors_diff.get(pre_processor_name, False):
        InteractionUtils.show(f"After {pre_processor_name}", image)
```

**Preprocessor Order**: Defined in template.json `preProcessors` array

**Common Order**:
1. AutoRotate (fix rotation)
2. CropOnMarkers (crop to markers)
3. GaussianBlur (reduce noise)
4. Contrast (adjust contrast)
5. Levels (adjust brightness)

---

### 4. Final Resize (Optional)

**Code Reference**: `src/processors/image/coordinator.py:102-110`

```python
# Resize to output requirements if specified
if template_layout.output_image_shape:
    context.gray_image = ImageUtils.resize_to_shape(
        template_layout.output_image_shape, context.gray_image
    )
```

**Purpose**: Resize to different dimensions for detection (if needed)

---

## Preprocessor Base Class

### ImageTemplatePreprocessor

**Code Reference**: `src/processors/image/base.py`

**Purpose**: Base class for all image preprocessors

**Interface**:
```python
class ImageTemplatePreprocessor(Processor):
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Process images and update template."""
        gray_image, colored_image, template = self.apply_filter(
            context.gray_image,
            context.colored_image,
            context.template,
            context.file_path
        )
        context.gray_image = gray_image
        context.colored_image = colored_image
        context.template = template
        return context

    @abstractmethod
    def apply_filter(self, image, colored_image, template, file_path):
        """Apply preprocessing operation."""
        pass
```

**Key Methods**:
- `process(context)`: Unified interface for pipeline
- `apply_filter(image, colored_image, template, file_path)`: Actual preprocessing logic
- `get_relative_path(path)`: Resolve paths relative to template directory
- `exclude_files()`: Return files to exclude from processing

---

## Preprocessor Configuration

### Template JSON Format

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
    },
    {
      "name": "CropOnMarkers",
      "options": {
        "type": "FOUR_DOTS",
        "referenceImage": "omr_marker.jpg"
      }
    },
    {
      "name": "GaussianBlur",
      "options": {
        "kSize": [3, 3],
        "sigmaX": 0
      }
    }
  ]
}
```

---

## Preprocessor Types

### 1. Geometric Transformations

**AutoRotate**: Detect and fix rotation (0°, 90°, 180°, 270°)
**CropOnMarkers**: Crop based on marker detection
**CropPage**: Detect page contours and crop
**FeatureBasedAlignment**: SIFT-based alignment

**Effect**: Change image dimensions, shift field positions

---

### 2. Image Enhancement

**GaussianBlur**: Reduce noise with Gaussian blur
**MedianBlur**: Reduce noise with median blur
**Contrast**: Adjust image contrast
**Levels**: Adjust brightness levels

**Effect**: Improve image quality for detection, no position changes

---

### 3. Advanced Transformations

**WarpOnPoints**: Perspective transformation based on points
**PiecewiseAffine**: Delaunay triangulation-based warping

**Effect**: Complex geometric transformations, shift field positions

---

## Visualization (Debug Mode)

### Show Preprocessors Diff

**Config** (config.json):
```json
{
  "outputs": {
    "showPreprocessorsDiff": {
      "AutoRotate": true,
      "CropOnMarkers": true,
      "GaussianBlur": false
    }
  }
}
```

**Behavior**: Show before/after images for specified preprocessors

**Use Case**: Debug preprocessing pipeline, tune parameters

---

## Browser Migration Notes

### TypeScript PreprocessingCoordinator

```typescript
class PreprocessingCoordinator implements Processor {
    private template: Template;
    private tuningConfig: TuningConfig;

    constructor(template: Template) {
        this.template = template;
        this.tuningConfig = template.tuningConfig;
    }

    getName(): string {
        return 'Preprocessing';
    }

    async process(context: ProcessingContext): Promise<ProcessingContext> {
        // 1. Copy template layout for mutation
        const nextTemplateLayout = context.template.templateLayout.getCopyForShifting();
        nextTemplateLayout.resetAllShifts();

        // 2. Resize images
        let grayImage = resizeToShape(
            nextTemplateLayout.processingImageShape,
            context.grayImage
        );
        let coloredImage = this.tuningConfig.outputs.coloredOutputsEnabled
            ? resizeToShape(nextTemplateLayout.processingImageShape, context.coloredImage)
            : context.coloredImage;

        // 3. Update context
        context.grayImage = grayImage;
        context.coloredImage = coloredImage;
        context.template.templateLayout = nextTemplateLayout;

        // 4. Run preprocessors
        for (const preprocessor of nextTemplateLayout.preProcessors) {
            const name = preprocessor.getName();

            // Show before (if configured)
            if (this.shouldShowDiff(name)) {
                this.showImage(`Before ${name}`, context.grayImage);
            }

            // Process
            context = await preprocessor.process(context);

            // Show after (if configured)
            if (this.shouldShowDiff(name)) {
                this.showImage(`After ${name}`, context.grayImage);
            }
        }

        return context;
    }

    private shouldShowDiff(preprocessorName: string): boolean {
        return this.tuningConfig.outputs.showPreprocessorsDiff[preprocessorName] ?? false;
    }

    private showImage(title: string, mat: cv.Mat): void {
        // Display in canvas or UI
        const canvas = document.createElement('canvas');
        cv.imshow(canvas, mat);
        // Append to debug panel or modal
    }
}
```

### Preprocessor Base Class (TypeScript)

```typescript
abstract class ImageTemplatePreprocessor implements Processor {
    protected options: Record<string, any>;
    protected relativeDir: string;
    protected tuningConfig: TuningConfig;

    constructor(options: Record<string, any>, relativeDir: string, tuningConfig: TuningConfig) {
        this.options = options;
        this.relativeDir = relativeDir;
        this.tuningConfig = tuningConfig;
    }

    abstract getName(): string;

    async process(context: ProcessingContext): Promise<ProcessingContext> {
        const { grayImage, coloredImage, template } = await this.applyFilter(
            context.grayImage,
            context.coloredImage,
            context.template,
            context.filePath
        );

        context.grayImage = grayImage;
        context.coloredImage = coloredImage;
        context.template = template;

        return context;
    }

    abstract applyFilter(
        image: cv.Mat,
        coloredImage: cv.Mat,
        template: Template,
        filePath: string
    ): Promise<{
        grayImage: cv.Mat;
        coloredImage: cv.Mat;
        template: Template;
    }>;

    protected getRelativePath(path: string): string {
        return `${this.relativeDir}/${path}`;
    }

    excludeFiles(): string[] {
        return [];
    }
}
```

---

## Summary

**PreprocessingCoordinator**: Orchestrates preprocessing pipeline
**Flow**: Copy template → Resize → Execute preprocessors → Optional final resize
**Preprocessor Types**: Geometric transformations, image enhancement, advanced warping
**Mutation**: Template layout copied and shifted during preprocessing
**Visualization**: Show before/after diffs for debugging

**Browser Migration**:
- Use async/await for preprocessor execution
- Implement same preprocessor base class pattern
- Display debug images in UI instead of cv2.imshow
- Use OpenCV.js for image operations
- Maintain same preprocessor sequence

**Key Takeaway**: Preprocessing transforms raw scans into clean, standardized images. Browser version should maintain same pipeline structure but use OpenCV.js for image operations.
