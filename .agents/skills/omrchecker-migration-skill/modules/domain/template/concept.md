# Template Entity - Concept

**Module**: Domain / Template
**Python Reference**: `src/processors/template/template.py`, `src/schemas/models/template.py`
**Last Updated**: 2026-02-20

---

## Overview

The Template is the central entity in OMRChecker that defines the structure and layout of OMR sheets. It encapsulates template configuration (from template.json), field blocks, fields, preprocessing logic, and drawing utilities.

**Key Responsibilities**:
1. **Layout Definition**: Defines field blocks, fields, and their positions
2. **Preprocessing Orchestration**: Manages image preprocessors (rotation, cropping, filtering)
3. **Pipeline Coordination**: Orchestrates the entire processing pipeline
4. **Directory Management**: Handles output directories and file organization
5. **State Management**: Maintains template state and field shifts during processing

---

## Template Architecture

### Template Class

**Code Reference**: `src/processors/template/template.py:17-252`

**Composition**:
```
Template
├── TemplateLayout (layout management)
├── TemplateDrawing (visualization)
├── TemplateDirectoryHandler (file I/O)
├── ProcessingPipeline (processing orchestration)
└── SaveImageOps (image saving utilities)
```

**Attributes**:
- `path` (Path): Path to template.json file
- `tuning_config` (TuningConfig): Global configuration settings
- `args` (dict): CLI arguments
- `template_layout` (TemplateLayout): Layout and field structure
- `drawing` (TemplateDrawing): Drawing utilities
- `directory_handler` (TemplateDirectoryHandler): Directory management
- `pipeline` (ProcessingPipeline): Processing pipeline
- `save_image_ops` (SaveImageOps): Image saving operations

**Re-exported from TemplateLayout**:
- `field_blocks` (list[FieldBlock]): All field blocks in template
- `all_fields` (list[Field]): All fields across all blocks
- `alignment` (dict): Alignment configuration
- `global_empty_val` (str): Default empty value for unanswered fields
- `output_columns` (list[str]): Output column ordering
- `template_dimensions` (list[int]): Original template dimensions [width, height]
- `all_field_detection_types` (set): All detection types used in template

---

## TemplateConfig Dataclass

**Code Reference**: `src/schemas/models/template.py:86-171`

**Purpose**: Type-safe representation of template.json structure

**Required Fields**:
- `bubble_dimensions` (list[int]): Default bubble size [width, height]
- `template_dimensions` (list[int]): Page size [width, height]

**Optional Fields**:
- `alignment` (AlignmentConfig): Alignment margins and max displacement
- `conditional_sets` (list): Conditional question sets
- `custom_labels` (dict): Custom label definitions
- `custom_bubble_field_types` (dict): Custom bubble field type definitions
- `empty_value` (str): Value for empty/unanswered fields (default: "")
- `field_blocks` (dict): Field block definitions (core structure)
- `field_blocks_offset` (list[int]): Global offset for all field blocks
- `output_columns` (OutputColumnsConfig): Column ordering configuration
- `pre_processors` (list): Image preprocessors to apply
- `processing_image_shape` (list[int]): Image shape for processing [height, width]
- `sort_files` (SortFilesConfig): File sorting configuration

---

## Template Lifecycle

### 1. Initialization

**Code Reference**: `src/processors/template/template.py:18-48`

```python
template = Template(template_path, tuning_config, args)
```

**Steps**:
1. Load template.json from `template_path`
2. Initialize SaveImageOps for debug image saving
3. Create TemplateLayout (parses JSON, creates field blocks/fields)
4. Initialize TemplateDrawing for visualization
5. Initialize TemplateDirectoryHandler for file management
6. Initialize ProcessingPipeline with all processors

**Dependencies**:
- `tuning_config`: Global configuration (thresholds, output settings)
- `args`: CLI arguments (ML models, debug flags, etc.)

---

### 2. Directory Setup

**Code Reference**: `src/processors/template/template.py:53-60`

```python
template.reset_and_setup_for_directory(output_dir)
```

**Purpose**: Reset template state and setup output directories for a new batch of files

**Steps**:
1. Reset all field shifts to [0, 0]
2. Create output directories based on output_mode:
   - `Results/`: Processed OMR results
   - `Errors/`: Files with errors (marker detection failures)
   - `MultiMarked/`: Files with multiple marked bubbles
   - `SavedMarked/`: Debug images with marked bubbles
   - `Evaluations/`: Evaluation results (if enabled)
   - `ImageMetrics/`: Bubble metrics for visualization

---

### 3. File Processing

**Code Reference**: `src/processors/template/template.py:123-137`

```python
context = template.process_file(file_path, gray_image, colored_image)
```

**Purpose**: Process a single OMR image through entire pipeline

**Pipeline Stages**:
1. **Preprocessing**: Image rotation, cropping, filtering, warping
2. **Alignment**: Feature-based alignment with reference image
3. **Detection**: Bubble/barcode/OCR detection and interpretation
4. **(Optional) Training Data Collection**: Collect labeled data for ML

**Returns**: ProcessingContext with:
- `omr_response` (dict): Detected field values
- `is_multi_marked` (bool): Whether multiple bubbles marked
- `field_id_to_interpretation` (dict): Per-field bubble interpretations
- `score` (float): Evaluation score (if evaluation enabled)
- `evaluation_meta` (dict): Evaluation details

---

### 4. Metrics Export

**Code Reference**: `src/processors/template/template.py:170-251`

```python
template.export_omr_metrics_for_file(
    file_path, evaluation_meta, field_id_to_interpretation, context
)
```

**Purpose**: Export bubble-level metrics for visualization

**Output**: JavaScript file at `outputs/ImageMetrics/metrics-{filename}.js`

**Content**:
- `template_meta`: Bubble interpretations, thresholds, multi-mark flags
- `evaluation_meta`: Scores, correct/incorrect answers, marking scheme

**Use Case**: HTML visualization dashboard (interactive bubble inspection)

---

## Template JSON Structure

### Minimal Template Example

```json
{
  "templateDimensions": [1200, 1600],
  "bubbleDimensions": [32, 32],
  "fieldBlocks": {
    "q1-10": {
      "origin": [100, 200],
      "fieldLabels": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"],
      "labelsGap": 50,
      "bubbleValues": ["A", "B", "C", "D"],
      "direction": "vertical",
      "fieldDetectionType": "BUBBLES_THRESHOLD",
      "bubbleFieldType": "StandardBubble"
    }
  }
}
```

### Full Template Example

**Code Reference**: `samples/community/UPI-1/template.json`

```json
{
  "templateDimensions": [1200, 1600],
  "bubbleDimensions": [32, 32],
  "processingImageShape": [900, 650],
  "emptyValue": "",
  "outputColumns": {
    "customOrder": ["Roll", "q1", "q2", "q3", "q4", "q5"],
    "sortType": "CUSTOM"
  },
  "preProcessors": [
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
  ],
  "fieldBlocks": {
    "Roll": {
      "origin": [50, 100],
      "fieldLabels": ["Roll"],
      "bubbleValues": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
      "direction": "vertical",
      "fieldDetectionType": "BUBBLES_THRESHOLD",
      "bubbleFieldType": "StandardBubble",
      "bubbleDimensions": [28, 28],
      "bubblesGap": 40
    },
    "q1-5": {
      "origin": [200, 100],
      "fieldLabels": ["q1", "q2", "q3", "q4", "q5"],
      "labelsGap": 60,
      "bubbleValues": ["A", "B", "C", "D"],
      "direction": "vertical",
      "fieldDetectionType": "BUBBLES_THRESHOLD",
      "bubbleFieldType": "StandardBubble"
    }
  },
  "customBubbleFieldTypes": {
    "CustomType1": {
      "origin": [0, 0],
      "bubblesGap": 0,
      "labelsGap": 0,
      "fieldLabels": ["CUSTOM_1"]
    }
  },
  "alignment": {
    "margins": {
      "top": 50,
      "bottom": 50,
      "left": 50,
      "right": 50
    },
    "maxDisplacement": 10,
    "referenceImage": "reference.jpg"
  }
}
```

---

## Key Methods

### get_pre_processors()

**Code Reference**: `src/processors/template/template.py:71-73`

**Purpose**: Get list of image preprocessors

**Returns**: list[ImageTemplatePreprocessor]

**Common Preprocessors**:
- AutoRotate: Automatic rotation detection
- CropOnMarkers: Crop based on marker dots/lines
- CropPage: Detect and crop page contours
- GaussianBlur: Blur for noise reduction
- Contrast: Adjust contrast
- Levels: Adjust brightness levels
- FeatureBasedAlignment: SIFT-based alignment

---

### get_concatenated_omr_response()

**Code Reference**: `src/processors/layout/template_layout.py` (re-exported)

**Purpose**: Format OMR response as human-readable string

**Example Output**:
```
Roll: 12345
q1: A
q2: B
q3: C
q4: A
q5: D
```

---

### reset_all_shifts()

**Code Reference**: `src/processors/layout/template_layout.py`

**Purpose**: Reset all field block shifts to [0, 0]

**When Used**: Before processing a new directory (shifts are per-file mutations)

---

## Template Mutations

**Important**: Template layout is mutable during preprocessing

**Why Mutable**:
- Preprocessors can shift field blocks (CropOnMarkers, alignment)
- Each file may have different shifts
- Template is copied before processing each file

**Copy Strategy**:
```python
# Shallow copy template layout
next_template_layout = template_layout.get_copy_for_shifting()

# Deep copy only mutable parts (field blocks)
template_layout.field_blocks = [
    field_block.get_copy_for_shifting() for field_block in self.field_blocks
]
```

**Code Reference**: `src/processors/layout/template_layout.py:88-96`

---

## Browser Migration Notes

### JavaScript Template Class

```typescript
class Template {
    path: string;
    config: TemplateConfig;
    layout: TemplateLayout;
    drawing: TemplateDrawing;
    pipeline: ProcessingPipeline;

    constructor(templatePath: string, tuningConfig: TuningConfig, args?: Record<string, any>) {
        this.path = templatePath;
        this.config = TemplateConfig.fromJSON(templatePath);
        this.layout = new TemplateLayout(this, this.config);
        this.drawing = new TemplateDrawing(this);
        this.pipeline = new ProcessingPipeline(this, args);
    }

    async processFile(
        filePath: string,
        grayImage: cv.Mat,
        coloredImage: cv.Mat
    ): Promise<ProcessingContext> {
        return this.pipeline.processFile(filePath, grayImage, coloredImage);
    }

    resetForDirectory(): void {
        this.layout.resetAllShifts();
    }
}
```

### Template JSON Loading

```typescript
class TemplateConfig {
    static async fromJSON(templatePath: string): Promise<TemplateConfig> {
        // Browser: Load from File API or fetch
        const response = await fetch(templatePath);
        const data = await response.json();
        return TemplateConfig.parse(data);
    }

    static parse(data: any): TemplateConfig {
        // Use Zod for validation
        const schema = z.object({
            templateDimensions: z.tuple([z.number(), z.number()]),
            bubbleDimensions: z.tuple([z.number(), z.number()]),
            fieldBlocks: z.record(z.any()),
            preProcessors: z.array(z.any()).default([]),
            // ... other fields
        });

        return schema.parse(data);
    }
}
```

### File System Abstraction

```typescript
// Browser: No file system, use in-memory directories
class DirectoryHandler {
    private results: Map<string, any> = new Map();
    private errors: Map<string, any> = new Map();
    private multiMarked: Map<string, any> = new Map();

    saveResult(filename: string, result: any): void {
        this.results.set(filename, result);
    }

    downloadAllResults(): void {
        const csv = this.resultsToCSV();
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'results.csv';
        a.click();
    }
}
```

### Preprocessor Loading

```typescript
// Browser: Dynamic imports instead of PROCESSOR_MANAGER
const PREPROCESSOR_MAP = {
    'AutoRotate': () => import('./preprocessors/AutoRotate'),
    'CropOnMarkers': () => import('./preprocessors/CropOnMarkers'),
    'GaussianBlur': () => import('./preprocessors/GaussianBlur'),
    // ...
};

async function loadPreprocessor(name: string): Promise<Preprocessor> {
    const module = await PREPROCESSOR_MAP[name]();
    return new module.default();
}
```

---

## Summary

**Template Entity**: Central orchestrator for OMR processing
**Composition**: Layout + Drawing + DirectoryHandler + Pipeline
**Configuration**: Type-safe dataclass from template.json
**Lifecycle**: Initialize → Setup Directory → Process Files → Export Metrics
**Mutability**: Template layout copied and shifted per file
**Output**: OMR responses, evaluation scores, bubble metrics

**Browser Migration**:
- Use Zod for template.json validation
- Replace file system with in-memory storage + downloads
- Use dynamic imports for preprocessors
- Maintain same template structure and lifecycle

**Key Takeaway**: Template is the single source of truth for OMR sheet structure. Browser version should maintain same composition and lifecycle, but adapt I/O operations for browser environment.
