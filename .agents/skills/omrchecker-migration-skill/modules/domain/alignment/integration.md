# Alignment Flow - Integration Points

## Pipeline Integration

### Position in Processing Pipeline

```
Entry Point
  ↓
Pre-processors (AutoRotate, CropOnMarkers, etc.)
  ↓
[ALIGNMENT PROCESSOR] ← YOU ARE HERE
  ↓
Detection (Bubble/Barcode/OCR)
  ↓
Evaluation
  ↓
Output (CSV)
```

**Key Point**: Alignment happens AFTER preprocessing, BEFORE detection

**Why?**
- Pre-processors normalize image (rotation, cropping)
- Alignment fine-tunes position for detection
- Detection uses aligned coordinates

---

## Integration with Template System

### Template Loading

**File**: `src/processors/template/template.py` (inferred)

```python
# Template JSON structure:
{
  "alignment": {
    "margins": {"top": 50, "bottom": 50, "left": 50, "right": 50},
    "max_displacement": 30,
    "gray_alignment_image": "<loaded at runtime>",
    "colored_alignment_image": "<loaded at runtime>"
  },
  "fieldBlocks": {
    "Q1-30": {
      "alignment": {
        "margins": {"top": 20, "bottom": 20, "left": 20, "right": 20},
        "max_displacement": 15
      }
    }
  }
}
```

**Loading Flow**:
```python
# 1. Parse template JSON
template_data = json.load(template_file)

# 2. Extract alignment config
alignment_config = template_data.get("alignment", {})

# 3. Load reference images
if "reference_image_path" in alignment_config:
    gray_alignment_image = cv2.imread(
        alignment_config["reference_image_path"],
        cv2.IMREAD_GRAYSCALE
    )
    colored_alignment_image = cv2.imread(
        alignment_config["reference_image_path"],
        cv2.IMREAD_COLOR
    )

    # 4. Store in template object
    template.alignment["gray_alignment_image"] = gray_alignment_image
    template.alignment["colored_alignment_image"] = colored_alignment_image
```

**Integration Point**: `AlignmentConfig` dataclass

**File**: `src/schemas/models/template.py:30-43`
```python
@dataclass
class AlignmentConfig:
    """Configuration for template alignment."""

    margins: AlignmentMarginsConfig = field(default_factory=AlignmentMarginsConfig)
    max_displacement: int = 10

    @classmethod
    def from_dict(cls, data: dict) -> "AlignmentConfig":
        margins_data = data.get("margins", {})
        return cls(
            margins=AlignmentMarginsConfig.from_dict(margins_data),
            max_displacement=data.get("max_displacement", 10),
        )
```

---

## Integration with ProcessingContext

### Context Flow

**Input Context**:
```python
class ProcessingContext:
    gray_image: np.ndarray          # From pre-processors
    colored_image: np.ndarray       # From pre-processors
    template: Template              # Contains alignment config
    # ... other fields
```

**Alignment Processing**:
```python
def process(self, context: ProcessingContext) -> ProcessingContext:
    # Read from context
    gray_image = context.gray_image
    colored_image = context.colored_image
    template = context.template

    # Apply alignment
    if "gray_alignment_image" in template.alignment:
        gray_image, colored_image, template = apply_template_alignment(...)

    # Write back to context
    context.gray_image = gray_image
    context.colored_image = colored_image
    context.template = template  # Now contains shifts

    return context
```

**Output Context**:
```python
# Updated template with shifts:
context.template.field_blocks[0].fields[0].scan_boxes[0].shifts = [-3, 2]
# Detection processors will use these shifts
```

---

## Integration with Field Blocks

### Field Block Structure

**File**: `src/processors/layout/field_block/` (inferred)

```python
class FieldBlock:
    name: str
    origin: list[int]                    # [x, y]
    bounding_box_origin: list[int]       # [x, y]
    bounding_box_dimensions: list[int]   # [width, height]
    fields: list[Field]
    alignment: dict                      # Override config
    shifts: list[int]                    # Updated by alignment
```

**Alignment Override**:
```python
# Template-level default:
template.alignment = {
    "margins": {"top": 50, "bottom": 50, "left": 50, "right": 50},
    "max_displacement": 30
}

# Field block override:
field_block.alignment = {
    "margins": {"top": 20, "bottom": 20, "left": 20, "right": 20},
    "max_displacement": 15
}

# Resolution:
margins = field_block.alignment.get("margins", template.alignment["margins"])
# Uses field block's 20-pixel margins, not template's 50-pixel
```

**Shift Application**:
```python
# Alignment writes:
field_block.shifts = [dx, dy]

# AND also writes to all scan boxes:
for field in field_block.fields:
    for scan_box in field.scan_boxes:
        scan_box.shifts = [dx, dy]
```

---

## Integration with Detection

### Bubble Detection Usage

**File**: `src/processors/detection/bubbles_threshold/` (inferred)

```python
def detect_bubbles(field_block, gray_image):
    for field in field_block.fields:
        for scan_box in field.scan_boxes:
            # Get template position
            template_x, template_y = scan_box.position

            # Apply alignment shift
            shift_x, shift_y = scan_box.shifts  # From alignment

            # Final position
            actual_x = template_x + shift_x
            actual_y = template_y + shift_y

            # Extract ROI
            roi = gray_image[
                actual_y - radius : actual_y + radius,
                actual_x - radius : actual_x + radius
            ]

            # Detect bubble darkness
            darkness = calculate_darkness(roi)
```

**Key Integration**: `scan_box.shifts` bridge alignment → detection

---

## Integration with Drawing/Visualization

### Drawing Shifted Positions

**File**: `src/processors/layout/field_block/drawing.py` (inferred)

```python
class FieldBlockDrawing:
    def draw_field_block(self, image, shifted=False, thickness=2):
        for field in self.field_block.fields:
            for scan_box in field.scan_boxes:
                x, y = scan_box.position

                if shifted:
                    # Apply alignment shifts
                    x += scan_box.shifts[0]
                    y += scan_box.shifts[1]

                # Draw bubble at shifted position
                cv2.circle(image, (x, y), radius, color, thickness)
```

**Usage**:
```python
# Draw template positions (before alignment):
field_block.drawing.draw_field_block(image, shifted=False)

# Draw aligned positions (after alignment):
field_block.drawing.draw_field_block(image, shifted=True)
```

---

## Integration with Configuration System

### Tuning Config

**File**: `src/schemas/models/config.py` (inferred)

```python
class TuningConfig:
    class OutputsConfig:
        show_image_level: int  # Controls visualization verbosity
        # 0: No images
        # 2: Before/after field blocks
        # 5: Triangle warping visualization
        # 6: Full SIFT match visualization
```

**Integration**:
```python
# AlignmentProcessor receives tuning_config:
def __init__(self, template):
    self.template = template
    self.tuning_config = template.tuning_config

def process(self, context):
    apply_template_alignment(
        gray_image, colored_image, template,
        self.tuning_config  # Passed for visualization
    )
```

**Visualization Control**:
```python
# In k_nearest_interpolation.py:53
if config.outputs.show_image_level >= 2:
    # Show before/after field blocks

# In sift_matcher.py:119
if config.outputs.show_image_level >= 6:
    # Show SIFT feature matches
    display_feature_matches = cv2.drawMatches(...)
    InteractionUtils.show(...)
```

---

## Integration with Image Utilities

### ImageUtils Dependencies

**File**: `src/utils/image.py`

**Used Methods**:

1. **resize_to_dimensions**:
```python
# template_alignment.py:33-39
(gray_image, colored_image, gray_alignment_image, colored_alignment_image) = (
    ImageUtils.resize_to_dimensions(
        template.template_dimensions,
        gray_image, colored_image,
        gray_alignment_image, colored_alignment_image,
    )
)
```

2. **overlay_image**:
```python
# alignment/utils.py:12-14
overlay = ImageUtils.overlay_image(
    block_gray_alignment_image,
    block_gray_image
)
```

3. **get_padded_hstack**:
```python
# k_nearest_interpolation.py:82-84
InteractionUtils.show(
    f"Field Block shifts: {average_shifts}",
    ImageUtils.get_padded_hstack([
        block_gray_image_before,
        block_gray_image_after
    ])
)
```

---

## Integration with Math Utilities

### MathUtils Dependencies

**File**: `src/utils/math.py`

**Used Methods**:

1. **distance**:
```python
# sift_matcher.py:69
if MathUtils.distance(source_feature_point, destination_feature_point) <= max_displacement:
    good.append(m)
```

2. **subtract_points**:
```python
# k_nearest_interpolation.py:47
displacement = MathUtils.subtract_points(anchor_point, displaced_point)

# k_nearest_interpolation.py:50-51
block_image_shifts = MathUtils.subtract_points(
    [margins["left"], margins["top"]],
    field_block.origin
)
```

3. **add_points**:
```python
# k_nearest_interpolation.py:113
field_block_position = MathUtils.add_points(
    field_block.origin,
    block_image_shifts
)
```

4. **rectangle_contains**:
```python
# piecewise_affine_delaunay.py:51-52
if MathUtils.rectangle_contains(source_point, warped_rectangle) and
   MathUtils.rectangle_contains(destination_point, warped_rectangle):
    parsed_displacement_pairs.append([source_point, destination_point])
```

---

## Integration with Interaction Utilities

### InteractionUtils Dependencies

**File**: `src/utils/interaction.py`

**Used Method**:
```python
# sift_matcher.py:138-140
InteractionUtils.show(
    f"Matches for {field_block_name}",
    display_feature_matches,
    0  # waitKey timeout
)

# alignment/utils.py:17-28
InteractionUtils.show(
    "Alignment For Field Block",
    ImageUtils.get_padded_hstack([...]),
)
```

**Purpose**: Display debug visualizations (non-blocking for batch mode)

---

## Integration with Logging System

### Logger Integration

**File**: `src/utils/logger.py`

**Usage Patterns**:

1. **Debug Level** (verbose details):
```python
# template_alignment.py:69-76
logger.debug(
    field_block.name,
    field_block.to_json(),
    gray_image.shape,
    zone_start,
    zone_end,
)
```

2. **Info Level** (important events):
```python
# phase_correlation.py:37
logger.info(field_block.name, field_block.shifts)
```

3. **Critical Level** (failures):
```python
# sift_matcher.py:114-116
logger.critical(
    f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}"
)
```

---

## Integration with Schema System

### Pydantic Models

**File**: `src/schemas/models/template.py`

**Dataclass Hierarchy**:
```python
@dataclass
class AlignmentMarginsConfig:
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            top=data.get("top", 0),
            bottom=data.get("bottom", 0),
            left=data.get("left", 0),
            right=data.get("right", 0),
        )

@dataclass
class AlignmentConfig:
    margins: AlignmentMarginsConfig
    max_displacement: int = 10

    @classmethod
    def from_dict(cls, data: dict):
        margins_data = data.get("margins", {})
        return cls(
            margins=AlignmentMarginsConfig.from_dict(margins_data),
            max_displacement=data.get("max_displacement", 10),
        )

@dataclass
class TemplateConfig:
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    # ... other fields

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            alignment=AlignmentConfig.from_dict(data.get("alignment", {})),
            # ... other fields
        )
```

**Validation**: Pydantic ensures types match, defaults applied

---

## Integration with Exception System

### Error Handling

**File**: `src/exceptions.py`

**Used Exception**:
```python
# sift_matcher.py:6, 194-200
from src.utils.exceptions import ImageProcessingError

if warped_block_image.shape != block_gray_image.shape:
    raise ImageProcessingError(
        f"Warped block image shape {warped_block_image.shape} does not match...",
        context={
            "warped_shape": warped_block_image.shape,
            "original_shape": block_gray_image.shape,
        },
    )
```

**Error Types**:
- `ImageProcessingError`: Image operation failures
- `ConfigurationError`: Invalid alignment config (not used yet, but could be)

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ Template JSON                                           │
│ - alignment config                                      │
│ - field blocks                                          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────┐
│ Template Parser (from_dict)                             │
│ - AlignmentConfig.from_dict()                           │
│ - Load reference images                                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────┐
│ ProcessingContext                                       │
│ - gray_image (from pre-processors)                      │
│ - template (with alignment config)                      │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────┐
│ AlignmentProcessor.process()                            │
│ - Check for alignment image                             │
│ - Call apply_template_alignment()                       │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────┐
│ apply_template_alignment()                              │
│ - Resize images (ImageUtils)                            │
│ - For each field block:                                 │
│   - Extract ROI                                         │
│   - Call apply_k_nearest_interpolation_inplace()        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────┐
│ apply_k_nearest_interpolation_inplace()                 │
│ - SiftMatcher.get_matches()                             │
│ - find_k_nearest_anchors() (MathUtils)                  │
│ - shift_by_field_blocks()                               │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────┐
│ Update scan_box.shifts                                  │
│ - All scan boxes in field block                         │
│ - shifts = [dx, dy]                                     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────┐
│ Return updated context                                  │
│ - template now contains shifts                          │
│ - Pass to detection processors                          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────┐
│ Detection Processors                                    │
│ - Use scan_box.position + scan_box.shifts               │
│ - Detect bubbles at aligned positions                   │
└─────────────────────────────────────────────────────────┘
```

---

## File Dependencies Map

```
AlignmentProcessor (processor.py)
├── Template (src/processors/template/template.py)
│   ├── AlignmentConfig (src/schemas/models/template.py)
│   └── FieldBlock (src/processors/layout/field_block/)
│
├── apply_template_alignment (template_alignment.py)
│   ├── ImageUtils.resize_to_dimensions (src/utils/image.py)
│   └── apply_k_nearest_interpolation_inplace (k_nearest_interpolation.py)
│       ├── SiftMatcher.get_matches (sift_matcher.py)
│       │   ├── cv2.SIFT_create()
│       │   ├── cv2.FlannBasedMatcher()
│       │   └── MathUtils.distance (src/utils/math.py)
│       │
│       ├── find_k_nearest_anchors (k_nearest_interpolation.py)
│       │   └── MathUtils.distance (src/utils/math.py)
│       │
│       └── shift_by_field_blocks (k_nearest_interpolation.py)
│           └── MathUtils.add_points (src/utils/math.py)
│
├── ProcessingContext (src/processors/base.py)
│
├── logger (src/utils/logger.py)
└── InteractionUtils (src/utils/interaction.py)
```

---

## Browser Integration Considerations

### Template Loading in Browser

**Challenge**: No file system access

**Solution**:
```javascript
// Option 1: Embed alignment image as base64 in template JSON
{
  "alignment": {
    "referenceImageBase64": "data:image/png;base64,iVBORw0KG..."
  }
}

// Load in browser:
const img = new Image();
img.onload = () => {
  const mat = cv.imread(img);
  template.alignment.grayAlignmentImage = mat;
};
img.src = template.alignment.referenceImageBase64;

// Option 2: Fetch separately
const response = await fetch('template-reference.png');
const blob = await response.blob();
const img = await createImageBitmap(blob);
const mat = cv.matFromImageBitmap(img);
```

### Context Management in Browser

**Challenge**: No mutable shared state (Web Workers)

**Solution**:
```javascript
// Main thread:
const worker = new Worker('alignment-worker.js');

worker.postMessage({
  type: 'align',
  grayImage: grayImageData,  // ImageData
  template: templateJSON,
});

worker.onmessage = (event) => {
  const { shifts } = event.data;
  // Apply shifts to template
  template.fieldBlocks.forEach((block, i) => {
    block.shifts = shifts[i];
  });
};

// Worker:
onmessage = (event) => {
  const { grayImage, template } = event.data;
  const shifts = performAlignment(grayImage, template);
  postMessage({ shifts });
};
```

### Memory Management

**Critical**: Explicitly delete cv.Mat objects
```javascript
function alignFieldBlock(grayImage, alignmentImage) {
  const roi = grayImage.roi(rect);
  const keypoints = new cv.KeyPointVector();
  const descriptors = new cv.Mat();

  try {
    detector.detectAndCompute(roi, new cv.Mat(), keypoints, descriptors);
    // ... use keypoints and descriptors
  } finally {
    // ALWAYS clean up
    roi.delete();
    keypoints.delete();
    descriptors.delete();
  }
}
```

---

## API Contract

### Input Requirements

```python
# AlignmentProcessor.process() expects:
context.gray_image: np.ndarray          # Shape: (H, W), dtype: uint8
context.colored_image: np.ndarray       # Shape: (H, W, 3), dtype: uint8
context.template.alignment: dict        # Contains:
    # - "gray_alignment_image": np.ndarray
    # - "colored_alignment_image": np.ndarray
    # - "margins": dict {top, bottom, left, right}
    # - "max_displacement": int
context.template.field_blocks: list     # Each has:
    # - name: str
    # - origin: [int, int]
    # - bounding_box_origin: [int, int]
    # - bounding_box_dimensions: [int, int]
    # - alignment: dict (optional, overrides template)
    # - fields: list of Field objects
```

### Output Guarantees

```python
# AlignmentProcessor.process() returns:
context.template.field_blocks[i].shifts: list[int, int]  # For each field block
context.template.field_blocks[i].fields[j].scan_boxes[k].shifts: list[int, int]  # For each scan box

# Invariants:
# - If alignment succeeds: shifts = [dx, dy] where |dx|, |dy| <= max_displacement
# - If alignment fails: shifts = [0, 0]
# - shifts are always int32 (not float)
# - gray_image and colored_image unchanged (except resize)
```

---

## Testing Integration Points

### Unit Test Interfaces

```python
# Test AlignmentProcessor:
def test_alignment_processor():
    template = load_template("test_template.json")
    context = ProcessingContext(
        gray_image=load_image("scan.png"),
        colored_image=load_image("scan_color.png"),
        template=template,
    )

    processor = AlignmentProcessor(template)
    result_context = processor.process(context)

    # Assert shifts applied
    assert result_context.template.field_blocks[0].shifts != [0, 0]

# Test apply_k_nearest_interpolation_inplace:
def test_k_nearest_interpolation():
    field_block = create_test_field_block()
    gray_image = create_test_image()
    alignment_image = create_reference_image()

    shifts = apply_k_nearest_interpolation_inplace(
        field_block, gray_image, alignment_image,
        max_displacement=30, margins={...}, config=test_config
    )

    assert isinstance(shifts, np.ndarray)
    assert shifts.shape == (2,)
    assert abs(shifts[0]) <= 30 and abs(shifts[1]) <= 30
```

### Integration Test Points

1. **Pre-processor → Alignment**: Verify resized images compatible
2. **Alignment → Detection**: Verify shifts applied correctly
3. **Template Loading → Alignment**: Verify alignment config parsed
4. **Config → Visualization**: Verify show_image_level respected

---

## Related Integration Docs

- **Pipeline Integration**: See `modules/domain/pipeline/integration.md`
- **Template System**: See `modules/domain/template/integration.md`
- **Detection System**: See `modules/domain/detection/integration.md`
- **Context Management**: See `modules/domain/processing-context/integration.md`
