# ProcessingContext Entity - Concept

**Module**: Domain / ProcessingContext
**Python Reference**: `src/processors/base.py:11-43`
**Last Updated**: 2026-02-20

---

## Overview

ProcessingContext is a dataclass that encapsulates all data flowing through the processing pipeline. It provides a unified interface for passing images, results, and metadata between processors without changing method signatures.

**Key Responsibilities**:
1. **Data Encapsulation**: Hold all processing inputs and outputs in one object
2. **Type Safety**: Provide typed fields for pipeline data
3. **Extensibility**: Support metadata dictionary for custom processor data
4. **State Management**: Track processing state across pipeline stages
5. **Immutability Pattern**: Processors return updated context (functional style)

---

## ProcessingContext Architecture

### Dataclass Definition

**Code Reference**: `src/processors/base.py:11-43`

```python
@dataclass
class ProcessingContext:
    """Context object that flows through all processors."""

    # Input data
    file_path: Path | str
    gray_image: MatLike
    colored_image: MatLike
    template: Any  # Template type (avoiding circular import)

    # Processing results (populated by processors)
    omr_response: dict[str, str] = field(default_factory=dict)
    is_multi_marked: bool = False
    field_id_to_interpretation: dict[str, Any] = field(default_factory=dict)

    # Evaluation results (populated by EvaluationProcessor)
    score: float = 0.0
    evaluation_meta: dict[str, Any] | None = None
    evaluation_config_for_response: Any = None
    default_answers_summary: str = ""

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
```

---

## Field Categories

### 1. Input Fields (Immutable)

**Set Once**: During context initialization

**Fields**:
- `file_path` (Path | str): Path to OMR image being processed
- `gray_image` (MatLike): Grayscale image (OpenCV Mat)
- `colored_image` (MatLike): Color image for visualization
- `template` (Template): Template object with layout and configuration

**Note**: Images may be mutated by preprocessors (rotation, cropping, etc.)

---

### 2. Detection Results (Mutable)

**Populated By**: ReadOMRProcessor

**Fields**:
- `omr_response` (dict[str, str]): Detected field values
  - Example: `{"Roll": "12345", "q1": "A", "q2": "B", "q3": "C"}`
- `is_multi_marked` (bool): Whether multiple bubbles marked in any field
- `field_id_to_interpretation` (dict[str, Any]): Per-field bubble interpretations
  - Example: `{"q1": [{"bubble": "A", "darkness": 0.85}], "q2": [...]}`

**Usage**: Main OMR detection output

---

### 3. Evaluation Results (Optional)

**Populated By**: EvaluationProcessor (if evaluation enabled)

**Fields**:
- `score` (float): Total score for this OMR sheet
- `evaluation_meta` (dict[str, Any]): Evaluation details
  - Correct/incorrect answers
  - Per-section scores
  - Marking scheme applied
- `evaluation_config_for_response` (EvaluationConfig): Selected evaluation config
- `default_answers_summary` (str): Summary of default answers used

**Use Case**: Grading OMR sheets against answer keys

---

### 4. Metadata (Extensible)

**Purpose**: Store custom processor data without polluting main fields

**Examples**:
```python
# Store intermediate preprocessing results
context.metadata["original_image_shape"] = image.shape
context.metadata["rotation_angle"] = 90
context.metadata["crop_coordinates"] = [x, y, w, h]

# Store detection aggregates
context.metadata["directory_level_interpretation_aggregates"] = aggregates
context.metadata["bubble_threshold_stats"] = stats

# Store ML model outputs
context.metadata["ml_bubble_detections"] = detections
context.metadata["field_block_shifts"] = shifts
```

**Type**: `dict[str, Any]` (flexible, but use with care)

---

## Context Flow Through Pipeline

### Pipeline Stages

**Code Reference**: `src/processors/pipeline.py:117-157`

```python
def process_file(self, file_path, gray_image, colored_image) -> ProcessingContext:
    # 1. Create initial context
    context = ProcessingContext(
        file_path=file_path,
        gray_image=gray_image,
        colored_image=colored_image,
        template=self.template,
    )

    # 2. Execute each processor in sequence
    for processor in self.processors:
        context = processor.process(context)  # Functional style

    # 3. Return final context with all results
    return context
```

### Stage 1: Preprocessing

**Processor**: PreprocessingCoordinator
**Input**: Original images
**Output**: Preprocessed images, updated template with shifts
**Mutations**:
- `context.gray_image` → rotated, cropped, filtered
- `context.colored_image` → same transformations (if enabled)
- `context.template.template_layout` → shifted field blocks

---

### Stage 2: Alignment

**Processor**: AlignmentProcessor
**Input**: Preprocessed images
**Output**: Aligned images
**Mutations**:
- `context.gray_image` → aligned to reference
- `context.colored_image` → aligned (if enabled)
- `context.metadata["alignment_successful"]` → True/False

---

### Stage 3: Detection

**Processor**: ReadOMRProcessor
**Input**: Aligned images
**Output**: OMR response and interpretations
**Mutations**:
- `context.omr_response` → detected field values
- `context.is_multi_marked` → multi-mark flag
- `context.field_id_to_interpretation` → bubble interpretations
- `context.metadata["directory_level_interpretation_aggregates"]` → aggregates

---

### Stage 4: Evaluation (Optional)

**Processor**: EvaluationProcessor
**Input**: OMR response
**Output**: Score and evaluation details
**Mutations**:
- `context.score` → calculated score
- `context.evaluation_meta` → grading details
- `context.evaluation_config_for_response` → selected config
- `context.default_answers_summary` → summary text

---

## Post-Init Validation

**Code Reference**: `src/processors/base.py:39-42`

```python
def __post_init__(self) -> None:
    """Convert file_path to string if it's a Path."""
    if isinstance(self.file_path, Path):
        self.file_path = str(self.file_path)
```

**Purpose**: Ensure file_path is always string for consistency

**Why Needed**: Some parts of codebase expect string, others expect Path

---

## Usage Patterns

### Pattern 1: Processor Implementation

```python
class CustomProcessor(Processor):
    def process(self, context: ProcessingContext) -> ProcessingContext:
        # 1. Read from context
        image = context.gray_image
        template = context.template

        # 2. Process
        result = do_custom_processing(image, template)

        # 3. Update context (optional: create new context)
        context.metadata["custom_result"] = result

        # 4. Return updated context
        return context
```

---

### Pattern 2: Accessing Results

```python
# After pipeline execution
context = template.process_file(file_path, gray_image, colored_image)

# Access detection results
omr_response = context.omr_response
is_multi_marked = context.is_multi_marked

# Access evaluation results (if enabled)
if context.score > 0:
    print(f"Score: {context.score}")
    print(f"Evaluation: {context.evaluation_meta}")

# Access metadata
if "alignment_successful" in context.metadata:
    print(f"Alignment: {context.metadata['alignment_successful']}")
```

---

### Pattern 3: Metadata Storage

```python
# Store intermediate results
context.metadata["preprocessing"] = {
    "rotation_applied": True,
    "rotation_angle": 90,
    "crop_applied": True,
    "crop_region": [100, 100, 800, 1200],
}

# Store aggregates for later use
context.metadata["bubble_stats"] = {
    "total_bubbles": 100,
    "marked_bubbles": 45,
    "confidence_mean": 0.87,
}
```

---

## Browser Migration Notes

### TypeScript ProcessingContext

```typescript
interface ProcessingContext {
    // Input data
    filePath: string;
    grayImage: cv.Mat;
    coloredImage: cv.Mat;
    template: Template;

    // Processing results
    omrResponse: Record<string, string>;
    isMultiMarked: boolean;
    fieldIdToInterpretation: Record<string, any>;

    // Evaluation results
    score: number;
    evaluationMeta: Record<string, any> | null;
    evaluationConfigForResponse: any | null;
    defaultAnswersSummary: string;

    // Metadata
    metadata: Record<string, any>;
}

function createProcessingContext(
    filePath: string,
    grayImage: cv.Mat,
    coloredImage: cv.Mat,
    template: Template
): ProcessingContext {
    return {
        filePath,
        grayImage,
        coloredImage,
        template,
        omrResponse: {},
        isMultiMarked: false,
        fieldIdToInterpretation: {},
        score: 0.0,
        evaluationMeta: null,
        evaluationConfigForResponse: null,
        defaultAnswersSummary: "",
        metadata: {}
    };
}
```

### Immutable Updates (Optional)

```typescript
// Functional style: return new context
function updateContext(
    context: ProcessingContext,
    updates: Partial<ProcessingContext>
): ProcessingContext {
    return { ...context, ...updates };
}

// Usage in processor
class CustomProcessor implements Processor {
    process(context: ProcessingContext): ProcessingContext {
        const result = doProcessing(context.grayImage);

        return updateContext(context, {
            metadata: {
                ...context.metadata,
                customResult: result
            }
        });
    }
}
```

### Zod Schema for Validation

```typescript
import { z } from 'zod';

const ProcessingContextSchema = z.object({
    filePath: z.string(),
    grayImage: z.any(), // cv.Mat (not serializable)
    coloredImage: z.any(),
    template: z.any(),
    omrResponse: z.record(z.string()).default({}),
    isMultiMarked: z.boolean().default(false),
    fieldIdToInterpretation: z.record(z.any()).default({}),
    score: z.number().default(0.0),
    evaluationMeta: z.record(z.any()).nullable().default(null),
    evaluationConfigForResponse: z.any().nullable().default(null),
    defaultAnswersSummary: z.string().default(""),
    metadata: z.record(z.any()).default({})
});
```

---

## Summary

**ProcessingContext**: Unified data container for pipeline
**Purpose**: Pass all data between processors without signature changes
**Categories**: Input, Detection Results, Evaluation, Metadata
**Pattern**: Functional style (processors return updated context)
**Extensibility**: Metadata dictionary for custom processor data

**Browser Migration**:
- Use TypeScript interface for type safety
- Consider immutable updates (functional style)
- Validate with Zod if needed
- Maintain same field structure

**Key Takeaway**: ProcessingContext is the backbone of the unified processor architecture. It eliminates tuple returns and provides clean, type-safe data flow through the pipeline.
