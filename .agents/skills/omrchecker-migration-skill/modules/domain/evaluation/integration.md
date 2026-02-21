# Evaluation Config - Integration

**Module**: Domain / Evaluation
**Python Reference**: `src/processors/evaluation/`
**Last Updated**: 2026-02-21

---

## Overview

This document details how the evaluation system integrates with other OMRChecker components and the broader processing pipeline.

---

## Integration Point 1: Pipeline Integration

### EvaluationProcessor in Pipeline

**Code Reference**: `src/processors/evaluation/processor.py:9-90`

**Pipeline Position**: After ReadOMR (detection), before output generation

**Processing Flow**:
```
Template.process_file()
└─> ProcessingPipeline.process_file()
    └─> Preprocessing (rotate, crop, align)
    └─> ReadOMR (detect bubbles)
    └─> EvaluationProcessor.process()  <-- Evaluation here
    └─> Output generation (CSV, images)
```

**Interface**:
```python
class EvaluationProcessor(Processor):
    def __init__(self, evaluation_config: EvaluationConfig) -> None:
        self.evaluation_config = evaluation_config

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # 1. Get OMR response from context
        concatenated_omr_response = context.omr_response
        file_path = context.file_path

        # 2. Get evaluation config for this response
        evaluation_config_for_response = (
            self.evaluation_config.get_evaluation_config_for_response(
                concatenated_omr_response, file_path
            )
        )

        # 3. Evaluate and score
        score, evaluation_meta = evaluate_concatenated_response(
            concatenated_omr_response, evaluation_config_for_response
        )

        # 4. Update context
        context.score = score
        context.evaluation_meta = evaluation_meta
        context.evaluation_config_for_response = evaluation_config_for_response

        return context
```

**Context Fields**:
- **Input**: `context.omr_response` (dict from ReadOMR)
- **Output**: `context.score`, `context.evaluation_meta`, `context.evaluation_config_for_response`

---

### Conditional Processor Creation

**Code Reference**: `src/processors/pipeline.py` (ProcessingPipeline initialization)

**Logic**:
```python
# In TemplateFileRunner.__init__
evaluation_config = None
if local_evaluation_path and local_evaluation_path.exists():
    evaluation_config = EvaluationConfig(
        curr_dir, local_evaluation_path, template, tuning_config
    )

# Only create EvaluationProcessor if evaluation.json exists
processors = [
    PreprocessingCoordinator(...),
    AlignmentProcessor(...),
    ReadOMRProcessor(...),
]

if evaluation_config:
    processors.append(EvaluationProcessor(evaluation_config))

pipeline = ProcessingPipeline(processors)
```

**Behavior**: If no evaluation.json, evaluation is skipped entirely

---

## Integration Point 2: Template Integration

### Template Provides

**1. Field Structure**:
- `template.field_blocks`: All field blocks
- `template.all_fields`: All fields (used to validate questions exist)

**2. Empty Value**:
- `template.global_empty_val`: Default empty value (e.g., "")
- Used in verdict determination (unmarked detection)

**3. Directory Management**:
- `template.get_evaluations_dir()`: Directory for evaluation outputs
- Used for explanation CSV export

**4. Image Processing** (for image-based answer keys):
- `template.process_file()`: Full pipeline for answer key image
- Returns `ProcessingContext` with `omr_response`

---

### Integration Example: Image-Based Answer Key

**Code Reference**: `evaluation_config_for_set.py:166-232`

```python
# In parse_csv_question_answers
image_path = curr_dir.joinpath(answer_key_image_path)

# Read image
gray_image, colored_image = ImageUtils.read_image_util(image_path, tuning_config)

# Process through template (full pipeline)
context = template.process_file(image_path, gray_image, colored_image)
concatenated_omr_response = context.omr_response

# Extract answers
answers_in_order = [concatenated_omr_response[q] for q in questions_in_order]
```

**Dependencies**: Template preprocessing, alignment, detection all run on answer key image

---

## Integration Point 3: TuningConfig Integration

### TuningConfig Provides

**1. Multi-Mark Filtering**:
```python
# In validate_answers
if tuning_config.outputs.filter_out_multimarked_files:
    # Check answer key for multi-marked answers
    # Raise ConfigError if found
```

**2. Image Processing Settings**:
- Used when processing answer key image
- Controls preprocessing, alignment, detection thresholds

---

## Integration Point 4: Output Integration

### Visual Outputs on Images

**Code Reference**: `src/processors/template/template_drawing.py`

**Draw Score**:
```python
# In template.draw_final_score
if evaluation_config_for_response.draw_score["enabled"]:
    score_format, position, size, thickness = (
        evaluation_config_for_response.get_formatted_score(score)
    )
    ImageUtils.draw_text(
        image, score_format, position, size, thickness, color=CLR_BLACK
    )
```

**Draw Answers Summary**:
```python
# In template.draw_answers_summary
if evaluation_config_for_response.draw_answers_summary["enabled"]:
    answers_format, position, size, thickness = (
        evaluation_config_for_response.get_formatted_answers_summary()
    )
    ImageUtils.draw_text(
        image, answers_format, position, size, thickness, color=CLR_BLACK
    )
```

**Draw Question Verdicts**:
```python
# In template.draw_question_verdicts
if evaluation_config_for_response.draw_question_verdicts["enabled"]:
    for field in template.all_fields:
        question_meta = evaluation_meta["questions_meta"].get(field.field_id)
        if question_meta:
            symbol, color, symbol_color, thickness_factor = (
                evaluation_config_for_response.get_evaluation_meta_for_question(
                    question_meta, is_field_marked, image_type
                )
            )
            # Draw symbols and colors on bubbles
```

**Integration**: Evaluation metadata drives visual rendering

---

### CSV Output Integration

**Code Reference**: `src/processors/template/template.py` (export_omr_results)

**CSV Columns**:
- All template output_columns (Roll, Name, q1, q2, ...)
- `score` column (if evaluation enabled)

**Example**:
```csv
Roll,Name,q1,q2,q3,q4,q5,score
12345,John,A,B,C,D,E,24.0
12346,Jane,B,C,A,D,E,21.0
```

**Integration**: `context.score` written to CSV

---

### Metrics Export Integration

**Code Reference**: `src/processors/template/template.py:170-251`

**ImageMetrics JavaScript File**:
```javascript
// outputs/ImageMetrics/metrics-filename.js
const templateMeta = {
    // From ReadOMR
    bubbleInterpretations: [...],
    multiMarkedFieldIds: [...],
    // From Evaluation
    score: 24.0,
    evaluationMeta: {
        questions_meta: {
            q1: { verdict: "ANSWER_MATCH", delta: 3.0, ... },
            q2: { verdict: "NO_ANSWER_MATCH", delta: -1.0, ... }
        }
    }
};
```

**Usage**: HTML visualization dashboard (interactive bubble inspection with scores)

---

## Integration Point 5: Error Handling Integration

### Exception Hierarchy

**Code Reference**: `src/exceptions.py`

**Evaluation-Specific Exceptions**:
- `ConfigError`: Invalid evaluation.json configuration
- `EvaluationError`: Evaluation logic errors (missing questions, etc.)
- `InputFileNotFoundError`: Answer key CSV/image not found
- `ImageReadError`: Answer key image unreadable

**Integration with Pipeline**:
```python
# In TemplateFileRunner.process_file
try:
    context = template.process_file(...)
except ConfigError as e:
    logger.error(f"Configuration error: {e}")
    # Skip file, continue processing
except EvaluationError as e:
    logger.error(f"Evaluation error: {e}")
    # Skip file, continue processing
```

**Error Context**: All evaluation errors include rich context (question names, file paths, etc.)

---

## Integration Point 6: Logging Integration

### Logger Integration

**Code Reference**: `src/utils/logger.py`

**Evaluation Logging**:
```python
# Info level
logger.info(f"Read Response: \n{concatenated_omr_response}")

# Debug level
logger.debug("evaluation_json_for_set", set_name, evaluation_json_for_set)
logger.debug("merged_evaluation_json", merged_evaluation_json)

# Warning level
logger.warning(f"No answer given for potential questions: {missing_prefixed_questions}")

# Error level
logger.error(f"Found empty answers for the questions: {empty_answered_questions}")

# Critical level
logger.critical(f"Missing OMR response for: {missing_questions}")
```

**Rich Console Output**: Explanation table printed via Rich library

---

## Integration Point 7: Schema Validation Integration

### JSON Schema Validation

**Code Reference**: `src/schemas/evaluation_schema.py`

**Validation Flow**:
```python
# In open_evaluation_with_defaults
def open_evaluation_with_defaults(evaluation_path):
    # 1. Load JSON
    evaluation_json = load_json(evaluation_path)

    # 2. Validate against schema
    validate_json_against_schema(evaluation_json, EVALUATION_SCHEMA)

    # 3. Merge with defaults
    merged = OVERRIDE_MERGER.merge(EVALUATION_CONFIG_DEFAULTS, evaluation_json)

    return merged
```

**Integration**: Schema validation catches config errors early (before pipeline starts)

---

## Integration Point 8: Field Parsing Integration

### Field String Parsing

**Code Reference**: `src/utils/parsing.py:parse_fields`

**Usage in Evaluation**:
```python
# Parse questions_in_order
questions_in_order = parse_fields("questions_in_order", ["q1..10", "q15..20"])
# Returns: ["q1", "q2", ..., "q10", "q15", "q16", ..., "q20"]

# Parse section questions
section_questions = parse_fields("SECTION_A", ["q1..5", "q8", "q10..12"])
# Returns: ["q1", "q2", "q3", "q4", "q5", "q8", "q10", "q11", "q12"]
```

**Integration**: Reuses template field parsing utilities

---

## Integration Point 9: Image Utils Integration

### ImageUtils for Answer Key Images

**Code Reference**: `src/utils/image.py`

**Usage**:
```python
# Read answer key image
gray_image, colored_image = ImageUtils.read_image_util(image_path, tuning_config)

# Draw score on image
ImageUtils.draw_text(image, score_text, position, size, thickness, color)
```

**Integration**: Reuses template image utilities

---

## Integration Point 10: File Organization Integration

### Exclude Files from Processing

**Code Reference**: `src/entry.py` (file discovery loop)

**Integration**:
```python
# Get exclude files from evaluation config
exclude_files = []
if evaluation_config:
    exclude_files = evaluation_config.get_exclude_files()

# During file discovery
for file_path in all_input_files:
    if file_path in exclude_files:
        logger.info(f"Skipping answer key image: {file_path}")
        continue  # Don't process answer key as student sheet
```

**Purpose**: Prevent answer key images from being processed as student sheets

---

## Data Flow Diagram

```
┌─────────────────┐
│ evaluation.json │
└────────┬────────┘
         │
         v
┌─────────────────────┐
│ EvaluationConfig    │ <--- Template (for image-based keys)
│   - Load JSON       │ <--- TuningConfig (for validation)
│   - Parse questions │
│   - Parse answers   │
│   - Build matchers  │
└────────┬────────────┘
         │
         v
┌─────────────────────────────┐
│ ProcessingPipeline          │
│   1. Preprocessing          │
│   2. Alignment              │
│   3. Detection (ReadOMR)    │
│   4. Evaluation ────────────┼─> EvaluationProcessor
│      - Get config for file  │       │
│      - Match answers        │       │
│      - Calculate score      │       │
│   5. Output Generation      │       │
└─────────────┬───────────────┘       │
              │                       │
              v                       v
    ┌──────────────────┐    ┌──────────────────┐
    │ ProcessingContext│<───│ evaluation_meta   │
    │   - omr_response │    │   - score         │
    │   - score        │    │   - questions_meta│
    │   - eval_meta    │    └──────────────────┘
    └────────┬─────────┘
             │
             v
    ┌────────────────┐
    │ Outputs        │
    │  - CSV         │
    │  - Images      │
    │  - Metrics JS  │
    └────────────────┘
```

---

## Browser Integration Notes

### Module Dependencies

**Browser Architecture**:
```typescript
// evaluation/
├── EvaluationConfig.ts
├── EvaluationConfigForSet.ts
├── AnswerMatcher.ts
├── SectionMarkingScheme.ts
├── EvaluationMeta.ts
└── EvaluationProcessor.ts

// Integration with other modules
import { Template } from '../template/Template';
import { TuningConfig } from '../config/TuningConfig';
import { ProcessingContext } from '../processing-context/ProcessingContext';
import { Processor } from '../pipeline/Processor';
```

---

### Pipeline Integration (Browser)

```typescript
class ProcessingPipeline {
    private processors: Processor[] = [];

    constructor(template: Template, evaluationConfig?: EvaluationConfig) {
        this.processors = [
            new PreprocessingCoordinator(template),
            new AlignmentProcessor(template),
            new ReadOMRProcessor(template),
        ];

        if (evaluationConfig) {
            this.processors.push(new EvaluationProcessor(evaluationConfig));
        }
    }

    async processFile(
        filePath: string,
        grayImage: cv.Mat,
        coloredImage: cv.Mat
    ): Promise<ProcessingContext> {
        let context = new ProcessingContext(filePath, grayImage, coloredImage);

        for (const processor of this.processors) {
            context = await processor.process(context);
        }

        return context;
    }
}
```

---

### Output Integration (Browser)

```typescript
// CSV export
function exportCSV(results: ProcessingContext[]): void {
    const rows = results.map(ctx => ({
        ...ctx.omrResponse,
        score: ctx.score || 0
    }));

    const csv = Papa.unparse(rows, { header: true });
    downloadFile(csv, 'results.csv', 'text/csv');
}

// Metrics export
function exportMetrics(context: ProcessingContext, fileName: string): void {
    const metrics = {
        templateMeta: {
            bubbleInterpretations: context.fieldIdToInterpretation,
            multiMarkedFieldIds: context.multiMarkedFieldIds,
        },
        evaluationMeta: context.evaluationMeta
    };

    const js = `const templateMeta = ${JSON.stringify(metrics, null, 2)};`;
    downloadFile(js, `metrics-${fileName}.js`, 'application/javascript');
}

// Download helper
function downloadFile(content: string, filename: string, mimeType: string): void {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}
```

---

### Visual Output Integration (Browser)

```typescript
// Draw on Canvas
function drawEvaluationOutputs(
    canvas: HTMLCanvasElement,
    context: ProcessingContext,
    evaluationConfig: EvaluationConfigForSet
): void {
    const ctx = canvas.getContext('2d')!;

    // Draw score
    if (evaluationConfig.drawScore.enabled) {
        const { text, position, size } = evaluationConfig.getFormattedScore(context.score);
        ctx.font = `${size * 16}px Arial`;
        ctx.fillText(text, position[0], position[1]);
    }

    // Draw answers summary
    if (evaluationConfig.drawAnswersSummary.enabled) {
        const { text, position, size } = evaluationConfig.getFormattedAnswersSummary();
        ctx.font = `${size * 16}px Arial`;
        ctx.fillText(text, position[0], position[1]);
    }

    // Draw question verdicts (symbols on bubbles)
    // ... similar to Python implementation
}
```

---

## Summary

**Key Integration Points**:
1. **Pipeline**: EvaluationProcessor in processing pipeline after ReadOMR
2. **Template**: Empty value, field structure, directory management, image processing
3. **TuningConfig**: Multi-mark filtering, image processing settings
4. **Output**: Visual (images), tabular (CSV), metrics (JS)
5. **Error Handling**: Exception hierarchy, rich context
6. **Logging**: Rich console output, explanation tables
7. **Schema Validation**: JSON Schema at initialization
8. **Field Parsing**: Reuse template utilities
9. **Image Utils**: Reuse template image utilities
10. **File Organization**: Exclude answer key images

**Browser Migration**:
- Same pipeline architecture (processors in sequence)
- Same context-based data flow
- Same output formats (CSV, images, metrics)
- Canvas-based visual rendering instead of OpenCV drawing
- Blob/download instead of file writes
- Native fetch/File API instead of pathlib

**Key Takeaway**: Evaluation system is well-integrated with minimal coupling. Browser version maintains same architecture with browser-native I/O operations.
