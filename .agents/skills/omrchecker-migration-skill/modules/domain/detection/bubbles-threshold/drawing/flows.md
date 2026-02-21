# Bubble Detection Drawing - Visual Feedback Flow

## Overview

The Bubble Detection Drawing module provides visual feedback by rendering bubble detection results onto the OMR sheet image. It supports two modes: **with verdicts** (evaluation mode with answer key) and **without verdicts** (detection-only mode).

**Python Reference**: `src/processors/detection/bubbles_threshold/interpretation_drawing.py`

---

## Core Concepts

### 1. BubblesFieldInterpretationDrawing Class

Extends the base `FieldInterpretationDrawing` class to provide bubble-specific visual rendering.

```python
class BubblesFieldInterpretationDrawing(FieldInterpretationDrawing):
    def __init__(self, field_interpretation) -> None:
        super().__init__(field_interpretation)
```

### 2. Drawing Modes

#### Mode 1: Without Verdicts (Detection-Only)
- Shows detected bubbles with filled gray boxes
- Shows undetected bubbles with hollow boxes
- Displays bubble values as text overlay

#### Mode 2: With Verdicts (Evaluation Mode)
- Color-coded boxes based on correctness
- Verdict symbols (+, -, o, *) inside bubbles
- Enhanced bounding box for expected answers
- Optional answer group indicators

---

## Drawing Flow: Main Entry Point

### `draw_field_interpretation()`

**Entry point** for rendering a single field's bubble interpretations.

```python
def draw_field_interpretation(
    self, marked_image, image_type, evaluation_meta, evaluation_config_for_response
) -> None:
    field_label = self.field.field_label
    bubble_interpretations = self.field_interpretation.bubble_interpretations

    # Determine if verdicts should be drawn
    should_draw_question_verdicts = (
        evaluation_meta is not None and
        evaluation_config_for_response is not None
    )
    question_has_verdict = (
        evaluation_meta is not None and
        field_label in evaluation_meta["questions_meta"]
    )

    # Route to appropriate drawing method
    if should_draw_question_verdicts and question_has_verdict and
       evaluation_config_for_response.draw_question_verdicts["enabled"]:
        # Draw with evaluation verdicts
        question_meta = evaluation_meta["questions_meta"][field_label]
        self.draw_bubbles_and_detections_with_verdicts(...)
    else:
        # Draw without verdicts (detection-only)
        self.draw_bubbles_and_detections_without_verdicts(...)
```

**Flow Diagram**:
```
draw_field_interpretation()
  │
  ├─► Check evaluation_meta exists
  ├─► Check evaluation_config_for_response exists
  ├─► Check field_label in questions_meta
  ├─► Check draw_question_verdicts enabled
  │
  ├─► YES: draw_bubbles_and_detections_with_verdicts()
  │        ├─► For each bubble: draw_unit_bubble_interpretation_with_verdicts()
  │        └─► If enabled: draw_answer_groups_for_bubbles()
  │
  └─► NO: draw_bubbles_and_detections_without_verdicts()
           └─► For each bubble: Draw simple filled/hollow box
```

---

## Drawing Without Verdicts (Detection-Only Mode)

### `draw_bubbles_and_detections_without_verdicts()`

Renders bubbles with minimal visual feedback for detection results.

```python
@staticmethod
def draw_bubbles_and_detections_without_verdicts(
    marked_image,
    bubble_interpretations: list[BubbleInterpretation],
    evaluation_config_for_response: EvaluationConfigForSet,
) -> None:
    for bubble_interpretation in bubble_interpretations:
        bubble = bubble_interpretation.item_reference
        bubble_dimensions = bubble.dimensions
        shifted_position = tuple(bubble.get_shifted_position())
        bubble_value = str(bubble.bubble_value)

        if bubble_interpretation.is_attempted:
            # Draw filled gray box for detected bubbles
            DrawingUtils.draw_box(
                marked_image, shifted_position, bubble_dimensions,
                color=CLR_GRAY, style="BOX_FILLED", thickness_factor=1/12
            )

            # Draw bubble value text (if enabled)
            if evaluation_config_for_response is None or
               evaluation_config_for_response.draw_detected_bubble_texts["enabled"]:
                DrawingUtils.draw_text(
                    marked_image, bubble_value, shifted_position,
                    text_size=TEXT_SIZE, color=CLR_NEAR_BLACK,
                    thickness=int(1 + 3.5 * TEXT_SIZE)
                )
        else:
            # Draw hollow box for undetected bubbles
            DrawingUtils.draw_box(
                marked_image, shifted_position, bubble_dimensions,
                style="BOX_HOLLOW", thickness_factor=1/10
            )
```

**Visual Output**:
```
┌─────┐  ┌─────┐  ┏━━━━━┓  ┌─────┐
│  A  │  │  B  │  ┃  C  ┃  │  D  │   ← Detected: C (filled gray box)
└─────┘  └─────┘  ┗━━━━━┛  └─────┘   ← Undetected: A, B, D (hollow)
```

**Configuration**:
- `draw_detected_bubble_texts.enabled`: Controls text overlay (default: true)
- `CLR_GRAY = (130, 130, 130)`: Fill color for detected bubbles
- `CLR_NEAR_BLACK = (20, 20, 10)`: Text color
- `TEXT_SIZE = 0.95`: Font size multiplier

---

## Drawing With Verdicts (Evaluation Mode)

### `draw_bubbles_and_detections_with_verdicts()`

Orchestrates verdict-based rendering for all bubbles in a field.

```python
@staticmethod
def draw_bubbles_and_detections_with_verdicts(
    marked_image, image_type,
    bubble_interpretations: list[BubbleInterpretation],
    question_meta, evaluation_config_for_response,
) -> None:
    # Draw each bubble with verdict
    for bubble_interpretation in bubble_interpretations:
        BubblesFieldInterpretationDrawing.draw_unit_bubble_interpretation_with_verdicts(
            bubble_interpretation, marked_image,
            evaluation_config_for_response, question_meta, image_type
        )

    # Draw answer group indicators (optional)
    if evaluation_config_for_response.draw_answer_groups["enabled"]:
        BubblesFieldInterpretationDrawing.draw_answer_groups_for_bubbles(
            marked_image, image_type, question_meta,
            bubble_interpretations, evaluation_config_for_response
        )
```

**Flow**:
1. Draw each bubble with verdict symbols and colors
2. Optionally draw answer group edge indicators

---

## Drawing Individual Bubble With Verdict

### `draw_unit_bubble_interpretation_with_verdicts()`

Core rendering logic for a single bubble in evaluation mode.

```python
@staticmethod
def draw_unit_bubble_interpretation_with_verdicts(
    bubble_interpretation, marked_image,
    evaluation_config_for_response, question_meta, image_type,
    thickness_factor=1/12,
) -> None:
    bonus_type = question_meta["bonus_type"]
    bubble = bubble_interpretation.item_reference
    bubble_dimensions = bubble.dimensions
    shifted_position = tuple(bubble.get_shifted_position())
    bubble_value = str(bubble.bubble_value)

    # Step 1: Draw enhanced box for expected answer
    if AnswerMatcher.is_part_of_some_answer(question_meta, bubble_value):
        DrawingUtils.draw_box(
            marked_image, shifted_position, bubble_dimensions,
            CLR_BLACK, style="BOX_HOLLOW", thickness_factor=0
        )

    # Step 2: Draw filled box with verdict color (for marked or bonus)
    if bubble_interpretation.is_attempted or bonus_type is not None:
        # Get verdict colors and symbols
        (verdict_symbol, verdict_color, verdict_symbol_color, thickness_factor) =
            evaluation_config_for_response.get_evaluation_meta_for_question(
                question_meta, bubble_interpretation.is_attempted, image_type
            )

        # Draw colored box
        if verdict_color != "":
            position, position_diagonal = DrawingUtils.draw_box(
                marked_image, shifted_position, bubble_dimensions,
                color=verdict_color, style="BOX_FILLED",
                thickness_factor=thickness_factor
            )

            # Draw verdict symbol (+, -, o, *)
            if verdict_symbol != "":
                DrawingUtils.draw_symbol(
                    marked_image, verdict_symbol,
                    position, position_diagonal,
                    color=verdict_symbol_color
                )

        # Draw bubble value text
        if (bubble_interpretation.is_attempted and
            evaluation_config_for_response.draw_detected_bubble_texts["enabled"]):
            DrawingUtils.draw_text(
                marked_image, bubble_value, shifted_position,
                text_size=TEXT_SIZE, color=CLR_NEAR_BLACK,
                thickness=int(1 + 3.5 * TEXT_SIZE)
            )
    else:
        # Step 3: Draw hollow box for unmarked bubbles
        DrawingUtils.draw_box(
            marked_image, shifted_position, bubble_dimensions,
            style="BOX_HOLLOW", thickness_factor=1/10
        )
```

**Rendering Steps**:
```
1. Enhanced Box (if expected answer)
   ┏━━━━━┓  ← Thick black border
   ┃     ┃
   ┗━━━━━┛

2. Verdict Box + Symbol (if marked/bonus)
   ┏━━━━━┓
   ┃  +  ┃  ← Green fill + plus symbol (correct)
   ┗━━━━━┛

   ┏━━━━━┓
   ┃  -  ┃  ← Red fill + minus symbol (incorrect)
   ┗━━━━━┛

   ┏━━━━━┓
   ┃  o  ┃  ← Neutral fill + circle symbol (neutral)
   ┗━━━━━┛

   ┏━━━━━┓
   ┃  *  ┃  ← Bonus fill + star symbol (bonus)
   ┗━━━━━┛

3. Hollow Box (if unmarked)
   ┌─────┐
   │     │
   └─────┘
```

---

## Verdict Colors and Symbols

### Color Mapping

Colors are configured in `evaluation_config_for_response` and depend on `question_meta` verdict.

**Verdict Colors** (from `verdict_colors`):
```python
{
    "correct": (100, 200, 100),    # CLR_GREEN-like
    "incorrect": (20, 20, 255),    # CLR_DARK_RED
    "neutral": (130, 130, 130),    # CLR_GRAY (or inherit from incorrect)
    "bonus": (255, 165, 0),        # Orange-like
}
```

**Symbol Colors** (from `verdict_symbol_colors`):
```python
{
    "positive": (0, 200, 0),       # Green
    "negative": (0, 0, 255),       # Red
    "neutral": (100, 100, 100),    # Gray
    "bonus": (255, 200, 0),        # Yellow/Orange
}
```

**Grayscale Mode**:
```python
if image_type == "GRAYSCALE":
    color = CLR_WHITE         # (255, 255, 255)
    symbol_color = CLR_BLACK  # (0, 0, 0)
```

### Symbol Mapping

Symbols are determined by `delta` (score change) and `bonus_type`:

| Condition | Symbol | Meaning |
|-----------|--------|---------|
| `delta > 0` | `+` | Positive (correct) |
| `delta < 0` | `-` | Negative (incorrect) |
| `delta == 0` | `o` | Neutral |
| `bonus_type == "BONUS_FOR_ALL"` (unmarked) | `+` | Bonus awarded to all |
| `bonus_type == "BONUS_ON_ATTEMPT"` (unmarked, schema verdict unmarked) | `o` | Neutral (no attempt) |
| `bonus_type == "BONUS_ON_ATTEMPT"` (unmarked, schema verdict marked) | `*` | Bonus on attempt |

---

## Answer Group Indicators

### `draw_answer_groups_for_bubbles()`

Draws colored edge indicators for questions with multiple correct answers.

```python
@staticmethod
def draw_answer_groups_for_bubbles(
    marked_image, image_type, question_meta,
    bubble_interpretations: list[BubbleInterpretation],
    evaluation_config_for_response: EvaluationConfigForSet,
) -> None:
    answer_type = question_meta["answer_type"]
    if answer_type == AnswerType.STANDARD:
        return  # No grouping for standard questions

    box_edges = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
    color_sequence = evaluation_config_for_response.draw_answer_groups["color_sequence"]

    # Override colors for grayscale
    if image_type == "GRAYSCALE":
        color_sequence = [CLR_WHITE] * len(color_sequence)

    # Draw group indicator for each bubble
    for bubble_interpretation in bubble_interpretations:
        bubble = bubble_interpretation.item_reference
        bubble_dimensions = bubble.dimensions
        shifted_position = tuple(bubble.get_shifted_position())
        bubble_value = str(bubble.bubble_value)

        # Find which answer groups this bubble belongs to
        matched_groups = AnswerMatcher.get_matched_answer_groups(
            question_meta, bubble_value
        )

        # Draw edge indicator for each matched group
        for answer_index in matched_groups:
            box_edge = box_edges[answer_index % 4]
            color = color_sequence[answer_index % 4]
            DrawingUtils.draw_group(
                marked_image, shifted_position,
                bubble_dimensions, box_edge, color
            )
```

**Visual Example** (Multiple Correct Answers: ['A', 'B', 'AB']):
```
Group 0: A   → TOP edge (Red)
Group 1: B   → RIGHT edge (Blue)
Group 2: AB  → BOTTOM edge (Green)

┏━━━━━┓  ┌─────┐  ┌─────┐  ┌─────┐
┃  A  ┃  │  B ││  │  C  │  │  D  │
┗━━━━━┛  └────┘│  └─────┘  └─────┘
               │
        └──────┘

Red top edge: A is correct (group 0)
Blue right edge: B is correct (group 1)
AB would have green bottom edge (group 2)
```

**Configuration**:
```python
draw_answer_groups = {
    "enabled": True,
    "color_sequence": [
        (255, 0, 0),    # Red - Group 0
        (0, 0, 255),    # Blue - Group 1
        (0, 255, 0),    # Green - Group 2
        (255, 255, 0),  # Yellow - Group 3
    ]
}
```

**Limitations**:
- Only supports up to 4 answer groups (uses 4 edges: TOP, RIGHT, BOTTOM, LEFT)
- Only applies to `MULTIPLE_CORRECT` and `MULTIPLE_CORRECT_WEIGHTED` answer types

---

## DrawingUtils Integration

### Box Drawing

```python
DrawingUtils.draw_box(
    image, position, box_dimensions,
    color=CLR_GRAY, style="BOX_FILLED", thickness_factor=1/12
)
```

**Styles**:
- `BOX_HOLLOW`: Outline only (default color: `CLR_GRAY`)
- `BOX_FILLED`: Filled rectangle (default color: `CLR_DARK_GRAY`, border=-1)

**Thickness Factor**:
- Applied to create inset: `position = (x + w * thickness_factor, y + h * thickness_factor)`
- `thickness_factor=0`: Full box (no inset)
- `thickness_factor=1/12`: Standard inset
- `thickness_factor=1/10`: Slightly larger inset for hollow boxes

### Text Drawing

```python
DrawingUtils.draw_text(
    image, text_value, position,
    text_size=TEXT_SIZE, thickness=2,
    color=CLR_BLACK, line_type=cv2.LINE_AA
)
```

**Parameters**:
- `text_size`: Font scale (default: `TEXT_SIZE = 0.95`)
- `thickness`: Line thickness (calculated as `int(1 + 3.5 * TEXT_SIZE)` for bubble values)
- `font_face`: Default is `cv2.FONT_HERSHEY_SIMPLEX`
- `line_type`: `cv2.LINE_AA` for anti-aliased text

### Symbol Drawing

```python
DrawingUtils.draw_symbol(
    image, symbol, position, position_diagonal,
    color=CLR_BLACK
)
```

**Centering Logic**:
```python
center_position = (
    (position[0] + position_diagonal[0] - size_x) // 2,
    (position[1] + position_diagonal[1] + size_y) // 2
)
```

### Group Edge Drawing

```python
DrawingUtils.draw_group(
    image, start, bubble_dimensions, box_edge, color,
    thickness=3, thickness_factor=7/10
)
```

**Edge Positions**:
- `TOP`: Draws line on top edge from 30% to 70% of width
- `RIGHT`: Draws line on right edge from 30% to 70% of height
- `BOTTOM`: Draws line on bottom edge from 30% to 70% of width
- `LEFT`: Draws line on left edge from 30% to 70% of height

---

## Browser Migration

### 1. Canvas API Rendering

Replace OpenCV drawing with Canvas 2D context:

```typescript
// TypeScript interfaces
interface BubbleInterpretation {
    item_reference: Bubble;
    is_attempted: boolean;
}

interface Bubble {
    bubble_value: string;
    dimensions: [number, number];
    get_shifted_position(): [number, number];
}

// Drawing without verdicts
function drawBubblesWithoutVerdicts(
    ctx: CanvasRenderingContext2D,
    bubbleInterpretations: BubbleInterpretation[],
    config: EvaluationConfig | null
): void {
    for (const bubbleInterp of bubbleInterpretations) {
        const bubble = bubbleInterp.item_reference;
        const [x, y] = bubble.get_shifted_position();
        const [w, h] = bubble.dimensions;
        const value = bubble.bubble_value;

        if (bubbleInterp.is_attempted) {
            // Filled gray box
            const thicknessFactor = 1 / 12;
            const insetX = x + w * thicknessFactor;
            const insetY = y + h * thicknessFactor;
            const insetW = w * (1 - 2 * thicknessFactor);
            const insetH = h * (1 - 2 * thicknessFactor);

            ctx.fillStyle = 'rgb(130, 130, 130)'; // CLR_GRAY
            ctx.fillRect(insetX, insetY, insetW, insetH);

            // Draw text (if enabled)
            if (config === null || config.draw_detected_bubble_texts.enabled) {
                ctx.fillStyle = 'rgb(20, 20, 10)'; // CLR_NEAR_BLACK
                ctx.font = `${0.95 * 16}px sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(value, x + w / 2, y + h / 2);
            }
        } else {
            // Hollow box
            const thicknessFactor = 1 / 10;
            const insetX = x + w * thicknessFactor;
            const insetY = y + h * thicknessFactor;
            const insetW = w * (1 - 2 * thicknessFactor);
            const insetH = h * (1 - 2 * thicknessFactor);

            ctx.strokeStyle = 'rgb(130, 130, 130)';
            ctx.lineWidth = 3;
            ctx.strokeRect(insetX, insetY, insetW, insetH);
        }
    }
}
```

### 2. Verdict Drawing with Canvas

```typescript
function drawBubbleWithVerdict(
    ctx: CanvasRenderingContext2D,
    bubbleInterp: BubbleInterpretation,
    questionMeta: QuestionMeta,
    config: EvaluationConfig,
    imageType: 'COLOR' | 'GRAYSCALE'
): void {
    const bubble = bubbleInterp.item_reference;
    const [x, y] = bubble.get_shifted_position();
    const [w, h] = bubble.dimensions;
    const value = bubble.bubble_value;

    // Step 1: Enhanced box for expected answer
    if (AnswerMatcher.isPartOfSomeAnswer(questionMeta, value)) {
        ctx.strokeStyle = 'rgb(0, 0, 0)'; // CLR_BLACK
        ctx.lineWidth = 5;
        ctx.strokeRect(x, y, w, h);
    }

    // Step 2: Verdict box and symbol
    if (bubbleInterp.is_attempted || questionMeta.bonus_type !== null) {
        const { symbol, color, symbolColor } =
            config.getEvaluationMetaForQuestion(
                questionMeta, bubbleInterp.is_attempted, imageType
            );

        if (color) {
            // Draw filled box
            const thicknessFactor = 1 / 12;
            const insetX = x + w * thicknessFactor;
            const insetY = y + h * thicknessFactor;
            const insetW = w * (1 - 2 * thicknessFactor);
            const insetH = h * (1 - 2 * thicknessFactor);

            ctx.fillStyle = color;
            ctx.fillRect(insetX, insetY, insetW, insetH);

            // Draw symbol
            if (symbol) {
                ctx.fillStyle = symbolColor;
                ctx.font = `${0.95 * 20}px sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(symbol, x + w / 2, y + h / 2);
            }
        }

        // Draw bubble value
        if (bubbleInterp.is_attempted && config.draw_detected_bubble_texts.enabled) {
            ctx.fillStyle = 'rgb(20, 20, 10)';
            ctx.font = `${0.95 * 16}px sans-serif`;
            ctx.fillText(value, x + w / 2, y + h / 2);
        }
    } else {
        // Hollow box for unmarked
        ctx.strokeStyle = 'rgb(130, 130, 130)';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
    }
}
```

### 3. Answer Group Edge Drawing

```typescript
function drawAnswerGroups(
    ctx: CanvasRenderingContext2D,
    bubbleInterpretations: BubbleInterpretation[],
    questionMeta: QuestionMeta,
    config: EvaluationConfig,
    imageType: 'COLOR' | 'GRAYSCALE'
): void {
    if (questionMeta.answer_type === 'standard') return;

    const boxEdges = ['TOP', 'RIGHT', 'BOTTOM', 'LEFT'];
    let colorSequence = config.draw_answer_groups.color_sequence;

    if (imageType === 'GRAYSCALE') {
        colorSequence = ['rgb(255, 255, 255)', 'rgb(255, 255, 255)',
                        'rgb(255, 255, 255)', 'rgb(255, 255, 255)'];
    }

    for (const bubbleInterp of bubbleInterpretations) {
        const bubble = bubbleInterp.item_reference;
        const [x, y] = bubble.get_shifted_position();
        const [w, h] = bubble.dimensions;
        const value = bubble.bubble_value;

        const matchedGroups = AnswerMatcher.getMatchedAnswerGroups(questionMeta, value);

        for (const answerIndex of matchedGroups) {
            const edge = boxEdges[answerIndex % 4];
            const color = colorSequence[answerIndex % 4];
            const thicknessFactor = 0.7;

            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.beginPath();

            switch (edge) {
                case 'TOP':
                    ctx.moveTo(x + w * (1 - thicknessFactor), y);
                    ctx.lineTo(x + w * thicknessFactor, y);
                    break;
                case 'RIGHT':
                    ctx.moveTo(x + w, y + h * (1 - thicknessFactor));
                    ctx.lineTo(x + w, y + h * thicknessFactor);
                    break;
                case 'BOTTOM':
                    ctx.moveTo(x + w * thicknessFactor, y + h);
                    ctx.lineTo(x + w * (1 - thicknessFactor), y + h);
                    break;
                case 'LEFT':
                    ctx.moveTo(x, y + h * thicknessFactor);
                    ctx.lineTo(x, y + h * (1 - thicknessFactor));
                    break;
            }

            ctx.stroke();
        }
    }
}
```

### 4. Color Utilities

```typescript
function rgbToBgr(rgb: [number, number, number]): string {
    return `rgb(${rgb[2]}, ${rgb[1]}, ${rgb[0]})`;
}

function hexToRgb(hex: string): string {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgb(${r}, ${g}, ${b})`;
}
```

---

## Configuration Reference

### Evaluation Config Drawing Options

```json
{
  "outputs_configuration": {
    "draw_detected_bubble_texts": {
      "enabled": true
    },
    "draw_question_verdicts": {
      "enabled": true,
      "verdict_colors": {
        "correct": "#64C864",
        "incorrect": "#FF6464",
        "neutral": "#828282",
        "bonus": "#FFA500"
      },
      "verdict_symbol_colors": {
        "positive": "#00C800",
        "negative": "#FF0000",
        "neutral": "#646464",
        "bonus": "#FFC800"
      }
    },
    "draw_answer_groups": {
      "enabled": true,
      "color_sequence": [
        "#FF0000",
        "#0000FF",
        "#00FF00",
        "#FFFF00"
      ]
    }
  }
}
```

---

## Summary

The Bubble Detection Drawing module provides:

1. **Two Drawing Modes**:
   - Detection-only (simple filled/hollow boxes)
   - Evaluation mode (color-coded verdicts with symbols)

2. **Visual Feedback Elements**:
   - Box styles (filled/hollow)
   - Verdict colors (correct/incorrect/neutral/bonus)
   - Verdict symbols (+/-/o/*)
   - Bubble value text overlay
   - Answer group edge indicators

3. **Browser Migration**:
   - Replace OpenCV with Canvas 2D API
   - Use `fillRect()` and `strokeRect()` for boxes
   - Use `fillText()` for text rendering
   - Implement color mapping utilities

4. **Performance Considerations**:
   - Drawing is stateless (no persistent state)
   - Uses DrawingUtils for reusable primitives
   - Supports grayscale optimization
   - Edge indicators limited to 4 groups
