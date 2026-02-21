# Drawing Utils Flows

**Module**: Domain - Utils - Drawing
**Python Reference**: `src/utils/drawing.py`
**Last Updated**: 2026-02-21

---

## Overview

DrawingUtils provides OpenCV-based drawing operations for visualization and debugging of OMR processing. All methods are static and support drawing shapes, text, arrows, and colored outputs on images.

**Primary Use Cases**:
- Debug visualization during preprocessing
- Colored output for matched templates
- Visual feedback for bubble detection
- Annotating field blocks and markers

---

## Core Drawing Operations

### 1. draw_text() - Text Rendering

```
START: DrawingUtils.draw_text(image, text_value, position, ...)
│
├─► STEP 1: Handle Centered Text
│   │
│   │ If centered=True:
│   │   │
│   │   ├─ Validate position is not callable
│   │   │  (centered requires concrete position)
│   │   │
│   │   │  If callable(position):
│   │   │    ├─► RAISE ImageProcessingError
│   │   │    └─► END (error)
│   │   │
│   │   ├─ Store original position
│   │   │  text_position = position
│   │   │
│   │   └─► Create centering function
│   │       def position(size_x, size_y):
│   │           return (
│   │               text_position[0] - size_x // 2,  # Center X
│   │               text_position[1] + size_y // 2   # Center Y
│   │           )
│   │
│   └─ Else: Continue with original position
│
├─► STEP 2: Calculate Position (if callable)
│   │
│   │ If callable(position):
│   │   │
│   │   ├─ Get text size
│   │   │  size_x, size_y = cv2.getTextSize(
│   │   │      text_value,
│   │   │      font_face,
│   │   │      text_size,
│   │   │      thickness
│   │   │  )[0]
│   │   │
│   │   └─► Call position function
│   │       position = position(size_x, size_y)
│   │
│   └─ Else: Use position directly
│
├─► STEP 3: Convert Position to Integers
│   │
│   position = (int(position[0]), int(position[1]))
│   │
│   │ OpenCV requires integer coordinates
│
└─► STEP 4: Draw Text
    │
    cv2.putText(
        image,
        text_value,
        position,
        font_face,        # Default: FONT_HERSHEY_SIMPLEX
        text_size,        # Default: TEXT_SIZE (0.95)
        color,            # Default: CLR_BLACK (0,0,0)
        thickness,        # Default: 2
        lineType=line_type # Default: LINE_AA (anti-aliased)
    )
    │
    └─► END (modifies image in-place)
```

**Parameters**:
```python
text_value: str             # Text to draw
position: tuple | callable  # (x, y) or function(size_x, size_y) -> (x, y)
text_size: float = 0.95     # Font scale
thickness: int = 2          # Text thickness in pixels
centered: bool = False      # Center text at position
color: tuple = CLR_BLACK    # BGR color tuple
line_type: int = LINE_AA    # Anti-aliased line type
font_face: int = FONT_HERSHEY_SIMPLEX
```

**Example Usage**:
```python
# Simple text
DrawingUtils.draw_text(image, "Q1", (10, 30))

# Centered text
DrawingUtils.draw_text(image, "A", (50, 50), centered=True)

# Callable position (bottom-right aligned)
def bottom_right(size_x, size_y):
    return (width - size_x - 10, height - 10)
DrawingUtils.draw_text(image, "Score: 85", bottom_right)
```

---

### 2. draw_text_responsive() - Boundary-Aware Text

```
START: DrawingUtils.draw_text_responsive(image, text, position, *args, **kwargs)
│
├─► STEP 1: Get Image Dimensions
│   │
│   h, w = image.shape[:2]
│
├─► STEP 2: Create Boundary-Aware Position Function
│   │
│   def text_position(size_x: int, size_y: int) -> tuple[int, int]:
│       │
│       ├─ Calculate overflow in X direction
│       │  overflow_x = max(0, position[0] + size_x - w)
│       │
│       ├─ Calculate overflow in Y direction
│       │  overflow_y = max(0, position[1] + size_y - h)
│       │
│       └─► Return adjusted position
│           return (
│               position[0] - overflow_x,  # Shift left if overflows right
│               position[1] - overflow_y   # Shift up if overflows bottom
│           )
│
└─► STEP 3: Draw Text with Adjusted Position
    │
    DrawingUtils.draw_text(image, text, text_position, *args, **kwargs)
    │
    └─► END
```

**Use Case**: Prevent text from being clipped at image boundaries

**Example**:
```python
# Text near right edge - automatically shifts left if needed
DrawingUtils.draw_text_responsive(
    image,
    "Long Text Here",
    (width - 10, 50)  # Would overflow, but gets adjusted
)
```

---

### 3. draw_box() - Rectangle Drawing

```
START: DrawingUtils.draw_box(image, position, box_dimensions, ...)
│
├─► STEP 1: Extract Parameters
│   │
│   x, y = position
│   box_w, box_h = box_dimensions
│
├─► STEP 2: Apply Thickness Factor (inset)
│   │
│   │ thickness_factor = 1/12 (default)
│   │ Creates inset rectangle within box
│   │
│   position = (
│       int(x + box_w * thickness_factor),      # Shift right
│       int(y + box_h * thickness_factor)       # Shift down
│   )
│   │
│   position_diagonal = (
│       int(x + box_w - box_w * thickness_factor),  # Shift left from far edge
│       int(y + box_h - box_h * thickness_factor)   # Shift up from far edge
│   )
│   │
│   │ Example: box_w=120, thickness_factor=1/12
│   │ Inset = 120 * (1/12) = 10 pixels on each side
│   │ Drawable area: 100x100 inside 120x120 box
│
├─► STEP 3: Apply Centering (if requested)
│   │
│   │ If centered=True:
│   │   │
│   │   ├─ Calculate centered positions
│   │   │  centered_position = [
│   │   │      (3 * position[0] - position_diagonal[0]) // 2,
│   │   │      (3 * position[1] - position_diagonal[1]) // 2
│   │   │  ]
│   │   │  # Shifts box toward top-left
│   │   │
│   │   └─► centered_diagonal = [
│   │       (position[0] + position_diagonal[0]) // 2,
│   │       (position[1] + position_diagonal[1]) // 2
│   │   ]
│   │   # Midpoint becomes new diagonal
│   │   │
│   │   position = centered_position
│   │   position_diagonal = centered_diagonal
│   │
│   └─ Else: Use calculated positions
│
├─► STEP 4: Determine Color and Border
│   │
│   │ If style == "BOX_HOLLOW":
│   │   color = CLR_GRAY (if not provided)
│   │   border = 3 (default)
│   │
│   │ Elif style == "BOX_FILLED":
│   │   color = CLR_DARK_GRAY (if not provided)
│   │   border = -1  # Fill rectangle
│   │
│   └─ Else: Use provided values
│
├─► STEP 5: Draw Rectangle
│   │
│   DrawingUtils.draw_box_diagonal(
│       image,
│       position,
│       position_diagonal,
│       color,
│       border
│   )
│   # Calls cv2.rectangle internally
│
└─► STEP 6: Return Positions
    │
    RETURN (position, position_diagonal)
    │
    └─► END
```

**Parameters**:
```python
position: tuple             # (x, y) top-left corner
box_dimensions: tuple       # (width, height)
color: tuple = None         # BGR color (auto-selected if None)
style: str = "BOX_HOLLOW"   # "BOX_HOLLOW" or "BOX_FILLED"
thickness_factor: float = 1/12  # Inset ratio
border: int = 3             # Border thickness (ignored if FILLED)
centered: bool = False      # Center the drawn box
```

**Visual Example**:
```
Original box: (100, 100) with dimensions (120, 120)

After thickness_factor=1/12 (inset 10px):
┌─────────────────────────┐ (100, 100) outer box
│  ┌─────────────────┐    │
│  │                 │    │ (110, 110) inner box start
│  │   DRAWN BOX     │    │
│  │   100x100       │    │ (210, 210) inner box end
│  │                 │    │
│  └─────────────────┘    │
└─────────────────────────┘ (220, 220) outer box end

Centered=True: Shifts entire box so center aligns with original position
```

---

### 4. draw_box_diagonal() - Low-Level Rectangle

```
START: DrawingUtils.draw_box_diagonal(image, position, position_diagonal, color, border)
│
├─► STEP 1: Convert to Integer Coordinates
│   │
│   start = [int(position[0]), int(position[1])]
│   end = [int(position_diagonal[0]), int(position_diagonal[1])]
│
└─► STEP 2: Draw Rectangle
    │
    cv2.rectangle(image, start, end, color, border)
    │
    │ border > 0: Hollow rectangle
    │ border = -1: Filled rectangle
    │
    └─► END (modifies image in-place)
```

**Use Case**: Direct rectangle drawing with diagonal corners

---

### 5. draw_contour() - Contour Drawing

```
START: DrawingUtils.draw_contour(image, contour, color, thickness)
│
├─► STEP 1: Validate Contour
│   │
│   │ If None in contour:
│   │   │
│   │   ├─► RAISE ImageProcessingError(
│   │   │       "Invalid contour provided",
│   │   │       context={"contour": str(contour)}
│   │   │   )
│   │   └─► END (error)
│   │
│   └─ Else: Continue
│
├─► STEP 2: Convert to Integer Points
│   │
│   contour = np.intp(contour)
│   # Ensures all points are integer type
│
└─► STEP 3: Draw Contour
    │
    cv2.drawContours(
        image,
        [contour],           # List of contours (single contour here)
        contourIdx=-1,       # Draw all contours in list
        color=color,         # Default: CLR_GREEN
        thickness=thickness  # Default: 2
    )
    │
    └─► END (modifies image in-place)
```

**Parameters**:
```python
contour: np.ndarray    # Contour points [[x1,y1], [x2,y2], ...]
color: tuple = CLR_GREEN
thickness: int = 2
```

**Use Case**: Drawing detected markers, page boundaries, field blocks

---

### 6. draw_line() - Line Drawing

```
START: DrawingUtils.draw_line(image, start, end, color, thickness)
│
└─► cv2.line(image, start, end, color, thickness)
    │
    └─► END (modifies image in-place)
```

**Parameters**:
```python
start: tuple           # (x1, y1)
end: tuple             # (x2, y2)
color: tuple = CLR_BLACK
thickness: int = 3
```

---

### 7. draw_polygon() - Polygon Drawing

```
START: DrawingUtils.draw_polygon(image, points, color, thickness, closed)
│
├─► STEP 1: Get Point Count
│   │
│   n = len(points)
│
└─► STEP 2: Draw Lines Between Consecutive Points
    │
    For i in range(n):
        │
        ├─ Check if should draw this line
        │  │
        │  │ If not closed and i == n-1:
        │  │   └─► Skip last line (don't close polygon)
        │  │
        │  └─ Else: Draw line
        │
        └─► DrawingUtils.draw_line(
                image,
                points[i % n],           # Current point
                points[(i + 1) % n],     # Next point (wraps)
                color,
                thickness
            )
    │
    └─► END
```

**Parameters**:
```python
points: list[tuple]    # [(x1,y1), (x2,y2), ...]
color: tuple = CLR_BLACK
thickness: int = 1
closed: bool = True    # Connect last point to first
```

**Use Case**: Drawing field boundaries, custom shapes

---

### 8. draw_arrows() - Arrow Drawing

```
START: DrawingUtils.draw_arrows(image, start_points, end_points, ...)
│
├─► STEP 1: Convert to Tuple Points
│   │
│   start_points = MathUtils.get_tuple_points(start_points)
│   end_points = MathUtils.get_tuple_points(end_points)
│   │
│   │ Converts numpy arrays to list of integer tuples
│   │ [(x1, y1), (x2, y2), ...]
│
├─► STEP 2: Draw Each Arrow
│   │
│   For (start_point, end_point) in zip(start_points, end_points):
│       │
│       image = cv2.arrowedLine(
│           image,
│           start_point,       # Arrow tail
│           end_point,         # Arrow head
│           color,             # Default: CLR_GREEN
│           thickness,         # Default: 2
│           line_type,         # Default: LINE_AA
│           tipLength=tip_length  # Default: 0.1
│       )
│
└─► STEP 3: Return Modified Image
    │
    RETURN image
    │
    └─► END
```

**Parameters**:
```python
start_points: list | np.ndarray  # Arrow tails
end_points: list | np.ndarray    # Arrow heads
color: tuple = CLR_GREEN
thickness: int = 2
line_type: int = cv2.LINE_AA
tip_length: float = 0.1          # Arrow tip size ratio
```

**Use Case**: Showing alignment vectors, warp directions

---

### 9. draw_matches() - Template Matching Visualization

```
START: DrawingUtils.draw_matches(image, from_points, warped_image, to_points)
│
├─► STEP 1: Create Horizontal Stack
│   │
│   horizontal_stack = ImageUtils.get_padded_hstack([image, warped_image])
│   │
│   │ Combines two images side-by-side with padding
│   │ Result: [image | warped_image]
│
├─► STEP 2: Get Image Width
│   │
│   _h, w = image.shape[:2]
│   # Need width to offset points in second image
│
├─► STEP 3: Convert Points to Tuples
│   │
│   from_points = MathUtils.get_tuple_points(from_points)
│   to_points = MathUtils.get_tuple_points(to_points)
│
└─► STEP 4: Draw Matching Lines
    │
    For (from_point, to_point) in zip(from_points, to_points):
        │
        horizontal_stack = cv2.line(
            horizontal_stack,
            from_point,                    # Point in left image
            (w + to_point[0], to_point[1]), # Point in right image (offset by w)
            color=CLR_GREEN,
            thickness=3
        )
    │
    RETURN horizontal_stack
    │
    └─► END
```

**Use Case**: Visualizing SIFT matches, alignment points

**Visual Example**:
```
┌──────────────┬──────────────┐
│              │              │
│  Original    │  Warped      │
│  Image       │  Image       │
│      •───────┼──────•       │  Green lines connect matching points
│         •────┼───•          │
│              │              │
└──────────────┴──────────────┘
```

---

### 10. draw_symbol() - Symbol Drawing

```
START: DrawingUtils.draw_symbol(image, symbol, position, position_diagonal, color)
│
├─► STEP 1: Create Centering Function
│   │
│   def center_position(size_x: int, size_y: int) -> tuple[int, int]:
│       │
│       └─► Calculate center between position and position_diagonal
│           return (
│               (position[0] + position_diagonal[0] - size_x) // 2,  # Center X
│               (position[1] + position_diagonal[1] + size_y) // 2   # Center Y
│           )
│
└─► STEP 2: Draw Text Centered
    │
    DrawingUtils.draw_text(image, symbol, center_position, color=color)
    │
    └─► END
```

**Use Case**: Drawing marks like "✓", "✗", "?" in boxes

---

### 11. draw_group() - Group Edge Marking

```
START: DrawingUtils.draw_group(image, start, bubble_dimensions, box_edge, color, ...)
│
├─► STEP 1: Extract Parameters
│   │
│   start_x, start_y = start
│   box_w, box_h = bubble_dimensions
│
├─► STEP 2: Calculate Line Position Based on Edge
│   │
│   │ If box_edge == "TOP":
│   │   │
│   │   end_position = (start_x + int(box_w * thickness_factor), start_y)
│   │   start = (start_x + int(box_w * (1 - thickness_factor)), start_y)
│   │   │
│   │   │ Draws horizontal line at top edge
│   │   │ Covers center 70% of width (default thickness_factor=7/10)
│   │   │
│   │   └─► DrawingUtils.draw_line(image, start, end_position, color, thickness)
│   │
│   │ Elif box_edge == "RIGHT":
│   │   │
│   │   start = (start_x + box_w, start_y)
│   │   end_position = (start_x, int(start_y + box_h * thickness_factor))
│   │   start = (start_x, int(start_y + box_h * (1 - thickness_factor)))
│   │   │
│   │   │ Draws vertical line at right edge
│   │   │ Note: Unusual logic - seems to reset start?
│   │   │
│   │   └─► DrawingUtils.draw_line(image, start, end_position, color, thickness)
│   │
│   │ Elif box_edge == "BOTTOM":
│   │   │
│   │   start = (start_x, start_y + box_h)
│   │   end_position = (int(start_x + box_w * thickness_factor), start_y)
│   │   start = (int(start_x + box_w * (1 - thickness_factor)), start_y)
│   │   │
│   │   │ Note: Sets start_y back to original (seems incorrect)
│   │   │
│   │   └─► DrawingUtils.draw_line(image, start, end_position, color, thickness)
│   │
│   │ Elif box_edge == "LEFT":
│   │   │
│   │   end_position = (start_x, int(start_y + box_h * thickness_factor))
│   │   start = (start_x, int(start_y + box_h * (1 - thickness_factor)))
│   │   │
│   │   │ Draws vertical line at left edge
│   │   │ Covers center 70% of height
│   │   │
│   │   └─► DrawingUtils.draw_line(image, start, end_position, color, thickness)
│   │
│   └─ Else: No drawing (invalid edge type)
│
└─► END
```

**Parameters**:
```python
start: tuple               # (x, y) bubble position
bubble_dimensions: tuple   # (width, height)
box_edge: str              # "TOP", "RIGHT", "BOTTOM", "LEFT"
color: tuple               # BGR color
thickness: int = 3         # Line thickness
thickness_factor: float = 7/10  # Line coverage ratio
```

**Use Case**: Marking groups of bubbles (e.g., multi-digit numbers)

**Note**: Implementation has some apparent bugs in RIGHT and BOTTOM cases where start position is reassigned incorrectly. May need verification in actual usage.

---

## Color Constants

**Defined in**: `src/utils/constants.py`

```python
CLR_BLACK = (0, 0, 0)
CLR_DARK_GRAY = (100, 100, 100)
CLR_DARK_BLUE = (255, 20, 20)      # Note: BGR format
CLR_DARK_GREEN = (20, 255, 20)
CLR_DARK_RED = (20, 20, 255)
CLR_GRAY = (130, 130, 130)
CLR_LIGHT_GRAY = (200, 200, 200)
CLR_GREEN = (100, 200, 100)
CLR_WHITE = (255, 255, 255)
```

**Important**: OpenCV uses BGR (Blue, Green, Red) order, not RGB!

---

## Integration Patterns

### Pattern 1: Debug Visualization

```python
# In preprocessing processor
if self.show_visualization:
    debug_image = colored_image.copy()

    # Draw detected markers
    for marker in markers:
        DrawingUtils.draw_contour(
            debug_image,
            marker.contour,
            CLR_GREEN,
            thickness=2
        )

    # Add labels
    DrawingUtils.draw_text(
        debug_image,
        f"Markers: {len(markers)}",
        (10, 30)
    )

    ImageUtils.save_img("debug_markers.jpg", debug_image)
```

### Pattern 2: Bubble Detection Feedback

```python
# Color-code bubbles by detection result
for bubble, result in zip(bubbles, results):
    if result.is_filled:
        color = CLR_GREEN  # Filled bubble
    elif result.is_multi_marked:
        color = CLR_DARK_RED  # Error
    else:
        color = CLR_GRAY  # Empty bubble

    DrawingUtils.draw_box(
        colored_image,
        bubble.position,
        bubble.dimensions,
        color=color,
        style="BOX_HOLLOW"
    )
```

### Pattern 3: Alignment Visualization

```python
# Show alignment matches
match_visualization = DrawingUtils.draw_matches(
    original_image,
    template_keypoints,
    warped_image,
    image_keypoints
)

# Add arrow overlays for warp vectors
DrawingUtils.draw_arrows(
    match_visualization,
    start_points=original_positions,
    end_points=warped_positions,
    color=CLR_DARK_BLUE,
    thickness=2
)
```

---

## Browser Migration

### Canvas API Implementation

```javascript
class DrawingUtils {
  // Text drawing
  static drawText(canvas, ctx, text, position, options = {}) {
    const {
      fontSize = 16,
      fontFamily = 'Arial',
      color = '#000000',
      centered = false,
      bold = false
    } = options;

    ctx.font = `${bold ? 'bold ' : ''}${fontSize}px ${fontFamily}`;
    ctx.fillStyle = color;

    let [x, y] = position;

    if (centered) {
      const metrics = ctx.measureText(text);
      x -= metrics.width / 2;
      y += fontSize / 2;
    }

    // Handle callable position
    if (typeof position === 'function') {
      const metrics = ctx.measureText(text);
      [x, y] = position(metrics.width, fontSize);
    }

    ctx.fillText(text, x, y);
  }

  // Rectangle drawing
  static drawBox(canvas, ctx, position, dimensions, options = {}) {
    const {
      color = '#808080',
      style = 'BOX_HOLLOW',
      lineWidth = 3,
      thicknessFactor = 1/12,
      centered = false
    } = options;

    const [x, y] = position;
    const [w, h] = dimensions;

    // Apply inset
    const insetX = w * thicknessFactor;
    const insetY = h * thicknessFactor;

    let rectX = x + insetX;
    let rectY = y + insetY;
    let rectW = w - 2 * insetX;
    let rectH = h - 2 * insetY;

    // Apply centering
    if (centered) {
      const centerX = x + w / 2;
      const centerY = y + h / 2;
      rectX = centerX - rectW / 2;
      rectY = centerY - rectH / 2;
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;

    if (style === 'BOX_FILLED') {
      ctx.fillStyle = color;
      ctx.fillRect(rectX, rectY, rectW, rectH);
    } else {
      ctx.strokeRect(rectX, rectY, rectW, rectH);
    }
  }

  // Contour drawing (polygon)
  static drawContour(canvas, ctx, contour, options = {}) {
    const { color = '#64C864', lineWidth = 2 } = options;

    if (!contour || contour.length === 0) return;

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();

    const [startX, startY] = contour[0];
    ctx.moveTo(startX, startY);

    for (let i = 1; i < contour.length; i++) {
      const [x, y] = contour[i];
      ctx.lineTo(x, y);
    }

    ctx.closePath();
    ctx.stroke();
  }

  // Arrow drawing
  static drawArrow(canvas, ctx, start, end, options = {}) {
    const {
      color = '#64C864',
      lineWidth = 2,
      tipLength = 0.1
    } = options;

    const [x1, y1] = start;
    const [x2, y2] = end;

    // Draw line
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    // Draw arrow head
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const length = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
    const arrowLength = length * tipLength;

    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(
      x2 - arrowLength * Math.cos(angle - Math.PI / 6),
      y2 - arrowLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.moveTo(x2, y2);
    ctx.lineTo(
      x2 - arrowLength * Math.cos(angle + Math.PI / 6),
      y2 - arrowLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.stroke();
  }

  // Match visualization
  static drawMatches(canvas1, canvas2, fromPoints, toPoints) {
    // Create combined canvas
    const combinedCanvas = document.createElement('canvas');
    const w1 = canvas1.width;
    const w2 = canvas2.width;
    const h = Math.max(canvas1.height, canvas2.height);

    combinedCanvas.width = w1 + w2;
    combinedCanvas.height = h;
    const ctx = combinedCanvas.getContext('2d');

    // Draw both images
    ctx.drawImage(canvas1, 0, 0);
    ctx.drawImage(canvas2, w1, 0);

    // Draw matching lines
    ctx.strokeStyle = '#64C864';
    ctx.lineWidth = 3;

    for (let i = 0; i < fromPoints.length; i++) {
      const [x1, y1] = fromPoints[i];
      const [x2, y2] = toPoints[i];

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(w1 + x2, y2);
      ctx.stroke();
    }

    return combinedCanvas;
  }
}
```

### Key Differences from OpenCV

| OpenCV (Python) | Canvas API (JavaScript) |
|----------------|-------------------------|
| `cv2.putText()` | `ctx.fillText()` |
| `cv2.rectangle()` | `ctx.strokeRect()` / `ctx.fillRect()` |
| `cv2.line()` | `ctx.lineTo()` + `ctx.stroke()` |
| `cv2.drawContours()` | Path with `ctx.lineTo()` loop |
| `cv2.arrowedLine()` | Manual arrow head calculation |
| BGR color tuples | Hex strings or RGB strings |
| In-place modification | Returns new canvas or modifies ctx |
| Anti-aliasing via LINE_AA | Automatic in Canvas |

### Color Conversion

```javascript
// OpenCV BGR to Canvas CSS
function bgrToColor(bgr) {
  const [b, g, r] = bgr;
  return `rgb(${r}, ${g}, ${b})`;
}

// Color constants
const CLR_BLACK = 'rgb(0, 0, 0)';
const CLR_GREEN = 'rgb(100, 200, 100)';
const CLR_GRAY = 'rgb(130, 130, 130)';
const CLR_DARK_GRAY = 'rgb(100, 100, 100)';
const CLR_WHITE = 'rgb(255, 255, 255)';
```

---

## Performance Considerations

### Drawing Operations Complexity

```
Operation         | Time Complexity | Notes
------------------|-----------------|---------------------------
draw_text()       | O(1)           | Single OpenCV call
draw_box()        | O(1)           | Single rectangle
draw_contour()    | O(n)           | n = contour points
draw_polygon()    | O(n)           | n lines to draw
draw_arrows()     | O(m)           | m = number of arrows
draw_matches()    | O(k)           | k = number of matches
```

### Memory Usage

- All operations modify image in-place (except `draw_matches`)
- Colored outputs require 3x memory (3 channels vs 1)
- Drawing doesn't allocate significant new memory

### Browser Optimization

```javascript
// Use OffscreenCanvas for background drawing
const offscreen = new OffscreenCanvas(width, height);
const ctx = offscreen.getContext('2d');

// Batch drawing operations
ctx.beginPath();
for (const contour of contours) {
  // Draw all contours in one path
}
ctx.stroke(); // Single stroke operation

// Cache font measurements
const textMetrics = new Map();
function getCachedTextWidth(text, font) {
  const key = `${text}-${font}`;
  if (!textMetrics.has(key)) {
    ctx.font = font;
    textMetrics.set(key, ctx.measureText(text).width);
  }
  return textMetrics.get(key);
}
```

---

## Related Documentation

- **Image Utils**: `../image/flows.md`
- **Math Utils**: `../math/flows.md`
- **Constants**: `src/utils/constants.py`
- **Debug System**: `../../../technical/debugging/debug-system.md`

---

## Summary

DrawingUtils provides:

1. **Text Rendering**: Simple, centered, and responsive text
2. **Shape Drawing**: Rectangles, contours, lines, polygons
3. **Arrows**: Directional indicators for alignment
4. **Colored Outputs**: BGR color support for visualization
5. **Debug Visualization**: Match visualization, symbol drawing
6. **Browser Migration**: Canvas API with similar functionality

**Key Design Principles**:
- All static methods (no instance state)
- In-place image modification (efficient)
- Flexible position handling (concrete or callable)
- OpenCV anti-aliasing support
- Color constants for consistency
