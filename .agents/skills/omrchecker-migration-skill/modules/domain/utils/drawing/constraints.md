# Drawing Utils Constraints

**Module**: Domain - Utils - Drawing
**Python Reference**: `src/utils/drawing.py`
**Last Updated**: 2026-02-21

---

## Input Constraints

### Image Parameter

**Type**: `np.ndarray` (OpenCV Mat)
**Expected Format**:
- Grayscale: shape `(height, width)`, dtype `uint8`
- Color: shape `(height, width, 3)`, dtype `uint8` (BGR order)

**Constraints**:
```python
# Valid images
gray_image.shape = (1000, 800)        # Grayscale
colored_image.shape = (1000, 800, 3)  # BGR color

# All drawing functions modify image IN-PLACE
# No return value (except draw_matches, draw_arrows)
```

**Mutation Guarantee**:
```python
original = image.copy()
DrawingUtils.draw_text(image, "Test", (10, 30))
assert not np.array_equal(image, original)  # Image is modified
```

---

## Position Constraints

### Concrete Position

**Type**: `tuple[int, int]` or `list[int, int]`
**Format**: `(x, y)` where x is horizontal, y is vertical
**Valid Range**:
```python
0 <= x < image.width
0 <= y < image.height
```

**Out-of-Bounds Behavior**:
```python
# OpenCV clips drawing to image boundaries
DrawingUtils.draw_text(image, "Text", (-10, -10))  # Clipped at (0, 0)
DrawingUtils.draw_text(image, "Text", (10000, 10000))  # Clipped at edge

# No exception raised, but may be invisible
```

**Integer Conversion**:
```python
# All positions are converted to integers internally
position = (10.7, 20.3)
# Becomes (10, 20) via int() conversion
```

### Callable Position

**Type**: `Callable[[int, int], tuple[int, int]]`
**Signature**: `def position(size_x: int, size_y: int) -> tuple[int, int]`

**Parameters**:
- `size_x`: Width of the element being drawn (text width, box width)
- `size_y`: Height of the element being drawn (text height, box height)

**Constraints**:
```python
# Callable position only supported by:
# - draw_text()
# - draw_text_responsive()
# - draw_symbol()

# NOT supported by:
# - draw_box() (must be concrete)
# - draw_line() (must be concrete)
# - draw_contour() (must be concrete)
```

**Centered Constraint**:
```python
# centered=True and callable position are mutually exclusive
DrawingUtils.draw_text(
    image,
    "Text",
    lambda w, h: (10, 20),  # Callable
    centered=True           # ERROR: ImageProcessingError raised
)
```

---

## Text Drawing Constraints

### text_value

**Type**: `str`
**Constraints**:
```python
# Any string is valid, but:
# - Empty string draws nothing
# - Newlines are NOT supported (single line only)
# - Special characters depend on font support

# Examples
"Q1"           # Valid
"Score: 85"    # Valid
"Multi\nLine"  # Only "Multi" is drawn, "\nLine" ignored
""             # Valid but invisible
"⚠️"            # Valid if font supports emoji (unlikely in OpenCV)
```

### text_size

**Type**: `float`
**Default**: `TEXT_SIZE = 0.95` (from constants)
**Valid Range**: `0.1 to 10.0` (recommended)

**Scale Interpretation**:
```python
# text_size is a scale factor, not pixel height
text_size=0.5   # Small text (~8px height)
text_size=0.95  # Default (~15px height)
text_size=1.0   # Standard (~16px height)
text_size=2.0   # Large (~32px height)

# Actual pixel height depends on font_face
```

**Constraint**: Must be positive, zero or negative not supported

### thickness

**Type**: `int`
**Default**: `2`
**Valid Range**: `1 to 10` (recommended)

**Effects**:
```python
thickness=1  # Thin, may be hard to read
thickness=2  # Default, good readability
thickness=3  # Bold
thickness=5  # Very bold, may blur at small sizes
thickness=-1 # NOT valid for text (only for rectangles)
```

### font_face

**Type**: `int` (OpenCV font constant)
**Default**: `cv2.FONT_HERSHEY_SIMPLEX`

**Valid Values**:
```python
cv2.FONT_HERSHEY_SIMPLEX       # Clean, readable (default)
cv2.FONT_HERSHEY_PLAIN         # Small, simple
cv2.FONT_HERSHEY_DUPLEX        # More complex
cv2.FONT_HERSHEY_COMPLEX       # Serif-like
cv2.FONT_HERSHEY_TRIPLEX       # More elaborate
cv2.FONT_HERSHEY_COMPLEX_SMALL # Smaller complex
cv2.FONT_HERSHEY_SCRIPT_SIMPLEX # Handwriting-like
cv2.FONT_HERSHEY_SCRIPT_COMPLEX # Fancy handwriting

# Italic variants (add to above)
cv2.FONT_ITALIC  # Can be OR'd: FONT_HERSHEY_SIMPLEX | FONT_ITALIC
```

**Constraint**: Only built-in OpenCV fonts supported (no custom fonts)

### line_type

**Type**: `int`
**Default**: `cv2.LINE_AA` (anti-aliased)

**Valid Values**:
```python
cv2.FILLED   # -1 (not applicable to text)
cv2.LINE_4   # 4-connected line (aliased, fast)
cv2.LINE_8   # 8-connected line (aliased)
cv2.LINE_AA  # Anti-aliased (smooth, slower)
```

**Recommendation**: Always use `LINE_AA` for text quality

---

## Box Drawing Constraints

### box_dimensions

**Type**: `tuple[int, int]`
**Format**: `(width, height)`
**Valid Range**:
```python
width > 0
height > 0

# Typical values:
(50, 50)   # Small bubble
(120, 40)  # Wide field
(200, 200) # Large block
```

### thickness_factor

**Type**: `float`
**Default**: `1/12 ≈ 0.0833`
**Valid Range**: `0.0 to 0.5`

**Purpose**: Creates inset rectangle within original box
**Formula**:
```python
inset_x = box_width * thickness_factor
inset_y = box_height * thickness_factor

# Drawable area:
drawn_width = box_width - 2 * inset_x
drawn_height = box_height - 2 * inset_y

# Example: box_width=120, thickness_factor=1/12
inset_x = 120 * (1/12) = 10
drawn_width = 120 - 20 = 100
```

**Constraints**:
```python
# If thickness_factor >= 0.5, box collapses to point or inverts
thickness_factor=0.0    # No inset, full box size
thickness_factor=0.1    # 10% inset on each side
thickness_factor=0.5    # 50% inset, box becomes a point
thickness_factor=0.6    # INVALID: negative width/height
```

**Recommendation**: Keep between `0.05` and `0.2`

### style

**Type**: `str`
**Valid Values**: Exactly 2 options

```python
"BOX_HOLLOW"  # Outline only (default)
"BOX_FILLED"  # Solid filled rectangle

# Any other value defaults to custom color/border
```

**Behavior**:
```python
if style == "BOX_HOLLOW":
    color = CLR_GRAY if color is None else color
    border = 3  # Use provided border value

elif style == "BOX_FILLED":
    color = CLR_DARK_GRAY if color is None else color
    border = -1  # Override to filled

else:
    # Use provided color and border as-is
    pass
```

### border

**Type**: `int`
**Default**: `3`
**Valid Values**:
```python
border > 0   # Hollow rectangle with border thickness
border = -1  # Filled rectangle (ignores style)

# Examples
border=1  # Thin outline
border=3  # Standard outline
border=5  # Thick outline
border=-1 # Filled
```

**Constraint**: `border=0` is technically valid but draws nothing (useless)

### centered

**Type**: `bool`
**Default**: `False`

**Behavior**:
```python
# centered=False: position is top-left corner
DrawingUtils.draw_box(image, (100, 100), (50, 50))
# Draws box from (100, 100) to (150, 150)

# centered=True: position is center point
DrawingUtils.draw_box(image, (100, 100), (50, 50), centered=True)
# Draws box centered at (100, 100)
# Actual corners: (~75, ~75) to (~125, ~125)
```

**Formula** (when centered=True):
```python
# After applying thickness_factor:
centered_position = [
    (3 * position[0] - position_diagonal[0]) // 2,
    (3 * position[1] - position_diagonal[1]) // 2
]

centered_diagonal = [
    (position[0] + position_diagonal[0]) // 2,
    (position[1] + position_diagonal[1]) // 2
]
```

---

## Contour Drawing Constraints

### contour

**Type**: `np.ndarray`
**Expected Shape**: `(n, 1, 2)` or `(n, 2)` where n ≥ 3
**Data Type**: Any numeric type (converted to `np.intp`)

**Valid Formats**:
```python
# OpenCV contour format (n, 1, 2)
contour = np.array([
    [[10, 20]],
    [[30, 40]],
    [[50, 60]],
    [[70, 80]]
])

# Simple points format (n, 2)
contour = np.array([
    [10, 20],
    [30, 40],
    [50, 60],
    [70, 80]
])

# Both are valid and converted to np.intp
```

**Validation**:
```python
# None values are NOT allowed
contour = [[10, 20], None, [30, 40]]  # RAISES ImageProcessingError

# Check performed:
if None in contour:
    raise ImageProcessingError(
        "Invalid contour provided",
        context={"contour": str(contour)}
    )
```

**Minimum Points**: At least 1 point (but 3+ recommended for visible polygon)

---

## Arrow Drawing Constraints

### start_points / end_points

**Type**: `list | np.ndarray`
**Expected Format**: List of points or numpy array of shape `(n, 2)`

**Conversion**:
```python
# Input can be numpy array
start_points = np.array([[10, 20], [30, 40]])

# Converted to list of integer tuples
start_points = MathUtils.get_tuple_points(start_points)
# Result: [(10, 20), (30, 40)]
```

**Constraint**: `len(start_points) == len(end_points)`
```python
# Valid
draw_arrows(image, [(10,20), (30,40)], [(50,60), (70,80)])

# Invalid (length mismatch) - will raise ValueError in zip()
draw_arrows(image, [(10,20)], [(50,60), (70,80)])
```

### tip_length

**Type**: `float`
**Default**: `0.1`
**Valid Range**: `0.0 to 1.0`

**Interpretation**: Ratio of arrow tip length to total arrow length
```python
tip_length=0.05  # Small tip (5% of arrow)
tip_length=0.1   # Default (10% of arrow)
tip_length=0.2   # Large tip (20% of arrow)
tip_length=0.5   # Very large tip (50% of arrow)
tip_length=1.0   # Tip same length as arrow (extreme)
```

**Constraint**: Values > 1.0 create tips longer than arrow itself (rarely useful)

---

## Color Constraints

### Color Format

**Type**: `tuple[int, int, int]`
**Format**: `(B, G, R)` - **BGR order** (OpenCV convention)
**Valid Range**: Each component `0-255`

**Common Mistake**:
```python
# WRONG: RGB order
color = (255, 0, 0)  # Intended red, but shows BLUE in OpenCV

# CORRECT: BGR order
color = (0, 0, 255)  # Red in BGR
```

**Predefined Colors** (from constants.py):
```python
CLR_BLACK = (0, 0, 0)
CLR_WHITE = (255, 255, 255)
CLR_GRAY = (130, 130, 130)
CLR_DARK_GRAY = (100, 100, 100)
CLR_LIGHT_GRAY = (200, 200, 200)

CLR_GREEN = (100, 200, 100)       # BGR
CLR_DARK_GREEN = (20, 255, 20)    # BGR
CLR_DARK_BLUE = (255, 20, 20)     # BGR (blue, not red!)
CLR_DARK_RED = (20, 20, 255)      # BGR
```

**Transparency**:
```python
# OpenCV drawing functions do NOT support alpha channel
# 4-tuple (B, G, R, A) will cause error or ignore alpha

# For transparency, must composite manually:
overlay = image.copy()
DrawingUtils.draw_box(overlay, pos, dims, color=CLR_GREEN)
cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)
```

---

## Polygon Drawing Constraints

### points

**Type**: `list[tuple[int, int]]`
**Minimum Length**: 2 (for line), 3 (for triangle)

**Examples**:
```python
# Triangle
points = [(10, 20), (30, 40), (50, 20)]

# Square
points = [(10, 10), (50, 10), (50, 50), (10, 50)]

# Line (with closed=False)
points = [(10, 20), (30, 40)]
```

### closed

**Type**: `bool`
**Default**: `True`

**Behavior**:
```python
# closed=True: Connects last point back to first
points = [(10,10), (50,10), (50,50)]
# Draws: (10,10)→(50,10)→(50,50)→(10,10) [triangle]

# closed=False: Doesn't connect last to first
points = [(10,10), (50,10), (50,50)]
# Draws: (10,10)→(50,10)→(50,50) [open path]
```

---

## Group Drawing Constraints

### box_edge

**Type**: `str`
**Valid Values**: Exactly 4 options

```python
"TOP"     # Draw on top edge
"RIGHT"   # Draw on right edge
"BOTTOM"  # Draw on bottom edge
"LEFT"    # Draw on left edge
```

**Case Sensitive**: Must be uppercase

**Invalid Edge**:
```python
DrawingUtils.draw_group(..., box_edge="top")  # Invalid (lowercase)
DrawingUtils.draw_group(..., box_edge="MIDDLE")  # Invalid (not an edge)
# Result: No drawing, silently ignored
```

### thickness_factor (for group)

**Type**: `float`
**Default**: `7/10 = 0.7`
**Valid Range**: `0.0 to 1.0`

**Purpose**: Determines line coverage on edge
```python
# thickness_factor=0.7: Line covers center 70% of edge
# For 100px wide box:
#   Line starts at 15px from left
#   Line ends at 85px from left
#   Total line width: 70px (70%)

# thickness_factor=1.0: Line covers full edge
# thickness_factor=0.5: Line covers center 50%
# thickness_factor=0.0: Line has zero length (invisible)
```

**Implementation Note**:
```python
# Current implementation has potential bugs in BOTTOM and RIGHT cases
# where start position is reassigned incorrectly

# Example from code (BOTTOM case):
start = (start_x, start_y + box_h)  # Correct: bottom edge
end_position = (int(start_x + box_w * thickness_factor), start_y)  # BUG: uses start_y not start_y+box_h
start = (int(start_x + box_w * (1 - thickness_factor)), start_y)  # BUG: overwrites start

# Likely intended behavior needs verification
```

---

## Performance Constraints

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `draw_text()` | O(1) | Single OpenCV call |
| `draw_box()` | O(1) | Single rectangle |
| `draw_box_diagonal()` | O(1) | Single rectangle |
| `draw_contour()` | O(n) | n = contour points |
| `draw_line()` | O(1) | Single line |
| `draw_polygon()` | O(n) | n = polygon points |
| `draw_arrows()` | O(m) | m = number of arrows |
| `draw_matches()` | O(k + s) | k = matches, s = stack operation |
| `draw_symbol()` | O(1) | Delegates to draw_text |
| `draw_group()` | O(1) | Single line |
| `draw_text_responsive()` | O(1) | Delegates to draw_text |

### Memory Constraints

**In-Place Modification**:
```python
# Most functions modify image in-place (no new allocation)
original_id = id(image)
DrawingUtils.draw_text(image, "Test", (10, 30))
assert id(image) == original_id  # Same object

# Exception: draw_matches creates new image
match_viz = DrawingUtils.draw_matches(img1, from_pts, img2, to_pts)
# match_viz is a NEW image (horizontal stack)
```

**Memory Usage**:
```python
# Colored output requirement:
# - Grayscale image: H × W × 1 byte
# - Colored image: H × W × 3 bytes (3x memory)

# Example: 1000×800 image
gray_memory = 1000 × 800 × 1 = 800 KB
color_memory = 1000 × 800 × 3 = 2.4 MB

# Drawing on grayscale converts to color if colored_outputs_enabled
```

### Browser Constraints

**Canvas API Limits**:
```javascript
// Maximum canvas size varies by browser
// Chrome: 32,767 × 32,767 pixels
// Firefox: 32,767 × 32,767 pixels
// Safari: 4,194,304 total pixels

// For OMR images (typically 2000×3000):
const maxSafe = 10000;  // Conservative limit
if (width > maxSafe || height > maxSafe) {
    console.warn('Image too large for Canvas API');
}
```

**Rendering Performance**:
```javascript
// Drawing on Canvas is slower than OpenCV
// Optimization strategies:

// 1. Batch operations
ctx.beginPath();
for (const contour of contours) {
    // Add all paths
}
ctx.stroke();  // Single stroke call

// 2. Use OffscreenCanvas for background work
const offscreen = new OffscreenCanvas(w, h);

// 3. Avoid frequent context switches
ctx.strokeStyle = color;
// Draw all items with same color
```

---

## Thread Safety Constraints

### Thread Safety

**Status**: Fully thread-safe (with caveats)

**Safe**:
```python
# Static methods with no shared state
# Each call operates on different image buffers

from concurrent.futures import ThreadPoolExecutor

def draw_on_image(image, text):
    DrawingUtils.draw_text(image, text, (10, 30))
    return image

# Safe: Each thread draws on different image
with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(draw_on_image, images, texts)
```

**Unsafe**:
```python
# Same image object accessed by multiple threads
shared_image = cv2.imread("template.jpg")

def draw_on_shared(text):
    DrawingUtils.draw_text(shared_image, text, (10, 30))  # RACE CONDITION

# Unsafe: Multiple threads modify same image
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(draw_on_shared, texts)  # CORRUPTED OUTPUT
```

**Recommendation**:
- Each thread should have its own image copy
- Or use locks when drawing on shared images

**Browser Implication**:
```javascript
// Web Workers cannot directly access DOM Canvas
// Must use OffscreenCanvas or ImageBitmap

// Transfer canvas to worker
const offscreen = canvas.transferControlToOffscreen();
worker.postMessage({ canvas: offscreen }, [offscreen]);
```

---

## Error Handling Constraints

### Exceptions Raised

**ImageProcessingError**:
```python
# Raised by draw_contour() when contour contains None
if None in contour:
    raise ImageProcessingError(
        "Invalid contour provided",
        context={"contour": str(contour)}
    )

# Raised by draw_text() when centered=True and position is callable
if centered and callable(position):
    raise ImageProcessingError(
        f"centered={centered} but position is a callable: {position}",
        context={"centered": centered, "position": str(position)}
    )
```

**OpenCV Errors**:
```python
# OpenCV may crash on extreme inputs
# - Very large coordinates (> 2^31)
# - Invalid image buffer (corrupted data)
# - Null image pointer

# No Python exception, process may crash!
# Recommendation: Validate inputs before drawing
```

### Silent Failures

**Out-of-Bounds Drawing**:
```python
# OpenCV clips to image boundaries without error
DrawingUtils.draw_text(image, "Text", (10000, 10000))
# No exception, but text is invisible (drawn outside image)
```

**Invalid Box Dimensions**:
```python
# Negative dimensions may cause undefined behavior
DrawingUtils.draw_box(image, (10, 10), (-50, -50))
# OpenCV behavior is undefined (may crash or clip)
```

**Invalid Font**:
```python
# Invalid font_face value uses default
DrawingUtils.draw_text(image, "Text", (10, 30), font_face=999)
# May use default font or crash (OpenCV version dependent)
```

---

## Validation Recommendations

### Pre-Drawing Validation

```python
def validate_drawing_inputs(image, position, dimensions=None):
    """Validate inputs before drawing operations."""

    # Check image
    assert image is not None, "Image is None"
    assert image.size > 0, "Image is empty"
    assert image.dtype == np.uint8, f"Image dtype {image.dtype}, expected uint8"

    # Check position
    h, w = image.shape[:2]
    x, y = position
    assert 0 <= x < w, f"X position {x} out of bounds [0, {w})"
    assert 0 <= y < h, f"Y position {y} out of bounds [0, {h})"

    # Check dimensions if provided
    if dimensions:
        box_w, box_h = dimensions
        assert box_w > 0, f"Width {box_w} must be positive"
        assert box_h > 0, f"Height {box_h} must be positive"
        assert x + box_w <= w, f"Box extends beyond image width"
        assert y + box_h <= h, f"Box extends beyond image height"
```

### TypeScript Type Guards

```typescript
interface DrawingOptions {
    color?: string;
    lineWidth?: number;
    fontSize?: number;
    centered?: boolean;
}

function validatePosition(
    pos: [number, number],
    width: number,
    height: number
): void {
    const [x, y] = pos;
    if (x < 0 || x >= width || y < 0 || y >= height) {
        console.warn(`Position (${x}, ${y}) out of bounds`);
    }
}

function validateColor(color: string): void {
    const validFormats = [
        /^#[0-9A-Fa-f]{6}$/,        // #RRGGBB
        /^rgb\(\d+,\s*\d+,\s*\d+\)$/, // rgb(r, g, b)
    ];

    if (!validFormats.some(regex => regex.test(color))) {
        throw new Error(`Invalid color format: ${color}`);
    }
}
```

---

## Browser-Specific Constraints

### Canvas Coordinate System

```javascript
// Canvas uses same coordinate system as OpenCV
// (0, 0) is top-left
// X increases to the right
// Y increases downward

// But text baseline is different:
// OpenCV: position is bottom-left of text
// Canvas: position is baseline (slightly above bottom)

// Adjustment needed:
function drawTextOpenCVStyle(ctx, text, x, y) {
    const metrics = ctx.measureText(text);
    const height = metrics.actualBoundingBoxAscent +
                   metrics.actualBoundingBoxDescent;

    // Adjust Y to match OpenCV behavior
    ctx.fillText(text, x, y);
}
```

### Font Differences

```javascript
// OpenCV: Built-in bitmap fonts (limited)
// Canvas: Any web font (extensive)

// Font mapping:
const FONT_MAP = {
    'FONT_HERSHEY_SIMPLEX': 'Arial',
    'FONT_HERSHEY_COMPLEX': 'Times New Roman',
    'FONT_HERSHEY_SCRIPT_SIMPLEX': 'Brush Script MT',
};

// Size mapping (approximate):
function cvTextSizeToFontSize(cvSize) {
    return cvSize * 16;  // TEXT_SIZE=0.95 → 15px
}
```

### Color Space

```javascript
// OpenCV: BGR color space
// Canvas: RGB color space

// Conversion required:
function bgrToRgb([b, g, r]) {
    return `rgb(${r}, ${g}, ${b})`;
}

// Color constants (converted):
const CLR_BLACK = 'rgb(0, 0, 0)';
const CLR_GREEN = 'rgb(100, 200, 100)';  // Note: reversed from (100,200,100)
const CLR_DARK_BLUE = 'rgb(20, 20, 255)'; // Was (255,20,20) in BGR
```

---

## Summary of Critical Constraints

| Constraint | Value/Rule | Impact |
|------------|-----------|---------|
| Image modification | In-place (except draw_matches) | Efficient, but mutates input |
| Position format | (x, y) or callable | Flexible positioning |
| Color format | BGR tuple (0-255) | OpenCV convention |
| Text centered | Cannot combine with callable | Raises error |
| Contour validation | No None values | Raises ImageProcessingError |
| Arrow points | Equal length lists | zip() requirement |
| Box thickness_factor | 0.0 to 0.5 | > 0.5 causes collapse |
| Thread safety | Safe if separate images | Unsafe on shared image |
| OpenCV errors | May crash process | Validate inputs |
| Browser coordinate | Same as OpenCV | Top-left origin |
| Browser color | RGB not BGR | Conversion needed |
| Browser fonts | Web fonts vs bitmap | Mapping required |

---

## Related Constraints

- **Image Utils**: `../image/constraints.md`
- **Math Utils**: `../math/constraints.md`
- **OpenCV Operations**: `../../../technical/opencv/opencv-operations.md`
- **Browser Canvas API**: `../../../migration/browser-adaptations.md`
