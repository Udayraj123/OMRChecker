# Bubble Detection Drawing - Constraints & Limitations

## Performance Constraints

### 1. Rendering Performance

**OpenCV Drawing Operations**:
- Each bubble requires 1-3 drawing operations:
  - 1 box drawing (`cv2.rectangle`)
  - 1 text rendering (`cv2.putText`) - optional
  - 1 symbol rendering (`cv2.putText`) - optional
- For a field with 10 bubbles: 10-30 drawing operations
- For a sheet with 100 questions (400 bubbles): 400-1200 operations

**Time Complexity**:
- `draw_field_interpretation()`: O(n) where n = number of bubbles
- `draw_answer_groups_for_bubbles()`: O(n × m) where m = matched answer groups per bubble
- Overall per field: O(n × m), typically O(n) as m ≤ 4

**Memory Usage**:
- Drawing modifies image in-place (no additional memory allocation)
- Text size calculation allocates temporary tuples: `cv2.getTextSize()`
- Negligible memory overhead per bubble (~100 bytes for position/dimension tuples)

### 2. Browser Performance Constraints

**Canvas API Performance**:
- Canvas operations are synchronous and block main thread
- Rendering 400 bubbles can take 50-200ms depending on complexity
- Anti-aliased text rendering is expensive

**Optimization Strategies**:
```typescript
// 1. Batch canvas operations
ctx.save();
// ... perform all drawing ...
ctx.restore();

// 2. Use OffscreenCanvas for parallel rendering
const offscreen = new OffscreenCanvas(width, height);
const ctx = offscreen.getContext('2d');
// ... draw on offscreen ...
mainCtx.drawImage(offscreen, 0, 0);

// 3. Disable anti-aliasing for filled boxes
ctx.imageSmoothingEnabled = false;

// 4. Cache common text measurements
const textMetricsCache = new Map<string, TextMetrics>();
```

**Performance Targets**:
- Target: < 100ms for full sheet rendering (400 bubbles)
- Acceptable: < 500ms for full sheet
- Critical threshold: > 1000ms (user perceives lag)

### 3. Image Size Constraints

**Resolution Impact**:
- Drawing operations are resolution-dependent
- Higher resolution = more pixels to fill = slower rendering
- Typical OMR sheet: 2000×3000px (6MP)
- Large sheet: 4000×6000px (24MP)

**Scaling Considerations**:
```python
# Python: Text size is relative
text_size = TEXT_SIZE  # 0.95
thickness = int(1 + 3.5 * TEXT_SIZE)  # ~4 pixels

# Browser: Need to scale based on canvas resolution
const scaleFactor = canvasWidth / 2000;  // Assume 2000px baseline
const textSize = 0.95 * 16 * scaleFactor;
const thickness = Math.round((1 + 3.5 * 0.95) * scaleFactor);
```

---

## Edge Cases

### 1. Overlapping Bubbles

**Problem**: Bubble positions too close, causing overlapping rendering.

**Detection**:
```python
# No explicit overlap detection in current implementation
# Bubbles are drawn in order, later bubbles may overdraw earlier ones
```

**Browser Handling**:
```typescript
// Option 1: Detect and skip overlapping bubbles
function detectOverlap(
    pos1: [number, number], dim1: [number, number],
    pos2: [number, number], dim2: [number, number]
): boolean {
    const [x1, y1] = pos1;
    const [w1, h1] = dim1;
    const [x2, y2] = pos2;
    const [w2, h2] = dim2;

    return !(x1 + w1 < x2 || x2 + w2 < x1 || y1 + h1 < y2 || y2 + h2 < y1);
}

// Option 2: Use transparency to show overlaps
ctx.globalAlpha = 0.8;  // Make overlapping areas visible
```

**Mitigation**: Template validation should prevent overlapping bubble definitions.

### 2. Empty Bubble Interpretations

**Problem**: Field has no bubble interpretations (`bubble_interpretations = []`).

**Handling**:
```python
for bubble_interpretation in bubble_interpretations:
    # Loop body never executes if list is empty
    # No error thrown, simply no drawing occurs
```

**Browser Handling**: Same as Python - empty loop, no rendering.

### 3. Missing Question Meta

**Problem**: `field_label` not in `evaluation_meta["questions_meta"]`.

**Handling**:
```python
question_has_verdict = (
    evaluation_meta is not None and
    field_label in evaluation_meta["questions_meta"]
)

if should_draw_question_verdicts and question_has_verdict:
    # Draw with verdicts
else:
    # Fallback to detection-only mode
```

**Browser Handling**: Same logic - gracefully degrade to detection-only mode.

### 4. Invalid Bubble Positions

**Problem**: `shifted_position` or `dimensions` are negative or out of bounds.

**Python Behavior**:
```python
# OpenCV clips coordinates automatically
cv2.rectangle(image, (x, y), (x2, y2), color, border)
# If x < 0, OpenCV clips to 0
# If x2 > width, OpenCV clips to width
```

**Browser Handling**:
```typescript
// Canvas does NOT clip automatically
function clampRect(
    x: number, y: number, w: number, h: number,
    canvasWidth: number, canvasHeight: number
): [number, number, number, number] {
    const x1 = Math.max(0, x);
    const y1 = Math.max(0, y);
    const x2 = Math.min(canvasWidth, x + w);
    const y2 = Math.min(canvasHeight, y + h);
    return [x1, y1, Math.max(0, x2 - x1), Math.max(0, y2 - y1)];
}
```

### 5. Text Overflow

**Problem**: Bubble value text is too long to fit inside bubble.

**Python Behavior**:
```python
# No text overflow handling
# Text may extend outside bubble boundaries
DrawingUtils.draw_text(marked_image, bubble_value, shifted_position, ...)
```

**Browser Handling**:
```typescript
// Option 1: Truncate text
function truncateText(
    ctx: CanvasRenderingContext2D,
    text: string, maxWidth: number
): string {
    const metrics = ctx.measureText(text);
    if (metrics.width <= maxWidth) return text;

    let truncated = text;
    while (truncated.length > 0) {
        truncated = truncated.slice(0, -1);
        if (ctx.measureText(truncated + '...').width <= maxWidth) {
            return truncated + '...';
        }
    }
    return '';
}

// Option 2: Scale font size
function getFontSizeToFit(
    ctx: CanvasRenderingContext2D,
    text: string, maxWidth: number, maxHeight: number, baseFontSize: number
): number {
    let fontSize = baseFontSize;
    ctx.font = `${fontSize}px sans-serif`;
    while (fontSize > 8) {
        const metrics = ctx.measureText(text);
        if (metrics.width <= maxWidth * 0.9) break;
        fontSize -= 1;
        ctx.font = `${fontSize}px sans-serif`;
    }
    return fontSize;
}
```

### 6. Answer Group Limit

**Problem**: Question has more than 4 correct answer groups.

**Current Limitation**:
```python
box_edges = ["TOP", "RIGHT", "BOTTOM", "LEFT"]  # Only 4 edges
# Only supports up to 4 answer groups
for answer_index in matched_groups:
    box_edge = box_edges[answer_index % 4]  # Wraps around using modulo
```

**Impact**: Groups 4+ will reuse edges (group 4 uses TOP again, group 5 uses RIGHT, etc.).

**Browser Alternative**:
```typescript
// Use corner markers instead of edge lines
const cornerPositions = [
    'top-left', 'top-right', 'bottom-right', 'bottom-left',
    'top-center', 'right-center', 'bottom-center', 'left-center'
];

function drawGroupMarker(
    ctx: CanvasRenderingContext2D,
    x: number, y: number, w: number, h: number,
    position: string, color: string
): void {
    const markerSize = 8;
    ctx.fillStyle = color;

    let mx: number, my: number;
    switch (position) {
        case 'top-left': [mx, my] = [x, y]; break;
        case 'top-right': [mx, my] = [x + w - markerSize, y]; break;
        // ... other positions
    }

    ctx.fillRect(mx, my, markerSize, markerSize);
}
```

### 7. Grayscale Mode Edge Cases

**Problem**: Verdict colors become indistinguishable in grayscale.

**Current Handling**:
```python
if image_type == "GRAYSCALE":
    color = CLR_WHITE         # All filled boxes become white
    symbol_color = CLR_BLACK  # All symbols become black
```

**Impact**: Cannot distinguish correct/incorrect/neutral by color alone - must rely on symbols.

**Browser Enhancement**:
```typescript
// Use different fill patterns instead of solid colors
function drawGrayscaleVerdict(
    ctx: CanvasRenderingContext2D,
    x: number, y: number, w: number, h: number,
    verdict: 'correct' | 'incorrect' | 'neutral' | 'bonus'
): void {
    switch (verdict) {
        case 'correct':
            // Solid white fill
            ctx.fillStyle = 'rgb(255, 255, 255)';
            ctx.fillRect(x, y, w, h);
            break;

        case 'incorrect':
            // Diagonal stripe pattern
            ctx.fillStyle = ctx.createPattern(createStripePattern(), 'repeat');
            ctx.fillRect(x, y, w, h);
            break;

        case 'neutral':
            // Dotted pattern
            ctx.fillStyle = ctx.createPattern(createDotPattern(), 'repeat');
            ctx.fillRect(x, y, w, h);
            break;

        case 'bonus':
            // Light gray fill
            ctx.fillStyle = 'rgb(200, 200, 200)';
            ctx.fillRect(x, y, w, h);
            break;
    }
}
```

---

## Canvas API Browser Compatibility

### 1. Canvas 2D Context Support

**Basic Support** (all modern browsers):
- `fillRect()`, `strokeRect()` - ✅ Universal support
- `fillText()`, `strokeText()` - ✅ Universal support (IE9+)
- `measureText()` - ✅ Universal support

**Advanced Features**:
- `globalAlpha` - ✅ Universal support
- `imageSmoothingEnabled` - ✅ Chrome 30+, Firefox 51+, Safari 9+
- `textBaseline`, `textAlign` - ✅ Universal support
- `save()`, `restore()` - ✅ Universal support

**Feature Detection**:
```typescript
function checkCanvasSupport(): boolean {
    const canvas = document.createElement('canvas');
    return !!(canvas.getContext && canvas.getContext('2d'));
}

function checkTextMetricsSupport(): boolean {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return false;

    const metrics = ctx.measureText('test');
    return typeof metrics.width === 'number';
}
```

### 2. OffscreenCanvas Support

**Support Status**:
- ✅ Chrome 69+
- ✅ Firefox 105+
- ✅ Safari 16.4+
- ❌ IE (no support)

**Fallback Strategy**:
```typescript
function createRenderingContext(
    width: number, height: number
): CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D {
    if (typeof OffscreenCanvas !== 'undefined') {
        const offscreen = new OffscreenCanvas(width, height);
        return offscreen.getContext('2d')!;
    } else {
        // Fallback to regular canvas
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        return canvas.getContext('2d')!;
    }
}
```

### 3. Font Rendering Differences

**Cross-Browser Font Variations**:
- Font family availability varies by OS
- Font metrics (width, height) differ slightly across browsers
- Anti-aliasing algorithms differ

**Mitigation**:
```typescript
// Use web-safe fonts with fallbacks
const fontFamily = 'Arial, Helvetica, sans-serif';

// Or load custom font via CSS Font Loading API
async function loadFont(): Promise<void> {
    const font = new FontFace(
        'CustomFont',
        'url(/fonts/custom.woff2)',
        { weight: '400' }
    );
    await font.load();
    document.fonts.add(font);
}

// Use before rendering
await loadFont();
ctx.font = `16px CustomFont, Arial, sans-serif`;
```

### 4. Color Space Handling

**RGB vs BGR**:
- Python OpenCV uses BGR order: `(B, G, R)`
- Canvas API uses RGB order: `rgb(R, G, B)`

**Conversion Required**:
```typescript
// Python: CLR_DARK_RED = (20, 20, 255)  # BGR
// Browser: rgb(255, 20, 20)             # RGB

function bgrToRgb(bgr: [number, number, number]): string {
    const [b, g, r] = bgr;
    return `rgb(${r}, ${g}, ${b})`;
}

// Example constants
const CLR_DARK_RED = bgrToRgb([20, 20, 255]);  // "rgb(255, 20, 20)"
const CLR_GREEN = bgrToRgb([100, 200, 100]);   // "rgb(100, 200, 100)"
```

### 5. High DPI / Retina Display Support

**Problem**: Canvas drawing appears blurry on high DPI displays.

**Solution**:
```typescript
function createHiDPICanvas(
    width: number, height: number
): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    const dpr = window.devicePixelRatio || 1;

    // Set canvas size in memory (scaled up)
    canvas.width = width * dpr;
    canvas.height = height * dpr;

    // Set display size (CSS pixels)
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    // Scale all drawing operations
    ctx.scale(dpr, dpr);

    return { canvas, ctx };
}

// Usage
const { canvas, ctx } = createHiDPICanvas(2000, 3000);
// Draw at logical resolution (2000×3000)
// Automatically renders at physical resolution (4000×6000 on 2x display)
```

---

## Configuration Constraints

### 1. Verdict Colors Validation

**Required Format**:
```json
{
  "verdict_colors": {
    "correct": "#64C864",     // Hex string or null
    "incorrect": "#FF6464",
    "neutral": null,          // Falls back to incorrect color
    "bonus": "#FFA500"
  }
}
```

**Validation**:
```typescript
function validateVerdictColors(colors: Record<string, string | null>): void {
    const required = ['correct', 'incorrect', 'bonus'];
    for (const key of required) {
        if (!(key in colors)) {
            throw new Error(`Missing required verdict color: ${key}`);
        }
    }

    // Neutral defaults to incorrect if null
    if (colors.neutral === null) {
        colors.neutral = colors.incorrect;
    }

    // Validate hex format
    const hexPattern = /^#[0-9A-Fa-f]{6}$/;
    for (const [key, value] of Object.entries(colors)) {
        if (value !== null && !hexPattern.test(value)) {
            throw new Error(`Invalid hex color for ${key}: ${value}`);
        }
    }
}
```

### 2. Draw Answer Groups Constraints

**Color Sequence Limit**:
```python
# Only 4 colors supported (maps to 4 edges)
color_sequence = [
    (255, 0, 0),    # Red
    (0, 0, 255),    # Blue
    (0, 255, 0),    # Green
    (255, 255, 0),  # Yellow
]
# Groups beyond 4 will wrap around (modulo 4)
```

**Browser Validation**:
```typescript
function validateColorSequence(colors: string[]): void {
    if (colors.length !== 4) {
        console.warn(
            `Expected 4 colors in color_sequence, got ${colors.length}. ` +
            `Only first 4 will be used.`
        );
    }

    const hexPattern = /^#[0-9A-Fa-f]{6}$/;
    for (let i = 0; i < Math.min(4, colors.length); i++) {
        if (!hexPattern.test(colors[i])) {
            throw new Error(`Invalid hex color at index ${i}: ${colors[i]}`);
        }
    }
}
```

### 3. Text Size Constraints

**Minimum/Maximum Text Size**:
```python
# No explicit constraints in Python
TEXT_SIZE = 0.95  # Typical value
# Range: 0.5 (small) to 2.0 (large)
```

**Browser Best Practices**:
```typescript
const MIN_TEXT_SIZE = 8;   // Minimum readable size (pixels)
const MAX_TEXT_SIZE = 48;  // Maximum reasonable size

function clampTextSize(
    baseSize: number, scaleFactor: number
): number {
    const scaled = baseSize * scaleFactor;
    return Math.max(MIN_TEXT_SIZE, Math.min(MAX_TEXT_SIZE, scaled));
}
```

---

## Memory Constraints

### 1. Image Buffer Size

**Python**:
- NumPy array: `height × width × channels × bytes_per_pixel`
- Example: 3000×2000×3×1 = 18MB per image
- Drawing modifies in-place (no additional allocation)

**Browser**:
- Canvas ImageData: `width × height × 4` bytes (RGBA)
- Example: 3000×2000×4 = 24MB per canvas
- JavaScript heap limit: 1-4GB depending on browser

**Browser Memory Management**:
```typescript
class BubbleRenderer {
    private imageDataCache: ImageData | null = null;

    renderBubbles(
        canvas: HTMLCanvasElement,
        bubbles: BubbleInterpretation[]
    ): void {
        const ctx = canvas.getContext('2d')!;

        // Batch all drawing operations
        ctx.save();
        for (const bubble of bubbles) {
            this.drawBubble(ctx, bubble);
        }
        ctx.restore();

        // Clear cache after rendering
        this.imageDataCache = null;
    }

    dispose(): void {
        this.imageDataCache = null;
        // Allow garbage collection
    }
}
```

### 2. Drawing Operation Limits

**No Hard Limits**, but practical constraints:
- 10,000+ drawing operations = noticeable lag
- 100,000+ operations = browser may freeze
- Typical OMR sheet: 400-1000 operations (manageable)

**Optimization for Large Sheets**:
```typescript
// Use requestAnimationFrame for chunked rendering
async function renderBubblesChunked(
    ctx: CanvasRenderingContext2D,
    bubbles: BubbleInterpretation[],
    chunkSize: number = 50
): Promise<void> {
    for (let i = 0; i < bubbles.length; i += chunkSize) {
        const chunk = bubbles.slice(i, i + chunkSize);

        await new Promise(resolve => {
            requestAnimationFrame(() => {
                for (const bubble of chunk) {
                    drawBubble(ctx, bubble);
                }
                resolve(undefined);
            });
        });
    }
}
```

---

## Summary of Key Constraints

| Constraint | Python | Browser | Mitigation |
|------------|--------|---------|------------|
| **Drawing Performance** | Fast (native C++) | Slower (JavaScript) | Use OffscreenCanvas, batch operations |
| **Answer Group Limit** | 4 edges (hardcoded) | 4 edges (same) | Use corner markers for 8+ groups |
| **Text Overflow** | No handling | No handling | Implement truncation or font scaling |
| **High DPI Support** | N/A | Canvas blurry | Use `devicePixelRatio` scaling |
| **Color Space** | BGR order | RGB order | Convert BGR → RGB |
| **Memory Usage** | 18MB/image | 24MB/canvas | Clear caches, use Web Workers |
| **Grayscale Verdicts** | Color-only | Color-only | Use fill patterns instead |
| **Canvas Support** | N/A | IE9+ | Feature detection + fallback |
| **Font Consistency** | OS fonts | OS fonts | Load custom web fonts |
| **Overlapping Bubbles** | No detection | No detection | Template validation |

**Browser Compatibility Target**:
- Minimum: Chrome 69+, Firefox 105+, Safari 14+
- Recommended: Latest versions of all major browsers
- Graceful degradation for older browsers (fallback to basic rectangles)
