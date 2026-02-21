# Math Utils Constraints

**Module**: Domain - Utils - Math
**Python Reference**: `src/utils/math.py`
**Last Updated**: 2026-02-21

---

## Class Design Constraints

### Static-Only Class

**Pattern**: All methods are `@staticmethod`

```python
class MathUtils:
    """Static-only class"""

    @staticmethod
    def distance(point1, point2):
        # No instance state, no self parameter
        return math.hypot(...)
```

**Constraints**:
- No instance variables
- No `__init__` method
- All methods are stateless
- No class-level mutable state

**Benefits**:
- Thread-safe by default
- No initialization required
- Clear API: `MathUtils.distance(p1, p2)`
- Suitable for concurrent use

**Browser Migration**: Can be implemented as plain functions or namespace object.

```typescript
// Option 1: Namespace object
const MathUtils = {
  distance(point1: Point, point2: Point): number {
    // ...
  }
};

// Option 2: Exported functions
export function distance(point1: Point, point2: Point): number {
  // ...
}
```

---

## Input Constraints

### Point Representation

**Type**: Tuple or List of two numbers

```python
# Valid point formats
point = (100, 200)        # Tuple
point = [100, 200]        # List
point = (100.5, 200.7)    # Float coordinates
point = [100, 200]        # Int coordinates
```

**Constraints**:
```python
# Point must have exactly 2 elements
len(point) == 2

# Elements must be numeric
isinstance(point[0], (int, float))
isinstance(point[1], (int, float))

# No constraints on range (can be negative, large, etc.)
```

**Coordinate System**:
- Origin (0, 0) is top-left (OpenCV convention)
- X increases rightward
- Y increases downward
- Coordinates can be negative (e.g., after shifting)

**Browser Migration**: Use arrays or objects

```typescript
// Option 1: Array (matches Python list)
type Point = [number, number];
const p: Point = [100, 200];

// Option 2: Object (more readable)
interface Point {
  x: number;
  y: number;
}
const p: Point = { x: 100, y: 200 };
```

### Edge Line Representation

**Type**: Tuple of two points

```python
edge_line = [(x1, y1), (x2, y2)]  # Start and end points
edge_line = [[100, 200], [150, 250]]
```

**Constraints**:
```python
len(edge_line) == 2
len(edge_line[0]) == 2
len(edge_line[1]) == 2
```

**Invariants**:
- Start and end can be the same (zero-length line)
- No ordering requirement (start/end are interchangeable for some operations)

### Rectangle Representation

**Format 1**: List of 4 points (ordered)

```python
rectangle = [
    [x1, y1],  # top-left
    [x2, y2],  # top-right
    [x3, y3],  # bottom-right
    [x4, y4]   # bottom-left
]
```

**Constraints**:
```python
len(rectangle) == 4  # Exactly 4 corners
# Points are ordered: tl, tr, br, bl
```

**Format 2**: 4-tuple (for containment check)

```python
rect = (x1, y1, x2, y2)  # (left, top, right, bottom)
```

**Constraints**:
```python
len(rect) == 4
x1 <= x2  # left <= right
y1 <= y2  # top <= bottom
```

### List of Points

**Type**: List or NumPy array

```python
points = [[x1, y1], [x2, y2], [x3, y3], ...]  # Python list
points = np.array([[x1, y1], [x2, y2], ...])  # NumPy array
```

**Constraints**:
```python
# Can be empty
len(points) >= 0

# Each point must have 2 elements
all(len(point) == 2 for point in points)
```

**Browser Migration**: Use regular arrays

```typescript
const points: Point[] = [
  [100, 200],
  [150, 250],
  [200, 300]
];
```

---

## Output Constraints

### Distance

**Type**: `float`
**Range**: `[0, ∞)`

**Constraints**:
```python
distance >= 0  # Always non-negative
distance == 0  # If and only if points are identical
```

**Precision**: Limited by floating-point precision (typically 15-17 decimal digits)

**Browser Migration**: JavaScript `number` has same precision as Python `float` (IEEE 754 double).

### Points (Modified/Generated)

**Type**: `list[float, float]` or `tuple[int, int]`

**Constraints**:
```python
# Lists (mutable)
point = [x, y]  # Can be modified
type(point) == list

# Tuples (immutable)
point = (x, y)  # Cannot be modified
type(point) == tuple
```

**Return Type Patterns**:
- `add_points()`: Returns `list[float, float]`
- `subtract_points()`: Returns `list[float, float]`
- `get_tuple_points()`: Returns `list[tuple[int, int]]`

**Browser Migration**: Always use arrays (JavaScript doesn't have tuples).

```typescript
function addPoints(origin: Point, point: Point): Point {
  return [origin[0] + point[0], origin[1] + point[1]];
}
```

### Ordered Rectangle

**Type**: `(np.ndarray, list[int])`

**Constraints**:
```python
# First element: ordered points
rect: np.ndarray
rect.shape == (4, 2)
rect.dtype == dtype parameter ("int" or "float")

# Second element: original indices
ordered_indices: list[int]
len(ordered_indices) == 4
all(0 <= i < 4 for i in ordered_indices)
```

**Ordering Guarantee**: Always returns points in order (tl, tr, br, bl)

**Browser Migration**: Return object with both arrays

```typescript
interface OrderedRectangle {
  rect: Point[];
  orderedIndices: number[];
}

function orderFourPoints(points: Point[]): OrderedRectangle {
  // ...
  return { rect, orderedIndices };
}
```

### Bounding Box

**Type**: `(np.ndarray, tuple[int, int])`

**Constraints**:
```python
# First element: bounding box points
bounding_box: np.ndarray
bounding_box.shape == (4, 2)
# Points ordered: (tl, tr, br, bl)

# Second element: dimensions
box_dimensions: tuple[int, int]
box_dimensions = (width, height)
width >= 0
height >= 0
```

**Invariants**:
```python
width == bounding_box[1][0] - bounding_box[0][0]
height == bounding_box[2][1] - bounding_box[1][1]
```

### Boolean Results

**Type**: `bool`
**Values**: `True` or `False`

**Methods**:
- `validate_rect()`: True if valid rectangle
- `check_max_cosine()`: True if angles acceptable
- `rectangle_contains()`: True if point inside
- `check_collinear_points()`: True if points on same line

**Browser Migration**: Direct equivalence (JavaScript `boolean`).

---

## Algorithm-Specific Constraints

### Distance Calculation

**Method**: `math.hypot(dx, dy)`

**Advantages**:
- More accurate than `sqrt(dx² + dy²)`
- Handles overflow/underflow better
- Built-in to Python standard library

**Constraints**:
```python
# Finite inputs required
assert not math.isnan(dx)
assert not math.isnan(dy)
assert not math.isinf(dx)
assert not math.isinf(dy)
```

**Browser Migration**: Use `Math.hypot()`

```typescript
function distance(p1: Point, p2: Point): number {
  const dx = p1[0] - p2[0];
  const dy = p1[1] - p2[1];
  return Math.hypot(dx, dy);  // Native browser support
}
```

**Browser Support**: `Math.hypot()` available in all modern browsers (ES2015).

### Order Four Points Algorithm

**Assumptions**:
- Points form a convex quadrilateral
- No three points are collinear
- Works best for rectangles or near-rectangles

**Algorithm Guarantees**:
```python
# Point with min sum is top-left
sum(tl) == min(sum(p) for p in points)

# Point with max sum is bottom-right
sum(br) == max(sum(p) for p in points)

# Point with min diff is top-right
diff(tr) == min(p[0] - p[1] for p in points)

# Point with max diff is bottom-left
diff(bl) == max(p[0] - p[1] for p in points)
```

**Edge Cases**:
```python
# Square: Algorithm still works (deterministic)
points = [[0, 0], [100, 0], [100, 100], [0, 100]]
# Returns: [[0, 0], [100, 0], [100, 100], [0, 100]]

# Non-convex quadrilateral: May produce unexpected ordering
# (Not designed for non-convex shapes)
```

**Browser Migration**: Same algorithm works with JavaScript arrays

```typescript
function orderFourPoints(points: Point[]): OrderedRectangle {
  const sumPoints = points.map(p => p[0] + p[1]);
  const diffPoints = points.map(p => p[0] - p[1]);

  const orderedIndices = [
    sumPoints.indexOf(Math.min(...sumPoints)),    // tl
    diffPoints.indexOf(Math.min(...diffPoints)),  // tr
    sumPoints.indexOf(Math.max(...sumPoints)),    // br
    diffPoints.indexOf(Math.max(...diffPoints))   // bl
  ];

  const rect = orderedIndices.map(i => points[i]);
  return { rect, orderedIndices };
}
```

### Rectangle Validation (Max Cosine)

**Threshold**: `MAX_COSINE = 0.35`

**Physical Meaning**:
```python
# Cosine of angle:
cos(0°) = 1.0    # Straight line (not a corner)
cos(90°) = 0.0   # Perfect right angle ✓
cos(180°) = -1.0 # Opposite direction

# MAX_COSINE = 0.35 corresponds to:
# angle = arccos(0.35) ≈ 69.5°
# or equivalently: 180° - 69.5° = 110.5°

# Accept angles in range: ~70° to ~110°
# Reject angles outside this range
```

**Constraint**:
```python
if max_cosine >= 0.35:
    # At least one angle too far from 90°
    return False  # Not rectangular
```

**Tuning**:
- **Stricter** (0.2): Only accept very rectangular shapes
- **Looser** (0.5): Accept more distorted quadrilaterals
- **Current** (0.35): Balanced for typical OMR sheets

**Browser Migration**: Use `Math.acos()` for debugging

```typescript
const MAX_COSINE = 0.35;

function checkMaxCosine(points: Point[]): boolean {
  let maxCosine = 0;

  for (let i = 2; i < 5; i++) {
    const cosine = Math.abs(
      angle(points[i % 4], points[i - 2], points[i - 1])
    );
    maxCosine = Math.max(cosine, maxCosine);
  }

  if (maxCosine >= MAX_COSINE) {
    console.warn("Quadrilateral is not a rectangle.");
    return false;
  }
  return true;
}
```

### Angle Calculation

**Formula**: Cosine of angle using dot product

```python
cos(θ) = (v1 · v2) / (|v1| × |v2|)
```

**Numerical Stability**:
```python
# Add small epsilon to prevent division by zero
denominator = magnitude_product + 1e-10

# For typical coordinates (0-10000), this is negligible
# For degenerate cases (zero-length vectors), prevents NaN
```

**Edge Cases**:
```python
# Zero-length vector
angle([0, 0], [0, 0], [0, 0])  # Returns 0.0 (due to epsilon)

# Very small vectors
angle([0, 0], [0.001, 0], [0, 0.001])  # Still works (epsilon)
```

**Browser Migration**: Same formula, same epsilon

```typescript
function angle(p1: Point, p2: Point, p0: Point): number {
  const dx1 = p1[0] - p0[0];
  const dy1 = p1[1] - p0[1];
  const dx2 = p2[0] - p0[0];
  const dy2 = p2[1] - p0[1];

  const dotProduct = dx1 * dx2 + dy1 * dy2;
  const magnitude = Math.sqrt(
    (dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10
  );

  return dotProduct / magnitude;
}
```

### Collinearity Check

**Algorithm**: Cross product == 0

```python
# Three points are collinear if:
(y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)
```

**Floating Point Concern**: Exact equality may fail for float coordinates

**Current Implementation**: Uses `==` (exact comparison)

**Limitation**:
```python
# May fail for nearly-collinear float points
p1 = [0.0, 0.0]
p2 = [5.0, 5.0]
p3 = [10.0, 10.00001]  # Nearly collinear

# Exact check may return False (floating-point error)
```

**Recommendation**: For production, consider epsilon-based comparison

```python
def check_collinear_points_robust(p1, p2, p3, epsilon=1e-6):
    cross_product = (
        (p1[1] - p2[1]) * (p1[0] - p3[0]) -
        (p1[1] - p3[1]) * (p1[0] - p2[0])
    )
    return abs(cross_product) < epsilon
```

**Browser Migration**: Same formula, consider epsilon

```typescript
function checkCollinearPoints(
  p1: Point,
  p2: Point,
  p3: Point,
  epsilon = 1e-6
): boolean {
  const crossProduct =
    (p1[1] - p2[1]) * (p1[0] - p3[0]) -
    (p1[1] - p3[1]) * (p1[0] - p2[0]);
  return Math.abs(crossProduct) < epsilon;
}
```

---

## Performance Constraints

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| distance | O(1) | Simple arithmetic |
| add_points | O(1) | Two additions |
| subtract_points | O(1) | Two subtractions |
| shift_points_from_origin | O(n) | n = number of points |
| get_point_on_line_by_ratio | O(1) | Linear interpolation |
| order_four_points | O(1) | Fixed 4 points |
| get_bounding_box_of_points | O(n) | Find min/max |
| validate_rect | O(1) | Check 4 corners |
| check_max_cosine | O(1) | 3 angle checks |
| angle | O(1) | Dot product |
| check_collinear_points | O(1) | Cross product |
| to_bgr | O(1) | Color conversion |
| chunks | O(n/k) | Lazy generator, k = chunk size |
| get_tuple_points | O(n) | Convert n points |

**Overall**: Most operations are O(1) or O(n) with small constants.

### Memory Constraints

**Python**:
```python
# Most operations create new lists (no mutation)
new_points = shift_points_from_origin(origin, points)
# Memory: O(n) new list

# NumPy operations
rect, indices = order_four_points(points)
# Memory: O(1) (4 points always)

# Generator (chunks)
gen = chunks(large_list, 100)
# Memory: O(1) until consumed
```

**Browser**:
```typescript
// Similar memory patterns
// Spread operator creates copies
const sortedValues = [...values].sort((a, b) => a - b);
// Memory: O(n)

// For large arrays, consider in-place operations
values.sort((a, b) => a - b);  // Mutates original
```

### Typical Performance

**Scenarios**:
```python
# Shift 100 field coordinates: ~10 μs
shift_points_from_origin(origin, 100 points)

# Order rectangle corners: ~1 μs
order_four_points(4 points)

# Validate rectangle: ~5 μs
validate_rect(approx)

# Distance calculation: ~0.1 μs
distance(p1, p2)
```

**Browser Performance**: Similar or slightly slower (interpreted vs compiled).

**Optimization**: For browser, use TypedArrays for bulk operations:

```typescript
// For large point arrays
const points = new Float64Array(n * 2);  // Flat [x1, y1, x2, y2, ...]
// ~2x faster than array of arrays
```

---

## Numerical Constraints

### Floating Point Precision

**Type**: `float` (Python) = IEEE 754 double (64-bit)

**Range**:
- Minimum: ~2.2 × 10⁻³⁰⁸
- Maximum: ~1.8 × 10³⁰⁸
- Precision: ~15-17 decimal digits

**Implications for OMR**:
```python
# Typical coordinate range: 0 to 10000 pixels
# Precision at 10000: ~10⁻¹¹ pixels (negligible)

# Distance calculation
distance = math.hypot(100, 200)  # Exact representation
distance = 223.606797749979      # 15 significant digits

# Comparison
assert distance == 223.606797749979  # May fail due to rounding
assert abs(distance - 223.606797749979) < 1e-10  # Better
```

**Browser**: Same precision (JavaScript `number`).

### Integer Conversion

**Pattern**: `int(float_value)` truncates toward zero

```python
int(100.9) == 100   # Truncate
int(-100.9) == -100 # Truncate toward zero

# For rounding:
round(100.9) == 101
round(100.5) == 100  # Banker's rounding (even)
```

**Use Cases**:
```python
# get_tuple_points: Convert float coordinates to int
tuple_points = get_tuple_points([[100.7, 200.9]])
# Returns: [(100, 200)]

# to_bgr: Scale 0-1 to 0-255
r = int(1.0 * 255)  # 255
```

**Browser**: `Math.trunc()` or `parseInt()`

```typescript
const x = Math.trunc(100.9);  // 100 (same as Python)
const y = Math.floor(100.9);  // 100 (rounds down)
const z = Math.round(100.9);  // 101 (rounds to nearest)
```

### Overflow/Underflow

**Not a concern for typical OMR coordinates**:
```python
# Typical ranges:
# - Image dimensions: 0 to 20000 pixels
# - Coordinates: -5000 to 20000 (can be negative after shifting)
# - Distances: 0 to 30000

# All well within float64 range
```

**Edge Cases**:
```python
# Very large coordinates (from incorrect processing)
point = (1e100, 1e100)
distance(point, (0, 0))  # Still works: 1.41e100

# Infinity
point = (float('inf'), 0)
distance(point, (0, 0))  # Returns inf
```

**Browser**: Same behavior with `Infinity`.

---

## Browser Migration Constraints

### Math API Equivalence

| Python | JavaScript | Notes |
|--------|-----------|-------|
| `math.hypot(x, y)` | `Math.hypot(x, y)` | Direct equivalence |
| `math.sqrt(x)` | `Math.sqrt(x)` | Direct equivalence |
| `abs(x)` | `Math.abs(x)` | Direct equivalence |
| `min(a, b)` | `Math.min(a, b)` | Direct equivalence |
| `max(a, b)` | `Math.max(a, b)` | Direct equivalence |
| `int(x)` | `Math.trunc(x)` | Truncate toward zero |
| `sorted(list)` | `[...list].sort()` | Copy then sort |

**All Math operations have direct browser equivalents.**

### NumPy Alternatives

**Python NumPy**:
```python
points = np.array([[1, 2], [3, 4]])
points.sum(axis=1)  # [3, 7]
np.diff(points, axis=1)  # [[-1], [-1]]
np.min(points, axis=0)  # [1, 2]
np.max(points, axis=0)  # [3, 4]
```

**Browser JavaScript**:
```typescript
const points = [[1, 2], [3, 4]];

// sum(axis=1)
const sums = points.map(p => p[0] + p[1]);  // [3, 7]

// diff(axis=1)
const diffs = points.map(p => p[0] - p[1]);  // [-1, -1]

// min(axis=0)
const minPoint = [
  Math.min(...points.map(p => p[0])),  // 1
  Math.min(...points.map(p => p[1]))   // 2
];

// max(axis=0)
const maxPoint = [
  Math.max(...points.map(p => p[0])),  // 3
  Math.max(...points.map(p => p[1]))   // 4
];
```

**No external library required for basic operations.**

**For complex operations**: Consider `ndarray.js` or `TensorFlow.js` (see `modules/technical/numpy/array-operations.md`).

### Color Conversion

**Python**: Uses `matplotlib.colors.to_rgb()`

```python
from matplotlib import colors

r, g, b = colors.to_rgb("red")  # (1.0, 0.0, 0.0)
```

**Browser**: Need color parsing library or manual implementation

```typescript
// Option 1: Manual for common colors
const COLOR_MAP: Record<string, [number, number, number]> = {
  "red": [1.0, 0.0, 0.0],
  "green": [0.0, 1.0, 0.0],
  "blue": [0.0, 0.0, 1.0],
  // ... add more as needed
};

// Option 2: Use tinycolor2 or chroma.js (small libraries)
import tinycolor from "tinycolor2";

function toBGR(anyColor: string): [number, number, number] {
  const rgb = tinycolor(anyColor).toRgb();
  const r = Math.round(rgb.r);
  const g = Math.round(rgb.g);
  const b = Math.round(rgb.b);
  return [b, g, r];  // BGR order for OpenCV.js
}
```

**Recommendation**: Use small color library (~2KB) or manual map for known colors.

### Type Safety

**TypeScript Interfaces**:
```typescript
type Point = [number, number];

interface Rectangle {
  topLeft: Point;
  topRight: Point;
  bottomRight: Point;
  bottomLeft: Point;
}

interface BoundingBoxResult {
  boundingBox: Point[];
  dimensions: [number, number];
}

interface OrderedRectangleResult {
  rect: Point[];
  orderedIndices: number[];
}
```

**Runtime Validation**:
```typescript
function validatePoint(point: unknown): asserts point is Point {
  if (!Array.isArray(point) || point.length !== 2) {
    throw new Error("Point must be array of length 2");
  }
  if (typeof point[0] !== "number" || typeof point[1] !== "number") {
    throw new Error("Point coordinates must be numbers");
  }
}
```

---

## Concurrency Constraints

### Thread Safety

**All methods are thread-safe**:
- No shared state
- No mutable class variables
- All methods are pure functions (same input → same output)

**Python**:
```python
# Safe for concurrent use
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    distances = executor.map(
        lambda p: MathUtils.distance(p, origin),
        points
    )
```

**Browser**:
```typescript
// Safe for Web Workers
// math-utils.worker.ts
self.onmessage = (e) => {
  const { points, origin } = e.data;
  const distances = points.map(p => distance(p, origin));
  self.postMessage(distances);
};
```

**No locks or synchronization required.**

---

## Testing Constraints

### Determinism

**All functions are deterministic**:
```python
# Same input always produces same output
assert distance(p1, p2) == distance(p1, p2)
```

**No randomness, no time-based logic, no external state.**

### Test Cases

**Recommended Test Coverage**:

```python
# Distance
test_distance_zero()       # Same point
test_distance_unit()       # Known distances
test_distance_negative()   # Negative coordinates

# Point arithmetic
test_add_points_zero()     # Add zero offset
test_add_points_positive() # Normal case
test_add_points_negative() # Negative coordinates

# Rectangle ordering
test_order_square()        # Perfect square
test_order_rectangle()     # Normal rectangle
test_order_rotated()       # Rotated rectangle

# Validation
test_validate_perfect_rect()     # 90° angles
test_validate_near_rect()        # ~85° angles
test_validate_not_rect()         # 60° angles

# Edge cases
test_collinear_horizontal() # Horizontal line
test_collinear_vertical()   # Vertical line
test_collinear_diagonal()   # Diagonal line
test_empty_list()           # Empty input
test_single_point()         # Single point
```

**Snapshot Testing**: Useful for complex operations

```python
def test_order_four_points_snapshot():
    points = [[300, 100], [100, 100], [300, 400], [100, 400]]
    rect, indices = MathUtils.order_four_points(points)

    # Snapshot expected output
    assert_snapshot(rect, "ordered_rect.json")
    assert_snapshot(indices, "ordered_indices.json")
```

---

## Summary of Critical Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Point format | `[x, y]` or `(x, y)` | OpenCV compatibility |
| Coordinate system | Origin top-left | OpenCV convention |
| Static methods | All methods | Thread-safe, stateless |
| MAX_COSINE | 0.35 | ~70° to ~110° angles accepted |
| Epsilon (angle) | 1e-10 | Prevent division by zero |
| Float precision | 15-17 digits | IEEE 754 double |
| Time complexity | O(1) or O(n) | Fast for all operations |
| Memory | O(n) or O(1) | Minimal overhead |
| Thread safety | Always safe | No shared state |
| Determinism | Always | Pure functions |
| Browser support | Math API | All modern browsers |

---

## Related Constraints

- **OpenCV Operations**: `modules/technical/opencv/opencv-operations.md`
- **NumPy Arrays**: `modules/technical/numpy/array-operations.md`
- **Geometry Utils**: `modules/domain/utils/geometry/constraints.md` (pending)
- **Drawing Utils**: `modules/domain/utils/drawing/constraints.md` (pending)
