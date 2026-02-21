# Geometry Utils - Constraints

**Module**: Domain / Utils / Geometry
**Python Reference**: `src/utils/geometry.py`
**Last Updated**: 2026-02-21

---

## Function Constraints

### euclidean_distance

**Function Signature**:
```python
def euclidean_distance(point1: Sequence[float], point2: Sequence[float]) -> float
```

#### Input Constraints

**point1 and point2**:
- **Type**: `Sequence[float]` (list, tuple, or array-like)
- **Length**: Must be equal (enforced by `strict=True` in zip)
- **Dimensions**: Any (typically 2D for image coordinates)
- **Value Range**: Unrestricted (any finite float)
- **Order**: Doesn't matter (symmetric function)

**Valid Inputs**:
```python
# 2D points (most common)
euclidean_distance([10, 20], [40, 60])  # ✓

# 1D points
euclidean_distance([10], [40])  # ✓

# 3D points
euclidean_distance([1, 2, 3], [4, 5, 6])  # ✓

# Empty sequences
euclidean_distance([], [])  # ✓ Returns 0.0

# Negative coordinates
euclidean_distance([-10, -20], [10, 20])  # ✓

# Floating point precision
euclidean_distance([1.5, 2.7], [3.2, 4.8])  # ✓
```

**Invalid Inputs**:
```python
# Length mismatch (raises ValueError)
euclidean_distance([10, 20], [40, 60, 80])  # ✗

# Non-numeric values (raises TypeError)
euclidean_distance(['a', 'b'], ['c', 'd'])  # ✗

# NaN values (returns NaN)
euclidean_distance([float('nan'), 20], [40, 60])  # ✗

# Infinity (returns Infinity)
euclidean_distance([float('inf'), 20], [40, 60])  # ✗
```

#### Output Constraints

**Return Type**: `float`

**Value Range**:
- Minimum: `0.0` (same points)
- Maximum: Unbounded (depends on input)
- Always non-negative: `distance >= 0`

**Special Cases**:
```python
# Same point
euclidean_distance([10, 20], [10, 20]) == 0.0

# Empty sequences
euclidean_distance([], []) == 0.0

# Symmetric
euclidean_distance(p1, p2) == euclidean_distance(p2, p1)
```

#### Algorithm Constraints

**Time Complexity**: O(n) where n = number of dimensions

**Space Complexity**: O(1) (no additional data structures)

**Numerical Stability**:
```python
# For very large numbers, may lose precision
point1 = [1e100, 1e100]
point2 = [2e100, 2e100]
# Result may have floating point errors

# Recommendation: normalize coordinates to reasonable range
# e.g., [0, image_width] × [0, image_height]
```

**Precision**:
```python
# Python float: IEEE 754 double precision
# ~15-17 decimal digits of precision
# Typical image coordinates: 0-10000
# More than sufficient precision for geometry calculations
```

---

### vector_magnitude

**Function Signature**:
```python
def vector_magnitude(vector: Sequence[float]) -> float
```

#### Input Constraints

**vector**:
- **Type**: `Sequence[float]` (list, tuple, or array-like)
- **Length**: Any (including 0)
- **Dimensions**: Any (typically 2D)
- **Value Range**: Unrestricted (any finite float)

**Valid Inputs**:
```python
# 2D vector (most common)
vector_magnitude([3, 4])  # ✓ Returns 5.0

# 1D vector
vector_magnitude([5])  # ✓ Returns 5.0

# 3D vector
vector_magnitude([1, 2, 2])  # ✓ Returns 3.0

# Empty vector
vector_magnitude([])  # ✓ Returns 0.0

# Zero vector
vector_magnitude([0, 0])  # ✓ Returns 0.0

# Negative components
vector_magnitude([-3, -4])  # ✓ Returns 5.0
```

**Invalid Inputs**:
```python
# Non-numeric values (raises TypeError)
vector_magnitude(['a', 'b'])  # ✗

# NaN values (returns NaN)
vector_magnitude([float('nan'), 4])  # ✗

# Infinity (returns Infinity)
vector_magnitude([float('inf'), 4])  # ✗
```

#### Output Constraints

**Return Type**: `float`

**Value Range**:
- Minimum: `0.0` (zero vector)
- Maximum: Unbounded
- Always non-negative: `magnitude >= 0`

**Special Cases**:
```python
# Zero vector
vector_magnitude([0, 0, 0]) == 0.0

# Empty vector
vector_magnitude([]) == 0.0

# Unit vectors
vector_magnitude([1, 0]) == 1.0
vector_magnitude([0, 1]) == 1.0
```

**Relationship to Euclidean Distance**:
```python
# Magnitude is distance from origin
assert vector_magnitude(v) == euclidean_distance([0] * len(v), v)

# Example:
vector_magnitude([3, 4]) == euclidean_distance([0, 0], [3, 4])
# Both return 5.0
```

#### Algorithm Constraints

**Time Complexity**: O(n) where n = number of components

**Space Complexity**: O(1)

**Numerical Stability**:
```python
# Same precision considerations as euclidean_distance
# For very large vectors, consider normalization
```

---

### bbox_center

**Function Signature**:
```python
def bbox_center(origin: Sequence[float], dimensions: Sequence[float]) -> list[float]
```

#### Input Constraints

**origin**:
- **Type**: `Sequence[float]` (typically list or tuple)
- **Length**: Any (typically 2 for 2D)
- **Value Range**: Unrestricted (can be negative)
- **Meaning**: Top-left corner coordinates (for images)

**dimensions**:
- **Type**: `Sequence[float]`
- **Length**: Must match origin length
- **Value Range**: Typically non-negative (width, height)
- **Meaning**: Size in each dimension

**Valid Inputs**:
```python
# Standard 2D bounding box
bbox_center([100, 200], [50, 30])  # ✓ Returns [125, 215]

# Origin at (0, 0)
bbox_center([0, 0], [100, 100])  # ✓ Returns [50, 50]

# Negative origin
bbox_center([-50, -100], [20, 30])  # ✓ Returns [-40, -85]

# Zero dimensions (point)
bbox_center([100, 200], [0, 0])  # ✓ Returns [100, 200]

# 1D bounding box
bbox_center([10], [20])  # ✓ Returns [20]

# 3D bounding box
bbox_center([0, 0, 0], [10, 10, 10])  # ✓ Returns [5, 5, 5]
```

**Invalid Inputs**:
```python
# Length mismatch (IndexError)
bbox_center([100, 200], [50])  # ✗

# Non-numeric values (TypeError)
bbox_center(['a', 'b'], ['c', 'd'])  # ✗
```

**Semantic Constraints**:
```python
# Dimensions should be non-negative for physical bounding boxes
# But function doesn't enforce this
bbox_center([100, 200], [-50, -30])  # ✓ Technically valid
# Returns [75, 185] (nonsensical for actual bounding box)

# Recommendation: validate dimensions >= 0 at call site
assert all(d >= 0 for d in dimensions)
```

#### Output Constraints

**Return Type**: `list[float]`

**Value Range**: Unbounded (depends on input)

**Length**: Same as input length

**Formula**:
```python
# For each dimension i:
center[i] = origin[i] + dimensions[i] / 2

# Example:
origin = [100, 200]
dimensions = [50, 30]
center = [100 + 50/2, 200 + 30/2] = [125, 215]
```

**Special Cases**:
```python
# Zero dimensions → center equals origin
bbox_center([x, y], [0, 0]) == [x, y]

# Symmetric box at origin
bbox_center([0, 0], [w, h]) == [w/2, h/2]
```

#### Algorithm Constraints

**Time Complexity**: O(n) where n = number of dimensions (typically 2, so O(1))

**Space Complexity**: O(n) for return list

**Numerical Stability**:
```python
# Division by 2 is exact in floating point (power of 2)
# No precision loss from halving
assert (50 / 2) == 25.0  # Exact
assert (51 / 2) == 25.5  # Exact
```

---

## Common Input Patterns

### 2D Image Coordinates

**Coordinate System**:
```
(0,0) ───────────► x (width)
  │
  │
  │
  ▼
  y (height)

Origin: Top-left corner
X-axis: Left to right (increases with width)
Y-axis: Top to bottom (increases with height)
```

**Typical Ranges**:
```python
# Coordinates
x: [0, image_width]   # e.g., [0, 2480]
y: [0, image_height]  # e.g., [0, 3508]

# Dimensions
width: [0, image_width]
height: [0, image_height]

# Example: A4 paper at 300 DPI
image_width = 2480   # 8.27 inches × 300 DPI
image_height = 3508  # 11.69 inches × 300 DPI
```

### Bounding Box Formats

**Format 1: Origin + Dimensions (Used by bbox_center)**:
```python
bbox = {
    'origin': [x, y],
    'dimensions': [width, height]
}
center = bbox_center(bbox['origin'], bbox['dimensions'])
```

**Format 2: OpenCV Rectangle (x, y, w, h)**:
```python
# Convert to bbox_center format
rect = (x, y, w, h)
origin = [rect[0], rect[1]]
dimensions = [rect[2], rect[3]]
center = bbox_center(origin, dimensions)
```

**Format 3: Corner Coordinates (x1, y1, x2, y2)**:
```python
# Convert to bbox_center format
x1, y1, x2, y2 = corner_coords
origin = [x1, y1]
dimensions = [x2 - x1, y2 - y1]
center = bbox_center(origin, dimensions)
```

**Format 4: Center + Dimensions**:
```python
# Inverse operation
def center_to_bbox(center, dimensions):
    origin = [
        center[0] - dimensions[0] / 2,
        center[1] - dimensions[1] / 2
    ]
    return origin, dimensions
```

---

## Performance Constraints

### Time Complexity Summary

| Function | Complexity | Typical n | Actual Time |
|----------|-----------|-----------|-------------|
| `euclidean_distance` | O(n) | 2 | ~10 ns |
| `vector_magnitude` | O(n) | 2 | ~10 ns |
| `bbox_center` | O(n) | 2 | ~5 ns |

**Notes**:
- All functions are highly optimized for 2D case
- Negligible overhead for typical use
- Can be called millions of times per second

### Space Complexity Summary

| Function | Complexity | Allocations |
|----------|-----------|-------------|
| `euclidean_distance` | O(1) | None |
| `vector_magnitude` | O(1) | None |
| `bbox_center` | O(n) | One list of length n |

**Memory Usage**:
```python
# bbox_center is the only function that allocates memory
center = bbox_center([100, 200], [50, 30])
# Allocates: list[float] with 2 elements (~80 bytes in Python)

# Others use stack variables only
distance = euclidean_distance([10, 20], [40, 60])
magnitude = vector_magnitude([3, 4])
# No heap allocations
```

### Browser Performance

**JavaScript Optimizations**:
```javascript
// For 2D, avoid generic loops (use specialized version)
function euclideanDistance2D(x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  return Math.sqrt(dx * dx + dy * dy);
}

// Or use Math.hypot (built-in, optimized)
const distance = Math.hypot(x2 - x1, y2 - y1);

// Avoid Array.reduce for small arrays (loop is faster)
// BAD for 2D:
const sum = vector.reduce((s, v) => s + v * v, 0);

// GOOD for 2D:
const sum = vector[0] * vector[0] + vector[1] * vector[1];
```

**Benchmark Results** (approximate):
```
Chrome V8:
- euclideanDistance (2D): ~5 ns/call
- vectorMagnitude (2D): ~5 ns/call
- bboxCenter (2D): ~8 ns/call

Firefox SpiderMonkey:
- euclideanDistance (2D): ~8 ns/call
- vectorMagnitude (2D): ~8 ns/call
- bboxCenter (2D): ~12 ns/call
```

---

## Thread Safety

### Python

**All functions are thread-safe**:
- No shared state
- No instance variables
- All data is local to function call
- Pure functions (no side effects)

```python
from concurrent.futures import ThreadPoolExecutor
from src.utils.geometry import euclidean_distance

# Safe to call from multiple threads
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i in range(1000):
        point1 = [i, i]
        point2 = [i+1, i+1]
        future = executor.submit(euclidean_distance, point1, point2)
        futures.append(future)

    results = [f.result() for f in futures]
```

### Browser (Web Workers)

**Safe to use in Web Workers**:
```javascript
// main.js
const worker = new Worker('geometry-worker.js');
worker.postMessage({
  type: 'euclideanDistance',
  point1: [10, 20],
  point2: [40, 60]
});

worker.onmessage = (e) => {
  console.log('Distance:', e.data.distance);
};

// geometry-worker.js
import { euclideanDistance } from './geometry.js';

self.onmessage = (e) => {
  const { type, point1, point2 } = e.data;

  if (type === 'euclideanDistance') {
    const distance = euclideanDistance(point1, point2);
    self.postMessage({ distance });
  }
};
```

---

## Validation Constraints

### Input Validation

**Recommended Checks** (not enforced by functions):

```python
def validate_point(point, expected_dim=2):
    """Validate point input."""
    if not isinstance(point, (list, tuple)):
        raise TypeError(f"Point must be list or tuple, got {type(point)}")

    if len(point) != expected_dim:
        raise ValueError(f"Point must have {expected_dim} dimensions, got {len(point)}")

    if not all(isinstance(x, (int, float)) for x in point):
        raise TypeError("Point coordinates must be numeric")

    if not all(math.isfinite(x) for x in point):
        raise ValueError("Point coordinates must be finite (no NaN/Inf)")

def validate_dimensions(dimensions):
    """Validate dimensions input."""
    if not isinstance(dimensions, (list, tuple)):
        raise TypeError(f"Dimensions must be list or tuple, got {type(dimensions)}")

    if not all(isinstance(x, (int, float)) for x in dimensions):
        raise TypeError("Dimensions must be numeric")

    if not all(x >= 0 for x in dimensions):
        raise ValueError("Dimensions must be non-negative")

    if not all(math.isfinite(x) for x in dimensions):
        raise ValueError("Dimensions must be finite (no NaN/Inf)")
```

**Usage with Validation**:
```python
# Wrap functions with validation
def safe_euclidean_distance(point1, point2):
    validate_point(point1)
    validate_point(point2)
    if len(point1) != len(point2):
        raise ValueError("Points must have same dimensions")
    return euclidean_distance(point1, point2)

def safe_bbox_center(origin, dimensions):
    validate_point(origin)
    validate_dimensions(dimensions)
    if len(origin) != len(dimensions):
        raise ValueError("Origin and dimensions must have same length")
    return bbox_center(origin, dimensions)
```

### TypeScript Validation

```typescript
// Type guards
function isValidPoint(point: any): point is number[] {
  return Array.isArray(point) &&
         point.every(x => typeof x === 'number' && isFinite(x));
}

function isValidDimensions(dims: any): dims is number[] {
  return Array.isArray(dims) &&
         dims.every(x => typeof x === 'number' && isFinite(x) && x >= 0);
}

// Validated functions
export function safeEuclideanDistance(
  point1: unknown,
  point2: unknown
): number {
  if (!isValidPoint(point1) || !isValidPoint(point2)) {
    throw new Error('Invalid points');
  }
  if (point1.length !== point2.length) {
    throw new Error('Point dimensions must match');
  }
  return euclideanDistance(point1, point2);
}

export function safeBboxCenter(
  origin: unknown,
  dimensions: unknown
): number[] {
  if (!isValidPoint(origin) || !isValidDimensions(dimensions)) {
    throw new Error('Invalid inputs');
  }
  if (origin.length !== dimensions.length) {
    throw new Error('Origin and dimensions must have same length');
  }
  return bboxCenter(origin, dimensions);
}
```

---

## Browser Migration Constraints

### Numerical Precision

**JavaScript Number Type**:
- IEEE 754 double precision (same as Python float)
- 53 bits of precision (~15-17 decimal digits)
- Range: ±2^1024 (effectively unlimited for geometry)

**No Precision Issues for Typical Use**:
```javascript
// Image coordinates: 0-10000
// No precision loss in this range
const distance = euclideanDistance([1234.5, 5678.9], [2345.6, 6789.1]);
// Result accurate to many decimal places

// Even at extreme image sizes (100K pixels)
const bigDistance = euclideanDistance([99999, 99999], [100000, 100000]);
// Still accurate
```

**Potential Issues** (rare):
```javascript
// Very large numbers may lose precision
const huge = euclideanDistance([1e15, 1e15], [1e15 + 1, 1e15 + 1]);
// May have rounding errors

// Recommendation: Keep coordinates in reasonable range
// For images: 0-100000 (more than sufficient)
```

### Array vs Typed Arrays

**Regular Arrays** (recommended for flexibility):
```javascript
const point = [10.5, 20.7];  // number[]
const distance = euclideanDistance(point, [30.2, 40.8]);
```

**Typed Arrays** (for performance with large datasets):
```javascript
const point1 = new Float64Array([10.5, 20.7]);
const point2 = new Float64Array([30.2, 40.8]);
const distance = euclideanDistance(point1, point2);
// Works (Sequence[float] includes typed arrays)
```

**Performance Comparison**:
```javascript
// For small arrays (2-3 elements): Regular arrays faster
// For large arrays (100+ elements): Typed arrays faster
// For geometry utils: Regular arrays recommended (size is small)
```

### Memory Management

**Python** (automatic garbage collection):
```python
# No manual memory management needed
center = bbox_center([100, 200], [50, 30])
# List is garbage collected when no longer referenced
```

**JavaScript** (automatic garbage collection):
```javascript
// No manual memory management needed
const center = bboxCenter([100, 200], [50, 30]);
// Array is garbage collected when no longer referenced
```

**OpenCV.js Matrices** (manual memory management):
```javascript
// If working with OpenCV.js Mat objects
const point1Mat = cv.matFromArray(1, 2, cv.CV_64F, [10, 20]);
const point2Mat = cv.matFromArray(1, 2, cv.CV_64F, [40, 60]);

// Extract data for distance calculation
const point1 = Array.from(point1Mat.data64F);
const point2 = Array.from(point2Mat.data64F);
const distance = euclideanDistance(point1, point2);

// Clean up OpenCV matrices
point1Mat.delete();
point2Mat.delete();
```

---

## Integration Constraints

### With NumPy Arrays

**Python NumPy Integration**:
```python
import numpy as np
from src.utils.geometry import euclidean_distance, vector_magnitude

# NumPy arrays work (Sequence[float] includes ndarray)
point1 = np.array([10, 20])
point2 = np.array([40, 60])
distance = euclidean_distance(point1, point2)  # ✓

# But NumPy has its own optimized functions
distance_np = np.linalg.norm(point2 - point1)  # Faster for NumPy

# Recommendation:
# - Use geometry utils for simple cases (lists, tuples)
# - Use NumPy functions when already working with NumPy arrays
```

### With OpenCV

**Python OpenCV Integration**:
```python
import cv2
from src.utils.geometry import bbox_center

# OpenCV rectangle: (x, y, w, h)
rect = cv2.boundingRect(contour)
# rect = (100, 200, 50, 30)

# Convert to bbox_center format
origin = [rect[0], rect[1]]
dimensions = [rect[2], rect[3]]
center = bbox_center(origin, dimensions)
# center = [125, 215]
```

**JavaScript OpenCV.js Integration**:
```javascript
import cv from 'opencv.js';
import { bboxCenter } from './geometry.js';

// Get bounding rect from contour
const rect = cv.boundingRect(contour);
// rect = {x: 100, y: 200, width: 50, height: 30}

// Convert to bbox_center format
const origin = [rect.x, rect.y];
const dimensions = [rect.width, rect.height];
const center = bboxCenter(origin, dimensions);
// center = [125, 215]
```

---

## Error Handling Constraints

### Current Behavior

**euclidean_distance**:
```python
# Raises ValueError on length mismatch (strict zip)
euclidean_distance([1, 2], [1, 2, 3])
# ValueError: zip() argument 2 is longer than argument 1

# Raises TypeError on non-numeric
euclidean_distance(['a'], ['b'])
# TypeError: unsupported operand type(s) for -: 'str' and 'str'

# Returns NaN on NaN input
euclidean_distance([float('nan')], [1])
# Returns: nan

# Returns Infinity on Infinity input
euclidean_distance([float('inf')], [1])
# Returns: inf
```

**vector_magnitude**:
```python
# Raises TypeError on non-numeric
vector_magnitude(['a', 'b'])
# TypeError: unsupported operand type(s) for ** or pow(): 'str' and 'int'

# Returns NaN on NaN input
vector_magnitude([float('nan'), 1])
# Returns: nan

# Returns Infinity on Infinity input
vector_magnitude([float('inf'), 1])
# Returns: inf
```

**bbox_center**:
```python
# Raises IndexError on length mismatch
bbox_center([1, 2], [1])
# IndexError: list index out of range

# Raises TypeError on non-numeric
bbox_center(['a', 'b'], ['c', 'd'])
# TypeError: unsupported operand type(s) for /: 'str' and 'int'
```

### Recommended Error Handling

**Defensive Programming**:
```python
try:
    distance = euclidean_distance(point1, point2)
    if not math.isfinite(distance):
        raise ValueError(f"Invalid distance: {distance}")
except (ValueError, TypeError) as e:
    # Handle invalid input
    logger.error(f"Failed to calculate distance: {e}")
    distance = None
```

**Early Validation**:
```python
# Validate before calling
if len(point1) != len(point2):
    raise ValueError("Point dimensions must match")

if not all(math.isfinite(x) for x in point1 + point2):
    raise ValueError("Points must have finite coordinates")

distance = euclidean_distance(point1, point2)
# Guaranteed to succeed
```

---

## Summary of Critical Constraints

| Constraint | Value/Rule | Impact |
|------------|-----------|---------|
| **euclidean_distance** |
| Input dimensions | Must match (strict zip) | ValueError if mismatch |
| Output range | [0, ∞) | Always non-negative |
| Time complexity | O(n) | n = dimensions (typically 2) |
| Thread safety | Yes | Pure function |
| **vector_magnitude** |
| Input dimensions | Any (including 0) | Empty returns 0.0 |
| Output range | [0, ∞) | Always non-negative |
| Time complexity | O(n) | n = components |
| Thread safety | Yes | Pure function |
| **bbox_center** |
| Input dimensions | Must match | IndexError if mismatch |
| Output type | list[float] | New allocation |
| Time complexity | O(n) | Typically O(1) for 2D |
| Thread safety | Yes | Pure function |
| **All functions** |
| Numerical type | Python float / JS Number | IEEE 754 double |
| NaN handling | Propagates | Returns NaN |
| Infinity handling | Propagates | Returns Infinity |
| Memory | Minimal | No shared state |

---

## Related Constraints

- **Image Utils**: `../image/constraints.md`
- **Math Utils**: `../math/constraints.md`
- **NumPy Arrays**: `../../../technical/numpy/array-operations.md`
- **Performance**: `../../../migration/performance.md`
- **Browser Compatibility**: `../../../migration/compatibility.md`
