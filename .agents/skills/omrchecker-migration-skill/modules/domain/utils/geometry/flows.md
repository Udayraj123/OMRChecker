# Geometry Utils - Flows

**Module**: Domain / Utils / Geometry
**Python Reference**: `src/utils/geometry.py`
**Last Updated**: 2026-02-21

---

## Overview

Geometry utility functions provide common geometric calculations used across ML processors for detection and alignment tasks. These utilities maintain consistency and reduce code duplication by centralizing fundamental geometric operations.

**Use Cases**:
- Calculate distances between detected features
- Measure vector magnitudes for alignment
- Find bounding box centers for field positioning

---

## Function Flows

### 1. Euclidean Distance

**Purpose**: Calculate the straight-line distance between two points in 2D space.

**Function Signature**:
```python
def euclidean_distance(point1: Sequence[float], point2: Sequence[float]) -> float
```

**Flow**:
```
START: euclidean_distance(point1, point2)
│
├─► STEP 1: Pair Coordinates
│   │
│   zip(point1, point2, strict=True)
│   │
│   │ Example:
│   │ point1 = [10, 20]
│   │ point2 = [40, 60]
│   │ pairs = [(10, 40), (20, 60)]
│
├─► STEP 2: Calculate Squared Differences
│   │
│   For each (a, b) in pairs:
│   │   diff_squared = (a - b) ** 2
│   │
│   │ Example:
│   │ (10 - 40)² = (-30)² = 900
│   │ (20 - 60)² = (-40)² = 1600
│
├─► STEP 3: Sum Squared Differences
│   │
│   sum_of_squares = sum(diff_squared)
│   │
│   │ Example:
│   │ 900 + 1600 = 2500
│
└─► STEP 4: Return Square Root
    │
    distance = sum_of_squares ** 0.5
    │
    │ Example:
    │ √2500 = 50.0
    │
    RETURN 50.0

END
```

**Mathematical Formula**:
```
d = √[(x₂ - x₁)² + (y₂ - y₁)²]

For n-dimensional points:
d = √[Σᵢ(aᵢ - bᵢ)²]
```

**Usage Example**:
```python
# Calculate distance between two bubble centers
bubble1_center = [100, 150]
bubble2_center = [100, 200]
distance = euclidean_distance(bubble1_center, bubble2_center)
# distance = 50.0 (vertical distance)

# Calculate distance between field corners
corner1 = [10, 10]
corner2 = [40, 40]
distance = euclidean_distance(corner1, corner2)
# distance = 42.426... (diagonal distance)
```

**Common Use Cases**:
- Measuring separation between detected bubbles
- Calculating nearest neighbor distances in SIFT matching
- Validating field dimensions after warping
- Detecting outlier points in alignment

---

### 2. Vector Magnitude

**Purpose**: Calculate the length (magnitude) of a vector.

**Function Signature**:
```python
def vector_magnitude(vector: Sequence[float]) -> float
```

**Flow**:
```
START: vector_magnitude(vector)
│
├─► STEP 1: Square Each Component
│   │
│   For each x in vector:
│   │   squared = x ** 2
│   │
│   │ Example:
│   │ vector = [3, 4]
│   │ squared_components = [9, 16]
│
├─► STEP 2: Sum All Squares
│   │
│   sum_of_squares = sum(squared)
│   │
│   │ Example:
│   │ 9 + 16 = 25
│
└─► STEP 3: Return Square Root
    │
    magnitude = sum_of_squares ** 0.5
    │
    │ Example:
    │ √25 = 5.0
    │
    RETURN 5.0

END
```

**Mathematical Formula**:
```
For 2D vector [dx, dy]:
magnitude = √(dx² + dy²)

For n-dimensional vector:
magnitude = √(Σᵢ xᵢ²)
```

**Usage Example**:
```python
# Calculate displacement magnitude
displacement_vector = [3, 4]
magnitude = vector_magnitude(displacement_vector)
# magnitude = 5.0

# Calculate shift magnitude from alignment
shift_vector = [10, 15]
shift_distance = vector_magnitude(shift_vector)
# shift_distance = 18.027...

# Unit vector calculation (normalize)
vector = [3, 4]
mag = vector_magnitude(vector)
unit_vector = [x / mag for x in vector]
# unit_vector = [0.6, 0.8]
```

**Common Use Cases**:
- Measuring alignment shift distances
- Normalizing vectors to unit length
- Calculating feature descriptor magnitudes
- Validating perspective transform scales

**Relationship to Euclidean Distance**:
```python
# These are equivalent:
magnitude = vector_magnitude([dx, dy])
distance = euclidean_distance([0, 0], [dx, dy])

# vector_magnitude is essentially distance from origin
```

---

### 3. Bounding Box Center

**Purpose**: Calculate the center point of a bounding box given its origin and dimensions.

**Function Signature**:
```python
def bbox_center(origin: Sequence[float], dimensions: Sequence[float]) -> list[float]
```

**Flow**:
```
START: bbox_center(origin, dimensions)
│
├─► STEP 1: Calculate X-Center
│   │
│   center_x = origin[0] + dimensions[0] / 2
│   │
│   │ Example:
│   │ origin = [100, 200]
│   │ dimensions = [50, 30]
│   │ center_x = 100 + 50/2 = 100 + 25 = 125
│
├─► STEP 2: Calculate Y-Center
│   │
│   center_y = origin[1] + dimensions[1] / 2
│   │
│   │ Example:
│   │ center_y = 200 + 30/2 = 200 + 15 = 215
│
└─► STEP 3: Return Center Point
    │
    center = [center_x, center_y]
    │
    │ Example:
    │ center = [125, 215]
    │
    RETURN [125, 215]

END
```

**Mathematical Formula**:
```
For 2D bounding box:
center_x = origin_x + width / 2
center_y = origin_y + height / 2

General formula:
centerᵢ = originᵢ + dimensionsᵢ / 2
```

**Bounding Box Representation**:
```
Origin (x, y)
    ├──────────────┐
    │              │  height
    │   • center   │
    │              │
    └──────────────┘
        width

origin = [x, y]
dimensions = [width, height]
center = [x + width/2, y + height/2]
```

**Usage Example**:
```python
# Calculate bubble center from detection results
bubble_origin = [100, 200]  # Top-left corner
bubble_size = [20, 20]      # Width, height
center = bbox_center(bubble_origin, bubble_size)
# center = [110, 210]

# Calculate field block center
field_origin = [50, 100]
field_dimensions = [200, 300]
center = bbox_center(field_origin, field_dimensions)
# center = [150, 250]

# ML detection bounding box center
detected_box = {
    'origin': [30, 40],
    'dimensions': [60, 80]
}
center = bbox_center(detected_box['origin'], detected_box['dimensions'])
# center = [60, 80]
```

**Common Use Cases**:
- Converting ML detection boxes to center points
- Calculating field block centroids
- Finding bubble centers for threshold analysis
- Positioning labels and annotations
- Computing centers for alignment reference points

**Coordinate Systems**:
```python
# OpenCV/Image coordinates (origin at top-left)
origin = [x, y]  # x: right, y: down
center = bbox_center(origin, [w, h])

# Typical usage with OpenCV rectangle
rect = (x, y, w, h)  # OpenCV rectangle format
origin = [rect[0], rect[1]]
dimensions = [rect[2], rect[3]]
center = bbox_center(origin, dimensions)
```

---

## Integration Examples

### Example 1: Bubble Clustering by Distance

```python
from src.utils.geometry import euclidean_distance

def cluster_bubbles_by_distance(bubble_centers, max_distance):
    """Group bubbles that are close together."""
    clusters = []

    for center in bubble_centers:
        # Find nearest cluster
        nearest_cluster = None
        min_dist = float('inf')

        for cluster in clusters:
            # Calculate distance to cluster centroid
            centroid = calculate_centroid(cluster)
            dist = euclidean_distance(center, centroid)

            if dist < min_dist:
                min_dist = dist
                nearest_cluster = cluster

        # Add to nearest cluster or create new one
        if nearest_cluster and min_dist < max_distance:
            nearest_cluster.append(center)
        else:
            clusters.append([center])

    return clusters
```

### Example 2: Validate Field Dimensions

```python
from src.utils.geometry import vector_magnitude

def validate_field_dimensions(field_origin, field_dimensions, expected_size):
    """Check if field dimensions are within expected range."""
    actual_magnitude = vector_magnitude(field_dimensions)
    expected_magnitude = vector_magnitude(expected_size)

    # Allow 10% tolerance
    tolerance = 0.1
    ratio = actual_magnitude / expected_magnitude

    is_valid = (1 - tolerance) <= ratio <= (1 + tolerance)

    return {
        'valid': is_valid,
        'actual_magnitude': actual_magnitude,
        'expected_magnitude': expected_magnitude,
        'ratio': ratio
    }
```

### Example 3: Convert ML Detections to Centers

```python
from src.utils.geometry import bbox_center

def process_ml_detections(detections):
    """Convert ML bounding boxes to center points."""
    field_centers = []

    for detection in detections:
        # ML detection format: {x, y, w, h, confidence, class}
        origin = [detection['x'], detection['y']]
        dimensions = [detection['w'], detection['h']]

        center = bbox_center(origin, dimensions)

        field_centers.append({
            'center': center,
            'confidence': detection['confidence'],
            'class': detection['class']
        })

    return field_centers
```

### Example 4: Alignment Shift Validation

```python
from src.utils.geometry import vector_magnitude

def validate_alignment_shift(shift_vector, max_shift):
    """Check if alignment shift is within acceptable range."""
    shift_magnitude = vector_magnitude(shift_vector)

    if shift_magnitude > max_shift:
        return {
            'valid': False,
            'magnitude': shift_magnitude,
            'reason': f'Shift {shift_magnitude:.1f}px exceeds maximum {max_shift}px'
        }

    return {
        'valid': True,
        'magnitude': shift_magnitude
    }
```

---

## Browser Migration

### JavaScript Implementation

```javascript
// geometry.js

/**
 * Calculate Euclidean distance between two points
 * @param {number[]} point1 - First point [x, y]
 * @param {number[]} point2 - Second point [x, y]
 * @returns {number} Distance
 */
export function euclideanDistance(point1, point2) {
  if (point1.length !== point2.length) {
    throw new Error('Points must have same dimensions');
  }

  let sumOfSquares = 0;
  for (let i = 0; i < point1.length; i++) {
    const diff = point1[i] - point2[i];
    sumOfSquares += diff * diff;
  }

  return Math.sqrt(sumOfSquares);
}

/**
 * Calculate vector magnitude (length)
 * @param {number[]} vector - Vector components
 * @returns {number} Magnitude
 */
export function vectorMagnitude(vector) {
  let sumOfSquares = 0;
  for (const component of vector) {
    sumOfSquares += component * component;
  }

  return Math.sqrt(sumOfSquares);
}

/**
 * Calculate bounding box center point
 * @param {number[]} origin - Box origin [x, y]
 * @param {number[]} dimensions - Box dimensions [width, height]
 * @returns {number[]} Center point [x, y]
 */
export function bboxCenter(origin, dimensions) {
  return [
    origin[0] + dimensions[0] / 2,
    origin[1] + dimensions[1] / 2
  ];
}
```

### TypeScript Implementation

```typescript
// geometry.ts

/**
 * Calculate Euclidean distance between two points
 */
export function euclideanDistance(
  point1: readonly number[],
  point2: readonly number[]
): number {
  if (point1.length !== point2.length) {
    throw new Error(`Point dimensions mismatch: ${point1.length} vs ${point2.length}`);
  }

  const sumOfSquares = point1.reduce((sum, val, i) => {
    const diff = val - point2[i];
    return sum + diff * diff;
  }, 0);

  return Math.sqrt(sumOfSquares);
}

/**
 * Calculate vector magnitude (length)
 */
export function vectorMagnitude(vector: readonly number[]): number {
  const sumOfSquares = vector.reduce((sum, val) => sum + val * val, 0);
  return Math.sqrt(sumOfSquares);
}

/**
 * Calculate bounding box center point
 */
export function bboxCenter(
  origin: readonly number[],
  dimensions: readonly number[]
): number[] {
  return [
    origin[0] + dimensions[0] / 2,
    origin[1] + dimensions[1] / 2
  ];
}

// Type definitions
export type Point2D = readonly [number, number];
export type Vector2D = readonly [number, number];
export type BBox2D = {
  readonly origin: Point2D;
  readonly dimensions: readonly [number, number];
};

// Typed variants
export function euclideanDistance2D(point1: Point2D, point2: Point2D): number {
  return euclideanDistance(point1, point2);
}

export function bboxCenter2D(bbox: BBox2D): Point2D {
  const center = bboxCenter(bbox.origin, bbox.dimensions);
  return [center[0], center[1]];
}
```

### Key Migration Differences

1. **Array Operations**:
   ```javascript
   // Python: generator expression with zip
   sum((a - b) ** 2 for a, b in zip(point1, point2, strict=True)) ** 0.5

   // JavaScript: reduce or loop
   Math.sqrt(point1.reduce((sum, val, i) => {
     const diff = val - point2[i];
     return sum + diff * diff;
   }, 0));
   ```

2. **Strict Zip**:
   ```python
   # Python: strict=True raises ValueError if lengths differ
   zip(point1, point2, strict=True)

   # JavaScript: must manually check
   if (point1.length !== point2.length) {
     throw new Error('Length mismatch');
   }
   ```

3. **Power Operator**:
   ```javascript
   // Python: ** operator
   x ** 2
   x ** 0.5

   // JavaScript: Math functions or ** operator (ES2016+)
   x * x        // For squaring, often faster
   Math.sqrt(x)  // For square root
   x ** 2       // Also works, but less common
   ```

4. **Type Annotations**:
   ```typescript
   // Python: Sequence[float] (duck typing)
   def euclidean_distance(point1: Sequence[float], point2: Sequence[float]) -> float

   // TypeScript: readonly number[] (structural typing)
   function euclideanDistance(point1: readonly number[], point2: readonly number[]): number
   ```

---

## Performance Considerations

### Time Complexity

| Function | Complexity | Notes |
|----------|-----------|-------|
| `euclidean_distance` | O(n) | n = number of dimensions (typically 2) |
| `vector_magnitude` | O(n) | n = number of components |
| `bbox_center` | O(1) | Fixed 2D calculation |

### Space Complexity

| Function | Complexity | Notes |
|----------|-----------|-------|
| `euclidean_distance` | O(1) | No additional structures |
| `vector_magnitude` | O(1) | No additional structures |
| `bbox_center` | O(1) | Returns new list with 2 elements |

### Browser Performance

```javascript
// Optimized for 2D (most common case)
function euclideanDistance2DFast(x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  return Math.sqrt(dx * dx + dy * dy);
}

// Avoid Math.pow for squaring (slower than x * x)
// BAD:  Math.pow(diff, 2)
// GOOD: diff * diff

// Use Math.hypot for built-in optimization (ES2015+)
const distance = Math.hypot(x2 - x1, y2 - y1);
```

---

## Edge Cases

### 1. Empty Sequences

```python
# euclidean_distance with empty points
euclidean_distance([], [])
# Returns: 0.0 (sum of empty sequence is 0)

# vector_magnitude with empty vector
vector_magnitude([])
# Returns: 0.0 (magnitude of zero vector)
```

### 2. Single Dimension

```python
# 1D distance (valid)
euclidean_distance([10], [40])
# Returns: 30.0

# 1D magnitude
vector_magnitude([5])
# Returns: 5.0
```

### 3. Length Mismatch

```python
# Python with strict=True
euclidean_distance([10, 20], [40, 60, 80])
# Raises: ValueError (strict zip)

# Browser should validate
if (point1.length !== point2.length) {
  throw new Error('Point dimensions must match');
}
```

### 4. Negative Coordinates

```python
# Negative coordinates (valid)
euclidean_distance([-10, -20], [10, 20])
# Returns: 44.721... (correct distance)

bbox_center([-50, -100], [20, 30])
# Returns: [-40, -85] (center in negative space)
```

### 5. Zero Dimensions

```python
# Zero-size bounding box
bbox_center([100, 200], [0, 0])
# Returns: [100, 200] (origin is center)
```

### 6. Large Numbers

```python
# Large coordinates (potential overflow in other languages)
euclidean_distance([1e10, 1e10], [2e10, 2e10])
# JavaScript: May lose precision with very large numbers
# Use careful scaling for extreme values
```

---

## Testing Patterns

### Unit Tests

```python
import pytest
from src.utils.geometry import euclidean_distance, vector_magnitude, bbox_center

def test_euclidean_distance_basic():
    assert euclidean_distance([0, 0], [3, 4]) == 5.0
    assert euclidean_distance([10, 20], [10, 20]) == 0.0

def test_vector_magnitude_basic():
    assert vector_magnitude([3, 4]) == 5.0
    assert vector_magnitude([0, 0]) == 0.0

def test_bbox_center_basic():
    assert bbox_center([100, 200], [50, 30]) == [125, 215]
    assert bbox_center([0, 0], [100, 100]) == [50, 50]

def test_edge_cases():
    # Empty
    assert euclidean_distance([], []) == 0.0
    assert vector_magnitude([]) == 0.0

    # Negative
    assert bbox_center([-50, -100], [20, 30]) == [-40, -85]

    # Zero dimensions
    assert bbox_center([100, 200], [0, 0]) == [100, 200]
```

### Browser Tests (Jest/Vitest)

```javascript
import { euclideanDistance, vectorMagnitude, bboxCenter } from './geometry';

describe('Geometry Utils', () => {
  test('euclideanDistance calculates correct distance', () => {
    expect(euclideanDistance([0, 0], [3, 4])).toBeCloseTo(5.0);
    expect(euclideanDistance([10, 20], [10, 20])).toBe(0);
  });

  test('vectorMagnitude calculates correct magnitude', () => {
    expect(vectorMagnitude([3, 4])).toBeCloseTo(5.0);
    expect(vectorMagnitude([0, 0])).toBe(0);
  });

  test('bboxCenter calculates correct center', () => {
    expect(bboxCenter([100, 200], [50, 30])).toEqual([125, 215]);
    expect(bboxCenter([0, 0], [100, 100])).toEqual([50, 50]);
  });

  test('handles edge cases', () => {
    expect(euclideanDistance([], [])).toBe(0);
    expect(vectorMagnitude([])).toBe(0);
    expect(bboxCenter([-50, -100], [20, 30])).toEqual([-40, -85]);
  });

  test('throws on dimension mismatch', () => {
    expect(() => euclideanDistance([1, 2], [1, 2, 3])).toThrow();
  });
});
```

---

## Related Documentation

- **Image Utils**: `../image/flows.md`
- **Drawing Utils**: `../drawing/flows.md`
- **Math Utils**: `../math/flows.md`
- **Alignment Flows**: `../../alignment/flows.md`
- **Detection Flows**: `../../detection/concept.md`

---

## Summary

Geometry utilities provide fundamental geometric operations:

1. **Euclidean Distance**: Calculate straight-line distance between points
2. **Vector Magnitude**: Calculate length of vectors
3. **Bounding Box Center**: Calculate center of rectangular regions

**Key Characteristics**:
- Simple, pure functions with no side effects
- O(n) or O(1) time complexity
- Thread-safe (no shared state)
- Easy browser migration with standard Math functions

**Common Applications**:
- Distance measurements in bubble detection
- Shift magnitude calculation in alignment
- Center point calculation for ML bounding boxes
- Validation of geometric constraints
