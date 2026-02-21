# Math Utils Flows

**Module**: Domain - Utils - Math
**Python Reference**: `src/utils/math.py`
**Last Updated**: 2026-02-21

---

## Overview

The `MathUtils` class provides static mathematical utility functions for geometric operations, point manipulation, and color conversion. All methods are stateless and designed for integration with OpenCV and OMRChecker's coordinate systems.

**Key Features**:
- Point arithmetic (addition, subtraction)
- Distance and angle calculations
- Rectangle and polygon operations
- Color conversions
- List chunking utilities

---

## Flow 1: Point Distance Calculation

**Method**: `MathUtils.distance(point1, point2)`
**Purpose**: Calculate Euclidean distance between two points

### Input

```python
point1: tuple[float, float]  # (x1, y1)
point2: tuple[float, float]  # (x2, y2)
```

### Processing Steps

```
START: distance(point1, point2)
│
├─► Step 1: Extract Coordinates
│   x1, y1 = point1[0], point1[1]
│   x2, y2 = point2[0], point2[1]
│
│   Example: point1 = (10, 20), point2 = (40, 50)
│
├─► Step 2: Calculate Differences
│   dx = x2 - x1 = 40 - 10 = 30
│   dy = y2 - y1 = 50 - 20 = 30
│
├─► Step 3: Compute Euclidean Distance
│   distance = math.hypot(dx, dy)
│   distance = √(dx² + dy²)
│   distance = √(30² + 30²)
│   distance = √(900 + 900)
│   distance = √1800
│   distance = 42.426...
│
└─► Step 4: Return Result
    return 42.426

END
```

### Output

```python
distance: float  # 42.426
```

**Algorithm**: Uses `math.hypot()` which provides accurate floating-point calculation and handles overflow better than manual sqrt(dx²+dy²).

---

## Flow 2: Point Addition and Shifting

**Methods**: `add_points()`, `shift_points_from_origin()`
**Purpose**: Translate points by adding offsets

### Flow 2A: Add Single Point

**Method**: `MathUtils.add_points(new_origin, point)`

```
START: add_points(new_origin, point)
│
├─► Input
│   new_origin = (100, 200)  # Offset to add
│   point = (30, 40)         # Point to shift
│
├─► Calculate New Coordinates
│   new_x = new_origin[0] + point[0] = 100 + 30 = 130
│   new_y = new_origin[1] + point[1] = 200 + 40 = 240
│
└─► Return Shifted Point
    return [130, 240]

END
```

### Flow 2B: Shift Multiple Points

**Method**: `MathUtils.shift_points_from_origin(new_origin, list_of_points)`

```
START: shift_points_from_origin(new_origin, list_of_points)
│
├─► Input
│   new_origin = (100, 200)
│   list_of_points = [
│       (10, 20),
│       (30, 40),
│       (50, 60)
│   ]
│
├─► Apply Shift to Each Point
│   result = []
│
│   For each point in list_of_points:
│       shifted_point = add_points(new_origin, point)
│       result.append(shifted_point)
│
│   Iteration 1: (10, 20) → [110, 220]
│   Iteration 2: (30, 40) → [130, 240]
│   Iteration 3: (50, 60) → [150, 260]
│
└─► Return Shifted Points
    return [
        [110, 220],
        [130, 240],
        [150, 260]
    ]

END
```

**Use Case**: Shift field coordinates after image cropping or preprocessing.

---

## Flow 3: Point Subtraction and Origin Shifting

**Methods**: `subtract_points()`, `shift_points_to_origin()`
**Purpose**: Translate points by subtracting offsets (inverse of Flow 2)

### Flow 3A: Subtract Single Point

**Method**: `MathUtils.subtract_points(point, new_origin)`

```
START: subtract_points(point, new_origin)
│
├─► Input
│   point = (150, 260)       # Point to shift
│   new_origin = (100, 200)  # Offset to subtract
│
├─► Calculate Relative Coordinates
│   rel_x = point[0] - new_origin[0] = 150 - 100 = 50
│   rel_y = point[1] - new_origin[1] = 260 - 200 = 60
│
└─► Return Relative Point
    return [50, 60]

END
```

### Flow 3B: Shift Multiple Points to Origin

**Method**: `MathUtils.shift_points_to_origin(new_origin, list_of_points)`

```
START: shift_points_to_origin(new_origin, list_of_points)
│
├─► Input
│   new_origin = (100, 200)
│   list_of_points = [
│       (110, 220),
│       (130, 240),
│       (150, 260)
│   ]
│
├─► Apply Subtraction to Each Point
│   result = []
│
│   For each point in list_of_points:
│       relative_point = subtract_points(point, new_origin)
│       result.append(relative_point)
│
│   Iteration 1: (110, 220) → [10, 20]
│   Iteration 2: (130, 240) → [30, 40]
│   Iteration 3: (150, 260) → [50, 60]
│
└─► Return Relative Points
    return [
        [10, 20],
        [30, 40],
        [50, 60]
    ]

END
```

**Use Case**: Convert absolute coordinates to relative coordinates after cropping.

---

## Flow 4: Point on Line by Ratio

**Method**: `MathUtils.get_point_on_line_by_ratio(edge_line, length_ratio)`
**Purpose**: Find point along a line segment at specified ratio

### Input

```python
edge_line: tuple[point, point]  # [(x1, y1), (x2, y2)]
length_ratio: float             # 0.0 to 1.0
```

### Processing Steps

```
START: get_point_on_line_by_ratio(edge_line, length_ratio)
│
├─► Step 1: Extract Endpoints
│   start, end = edge_line
│   start = (100, 100)
│   end = (200, 300)
│
├─► Step 2: Input Ratio
│   length_ratio = 0.25  # 25% along the line
│
├─► Step 3: Linear Interpolation
│   │
│   │ Formula: P = start + ratio × (end - start)
│   │
│   new_x = start[0] + (end[0] - start[0]) × length_ratio
│          = 100 + (200 - 100) × 0.25
│          = 100 + 100 × 0.25
│          = 100 + 25
│          = 125
│
│   new_y = start[1] + (end[1] - start[1]) × length_ratio
│          = 100 + (300 - 100) × 0.25
│          = 100 + 200 × 0.25
│          = 100 + 50
│          = 150
│
└─► Step 4: Return Interpolated Point
    return [125, 150]

END
```

### Output

```python
point: list[float, float]  # [125, 150]
```

### Ratio Examples

```python
length_ratio = 0.0   # Returns start point (100, 100)
length_ratio = 0.25  # Returns (125, 150)
length_ratio = 0.5   # Returns midpoint (150, 200)
length_ratio = 0.75  # Returns (175, 250)
length_ratio = 1.0   # Returns end point (200, 300)
```

**Use Case**: Find marker positions on template edges, crop boundaries.

---

## Flow 5: Order Four Points

**Method**: `MathUtils.order_four_points(points, dtype="int")`
**Purpose**: Order four corner points in standard sequence (tl, tr, br, bl)

### Input

```python
points: list[point]  # Unordered 4 points
dtype: str          # "int" or "float"
```

### Processing Steps

```
START: order_four_points(points, dtype="int")
│
├─► Step 1: Convert to NumPy Array
│   points = np.array(points, dtype=dtype)
│
│   Example input (unordered):
│   points = [
│       [300, 100],  # top-right
│       [100, 100],  # top-left
│       [300, 400],  # bottom-right
│       [100, 400]   # bottom-left
│   ]
│
├─► Step 2: Calculate Sum of Coordinates
│   sum_of_points = points.sum(axis=1)
│
│   [300+100, 100+100, 300+400, 100+400]
│   = [400, 200, 700, 500]
│
│   Observation:
│   - Smallest sum (200) → top-left (100, 100)
│   - Largest sum (700) → bottom-right (300, 400)
│
├─► Step 3: Calculate Difference of Coordinates
│   diff = np.diff(points, axis=1)  # x - y for each point
│
│   [[300-100], [100-100], [300-400], [100-400]]
│   = [200, 0, -100, -300]
│
│   Observation:
│   - Largest diff (200) → top-right (300, 100)
│   - Smallest diff (-300) → bottom-left (100, 400)
│
├─► Step 4: Find Ordered Indices
│   ordered_indices = [
│       np.argmin(sum_of_points),   # top-left: index 1
│       np.argmin(diff),             # top-right: index 0
│       np.argmax(sum_of_points),   # bottom-right: index 2
│       np.argmax(diff)              # bottom-left: index 3
│   ]
│   = [1, 0, 2, 3]
│
├─► Step 5: Reorder Points
│   rect = points[ordered_indices]
│   rect = [
│       points[1],  # [100, 100] - top-left
│       points[0],  # [300, 100] - top-right
│       points[2],  # [300, 400] - bottom-right
│       points[3]   # [100, 400] - bottom-left
│   ]
│
└─► Step 6: Return Ordered Rectangle
    return rect, ordered_indices

END
```

### Output

```python
rect: np.ndarray           # Ordered points (tl, tr, br, bl)
ordered_indices: list[int] # [1, 0, 2, 3]
```

### Visual Representation

```
Before ordering:
points[0] = (300, 100)  ┐
points[1] = (100, 100)  │  Unordered
points[2] = (300, 400)  │
points[3] = (100, 400)  ┘

After ordering:
rect[0] = (100, 100)  ← top-left
rect[1] = (300, 100)  ← top-right
rect[2] = (300, 400)  ← bottom-right
rect[3] = (100, 400)  ← bottom-left
```

**Use Case**: Normalize corner points from contour detection for perspective transform.

---

## Flow 6: Bounding Box of Points

**Method**: `MathUtils.get_bounding_box_of_points(points)`
**Purpose**: Find axis-aligned bounding box containing all points

### Input

```python
points: np.ndarray  # N×2 array of points
```

### Processing Steps

```
START: get_bounding_box_of_points(points)
│
├─► Step 1: Input Points
│   points = [
│       [120, 150],
│       [180, 200],
│       [140, 180],
│       [200, 220]
│   ]
│
├─► Step 2: Find Min/Max Coordinates
│   min_x, min_y = np.min(points, axis=0)
│   min_x = 120, min_y = 150
│
│   max_x, max_y = np.max(points, axis=0)
│   max_x = 200, max_y = 220
│
├─► Step 3: Create Bounding Box
│   bounding_box = [
│       (min_x, min_y),  # top-left: (120, 150)
│       (max_x, min_y),  # top-right: (200, 150)
│       (max_x, max_y),  # bottom-right: (200, 220)
│       (min_x, max_y)   # bottom-left: (120, 220)
│   ]
│
├─► Step 4: Calculate Dimensions
│   width = max_x - min_x = 200 - 120 = 80
│   height = max_y - min_y = 220 - 150 = 70
│   box_dimensions = (80, 70)
│
└─► Step 5: Return Box and Dimensions
    return bounding_box, box_dimensions

END
```

### Output

```python
bounding_box: np.ndarray      # 4×2 array (tl, tr, br, bl)
box_dimensions: tuple[int]    # (width, height)
```

### Visual Representation

```
Input points (scattered):
    (120, 150)
         │
         ├─── (140, 180)
         │         │
         │    (180, 200)
         │         │
         └───────────────── (200, 220)

Bounding box:
(120,150) ────────── (200,150)
    │                    │
    │    Input Points    │
    │                    │
(120,220) ────────── (200,220)
```

**Use Case**: Find minimal rectangle containing field block or markers.

---

## Flow 7: Rectangle Point Generation

**Methods**: `get_rectangle_points()`, `get_rectangle_points_from_box()`
**Purpose**: Generate 4 corner points of rectangle from position and dimensions

### Flow 7A: From X, Y, W, H

**Method**: `MathUtils.get_rectangle_points(x, y, w, h)`

```
START: get_rectangle_points(x=100, y=200, w=50, h=30)
│
├─► Step 1: Input Parameters
│   x = 100     # Top-left X
│   y = 200     # Top-left Y
│   w = 50      # Width
│   h = 30      # Height
│
├─► Step 2: Calculate Four Corners
│   top_left = [x, y] = [100, 200]
│   top_right = [x + w, y] = [150, 200]
│   bottom_right = [x + w, y + h] = [150, 230]
│   bottom_left = [x, y + h] = [100, 230]
│
└─► Step 3: Return Ordered Points
    return [
        [100, 200],  # tl
        [150, 200],  # tr
        [150, 230],  # br
        [100, 230]   # bl
    ]

END
```

### Flow 7B: From Origin and Dimensions

**Method**: `MathUtils.get_rectangle_points_from_box(origin, dimensions)`

```
START: get_rectangle_points_from_box(origin, dimensions)
│
├─► Step 1: Extract Parameters
│   origin = (100, 200)
│   dimensions = (50, 30)
│
│   x, y = origin
│   w, h = dimensions
│
├─► Step 2: Delegate to get_rectangle_points
│   return get_rectangle_points(x, y, w, h)
│
└─► Return Result
    return [[100, 200], [150, 200], [150, 230], [100, 230]]

END
```

**Use Case**: Generate scan box coordinates from template configuration.

---

## Flow 8: Select Edge from Rectangle

**Method**: `MathUtils.select_edge_from_rectangle(rectangle, edge_type)`
**Purpose**: Extract specific edge from rectangle as line segment

### Input

```python
rectangle: list[point]  # [tl, tr, br, bl]
edge_type: EdgeType     # TOP, RIGHT, BOTTOM, LEFT
```

### Processing Steps

```
START: select_edge_from_rectangle(rectangle, edge_type)
│
├─► Step 1: Unpack Rectangle
│   tl, tr, br, bl = rectangle
│
│   tl = [100, 200]  # top-left
│   tr = [150, 200]  # top-right
│   br = [150, 230]  # bottom-right
│   bl = [100, 230]  # bottom-left
│
├─► Step 2: Select Edge Based on Type
│
│   If edge_type == EdgeType.TOP:
│       return [tl, tr] = [[100, 200], [150, 200]]
│
│   If edge_type == EdgeType.RIGHT:
│       return [tr, br] = [[150, 200], [150, 230]]
│
│   If edge_type == EdgeType.BOTTOM:
│       return [br, bl] = [[150, 230], [100, 230]]
│
│   If edge_type == EdgeType.LEFT:
│       return [bl, tl] = [[100, 230], [100, 200]]
│
│   Default (fallback to TOP):
│       return [tl, tr]
│
└─► Return Edge Line
    return edge_line

END
```

### Output

```python
edge_line: list[point, point]  # Line segment as [start, end]
```

### Visual Representation

```
Rectangle:
tl ──────TOP────── tr
│                  │
LEFT             RIGHT
│                  │
bl ─────BOTTOM──── br

EdgeType.TOP → [tl, tr]
EdgeType.RIGHT → [tr, br]
EdgeType.BOTTOM → [br, bl]
EdgeType.LEFT → [bl, tl]
```

**Use Case**: Find edge for marker alignment, boundary detection.

---

## Flow 9: Rectangle Contains Point

**Method**: `MathUtils.rectangle_contains(point, rect)`
**Purpose**: Check if point lies inside rectangle

### Input

```python
point: tuple[float, float]  # (x, y)
rect: tuple[float, float, float, float]  # (x1, y1, x2, y2)
```

### Processing Steps

```
START: rectangle_contains(point, rect)
│
├─► Step 1: Extract Coordinates
│   point = (125, 215)
│   rect = (100, 200, 150, 230)
│
│   rect_start = rect[0:2] = (100, 200)  # top-left
│   rect_end = rect[2:4] = (150, 230)    # bottom-right
│
├─► Step 2: Check Boundaries
│   │
│   │ Point is INSIDE if:
│   │ point.x >= rect_start.x AND
│   │ point.y >= rect_start.y AND
│   │ point.x <= rect_end.x AND
│   │ point.y <= rect_end.y
│   │
│   Check conditions:
│   ✓ 125 >= 100 (point.x >= rect_start.x)
│   ✓ 215 >= 200 (point.y >= rect_start.y)
│   ✓ 125 <= 150 (point.x <= rect_end.x)
│   ✓ 215 <= 230 (point.y <= rect_end.y)
│
├─► Step 3: Implement via Negation
│   │ (Code uses negation of OUTSIDE conditions)
│   │
│   is_outside = (
│       point[0] < rect_start[0] OR  # Left of rect
│       point[1] < rect_start[1] OR  # Above rect
│       point[0] > rect_end[0] OR    # Right of rect
│       point[1] > rect_end[1]       # Below rect
│   )
│
│   is_outside = False
│   is_inside = not is_outside = True
│
└─► Step 4: Return Result
    return True

END
```

### Output

```python
contains: bool  # True if point inside rectangle
```

### Edge Cases

```python
# Point on boundary
rectangle_contains((100, 200), (100, 200, 150, 230))  # False (< check)

# Point at corner
rectangle_contains((100, 200), (100, 200, 150, 230))  # False

# Point outside
rectangle_contains((50, 215), (100, 200, 150, 230))   # False (left)
rectangle_contains((160, 215), (100, 200, 150, 230))  # False (right)
```

**Use Case**: Hit testing for field blocks, marker validation.

---

## Flow 10: Validate Rectangle (Max Cosine Check)

**Methods**: `validate_rect()`, `check_max_cosine()`, `angle()`
**Purpose**: Validate if quadrilateral is approximately rectangular

### Input

```python
approx: np.ndarray  # 4×2 array of corner points
```

### Processing Steps

```
START: validate_rect(approx)
│
├─► Step 1: Check Point Count
│   if len(approx) != 4:
│       return False
│
├─► Step 2: Reshape and Check Angles
│   points = approx.reshape(4, 2)
│   return check_max_cosine(points)
│
│ ┌──────────────────────────────────────────────┐
│ │ check_max_cosine(approx)                     │
│ └──────────────────────────────────────────────┘
│ │
│ ├─► Initialize Tracking Variables
│ │   max_cosine = 0
│ │   min_cosine = 1.5
│ │   MAX_COSINE = 0.35  # Threshold for rectangularity
│ │
│ ├─► Check Each Corner Angle
│ │   For i in range(2, 5):  # Check angles at corners 2, 3, 0
│ │       │
│ │       │ At corner approx[i-1], measure angle between:
│ │       │ - Vector from approx[i-1] to approx[i % 4]
│ │       │ - Vector from approx[i-1] to approx[i-2]
│ │       │
│ │       cosine = abs(angle(
│ │           approx[i % 4],     # Next point
│ │           approx[i - 2],     # Previous point
│ │           approx[i - 1]      # Current corner
│ │       ))
│ │
│ │       max_cosine = max(cosine, max_cosine)
│ │       min_cosine = min(cosine, min_cosine)
│ │
│ ├─► Check Rectangularity
│ │   if max_cosine >= 0.35:
│ │       logger.warning("Quadrilateral is not a rectangle.")
│ │       return False
│ │
│ │   return True  # All angles close to 90°
│
└─► Return Validation Result
    return is_rectangular

END
```

### Angle Calculation Details

**Method**: `MathUtils.angle(p_1, p_2, p_0)`

```
START: angle(p_1, p_2, p_0)
│
├─► Step 1: Calculate Vectors from p_0
│   │
│   │ Vector 1: p_0 → p_1
│   dx1 = float(p_1[0] - p_0[0])
│   dy1 = float(p_1[1] - p_0[1])
│   │
│   │ Vector 2: p_0 → p_2
│   dx2 = float(p_2[0] - p_0[0])
│   dy2 = float(p_2[1] - p_0[1])
│
├─► Step 2: Calculate Dot Product
│   dot_product = dx1 * dx2 + dy1 * dy2
│
├─► Step 3: Calculate Magnitudes
│   mag1_squared = dx1 * dx1 + dy1 * dy1
│   mag2_squared = dx2 * dx2 + dy2 * dy2
│
│   magnitude_product = sqrt(mag1_squared * mag2_squared)
│
├─► Step 4: Calculate Cosine of Angle
│   │
│   │ cos(θ) = dot_product / magnitude_product
│   │ Add small epsilon (1e-10) to prevent division by zero
│   │
│   cosine = dot_product / (magnitude_product + 1e-10)
│
└─► Return Cosine
    return cosine

END
```

### Interpretation

```python
# Cosine values and their meaning:
cos(0°) = 1.0      # Vectors aligned (not a corner)
cos(45°) = 0.707   # 45° angle (not rectangular)
cos(90°) = 0.0     # Perfect right angle ✓
cos(135°) = -0.707 # 135° angle (not rectangular)
cos(180°) = -1.0   # Vectors opposite (not a corner)

# MAX_COSINE = 0.35 means:
# Accept angles in range: 90° ± ~20°
# Reject angles outside: ~70° to ~110°
```

**Use Case**: Validate detected contours before perspective transform.

---

## Flow 11: Check Collinear Points

**Method**: `MathUtils.check_collinear_points(point1, point2, point3)`
**Purpose**: Check if three points lie on same line

### Input

```python
point1, point2, point3: tuple[float, float]
```

### Processing Steps

```
START: check_collinear_points(point1, point2, point3)
│
├─► Step 1: Extract Coordinates
│   [x1, y1] = point1 = [0, 0]
│   [x2, y2] = point2 = [5, 5]
│   [x3, y3] = point3 = [10, 10]
│
├─► Step 2: Calculate Cross Product
│   │
│   │ Three points are collinear if:
│   │ (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)
│   │
│   │ This is equivalent to checking if the cross product
│   │ of vectors (p1→p2) and (p1→p3) is zero.
│   │
│   left_side = (y1 - y2) * (x1 - x3)
│              = (0 - 5) * (0 - 10)
│              = (-5) * (-10)
│              = 50
│
│   right_side = (y1 - y3) * (x1 - x2)
│               = (0 - 10) * (0 - 5)
│               = (-10) * (-5)
│               = 50
│
├─► Step 3: Compare
│   are_collinear = (left_side == right_side)
│                 = (50 == 50)
│                 = True
│
└─► Return Result
    return True

END
```

### Output

```python
collinear: bool  # True if points are on same line
```

### Examples

```python
# Collinear (on same line)
check_collinear_points([0, 0], [5, 5], [10, 10])    # True (diagonal)
check_collinear_points([0, 0], [5, 0], [10, 0])     # True (horizontal)
check_collinear_points([0, 0], [0, 5], [0, 10])     # True (vertical)

# Not collinear
check_collinear_points([0, 0], [5, 5], [10, 0])     # False (triangle)
```

**Use Case**: Validate warp points, detect degenerate transformations.

---

## Flow 12: Color Conversion (to BGR)

**Method**: `MathUtils.to_bgr(any_color)`
**Purpose**: Convert any color format to BGR tuple for OpenCV

### Input

```python
any_color: str or tuple  # matplotlib color format
```

### Processing Steps

```
START: to_bgr(any_color)
│
├─► Step 1: Convert to RGB using matplotlib
│   │
│   │ matplotlib.colors.to_rgb() accepts:
│   │ - Color names: "red", "blue", "green"
│   │ - Hex codes: "#FF0000", "#00FF00"
│   │ - RGB tuples: (1.0, 0.0, 0.0)
│   │ - Single letter: "r", "g", "b"
│   │
│   any_color = "red"
│   rgb = colors.to_rgb("red")
│   rgb = (1.0, 0.0, 0.0)  # RGB normalized to [0, 1]
│
├─► Step 2: Scale to 0-255 Range
│   r = int(1.0 * 255) = 255
│   g = int(0.0 * 255) = 0
│   b = int(0.0 * 255) = 0
│
│   (r, g, b) = (255, 0, 0)
│
├─► Step 3: Swap to BGR Order
│   │
│   │ OpenCV uses BGR instead of RGB
│   │
│   bgr = (b, g, r)
│        = (0, 0, 255)
│
└─► Return BGR Tuple
    return (0, 0, 255)

END
```

### Output

```python
bgr: tuple[int, int, int]  # (B, G, R) in range [0, 255]
```

### Examples

```python
to_bgr("red")       # (0, 0, 255)   - BGR for red
to_bgr("green")     # (0, 255, 0)   - BGR for green
to_bgr("blue")      # (255, 0, 0)   - BGR for blue
to_bgr("#FF8000")   # (0, 128, 255) - BGR for orange
to_bgr("white")     # (255, 255, 255)
to_bgr("black")     # (0, 0, 0)
```

**Use Case**: Convert named colors for OpenCV drawing functions.

---

## Flow 13: Chunk List

**Method**: `MathUtils.chunks(input_list, chunk_size)`
**Purpose**: Split list into fixed-size chunks (generator)

### Input

```python
input_list: list      # Any list
chunk_size: int       # Size of each chunk
```

### Processing Steps

```
START: chunks(input_list, chunk_size)
│
├─► Step 1: Ensure Positive Chunk Size
│   chunk_size = max(1, chunk_size)
│
│   Example: chunk_size = 3
│
├─► Step 2: Input List
│   input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
│
├─► Step 3: Generate Chunks
│   │
│   │ Generator expression:
│   │ for i in range(0, len(input_list), chunk_size):
│   │     yield input_list[i : i + chunk_size]
│   │
│   Iteration 1 (i=0):
│       yield input_list[0:3] = [1, 2, 3]
│
│   Iteration 2 (i=3):
│       yield input_list[3:6] = [4, 5, 6]
│
│   Iteration 3 (i=6):
│       yield input_list[6:9] = [7, 8, 9]
│
│   Iteration 4 (i=9):
│       yield input_list[9:12] = [10]  # Last chunk may be smaller
│
└─► Return Generator
    return generator

END
```

### Output

```python
chunks: Generator[list]  # Yields chunks on iteration
```

### Usage Example

```python
# Convert generator to list
input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = list(MathUtils.chunks(input_list, 3))
# result = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]

# Use in loop
for chunk in MathUtils.chunks(input_list, 4):
    process(chunk)
# Iteration 1: [1, 2, 3, 4]
# Iteration 2: [5, 6, 7, 8]
# Iteration 3: [9, 10]
```

**Use Case**: Batch processing of fields, parallel processing coordination.

---

## Flow 14: Get Tuple Points

**Method**: `MathUtils.get_tuple_points(points)`
**Purpose**: Convert list of points to list of integer tuples

### Input

```python
points: list[list[float]]  # Points as lists
```

### Processing Steps

```
START: get_tuple_points(points)
│
├─► Step 1: Input Points
│   points = [
│       [100.5, 200.7],
│       [150.2, 205.9],
│       [125.8, 230.1]
│   ]
│
├─► Step 2: Convert Each Point
│   result = []
│
│   For each point in points:
│       x = int(point[0])
│       y = int(point[1])
│       tuple_point = (x, y)
│       result.append(tuple_point)
│
│   Iteration 1: [100.5, 200.7] → (100, 200)
│   Iteration 2: [150.2, 205.9] → (150, 205)
│   Iteration 3: [125.8, 230.1] → (125, 230)
│
└─► Return Tuple Points
    return [
        (100, 200),
        (150, 205),
        (125, 230)
    ]

END
```

### Output

```python
tuple_points: list[tuple[int, int]]  # Points as integer tuples
```

**Use Case**: Convert points for OpenCV drawing functions (require integer tuples).

---

## Summary

The Math Utils flows provide:

1. **Point Arithmetic**: Distance, addition, subtraction, shifting
2. **Geometric Operations**: Interpolation, ordering, bounding boxes
3. **Rectangle Operations**: Generation, edge selection, containment checking
4. **Validation**: Rectangle validation, collinearity checks
5. **Color Conversion**: matplotlib to OpenCV BGR format
6. **Utility Functions**: List chunking, type conversion

**All operations are stateless and suitable for concurrent use.**

---

## Related Flows

- **Geometry Utils**: `modules/domain/utils/geometry/flows.md` (pending)
- **Drawing Utils**: `modules/domain/utils/drawing/flows.md` (pending)
- **Image Warping**: `modules/domain/preprocessing/warping/flows.md`
- **Alignment**: `modules/domain/alignment/flows.md`
