# Phase Correlation Alignment - Execution Flow

## Overview

**Phase Correlation** is a frequency-domain method for detecting rigid translation (shift) between two images using Fast Fourier Transform (FFT).

**File**: `src/processors/alignment/phase_correlation.py`

**Status**: Commented out in current codebase (Method 1), replaced by K-Nearest Interpolation

**Advantages**:
- No feature detection needed
- Fast (FFT-based)
- Works on feature-poor regions

**Limitations**:
- **Only detects translation** (no rotation, scale, or warping)
- Assumes rigid shift across entire image
- Less accurate for local distortions

---

## Algorithm Flow

### Step 1: Compute Phase Correlation

**Code**: `phase_correlation.py:9-11`

```python
def phase_correlation(a, b):
    correlation_r = np.fft.fft2(a) * np.fft.fft2(b).conj()
    return np.abs(np.fft.ifft2(correlation_r))
```

#### Mathematical Breakdown

```
1. Forward FFT on both images:
   F_a = FFT2D(image_a)
   F_b = FFT2D(image_b)

2. Cross-power spectrum:
   R = F_a × F_b* (conjugate)

3. Inverse FFT:
   correlation = |IFFT2D(R)|

4. Find peak in correlation:
   shift = argmax(correlation)
```

**Why it works**: Translation in spatial domain = phase shift in frequency domain

**Example**:
```python
# Image A: Reference
# Image B: Shifted by [5, -3] pixels

correlation = phase_correlation(A, B)
# correlation is 2D array with peak at [5, -3]
```

---

### Step 2: Find Shift from Correlation Peak

**Code**: `phase_correlation.py:14-28`

```python
def get_phase_correlation_shifts(alignment_image, gray_image):
    corr = phase_correlation(alignment_image, gray_image)

    shape = corr.shape
    maxima = np.unravel_index(np.argmax(corr), shape)

    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.stack(maxima).astype(np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
    x, y = shifts[:2]

    corr_image = cv2.normalize(corr, None, 0, 255, norm_type=cv2.NORM_MINMAX)

    return [int(x), int(y)], corr_image
```

#### Shift Calculation

**1. Find Peak Location**:
```python
maxima = np.unravel_index(np.argmax(corr), shape)
# Example: maxima = (5, 397) in 400×300 image
```

**2. Wrap Negative Shifts**:
```python
# FFT output wraps around:
# Shift of +5 → maxima[0] = 5
# Shift of -3 → maxima[1] = 397 (wrapped: 400 - 3)

midpoints = [200, 150]  # Half of [400, 300]

if maxima > midpoints:
    shifts -= shape

# Example:
shifts[1] = 397
397 > 150  → True
shifts[1] = 397 - 400 = -3
```

**Final Output**: `shifts = [5, -3]` (x, y displacement)

---

### Step 3: Apply Affine Transform

**Code**: `phase_correlation.py:31-53`

```python
def apply_phase_correlation_shifts(field_block, block_gray_alignment_image, block_gray_image):
    field_block.shifts, corr_image = get_phase_correlation_shifts(
        block_gray_alignment_image, block_gray_image
    )

    logger.info(field_block.name, field_block.shifts)

    # Create affine transformation matrix
    correlation_m = np.float32([
        [1, 0, -1 * field_block.shifts[0]],
        [0, 1, -1 * field_block.shifts[1]]
    ])

    # Apply transformation
    shifted_block_image = cv2.warpAffine(
        block_gray_image,
        correlation_m,
        (block_gray_image.shape[1], block_gray_image.shape[0]),
    )

    InteractionUtils.show("Correlation", corr_image, 0)
    show_displacement_overlay(
        block_gray_alignment_image, block_gray_image, shifted_block_image
    )

    return shifted_block_image
```

#### Affine Matrix

```python
# Translation matrix:
M = [[1, 0, tx],
     [0, 1, ty]]

# Where:
tx = -shifts[0]  # Negative to reverse shift
ty = -shifts[1]

# Example: shifts = [5, -3]
M = [[1, 0, -5],
     [0, 1, 3]]
```

**cv2.warpAffine**: Applies transformation to shift image

---

## Why Commented Out?

**Current code** (`template_alignment.py:88-91`):
```python
# Method 1
# TODO: move to a processor:
# warped_block_image = apply_phase_correlation_shifts(
#     field_block, block_gray_alignment_image, block_gray_image
# )
```

**Reasons**:
1. **Rigid translation only**: Can't handle rotation, warping
2. **Global shift assumption**: Assumes entire field block shifts uniformly
3. **Replaced by SIFT**: K-Nearest Interpolation handles local variations

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Time |
|-----------|------------|------|
| FFT2D (forward) | O(n log n) | 5-10ms |
| FFT2D (inverse) | O(n log n) | 5-10ms |
| Peak finding | O(n) | < 1ms |
| **Total** | **O(n log n)** | **10-20ms** |

**Faster than SIFT** (20-80ms)

### Memory Usage

```python
# 400×300 image:
# FFT complex array: 400 × 300 × 16 bytes (complex128) = 1.9 MB
# Correlation array: 400 × 300 × 8 bytes (float64) = 0.96 MB
# Total: ~3 MB per field block
```

---

## Edge Cases

### 1. No Translation (Perfect Alignment)

```python
shifts = get_phase_correlation_shifts(image, image)
# shifts = [0, 0]
# Peak at center of correlation matrix
```

### 2. Translation Exceeds Image Size

```python
# Image: 400×300
# Actual shift: [450, 200]  ← Exceeds width

# Wrapped shift detected: [50, 200]  ← Wrong!
# Phase correlation fails for shifts > half image dimension
```

**Limitation**: Only reliable for shifts < image_dimension / 2

### 3. Rotation Present

```python
# Image rotated by 10°
# Phase correlation detects "best-fit" translation
# Result: Incorrect shift, doesn't fix rotation
```

### 4. Non-Uniform Shift

```python
# Top of image: shift [5, 3]
# Bottom of image: shift [8, 6]  ← Different!

# Phase correlation computes average/dominant shift
# Result: Partial correction, not perfect
```

---

## Browser Migration

### FFT Libraries

**Python**: `np.fft.fft2()`

**Browser Options**:

1. **fft.js**:
```javascript
import { FFT } from 'fft.js';

function phaseCorrelation(imageA, imageB) {
  const fft = new FFT(width * height);

  const fftA = fft.createComplexArray();
  fft.realTransform(fftA, imageA);

  const fftB = fft.createComplexArray();
  fft.realTransform(fftB, imageB);

  // Cross-power spectrum
  const R = multiplyConjugate(fftA, fftB);

  // Inverse FFT
  const correlation = new Float32Array(width * height);
  fft.inverseTransform(correlation, R);

  return correlation;
}
```

2. **ml-fft**:
```javascript
import { FFT } from 'ml-fft';

const fft = new FFT(width, height);
const result = fft.forward(imageData);
```

3. **OpenCV.js dft()**:
```javascript
// If available in build
const fftA = new cv.Mat();
cv.dft(imageA, fftA, cv.DFT_COMPLEX_OUTPUT);
```

---

## When to Use Phase Correlation

### Good Use Cases

1. **Simple translation**: Image only shifted, no rotation/warping
2. **Feature-poor regions**: Solid backgrounds, gradients
3. **Speed critical**: Need fast alignment
4. **Fallback method**: When SIFT fails (< MIN_MATCH_COUNT)

### Bad Use Cases

1. **Local distortions**: Different parts shift differently
2. **Rotation present**: Phase correlation can't handle
3. **Scale changes**: Can't detect DPI mismatch
4. **Severe warping**: Page curl, perspective distortion

---

## Integration Points

### Potential Usage (If Re-enabled)

```python
# In template_alignment.py:
for field_block in template.field_blocks:
    # Try SIFT first:
    displacement_pairs = SiftMatcher.get_matches(...)

    if len(displacement_pairs) < MIN_MATCH_COUNT:
        # Fallback to phase correlation:
        warped_block_image = apply_phase_correlation_shifts(
            field_block, block_gray_alignment_image, block_gray_image
        )
        # Use global shift for all scan boxes
```

---

## Testing Considerations

```python
def test_phase_correlation_zero_shift():
    shifts, _ = get_phase_correlation_shifts(image, image)
    assert shifts == [0, 0]

def test_phase_correlation_known_shift():
    shifted_image = translate_image(image, dx=10, dy=5)
    shifts, _ = get_phase_correlation_shifts(image, shifted_image)
    assert abs(shifts[0] - 10) < 1
    assert abs(shifts[1] - 5) < 1

def test_phase_correlation_rotation_fails():
    rotated_image = rotate_image(image, angle=10)
    shifts, _ = get_phase_correlation_shifts(image, rotated_image)
    # Shifts will be incorrect (can't handle rotation)
    # This test documents the limitation
```

---

## Related

- **SIFT Alignment** (`../sift/flows.md`) - Feature-based alternative
- **K-Nearest Interpolation** (`../k-nearest/flows.md`) - Current method
- **FFT Operations** (`modules/technical/numpy/array-operations.md`)
