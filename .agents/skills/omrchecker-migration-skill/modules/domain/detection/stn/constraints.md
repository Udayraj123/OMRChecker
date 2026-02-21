# STN Module - Constraints & Edge Cases

## Core Constraints

### 1. Input Dimensions

**Fixed at Training Time**:
```python
stn = SpatialTransformerNetwork(input_size=(640, 640))
# Expects 640×640 inputs during inference
```

**Why Fixed?**:
- Localization CNN learns spatial patterns at specific scales
- AdaptiveAvgPool2d(4, 4) adapts output, but learned features assume training size
- Different sizes may produce suboptimal transformations

**Workaround**:
```python
# Resize to training size before inference:
image_resized = cv2.resize(image, (640, 640))
transformed = apply_stn_to_image(stn, image_resized)

# Resize back to original:
transformed_original_size = cv2.resize(transformed, (original_h, original_w))
```

**Trade-off**: Resizing can introduce artifacts, especially for small text/bubbles

---

### 2. Model Size Constraint

**Lightweight Design**: ~10,000 parameters

**Rationale**:
- Minimize inference overhead (STN adds to detection pipeline)
- Fast training (can retrain on new OMR datasets)
- Small model file (~40 KB)

**Limitations**:
- Cannot learn very complex, non-affine transformations
- Limited capacity for highly distorted sheets (bent, folded)
- May struggle with extreme perspective (> 45° rotation)

**When STN Fails**:
```
Extreme perspective (mobile photo from steep angle)
  ↓
STN cannot correct fully (limited by affine constraints)
  ↓
Fallback: Use more aggressive preprocessing (CropOnMarkers, WarpOnPoints)
```

---

### 3. Affine Transformation Constraints

**Affine Limitations**:
- Cannot model curved/bent pages (requires piecewise affine or TPS)
- Assumes planar transformations
- Parallel lines remain parallel after transformation

**What Affine CAN Do**:
- Rotation (any angle)
- Translation (any distance)
- Scaling (any factor)
- Shear (skewing)

**What Affine CANNOT Do**:
- Perspective distortion (needs homography, not affine)
- Local warping (bent corners)
- Non-linear distortions (curved paper)

**Example Failure**:
```
OMR sheet bent in middle (U-shape):
  Left side: rotated -5°
  Right side: rotated +5°
  ↓
Single affine cannot model this (would need two affine transforms)
  ↓
STN applies average correction, both sides still slightly misaligned
```

---

### 4. Translation-Only STN Constraints

**Restricted to Translation**: Only tx, ty (no rotation/scale)

**Use Cases**:
- Post-SIFT alignment (SIFT already handles rotation/scale)
- Flatbed scans (minimal rotation)
- Faster inference (simpler grid generation)

**Limitations**:
```python
# If image has 5° rotation:
stn_trans = TranslationOnlySTN()
output = stn_trans(rotated_image)
# Output still has 5° rotation! Cannot correct.

# Need full affine STN:
stn = SpatialTransformerNetwork()
output = stn(rotated_image)
# Can correct rotation
```

---

### 5. Regularization Weight Constraint

**Balance Between Correction and Stability**:

```python
# Too low (e.g., 0.01):
regularization_weight = 0.01
# → STN learns extreme transforms (overfitting)
# → Unrealistic warping

# Too high (e.g., 1.0):
regularization_weight = 1.0
# → STN stays near identity (underfitting)
# → Doesn't correct misalignment

# Recommended: 0.05 - 0.2
regularization_weight = 0.1  # Good balance
```

**Effect on Training**:
```python
# Without regularization:
theta = [[2.3, 1.1, 0.7],   # Over-correction
         [1.0, 0.4, -0.8]]

# With regularization (0.1):
theta = [[1.05, -0.02, 0.03],  # Subtle correction
         [0.02, 0.98, -0.01]]
```

---

### 6. Gradient Flow Constraint

**Bilinear Sampling is Differentiable**:
- Enables end-to-end training with backpropagation
- Gradients flow through transformation

**But**:
- Gradient magnitude depends on image content
- Blank regions → weak gradients
- High-frequency details → stronger gradients

**Implication**:
```python
# Training on blank field blocks:
# → Weak gradients → slow learning

# Training on text/bubble-rich regions:
# → Strong gradients → faster learning

# Solution: Diverse training data
```

---

### 7. Coordinate Normalization Constraint

**Normalized Range**: [-1, 1]

**Why?**:
- Resolution-independent
- Numerical stability (avoids large numbers)
- Consistent across different image sizes

**Conversion**:
```python
# Pixel coordinates (640×640 image):
pixel_x = 320  # Center
pixel_y = 480  # 3/4 down

# Normalized coordinates:
norm_x = (pixel_x / 320) - 1 = 0.0   # Center
norm_y = (pixel_y / 320) - 1 = 0.5   # Halfway from center to bottom

# Reverse:
pixel_x = (norm_x + 1) * 320 = 320
pixel_y = (norm_y + 1) * 320 = 480
```

**Edge Case**: Non-square images
```python
# 640×480 image:
# X: [-1, 1] maps to [0, 640]
# Y: [-1, 1] maps to [0, 480]
# → Aspect ratio preserved
```

---

### 8. Boundary Handling Constraint

**Default: Zero Padding**

```python
# If transformation shifts image right:
theta = [[1, 0, 0.5], [0, 1, 0]]  # Shift right 50%
# Left side of output: black border (pixels from outside image)

# grid_sample behavior:
output = F.grid_sample(x, grid, padding_mode='zeros')
# Out-of-bounds → 0 (black for grayscale)
```

**Alternative Padding Modes**:
```python
# Border (repeat edge pixels):
output = F.grid_sample(x, grid, padding_mode='border')
# Out-of-bounds → nearest edge pixel

# Reflection:
output = F.grid_sample(x, grid, padding_mode='reflection')
# Out-of-bounds → mirror reflection
```

**Recommendation for OMR**: Use `zeros` (default)
- Black borders clearly indicate out-of-bounds
- Doesn't create false features

---

### 9. Memory Constraints

**Per-Image Memory** (640×640):
```python
# PyTorch memory (GPU):
# Input: 640×640×4 bytes = 1.6 MB
# Localization features: 32×4×4×4 = 2 KB
# Grid: 640×640×2×4 = 3.2 MB
# Output: 640×640×4 = 1.6 MB
# Total: ~6.5 MB per image

# Batch of 16 images: 16 × 6.5 = 104 MB
# GPU with 4 GB: Can process ~60 images in parallel
```

**Browser Constraints** (WASM heap):
```javascript
// Typical browser: 2 GB heap limit
// Large image (2000×3000):
// Single image: 2000×3000×4 = 24 MB
// STN processing: ~100 MB peak
// Max concurrent: ~20 images (very conservative)
```

---

### 10. Training Data Constraint

**Requires Paired Data**:
```python
# Training sample:
{
  "misaligned_image": "scan_001_misaligned.jpg",  # Input
  "ground_truth": "scan_001_aligned.jpg"          # Target
}

# Or alignment parameters:
{
  "misaligned_image": "scan_001.jpg",
  "target_theta": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # Identity
}
```

**Challenge**: Creating ground truth
- Manual alignment (labor-intensive)
- Synthetic misalignment (may not match real distortions)
- Self-supervised (use detection loss, no ground truth)

**Current Status**: OMRChecker does NOT include STN training pipeline (only inference)

---

## Edge Cases

### 1. Identity Transformation (No Correction Needed)

**Scenario**: Image already perfectly aligned

```python
theta = [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0]]

# Check:
from src.processors.detection.models.stn_utils import is_identity_transform

if is_identity_transform(theta, tolerance=0.05):
    print("STN detected no misalignment, using original image")
    # Skip STN, save computation
```

**Why This Happens**:
- STN initialized to identity
- Well-aligned input → network predicts near-identity
- Regularization penalty keeps it close to identity

---

### 2. Extreme Misalignment (Beyond STN Capacity)

**Scenario**: 60° rotation, 50% scale change, large perspective

```python
# STN predicts:
theta = [[0.8, -0.6, 0.2],   # Max capacity
         [0.6,  0.7, -0.1]]

# But true transformation needs:
theta_true = [[0.5, -0.87, 0.4],  # 60° rotation
              [0.87, 0.5, -0.3]]

# Result: Partial correction, still misaligned
```

**Detection**:
```python
# Check if predicted transformation is extreme:
params = decompose_affine_matrix(theta)

if abs(params["rotation"]) > 30 or params["scale_x"] > 1.5 or params["scale_x"] < 0.5:
    print("Warning: STN predicting extreme transformation, may be unreliable")
    # Fallback to preprocessing-based alignment
```

---

### 3. Blank/Uniform Image (No Features)

**Scenario**: Solid white or black image

```python
# Input: all white (255)
blank_image = torch.ones(1, 1, 640, 640) * 255

# STN forward:
theta = stn.get_transformation_params(blank_image)
# Localization CNN extracts no meaningful features
# FC predicts unpredictable transformation (depends on weight initialization)

# Result: Random transformation (not useful)
```

**Mitigation**:
```python
# Check image variance before STN:
if np.std(image) < 5:  # Nearly uniform
    print("Image too uniform for STN, skipping")
    return image  # Use original
```

---

### 4. Different Input Size (Not Trained On)

**Scenario**: STN trained on 640×640, applied to 480×320

```python
stn = SpatialTransformerNetwork(input_size=(640, 640))

# Applied to 480×320:
image_tensor = torch.randn(1, 1, 480, 320)
output = stn(image_tensor)  # (1, 1, 480, 320)

# Works, but:
# - Localization CNN sees different spatial layout
# - Learned features may not transfer well
# - Transformation parameters may be suboptimal
```

**Best Practice**:
```python
# Always resize to training size:
def apply_stn_safe(stn, image, training_size=(640, 640)):
    original_size = image.shape[:2]

    # Resize to training size
    image_resized = cv2.resize(image, training_size[::-1])

    # Apply STN
    transformed_resized = apply_stn_to_image(stn, image_resized)

    # Resize back
    transformed = cv2.resize(transformed_resized, original_size[::-1])

    return transformed
```

---

### 5. Batch Size = 1 (BatchNorm Instability)

**Issue**: BatchNorm layers expect batch statistics

```python
# Training: batch_size = 32
# → BatchNorm computes mean/std across 32 images

# Inference: batch_size = 1
# → BatchNorm uses running statistics (frozen during eval mode)
# → Usually fine, but can be unstable if training stats differ from test
```

**Solution**: STN already in eval mode during inference
```python
stn.eval()  # Freezes BatchNorm
# Uses running_mean, running_var (learned during training)
```

**Edge Case**: If model not trained with eval mode
```python
# Replace BatchNorm with GroupNorm (batch-independent):
nn.GroupNorm(num_groups=4, num_channels=8)
# More stable for batch_size=1
```

---

### 6. Out-of-Bounds Translation (Image Shifts Outside View)

**Scenario**: Predicted translation shifts entire image out

```python
theta = [[1, 0, 1.5],   # Shift right 150%
         [0, 1, 1.0]]   # Shift down 100%

# Result:
# - Entire image content outside [-1, 1] range
# - Output: all black (zero padding)
```

**Prevention**: Clip translation during inference
```python
# Constrain translation to reasonable range:
theta[:, :, 2] = torch.clamp(theta[:, :, 2], min=-0.5, max=0.5)
# Max 50% shift (prevents complete out-of-bounds)
```

---

### 7. Non-Affine Distortions (Curved Paper)

**Limitation**: STN with affine transformation cannot model curves

```python
# Bent paper (quadratic curve):
# Top-left corner: -5° rotation
# Center: 0° rotation
# Bottom-right: +5° rotation

# Affine STN prediction:
theta = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # Average = identity
# → Doesn't correct either side well

# Better: Piecewise Affine STN (more complex)
# Or: Thin Plate Spline (TPS) transformation
```

---

### 8. Gradient Vanishing (Flat Regions During Training)

**Problem**: Blank field blocks provide weak gradients

```python
# Blank field block:
# → Localization CNN extracts low-variance features
# → FC layer gradients small
# → Slow learning for such samples
```

**Solution**: Data augmentation
```python
# Add noise to blank regions during training:
if np.std(image) < 10:
    image += np.random.normal(0, 5, image.shape)  # Add noise
# Now provides gradients for learning
```

---

### 9. Model Type Mismatch (Load Wrong Architecture)

**Scenario**: Load translation-only weights into affine STN

```python
# Saved: TranslationOnlySTN (fc_loc outputs 2 params)
# Loaded: SpatialTransformerNetwork (fc_loc expects 6 params)

stn = SpatialTransformerNetwork()
state_dict = torch.load("stn_translation_only.pt")
stn.load_state_dict(state_dict)  # ERROR: size mismatch

# RuntimeError: fc_loc.3.bias: size [6] doesn't match [2]
```

**Solution**: Use metadata JSON
```python
# stn_model.json:
{
  "transformation_type": "translation_only"
}

# load_stn_model() auto-detects and loads correct class
stn = load_stn_model("stn_model.pt")  # Auto-loads TranslationOnlySTN
```

---

### 10. Concurrent Inference (Thread Safety)

**Issue**: Multiple threads using same STN model

```python
# Python GIL protects for now, but not guaranteed
# Better: Separate model instance per thread

# Thread-local models:
import threading

_thread_local = threading.local()

def get_stn_model():
    if not hasattr(_thread_local, 'stn'):
        _thread_local.stn = load_stn_model("model.pt")
    return _thread_local.stn

# In each thread:
stn = get_stn_model()
output = stn(input_tensor)
```

---

## Browser-Specific Constraints

### 1. TensorFlow.js Model Conversion

**Constraint**: PyTorch → TensorFlow.js conversion required

**Process**:
```bash
# Step 1: Export PyTorch to ONNX
import torch.onnx

torch.onnx.export(
    stn,
    dummy_input,
    "stn_model.onnx",
    opset_version=11
)

# Step 2: Convert ONNX to TensorFlow.js
# Using onnx-tf and tensorflowjs_converter
pip install onnx-tf tensorflowjs

onnx-tf convert -i stn_model.onnx -o stn_tf
tensorflowjs_converter --input_format=tf_saved_model stn_tf stn_tfjs
```

**Limitations**:
- Not all PyTorch ops supported in ONNX/TensorFlow.js
- `F.affine_grid` may need custom implementation
- Bilinear sampling (`F.grid_sample`) may have slight numerical differences

---

### 2. Browser Memory Limits

**WebAssembly Heap**: Typically 2 GB limit

```javascript
// Large image (2000×3000):
// Peak memory: ~100 MB per image

// Max concurrent: ~20 images (conservative)

// Mitigation:
// - Process images sequentially
// - Downsample large images
// - Use Web Workers (separate heap per worker)
```

---

### 3. GPU Acceleration (WebGL/WebGPU)

**TensorFlow.js WebGL Backend**: GPU acceleration for STN

```javascript
await tf.setBackend('webgl');

// STN inference on GPU:
const stn = await tf.loadGraphModel('stn_tfjs/model.json');
const output = stn.predict(inputTensor);
// ~5-10ms on modern GPU (vs 50-100ms on CPU)
```

**Limitations**:
- WebGL may have texture size limits (4096×4096 on some devices)
- WebGPU (newer, better) not universally supported yet

---

### 4. Bilinear Sampling Differences

**PyTorch vs TensorFlow.js**: Slight numerical differences

```python
# PyTorch:
output_pt = F.grid_sample(x, grid, mode='bilinear', align_corners=False)

# TensorFlow.js equivalent:
# tf.image.resizeBilinear() or custom implementation
# May differ by ~0.1% due to interpolation edge cases
```

**Impact**: Negligible for OMR (detection threshold absorbs small differences)

---

### 5. Model Loading Time

**TensorFlow.js Model Loading**: Can be slow

```javascript
// Model size: ~100 KB (STN small)
// Load time: 50-200ms (network + parsing)

// Mitigation:
// - Cache model in IndexedDB
// - Preload during app initialization

const modelUrl = 'stn_tfjs/model.json';
const cachedModel = await caches.match(modelUrl);
if (cachedModel) {
  // Load from cache
} else {
  // Fetch and cache
  const model = await tf.loadGraphModel(modelUrl);
  // Store in IndexedDB/Cache API
}
```

---

### 6. Inference Latency

**Browser Performance**:
- **CPU (WASM)**: 50-100ms per image
- **GPU (WebGL)**: 5-15ms per image

**Batch Processing Trade-off**:
```javascript
// Sequential (user sees progress):
for (const image of images) {
  const transformed = await stn.predict(image);
  updateUI(transformed);
}

// Batch (faster total time, delayed UI):
const batch = tf.stack(images);
const outputs = await stn.predict(batch);
// Process all outputs
```

---

## Performance Constraints

### 1. Inference Speed

| Device | Backend | Time (640×640) |
|--------|---------|----------------|
| Desktop CPU (Python) | PyTorch | 35-80ms |
| Desktop GPU (Python) | PyTorch CUDA | 8-18ms |
| Browser CPU (WASM) | TensorFlow.js | 50-100ms |
| Browser GPU (WebGL) | TensorFlow.js | 5-15ms |
| Mobile CPU | TensorFlow.js | 100-300ms |
| Mobile GPU | TensorFlow.js | 20-50ms |

---

### 2. Training Time Constraint

**Not Included in OMRChecker**: Training pipeline not provided

**Estimated Training**:
- Dataset: 5000 OMR sheets with alignment labels
- Epochs: 50-100
- Batch size: 32
- GPU: NVIDIA RTX 3080
- Time: **2-4 hours**

**Why Not Included**:
- Most users use pre-trained models
- Training requires labeled data (expensive to create)
- Inference-only deployment simpler

---

### 3. Model Size Constraint

**Disk Size**: ~40 KB (PyTorch), ~100 KB (TensorFlow.js)

**Why So Small?**:
- Only ~10,000 parameters
- Lightweight by design (fast inference)

**Comparison**:
- YOLO-v5s: ~7 MB
- ResNet-50: ~100 MB
- STN: ~0.04 MB (175× smaller than YOLO!)

---

## Validation Checks

### Before Inference

```python
# 1. Check input shape
assert image.shape == (640, 640) or len(image.shape) == 2, "Expected 640×640 grayscale"

# 2. Check input range
assert image.min() >= 0 and image.max() <= 255, "Expected [0, 255] range"

# 3. Check model loaded
assert stn is not None, "STN model not loaded"

# 4. Check model in eval mode
assert not stn.training, "STN must be in eval mode"
```

### After Inference

```python
# 1. Check output shape
assert output.shape == input.shape, "Output shape mismatch"

# 2. Check for NaN/Inf
assert not np.isnan(output).any(), "Output contains NaN"
assert not np.isinf(output).any(), "Output contains Inf"

# 3. Check transformation validity
theta = get_transformation_matrix(stn, image)
params = decompose_affine_matrix(theta)

# Warn if extreme:
if abs(params["rotation"]) > 45:
    logger.warning(f"Large rotation detected: {params['rotation']:.2f}°")

if params["scale_x"] > 2.0 or params["scale_x"] < 0.5:
    logger.warning(f"Large scale change: {params['scale_x']:.2f}×")
```

---

## Summary Table

| Constraint | Value | Impact |
|------------|-------|--------|
| Model parameters | ~10,000 | Lightweight, fast inference |
| Input size (typical) | 640×640 | Fixed during training, resize if different |
| Transformation type | Affine or Translation-only | Cannot model non-linear distortions |
| Memory per image | ~6.5 MB | Batch processing limited by GPU/browser heap |
| Inference time (GPU) | 8-18ms | Minimal overhead in pipeline |
| Inference time (Browser GPU) | 5-15ms | Real-time capable |
| Regularization weight | 0.05-0.2 | Balance correction vs stability |
| Coordinate range | [-1, 1] | Normalized, resolution-independent |
| Boundary handling | Zero padding | Black borders for out-of-bounds |

---

## Related Constraints

- **Alignment Constraints** (`../../alignment/constraints.md`) - SIFT/alignment limits
- **ML Model Migration** (`../../../migration/ml-model-migration.md`) - PyTorch → TensorFlow.js
- **Performance Constraints** (`../../../migration/performance.md`) - Browser memory/speed limits
- **OpenCV Constraints** (`../../../technical/opencv/opencv-operations.md`) - Image warping limits
