# STN Module - Execution Flow

## Overview

**STN (Spatial Transformer Network)** is a neural network module that learns to apply spatial transformations to input images. In OMRChecker, STN is used for **residual alignment refinement** after initial geometric alignment, correcting subtle distortions (rotation, translation, scale, shear) that improve field block detection accuracy.

**Reference**: "Spatial Transformer Networks" (Jaderberg et al., 2015) - https://arxiv.org/abs/1506.02025

**Files**:
- `src/processors/detection/models/stn_module.py` - STN implementations
- `src/processors/detection/models/stn_utils.py` - Utility functions for loading, inference, visualization

---

## Architecture Overview

### Three-Stage Pipeline

```
Input Image (OMR sheet)
    ↓
1. Localization Network (Small CNN)
    → Predicts transformation parameters (θ)
    ↓
2. Grid Generator
    → Creates sampling grid from θ
    ↓
3. Sampler (Bilinear Interpolation)
    → Warps input image
    ↓
Transformed Image (aligned)
```

### Component Hierarchy

```
SpatialTransformerNetwork (Base)
├─> localization: CNN (8→16→32 channels)
├─> fc_loc: FC layers (512→64→6 params)
└─> forward(): Apply transformation

STNWithRegularization (Adds penalty for extreme transforms)
└─> compute_regularization_loss()

TranslationOnlySTN (Simplified, only tx/ty)
├─> localization: Same CNN
├─> fc_loc: FC layers (512→64→2 params)
└─> forward(): Apply translation-only transform

TranslationOnlySTNWithRegularization
└─> compute_regularization_loss()
```

---

## Model Variants

### 1. SpatialTransformerNetwork (Full Affine)

**Purpose**: Learn general affine transformations (rotation, scale, shear, translation)

**Parameters**: 6 (2×3 affine matrix)
```
θ = [θ11, θ12, θ13]  →  [[scale_x*cos(θ), -sin(θ), tx],
    [θ21, θ22, θ23]      [sin(θ), scale_y*cos(θ), ty]]
```

**Use Cases**:
- Mobile phone photos (rotation, perspective)
- Skewed scans (shear correction)
- Bent sheets (local warping)

---

### 2. TranslationOnlySTN (Simplified)

**Purpose**: Learn only positional shifts (no rotation/scaling)

**Parameters**: 2 (tx, ty)
```
θ = [tx, ty]  →  [[1, 0, tx],
                  [0, 1, ty]]
```

**Use Cases**:
- Flatbed scans (only small shifts)
- Post-SIFT refinement (SIFT handles rotation/scale)
- Faster training/inference

---

## Initialization Flow

### Step 1: Create STN Instance

```python
from src.processors.detection.models.stn_module import SpatialTransformerNetwork

stn = SpatialTransformerNetwork(
    input_channels=1,        # Grayscale OMR sheets
    input_size=(640, 640)    # Expected image dimensions
)
```

**Architecture Details**:

```python
# Localization Network (Feature Extractor)
self.localization = nn.Sequential(
    # Block 1: 640×640 → 320×320 → 160×160
    nn.Conv2d(1, 8, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(8),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, stride=2),

    # Block 2: 160×160 → 80×80 → 40×40
    nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
    nn.BatchNorm2d(16),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, stride=2),

    # Block 3: 40×40 → 40×40 → 4×4
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((4, 4)),  # Fixed 4×4 output
)

# Regressor (Transformation Predictor)
self.fc_loc = nn.Sequential(
    nn.Linear(32 * 4 * 4, 64),  # 512 → 64
    nn.ReLU(inplace=True),
    nn.Dropout(0.1),            # Prevent overfitting
    nn.Linear(64, 6),           # 64 → 6 parameters
)
```

**Parameter Count**: ~10,000 (lightweight for minimal overhead)

---

### Step 2: Initialize to Identity

**Critical**: STN starts with no transformation, learns gradually

```python
# Initialize final FC layer
self.fc_loc[3].weight.data.zero_()           # All weights = 0
self.fc_loc[3].bias.data.copy_(
    torch.tensor([1, 0, 0, 0, 1, 0])         # Identity: [[1,0,0], [0,1,0]]
)
```

**Why Identity?**
- Network starts with θ = identity matrix
- Input = output initially
- Learns to deviate only when beneficial
- Prevents random transformations during early training

---

## Forward Pass Flow

### Step-by-Step Execution

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Input image (batch, 1, 640, 640)

    Returns:
        Transformed image (batch, 1, 640, 640)
    """
    # STEP 1: Localization Network
    # Extract spatial features
    xs = self.localization(x)           # (batch, 32, 4, 4)

    # STEP 2: Flatten Features
    xs = xs.view(xs.size(0), -1)        # (batch, 512)

    # STEP 3: Predict Transformation
    theta = self.fc_loc(xs)              # (batch, 6)
    theta = theta.view(-1, 2, 3)         # (batch, 2, 3)

    # STEP 4: Generate Sampling Grid
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    # grid: (batch, H, W, 2) - normalized coords [-1, 1]

    # STEP 5: Sample from Input
    output = F.grid_sample(x, grid, align_corners=False)
    # Uses bilinear interpolation

    return output
```

---

### Detailed Step Breakdown

#### Step 1: Localization Network

**Purpose**: Extract spatial features that indicate misalignment

```python
# Input: (batch=4, channels=1, height=640, width=640)
x = torch.randn(4, 1, 640, 640)

# Pass through CNN:
xs = self.localization(x)
# Output: (batch=4, channels=32, height=4, width=4)
```

**What it learns**:
- Edge alignment (sheet boundaries)
- Marker positions (reference points)
- Field block locations
- Overall image orientation

---

#### Step 2: Flatten Features

```python
xs = xs.view(xs.size(0), -1)
# (4, 32, 4, 4) → (4, 512)
# Flatten spatial dimensions, keep batch
```

---

#### Step 3: Predict Transformation Parameters

```python
theta = self.fc_loc(xs)           # (4, 6)
theta = theta.view(-1, 2, 3)      # (4, 2, 3)

# Example output for one image:
theta[0] = [[1.02, -0.01,  0.03],   # Scale, rotation, tx
            [0.01,  0.98, -0.02]]   # Rotation, scale, ty

# Interpretation:
# - Scale X: 1.02 (2% larger)
# - Scale Y: 0.98 (2% smaller)
# - Rotation: ~0.01 radians ≈ 0.57°
# - Translation: (0.03, -0.02) in normalized coords
```

**Normalized Coordinates**:
- Range: [-1, 1]
- Center of image: (0, 0)
- Top-left: (-1, -1)
- Bottom-right: (1, 1)

**Example**:
```python
tx = 0.1  # 10% of image width to the right
ty = -0.05 # 5% of image height upward

# For 640×640 image:
# tx in pixels = 0.1 × 640 / 2 = 32 pixels
# ty in pixels = -0.05 × 640 / 2 = -16 pixels
```

---

#### Step 4: Generate Sampling Grid

```python
grid = F.affine_grid(theta, x.size(), align_corners=False)
# theta: (batch, 2, 3)
# x.size(): (batch, channels, height, width)
# grid: (batch, H, W, 2) - normalized (x, y) coordinates
```

**What it does**:
Creates a mesh grid of sampling coordinates

```python
# Example for 4×4 grid (actual: 640×640):
# Original grid (identity):
[(-1, -1), (-0.33, -1), (0.33, -1), (1, -1)]
[(-1, -0.33), (-0.33, -0.33), ...]
[(-1, 0.33), ...]
[(-1, 1), (-0.33, 1), (0.33, 1), (1, 1)]

# After θ = [[1, 0, 0.1], [0, 1, 0]] (shift right by 0.1):
[(-0.9, -1), (-0.23, -1), (0.43, -1), (1.1, -1)]
[(-0.9, -0.33), (-0.23, -0.33), ...]
...
```

**Boundary Handling**:
- Coordinates outside [-1, 1] are clamped or padded
- Default: zero padding (black border if image shifted out)

---

#### Step 5: Bilinear Sampling

```python
output = F.grid_sample(x, grid, align_corners=False)
```

**Purpose**: Sample from input image using transformed grid

**Bilinear Interpolation**:
```python
# For a grid point (x=0.5, y=0.3):
# x=0.5 → pixel column 320 (in 640px image)
# y=0.3 → pixel row 288

# Actual: x=320.3, y=288.7 (sub-pixel)
# Interpolate from 4 neighbors:
#   - (320, 288): weight = (1-0.3) × (1-0.7) = 0.21
#   - (321, 288): weight = 0.3 × (1-0.7) = 0.09
#   - (320, 289): weight = (1-0.3) × 0.7 = 0.49
#   - (321, 289): weight = 0.3 × 0.7 = 0.21
# Output = weighted average of 4 pixel values
```

**Result**: Smooth, differentiable warping (enables backpropagation)

---

## Inference Flow (Deployed STN)

### Step 1: Load Trained Model

```python
from src.processors.detection.models.stn_utils import load_stn_model

stn = load_stn_model(
    model_path="models/stn_omr_v1.pt",
    input_channels=1,
    input_size=(640, 640),
    device="cpu"  # or "cuda"
)
# Model automatically set to eval mode
```

**Auto-Detection**:
```python
# Checks for metadata JSON:
# models/stn_omr_v1.json:
{
  "transformation_type": "affine",  # or "translation_only"
  "trained_on": "omr_dataset_v2",
  "accuracy": 0.94
}

# Loads correct architecture based on type
```

---

### Step 2: Apply to Image

```python
from src.processors.detection.models.stn_utils import apply_stn_to_image
import cv2

# Read OMR sheet
image = cv2.imread("scan.jpg", cv2.IMREAD_GRAYSCALE)  # (480, 640)

# Apply STN transformation
transformed = apply_stn_to_image(stn, image, device="cpu")
# Output: (480, 640) - same size

# transformed is now better aligned for detection
```

**Under the Hood**:
```python
def apply_stn_to_image(model, image, device="cpu"):
    # 1. Convert numpy → torch
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    # (H, W) → (1, 1, H, W)

    # 2. Normalize to [0, 1]
    if image_tensor.max() > 1:
        image_tensor = image_tensor / 255.0

    # 3. Move to device
    image_tensor = image_tensor.to(device)

    # 4. Inference (no gradients)
    with torch.no_grad():
        transformed_tensor = model(image_tensor)

    # 5. Convert back to numpy
    transformed = transformed_tensor.cpu().squeeze(0).squeeze(0).numpy()

    # 6. Denormalize to [0, 255]
    transformed = (transformed * 255).clip(0, 255).astype(np.uint8)

    return transformed
```

---

### Step 3: Get Transformation Parameters (Optional)

```python
from src.processors.detection.models.stn_utils import get_transformation_matrix

theta = get_transformation_matrix(stn, image, device="cpu")
# theta: (2, 3) numpy array

# Example:
# [[1.03, -0.02,  0.05],
#  [0.02,  0.97, -0.03]]
```

**Decompose into Interpretable Parameters**:
```python
from src.processors.detection.models.stn_utils import decompose_affine_matrix

params = decompose_affine_matrix(theta, transformation_type="affine")
# {
#   "rotation": 1.15,        # degrees
#   "scale_x": 1.03,         # 3% larger
#   "scale_y": 0.97,         # 3% smaller
#   "shear": 0.02,           # degrees
#   "translation_x": 0.05,   # normalized
#   "translation_y": -0.03   # normalized
# }
```

---

## Regularization Flow (Training-Time)

### STNWithRegularization

**Purpose**: Prevent STN from learning unrealistic transformations

```python
from src.processors.detection.models.stn_module import STNWithRegularization

stn_reg = STNWithRegularization(
    input_channels=1,
    input_size=(640, 640),
    regularization_weight=0.1  # Penalty strength
)
```

### Forward Pass with Regularization

```python
# During training:
transformed, reg_loss = stn_reg.forward_with_regularization(x)

# Total loss:
detection_loss = detection_criterion(model_output, ground_truth)
total_loss = detection_loss + reg_loss

# Backpropagation:
total_loss.backward()
```

### Regularization Loss Calculation

```python
def compute_regularization_loss(self, theta: torch.Tensor) -> torch.Tensor:
    """
    Penalize deviation from identity transformation.

    Args:
        theta: (batch, 2, 3) affine matrices

    Returns:
        Scalar loss
    """
    # Identity transformation
    identity = torch.tensor(
        [[1, 0, 0],
         [0, 1, 0]],
        dtype=theta.dtype,
        device=theta.device
    ).unsqueeze(0).expand_as(theta)

    # L2 distance from identity
    reg_loss = torch.mean((theta - identity) ** 2)

    return self.regularization_weight * reg_loss
```

**Effect**:
```python
# Without regularization:
θ = [[2.5, 1.3, 0.8],   # Extreme scale/rotation
     [1.2, 0.3, -0.9]]

# With regularization (weight=0.1):
θ = [[1.05, -0.02, 0.03],  # Close to identity
     [0.02, 0.98, -0.01]]
```

---

## Translation-Only STN Flow

### Simplified Architecture

```python
from src.processors.detection.models.stn_module import TranslationOnlySTN

stn_trans = TranslationOnlySTN(
    input_channels=1,
    input_size=(640, 640)
)
```

**Key Difference**: Predicts only 2 parameters (tx, ty)

### Forward Pass

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Step 1: Extract features (same as full STN)
    xs = self.localization(x)
    xs = xs.view(xs.size(0), -1)

    # Step 2: Predict translation (2 params instead of 6)
    translation = self.fc_loc(xs)  # (batch, 2) → [tx, ty]

    # Step 3: Construct translation-only affine matrix
    batch_size = x.size(0)
    theta = torch.zeros(batch_size, 2, 3)
    theta[:, 0, 0] = 1.0  # No scaling
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = translation[:, 0]  # tx
    theta[:, 1, 2] = translation[:, 1]  # ty

    # Step 4-5: Same grid generation and sampling
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)
```

**Example Transformation**:
```python
# Predicted translation:
translation = [0.05, -0.02]  # Shift right 5%, up 2%

# Constructed affine matrix:
theta = [[1.0,  0.0,  0.05],
         [0.0,  1.0, -0.02]]

# No rotation, no scaling, only shift
```

### Get Translation Values

```python
# Get raw translation without matrix construction:
translation = stn_trans.get_translation_values(image_tensor)
# (batch, 2) → [[tx1, ty1], [tx2, ty2], ...]

# Example:
# [[0.05, -0.02],   # Image 1: shift right 5%, up 2%
#  [0.03,  0.01]]   # Image 2: shift right 3%, down 1%
```

---

## Visualization Flow

### Step 1: Create Visualization

```python
from src.processors.detection.models.stn_utils import visualize_transformation

# Original and transformed images
original = cv2.imread("scan.jpg", cv2.IMREAD_GRAYSCALE)
transformed = apply_stn_to_image(stn, original)

# Get transformation matrix
theta = get_transformation_matrix(stn, original)

# Create visualization
vis = visualize_transformation(
    original,
    transformed,
    theta,
    save_path="debug/stn_transform.jpg"
)
```

### Visualization Output

```
┌──────────────┬──────────────┬──────────────┐
│  Original    │ Transformed  │  Difference  │
│              │              │              │
│  [Image 1]   │  [Image 2]   │  [Image 3]   │
│              │              │              │
│ Matrix:      │              │              │
│ [1.03 -.02   │              │              │
│  .02   .97   │              │              │
└──────────────┴──────────────┴──────────────┘
```

**Components**:
1. **Original**: Input OMR sheet
2. **Transformed**: After STN alignment
3. **Difference**: Absolute difference (highlights changes)
4. **Matrix**: Affine transformation parameters

---

## Model Management Flow

### Save Model

```python
from src.processors.detection.models.stn_utils import save_stn_model

# After training:
save_stn_model(
    model=stn,
    save_path="models/stn_omr_v1.pt",
    metadata={
        "transformation_type": "affine",
        "trained_on": "omr_dataset_v2",
        "num_samples": 5000,
        "validation_accuracy": 0.94,
        "notes": "Trained on mobile scans"
    }
)
```

**Creates Two Files**:
1. `stn_omr_v1.pt` - PyTorch weights
2. `stn_omr_v1.json` - Metadata

```json
{
  "transformation_type": "affine",
  "trained_on": "omr_dataset_v2",
  "num_samples": 5000,
  "validation_accuracy": 0.94,
  "notes": "Trained on mobile scans"
}
```

### Load Model (Auto-Detection)

```python
# Automatically detects type from metadata:
stn = load_stn_model("models/stn_omr_v1.pt")

# Loads correct class:
# - "affine" → SpatialTransformerNetwork
# - "translation_only" → TranslationOnlySTN
```

---

## Edge Cases & Handling

### 1. Identity Transformation (No Change Needed)

```python
# STN predicts θ ≈ identity:
theta = [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0]]

# Check if transformation is identity:
from src.processors.detection.models.stn_utils import is_identity_transform

if is_identity_transform(theta, tolerance=0.1):
    print("No transformation needed, image already aligned")
    # Skip STN, use original image
```

---

### 2. Extreme Transformations (Regularization Helps)

**Problem**: STN learns unrealistic warping

**Solution**: Use STNWithRegularization

```python
# Without regularization:
theta = [[3.5, 2.1, 0.9],   # Extreme scale/rotation
         [2.0, 0.2, -1.5]]  # → Severely distorted image

# With regularization (weight=0.1):
theta = [[1.05, -0.02, 0.03],  # Reasonable correction
         [0.02, 0.98, -0.01]]
```

---

### 3. Out-of-Bounds Coordinates

**Scenario**: Transformation shifts image content outside boundaries

```python
# Predicted translation: tx=0.8 (80% shift right)
# Some pixels sample from outside image bounds

# grid_sample behavior (default: zeros padding):
# - Out-of-bounds coords → return 0 (black)
# - Creates black borders where content shifted out
```

**Alternatives** (if needed):
```python
# Border padding (repeat edge pixels):
output = F.grid_sample(x, grid, mode='bilinear', padding_mode='border')

# Reflection padding:
output = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection')
```

---

### 4. Different Input Sizes

**STN is NOT fully size-invariant**

```python
# Trained on 640×640:
stn = SpatialTransformerNetwork(input_size=(640, 640))

# Applied to 480×320:
image_tensor = torch.randn(1, 1, 480, 320)
output = stn(image_tensor)  # Works, but parameters may be suboptimal

# Why: AdaptiveAvgPool2d(4, 4) adapts, but learned features assume 640×640
```

**Best Practice**: Resize inputs to training size

```python
# Resize to 640×640 before STN:
image = cv2.resize(image, (640, 640))
transformed = apply_stn_to_image(stn, image)
# Resize back if needed:
transformed = cv2.resize(transformed, original_size)
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Typical Time (CPU) | Typical Time (GPU) |
|-----------|------------|---------------------|---------------------|
| Localization CNN | O(n) | 20-50ms | 5-10ms |
| FC layers | O(1) | < 1ms | < 1ms |
| Grid generation | O(H × W) | 5-10ms | 1-2ms |
| Bilinear sampling | O(H × W) | 10-20ms | 2-5ms |
| **Total (640×640)** | O(H × W) | **35-80ms** | **8-18ms** |

Where n = number of pixels (640×640 = 409,600)

### Space Complexity

```python
# Model parameters:
# Localization: ~8K params
# FC layers: ~2K params
# Total: ~10K params × 4 bytes = 40 KB

# Runtime memory (640×640 image):
# Input: 640×640×4 = 1.6 MB
# Features: 32×4×4×4 = 2 KB
# Grid: 640×640×2×4 = 3.2 MB
# Output: 640×640×4 = 1.6 MB
# Total: ~6.5 MB per image
```

### Parameter Count

```python
from src.processors.detection.models.stn_module import count_parameters

# Full affine STN:
stn = SpatialTransformerNetwork()
print(count_parameters(stn))  # ~10,000 params

# Translation-only STN:
stn_trans = TranslationOnlySTN()
print(count_parameters(stn_trans))  # ~9,800 params (slightly fewer)
```

---

## Integration with Detection Pipeline

### Typical Usage

```python
# In ML-based field block detection:

# 1. Initial alignment (SIFT/preprocessing)
aligned_image = alignment_processor.align(scanned_image, template)

# 2. STN refinement
stn = load_stn_model("models/stn_field_block.pt")
refined_image = apply_stn_to_image(stn, aligned_image)

# 3. Field block detection (YOLO/other ML)
field_blocks = detector.detect(refined_image)

# STN improves detection accuracy by correcting residual misalignment
```

---

## Related Flows

- **Alignment Flow** (`../../alignment/flows.md`) - Initial alignment (SIFT/Phase Correlation)
- **ML Field Block Detector** (`../ml-field-block/flows.md`) - Uses STN for refinement
- **Preprocessing Flow** (`../../preprocessing/flows.md`) - Geometric transformations
- **Image Utils** (`../../utils/image/flows.md`) - Image warping utilities

---

## Testing Considerations

### Unit Tests

```python
def test_stn_forward_pass():
    """Test STN processes images correctly"""
    stn = SpatialTransformerNetwork(input_channels=1, input_size=(640, 640))
    batch = torch.randn(4, 1, 640, 640)
    output = stn(batch)

    assert output.shape == (4, 1, 640, 640)  # Same shape
    assert not torch.isnan(output).any()      # No NaN values

def test_stn_identity_initialization():
    """Test STN starts with identity transformation"""
    stn = SpatialTransformerNetwork()
    identity_input = torch.randn(1, 1, 640, 640)

    theta = stn.get_transformation_params(identity_input)
    expected = torch.tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0)

    # Should be close to identity initially
    assert torch.allclose(theta, expected, atol=0.5)

def test_translation_only_stn():
    """Test translation-only variant"""
    stn = TranslationOnlySTN()
    batch = torch.randn(2, 1, 640, 640)

    translation = stn.get_translation_values(batch)
    assert translation.shape == (2, 2)  # (batch, 2)

    theta = stn.get_transformation_params(batch)
    assert theta[0, 0, 0] == 1.0  # No scaling
    assert theta[0, 1, 1] == 1.0

def test_regularization_loss():
    """Test regularization penalizes extreme transforms"""
    stn = STNWithRegularization(regularization_weight=0.1)

    # Extreme transform should have high loss
    extreme_theta = torch.tensor([[[3.0, 1.5, 0.8],
                                   [1.2, 0.3, -0.9]]])
    extreme_loss = stn.compute_regularization_loss(extreme_theta)

    # Identity transform should have zero loss
    identity_theta = torch.tensor([[[1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0]]])
    identity_loss = stn.compute_regularization_loss(identity_theta)

    assert extreme_loss > identity_loss
    assert identity_loss < 0.01  # Near zero
```

### Edge Case Tests

1. **Blank image** (no features)
2. **Identity transformation** (no alignment needed)
3. **Extreme transformations** (regularization test)
4. **Different input sizes** (640×640, 480×320, etc.)
5. **Batch processing** (multiple images)
