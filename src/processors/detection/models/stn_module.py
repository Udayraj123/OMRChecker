"""Spatial Transformer Network (STN) module for OMR sheet alignment refinement.

Implements a lightweight STN that learns to correct residual geometric distortions
(rotation, translation, scale, shear) after initial alignment, improving field
block detection accuracy on challenging images.

Reference: "Spatial Transformer Networks" (Jaderberg et al., 2015)
https://arxiv.org/abs/1506.02025
"""

import torch
import torch.nn.functional as F
from torch import nn


class SpatialTransformerNetwork(nn.Module):
    """Lightweight STN for OMR sheet alignment refinement.

    Learns affine transformations to correct residual misalignments in OMR sheets,
    improving detection accuracy on mobile photos, skewed scans, and bent sheets.

    Architecture:
        - Localization Network: Small CNN to predict transformation parameters
        - Grid Generator: Creates sampling grid from predicted affine matrix
        - Sampler: Bilinear interpolation to warp input image

    Attributes:
        localization: CNN that predicts transformation parameters
        fc_loc: Fully connected layers to output 2x3 affine matrix
    """

    def __init__(
        self, input_channels: int = 1, input_size: tuple[int, int] = (640, 640)
    ) -> None:
        """Initialize the STN module.

        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            input_size: Expected input image size (height, width)
        """
        super().__init__()

        self.input_channels = input_channels
        self.input_size = input_size

        # Localization network: Small CNN to extract spatial features
        # Design rationale: Lightweight to minimize inference overhead (~10K params)
        self.localization = nn.Sequential(
            # First conv block: Downsample spatial dimensions
            nn.Conv2d(input_channels, 8, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 640 -> 160 -> 80
            # Second conv block: Extract higher-level features
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 80 -> 40 -> 20
            # Third conv block: Final feature extraction
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # Fixed output size regardless of input
        )

        # Regressor: Predict 6 parameters of 2x3 affine transformation matrix
        # [θ11, θ12, θ13]  ->  [[scale_x*cos(θ), -sin(θ), tx],
        # [θ21, θ22, θ23]      [sin(θ), scale_y*cos(θ), ty]]
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Prevent overfitting to training distortions
            nn.Linear(64, 6),
        )

        # Initialize transformation to identity
        # This ensures STN starts with no transformation and learns gradually
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial transformation to input image.

        Args:
            x: Input image tensor of shape (batch, channels, height, width)

        Returns:
            Transformed image tensor of same shape as input
        """
        # Step 1: Predict transformation parameters
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)  # Flatten: (batch, 32*4*4)
        theta = self.fc_loc(xs)  # (batch, 6)

        # Step 2: Reshape to 2x3 affine transformation matrix
        theta = theta.view(-1, 2, 3)  # (batch, 2, 3)

        # Step 3: Generate sampling grid
        # Creates a normalized coordinate grid that will be used to sample from input
        grid = F.affine_grid(theta, x.size(), align_corners=False)

        # Step 4: Sample from input using bilinear interpolation
        # This applies the learned transformation to the input image
        return F.grid_sample(x, grid, align_corners=False)

    def get_transformation_params(self, x: torch.Tensor) -> torch.Tensor:
        """Get the predicted transformation parameters without applying them.

        Useful for debugging and visualization of learned transformations.

        Args:
            x: Input image tensor

        Returns:
            Affine transformation matrix of shape (batch, 2, 3)
        """
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        return theta.view(-1, 2, 3)


class STNWithRegularization(SpatialTransformerNetwork):
    """STN with additional regularization to prevent extreme transformations.

    Adds a penalty for transformations that deviate too much from identity,
    preventing the network from learning unrealistic warps.
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_size: tuple[int, int] = (640, 640),
        regularization_weight: float = 0.1,
    ) -> None:
        """Initialize STN with regularization.

        Args:
            input_channels: Number of input channels
            input_size: Expected input image size
            regularization_weight: Weight for transformation regularization loss
        """
        super().__init__(input_channels, input_size)
        self.regularization_weight = regularization_weight

    def compute_regularization_loss(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss to prevent extreme transformations.

        Penalizes deviation from identity transformation:
        - Identity matrix: [[1, 0, 0], [0, 1, 0]]
        - Large rotations/scales get penalized

        Args:
            theta: Predicted affine transformation matrix (batch, 2, 3)

        Returns:
            Scalar regularization loss
        """
        # Identity transformation
        identity = (
            torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=theta.dtype, device=theta.device)
            .unsqueeze(0)
            .expand_as(theta)
        )

        # L2 distance from identity
        reg_loss = torch.mean((theta - identity) ** 2)

        return self.regularization_weight * reg_loss

    def forward_with_regularization(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with regularization loss.

        Args:
            x: Input image tensor

        Returns:
            Tuple of (transformed_image, regularization_loss)
        """
        # Get transformation parameters
        theta = self.get_transformation_params(x)

        # Compute regularization loss
        reg_loss = self.compute_regularization_loss(theta)

        # Apply transformation
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_transformed = F.grid_sample(x, grid, align_corners=False)

        return x_transformed, reg_loss


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test STN module

    # Create STN instance
    stn = SpatialTransformerNetwork(input_channels=1, input_size=(640, 640))

    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 640, 640)
    test_output = stn(test_input)

    # Test transformation parameters
    theta = stn.get_transformation_params(test_input)

    # Test regularized version
    stn_reg = STNWithRegularization(input_channels=1)
    transformed, reg_loss = stn_reg.forward_with_regularization(test_input)
