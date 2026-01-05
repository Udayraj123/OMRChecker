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


class TranslationOnlySTN(nn.Module):
    """Lightweight STN that learns only translation (tx, ty) transformations.

    Simpler alternative to full affine STN when only positional corrections
    are needed (no rotation/scaling). Provides faster training and inference
    with more stable convergence.

    Architecture:
        - Localization Network: Same CNN as SpatialTransformerNetwork
        - Grid Generator: Creates sampling grid with translation-only matrix
        - Sampler: Bilinear interpolation to warp input image

    Attributes:
        localization: CNN that predicts transformation parameters
        fc_loc: Fully connected layers to output 2 translation parameters (tx, ty)
    """

    def __init__(
        self, input_channels: int = 1, input_size: tuple[int, int] = (640, 640)
    ) -> None:
        """Initialize the translation-only STN module.

        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            input_size: Expected input image size (height, width)
        """
        super().__init__()

        self.input_channels = input_channels
        self.input_size = input_size

        # Localization network: Same architecture as full affine STN
        # Reuse proven design for feature extraction
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

        # Regressor: Predict only 2 parameters (tx, ty) instead of 6
        # Translation is in normalized coordinates [-1, 1]
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Prevent overfitting to training distortions
            nn.Linear(64, 2),  # Only tx, ty
        )

        # Initialize translation to zero (no shift)
        # This ensures STN starts with identity and learns gradually
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply translation transformation to input image.

        Args:
            x: Input image tensor of shape (batch, channels, height, width)

        Returns:
            Transformed image tensor of same shape as input
        """
        # Step 1: Predict translation parameters
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)  # Flatten: (batch, 32*4*4)
        translation = self.fc_loc(xs)  # (batch, 2) -> [tx, ty]

        # Step 2: Construct 2x3 translation-only affine matrix
        # [[1, 0, tx],
        #  [0, 1, ty]]
        batch_size = x.size(0)
        theta = torch.zeros(batch_size, 2, 3, dtype=x.dtype, device=x.device)
        theta[:, 0, 0] = 1.0  # No horizontal scaling
        theta[:, 1, 1] = 1.0  # No vertical scaling
        theta[:, 0, 2] = translation[:, 0]  # Horizontal translation
        theta[:, 1, 2] = translation[:, 1]  # Vertical translation

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
            Translation-only affine transformation matrix of shape (batch, 2, 3)
        """
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        translation = self.fc_loc(xs)

        # Construct full 2x3 matrix
        batch_size = x.size(0)
        theta = torch.zeros(batch_size, 2, 3, dtype=x.dtype, device=x.device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = translation[:, 0]
        theta[:, 1, 2] = translation[:, 1]

        return theta

    def get_translation_values(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw translation values (tx, ty) without full matrix construction.

        Args:
            x: Input image tensor

        Returns:
            Translation tensor of shape (batch, 2) with [tx, ty] values
        """
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        return self.fc_loc(xs)


class TranslationOnlySTNWithRegularization(TranslationOnlySTN):
    """Translation-only STN with regularization to prevent extreme translations.

    Adds a penalty for large translation magnitudes, preventing the network
    from learning unrealistic shifts.
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_size: tuple[int, int] = (640, 640),
        regularization_weight: float = 0.1,
        max_translation: float = 0.2,
    ) -> None:
        """Initialize translation-only STN with regularization.

        Args:
            input_channels: Number of input channels
            input_size: Expected input image size
            regularization_weight: Weight for translation regularization loss
            max_translation: Maximum reasonable translation in normalized coords (e.g., 0.2 = 20% of image)
        """
        super().__init__(input_channels, input_size)
        self.regularization_weight = regularization_weight
        self.max_translation = max_translation

    def compute_regularization_loss(self, translation: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss to prevent extreme translations.

        Penalizes large translation magnitudes using L2 norm.
        Encourages small corrections rather than large shifts.

        Args:
            translation: Predicted translation values (batch, 2) with [tx, ty]

        Returns:
            Scalar regularization loss
        """
        # L2 magnitude penalty (distance from zero translation)
        translation_magnitude = torch.sqrt(torch.sum(translation**2, dim=1))
        reg_loss = torch.mean(translation_magnitude)

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
        # Get translation parameters
        translation = self.get_translation_values(x)

        # Compute regularization loss
        reg_loss = self.compute_regularization_loss(translation)

        # Apply transformation
        x_transformed = self.forward(x)

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

    # Create translation-only STN
    stn_trans = TranslationOnlySTN(input_channels=1, input_size=(640, 640))

    # Test forward pass
    test_output_trans = stn_trans(test_input)

    # Test transformation parameters
    theta_trans = stn_trans.get_transformation_params(test_input)
    translation_vals = stn_trans.get_translation_values(test_input)

    # Test regularized version
    stn_trans_reg = TranslationOnlySTNWithRegularization(input_channels=1)
    transformed_trans, reg_loss_trans = stn_trans_reg.forward_with_regularization(
        test_input
    )
