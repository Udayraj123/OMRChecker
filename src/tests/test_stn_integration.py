"""Unit tests for STN integration with YOLO field block detector."""

import numpy as np
import pytest
import torch

from src.processors.detection.models.stn_module import (
    SpatialTransformerNetwork,
    STNWithRegularization,
    count_parameters,
)
from src.processors.detection.models.stn_utils import (
    apply_stn_to_image,
    decompose_affine_matrix,
    is_identity_transform,
)


class TestSTNModule:
    """Test STN module functionality."""

    def test_stn_initialization(self):
        """Test STN model initializes correctly."""
        stn = SpatialTransformerNetwork(input_channels=1, input_size=(640, 640))

        assert stn.input_channels == 1
        assert stn.input_size == (640, 640)

        # Check parameter count (should be ~40K)
        params = count_parameters(stn)
        assert 30000 < params < 50000, f"Expected ~40K params, got {params}"

    def test_stn_forward_pass(self):
        """Test STN forward pass produces correct output shape."""
        stn = SpatialTransformerNetwork(input_channels=1, input_size=(640, 640))

        # Test input
        batch_size = 2
        test_input = torch.randn(batch_size, 1, 640, 640)

        # Forward pass
        output = stn(test_input)

        assert output.shape == test_input.shape
        assert output.dtype == test_input.dtype

    def test_stn_identity_initialization(self):
        """Test STN initializes to identity transformation."""
        stn = SpatialTransformerNetwork(input_channels=1, input_size=(640, 640))

        test_input = torch.randn(1, 1, 640, 640)
        theta = stn.get_transformation_params(test_input)

        # Should be close to identity [[1,0,0], [0,1,0]]
        identity = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)

        assert theta.shape == (1, 2, 3)
        assert torch.allclose(theta[0], identity, atol=0.1)

    def test_stn_with_regularization(self):
        """Test STN with regularization loss."""
        stn = STNWithRegularization(
            input_channels=1, input_size=(640, 640), regularization_weight=0.1
        )

        test_input = torch.randn(2, 1, 640, 640)
        transformed, reg_loss = stn.forward_with_regularization(test_input)

        assert transformed.shape == test_input.shape
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.numel() == 1  # Scalar
        assert reg_loss.item() >= 0  # Non-negative

    def test_stn_regularization_loss_computation(self):
        """Test regularization loss penalizes deviations from identity."""
        stn = STNWithRegularization(regularization_weight=1.0)

        # Identity transformation (no loss)
        identity_theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32)
        identity_loss = stn.compute_regularization_loss(identity_theta)
        assert identity_loss.item() < 0.01

        # Non-identity transformation (has loss)
        rotation_theta = torch.tensor(
            [[[0.9, -0.1, 0.1], [0.1, 0.9, 0.1]]], dtype=torch.float32
        )
        rotation_loss = stn.compute_regularization_loss(rotation_theta)
        assert rotation_loss.item() > 0.01

    def test_stn_gradient_flow(self):
        """Test gradients flow through STN properly."""
        stn = SpatialTransformerNetwork(input_channels=1, input_size=(640, 640))

        test_input = torch.randn(1, 1, 640, 640, requires_grad=True)
        output = stn(test_input)

        # Compute dummy loss and backprop
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert test_input.grad is not None
        for param in stn.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestSTNUtils:
    """Test STN utility functions."""

    def test_apply_stn_grayscale(self):
        """Test applying STN to grayscale numpy image."""
        stn = SpatialTransformerNetwork(input_channels=1, input_size=(640, 640))
        stn.eval()

        # Create test image
        image = np.random.randint(0, 255, (640, 640), dtype=np.uint8)

        # Apply STN
        transformed = apply_stn_to_image(stn, image, device="cpu")

        assert transformed.shape == image.shape
        assert transformed.dtype == np.uint8
        assert 0 <= transformed.min() <= transformed.max() <= 255

    def test_apply_stn_color(self):
        """Test applying STN to color numpy image."""
        stn = SpatialTransformerNetwork(input_channels=3, input_size=(640, 640))
        stn.eval()

        # Create test image
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Apply STN
        transformed = apply_stn_to_image(stn, image, device="cpu")

        assert transformed.shape == image.shape
        assert transformed.dtype == np.uint8

    def test_decompose_affine_identity(self):
        """Test affine matrix decomposition for identity."""
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        params = decompose_affine_matrix(identity)

        assert abs(params["rotation"]) < 0.1
        assert abs(params["scale_x"] - 1.0) < 0.1
        assert abs(params["scale_y"] - 1.0) < 0.1
        assert abs(params["shear"]) < 0.1
        assert abs(params["translation_x"]) < 0.1
        assert abs(params["translation_y"]) < 0.1

    def test_decompose_affine_rotation(self):
        """Test affine matrix decomposition for rotation."""
        # 45-degree rotation
        angle = np.pi / 4
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0]],
            dtype=np.float32,
        )

        params = decompose_affine_matrix(rotation)

        assert abs(params["rotation"] - 45.0) < 1.0  # ~45 degrees

    def test_decompose_affine_translation(self):
        """Test affine matrix decomposition for translation."""
        translation = np.array([[1, 0, 0.1], [0, 1, 0.2]], dtype=np.float32)

        params = decompose_affine_matrix(translation)

        assert abs(params["translation_x"] - 0.1) < 0.01
        assert abs(params["translation_y"] - 0.2) < 0.01

    def test_is_identity_transform_true(self):
        """Test identity transform detection for identity matrix."""
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        assert is_identity_transform(identity, tolerance=0.1)

    def test_is_identity_transform_false(self):
        """Test identity transform detection for non-identity matrix."""
        rotation = np.array([[0.9, -0.1, 0], [0.1, 0.9, 0]], dtype=np.float32)

        assert not is_identity_transform(rotation, tolerance=0.05)

    def test_is_identity_transform_tolerance(self):
        """Test identity transform detection with different tolerances."""
        near_identity = np.array([[1.05, 0, 0], [0, 1.05, 0]], dtype=np.float32)

        # Should fail with strict tolerance
        assert not is_identity_transform(near_identity, tolerance=0.01)

        # Should pass with loose tolerance
        assert is_identity_transform(near_identity, tolerance=0.1)


class TestSTNIntegration:
    """Test STN integration scenarios."""

    def test_stn_preserves_image_quality(self):
        """Test STN doesn't degrade image quality significantly."""
        stn = SpatialTransformerNetwork(input_channels=1, input_size=(640, 640))
        stn.eval()

        # Create test image with structure
        image = np.zeros((640, 640), dtype=np.uint8)
        image[200:400, 200:400] = 255  # White square

        # Apply STN (should be near-identity initially)
        transformed = apply_stn_to_image(stn, image, device="cpu")

        # Check similarity (should be very similar due to identity init)
        diff = np.abs(image.astype(float) - transformed.astype(float)).mean()
        assert diff < 10, f"Mean difference too large: {diff}"

    def test_stn_handles_different_sizes(self):
        """Test STN works with different input sizes."""
        stn = SpatialTransformerNetwork(input_channels=1, input_size=(640, 640))
        stn.eval()

        # Test with expected size
        image_640 = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
        transformed_640 = apply_stn_to_image(stn, image_640, device="cpu")
        assert transformed_640.shape == (640, 640)

        # Test with different size (should still work via adaptive pooling)
        image_1024 = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
        transformed_1024 = apply_stn_to_image(stn, image_1024, device="cpu")
        assert transformed_1024.shape == (1024, 1024)

    def test_stn_batch_processing(self):
        """Test STN can process batches efficiently."""
        stn = SpatialTransformerNetwork(input_channels=1, input_size=(640, 640))
        stn.eval()

        batch_size = 4
        test_batch = torch.randn(batch_size, 1, 640, 640)

        with torch.no_grad():
            output_batch = stn(test_batch)

        assert output_batch.shape == test_batch.shape

        # With identity initialization, output should be very similar to input
        # (grid_sample may introduce minor interpolation differences)
        for i in range(batch_size):
            diff = torch.abs(output_batch[i] - test_batch[i]).mean()
            assert diff < 0.1, f"Batch {i} differs too much: {diff}"


class TestSTNRobustness:
    """Test STN robustness to edge cases."""

    def test_stn_handles_zeros(self):
        """Test STN handles all-zero input gracefully."""
        stn = SpatialTransformerNetwork(input_channels=1, input_size=(640, 640))
        stn.eval()

        zeros = torch.zeros(1, 1, 640, 640)
        output = stn(zeros)

        assert output.shape == zeros.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_stn_handles_ones(self):
        """Test STN handles all-ones input gracefully."""
        stn = SpatialTransformerNetwork(input_channels=1, input_size=(640, 640))
        stn.eval()

        ones = torch.ones(1, 1, 640, 640)
        output = stn(ones)

        assert output.shape == ones.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_stn_numerical_stability(self):
        """Test STN maintains numerical stability."""
        stn = SpatialTransformerNetwork(input_channels=1, input_size=(640, 640))
        stn.eval()

        # Test with various input ranges
        test_inputs = [
            torch.randn(1, 1, 640, 640) * 0.01,  # Small values
            torch.randn(1, 1, 640, 640),  # Normal values
            torch.randn(1, 1, 640, 640) * 10,  # Large values
        ]

        for test_input in test_inputs:
            output = stn(test_input)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
