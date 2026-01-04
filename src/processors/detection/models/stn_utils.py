"""Utilities for loading, saving, and using STN models.

Provides helper functions for STN model management, inference,
and visualization of learned transformations.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from src.processors.detection.models.stn_module import (
    SpatialTransformerNetwork,
)
from src.utils.logger import logger


def load_stn_model(
    model_path: str | Path,
    input_channels: int = 1,
    input_size: tuple[int, int] = (640, 640),
    device: str = "cpu",
) -> SpatialTransformerNetwork:
    """Load a trained STN model from disk.

    Args:
        model_path: Path to saved model weights (.pt file)
        input_channels: Number of input channels (1 for grayscale)
        input_size: Expected input image size
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded STN model in eval mode

    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    model_path = Path(model_path)

    if not model_path.exists():
        msg = f"STN model not found at {model_path}"
        raise FileNotFoundError(msg)

    try:
        # Initialize model architecture
        model = SpatialTransformerNetwork(
            input_channels=input_channels, input_size=input_size
        )

        # Load trained weights
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        # Set to evaluation mode
        model.eval()
        model.to(device)

        logger.info(f"Loaded STN model from {model_path}")
        return model

    except Exception as e:
        msg = f"Failed to load STN model: {e}"
        raise RuntimeError(msg) from e


def save_stn_model(
    model: SpatialTransformerNetwork,
    save_path: str | Path,
    metadata: dict | None = None,
) -> None:
    """Save STN model weights and metadata to disk.

    Args:
        model: STN model to save
        save_path: Path to save model (.pt file)
        metadata: Optional metadata dictionary to save alongside model
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved STN model to {save_path}")

    # Save metadata if provided
    if metadata:
        import json

        metadata_path = save_path.with_suffix(".json")
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model metadata to {metadata_path}")


def apply_stn_to_image(
    model: SpatialTransformerNetwork,
    image: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Apply STN transformation to a single numpy image.

    Handles conversion between numpy/torch formats and applies the
    learned spatial transformation.

    Args:
        model: Trained STN model
        image: Input image as numpy array (H, W) or (H, W, C)
        device: Device to run inference on

    Returns:
        Transformed image as numpy array (same shape as input)
    """
    # Handle grayscale vs color images
    if len(image.shape) == 2:
        # Grayscale: add channel dimension
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        # Color: transpose from (H, W, C) to (C, H, W)
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    else:
        msg = f"Unexpected image shape: {image.shape}"
        raise ValueError(msg)

    # Normalize to [0, 1] if needed
    if image_tensor.max() > 1:
        image_tensor = image_tensor / 255.0

    # Move to device
    image_tensor = image_tensor.to(device)

    # Apply STN
    with torch.no_grad():
        transformed_tensor = model(image_tensor)

    # Convert back to numpy
    transformed = transformed_tensor.cpu().squeeze(0)

    if transformed.shape[0] == 1:
        # Grayscale: remove channel dimension
        transformed = transformed.squeeze(0).numpy()
    else:
        # Color: transpose back to (H, W, C)
        transformed = transformed.permute(1, 2, 0).numpy()

    # Denormalize to [0, 255]
    if transformed.max() <= 1:
        transformed = (transformed * 255).clip(0, 255).astype(np.uint8)
    else:
        transformed = transformed.clip(0, 255).astype(np.uint8)

    return transformed


def get_transformation_matrix(
    model: SpatialTransformerNetwork,
    image: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Get the predicted transformation matrix for an image.

    Useful for debugging and visualization.

    Args:
        model: Trained STN model
        image: Input image as numpy array
        device: Device to run inference on

    Returns:
        Affine transformation matrix as numpy array (2, 3)
    """
    # Prepare input
    if len(image.shape) == 2:
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    else:
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)

    if image_tensor.max() > 1:
        image_tensor = image_tensor / 255.0

    image_tensor = image_tensor.to(device)

    # Get transformation parameters
    with torch.no_grad():
        theta = model.get_transformation_params(image_tensor)

    return theta.cpu().numpy()[0]


def visualize_transformation(
    original: np.ndarray,
    transformed: np.ndarray,
    theta: np.ndarray,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Create a visualization comparing original and transformed images.

    Args:
        original: Original input image
        transformed: STN-transformed image
        theta: Transformation matrix (2, 3)
        save_path: Optional path to save visualization

    Returns:
        Visualization image showing original, transformed, and difference
    """
    # Ensure images are uint8
    if original.dtype != np.uint8:
        original = (
            (original * 255).astype(np.uint8)
            if original.max() <= 1
            else original.astype(np.uint8)
        )
    if transformed.dtype != np.uint8:
        transformed = (
            (transformed * 255).astype(np.uint8)
            if transformed.max() <= 1
            else transformed.astype(np.uint8)
        )

    # Convert grayscale to color for visualization
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    if len(transformed.shape) == 2:
        transformed = cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)

    # Compute difference (absolute)
    diff = cv2.absdiff(original, transformed)

    # Create side-by-side comparison
    h, w = original.shape[:2]
    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
    vis[:, :w] = original
    vis[:, w : 2 * w] = transformed
    vis[:, 2 * w :] = diff

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, "Original", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(vis, "Transformed", (w + 10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(vis, "Difference", (2 * w + 10, 30), font, 1, (0, 255, 0), 2)

    # Add transformation parameters
    y_offset = 60
    cv2.putText(vis, "Matrix:", (10, y_offset), font, 0.5, (255, 255, 255), 1)
    cv2.putText(
        vis,
        f"[{theta[0, 0]:.3f} {theta[0, 1]:.3f} {theta[0, 2]:.3f}]",
        (10, y_offset + 20),
        font,
        0.4,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        vis,
        f"[{theta[1, 0]:.3f} {theta[1, 1]:.3f} {theta[1, 2]:.3f}]",
        (10, y_offset + 40),
        font,
        0.4,
        (255, 255, 255),
        1,
    )

    # Save if requested
    if save_path:
        cv2.imwrite(str(save_path), vis)
        logger.info(f"Saved transformation visualization to {save_path}")

    return vis


def decompose_affine_matrix(theta: np.ndarray) -> dict:
    """Decompose affine transformation matrix into interpretable parameters.

    Extracts rotation, scale, shear, and translation from 2x3 affine matrix.

    Args:
        theta: Affine transformation matrix (2, 3)

    Returns:
        Dictionary with decomposed parameters:
            - rotation: Rotation angle in degrees
            - scale_x: X-axis scaling factor
            - scale_y: Y-axis scaling factor
            - shear: Shear angle in degrees
            - translation_x: X-axis translation
            - translation_y: Y-axis translation
    """
    # Extract components
    a, b, tx = theta[0]
    c, d, ty = theta[1]

    # Compute scale
    scale_x = np.sqrt(a**2 + c**2)
    scale_y = np.sqrt(b**2 + d**2)

    # Compute rotation (in degrees)
    rotation = np.arctan2(c, a) * 180 / np.pi

    # Compute shear
    shear = np.arctan2(b, d) * 180 / np.pi - rotation

    return {
        "rotation": rotation,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "shear": shear,
        "translation_x": tx,
        "translation_y": ty,
    }


def is_identity_transform(theta: np.ndarray, tolerance: float = 0.1) -> bool:
    """Check if transformation matrix is approximately identity.

    Useful for detecting when STN has learned to do nothing.

    Args:
        theta: Affine transformation matrix (2, 3)
        tolerance: Maximum deviation from identity to consider equal

    Returns:
        True if transformation is approximately identity
    """
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    diff = np.abs(theta - identity)
    return np.all(diff < tolerance)
