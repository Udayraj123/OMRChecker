"""Train STN module using existing augmented OMR data.

This script trains a Spatial Transformer Network to learn alignment corrections
from augmented OMR images with synthetic distortions (shifts, rotations).

The training approach:
1. Loads augmented images with known distortions (shift/rotation metadata)
2. Trains STN end-to-end with frozen YOLO to predict corrections
3. Optimizes alignment loss (bbox matching between STN+YOLO predictions and ground truth)

Usage:
    python train_stn_yolo.py --augmented-data outputs/training_data/yolo_field_blocks_augmented
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.processors.detection.models.stn_module import STNWithRegularization
from src.processors.detection.models.stn_utils import save_stn_model
from src.utils.logger import logger


class AugmentedOMRDataset(Dataset):
    """Dataset for STN training using augmented OMR images.

    Loads images and labels from YOLO-format augmented dataset,
    extracting ground truth bounding boxes for training.
    """

    def __init__(self, dataset_dir: Path, split: str = "train") -> None:
        """Initialize dataset.

        Args:
            dataset_dir: Path to augmented dataset directory
            split: Dataset split ("train" or "val")
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split

        # Load images and labels
        self.image_dir = self.dataset_dir / "images" / split
        self.label_dir = self.dataset_dir / "labels" / split

        if not self.image_dir.exists():
            msg = f"Image directory not found: {self.image_dir}"
            raise FileNotFoundError(msg)

        self.image_files = sorted(self.image_dir.glob("*.jpg"))
        logger.info(
            f"Loaded {len(self.image_files)} {split} images from {self.image_dir}"
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single training sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, bbox_tensor):
                - image_tensor: (1, H, W) grayscale image
                - bbox_tensor: (N, 5) ground truth boxes [class, x_center, y_center, width, height]
        """
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            msg = f"Failed to load image: {image_path}"
            raise ValueError(msg)

        # Load labels (YOLO format)
        label_path = self.label_dir / (image_path.stem + ".txt")
        bboxes = []

        if label_path.exists():
            with label_path.open() as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # YOLO format: class x_center y_center width height (normalized)
                        bboxes.append([float(x) for x in parts])

        # Convert to tensors
        image_tensor = (
            torch.from_numpy(image).float().unsqueeze(0) / 255.0
        )  # Normalize to [0,1]
        bbox_tensor = (
            torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 5))
        )

        return image_tensor, bbox_tensor


def compute_alignment_loss(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> torch.Tensor:
    """Compute alignment loss between predicted and ground truth boxes.

    Measures how well the STN-transformed image aligns detections with ground truth.
    Uses IoU loss + center distance loss for robust matching.

    Args:
        pred_boxes: Predicted bounding boxes from YOLO (N, 5) [class, x, y, w, h] normalized
        gt_boxes: Ground truth bounding boxes (M, 5) [class, x, y, w, h] normalized

    Returns:
        Scalar alignment loss
    """
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        # No boxes detected or no ground truth - high penalty
        return torch.tensor(10.0, device=pred_boxes.device)

    # Extract centers and sizes
    pred_centers = pred_boxes[:, 1:3]  # (N, 2)
    pred_sizes = pred_boxes[:, 3:5]  # (N, 2)
    gt_centers = gt_boxes[:, 1:3]  # (M, 2)
    gt_sizes = gt_boxes[:, 3:5]  # (M, 2)

    # Compute pairwise center distance (N, M)
    pred_centers_exp = pred_centers.unsqueeze(1)  # (N, 1, 2)
    gt_centers_exp = gt_centers.unsqueeze(0)  # (1, M, 2)
    center_dist = torch.sqrt(torch.sum((pred_centers_exp - gt_centers_exp) ** 2, dim=2))

    # Compute pairwise size similarity (N, M)
    pred_sizes_exp = pred_sizes.unsqueeze(1)
    gt_sizes_exp = gt_sizes.unsqueeze(0)
    size_diff = torch.sum(torch.abs(pred_sizes_exp - gt_sizes_exp), dim=2)

    # Combined matching cost
    matching_cost = center_dist + size_diff

    # Hungarian matching (simplified: greedy assignment)
    min_costs, _ = torch.min(matching_cost, dim=1)
    return torch.mean(min_costs)


def _parse_yolo_predictions(results, img_np: np.ndarray, device: str) -> torch.Tensor:
    """Parse YOLO prediction results into normalized boxes.

    Args:
        results: YOLO prediction results
        img_np: Input image as numpy array
        device: Device to create tensor on

    Returns:
        Tensor of predicted boxes (N, 5) [class, x_center, y_center, width, height] normalized
    """
    pred_boxes_list = []
    if results and len(results) > 0:
        for result in results:
            if hasattr(result, "boxes") and result.boxes is not None:
                for box in result.boxes:
                    # Get normalized xywh format
                    xywh = (
                        box.xywh[0].cpu().numpy()
                    )  # center_x, center_y, width, height (pixels)
                    class_id = int(box.cls[0])

                    # Normalize by image size
                    h, w = img_np.shape
                    norm_xywh = [
                        class_id,
                        xywh[0] / w,
                        xywh[1] / h,
                        xywh[2] / w,
                        xywh[3] / h,
                    ]
                    pred_boxes_list.append(norm_xywh)

    if pred_boxes_list:
        return torch.tensor(pred_boxes_list, device=device)
    return torch.zeros((0, 5), device=device)


def _process_single_image(
    transformed_image: torch.Tensor,
    gt_boxes: torch.Tensor,
    yolo_model,
    device: str,
) -> torch.Tensor:
    """Process a single image through YOLO and compute loss.

    Args:
        transformed_image: STN-transformed image tensor
        gt_boxes: Ground truth boxes for this image
        yolo_model: YOLO model for predictions
        device: Device to run on

    Returns:
        Alignment loss for this image
    """
    # Convert to numpy for YOLO
    img_np = (transformed_image.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # Run YOLO prediction
    with torch.no_grad():
        results = yolo_model.predict(img_np, conf=0.3, verbose=False)

    # Parse predictions
    pred_boxes = _parse_yolo_predictions(results, img_np, device)

    # Compute alignment loss
    if gt_boxes.shape[0] > 0:
        return compute_alignment_loss(pred_boxes, gt_boxes)
    return torch.tensor(0.0, device=device)


def train_epoch(
    stn_model: nn.Module,
    yolo_model,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
) -> dict:
    """Train STN for one epoch.

    Args:
        stn_model: STN module to train
        yolo_model: Frozen YOLO model for supervision
        dataloader: Training data loader
        optimizer: Optimizer for STN parameters
        device: Device to train on

    Returns:
        Dictionary with training metrics
    """
    stn_model.train()
    yolo_model.eval()  # Keep YOLO frozen

    total_loss = 0.0
    total_alignment_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0

    for batch_idx, (images, gt_boxes) in enumerate(dataloader):
        images = images.to(device)

        optimizer.zero_grad()

        # Forward pass: STN -> YOLO
        if isinstance(stn_model, STNWithRegularization):
            transformed_images, reg_loss = stn_model.forward_with_regularization(images)
        else:
            transformed_images = stn_model(images)
            reg_loss = torch.tensor(0.0, device=device)

        # Compute alignment loss for each image in batch
        batch_alignment_loss = torch.tensor(0.0, device=device)
        for i in range(images.shape[0]):
            img_gt_boxes = gt_boxes[i].to(device)
            img_loss = _process_single_image(
                transformed_images[i], img_gt_boxes, yolo_model, device
            )
            batch_alignment_loss += img_loss

        # Average alignment loss over batch
        batch_alignment_loss = batch_alignment_loss / images.shape[0]

        # Total loss
        loss = batch_alignment_loss + reg_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_alignment_loss += batch_alignment_loss.item()
        total_reg_loss += reg_loss.item()
        num_batches += 1

        if (batch_idx + 1) % 5 == 0:
            logger.info(
                f"Batch {batch_idx + 1}/{len(dataloader)}: "
                f"Loss={loss.item():.4f}, "
                f"Align={batch_alignment_loss.item():.4f}, "
                f"Reg={reg_loss.item():.4f}"
            )

    return {
        "loss": total_loss / num_batches,
        "alignment_loss": total_alignment_loss / num_batches,
        "reg_loss": total_reg_loss / num_batches,
    }


def validate(
    stn_model: nn.Module,
    yolo_model,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Validate STN model.

    Args:
        stn_model: STN module to validate
        yolo_model: YOLO model for evaluation
        dataloader: Validation data loader
        device: Device to run on

    Returns:
        Dictionary with validation metrics
    """
    stn_model.eval()
    yolo_model.eval()

    total_alignment_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, gt_boxes in dataloader:
            images = images.to(device)

            # Apply STN
            transformed_images = stn_model(images)

            # Compute alignment loss for each image
            batch_alignment_loss = torch.tensor(0.0, device=device)
            for i in range(images.shape[0]):
                img_gt_boxes = gt_boxes[i].to(device)
                img_loss = _process_single_image(
                    transformed_images[i], img_gt_boxes, yolo_model, device
                )
                batch_alignment_loss += img_loss

            batch_alignment_loss = batch_alignment_loss / images.shape[0]
            total_alignment_loss += batch_alignment_loss.item()
            num_batches += 1

    return {
        "val_alignment_loss": total_alignment_loss / num_batches,
    }


def _load_yolo_model(models_dir: Path) -> tuple[Any, Path] | tuple[None, None]:
    """Load YOLO model from directory.

    Args:
        models_dir: Directory containing YOLO models

    Returns:
        Tuple of (yolo_model, model_path) or (None, None) on failure
    """
    yolo_models = sorted(models_dir.glob("field_block_detector_*.pt"))

    if not yolo_models:
        logger.error(f"No YOLO field block detector found in {models_dir}")
        logger.error("Please train a field block detector first.")
        return None, None

    yolo_model_path = yolo_models[-1]  # Use latest
    logger.info(f"Using YOLO model: {yolo_model_path.name}")

    try:
        from ultralytics import YOLO

        yolo_model = YOLO(str(yolo_model_path))
        yolo_model.eval()  # Freeze YOLO
        logger.info("YOLO model loaded successfully")
        return yolo_model, yolo_model_path
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return None, None


def _load_datasets(
    data_path: Path, batch_size: int
) -> tuple[DataLoader, DataLoader] | tuple[None, None]:
    """Load train and validation datasets.

    Args:
        data_path: Path to augmented dataset
        batch_size: Batch size for data loaders

    Returns:
        Tuple of (train_loader, val_loader) or (None, None) on failure
    """
    try:
        train_dataset = AugmentedOMRDataset(data_path, split="train")
        val_dataset = AugmentedOMRDataset(data_path, split="val")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        return train_loader, val_loader
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return None, None


def _save_best_model(
    stn_model: nn.Module,
    val_loss: float,
    epoch: int,
    train_metrics: dict,
    models_dir: Path,
) -> str:
    """Save the best model with metadata.

    Args:
        stn_model: STN model to save
        val_loss: Validation loss
        epoch: Current epoch number
        train_metrics: Training metrics dictionary
        models_dir: Directory to save model

    Returns:
        Model filename
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    model_name = f"stn_refinement_{timestamp}.pt"
    model_path = models_dir / model_name

    metadata = {
        "epoch": epoch + 1,
        "val_alignment_loss": val_loss,
        "train_metrics": train_metrics,
        "timestamp": timestamp,
    }

    save_stn_model(stn_model, model_path, metadata)
    logger.info(f"✅ Saved best model: {model_name}")
    return model_name


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train STN for OMR alignment refinement"
    )
    parser.add_argument(
        "--augmented-data",
        type=str,
        default="outputs/training_data/yolo_field_blocks_augmented",
        help="Path to augmented dataset directory",
    )
    parser.add_argument(
        "--yolo-model", type=str, help="Path to trained YOLO field block detector"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/models",
        help="Output directory for trained model",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("STN Training for OMR Alignment Refinement")
    logger.info("=" * 80)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load YOLO model
    models_dir = Path(args.output_dir)
    yolo_model, _ = _load_yolo_model(models_dir)
    if yolo_model is None:
        return 1

    # Load datasets
    train_loader, val_loader = _load_datasets(
        Path(args.augmented_data), args.batch_size
    )
    if train_loader is None:
        return 1

    # Initialize STN model
    stn_model = STNWithRegularization(
        input_channels=1, input_size=(1024, 1024), regularization_weight=0.1
    ).to(device)

    logger.info(f"STN parameters: {sum(p.numel() for p in stn_model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(stn_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Training loop
    best_val_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-" * 40)

        # Train
        train_metrics = train_epoch(
            stn_model, yolo_model, train_loader, optimizer, device
        )
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, Alignment: {train_metrics['alignment_loss']:.4f}"
        )

        # Validate
        val_metrics = validate(stn_model, yolo_model, val_loader, device)
        logger.info(f"Val - Alignment Loss: {val_metrics['val_alignment_loss']:.4f}")

        # Learning rate scheduling
        scheduler.step(val_metrics["val_alignment_loss"])

        # Save best model
        if val_metrics["val_alignment_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_alignment_loss"]
            _save_best_model(stn_model, best_val_loss, epoch, train_metrics, models_dir)

        # Track history
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

    # Save training history

    history_path = models_dir / "stn_training_history.json"
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("✅ STN Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Best validation alignment loss: {best_val_loss:.4f}")
    logger.info(f"Training history saved to: {history_path}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
