"""Train field block detector with augmented data including shifts."""

from pathlib import Path

from src.training.trainer import AutoTrainer
from src.utils.logger import logger


def main() -> None:
    """Train field block detector with augmented dataset."""
    logger.info("=" * 80)
    logger.info("Training Field Block Detector with Augmented Data")
    logger.info("=" * 80)

    # Initialize trainer
    trainer = AutoTrainer(
        training_data_dir=Path("outputs/training_data"),
        epochs=50,  # More epochs for better convergence
        batch_size=16,
        image_size=640,
    )

    # Train field block detector
    logger.info("\nTraining with augmented dataset (including shift data)...")
    logger.info("  Dataset: outputs/training_data/yolo_field_blocks_augmented")
    logger.info("  Epochs: 50")
    logger.info("  Batch size: 16")
    logger.info("  Image size: 640")

    model_path, metrics = trainer.train_field_block_detector(
        dataset_path=Path("outputs/training_data/yolo_field_blocks_augmented"),
        epochs=50,
    )

    logger.info("\n" + "=" * 80)
    logger.info("✅ Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Model saved: {model_path}")
    logger.info("\nMetrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
