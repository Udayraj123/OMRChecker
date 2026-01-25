"""Export augmented data to YOLO format for training."""

from pathlib import Path

from src.processors.training.yolo_exporter import YOLOAnnotationExporter
from src.utils.logger import logger


def main() -> None:
    """Export augmented data with field block shifts to YOLO format."""
    logger.info("=" * 80)
    logger.info("Exporting Augmented Data to YOLO Format")
    logger.info("=" * 80)

    # Export field block dataset
    logger.info("\n1. Exporting Field Block Detection Dataset...")
    field_block_exporter = YOLOAnnotationExporter(
        dataset_dir=Path("outputs/training_data/yolo_field_blocks_augmented"),
        dataset_type="field_block",
    )

    field_block_exporter.convert_dataset(
        source_images_dir=Path("outputs/training_data/augmented/images"),
        source_labels_dir=Path("outputs/training_data/augmented/labels"),
    )

    logger.info("\n✅ Export complete!")
    logger.info(
        "   Field Block Dataset: outputs/training_data/yolo_field_blocks_augmented"
    )


if __name__ == "__main__":
    main()
