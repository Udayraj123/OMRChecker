"""CLI runner for OMRChecker - Command-line interface implementation.

This module contains the CLI-specific orchestration logic, delegating actual
processing to the original entry point logic.
"""

import sys
from pathlib import Path

from src.entry import entry_point
from src.utils.logger import logger


def run_cli(args: dict) -> None:
    """Run OMRChecker from command-line arguments.

    This is the main CLI entry point that handles different modes
    (process, auto-train, export-yolo, etc.) and delegates to appropriate handlers.

    Args:
        args: Parsed command-line arguments dictionary
    """
    # Setup debugging
    if args["debug"] is False:
        # Disable traceback limit
        sys.tracebacklimit = 0
        logger.set_log_levels({"debug": True})

    # Handle different modes
    mode = args.get("mode", "process")

    if mode == "auto-train":
        _run_auto_train_mode(args)
        return

    if mode == "auto-train-hierarchical":
        _run_hierarchical_train_mode(args)
        return

    if mode == "export-yolo":
        _run_export_yolo_mode(args)
        return

    if mode == "test-model":
        logger.info("Model testing mode - coming soon!")
        return

    # Default: process mode
    _run_process_mode(args)


def _run_process_mode(args: dict) -> None:
    """Run standard OMR processing mode.

    This processes input directories using the original entry point.
    """
    for root in args["input_paths"]:
        try:
            # Use original entry_point which has all the CSV writing logic
            entry_point(Path(root), args)

        except Exception:
            if args["debug"] is False:
                logger.critical(
                    "OMRChecker crashed. add --debug and run again to see error details"
                )
            raise


def _run_auto_train_mode(args: dict) -> None:
    """Run auto-training mode for ML models."""
    from src.training.trainer import AutoTrainer

    trainer = AutoTrainer(
        training_data_dir=args["training_data_dir"], epochs=args["epochs"]
    )
    trainer.train_from_collected_data()


def _run_hierarchical_train_mode(args: dict) -> None:
    """Run hierarchical auto-training mode (field blocks + bubbles)."""
    from src.training.trainer import AutoTrainer

    trainer = AutoTrainer(epochs=args["epochs"])

    # Train both stages
    results = trainer.train_hierarchical_pipeline(
        field_block_dataset=Path(args["output_dir"])
        / "training_data"
        / "field_blocks"
        / "dataset",
        bubble_dataset=Path(args["output_dir"])
        / "training_data"
        / "bubbles"
        / "dataset",
    )

    logger.info("=" * 60)
    logger.info("Hierarchical Training Results:")
    logger.info(f"  Field Block Model: {results['field_block_model']}")
    logger.info(f"  Bubble Model: {results['bubble_model']}")
    logger.info("=" * 60)


def _run_export_yolo_mode(args: dict) -> None:
    """Run YOLO dataset export mode."""
    from src.processors.training.yolo_exporter import YOLOAnnotationExporter

    training_data_dir = Path(args["training_data_dir"])
    dataset_dir = training_data_dir / "dataset"
    source_images = training_data_dir / "dataset/images"
    source_labels = training_data_dir / "dataset/labels"

    exporter = YOLOAnnotationExporter(dataset_dir)
    exporter.convert_dataset(source_images, source_labels)
    logger.info("YOLO dataset export complete!")


def _print_processing_summary(result) -> None:
    """Print a summary of processing results.

    Args:
        result: DirectoryProcessingResult object
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("Processing Summary:")
    logger.info(f"  Total files: {result.total_files}")
    logger.info(f"  Successful: {result.successful}")
    logger.info(f"  Errors: {result.errors}")
    logger.info(f"  Multi-marked: {result.multi_marked}")
    logger.info(f"  Processing time: {result.processing_time:.2f}s")
    if result.total_files > 0:
        rate = result.processing_time / result.total_files
        logger.info(f"  Rate: {rate:.2f}s per sheet")
    logger.info("=" * 60)
