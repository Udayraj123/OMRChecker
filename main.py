import argparse
import sys
from pathlib import Path

from src.entry import entry_point
from src.utils.constants import OUTPUT_MODES
from src.utils.logger import logger


def parse_args():
    # construct the argument parse and parse the arguments
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "-i",
        "--inputDir",
        default=["inputs"],
        # https://docs.python.org/3/library/argparse.html#nargs
        nargs="*",
        required=False,
        type=str,
        dest="input_paths",
        help="Specify an input directory.",
    )

    argparser.add_argument(
        "-d",
        "--debug",
        required=False,
        dest="debug",
        action="store_true",
        help="Enables debugging mode for showing detailed errors",
    )

    argparser.add_argument(
        "-o",
        "--outputDir",
        default="outputs",
        required=False,
        dest="output_dir",
        help="Specify an output directory.",
    )

    argparser.add_argument(
        "-m",
        "--outputMode",
        default="default",
        required=False,
        choices=[*list(OUTPUT_MODES.values())],
        dest="outputMode",
        help="Specify the output mode. Supported: moderation, default",
    )

    argparser.add_argument(
        "-l",
        "--setLayout",
        required=False,
        dest="setLayout",
        action="store_true",
        help="Set up OMR template layout - modify your json file and \
        run again until the template is set.",
    )

    # ML Training arguments
    argparser.add_argument(
        "--collect-training-data",
        required=False,
        dest="collect_training_data",
        action="store_true",
        help="Collect high-confidence detections for ML training",
    )

    argparser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        required=False,
        dest="confidence_threshold",
        help="Minimum confidence for including in training data (0.0-1.0)",
    )

    argparser.add_argument(
        "--mode",
        choices=[
            "process",
            "auto-train",
            "auto-train-hierarchical",
            "test-model",
            "export-yolo",
        ],
        default="process",
        required=False,
        dest="mode",
        help="Operation mode: process (default), auto-train, auto-train-hierarchical, test-model, or export-yolo",
    )

    argparser.add_argument(
        "--use-ml-fallback",
        type=str,
        required=False,
        dest="ml_model_path",
        help="Path to trained YOLO bubble model for low-confidence fallback",
    )

    argparser.add_argument(
        "--use-field-block-detection",
        action="store_true",
        required=False,
        dest="use_field_block_detection",
        help="Enable ML field block detection (Stage 1) during processing",
    )

    argparser.add_argument(
        "--field-block-model",
        type=str,
        required=False,
        dest="field_block_model_path",
        help="Path to trained YOLO field block model",
    )

    argparser.add_argument(
        "--enable-shift-detection",
        action="store_true",
        required=False,
        dest="enable_shift_detection",
        help="Enable ML-based field block shift detection and application (requires field block model)",
    )

    argparser.add_argument(
        "--fusion-strategy",
        choices=["confidence_weighted", "ml_fallback", "traditional_primary"],
        default="confidence_weighted",
        required=False,
        dest="fusion_strategy",
        help="Detection fusion strategy when using ML models",
    )

    argparser.add_argument(
        "--training-data-dir",
        type=str,
        default="outputs/training_data",
        required=False,
        dest="training_data_dir",
        help="Directory containing training data",
    )

    argparser.add_argument(
        "--epochs",
        type=int,
        default=100,
        required=False,
        dest="epochs",
        help="Number of training epochs for auto-train mode",
    )

    (
        args,
        unknown,
    ) = argparser.parse_known_args()

    args = vars(args)

    if len(unknown) > 0:
        argparser.print_help()
        msg = f"\nError: Unknown arguments: {unknown}"
        raise Exception(msg)

    if args["setLayout"] is True:
        if args["outputMode"] not in {OUTPUT_MODES.SET_LAYOUT, OUTPUT_MODES.DEFAULT}:
            msg = f"Error: --setLayout cannot be used together with --outputMode={args['outputMode']}"
            raise Exception(msg)
        args["outputMode"] = "setLayout"
    return args


def entry_point_for_args(args) -> None:
    if args["debug"] is False:
        # Disable traceback limit
        sys.tracebacklimit = 0
        logger.set_log_levels({"debug": True})

    # Handle different modes
    mode = args.get("mode", "process")

    if mode == "auto-train":
        # Auto-training mode
        from src.training.trainer import AutoTrainer

        trainer = AutoTrainer(
            training_data_dir=args["training_data_dir"], epochs=args["epochs"]
        )
        trainer.train_from_collected_data()
        return

    if mode == "auto-train-hierarchical":
        # Hierarchical auto-training mode (field blocks + bubbles)
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
        return

    if mode == "export-yolo":
        # Export to YOLO format mode
        from src.processors.training.yolo_exporter import (
            YOLOAnnotationExporter,
        )

        training_data_dir = Path(args["training_data_dir"])
        dataset_dir = training_data_dir / "dataset"
        source_images = training_data_dir / "dataset/images"
        source_labels = training_data_dir / "dataset/labels"

        exporter = YOLOAnnotationExporter(dataset_dir)
        exporter.convert_dataset(source_images, source_labels)
        logger.info("YOLO dataset export complete!")
        return

    if mode == "test-model":
        # Model testing mode
        logger.info("Model testing mode - coming soon!")
        return

    # Default: process mode
    for root in args["input_paths"]:
        try:
            entry_point(
                Path(root),
                args,
            )
        except Exception:
            if args["debug"] is False:
                logger.critical(
                    "OMRChecker crashed. add --debug and run again to see error details"
                )
            raise


if __name__ == "__main__":
    args = parse_args()
    entry_point_for_args(args)
