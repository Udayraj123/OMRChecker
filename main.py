#!/usr/bin/env python3
"""Main entry point for OMRChecker.

This module provides the command-line interface for the OMR Checker application.
It handles argument parsing and delegates to the CLI runner.

Usage:
    uv run main.py
    uv run main.py -i <input_dir>
    uv run main.py --inputDir <input_dir> --outputDir <output_dir>
    uv run main.py --setLayout -i <input_dir>
"""

import argparse
import sys
from pathlib import Path

from src.cli import run_cli
from src.utils.logger import logger


def parse_args() -> dict:
    """Parse command-line arguments and return as a dictionary.

    Returns:
        Dictionary containing parsed arguments:
            - input_paths: List of input directory paths
            - output_dir: Output directory path
            - debug: Debug mode flag
            - setLayout: Layout mode flag
            - outputMode: Output mode string
            - mode: Processing mode
            - collect_training_data: Training data collection flag
            - confidence_threshold: Confidence threshold for ML training
    """
    parser = argparse.ArgumentParser(
        description="OMRChecker - Evaluate OMR sheets fast and accurately",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default inputs directory
  uv run main.py

  # Process a specific directory
  uv run main.py -i inputs/sample1

  # Process with custom output directory
  uv run main.py -i inputs/sample1 -o outputs/results

  # Set up template layout (interactive mode)
  uv run main.py --setLayout -i inputs/sample1

  # Process with debug mode
  uv run main.py -i inputs/sample1 --debug


For more information, visit: https://github.com/Udayraj123/OMRChecker
        """,
    )

    # Input/Output arguments
    parser.add_argument(
        "-i",
        "--inputDir",
        dest="input_paths",
        type=str,
        nargs="+",
        default=["./inputs"],
        help="Path(s) to input directory containing OMR sheets (default: ./inputs)",
    )

    parser.add_argument(
        "-o",
        "--outputDir",
        dest="output_dir",
        type=str,
        default="./outputs",
        help="Path to output directory for results (default: ./outputs)",
    )

    # Mode arguments
    parser.add_argument(
        "--setLayout",
        action="store_true",
        help="Set up OMR template layout (interactive mode)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed logging",
    )

    # Output mode
    parser.add_argument(
        "--outputMode",
        type=str,
        default="default",
        choices=["default", "csv", "moderation"],
        help="Output mode for results (default: default)",
    )

    # Processing mode
    parser.add_argument(
        "--mode",
        type=str,
        default="process",
        choices=["process", "auto-train"],
        help="Processing mode (default: process)",
    )

    # ML Training arguments
    parser.add_argument(
        "--collect-training-data",
        dest="collect_training_data",
        action="store_true",
        help="Collect training data for ML models",
    )

    parser.add_argument(
        "--confidence-threshold",
        dest="confidence_threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for ML training (default: 0.8)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Convert to dictionary format expected by the rest of the codebase
    args_dict = {
        "input_paths": args.input_paths,
        "output_dir": args.output_dir,
        "debug": args.debug,
        "setLayout": args.setLayout,
        "outputMode": args.outputMode,
        "mode": args.mode,
        "collect_training_data": args.collect_training_data,
        "confidence_threshold": args.confidence_threshold,
    }

    return args_dict


def validate_paths(args: dict) -> None:
    """Validate input and output paths.

    Args:
        args: Dictionary containing parsed arguments

    Raises:
        SystemExit: If validation fails
    """
    errors = []

    # Validate input paths
    for input_path in args["input_paths"]:
        path = Path(input_path)
        if not path.exists():
            errors.append(f"Input directory not found: {input_path}")

    # Validate output directory parent exists
    output_path = Path(args["output_dir"])
    if output_path.exists() and not output_path.is_dir():
        errors.append(
            f"Output path exists but is not a directory: {args['output_dir']}"
        )

    if errors:
        logger.error("Validation errors:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)


def main() -> int:
    """Main entry point for the application.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Parse arguments
        args = parse_args()

        # Validate paths
        validate_paths(args)

        # Run the CLI
        run_cli(args)

        return 0

    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        return 130

    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"Error: {e}")
        logger.error("=" * 70)
        if args.get("debug", False):
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
