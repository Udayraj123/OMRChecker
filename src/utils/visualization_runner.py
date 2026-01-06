r"""CLI tool for workflow visualization.

This module provides a command-line interface for running OMR processing
with visualization tracking enabled, generating interactive HTML reports.

Usage:
    python -m src.utils.visualization_runner \\
        --input inputs/sample1/sample1.jpg \\
        --template inputs/sample1/template.json \\
        --config inputs/sample1/config.json \\
        --output outputs/visualization \\
        --capture-processors "AutoRotate,ReadOMR"
"""

import argparse
import sys
from pathlib import Path

from src.processors.visualization import export_to_html, track_workflow
from src.utils.logger import logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="OMR Workflow Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a single OMR file with all processors
  python -m src.utils.visualization_runner \\
      --input inputs/sample1/sample1.jpg \\
      --template inputs/sample1/template.json \\
      --output outputs/visualization

  # Visualize with specific processors only
  python -m src.utils.visualization_runner \\
      --input inputs/sample1/sample1.jpg \\
      --template inputs/sample1/template.json \\
      --capture-processors "AutoRotate,CropOnMarkers,ReadOMR" \\
      --output outputs/visualization

  # Customize image quality and size
  python -m src.utils.visualization_runner \\
      --input inputs/sample1/sample1.jpg \\
      --template inputs/sample1/template.json \\
      --max-width 1200 \\
      --quality 95 \\
      --no-colored \\
      --output outputs/visualization
        """,
    )

    parser.add_argument(
        "-i", "--input", required=True, type=Path, help="Path to input OMR image file"
    )

    parser.add_argument(
        "-t", "--template", required=True, type=Path, help="Path to template JSON file"
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to config JSON file (optional)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs/visualization"),
        help="Output directory for visualization (default: outputs/visualization)",
    )

    parser.add_argument(
        "--capture-processors",
        type=str,
        default="all",
        help="Comma-separated list of processor names to capture (default: all)",
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=800,
        help="Maximum width for captured images in pixels (default: 800)",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        choices=range(1, 101),
        metavar="[1-100]",
        help="JPEG quality for captured images (default: 85)",
    )

    parser.add_argument(
        "--no-colored",
        action="store_true",
        help="Do not capture colored images (only grayscale)",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open visualization in browser automatically",
    )

    parser.add_argument(
        "--no-json", action="store_true", help="Do not export JSON data file"
    )

    parser.add_argument(
        "--title", type=str, default=None, help="Custom title for the visualization"
    )

    return parser.parse_args()


def validate_paths(args):
    """Validate input file paths."""
    errors = []

    if not args.input.exists():
        errors.append(f"Input file not found: {args.input}")

    if not args.template.exists():
        errors.append(f"Template file not found: {args.template}")

    if args.config and not args.config.exists():
        errors.append(f"Config file not found: {args.config}")

    if errors:
        for error in errors:
            logger.error(error)
        sys.exit(1)


def parse_capture_processors(capture_str: str) -> list[str]:
    """Parse comma-separated processor names.

    Args:
        capture_str: Comma-separated processor names or "all"

    Returns:
        List of processor names
    """
    if capture_str.lower() == "all":
        return ["all"]

    # Split by comma and strip whitespace
    processors = [p.strip() for p in capture_str.split(",")]
    return [p for p in processors if p]  # Filter empty strings


def main():
    """Main entry point for the CLI tool."""
    args = parse_args()

    # Validate paths
    validate_paths(args)

    # Parse capture processors
    capture_processors = parse_capture_processors(args.capture_processors)

    logger.info("=" * 70)
    logger.info("OMR Workflow Visualization Tool")
    logger.info("=" * 70)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Template: {args.template}")
    logger.info(f"Config: {args.config or 'Default'}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Capturing processors: {', '.join(capture_processors)}")
    logger.info(f"Image settings: max_width={args.max_width}, quality={args.quality}")
    logger.info(f"Include colored: {not args.no_colored}")
    logger.info("=" * 70)

    try:
        # Track workflow
        logger.info("Starting workflow tracking...")
        session = track_workflow(
            file_path=args.input,
            template_path=args.template,
            config_path=args.config,
            capture_processors=capture_processors,
            max_image_width=args.max_width,
            include_colored=not args.no_colored,
            image_quality=args.quality,
        )
        logger.info(f"✓ Tracked {len(session.processor_states)} processor states")

        # Export to HTML
        logger.info("Generating visualization...")
        html_path = export_to_html(
            session=session,
            output_dir=args.output,
            title=args.title,
            open_browser=not args.no_browser,
            export_json=not args.no_json,
        )
        logger.info(f"✓ Visualization saved to: {html_path}")

        logger.info("=" * 70)
        logger.info("SUCCESS! Workflow visualization completed.")
        logger.info(f"Open the file in your browser: file://{html_path.absolute()}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"ERROR: Workflow visualization failed: {e}")
        logger.error("=" * 70)
        import traceback

        traceback.print_exc()
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
