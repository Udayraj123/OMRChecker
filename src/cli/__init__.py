"""CLI interface for OMRChecker."""

from pathlib import Path

from src.entry import entry_point

__all__ = ["run_cli"]


def run_cli(args: dict) -> None:
    """Run the CLI with the provided arguments.

    Args:
        args: Dictionary containing CLI arguments including:
            - input_paths: List of input directory paths
            - output_dir: Output directory path
            - debug: Debug mode flag
            - setLayout: Layout mode flag
            - outputMode: Output mode (default, csv, etc.)
            - mode: Processing mode
    """
    # Get the input paths from args
    input_paths = args.get("input_paths", [])

    if not input_paths:
        msg = "No input paths provided"
        raise ValueError(msg)

    # Process each input path
    for input_path in input_paths:
        input_dir = Path(input_path)
        entry_point(input_dir, args)
