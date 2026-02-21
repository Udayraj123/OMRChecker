#!/usr/bin/env python3
"""Bulk image preparation utility for OMRChecker.

This standalone utility helps prepare images before processing with OMRChecker:
  1. Bulk rename: Remove/replace non-UTF characters from filenames
  2. Bulk resize: Resize images based on file size threshold and dimensions

USAGE EXAMPLES:

  # Rename images, removing problematic characters from filenames:
  python prepare_images.py rename --path ./images

  # Rename recursively through subdirectories:
  python prepare_images.py rename --path ./images --recursive

  # Resize images larger than 500KB to fit 1920x1440 pixels:
  python prepare_images.py resize --path ./images \
    --max-size 500000 --max-width 1920 --max-height 1440
  # Note: 500000 bytes = ~500KB

  # Resize recursively:
  python prepare_images.py resize --path ./images \
    --max-size 500000 --max-width 1920 --max-height 1440 --recursive

SUPPORTED FORMATS:
  - PNG (.png)
  - JPEG (.jpg, .jpeg)
  - TIFF (.tiff, .tif)
  - PDF files are skipped with a message

REQUIREMENTS:
  - Python 3.7+
  - OpenCV (cv2)
  - NumPy

FEATURES:
  - Rename: Uses Unicode normalization to clean up filenames
  - Resize: Preserves aspect ratio, creates _resized copies
  - Recursive: Optional directory traversal
  - Detailed logging: Per-file status messages
  - Error handling: Skips problematic files with error messages
"""

import argparse
import os
import sys
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

# ============================================================================
# LOGGING UTILITIES
# ============================================================================


def log_status(status: str, message: str) -> None:
    """Print formatted status message to stdout.

    Args:
        status: Status type (RENAMED, SKIP, ERROR, RESIZED, INFO)
        message: Message content
    """
    print(f"[{status}] {message}")


def log_section(title: str) -> None:
    """Print a section header."""
    print(f"\n=== {title} ===")


def log_summary(processed: int, success: int, skipped: int, errors: int) -> None:
    """Print operation summary statistics."""
    log_section("Summary")
    print(f"Processed: {processed} files")
    print(f"Success: {success} files")
    print(f"Skipped: {skipped} files")
    print(f"Errors: {errors} files")


# ============================================================================
# FILE UTILITIES
# ============================================================================


def get_supported_image_extensions() -> Tuple[str, ...]:
    """Return supported image file extensions (case-insensitive)."""
    return (".png", ".jpg", ".jpeg", ".tiff", ".tif")


def find_image_files(directory: Path, recursive: bool = False) -> List[Path]:
    """Scan directory for image files.

    Args:
        directory: Directory path to scan
        recursive: If True, scan subdirectories recursively

    Returns:
        List of image file paths
    """
    if not directory.exists():
        return []

    image_files = []
    extensions = get_supported_image_extensions()

    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            image_files.append(file_path)

    return sorted(image_files)


# ============================================================================
# RENAME FUNCTIONALITY
# ============================================================================


def clean_filename(filename: str) -> str:
    """Remove/replace non-UTF characters from filename.

    Strategy:
    1. Separate name and extension
    2. Normalize unicode to NFD (decomposed form)
    3. Remove non-ASCII characters
    4. Replace spaces and problematic chars with underscores
    5. Recombine with original extension

    Args:
        filename: Original filename with extension

    Returns:
        Cleaned filename
    """
    name, ext = os.path.splitext(filename)

    # Normalize unicode: NFD decomposes characters (e.g., é → e + accent)
    normalized = unicodedata.normalize("NFD", name)

    # Remove non-ASCII bytes by encoding/decoding to ASCII
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")

    # Replace spaces and problematic characters
    cleaned = ascii_name.replace(" ", "_")

    # Remove any remaining special characters except underscores
    cleaned = "".join(c if c.isalnum() or c == "_" else "" for c in cleaned)

    # Remove consecutive underscores
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    # Strip leading/trailing underscores
    cleaned = cleaned.strip("_")

    # Fallback if cleaning removed all characters
    if not cleaned:
        cleaned = "unnamed_file"

    return cleaned + ext


def rename_files(
    directory: Path, recursive: bool = False, dry_run: bool = False
) -> None:
    """Bulk rename files, removing non-UTF characters from filenames.

    Args:
        directory: Directory to process
        recursive: If True, process subdirectories recursively
        dry_run: If True, preview changes without actually renaming
    """
    log_section("Bulk Rename Operation")
    print(f"Path: {directory}")
    print(f"Recursive: {recursive}")
    if dry_run:
        print("Mode: DRY-RUN (preview only, no changes made)")

    image_files = find_image_files(directory, recursive)

    if not image_files:
        log_status("INFO", "No image files found in specified directory")
        return

    processed = 0
    renamed = 0
    skipped = 0
    errors = 0

    for file_path in image_files:
        processed += 1
        original_name = file_path.name
        cleaned_name = clean_filename(original_name)

        # Skip if no changes needed
        if original_name == cleaned_name:
            log_status("SKIP", f"{original_name} (no changes needed)")
            skipped += 1
            continue

        try:
            new_path = file_path.parent / cleaned_name

            # Handle duplicate filenames (in case cleaned name already exists)
            if new_path.exists() and new_path != file_path:
                log_status(
                    "ERROR",
                    f"{original_name}: Target filename already exists "
                    f"({cleaned_name})",
                )
                errors += 1
                continue

            if not dry_run:
                file_path.rename(new_path)

            log_status("RENAMED", f"{original_name} → {cleaned_name}")
            renamed += 1

        except OSError as e:
            log_status("ERROR", f"{original_name}: {str(e)}")
            errors += 1

    log_summary(processed, renamed, skipped, errors)


# ============================================================================
# RESIZE FUNCTIONALITY
# ============================================================================


def get_image_dimensions(file_path: Path) -> Optional[Tuple[int, int]]:
    """Get image dimensions (width, height) using OpenCV.

    Returns None if file cannot be read.

    Args:
        file_path: Path to image file

    Returns:
        Tuple of (width, height) or None if unreadable
    """
    try:
        image = cv2.imread(str(file_path))
        if image is None:
            return None
        height, width = image.shape[:2]
        return width, height
    except Exception:
        return None


def read_and_get_dimensions(
    file_path: Path,
) -> Optional[Tuple]:
    """Read image and return (image_data, width, height).

    Returns None if file cannot be read.

    Args:
        file_path: Path to image file

    Returns:
        Tuple of (image, width, height) or None if unreadable
    """
    try:
        image = cv2.imread(str(file_path))
        if image is None:
            return None
        height, width = image.shape[:2]
        return image, width, height
    except Exception:
        return None


def calculate_resized_dimensions(
    original_width: int,
    original_height: int,
    max_width: int,
    max_height: int,
) -> Tuple[int, int]:
    """Calculate new dimensions preserving aspect ratio.

    Only downscales if original exceeds limits. Never upscales.

    Args:
        original_width: Original image width
        original_height: Original image height
        max_width: Maximum allowed width
        max_height: Maximum allowed height

    Returns:
        Tuple of (new_width, new_height)
    """
    # If already within limits, don't resize
    if original_width <= max_width and original_height <= max_height:
        return original_width, original_height

    # Calculate scale factors
    width_scale = max_width / original_width
    height_scale = max_height / original_height

    # Use the smaller scale to preserve aspect ratio
    scale = min(width_scale, height_scale)

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    return new_width, new_height


def resize_files(
    directory: Path,
    max_size_bytes: int,
    max_width: int,
    max_height: int,
    recursive: bool = False,
    dry_run: bool = False,
) -> None:
    """Bulk resize images exceeding file size threshold.

    Only processes files larger than max_size_bytes. Creates new files with
    _resized suffix, preserving originals.

    Args:
        directory: Directory to process
        max_size_bytes: File size threshold in bytes
        max_width: Maximum image width in pixels
        max_height: Maximum image height in pixels
        recursive: If True, process subdirectories recursively
        dry_run: If True, preview changes without actually resizing
    """
    log_section("Bulk Resize Operation")
    print(f"Path: {directory}")
    print(f"Recursive: {recursive}")
    print(f"Size threshold: {max_size_bytes / 1024:.1f} KB")
    print(f"Max dimensions: {max_width}x{max_height} pixels")
    if dry_run:
        print("Mode: DRY-RUN (preview only, no changes made)")

    image_files = find_image_files(directory, recursive)

    if not image_files:
        log_status("INFO", "No image files found in specified directory")
        return

    processed = 0
    resized = 0
    skipped = 0
    errors = 0

    for file_path in image_files:
        processed += 1
        original_name = file_path.name

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size <= max_size_bytes:
            size_kb = file_size / 1024
            threshold_kb = max_size_bytes / 1024
            log_status(
                "SKIP",
                f"{original_name} ({size_kb:.1f} KB < {threshold_kb:.1f} KB "
                "threshold)",
            )
            skipped += 1
            continue

        # Read image and get dimensions (single I/O operation)
        image_data = read_and_get_dimensions(file_path)
        if image_data is None:
            log_status(
                "ERROR",
                f"{original_name}: Unable to read image file",
            )
            errors += 1
            continue

        image, original_width, original_height = image_data

        # Calculate new dimensions
        new_width, new_height = calculate_resized_dimensions(
            original_width, original_height, max_width, max_height
        )

        # Skip if no resize needed
        if (new_width, new_height) == (original_width, original_height):
            log_status(
                "SKIP",
                f"{original_name} ({original_width}x{original_height} "
                "already within limits)",
            )
            skipped += 1
            continue

        try:
            # Resize using OpenCV (image already loaded)
            resized_image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

            # Create output filename with _resized suffix
            name, ext = os.path.splitext(original_name)
            resized_name = f"{name}_resized{ext}"
            resized_path = file_path.parent / resized_name

            original_kb = file_size / 1024

            if not dry_run:
                # Write resized image
                cv2.imwrite(str(resized_path), resized_image)
                new_file_size = os.path.getsize(resized_path)
                new_kb = new_file_size / 1024
            else:
                # Estimate file size in dry-run mode
                new_kb = None

            if new_kb is not None:
                log_status(
                    "RESIZED",
                    f"{original_name}: {original_width}x{original_height} → "
                    f"{new_width}x{new_height} ({original_kb:.1f} KB → "
                    f"{new_kb:.1f} KB)",
                )
            else:
                log_status(
                    "RESIZED",
                    f"{original_name}: {original_width}x{original_height} → "
                    f"{new_width}x{new_height} (~{original_kb:.1f} KB reduced)",
                )
            resized += 1

        except Exception as e:
            log_status("ERROR", f"{original_name}: {str(e)}")
            errors += 1

    log_summary(processed, resized, skipped, errors)


# ============================================================================
# CLI INTERFACE
# ============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare images for OMRChecker processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="operation", help="Operation to perform", required=True
    )

    # Rename subcommand
    rename_parser = subparsers.add_parser(
        "rename",
        help="Rename files removing non-UTF characters",
    )
    rename_parser.add_argument(
        "-p",
        "--path",
        type=Path,
        required=True,
        help="Input directory path",
    )
    rename_parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively process subdirectories",
    )
    rename_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually renaming",
    )

    # Resize subcommand
    resize_parser = subparsers.add_parser(
        "resize",
        help="Resize images exceeding size threshold",
    )
    resize_parser.add_argument(
        "-p",
        "--path",
        type=Path,
        required=True,
        help="Input directory path",
    )
    resize_parser.add_argument(
        "-s",
        "--max-size",
        type=int,
        required=True,
        help="Max file size threshold in bytes (e.g., 500000)",
    )
    resize_parser.add_argument(
        "-w",
        "--max-width",
        type=int,
        required=True,
        help="Max image width in pixels (e.g., 1920)",
    )
    resize_parser.add_argument(
        "-ht",
        "--max-height",
        type=int,
        required=True,
        help="Max image height in pixels (e.g., 1440)",
    )
    resize_parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively process subdirectories",
    )
    resize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually resizing",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Validate directory exists
        if not args.path.exists():
            log_status("ERROR", f"Directory not found: {args.path}")
            return 1

        if not args.path.is_dir():
            log_status("ERROR", f"Path is not a directory: {args.path}")
            return 1

        if args.operation == "rename":
            rename_files(args.path, args.recursive, args.dry_run)

        elif args.operation == "resize":
            # Validate resize arguments
            if args.max_size <= 0:
                log_status("ERROR", "--max-size must be greater than 0")
                return 1

            if args.max_width <= 0 or args.max_height <= 0:
                log_status(
                    "ERROR",
                    "--max-width and --max-height must be greater than 0",
                )
                return 1

            resize_files(
                args.path,
                args.max_size,
                args.max_width,
                args.max_height,
                args.recursive,
                args.dry_run,
            )

        return 0

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        return 1
    except Exception as e:
        log_status("ERROR", f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
