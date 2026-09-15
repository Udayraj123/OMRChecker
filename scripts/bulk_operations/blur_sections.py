"""
Bulk Image Section Blur Utility for OMRChecker
Author: Palak
Addresses part of issue #213 (bulk operation scripts for optimising outputs)

Blurs one or more rectangular regions (e.g. student name/signature boxes) across
a folder of images, so sheets can be safely shared/archived without exposing PII.

SUPPORTED FORMATS:
    - PNG (.png)
    - JPEG (.jpg, .jpeg)
    - TIFF (.tiff, .tif)

USAGE:
    # Preview only, no files written (uses default --path=inputs)
    python scripts/bulk_operations/blur_sections.py --box 50,20,200,60 --dry-run

    # Blur two regions, recursively, real run, writes to inputs/blurred/
    python scripts/bulk_operations/blur_sections.py \
        --box 50,20,200,60 --box 50,100,200,40 --recursive

    # In-place mode: overwrites originals, tracks processed files in a
    # manifest so reruns don't double-blur an already-blurred file
    python scripts/bulk_operations/blur_sections.py \
        --path ./inputs --box 50,20,200,60 --inplace

Boxes are given as "x,y,w,h" in pixels, relative to each image's own top-left
corner. Coordinates are clamped to image bounds rather than erroring, since a
single bad box shouldn't crash a bulk run over many images.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif")
DEFAULT_PATH = Path("inputs")
DEFAULT_OUTPUT_SUBDIR = "blurred"
DEFAULT_STRENGTH = 21
MANIFEST_FILENAME = ".blur_manifest.json"


# ==============================================================================
# Logging helpers (mirrors the style used in prepare_images.py from PR #267)
# ==============================================================================


def log_status(status: str, message: str) -> None:
    """Print formatted status message to stdout."""
    print(f"[{status}] {message}")


def log_section(title: str) -> None:
    """Print a section header."""
    print(f"\n=== {title} ===")


def log_summary(processed: int, success: int, skipped: int, errors: int) -> None:
    """Print operation summary statistics."""
    log_section("Summary")
    print(f"Processed: {processed} files")
    print(f"Blurred:   {success} files")
    print(f"Skipped:   {skipped} files")
    print(f"Errors:    {errors} files")


# ==============================================================================
# Box parsing
# ==============================================================================


def parse_box(box_str: str) -> Tuple[int, int, int, int]:
    """Parse a "x,y,w,h" string into a tuple of ints.

    Raises:
        ValueError: if the string isn't exactly 4 comma-separated integers,
            or if width/height are not positive.
    """
    parts = box_str.split(",")
    if len(parts) != 4:
        raise ValueError(
            f"Box must be 'x,y,w,h' (4 values), got {len(parts)}: '{box_str}'"
        )
    try:
        x, y, w, h = (int(p.strip()) for p in parts)
    except ValueError as e:
        raise ValueError(f"Box values must be integers: '{box_str}'") from e

    if w <= 0 or h <= 0:
        raise ValueError(f"Box width/height must be positive: '{box_str}'")
    if x < 0 or y < 0:
        raise ValueError(f"Box x/y must be non-negative: '{box_str}'")

    return x, y, w, h


def clamp_box_to_image(
    box: Tuple[int, int, int, int], img_width: int, img_height: int
) -> Optional[Tuple[int, int, int, int]]:
    """Clamp a box to fit within the image bounds.

    Returns None if the box falls entirely outside the image (nothing to blur).
    """
    x, y, w, h = box

    if x >= img_width or y >= img_height:
        return None

    clamped_w = min(w, img_width - x)
    clamped_h = min(h, img_height - y)

    if clamped_w <= 0 or clamped_h <= 0:
        return None

    return x, y, clamped_w, clamped_h


# ==============================================================================
# Manifest handling (for --inplace rerun tracking)
# ==============================================================================


def load_manifest(manifest_path: Path) -> Dict[str, dict]:
    """Load the processed-files manifest, or return empty dict if absent/corrupt."""
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log_status(
            "ERROR",
            f"Manifest at {manifest_path} unreadable ({type(e).__name__}), "
            f"treating as empty",
        )
        return {}


def save_manifest(manifest_path: Path, manifest: Dict[str, dict]) -> None:
    """Persist the processed-files manifest to disk."""
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    except OSError as e:
        log_status("ERROR", f"Could not write manifest: {type(e).__name__}: {e}")


# ==============================================================================
# Core blur logic
# ==============================================================================


def blur_region(
    image, box: Tuple[int, int, int, int], strength: int
):
    """Apply a Gaussian blur to a rectangular region of an image (in place).

    Args:
        image: OpenCV image (numpy array), modified in place.
        box: (x, y, w, h) region to blur, already clamped to image bounds.
        strength: Blur kernel size (odd, positive). Higher = blurrier.

    Returns:
        The same image object, for chaining.
    """
    x, y, w, h = box

    # Kernel size must be odd and >= 1 for GaussianBlur.
    k = strength if strength % 2 == 1 else strength + 1
    k = max(k, 1)

    region = image[y : y + h, x : x + w]
    if region.size == 0:
        return image

    blurred = cv2.GaussianBlur(region, (k, k), 0)
    image[y : y + h, x : x + w] = blurred
    return image


def find_image_files(directory: Path, recursive: bool) -> List[Path]:
    """Find all supported image files under directory."""
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def blur_files(
    directory: Path,
    boxes: List[Tuple[int, int, int, int]],
    strength: int,
    output_dir: Optional[Path],
    recursive: bool,
    dry_run: bool,
    inplace: bool,
) -> None:
    """Blur the given regions across all images in directory.

    In --inplace mode, originals are overwritten and a manifest file tracks
    which files were already processed, so reruns skip them instead of
    blurring an already-blurred file a second time.
    """
    log_section("Bulk Blur Operation")
    print(f"Path: {directory}")
    print(f"Regions: {boxes}")
    print(f"Strength: {strength}")
    print(f"Mode: {'in-place' if inplace else 'separate output dir'}")
    if not inplace:
        print(f"Output: {output_dir}")
    print(f"Recursive: {recursive}")
    if dry_run:
        print("Mode: DRY-RUN (preview only, no changes made)")

    image_files = find_image_files(directory, recursive)

    if not image_files:
        log_status("INFO", "No image files found in specified directory")
        return

    manifest_path = directory / MANIFEST_FILENAME
    manifest = load_manifest(manifest_path) if inplace else {}

    if not dry_run and not inplace:
        output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    success = 0
    skipped = 0
    errors = 0

    for file_path in image_files:
        processed += 1
        relative_name = file_path.name
        # Key manifest entries by path relative to the scanned directory, so
        # it stays stable across --recursive subfolders.
        manifest_key = str(file_path.relative_to(directory))

        if inplace and manifest_key in manifest:
            log_status(
                "SKIP", f"{relative_name}: already processed (in manifest)"
            )
            skipped += 1
            continue

        try:
            image = cv2.imread(str(file_path))
            if image is None:
                log_status("ERROR", f"{relative_name}: unreadable image file")
                errors += 1
                continue

            img_h, img_w = image.shape[:2]

            applied_any = False
            for box in boxes:
                clamped = clamp_box_to_image(box, img_w, img_h)
                if clamped is None:
                    log_status(
                        "SKIP",
                        f"{relative_name}: box {box} outside image bounds "
                        f"({img_w}x{img_h})",
                    )
                    continue
                blur_region(image, clamped, strength)
                applied_any = True

            if not applied_any:
                log_status("SKIP", f"{relative_name}: no valid regions to blur")
                skipped += 1
                continue

            if inplace:
                out_path = file_path
            else:
                out_path = output_dir / relative_name
                if out_path.exists():
                    log_status(
                        "ERROR",
                        f"{relative_name}: output already exists at {out_path}, "
                        f"not overwriting",
                    )
                    errors += 1
                    continue

            if not dry_run:
                cv2.imwrite(str(out_path), image)
                if inplace:
                    manifest[manifest_key] = {
                        "boxes": boxes,
                        "strength": strength,
                    }

            log_status("BLURRED", f"{relative_name} -> {out_path}")
            success += 1

        except (OSError, cv2.error) as e:
            log_status("ERROR", f"{relative_name}: {type(e).__name__}: {e}")
            errors += 1

    if inplace and not dry_run:
        save_manifest(manifest_path, manifest)

    log_summary(processed, success, skipped, errors)


# ==============================================================================
# CLI
# ==============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bulk-blur rectangular sections of images (e.g. names, "
        "signatures) before they're shared or archived.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help=f"Directory of input images (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--box",
        action="append",
        required=True,
        dest="boxes",
        metavar="x,y,w,h",
        help="Region to blur, in pixels. Repeatable for multiple regions.",
    )
    parser.add_argument(
        "--strength",
        type=int,
        default=DEFAULT_STRENGTH,
        help=f"Gaussian blur kernel size (odd number, default: {DEFAULT_STRENGTH}). "
        "Higher = blurrier/less reversible.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Where blurred images are written (default: <path>/{DEFAULT_OUTPUT_SUBDIR}). "
        "Ignored if --inplace is set.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite original files instead of writing to a separate folder. "
        f"Tracks processed files in <path>/{MANIFEST_FILENAME} so reruns "
        "don't blur an already-blurred file again.",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recurse into subdirectories"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would happen without writing any files",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.path.is_dir():
        log_status("ERROR", f"Not a directory: {args.path}")
        return 1

    if args.inplace and args.output_dir is not None:
        log_status("ERROR", "--output-dir is not allowed together with --inplace")
        return 1

    try:
        boxes = [parse_box(b) for b in args.boxes]
    except ValueError as e:
        log_status("ERROR", str(e))
        return 1

    output_dir = None if args.inplace else (args.output_dir or (args.path / DEFAULT_OUTPUT_SUBDIR))

    try:
        blur_files(
            directory=args.path,
            boxes=boxes,
            strength=args.strength,
            output_dir=output_dir,
            recursive=args.recursive,
            dry_run=args.dry_run,
            inplace=args.inplace,
        )
    except Exception as e:  # noqa: BLE001 - top-level CLI guard, logged with type
        log_status("ERROR", f"Unexpected {type(e).__name__} during run: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())