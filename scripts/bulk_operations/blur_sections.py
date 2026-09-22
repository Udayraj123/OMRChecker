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

EXIT CODE: non-zero if any file failed to process (unreadable, write failure,
output collision, etc.) so this is safe to use in automation/CI.
"""

import argparse
import hashlib
import json
import os
import sys
import uuid
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
# Box / strength parsing and validation
# ==============================================================================


def parse_box(box_str: str) -> Tuple[int, int, int, int]:
    """Parse a "x,y,w,h" string into a tuple of ints."""
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


def validate_strength(strength: int) -> None:
    """Reject non-positive blur strength.

    A strength <= 0 collapses to a 1-pixel (or invalid) blur kernel, which
    leaves the region visually unchanged while still being logged/counted
    as successfully blurred -- silently defeating the whole point of the
    tool. Fail loudly instead.
    """
    if strength <= 0:
        raise ValueError(f"--strength must be a positive integer, got {strength}")


def clamp_box_to_image(
    box: Tuple[int, int, int, int], img_width: int, img_height: int
) -> Optional[Tuple[int, int, int, int]]:
    """Clamp a box to fit within the image bounds. Returns None if entirely outside."""
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


def compute_config_hash(boxes: List[Tuple[int, int, int, int]], strength: int) -> str:
    """Hash the blur parameters, so a manifest entry only counts as 'done'
    for the exact region(s)/strength that were actually applied. Reruns with
    a new/changed region are reprocessed instead of silently skipped.
    """
    payload = json.dumps({"boxes": sorted(boxes), "strength": strength}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest(manifest_path: Path) -> Dict[str, dict]:
    """Load the processed-files manifest.

    Raises on a corrupt/unreadable manifest rather than silently treating it
    as empty -- an empty-on-failure fallback would cause --inplace to
    re-blur every already-completed file, stacking blur on blur.
    """
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(
            f"Manifest at {manifest_path} is corrupt or unreadable "
            f"({type(e).__name__}: {e}). Refusing to run in --inplace mode, "
            f"since that could re-blur already-completed files. Fix or "
            f"remove the manifest and retry."
        ) from e


def save_manifest(manifest_path: Path, manifest: Dict[str, dict]) -> None:
    """Persist the processed-files manifest to disk.

    Raises on write failure instead of swallowing it -- a manifest that
    silently fails to save would let --inplace runs continue untracked,
    causing later reruns to re-blur files whose "done" status was lost.
    """
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    except OSError as e:
        raise RuntimeError(
            f"Could not write manifest at {manifest_path}: {type(e).__name__}: {e}"
        ) from e


# ==============================================================================
# Core blur logic
# ==============================================================================


def blur_region(image, box: Tuple[int, int, int, int], strength: int):
    """Apply a Gaussian blur to a rectangular region of an image (in place)."""
    if strength <= 0:
        raise ValueError(f"Blur strength must be positive, got {strength}")

    x, y, w, h = box
    k = strength if strength % 2 == 1 else strength + 1

    region = image[y : y + h, x : x + w]
    if region.size == 0:
        return image

    blurred = cv2.GaussianBlur(region, (k, k), 0)
    image[y : y + h, x : x + w] = blurred
    return image


def write_image_atomic(out_path: Path, image) -> bool:
    """Write `image` to `out_path` without ever leaving a partially-written
    or corrupted file at that path.

    Encodes to a temporary sibling file first, verifies the write actually
    succeeded, and only then atomically replaces the destination. This
    matters most for --inplace mode: writing straight over the original and
    checking cv2.imwrite's result *after the fact* still risks leaving a
    truncated/corrupt original if the write fails partway through (e.g.
    disk full). Writing to a temp file means the original is only ever
    touched by a single atomic os.replace(), which either fully succeeds or
    leaves the original completely untouched.

    Returns True on success. On any failure, the temp file is cleaned up
    and the original destination (if it existed) is left untouched.
    """
    # Keep the real extension (.jpg/.png/etc.) so cv2.imwrite can infer the
    # correct codec -- it dispatches purely off the filename suffix.
    tmp_path = out_path.with_name(
        f".{out_path.stem}.{uuid.uuid4().hex}{out_path.suffix}"
    )
    try:
        write_ok = cv2.imwrite(str(tmp_path), image)
        if not write_ok or not tmp_path.exists() or tmp_path.stat().st_size == 0:
            return False
        os.replace(str(tmp_path), str(out_path))  # atomic on same filesystem
        return True
    except (OSError, cv2.error):
        return False
    finally:
        # If anything above failed before the replace, remove the leftover
        # temp file so it doesn't linger in the output directory.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def find_image_files(
    directory: Path, recursive: bool, output_dir: Optional[Path] = None
) -> List[Path]:
    """Find all supported image files under directory.

    - Excludes anything inside output_dir, so a recursive rerun doesn't
      rediscover previously-generated output and try to reprocess it.
    - Resolves symlinks and skips any file whose real target falls outside
      `directory`, so --inplace can't be tricked into overwriting a file
      elsewhere on disk via a symlink planted in the input tree.
    """
    pattern = "**/*" if recursive else "*"
    directory_resolved = directory.resolve()
    output_dir_resolved = output_dir.resolve() if output_dir else None

    files = []
    for p in sorted(directory.glob(pattern)):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if output_dir_resolved is not None:
            try:
                p.resolve().relative_to(output_dir_resolved)
                continue  # lives inside the output dir -- skip it
            except ValueError:
                pass  # not inside output_dir, keep going

        resolved = p.resolve()
        try:
            resolved.relative_to(directory_resolved)
        except ValueError:
            log_status(
                "SKIP",
                f"{p.name}: symlink target resolves outside the scanned "
                f"directory, refusing to touch it",
            )
            continue

        files.append(p)

    return files


def blur_files(
    directory: Path,
    boxes: List[Tuple[int, int, int, int]],
    strength: int,
    output_dir: Optional[Path],
    recursive: bool,
    dry_run: bool,
    inplace: bool,
) -> Tuple[int, int, int, int]:
    """Blur the given regions across all images in directory.

    In --inplace mode, originals are overwritten and a manifest file tracks
    which files (and with which exact parameters) were already processed,
    so reruns skip only genuinely-completed work.

    Returns (processed, success, skipped, errors).
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

    scan_output_dir = None if inplace else output_dir
    image_files = find_image_files(directory, recursive, output_dir=scan_output_dir)

    if not image_files:
        log_status("INFO", "No image files found in specified directory")
        return 0, 0, 0, 0

    manifest_path = directory / MANIFEST_FILENAME
    manifest = load_manifest(manifest_path) if inplace else {}
    current_hash = compute_config_hash(boxes, strength)

    if not dry_run and not inplace:
        output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    success = 0
    skipped = 0
    errors = 0

    for file_path in image_files:
        processed += 1
        relative_path = file_path.relative_to(directory)
        relative_name = str(relative_path)
        manifest_key = relative_path.as_posix()

        if inplace:
            existing_entry = manifest.get(manifest_key)
            if existing_entry is not None and existing_entry.get("hash") == current_hash:
                log_status(
                    "SKIP", f"{relative_name}: already processed with these exact "
                    f"parameters (in manifest)"
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
                # Preserve subdirectory structure so same-named files in
                # different subfolders don't collide on a single flat name.
                out_path = output_dir / relative_path
                if out_path.exists():
                    log_status(
                        "ERROR",
                        f"{relative_name}: output already exists at {out_path}, "
                        f"not overwriting",
                    )
                    errors += 1
                    continue
                if not dry_run:
                    out_path.parent.mkdir(parents=True, exist_ok=True)

            if dry_run:
                log_status("BLURRED", f"{relative_name} -> {out_path} (dry-run)")
                success += 1
                continue

            write_ok = write_image_atomic(out_path, image)
            if not write_ok:
                log_status(
                    "ERROR",
                    f"{relative_name}: write failed, output not written "
                    f"(original left untouched -- write went to a temp file "
                    f"first, so nothing was overwritten before verifying "
                    f"success)",
                )
                errors += 1
                continue

            if inplace:
                manifest[manifest_key] = {
                    "boxes": boxes,
                    "strength": strength,
                    "hash": current_hash,
                }
                # Save after every successful write, not just once at the end,
                # so an interrupted run doesn't lose track of files that were
                # already (successfully) blurred.
                save_manifest(manifest_path, manifest)

            log_status("BLURRED", f"{relative_name} -> {out_path}")
            success += 1

        except (OSError, cv2.error) as e:
            log_status("ERROR", f"{relative_name}: {type(e).__name__}: {e}")
            errors += 1

    log_summary(processed, success, skipped, errors)
    return processed, success, skipped, errors


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
        help=f"Gaussian blur kernel size (odd number, must be positive; "
        f"default: {DEFAULT_STRENGTH}). Higher = blurrier/less reversible.",
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
        f"Tracks processed files (and the exact parameters used) in "
        f"<path>/{MANIFEST_FILENAME} so reruns only skip genuinely-completed "
        "work.",
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
        validate_strength(args.strength)
        boxes = [parse_box(b) for b in args.boxes]
    except ValueError as e:
        log_status("ERROR", str(e))
        return 1

    output_dir = (
        None if args.inplace else (args.output_dir or (args.path / DEFAULT_OUTPUT_SUBDIR))
    )

    try:
        _, _, _, errors = blur_files(
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

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())