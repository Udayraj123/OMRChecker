"""Benchmark STN+YOLO vs YOLO-only performance.

Compares detection accuracy and inference time for field block detection
with and without STN preprocessing on a test dataset.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.logger import logger


def load_test_images(
    test_dir: Path, max_images: int = 10
) -> list[tuple[Path, np.ndarray]]:
    """Load test images from directory.

    Args:
        test_dir: Directory containing test images
        max_images: Maximum number of images to load

    Returns:
        List of (path, image) tuples
    """
    image_files = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))[
        :max_images
    ]

    images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append((img_path, img))

    logger.info(f"Loaded {len(images)} test images from {test_dir}")
    return images


def run_yolo_only(yolo_model, images: list[tuple[Path, np.ndarray]]) -> dict:
    """Run YOLO detection without STN.

    Args:
        yolo_model: YOLO model instance
        images: List of (path, image) tuples

    Returns:
        Dictionary with timing and detection results
    """
    logger.info("\n" + "=" * 60)
    logger.info("Running YOLO-Only Detection")
    logger.info("=" * 60)

    results = {
        "detections": [],
        "inference_times": [],
        "total_time": 0,
    }

    start_time = time.time()

    for img_path, image in images:
        img_start = time.time()

        # Run YOLO
        yolo_results = yolo_model.predict(image, conf=0.5, verbose=False, imgsz=1024)

        img_time = time.time() - img_start
        results["inference_times"].append(img_time)

        # Parse detections
        num_detections = 0
        if yolo_results and len(yolo_results) > 0:
            for result in yolo_results:
                if hasattr(result, "boxes") and result.boxes is not None:
                    num_detections = len(result.boxes)

        results["detections"].append(
            {
                "image": img_path.name,
                "num_boxes": num_detections,
                "time": img_time,
            }
        )

        logger.info(
            f"  {img_path.name}: {num_detections} boxes, {img_time * 1000:.1f}ms"
        )

    results["total_time"] = time.time() - start_time
    results["avg_time"] = np.mean(results["inference_times"])
    results["total_boxes"] = sum(d["num_boxes"] for d in results["detections"])

    logger.info(f"\nTotal time: {results['total_time']:.2f}s")
    logger.info(f"Average time per image: {results['avg_time'] * 1000:.1f}ms")
    logger.info(f"Total boxes detected: {results['total_boxes']}")

    return results


def run_stn_yolo(stn_model, yolo_model, images: list[tuple[Path, np.ndarray]]) -> dict:
    """Run YOLO detection with STN preprocessing.

    Args:
        stn_model: STN model instance
        yolo_model: YOLO model instance
        images: List of (path, image) tuples

    Returns:
        Dictionary with timing and detection results
    """
    logger.info("\n" + "=" * 60)
    logger.info("Running STN+YOLO Detection")
    logger.info("=" * 60)

    from src.processors.detection.models.stn_utils import apply_stn_to_image

    results = {
        "detections": [],
        "inference_times": [],
        "stn_times": [],
        "yolo_times": [],
        "total_time": 0,
    }

    start_time = time.time()

    for img_path, image in images:
        img_start = time.time()

        # Apply STN
        stn_start = time.time()
        transformed = apply_stn_to_image(stn_model, image, device="cpu")
        stn_time = time.time() - stn_start
        results["stn_times"].append(stn_time)

        # Run YOLO
        yolo_start = time.time()
        yolo_results = yolo_model.predict(
            transformed, conf=0.5, verbose=False, imgsz=1024
        )
        yolo_time = time.time() - yolo_start
        results["yolo_times"].append(yolo_time)

        img_time = time.time() - img_start
        results["inference_times"].append(img_time)

        # Parse detections
        num_detections = 0
        if yolo_results and len(yolo_results) > 0:
            for result in yolo_results:
                if hasattr(result, "boxes") and result.boxes is not None:
                    num_detections = len(result.boxes)

        results["detections"].append(
            {
                "image": img_path.name,
                "num_boxes": num_detections,
                "time": img_time,
                "stn_time": stn_time,
                "yolo_time": yolo_time,
            }
        )

        logger.info(
            f"  {img_path.name}: {num_detections} boxes, "
            f"{img_time * 1000:.1f}ms (STN: {stn_time * 1000:.1f}ms, YOLO: {yolo_time * 1000:.1f}ms)"
        )

    results["total_time"] = time.time() - start_time
    results["avg_time"] = np.mean(results["inference_times"])
    results["avg_stn_time"] = np.mean(results["stn_times"])
    results["avg_yolo_time"] = np.mean(results["yolo_times"])
    results["total_boxes"] = sum(d["num_boxes"] for d in results["detections"])

    logger.info(f"\nTotal time: {results['total_time']:.2f}s")
    logger.info(f"Average time per image: {results['avg_time'] * 1000:.1f}ms")
    logger.info(f"  STN overhead: {results['avg_stn_time'] * 1000:.1f}ms")
    logger.info(f"  YOLO time: {results['avg_yolo_time'] * 1000:.1f}ms")
    logger.info(f"Total boxes detected: {results['total_boxes']}")

    return results


def compare_results(yolo_results: dict, stn_yolo_results: dict) -> None:
    """Compare and print comparison of both approaches.

    Args:
        yolo_results: Results from YOLO-only
        stn_yolo_results: Results from STN+YOLO
    """
    logger.info("\n" + "=" * 60)
    logger.info("Performance Comparison")
    logger.info("=" * 60)

    # Timing comparison
    yolo_avg_ms = yolo_results["avg_time"] * 1000
    stn_yolo_avg_ms = stn_yolo_results["avg_time"] * 1000
    overhead_ms = stn_yolo_avg_ms - yolo_avg_ms
    overhead_pct = (overhead_ms / yolo_avg_ms) * 100

    logger.info("\n📊 Inference Time:")
    logger.info(f"  YOLO-only:     {yolo_avg_ms:.1f}ms per image")
    logger.info(f"  STN+YOLO:      {stn_yolo_avg_ms:.1f}ms per image")
    logger.info(f"  STN overhead:  {overhead_ms:.1f}ms (+{overhead_pct:.1f}%)")
    logger.info(f"    (STN alone:  {stn_yolo_results['avg_stn_time'] * 1000:.1f}ms)")

    # Detection count comparison
    yolo_boxes = yolo_results["total_boxes"]
    stn_yolo_boxes = stn_yolo_results["total_boxes"]
    box_diff = stn_yolo_boxes - yolo_boxes

    logger.info("\n📦 Detections:")
    logger.info(f"  YOLO-only:  {yolo_boxes} total boxes")
    logger.info(f"  STN+YOLO:   {stn_yolo_boxes} total boxes")

    if box_diff > 0:
        logger.info(f"  Difference: +{box_diff} boxes (STN found more)")
    elif box_diff < 0:
        logger.info(f"  Difference: {box_diff} boxes (STN found fewer)")
    else:
        logger.info("  Difference: 0 boxes (same detections)")

    # Per-image comparison
    logger.info("\n📸 Per-Image Comparison:")
    logger.info(f"{'Image':<30} {'YOLO':<10} {'STN+YOLO':<10} {'Diff':<8}")
    logger.info("-" * 60)

    for yolo_det, stn_det in zip(
        yolo_results["detections"], stn_yolo_results["detections"], strict=False
    ):
        img_name = yolo_det["image"]
        yolo_count = yolo_det["num_boxes"]
        stn_count = stn_det["num_boxes"]
        diff = stn_count - yolo_count
        diff_str = f"+{diff}" if diff > 0 else str(diff)

        logger.info(f"{img_name:<30} {yolo_count:<10} {stn_count:<10} {diff_str:<8}")

    # Summary
    logger.info("\n💡 Summary:")
    if abs(box_diff) <= 1:
        logger.info("  ✅ STN produces similar detection counts (within ±1)")
    elif box_diff > 0:
        logger.info(f"  ⚠️  STN detects {box_diff} more boxes - may improve recall")
    else:
        logger.info(
            f"  ⚠️  STN detects {abs(box_diff)} fewer boxes - may reduce false positives"
        )

    if overhead_pct < 20:
        logger.info(f"  ✅ STN overhead is acceptable ({overhead_pct:.1f}%)")
    elif overhead_pct < 50:
        logger.info(f"  ⚠️  STN adds moderate overhead ({overhead_pct:.1f}%)")
    else:
        logger.info(f"  ❌ STN adds significant overhead ({overhead_pct:.1f}%)")


def _find_yolo_model(args, models_dir: Path) -> Path | None:
    """Find YOLO model path.

    Args:
        args: Parsed command line arguments
        models_dir: Directory containing models

    Returns:
        Path to YOLO model or None if not found
    """
    if args.yolo_model:
        return Path(args.yolo_model)

    yolo_models = sorted(models_dir.glob("field_block_detector_*.pt"))
    if not yolo_models:
        logger.error("No YOLO model found!")
        return None
    return yolo_models[-1]


def _find_stn_model(args, models_dir: Path) -> Path | None:
    """Find STN model path.

    Args:
        args: Parsed command line arguments
        models_dir: Directory containing models

    Returns:
        Path to STN model or None if not found
    """
    if args.stn_model:
        return Path(args.stn_model)

    stn_models = sorted(models_dir.glob("stn_refinement_*.pt"))
    if not stn_models:
        logger.warning("\nNo STN model found - skipping STN+YOLO benchmark")
        logger.warning("Train an STN model first with: python train_stn_yolo.py")
        return None
    return stn_models[-1]


def _load_models(
    yolo_model_path: Path, stn_model_path: Path | None
) -> tuple[Any, Any] | tuple[None, None]:
    """Load YOLO and optional STN models.

    Args:
        yolo_model_path: Path to YOLO model
        stn_model_path: Path to STN model (optional)

    Returns:
        Tuple of (yolo_model, stn_model) or (None, None) on failure
    """
    # Load YOLO
    try:
        from ultralytics import YOLO

        yolo_model = YOLO(str(yolo_model_path))
        yolo_model.eval()
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return None, None

    # Load STN if path provided
    stn_model = None
    if stn_model_path and stn_model_path.exists():
        try:
            from src.processors.detection.models.stn_utils import (
                load_stn_model,
            )

            stn_model = load_stn_model(
                stn_model_path, input_channels=1, input_size=(1024, 1024), device="cpu"
            )
            logger.info(f"\nUsing STN model: {stn_model_path.name}")
        except Exception as e:
            logger.error(f"Failed to load STN model: {e}")
            return yolo_model, None
    elif stn_model_path:
        logger.warning(f"\nSTN model not found: {stn_model_path}")
        logger.warning("Skipping STN+YOLO benchmark")

    return yolo_model, stn_model


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description="Benchmark STN+YOLO vs YOLO-only")
    parser.add_argument(
        "--test-images",
        type=str,
        default="samples/2-omr-marker/ScanBatch2/inputs",
        help="Directory with test images",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        help="Path to YOLO model (auto-detect latest if not specified)",
    )
    parser.add_argument(
        "--stn-model",
        type=str,
        help="Path to STN model (auto-detect latest if not specified)",
    )
    parser.add_argument(
        "--max-images", type=int, default=10, help="Maximum number of test images"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("STN+YOLO Benchmarking Tool")
    logger.info("=" * 80)

    # Load test images
    test_dir = Path(args.test_images)
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return 1

    images = load_test_images(test_dir, max_images=args.max_images)
    if not images:
        logger.error("No images loaded!")
        return 1

    # Find models
    models_dir = Path("outputs/models")
    yolo_model_path = _find_yolo_model(args, models_dir)
    if yolo_model_path is None:
        return 1

    logger.info(f"Using YOLO model: {yolo_model_path.name}")

    stn_model_path = _find_stn_model(args, models_dir)

    # Load models
    yolo_model, stn_model = _load_models(yolo_model_path, stn_model_path)
    if yolo_model is None:
        return 1

    # Run YOLO-only benchmark
    yolo_results = run_yolo_only(yolo_model, images)

    # Run STN+YOLO benchmark if STN model loaded
    if stn_model is not None:
        stn_yolo_results = run_stn_yolo(stn_model, yolo_model, images)
        compare_results(yolo_results, stn_yolo_results)
    else:
        logger.info("\nSkipping STN+YOLO comparison (no STN model available)")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Benchmarking Complete!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
