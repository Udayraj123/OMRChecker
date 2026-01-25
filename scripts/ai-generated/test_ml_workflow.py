"""Test workflow for two-stage hierarchical YOLO detection.

This script demonstrates the complete workflow:
1. Collect training data (field blocks + bubbles)
2. Train both Stage 1 and Stage 2 models
3. Test performance and generate metrics
"""
# ruff: noqa: S607

import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import AutoTrainer
from src.utils.logger import logger


def collect_training_data() -> bool:
    """Step 1: Collect training data from sample OMR sheets."""
    logger.info("=" * 60)
    logger.info("STEP 1: Collecting Training Data")
    logger.info("=" * 60)

    # Use a good quality sample for training
    sample_path = "samples/2-omr-marker"

    # Collect field block data
    logger.info("Collecting field block data...")
    result = subprocess.run(  # noqa: S603
        [
            "uv",
            "run",
            "python",
            "main.py",
            "-i",
            f"{sample_path}/ScanBatch1",
            "-o",
            "outputs/ml_workflow_test",
            "--collect-field-block-data",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Field block collection failed: {result.stderr}")
        return False

    # Collect bubble data
    logger.info("Collecting bubble data...")
    result = subprocess.run(  # noqa: S603
        [
            "uv",
            "run",
            "python",
            "main.py",
            "-i",
            f"{sample_path}/ScanBatch1",
            "-o",
            "outputs/ml_workflow_test",
            "--collect-training-data",
            "--confidence-threshold",
            "0.85",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Bubble collection failed: {result.stderr}")
        return False

    logger.info("✅ Training data collection complete!")
    return True


def train_models() -> tuple[bool, bool]:
    """Step 2: Train both YOLO models."""
    logger.info("=" * 60)
    logger.info("STEP 2: Training YOLO Models")
    logger.info("=" * 60)

    # Check if we have enough data
    field_block_images = Path("outputs/training_data/field_blocks/images")
    bubble_images = Path("outputs/training_data/bubbles/images")

    if not field_block_images.exists() or not list(field_block_images.glob("*.jpg")):
        logger.warning(
            "No field block training data found. Skipping field block training."
        )
        field_block_trained = False
    else:
        logger.info(
            f"Found {len(list(field_block_images.glob('*.jpg')))} field block samples"
        )

        # Train field block detector (Stage 1)
        logger.info("Training Stage 1: Field Block Detector...")
        trainer = AutoTrainer(
            training_data_dir="outputs/training_data",
            epochs=50,  # Reduced for quick testing
            batch_size=8,
            image_size=1024,  # Larger for field blocks
        )

        try:
            trainer.train_field_block_model()
            field_block_trained = True
            logger.info("✅ Field block model trained!")
        except Exception as e:
            logger.error(f"Field block training failed: {e}")
            field_block_trained = False

    if not bubble_images.exists() or not list(bubble_images.glob("*.jpg")):
        logger.warning("No bubble training data found. Skipping bubble training.")
        bubble_trained = False
    else:
        logger.info(f"Found {len(list(bubble_images.glob('*.jpg')))} bubble samples")

        # Train bubble detector (Stage 2)
        logger.info("Training Stage 2: Bubble Detector...")
        trainer = AutoTrainer(
            training_data_dir="outputs/training_data",
            epochs=50,  # Reduced for quick testing
            batch_size=16,
            image_size=640,
        )

        try:
            trainer.train_bubble_model()
            bubble_trained = True
            logger.info("✅ Bubble model trained!")
        except Exception as e:
            logger.error(f"Bubble training failed: {e}")
            bubble_trained = False

    return field_block_trained, bubble_trained


def benchmark_performance(use_field_blocks: bool, use_bubbles: bool) -> dict:
    """Step 3: Test model performance on validation set.

    Args:
        use_field_blocks: Whether to use field block detection model
        use_bubbles: Whether to use bubble detection model

    Returns:
        Dictionary of benchmark results
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Testing Model Performance")
    logger.info("=" * 60)

    test_sample = "samples/2-omr-marker/ScanBatch2"

    # Find trained models
    models_dir = Path("outputs/models")
    field_block_model = None
    bubble_model = None

    if use_field_blocks:
        field_block_models = sorted(models_dir.glob("field_block_detector_*.pt"))
        if field_block_models:
            field_block_model = field_block_models[-1]  # Get latest
            logger.info(f"Using field block model: {field_block_model.name}")

    if use_bubbles:
        bubble_models = sorted(models_dir.glob("bubble_detector_*.pt"))
        if bubble_models:
            bubble_model = bubble_models[-1]  # Get latest
            logger.info(f"Using bubble model: {bubble_model.name}")

    # Test configurations
    test_configs = _prepare_test_configs(field_block_model, bubble_model)

    # Run benchmarks
    results = _run_benchmark_tests(test_configs, test_sample)

    # Generate report
    _print_comparison_report(results)

    return results


def _prepare_test_configs(field_block_model, bubble_model) -> list[dict]:
    """Prepare test configurations based on available models."""
    test_configs = [
        {
            "name": "Traditional Only (Baseline)",
            "args": [],
        },
    ]

    if field_block_model and bubble_model:
        test_configs.append(
            {
                "name": "Two-Stage ML (Field Blocks + Bubbles)",
                "args": [
                    "--use-field-block-detection",
                    "--field-block-model",
                    str(field_block_model),
                    "--use-ml-fallback",
                    str(bubble_model),
                    "--fusion-strategy",
                    "confidence_weighted",
                ],
            }
        )
    elif bubble_model:
        test_configs.append(
            {
                "name": "Single-Stage ML (Bubbles Only)",
                "args": [
                    "--use-ml-fallback",
                    str(bubble_model),
                    "--fusion-strategy",
                    "confidence_weighted",
                ],
            }
        )

    return test_configs


def _run_benchmark_tests(test_configs: list[dict], test_sample: str) -> dict:
    """Run benchmark tests for each configuration."""
    results = {}

    for config in test_configs:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing: {config['name']}")
        logger.info(f"{'=' * 60}")

        start_time = time.time()

        cmd = [
            "uv",
            "run",
            "python",
            "main.py",
            "-i",
            test_sample,
            "-o",
            f"outputs/ml_test_{config['name'].replace(' ', '_')}",
        ] + config["args"]

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)  # noqa: S603

        elapsed = time.time() - start_time

        if result.returncode == 0:
            results[config["name"]] = {
                "success": True,
                "time": elapsed,
                "output": result.stdout,
            }
            logger.info(f"✅ {config['name']} completed in {elapsed:.2f}s")
        else:
            results[config["name"]] = {
                "success": False,
                "time": elapsed,
                "error": result.stderr,
            }
            logger.error(f"❌ {config['name']} failed: {result.stderr}")

    return results


def _print_comparison_report(results: dict) -> None:
    """Print comparison report of benchmark results."""
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE COMPARISON REPORT")
    logger.info("=" * 60)

    for name, result in results.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Status: {'✅ Success' if result['success'] else '❌ Failed'}")
        logger.info(f"  Time: {result['time']:.2f}s")

        if result["success"]:
            # Extract metrics from output
            output = result["output"]
            if "Processing Speed" in output:
                for line in output.split("\n"):
                    if "Processing Rate" in line or "Processing Speed" in line:
                        logger.info(f"  {line.strip()}")


def main() -> int:
    """Run the complete ML workflow test."""
    logger.info("🚀 Starting Two-Stage Hierarchical YOLO Testing Workflow")
    logger.info("=" * 60)

    # Step 1: Collect training data
    if not collect_training_data():
        logger.error("Failed to collect training data. Exiting.")
        return 1

    # Step 2: Train models
    field_block_trained, bubble_trained = train_models()

    if not field_block_trained and not bubble_trained:
        logger.error("No models were trained. Exiting.")
        return 1

    # Step 3: Test performance
    _results = benchmark_performance(
        use_field_blocks=field_block_trained,
        use_bubbles=bubble_trained,
    )

    logger.info("\n" + "=" * 60)
    logger.info("✅ ML Workflow Test Complete!")
    logger.info("=" * 60)
    logger.info("\nModels saved in: outputs/models/")
    logger.info("Training data in: outputs/training_data/")
    logger.info("Test results in: outputs/ml_test_*/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
