"""Auto-trainer for YOLO bubble detection models.

Handles automated model training from collected training data.
"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from src.utils.logger import logger


class AutoTrainer:
    """Handles automated model training from collected data.

    Uses YOLO (ultralytics) for bubble detection training.
    """

    def __init__(
        self,
        training_data_dir: str | Path = "outputs/training_data",
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
    ) -> None:
        """Initialize the auto-trainer.

        Args:
            training_data_dir: Directory containing training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            image_size: Image size for YOLO training
        """
        self.training_data_dir = Path(training_data_dir)
        self.dataset_dir = self.training_data_dir / "dataset"
        self.models_dir = Path("outputs/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size

    def train_from_collected_data(self) -> dict:
        """Train YOLO model from collected training data.

        Returns:
            Dictionary with training results and metrics
        """
        # Check if dataset exists
        data_yaml = self.dataset_dir / "data.yaml"
        if not data_yaml.exists():
            msg = (
                f"Training data not found at {data_yaml}. "
                f"Please run data collection first with --collect-training-data"
            )
            raise FileNotFoundError(msg)

        logger.info("=" * 60)
        logger.info("Starting YOLO Auto-Training")
        logger.info("=" * 60)
        logger.info(f"Dataset: {self.dataset_dir}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Image size: {self.image_size}")

        try:
            # Import YOLO (lazy import to avoid dependency if not using ML features)
            from ultralytics import YOLO

            # Initialize pretrained model (YOLOv8 nano for speed)
            logger.info("Loading pretrained YOLOv8n model...")
            model = YOLO("yolov8n.pt")

            # Train on collected data
            logger.info("Starting training...")
            results = model.train(
                data=str(data_yaml),
                epochs=self.epochs,
                imgsz=self.image_size,
                batch=self.batch_size,
                device="cpu",  # Use CPU by default (GPU auto-detected if available)
                project=str(self.models_dir),
                name="bubble_detector",
                exist_ok=True,
                patience=20,  # Early stopping patience
                save=True,
                plots=True,
                verbose=True,
            )

            logger.info("Training complete!")

            # Validate the model
            logger.info("Validating trained model...")
            metrics = model.val()

            # Export model and metadata
            trained_model_path = self._export_trained_model(model, metrics, results)

            # Generate summary
            summary = self._generate_training_summary(
                metrics, results, trained_model_path
            )

            logger.info("=" * 60)
            logger.info("Training Summary")
            logger.info("=" * 60)
            logger.info(f"Model saved to: {trained_model_path}")
            logger.info(f"Precision: {summary['metrics']['precision']:.3f}")
            logger.info(f"Recall: {summary['metrics']['recall']:.3f}")
            logger.info(f"mAP50: {summary['metrics']['map50']:.3f}")
            logger.info(f"mAP50-95: {summary['metrics']['map50_95']:.3f}")
            logger.info("=" * 60)

        except ImportError as e:
            msg = "ML dependencies not installed. Run: uv sync --extra ml"
            logger.error(
                "ultralytics package not found. Install ML dependencies with: uv sync --extra ml"
            )
            raise ImportError(msg) from e
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        else:
            return summary

    def _export_trained_model(self, _model, metrics) -> Path:
        """Export trained model with metadata.

        Args:
            model: Trained YOLO model
            metrics: Validation metrics
            results: Training results

        Returns:
            Path to exported model
        """
        # Find the best weights
        project_dir = self.models_dir / "bubble_detector"
        best_weights = project_dir / "weights" / "best.pt"

        if not best_weights.exists():
            msg = f"Best weights not found at {best_weights}"
            raise FileNotFoundError(msg)

        # Create timestamped copy
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        model_name = f"bubble_detector_{timestamp}.pt"
        exported_path = self.models_dir / model_name

        shutil.copy2(best_weights, exported_path)

        # Also keep as "best.pt" for convenience
        best_path = self.models_dir / "best.pt"
        shutil.copy2(best_weights, best_path)

        # Save metadata
        metadata = {
            "training_date": timestamp,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            "model_path": str(exported_path),
            "metrics": {
                "precision": float(metrics.box.p[0])
                if hasattr(metrics.box, "p")
                else 0.0,
                "recall": float(metrics.box.r[0]) if hasattr(metrics.box, "r") else 0.0,
                "map50": float(metrics.box.map50)
                if hasattr(metrics.box, "map50")
                else 0.0,
                "map50_95": float(metrics.box.map)
                if hasattr(metrics.box, "map")
                else 0.0,
            },
        }

        metadata_path = self.models_dir / f"bubble_detector_{timestamp}_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        logger.info(f"Model exported to: {exported_path}")
        logger.info(f"Metadata saved to: {metadata_path}")

        return best_path

    def _generate_training_summary(self, metrics, _results, model_path: Path) -> dict:
        """Generate training summary.

        Args:
            metrics: Validation metrics
            results: Training results
            model_path: Path to saved model

        Returns:
            Dictionary with summary information
        """
        summary = {
            "status": "success",
            "model_path": str(model_path),
            "metrics": {
                "precision": float(metrics.box.p[0])
                if hasattr(metrics.box, "p")
                else 0.0,
                "recall": float(metrics.box.r[0]) if hasattr(metrics.box, "r") else 0.0,
                "map50": float(metrics.box.map50)
                if hasattr(metrics.box, "map50")
                else 0.0,
                "map50_95": float(metrics.box.map)
                if hasattr(metrics.box, "map")
                else 0.0,
            },
            "training_config": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "image_size": self.image_size,
            },
        }

        # Save summary to file
        summary_path = self.models_dir / "training_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        return summary

    def resume_training(self, checkpoint_path: str | Path) -> dict:
        """Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint model

        Returns:
            Dictionary with training results
        """
        logger.info(f"Resuming training from: {checkpoint_path}")

        try:
            from ultralytics import YOLO

            # Load checkpoint
            model = YOLO(str(checkpoint_path))

            # Continue training
            data_yaml = self.dataset_dir / "data.yaml"
            _results = model.train(  # Results unused, can be logged in future
                data=str(data_yaml),
                epochs=self.epochs,
                imgsz=self.image_size,
                batch=self.batch_size,
                device="cpu",
                project=str(self.models_dir),
                name="bubble_detector_resumed",
                exist_ok=True,
                resume=True,
            )

            metrics = model.val()
            trained_model_path = self._export_trained_model(model, metrics)
            return self._generate_training_summary(metrics, None, trained_model_path)

        except Exception as e:
            logger.error(f"Resume training failed: {e}")
            raise

    def train_field_block_detector(
        self,
        dataset_path: Path,
        epochs: int | None = None,
    ) -> tuple[Path, dict]:
        """Train Stage 1: Field Block Detector.

        Args:
            dataset_path: Path to field block dataset directory
            epochs: Number of training epochs (defaults to self.epochs)

        Returns:
            Tuple of (model_path, metrics_dict)
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            msg = "ML dependencies not installed. Run: uv sync --extra ml"
            logger.error(
                "ultralytics package not found. Install ML dependencies with: uv sync --extra ml"
            )
            raise ImportError(msg) from e

        logger.info("=" * 60)
        logger.info("Training Field Block Detector (Stage 1)")
        logger.info("=" * 60)

        # Use YOLOv8 medium (more capacity for field block detection)
        model = YOLO("yolov8m.pt")

        data_yaml = dataset_path / "data.yaml"
        if not data_yaml.exists():
            msg = f"Dataset configuration not found: {data_yaml}"
            raise FileNotFoundError(msg)

        # Train
        model.train(
            data=str(data_yaml),
            epochs=epochs or self.epochs,
            imgsz=1024,  # Larger for full OMR sheet
            batch=8,  # Smaller batch for larger images
            device="cpu",
            project=str(self.models_dir),
            name="field_block_detector",
            exist_ok=True,
            patience=20,
            save=True,
            plots=True,
            verbose=True,
        )

        logger.info("Field block detector training complete!")

        # Validate
        metrics = model.val()

        # Export best model
        project_dir = self.models_dir / "field_block_detector"
        best_weights = project_dir / "weights" / "best.pt"

        if not best_weights.exists():
            msg = f"Best weights not found at {best_weights}"
            raise FileNotFoundError(msg)

        # Create timestamped copy
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        model_name = f"field_block_detector_{timestamp}.pt"
        exported_path = self.models_dir / model_name

        shutil.copy2(best_weights, exported_path)

        logger.info(f"Field block model saved to: {exported_path}")

        return exported_path, metrics.results_dict

    def train_bubble_detector(
        self,
        dataset_path: Path,
        epochs: int | None = None,
    ) -> tuple[Path, dict]:
        """Train Stage 2: Bubble Detector.

        Args:
            dataset_path: Path to bubble dataset directory
            epochs: Number of training epochs (defaults to self.epochs)

        Returns:
            Tuple of (model_path, metrics_dict)
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            msg = "ML dependencies not installed. Run: uv sync --extra ml"
            logger.error(
                "ultralytics package not found. Install ML dependencies with: uv sync --extra ml"
            )
            raise ImportError(msg) from e

        logger.info("=" * 60)
        logger.info("Training Bubble Detector (Stage 2)")
        logger.info("=" * 60)

        # Use YOLOv8 nano (faster, bubbles are simpler)
        model = YOLO("yolov8n.pt")

        data_yaml = dataset_path / "data.yaml"
        if not data_yaml.exists():
            msg = f"Dataset configuration not found: {data_yaml}"
            raise FileNotFoundError(msg)

        # Train
        model.train(
            data=str(data_yaml),
            epochs=epochs or self.epochs,
            imgsz=640,  # Smaller for cropped blocks
            batch=16,
            device="cpu",
            project=str(self.models_dir),
            name="bubble_detector",
            exist_ok=True,
            patience=20,
            save=True,
            plots=True,
            verbose=True,
        )

        logger.info("Bubble detector training complete!")

        # Validate
        metrics = model.val()

        # Export best model
        project_dir = self.models_dir / "bubble_detector"
        best_weights = project_dir / "weights" / "best.pt"

        if not best_weights.exists():
            msg = f"Best weights not found at {best_weights}"
            raise FileNotFoundError(msg)

        # Create timestamped copy
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        model_name = f"bubble_detector_{timestamp}.pt"
        exported_path = self.models_dir / model_name

        shutil.copy2(best_weights, exported_path)

        logger.info(f"Bubble model saved to: {exported_path}")

        return exported_path, metrics.results_dict

    def train_hierarchical_pipeline(
        self,
        field_block_dataset: Path,
        bubble_dataset: Path,
    ) -> dict:
        """Train both stages sequentially (hierarchical pipeline).

        Args:
            field_block_dataset: Path to field block dataset
            bubble_dataset: Path to bubble dataset

        Returns:
            Dictionary with paths and metrics for both models
        """
        logger.info("=" * 60)
        logger.info("Starting Two-Stage Hierarchical Training")
        logger.info("=" * 60)

        # Stage 1: Field Blocks
        logger.info("\n🔷 STAGE 1: Field Block Detection")
        fb_model_path, fb_metrics = self.train_field_block_detector(field_block_dataset)

        # Stage 2: Bubbles
        logger.info("\n🔷 STAGE 2: Bubble Detection")
        bubble_model_path, bubble_metrics = self.train_bubble_detector(bubble_dataset)

        logger.info("\n" + "=" * 60)
        logger.info("✅ Hierarchical Training Complete!")
        logger.info("=" * 60)

        results = {
            "field_block_model": str(fb_model_path),
            "field_block_metrics": fb_metrics,
            "bubble_model": str(bubble_model_path),
            "bubble_metrics": bubble_metrics,
        }

        # Save combined summary
        summary_path = self.models_dir / "hierarchical_training_summary.json"
        summary_path.write_text(json.dumps(results, indent=2))

        logger.info(f"Training summary saved to: {summary_path}")

        return results
