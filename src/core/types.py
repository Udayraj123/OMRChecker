"""Shared type definitions for OMRChecker core library.

These types provide a clean interface between CLI and core processing logic,
and will be mirrored in the TypeScript port.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProcessorConfig:
    """Configuration for OMRProcessor.

    This mirrors the args dict but with a cleaner, typed interface.
    """

    # Input/Output paths
    input_dir: Path | None = None
    output_dir: Path = Path("outputs")

    # Processing modes
    debug: bool = False
    output_mode: str = "default"
    set_layout: bool = False

    # ML-related settings (future support)
    ml_model_path: str | None = None
    field_block_model_path: str | None = None
    use_field_block_detection: bool = False
    enable_shift_detection: bool = False
    fusion_strategy: str = "confidence_weighted"

    # Training-related settings
    collect_training_data: bool = False
    confidence_threshold: float = 0.85
    training_data_dir: str = "outputs/training_data"

    # Execution mode
    mode: str = "process"  # process, auto-train, test-model, export-yolo
    epochs: int = 100

    @classmethod
    def from_args(cls, args: dict) -> "ProcessorConfig":
        """Create ProcessorConfig from CLI args dict."""
        return cls(
            input_dir=Path(args["input_paths"][0]) if args.get("input_paths") else None,
            output_dir=Path(args.get("output_dir", "outputs")),
            debug=args.get("debug", False),
            output_mode=args.get("outputMode", "default"),
            set_layout=args.get("setLayout", False),
            ml_model_path=args.get("ml_model_path"),
            field_block_model_path=args.get("field_block_model_path"),
            use_field_block_detection=args.get("use_field_block_detection", False),
            enable_shift_detection=args.get("enable_shift_detection", False),
            fusion_strategy=args.get("fusion_strategy", "confidence_weighted"),
            collect_training_data=args.get("collect_training_data", False),
            confidence_threshold=args.get("confidence_threshold", 0.85),
            training_data_dir=args.get("training_data_dir", "outputs/training_data"),
            mode=args.get("mode", "process"),
            epochs=args.get("epochs", 100),
        )

    def to_args(self) -> dict:
        """Convert back to args dict for backward compatibility."""
        return {
            "input_paths": [str(self.input_dir)] if self.input_dir else [],
            "output_dir": str(self.output_dir),
            "debug": self.debug,
            "outputMode": self.output_mode,
            "setLayout": self.set_layout,
            "ml_model_path": self.ml_model_path,
            "field_block_model_path": self.field_block_model_path,
            "use_field_block_detection": self.use_field_block_detection,
            "enable_shift_detection": self.enable_shift_detection,
            "fusion_strategy": self.fusion_strategy,
            "collect_training_data": self.collect_training_data,
            "confidence_threshold": self.confidence_threshold,
            "training_data_dir": self.training_data_dir,
            "mode": self.mode,
            "epochs": self.epochs,
        }


@dataclass
class OMRResult:
    """Result of processing a single OMR sheet.

    This provides a clean, typed return value instead of relying on side effects.
    """

    # File information
    file_name: str
    file_path: Path
    output_path: Path | None = None

    # Processing status
    status: str = "success"  # success, error, multi_marked
    error: str | None = None

    # OMR response
    omr_response: dict[str, str] = field(default_factory=dict)
    raw_omr_response: dict[str, Any] = field(default_factory=dict)

    # Evaluation results
    score: float = 0.0
    evaluation_meta: dict[str, Any] | None = None

    # Additional metadata
    is_multi_marked: bool = False
    processing_time: float = 0.0
    field_interpretations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_name": self.file_name,
            "file_path": str(self.file_path),
            "output_path": str(self.output_path) if self.output_path else None,
            "status": self.status,
            "error": self.error,
            "omr_response": self.omr_response,
            "raw_omr_response": self.raw_omr_response,
            "score": self.score,
            "evaluation_meta": self.evaluation_meta,
            "is_multi_marked": self.is_multi_marked,
            "processing_time": self.processing_time,
            "field_interpretations": self.field_interpretations,
        }


@dataclass
class DirectoryProcessingResult:
    """Result of processing a directory of OMR sheets."""

    total_files: int = 0
    successful: int = 0
    errors: int = 0
    multi_marked: int = 0
    results: list[OMRResult] = field(default_factory=list)
    processing_time: float = 0.0

    def add_result(self, result: OMRResult) -> None:
        """Add a single file result."""
        self.results.append(result)
        self.total_files += 1

        if result.status == "success":
            if result.is_multi_marked:
                self.multi_marked += 1
            else:
                self.successful += 1
        else:
            self.errors += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_files": self.total_files,
            "successful": self.successful,
            "errors": self.errors,
            "multi_marked": self.multi_marked,
            "processing_time": self.processing_time,
            "results": [r.to_dict() for r in self.results],
        }
