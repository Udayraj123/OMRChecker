"""Typed dataclass models for configuration."""

import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from src.utils.logger import logger
from src.utils.serialization import dataclass_to_dict


@dataclass
class ThresholdingConfig:
    """Configuration for bubble thresholding algorithm."""

    GAMMA_LOW: float = 0.7
    MIN_GAP_TWO_BUBBLES: int = 30
    MIN_JUMP: int = 25
    CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY: int = 25
    MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK: int = 5
    GLOBAL_THRESHOLD_MARGIN: int = 10
    JUMP_DELTA: int = 30
    GLOBAL_PAGE_THRESHOLD: int = 200
    GLOBAL_PAGE_THRESHOLD_STD: int = 10
    MIN_JUMP_STD: int = 15
    JUMP_DELTA_STD: int = 5


@dataclass
class GroupingRule:
    """A single file grouping rule with dynamic path pattern."""

    name: str
    priority: int
    destination_pattern: str  # Full path pattern: "folder/{booklet}/roll_{roll}.jpg"
    matcher: dict  # { "formatString": "...", "matchRegex": "..." }
    action: str = "symlink"  # "symlink" or "copy"
    collision_strategy: str = "skip"  # "skip", "increment", or "overwrite"


@dataclass
class FileGroupingConfig:
    """Configuration for automatic file organization."""

    enabled: bool = False
    rules: list[GroupingRule] = field(default_factory=list)
    default_pattern: str = "ungrouped/{original_name}"  # Default for non-matching files

    # Always available fields in patterns (built-in)
    BUILTIN_FIELDS: ClassVar[set[str]] = {
        "file_path",
        "file_name",
        "file_stem",
        "original_name",
        "is_multi_marked",
    }

    # Fields that require evaluation to be enabled
    EVALUATION_FIELDS: ClassVar[set[str]] = {"score"}

    def validate(self, template=None, *, has_evaluation: bool = False) -> list[str]:
        """Validate the file grouping configuration.

        Args:
            template: Optional template object to check available OMR fields
            has_evaluation: Whether evaluation is enabled

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.enabled:
            return errors  # Skip validation if disabled

        # Validate default pattern
        pattern_errors = self._validate_pattern(
            self.default_pattern,
            "default_pattern",
            template,
            has_evaluation,
        )
        errors.extend(pattern_errors)

        # Validate each rule
        for i, rule in enumerate(self.rules, 1):
            rule_errors = self._validate_rule(rule, i, template, has_evaluation)
            errors.extend(rule_errors)

        # Check for duplicate priorities
        priorities = [rule.priority for rule in self.rules]
        if len(priorities) != len(set(priorities)):
            duplicates = [p for p in priorities if priorities.count(p) > 1]
            errors.append(
                f"Duplicate rule priorities found: {set(duplicates)}. "
                "Each rule should have a unique priority."
            )

        return errors

    def _validate_rule(
        self,
        rule: GroupingRule,
        rule_num: int,
        template,
        has_evaluation: bool,
    ) -> list[str]:
        """Validate a single grouping rule."""
        errors = []
        prefix = f"Rule #{rule_num} ('{rule.name}')"

        # Validate destination pattern
        pattern_errors = self._validate_pattern(
            rule.destination_pattern,
            f"{prefix} destination_pattern",
            template,
            has_evaluation,
        )
        errors.extend(pattern_errors)

        # Validate matcher format string
        matcher_format = rule.matcher.get("formatString", "")
        matcher_errors = self._validate_pattern(
            matcher_format,
            f"{prefix} matcher.formatString",
            template,
            has_evaluation,
            allow_empty=False,
        )
        errors.extend(matcher_errors)

        # Validate regex pattern
        try:
            re.compile(rule.matcher.get("matchRegex", ""))
        except re.error as e:
            errors.append(f"{prefix}: Invalid regex pattern in matcher.matchRegex: {e}")

        # Validate action
        if rule.action not in ("symlink", "copy"):
            errors.append(
                f"{prefix}: Invalid action '{rule.action}'. "
                "Must be 'symlink' or 'copy'."
            )

        # Validate collision strategy
        if rule.collision_strategy not in ("skip", "increment", "overwrite"):
            errors.append(
                f"{prefix}: Invalid collision_strategy '{rule.collision_strategy}'. "
                "Must be 'skip', 'increment', or 'overwrite'."
            )

        return errors

    def _validate_pattern(  # noqa: C901 - Complexity needed for thorough validation
        self,
        pattern: str,
        pattern_name: str,
        template,
        has_evaluation: bool,
        allow_empty: bool = True,
    ) -> list[str]:
        """Validate a pattern string for field availability."""
        errors = []

        if not pattern and not allow_empty:
            errors.append(f"{pattern_name}: Pattern cannot be empty")
            return errors

        if not pattern:
            return errors

        # Extract field names from pattern using string.Formatter
        try:
            formatter = string.Formatter()
            field_names = {
                field_name
                for _, field_name, _, _ in formatter.parse(pattern)
                if field_name
            }
        except (ValueError, KeyError) as e:
            errors.append(f"{pattern_name}: Invalid pattern syntax: {e}")
            return errors

        # Check each field
        for field_name in field_names:
            # Check if it's a built-in field
            if field_name in self.BUILTIN_FIELDS:
                continue

            # Check if it requires evaluation
            if field_name in self.EVALUATION_FIELDS:
                if not has_evaluation:
                    errors.append(
                        f"{pattern_name}: Field '{{{field_name}}}' requires evaluation.json "
                        f"to be present. Either add evaluation.json or remove this field from the pattern."
                    )
                continue

            # Check if it's an OMR field from template
            if template:
                # Get all fields from template
                template_fields = set()
                if hasattr(template, "all_fields"):
                    template_fields.update(template.all_fields)

                if field_name not in template_fields:
                    # Provide helpful error message
                    available = sorted(
                        self.BUILTIN_FIELDS | self.EVALUATION_FIELDS | template_fields
                    )
                    errors.append(
                        f"{pattern_name}: Field '{{{field_name}}}' not found in template. "
                        f"Available fields: {', '.join(f'{{{f}}}' for f in available)}"
                    )
            else:
                # No template available for validation - just warn
                logger.warning(
                    f"{pattern_name}: Cannot validate field '{{{field_name}}}' "
                    f"without template context"
                )

        return errors


@dataclass
class OutputsConfig:
    """Configuration for output behavior and visualization."""

    output_mode: str = "default"
    display_image_dimensions: list[int] = field(default_factory=lambda: [720, 1080])
    show_image_level: int = 0
    show_preprocessors_diff: dict[str, bool] = field(default_factory=dict)
    save_image_level: int = 1
    show_logs_by_type: dict[str, bool] = field(
        default_factory=lambda: {
            "critical": True,
            "error": True,
            "warning": True,
            "info": True,
            "debug": False,
        }
    )
    save_detections: bool = True
    colored_outputs_enabled: bool = False
    save_image_metrics: bool = False
    show_confidence_metrics: bool = False
    filter_out_multimarked_files: bool = False
    file_grouping: FileGroupingConfig = field(default_factory=FileGroupingConfig)


@dataclass
class ProcessingConfig:
    """Configuration for parallel processing."""

    max_parallel_workers: int = 1


@dataclass
class ShiftDetectionConfig:
    """Configuration for ML-based shift detection and application."""

    enabled: bool = False
    global_max_shift_pixels: int = 50  # Global limit for all blocks
    per_block_max_shift_pixels: dict[str, int] = field(
        default_factory=dict
    )  # Per-block overrides

    # Confidence adjustment on mismatch
    confidence_reduction_min: float = 0.1  # Min reduction (1 bubble diff)
    confidence_reduction_max: float = 0.5  # Max reduction (many diffs)

    # Comparison thresholds
    bubble_mismatch_threshold: int = 3  # Flag if >3 bubbles differ
    field_mismatch_threshold: int = 1  # Flag if any field response differs


@dataclass
class MLConfig:
    """Configuration for ML-based detection and training."""

    # General ML settings
    enabled: bool = False

    # Bubble detection (Stage 2) settings
    model_path: str | None = None
    confidence_threshold: float = 0.7
    use_for_low_confidence_only: bool = True

    # Field block detection (Stage 1) settings
    field_block_detection_enabled: bool = False
    field_block_model_path: str | None = None
    field_block_confidence_threshold: float = 0.75

    # Detection fusion settings
    fusion_enabled: bool = True
    fusion_strategy: str = "confidence_weighted"  # Options: confidence_weighted, ml_fallback, traditional_primary
    discrepancy_threshold: float = 0.3  # Flag if responses differ with high confidence

    # Shift detection settings
    shift_detection: ShiftDetectionConfig = field(default_factory=ShiftDetectionConfig)

    # Training data collection settings
    collect_training_data: bool = False
    min_training_confidence: float = 0.85
    training_data_dir: Path = Path("outputs/training_data")
    model_output_dir: Path = Path("outputs/models")

    # Hierarchical training settings
    collect_field_block_data: bool = True  # Collect both stages
    field_block_dataset_dir: Path = Path("outputs/training_data/field_blocks")
    bubble_dataset_dir: Path = Path("outputs/training_data/bubbles")


@dataclass
class VisualizationConfig:
    """Configuration for workflow visualization and debugging."""

    enabled: bool = False
    capture_processors: list[str] = field(default_factory=lambda: ["all"])
    capture_frequency: str = "on_change"  # Options: "always", "on_change"
    include_colored: bool = True
    max_image_width: int = 800
    embed_images: bool = True
    export_format: str = "html"  # Options: "html", "json"
    output_dir: Path = Path("outputs/visualization")
    auto_open_browser: bool = True


@dataclass
class Config:
    """Main configuration object for OMRChecker.

    This replaces the DotMap-based tuning_config throughout the codebase.
    """

    path: Path
    thresholding: ThresholdingConfig = field(default_factory=ThresholdingConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create Config from dictionary (typically from JSON).

        Args:
            data: Dictionary containing configuration data

        Returns:
            Config instance with nested dataclasses
        """
        outputs_data = data.get("outputs", {})

        # Parse file_grouping nested structure
        if "file_grouping" in outputs_data:
            grouping_data = outputs_data["file_grouping"]
            rules_data = grouping_data.get("rules", [])
            rules = [GroupingRule(**rule) for rule in rules_data]
            outputs_data["file_grouping"] = FileGroupingConfig(
                enabled=grouping_data.get("enabled", False),
                default_pattern=grouping_data.get(
                    "default_pattern", "ungrouped/{original_name}"
                ),
                rules=rules,
            )

        return cls(
            path=Path(data.get("path", "config.json")),
            thresholding=ThresholdingConfig(**data.get("thresholding", {})),
            outputs=OutputsConfig(**outputs_data),
            processing=ProcessingConfig(**data.get("processing", {})),
            ml=MLConfig(**data.get("ml", {})),
            visualization=VisualizationConfig(**data.get("visualization", {})),
        )

    def to_dict(self) -> dict:
        """Convert Config to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the config
        """
        return dataclass_to_dict(self)

    def get(self, key: str, default=None):
        """Get value by key (backwards compatibility).

        Args:
            key: The key to retrieve
            default: Default value if key not found

        Returns:
            The value if found, otherwise default
        """
        return getattr(self, key, default)
