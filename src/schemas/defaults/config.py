from pathlib import Path

from src.schemas.models.config import (
    Config,
    FileGroupingConfig,
    MLConfig,
    OutputsConfig,
    ProcessingConfig,
    ShiftDetectionConfig,
    ThresholdingConfig,
)
from src.utils.constants import SUPPORTED_PROCESSOR_NAMES

# Create default config instance
CONFIG_DEFAULTS = Config(
    path=Path("config.json"),
    thresholding=ThresholdingConfig(
        gamma_low=0.7,
        min_gap_two_bubbles=30,
        min_jump=25,
        confident_jump_surplus_for_disparity=25,
        min_jump_surplus_for_global_fallback=5,
        global_threshold_margin=10,
        jump_delta=30,
        # Note: tune this value to avoid empty bubble detections
        global_page_threshold=200,
        global_page_threshold_std=10,
        min_jump_std=15,
        jump_delta_std=5,
    ),
    outputs=OutputsConfig(
        output_mode="default",
        display_image_dimensions=[720, 1080],
        show_image_level=0,
        show_preprocessors_diff=dict.fromkeys(SUPPORTED_PROCESSOR_NAMES, False),
        save_image_level=1,
        show_logs_by_type={
            "critical": True,
            "error": True,
            "warning": True,
            "info": True,
            "debug": False,
        },
        save_detections=True,
        colored_outputs_enabled=False,
        save_image_metrics=False,
        show_confidence_metrics=False,
        filter_out_multimarked_files=False,
        file_grouping=FileGroupingConfig(
            enabled=False,
            default_pattern="ungrouped/{original_name}",
            rules=[],
        ),
    ),
    processing=ProcessingConfig(
        max_parallel_workers=1,  # Number of worker threads for parallel processing (1 = sequential)
    ),
    ml=MLConfig(
        enabled=False,
        model_path=None,
        confidence_threshold=0.7,
        use_for_low_confidence_only=True,
        collect_training_data=False,
        min_training_confidence=0.85,
        shift_detection=ShiftDetectionConfig(
            enabled=False,
            global_max_shift_pixels=50,
            per_block_max_shift_pixels={},
            confidence_reduction_min=0.1,
            confidence_reduction_max=0.5,
            bubble_mismatch_threshold=3,
            field_mismatch_threshold=1,
        ),
    ),
)
