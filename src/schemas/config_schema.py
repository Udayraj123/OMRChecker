from src.schemas.constants import load_common_defs
from src.utils.constants import OUTPUT_MODES, SUPPORTED_PROCESSOR_NAMES

CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/Udayraj123/OMRChecker/tree/master/src/schemas/config-schema.json",
    "$def": {
        # The common definitions go here
        **load_common_defs(["two_positive_integers"]),
    },
    "title": "Config Schema",
    "description": "OMRChecker config schema for custom tuning",
    "type": "object",
    "allOf": [
        {
            "additionalProperties": False,
            "properties": {
                "path": {
                    "description": "The path to the config file",
                    "type": "string",
                },
                "thresholding": {
                    "description": "The values used in the core algorithm of OMRChecker",
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        # TODO: rename all of these variables for better usability
                        "MIN_GAP_TWO_BUBBLES": {
                            "description": "Minimum difference between all mean values of the bubbles. Used for local thresholding of 2 or 1 bubbles",
                            "type": "integer",
                            "minimum": 10,
                            "maximum": 100,
                        },
                        "MIN_JUMP": {
                            "description": "Minimum difference between consecutive elements to be consider as a jump in a sorted array of mean values of the bubbles",
                            "type": "integer",
                            "minimum": 10,
                            "maximum": 100,
                        },
                        "MIN_JUMP_STD": {
                            "description": "The MIN_JUMP for the standard deviation plot",
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK": {
                            "description": "This value is added to jump value, only under-confident bubbles will use a fallback threshold",
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 20,
                        },
                        "GLOBAL_THRESHOLD_MARGIN": {
                            "description": 'This value determines if the calculated global threshold is "too close" to lower bubbles in confidence metrics ',
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 20,
                        },
                        "JUMP_DELTA": {
                            "description": "Note: JUMP_DELTA is deprecated, used only in plots currently to determine a stricter threshold",
                            "type": "integer",
                            "minimum": 10,
                            "maximum": 100,
                        },
                        "JUMP_DELTA_STD": {
                            "description": "JUMP_DELTA_STD is the minimum delta to be considered as a jump in the std plot",
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY": {
                            "description": "This value is added to jump value to distinguish safe detections vs underconfident detections",
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "GLOBAL_PAGE_THRESHOLD": {
                            "description": "This option decides the starting value to use before applying local outlier threshold",
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 255,
                        },
                        "GLOBAL_PAGE_THRESHOLD_STD": {
                            "description": "This option decides the starting value to use for standard deviation threshold which determines outliers",
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 60,
                        },
                        "GAMMA_LOW": {
                            "description": "Used in the CropOnDotLines processor to create a darker image for enhanced line detection (darker boxes)",
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                },
                "outputs": {
                    "description": "The configuration related to the outputs generated by OMRChecker",
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "output_mode": {
                            "type": "string",
                            "enum": [*list(OUTPUT_MODES.values())],
                            "description": "The output mode for the OMRChecker. Supported: default, moderation, setLayout",
                        },
                        "display_image_dimensions": {
                            "$ref": "#/$def/two_positive_integers",
                            "description": "The dimensions (width, height) for images displayed during the execution",
                        },
                        "show_logs_by_type": {
                            "description": "The toggles for enabling logs per level",
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "critical": {"type": "boolean"},
                                "error": {"type": "boolean"},
                                "warning": {"type": "boolean"},
                                "info": {"type": "boolean"},
                                "debug": {"type": "boolean"},
                            },
                        },
                        "show_image_level": {
                            "description": "The toggle level for showing debug images (higher means more debug images)",
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 6,
                        },
                        "save_image_level": {
                            "description": "The toggle level for saving debug images (higher means more debug images)",
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 6,
                        },
                        "colored_outputs_enabled": {
                            "description": "This option shows colored outputs while taking a small toll on the processing speeds",
                            "type": "boolean",
                        },
                        "save_detections": {
                            "description": "This option saves the detection outputs while taking a small toll on the processing speeds",
                            "type": "boolean",
                        },
                        # TODO: merge into one object
                        "save_image_metrics": {
                            "description": "This option exports the confidence metrics etc related to the images. These can be later used for deeper analysis/visualizations",
                            "type": "boolean",
                        },
                        "show_confidence_metrics": {
                            "description": "The toggle for enabling confidence metrics calculation",
                            "type": "boolean",
                        },
                        "filter_out_multimarked_files": {
                            "description": "This option moves files having multi-marked responses into a separate folder for manual checking, skipping evaluation",
                            "type": "boolean",
                        },
                        "file_grouping": {
                            "description": "Configuration for organizing processed files with dynamic patterns",
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "description": "Enable automatic file organization",
                                    "type": "boolean",
                                    "default": False,
                                },
                                "default_pattern": {
                                    "description": "Default pattern for files not matching any rule (use {original_name} for original filename)",
                                    "type": "string",
                                    "default": "ungrouped/{original_name}",
                                },
                                "rules": {
                                    "description": "List of grouping rules (evaluated by priority)",
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": [
                                            "name",
                                            "priority",
                                            "destination_pattern",
                                            "matcher",
                                        ],
                                        "properties": {
                                            "name": {
                                                "description": "Rule name for logging",
                                                "type": "string",
                                            },
                                            "priority": {
                                                "description": "Rule priority (lower = higher priority)",
                                                "type": "integer",
                                                "minimum": 0,
                                            },
                                            "destination_pattern": {
                                                "description": "Full path pattern with {field} placeholders (e.g., 'booklet_{code}/{roll}_{score}'). Extension auto-preserved if omitted.",
                                                "type": "string",
                                            },
                                            "action": {
                                                "description": "Action to perform (symlink or copy)",
                                                "enum": ["symlink", "copy"],
                                                "default": "symlink",
                                            },
                                            "collision_strategy": {
                                                "description": "How to handle filename collisions",
                                                "enum": [
                                                    "skip",
                                                    "increment",
                                                    "overwrite",
                                                ],
                                                "default": "skip",
                                            },
                                            "matcher": {
                                                "description": "Pattern matcher using format string + regex",
                                                "type": "object",
                                                "required": [
                                                    "formatString",
                                                    "matchRegex",
                                                ],
                                                "properties": {
                                                    "formatString": {
                                                        "description": "Format string with {field} placeholders",
                                                        "type": "string",
                                                    },
                                                    "matchRegex": {
                                                        "description": "Regex pattern to match formatted string",
                                                        "type": "string",
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                        "show_preprocessors_diff": {
                            "description": "This option shows a preview of the processed image for every preprocessor. Also granular at preprocessor level using a map",
                            "oneOf": [
                                {
                                    "type": "object",
                                    "patternProperties": {
                                        f"^({'|'.join(SUPPORTED_PROCESSOR_NAMES)})$": {
                                            "type": "boolean"
                                        }
                                    },
                                },
                                {"type": "boolean"},
                            ],
                        },
                    },
                },
                "processing": {
                    "description": "The configuration related to processing settings",
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "max_parallel_workers": {
                            "description": "Number of worker threads for parallel image processing per folder. Set to 1 for sequential processing. Automatically set to 1 when show_image_level > 0 (interactive mode)",
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 16,
                        },
                    },
                },
                "ml": {
                    "description": "The configuration related to ML-based detection and training",
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "enabled": {
                            "description": "Enable ML-based detection features",
                            "type": "boolean",
                        },
                        "model_path": {
                            "description": "Path to trained YOLO model for bubble detection",
                            "type": ["string", "null"],
                        },
                        "confidence_threshold": {
                            "description": "Minimum confidence threshold for ML predictions (0.0-1.0)",
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "use_for_low_confidence_only": {
                            "description": "Use ML only for low-confidence traditional detections",
                            "type": "boolean",
                        },
                        "collect_training_data": {
                            "description": "Collect high-confidence detections for ML training",
                            "type": "boolean",
                        },
                        "min_training_confidence": {
                            "description": "Minimum confidence score to include sample in training data (0.0-1.0)",
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "training_data_dir": {
                            "description": "Directory to store bubble training data",
                            "type": "string",
                        },
                        "model_output_dir": {
                            "description": "Directory to store trained models",
                            "type": "string",
                        },
                        "field_block_detection_enabled": {
                            "description": "Enable Stage 1 field block detection",
                            "type": "boolean",
                        },
                        "field_block_model_path": {
                            "description": "Path to trained YOLO model for field block detection",
                            "type": ["string", "null"],
                        },
                        "field_block_confidence_threshold": {
                            "description": "Minimum confidence threshold for field block detection (0.0-1.0)",
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "collect_field_block_data": {
                            "description": "Collect field block data for ML training",
                            "type": "boolean",
                        },
                        "field_block_dataset_dir": {
                            "description": "Directory to store field block training data",
                            "type": "string",
                        },
                        "bubble_dataset_dir": {
                            "description": "Directory to store bubble training data (alias for training_data_dir)",
                            "type": "string",
                        },
                        "shift_detection": {
                            "description": "Configuration for ML-based field block shift detection",
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "enabled": {
                                    "description": "Enable ML-based shift detection and application",
                                    "type": "boolean",
                                },
                                "global_max_shift_pixels": {
                                    "description": "Global maximum shift allowed in pixels for all field blocks",
                                    "type": "integer",
                                    "minimum": 0,
                                },
                                "per_block_max_shift_pixels": {
                                    "description": "Per-block maximum shift overrides (field_block_name: max_pixels)",
                                    "type": "object",
                                    "additionalProperties": {
                                        "type": "integer",
                                        "minimum": 0,
                                    },
                                },
                                "confidence_reduction_min": {
                                    "description": "Minimum confidence reduction when shift causes mismatch (0.0-1.0)",
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "confidence_reduction_max": {
                                    "description": "Maximum confidence reduction for severe mismatches (0.0-1.0)",
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "bubble_mismatch_threshold": {
                                    "description": "Flag mismatches if more than this many bubbles differ",
                                    "type": "integer",
                                    "minimum": 0,
                                },
                                "field_mismatch_threshold": {
                                    "description": "Flag mismatches if this many or more field responses differ",
                                    "type": "integer",
                                    "minimum": 0,
                                },
                            },
                        },
                        "fusion_enabled": {
                            "description": "Enable detection fusion between traditional and ML methods",
                            "type": "boolean",
                        },
                        "fusion_strategy": {
                            "description": "Strategy for fusing traditional and ML detections",
                            "type": "string",
                            "enum": [
                                "confidence_weighted",
                                "ml_fallback",
                                "traditional_primary",
                                "ml_only",
                            ],
                        },
                        "discrepancy_threshold": {
                            "description": "Threshold for flagging discrepancies between traditional and ML (0.0-1.0)",
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                },
                "visualization": {
                    "description": "Configuration for workflow visualization and debugging",
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "enabled": {
                            "description": "Enable workflow visualization",
                            "type": "boolean",
                        },
                        "capture_processors": {
                            "description": "List of processor names to capture or ['all'] for all processors",
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "capture_frequency": {
                            "description": "When to capture images: 'always' or 'on_change'",
                            "type": "string",
                            "enum": ["always", "on_change"],
                        },
                        "include_colored": {
                            "description": "Whether to capture colored images in addition to grayscale",
                            "type": "boolean",
                        },
                        "max_image_width": {
                            "description": "Maximum width for captured images in pixels",
                            "type": "integer",
                            "minimum": 100,
                            "maximum": 4000,
                        },
                        "embed_images": {
                            "description": "Whether to embed images in HTML (true) or reference externally (false)",
                            "type": "boolean",
                        },
                        "export_format": {
                            "description": "Format for exporting visualizations",
                            "type": "string",
                            "enum": ["html", "json"],
                        },
                        "output_dir": {
                            "description": "Directory to save visualizations",
                            "type": "string",
                        },
                        "auto_open_browser": {
                            "description": "Automatically open visualization in browser",
                            "type": "boolean",
                        },
                    },
                },
            },
        },
        {
            "if": {
                "properties": {
                    "outputs": {
                        "properties": {
                            "show_image_level": {
                                "type": "integer",
                                "exclusiveMinimum": 0,
                            },
                        },
                    },
                },
            },
            "then": {
                "properties": {
                    "processing": {
                        "properties": {
                            "max_parallel_workers": {
                                "const": 1,
                            },
                        },
                    },
                },
            },
        },
    ],
}
