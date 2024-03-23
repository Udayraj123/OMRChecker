from src.schemas.constants import two_positive_integers

CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/Udayraj123/OMRChecker/tree/master/src/schemas/config-schema.json",
    "title": "Config Schema",
    "description": "OMRChecker config schema for custom tuning",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "dimensions": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "display_image_shape": two_positive_integers,
                "processing_image_shape": two_positive_integers,
            },
        },
        "threshold_params": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "GAMMA_LOW": {"type": "number", "minimum": 0, "maximum": 1},
                # TODO: rename these variables for better usability
                "MIN_GAP": {"type": "integer", "minimum": 10, "maximum": 100},
                "MIN_JUMP": {"type": "integer", "minimum": 10, "maximum": 100},
                "MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 20,
                },
                "GLOBAL_THRESHOLD_MARGIN": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 20,
                },
                "CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                },
                "JUMP_DELTA": {"type": "integer", "minimum": 10, "maximum": 100},
                "PAGE_TYPE_FOR_THRESHOLD": {
                    "enum": ["white", "black"],
                    "type": "string",
                },
                "GLOBAL_PAGE_THRESHOLD_WHITE": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 255,
                },
                "GLOBAL_PAGE_THRESHOLD_BLACK": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 255,
                },
            },
        },
        "outputs": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "show_image_level": {"type": "integer", "minimum": 0, "maximum": 6},
                "save_image_level": {"type": "integer", "minimum": 0, "maximum": 6},
                # This option shows colored outputs while taking a small toll on the processing speeds
                "show_colored_outputs": {"type": "boolean"},
                "save_detections": {"type": "boolean"},
                "save_image_metrics": {"type": "boolean"},
                # This option moves multimarked files into a separate folder for manual checking, skipping evaluation
                "filter_out_multimarked_files": {"type": "boolean"},
            },
        },
    },
}
