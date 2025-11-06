"""
Default configuration loader for OMRChecker.

This loads defaults from the YAML config file and provides
a fallback mechanism in case keys are missing.
"""

from config.config_loader import load_config

# Load config from YAML
config = load_config()

DEFAULT_CONFIG = {
    "threshold_params": {
        "threshold_value": config["preprocessing"].get("threshold", 200),
        "lower_threshold": config["preprocessing"].get("canny_min", 185),
        "upper_threshold": config["preprocessing"].get("canny_max", 55),
        "PAGE_TYPE_FOR_THRESHOLD": config.get("omr", {}).get("page_type_for_threshold", "white"),
    },
    "paths": {
        "input_dir": config["paths"].get("input_dir", "input/"),
        "output_dir": config["paths"].get("output_dir", "output/"),
        "temp_dir": config["paths"].get("temp_dir", "temp/"),
        "logs_dir": config["paths"].get("logs_dir", "logs/"),
    },
    "omr": {
        "bubble_threshold": config["omr"].get("bubble_threshold", 0.7),
        "max_choices": config["omr"].get("max_choices", 4),
        "sheet_template": config["omr"].get("sheet_template", "templates/default_template.json"),
    },
    "logging": {
        "level": config["logging"].get("level", "INFO"),
        "save_logs": config["logging"].get("save_logs", True),
    },
    "cli": {
        "enable_argument_override": config["cli"].get("enable_argument_override", True),
    }
}
