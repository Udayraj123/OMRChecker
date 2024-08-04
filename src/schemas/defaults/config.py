from dotmap import DotMap

from src.utils.constants import SUPPORTED_PROCESSOR_NAMES

CONFIG_DEFAULTS = DotMap(
    {
        "thresholding": {
            "GAMMA_LOW": 0.7,
            "MIN_GAP_TWO_BUBBLES": 30,
            "MIN_JUMP": 25,
            "CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY": 25,
            "MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK": 5,
            "GLOBAL_THRESHOLD_MARGIN": 10,
            "JUMP_DELTA": 30,
            # Note: tune this value to avoid empty bubble detections
            "GLOBAL_PAGE_THRESHOLD": 200,
            "MIN_JUMP_STD": 15,
            "JUMP_DELTA_STD": 5,
            "GLOBAL_PAGE_THRESHOLD_STD": 10,
        },
        "outputs": {
            "display_image_dimensions": [720, 1080],
            "show_image_level": 0,
            "show_preprocessors_diff": {
                **{
                    processor_name: False
                    for processor_name in SUPPORTED_PROCESSOR_NAMES
                }
            },
            "save_image_level": 1,
            "show_logs_by_type": {
                "critical": True,
                "error": True,
                "warning": True,
                "info": True,
                "debug": False,
            },
            "save_detections": True,
            "colored_outputs_enabled": False,
            "save_image_metrics": False,
            "show_confidence_metrics": False,
            "filter_out_multimarked_files": False,
        },
    },
    _dynamic=False,
)
