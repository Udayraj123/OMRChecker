from dotmap import DotMap

CONFIG_DEFAULTS = DotMap(
    {
        "dimensions": {
            "display_image_shape": [2480, 1640],
            "processing_image_shape": [820, 666],
        },
        "thresholding": {
            "GAMMA_LOW": 0.7,
            "MIN_GAP": 30,
            "MIN_JUMP": 25,
            "CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY": 25,
            "MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK": 5,
            "GLOBAL_THRESHOLD_MARGIN": 10,
            "JUMP_DELTA": 30,
            # Note: tune this value to avoid empty bubble detections
            "GLOBAL_PAGE_THRESHOLD": 200,
        },
        "outputs": {
            "show_image_level": 0,
            "save_image_level": 0,
            "save_detections": True,
            "show_colored_outputs": True,
            "save_image_metrics": False,
            "filter_out_multimarked_files": False,
        },
    },
    _dynamic=False,
)
