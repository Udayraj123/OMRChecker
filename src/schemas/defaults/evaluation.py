from src.schemas.constants import (
    DEFAULT_ANSWERS_SUMMARY_FORMAT_STRING,
    DEFAULT_SCORE_FORMAT_STRING,
)

EVALUATION_CONFIG_DEFAULTS = {
    "options": {
        "should_explain_scoring": False,
    },
    "marking_schemes": {},
    "outputs_configuration": {
        "draw_score": {
            "enabled": False,
            "position": [200, 200],
            "score_format_string": DEFAULT_SCORE_FORMAT_STRING,
            "size": 1.5,
        },
        "draw_answers_summary": {
            "enabled": False,
            "position": [200, 600],
            "answers_summary_format_string": DEFAULT_ANSWERS_SUMMARY_FORMAT_STRING,
            "size": 1.0,
        },
        "verdict_colors": {
            "correct": "#00ff00",
            "incorrect": "#ff0000",
            "unmarked": "#0000ff",
        },
    },
}
