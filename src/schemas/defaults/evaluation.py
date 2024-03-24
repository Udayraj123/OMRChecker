from src.schemas.constants import (
    DEFAULT_ANSWERS_SUMMARY_FORMAT_STRING,
    DEFAULT_SCORE_FORMAT_STRING,
)

EVALUATION_CONFIG_DEFAULTS = {
    "options": {
        "should_explain_scoring": False,
        # TODO: move into "outputs_configuration"
        "score_format_string": DEFAULT_SCORE_FORMAT_STRING,
        "answers_summary_format_string": DEFAULT_ANSWERS_SUMMARY_FORMAT_STRING,
        "draw_answers_summary": False,
        "draw_score": False,
    },
    "marking_schemes": {},
    "outputs_configuration": {},
}
