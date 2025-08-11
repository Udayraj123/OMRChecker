from src.schemas.constants import (
    DEFAULT_ANSWERS_SUMMARY_FORMAT_STRING,
    DEFAULT_SCORE_FORMAT_STRING,
)

EVALUATION_CONFIG_DEFAULTS = {
    "options": {},
    "marking_schemes": {},
    "conditional_sets": [],
    "outputs_configuration": {
        "should_explain_scoring": False,
        "should_export_explanation_csv": False,
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
        "draw_question_verdicts": {
            "enabled": True,
            "verdict_colors": {
                "correct": "#00FF00",
                "neutral": None,
                "incorrect": "#FF0000",
                "bonus": "#00DDDD",
            },
            "verdict_symbol_colors": {
                "positive": "#000000",
                "neutral": "#000000",
                "negative": "#000000",
                "bonus": "#000000",
            },
            "draw_answer_groups": {
                "enabled": True,
                "color_sequence": ["#8DFBC4", "#F7FB8D", "#8D9EFB", "#EA666F"],
            },
        },
        "draw_detected_bubble_texts": {"enabled": True},
    },
}
