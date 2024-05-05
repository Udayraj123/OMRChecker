from src.schemas.constants import (
    ARRAY_OF_STRINGS,
    DEFAULT_SECTION_KEY,
    FIELD_STRING_TYPE,
    SCHEMA_VERDICTS_IN_ORDER,
    two_positive_integers,
)

marking_score_regex = "-?(\\d+)(/(\\d+))?"

marking_score = {
    "oneOf": [
        {"type": "string", "pattern": marking_score_regex},
        {"type": "number"},
    ]
}

marking_object_properties = {
    "additionalProperties": False,
    "required": SCHEMA_VERDICTS_IN_ORDER,
    "type": "object",
    "properties": {verdict: marking_score for verdict in SCHEMA_VERDICTS_IN_ORDER},
}
image_and_csv_options = {
    "additionalProperties": False,
    "required": ["answer_key_csv_path"],
    "dependentRequired": {
        "answer_key_image_path": [
            "answer_key_csv_path",
            "questions_in_order",
        ]
    },
    "type": "object",
    "properties": {
        "should_explain_scoring": {"type": "boolean"},
        "answer_key_csv_path": {"type": "string"},
        "answer_key_image_path": {"type": "string"},
        "questions_in_order": ARRAY_OF_STRINGS,
    },
}

EVALUATION_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/Udayraj123/OMRChecker/tree/master/src/schemas/evaluation-schema.json",
    "title": "Evaluation Schema",
    "description": "OMRChecker evaluation schema i.e. the marking scheme",
    "type": "object",
    "additionalProperties": False,
    "required": ["source_type", "options", "marking_schemes"],
    "properties": {
        "additionalProperties": False,
        "source_type": {"type": "string", "enum": ["csv", "image_and_csv", "custom"]},
        "options": {"type": "object"},
        "marking_schemes": {
            "type": "object",
            "required": [DEFAULT_SECTION_KEY],
            "patternProperties": {
                f"^{DEFAULT_SECTION_KEY}$": marking_object_properties,
                f"^(?!{DEFAULT_SECTION_KEY}$).*": {
                    "additionalProperties": False,
                    "required": ["marking", "questions"],
                    "type": "object",
                    "properties": {
                        "questions": {
                            "oneOf": [
                                FIELD_STRING_TYPE,
                                {
                                    "type": "array",
                                    "items": FIELD_STRING_TYPE,
                                },
                            ]
                        },
                        "marking": marking_object_properties,
                    },
                },
            },
        },
        "outputs_configuration": {
            "type": "object",
            "additionalProperties": False,
            "required": [],
            "properties": {
                "draw_score": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "enabled",
                    ],
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "position": two_positive_integers,
                        "score_format_string": {"type": "string"},
                        "size": {"type": "number"},
                    },
                    "allOf": [
                        {
                            "if": {"properties": {"enabled": {"const": True}}},
                            "then": {
                                "required": ["position", "score_format_string"],
                            },
                        }
                    ],
                },
                "draw_answers_summary": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "enabled",
                    ],
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "position": two_positive_integers,
                        "answers_summary_format_string": {"type": "string"},
                        "size": {"type": "number"},
                    },
                    "allOf": [
                        {
                            "if": {"properties": {"enabled": {"const": True}}},
                            "then": {
                                "required": [
                                    "position",
                                    "answers_summary_format_string",
                                ],
                            },
                        }
                    ],
                },
                "draw_question_verdicts": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["enabled"],
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "verdict_colors": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["correct", "neutral", "negative", "bonus"],
                            "properties": {
                                "correct": {"type": "string"},
                                "neutral": {"type": "string"},
                                "negative": {"type": "string"},
                                "bonus": {"type": "string"},
                            },
                        },
                        "verdict_symbol_colors": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["positive", "neutral", "negative", "bonus"],
                            "properties": {
                                "positive": {"type": "string"},
                                "neutral": {"type": "string"},
                                "negative": {"type": "string"},
                                "bonus": {"type": "string"},
                            },
                        },
                        "draw_answer_groups": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [],
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "color_sequence": {**ARRAY_OF_STRINGS,"minItems": 4,"maxItems": 4}
                            },
                            "allOf": [
                                {
                                    "if": {"properties": {"enabled": {"const": True}}},
                                    "then": {
                                        "required": ["color_sequence"],
                                    },
                                }
                            ],
                        },
                    },
                    "allOf": [
                        {
                            "if": {"properties": {"enabled": {"const": True}}},
                            "then": {
                                "required": [
                                    "verdict_colors",
                                    "verdict_symbol_colors",
                                    "draw_answer_groups",
                                ],
                            },
                        }
                    ],
                },
                "draw_detected_bubble_texts": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["enabled"],
                    "properties": {"enabled": {"type": "boolean"}},
                },
            },
        },
    },
    "allOf": [
        {
            "if": {"properties": {"source_type": {"const": "csv"}}},
            "then": {"properties": {"options": image_and_csv_options}},
        },
        {
            "if": {"properties": {"source_type": {"const": "image_and_csv"}}},
            "then": {"properties": {"options": image_and_csv_options}},
        },
        {
            "if": {"properties": {"source_type": {"const": "custom"}}},
            "then": {
                "properties": {
                    "options": {
                        "additionalProperties": False,
                        "required": ["answers_in_order", "questions_in_order"],
                        "type": "object",
                        "properties": {
                            "should_explain_scoring": {"type": "boolean"},
                            "answers_in_order": {
                                "oneOf": [
                                    {
                                        "type": "array",
                                        "items": {
                                            "oneOf": [
                                                # Standard answer type allows single correct answers. They can have multiple characters(multi-marked) as well.
                                                # Useful for any standard response e.g. 'A', '01', '99', 'AB', etc
                                                {"type": "string"},
                                                # Multiple correct answer type covers multiple correct answers
                                                # Useful for ambiguous/bonus questions e.g. ['A', 'B'], ['1', '01'], ['A', 'B', 'AB'], etc
                                                {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                    "minItems": 2,
                                                },
                                                # Multiple correct weighted answer covers multiple answers with weights
                                                # Useful for partial marking e.g. [['A', 2], ['B', 0.5], ['AB', 2.5]], [['1', 0.5], ['01', 1]], etc
                                                {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "array",
                                                        "items": False,
                                                        "minItems": 2,
                                                        "maxItems": 2,
                                                        "prefixItems": [
                                                            {"type": "string"},
                                                            marking_score,
                                                        ],
                                                    },
                                                },
                                            ],
                                        },
                                    },
                                ]
                            },
                            "questions_in_order": ARRAY_OF_STRINGS,
                        },
                    }
                }
            },
        },
    ],
}
