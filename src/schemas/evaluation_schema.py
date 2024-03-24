from src.schemas.constants import (
    ARRAY_OF_STRINGS,
    DEFAULT_SECTION_KEY,
    FIELD_STRING_TYPE,
    SCHEMA_VERDICTS_IN_ORDER,
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
    # TODO: can support streak marking if we allow array of marking_scores here
    "properties": {verdict: marking_score for verdict in SCHEMA_VERDICTS_IN_ORDER},
}

common_options_schema = {
    "draw_score": {"type": "boolean"},
    "draw_answers_summary": {"type": "boolean"},
    "answers_summary_format_string": {
        "type": "string",
    },
    "score_format_string": {
        "type": "string",
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
        "source_type": {"type": "string", "enum": ["csv", "custom"]},
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
    },
    "allOf": [
        {
            "if": {"properties": {"source_type": {"const": "csv"}}},
            "then": {
                "properties": {
                    "options": {
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
                            **common_options_schema,
                            "should_explain_scoring": {"type": "boolean"},
                            "answer_key_csv_path": {"type": "string"},
                            "answer_key_image_path": {"type": "string"},
                            "questions_in_order": ARRAY_OF_STRINGS,
                        },
                    }
                }
            },
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
                            **common_options_schema,
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
