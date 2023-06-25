from src.schemas.constants import (
    ARRAY_OF_STRINGS,
    DEFAULT_SECTION_KEY,
    FIELD_STRING_TYPE,
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
    "required": ["correct", "incorrect", "unmarked"],
    "type": "object",
    "properties": {
        "correct": marking_score,
        "incorrect": marking_score,
        "unmarked": marking_score,
    },
}

EVALUATION_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/Udayraj123/OMRChecker/tree/master/src/schemas/evaluation-schema.json",
    "title": "Evaluation Schema",
    "description": "OMRChecker evaluation schema i.e. the marking scheme",
    "type": "object",
    "additionalProperties": True,
    "required": ["source_type", "options", "marking_scheme"],
    "properties": {
        "additionalProperties": False,
        "source_type": {"type": "string", "enum": ["csv", "custom"]},
        "options": {"type": "object"},
        "marking_scheme": {
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
                            "should_explain_scoring": {"type": "boolean"},
                            "answers_in_order": {
                                "oneOf": [
                                    {
                                        "type": "array",
                                        "items": {
                                            "oneOf": [
                                                # "standard": single correct, multimarked single-correct
                                                # Example: "q1" --> 'AB'
                                                {"type": "string"},
                                                # "multiple-correct": multiple correct answers (for ambiguos/bonus questions)
                                                # Example: "q1" --> [ 'A', 'B' ]
                                                {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                    "minItems": 2,
                                                },
                                                # "multiple-correct-weighted": array of answer-wise weights
                                                # Example: "q1" --> [['A', 1], ['B', 2], ['C', 3]]
                                                {
                                                    "type": "array",
                                                    "items": False,
                                                    "maxItems": 2,
                                                    "minItems": 2,
                                                    "prefixItems": [
                                                        {"type": "string"},
                                                        {
                                                            "type": "array",
                                                            "items": marking_score,
                                                            "minItems": 1,
                                                            "maxItems": 3,
                                                        },
                                                    ],
                                                },
                                            ],
                                        },
                                    },
                                    {
                                        # TODO: answer_weight format
                                        "type": "array",  # two column array for weights
                                        "items": False,
                                        "maxItems": 2,
                                        "minItems": 2,
                                        "prefixItems": [
                                            {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "minItems": 2,
                                                "maxItems": 2,
                                            },
                                            {
                                                "type": "array",
                                                "items": marking_score,
                                                "minItems": 1,
                                                "maxItems": 3,
                                            },
                                        ],
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
