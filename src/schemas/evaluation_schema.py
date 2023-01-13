DEFAULT_SECTION_KEY = "DEFAULT"
BONUS_SECTION_PREFIX = "BONUS"
MARKING_VERDICT_TYPES = ["correct", "incorrect", "unmarked"]
array_of_strings = {
    "type": "array",
    "items": {"type": "string"},
}
marking_score = {
    "oneOf": [
        {"type": "string", "pattern": "-?(\\d+)(/(\\d+))?"},
        {"type": "number"},
    ]
}
marking_score_or_streak_array = {
    "oneOf": [marking_score, {"type": "array", "items": marking_score}]
}

marking_object_properties = {
    "additionalProperties": False,
    "required": ["correct", "incorrect", "unmarked"],
    "type": "object",
    "properties": {
        "correct": marking_score_or_streak_array,
        "incorrect": marking_score_or_streak_array,
        "unmarked": marking_score_or_streak_array,
    },
}

question_string_regex = "^([^\\.]+)*?([^\\.\\d]+(\\d+)\\.{2,3}(\\d+))*?$"

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
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "pattern": question_string_regex,
                                    },
                                },
                                {"type": "string", "pattern": question_string_regex},
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
                                "evaluation_columns",
                            ]
                        },
                        "type": "object",
                        "properties": {
                            "should_explain_scoring": {"type": "boolean"},
                            "answer_key_csv_path": {"type": "string"},
                            "answer_key_image_path": {"type": "string"},
                            "evaluation_columns": array_of_strings,
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
                                                # standard: single correct, multimarked correct
                                                {"type": "string"},
                                                # multiple correct
                                                {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                    "minItems": 2,
                                                },
                                                {
                                                    "type": "array",  # two column array for weights
                                                    "items": False,
                                                    "maxItems": 2,
                                                    "minItems": 2,
                                                    "prefixItems": [
                                                        # first item is string of correct answer
                                                        {"type": "string"},
                                                        {
                                                            "type": "array",
                                                            "items": marking_score_or_streak_array,
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
                                                "items": marking_score_or_streak_array,
                                                "minItems": 1,
                                                "maxItems": 3,
                                            },
                                        ],
                                    },
                                ]
                            },
                            "questions_in_order": array_of_strings,
                        },
                    }
                }
            },
        },
    ],
}
