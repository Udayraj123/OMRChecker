DEFAULT_SECTION_KEY = "DEFAULT"
BONUS_SECTION_PREFIX = "BONUS"
MARKING_VERDICT_TYPES = ["correct", "incorrect", "unmarked"]
array_of_strings = {
    "type": "array",
    "items": {"type": "string"},
}
marking_object_properties = {
    "additionalProperties": False,
    "required": ["correct", "incorrect", "unmarked"],
    "type": "object",
    "properties": {
        "correct": {
            "oneOf": [
                {"type": "string", "pattern": "-?(\\d+)(/(\\d+))?"},
                {"type": "number", "minimum": 0},
            ]
        },
        "incorrect": {
            "oneOf": [
                {"type": "string", "pattern": "-?(\\d+)(/(\\d+))?"},
                {"type": "number"},
            ]
        },
        "unmarked": {
            "oneOf": [
                {"type": "string", "pattern": "-?(\\d+)(/(\\d+))?"},
                {"type": "number", "minimum": 0},
            ]
        },
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
                DEFAULT_SECTION_KEY: marking_object_properties,
                "^(?!DEFAULT$).*": {
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
                            "answers_in_order": array_of_strings,
                            "questions_in_order": array_of_strings,
                        },
                    }
                }
            },
        },
    ],
}
