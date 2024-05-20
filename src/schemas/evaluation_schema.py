from src.schemas.constants import (
    DEFAULT_SECTION_KEY,
    SCHEMA_VERDICTS_IN_ORDER,
    load_common_defs,
)

marking_score_regex = "-?(\\d+)(/(\\d+))?"

marking_score = {
    "oneOf": [
        {
            "description": "The marking score as a string. We can pass natural fractions as well",
            "type": "string",
            "pattern": marking_score_regex,
        },
        {
            "description": "The marking score as a number. It can be negative as well",
            "type": "number",
        },
    ]
}

marking_object_properties = {
    "description": "The marking object describes verdict-wise score deltas",
    "required": SCHEMA_VERDICTS_IN_ORDER,
    "type": "object",
    "additionalProperties": False,
    "properties": {
        schema_verdict: marking_score for schema_verdict in SCHEMA_VERDICTS_IN_ORDER
    },
}
image_and_csv_options = {
    "description": "The options needed if source type is image and csv",
    "required": ["answer_key_csv_path"],
    "dependentRequired": {
        "answer_key_image_path": [
            "answer_key_csv_path",
            "questions_in_order",
        ]
    },
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer_key_csv_path": {
            "description": "The path to the answer key csv relative to the evaluation.json file",
            "type": "string",
        },
        "answer_key_image_path": {
            "description": "The path to the answer key image relative to the evaluation.json file",
            "type": "string",
        },
        "questions_in_order": {
            "$ref": "#/$def/array_of_strings",
            "description": "An array of fields to treat as questions when the answer key image is provided",
        },
    },
}

common_evaluation_schema_properties = {
    "source_type": {"type": "string", "enum": ["csv", "image_and_csv", "custom"]},
    "options": {"type": "object"},
    "marking_schemes": {
        "type": "object",
        "required": [DEFAULT_SECTION_KEY],
        "patternProperties": {
            f"^{DEFAULT_SECTION_KEY}$": {"$ref": "#/$def/marking_object_properties"},
            f"^(?!{DEFAULT_SECTION_KEY}$).*": {
                "description": "A section that defines custom marking for a subset of the questions",
                "additionalProperties": False,
                "required": ["marking", "questions"],
                "type": "object",
                "properties": {
                    "questions": {
                        "oneOf": [
                            {"$ref": "#/$def/field_string_type"},
                            {
                                "type": "array",
                                "items": {"$ref": "#/$def/field_string_type"},
                            },
                        ]
                    },
                    "marking": {"$ref": "#/$def/marking_object_properties"},
                },
            },
        },
    },
    "outputs_configuration": {
        "description": "The configuration for outputs produced from the evaluation",
        "type": "object",
        "required": [],
        "additionalProperties": False,
        "properties": {
            "should_explain_scoring": {
                "description": "Whether to print the table explaining question-wise verdicts",
                "type": "boolean",
            },
            "draw_score": {
                "description": "The configuration for drawing the final score",
                "type": "object",
                "required": [
                    "enabled",
                ],
                "additionalProperties": False,
                "properties": {
                    "enabled": {
                        "description": "The toggle for enabling the configuration",
                        "type": "boolean",
                    },
                    "position": {
                        "description": "The position of the score box",
                        "$ref": "#/$def/two_positive_integers",
                    },
                    "score_format_string": {
                        "description": "The format string to compose the score string. Supported variables - {score}",
                        "type": "string",
                    },
                    "size": {
                        "description": "The font size for the score box",
                        "type": "number",
                    },
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
                "description": "The configuration for drawing the answers summary",
                "type": "object",
                "required": [
                    "enabled",
                ],
                "additionalProperties": False,
                "properties": {
                    "enabled": {
                        "description": "The toggle for enabling the configuration",
                        "type": "boolean",
                    },
                    "position": {
                        "description": "The position of the answers summary box",
                        "$ref": "#/$def/two_positive_integers",
                    },
                    "answers_summary_format_string": {
                        "description": "The format string to compose the answer summary. Supported variables - {correct}, {incorrect}, {unmarked} ",
                        "type": "string",
                    },
                    "size": {
                        "description": "The font size for the answers summary box",
                        "type": "number",
                    },
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
                        "description": "The mapping from delta sign notions to the corresponding colors",
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["correct", "neutral", "incorrect", "bonus"],
                        "properties": {
                            "correct": {
                                "description": "The color of the bubble box when delta > 0",
                                "$ref": "#/$def/matplotlib_color",
                            },
                            "neutral": {
                                "description": "The color of the bubble box when delta == 0 (defaults to incorrect)",
                                "oneOf": [
                                    {"$ref": "#/$def/matplotlib_color"},
                                    # Allow null for default to incorrect color
                                    {"type": "null"},
                                ],
                            },
                            "incorrect": {
                                "description": "The color of the bubble box when delta < 0",
                                "$ref": "#/$def/matplotlib_color",
                            },
                            "bonus": {
                                "description": "The color of the bubble box when delta > 0 and question is part of a bonus scheme",
                                "$ref": "#/$def/matplotlib_color",
                            },
                        },
                    },
                    "verdict_symbol_colors": {
                        "description": "The mapping from verdict symbols(based on delta sign) to the corresponding colors",
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["positive", "neutral", "negative", "bonus"],
                        "properties": {
                            "positive": {
                                "description": "The color of '+' symbol when delta > 0",
                                "$ref": "#/$def/matplotlib_color",
                            },
                            "neutral": {
                                "description": "The color of 'o' symbol when delta == 0",
                                "$ref": "#/$def/matplotlib_color",
                            },
                            "negative": {
                                "description": "The color of '-' symbol when delta < 0",
                                "$ref": "#/$def/matplotlib_color",
                            },
                            "bonus": {
                                "description": "The color of '*' symbol when delta > 0 and question is part of a bonus scheme",
                                "$ref": "#/$def/matplotlib_color",
                            },
                        },
                    },
                    "draw_answer_groups": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [],
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "color_sequence": {
                                "type": "array",
                                "items": {
                                    "$ref": "#/$def/matplotlib_color",
                                },
                                "minItems": 4,
                                "maxItems": 4,
                            },
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
}
common_evaluation_schema_conditions = [
    {
        "if": {"properties": {"source_type": {"const": "csv"}}},
        "then": {"properties": {"options": {"$ref": "#/$def/image_and_csv_options"}}},
    },
    {
        "if": {"properties": {"source_type": {"const": "image_and_csv"}}},
        "then": {"properties": {"options": {"$ref": "#/$def/image_and_csv_options"}}},
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
                        "questions_in_order": {
                            "$ref": "#/$def/array_of_strings",
                            "description": "An array of fields to treat as questions specified in an order to apply evaluation",
                        },
                        "answers_in_order": {
                            "oneOf": [
                                {
                                    "description": "An array of answers in the same order as provided array of questions",
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
                    },
                }
            }
        },
    },
]

EVALUATION_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/Udayraj123/OMRChecker/tree/master/src/schemas/evaluation-schema.json",
    "$def": {
        # The common definitions go here
        **load_common_defs(
            [
                "array_of_strings",
                "two_positive_integers",
                "field_string_type",
                "matplotlib_color",
            ]
        ),
        "marking_object_properties": marking_object_properties,
        "marking_score": marking_score,
        "image_and_csv_options": image_and_csv_options,
    },
    "title": "Evaluation Schema",
    "description": "The OMRChecker evaluation schema",
    "type": "object",
    "required": ["source_type", "options", "marking_schemes"],
    "additionalProperties": False,
    "properties": {
        "additionalProperties": False,
        **common_evaluation_schema_properties,
        "conditionalSets": {
            "description": "An array of answer sets with their conditions. These will override the default values in case of any conflict",
            "type": "array",
            "items": {
                "description": "Each item represents a conditional evaluation schema to apply for the given matcher",
                "type": "object",
                "required": ["name", "matcher", "evaluation"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "matcher": {
                        "description": "Mapping response fields from default layout to the set name",
                        "type": "object",
                        "required": ["formatString", "matchRegex"],
                        "additionalProperties": False,
                        "properties": {
                            "formatString": {
                                "description": "Format string composed of the response variables to apply the regex on e.g. '{roll}-{barcode}'",
                                "type": "string",
                            },
                            # Example: match last four characters ".*-SET1"
                            "matchRegex": {
                                "description": "Mapping to use on the composed field string",
                                "type": "string",
                                "format": "regex",
                            },
                        },
                    },
                    "evaluation": {
                        # Note: even outputs_configuration is going to be different as per the set, allowing custom colors for different sets!
                        "description": "The custom evaluation schema to apply if given matcher is satisfied",
                        "type": "object",
                        "required": ["source_type", "options", "marking_schemes"],
                        "additionalProperties": False,
                        "properties": {
                            **common_evaluation_schema_properties,
                        },
                        "allOf": [*common_evaluation_schema_conditions],
                    },
                },
            },
        },
    },
    "allOf": [*common_evaluation_schema_conditions],
}
