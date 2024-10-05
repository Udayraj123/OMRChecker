from src.schemas.constants import (
    DEFAULT_SECTION_KEY,
    MARKING_SCHEME_TYPES_IN_ORDER,
    SCHEMA_VERDICTS_IN_ORDER,
    MarkingSchemeType,
    load_common_defs,
)

marking_score_regex = "-?(\\d+)(/(\\d+))?"

marking_score_without_streak = {
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

marking_score_with_streak = {
    **marking_score_without_streak,
    "oneOf": [
        *marking_score_without_streak["oneOf"],
        {
            "description": "An array will be used for consecutive streak based marking. Note: original order from questions_in_order will be used.",
            "type": "array",
            "items": {"oneOf": [*marking_score_without_streak["oneOf"]]},
        },
    ],
}

section_marking_without_streak_object = {
    "description": "The marking object describes verdict-wise score deltas",
    "required": SCHEMA_VERDICTS_IN_ORDER,
    "type": "object",
    "additionalProperties": False,
    "properties": {
        schema_verdict: {
            "$ref": "#/$def/marking_score_without_streak",
        }
        for schema_verdict in SCHEMA_VERDICTS_IN_ORDER
    },
}

section_marking_with_streak_object = {
    **section_marking_without_streak_object,
    "properties": {
        schema_verdict: {"$ref": "#/$def/marking_score_with_streak"}
        for schema_verdict in SCHEMA_VERDICTS_IN_ORDER
    },
}


custom_section_marking_object_conditions = [
    {
        "if": {"properties": {"marking_type": {"const": MarkingSchemeType.DEFAULT}}},
        "then": {
            "properties": {
                "marking": {"$ref": "#/$def/section_marking_without_streak_object"},
            }
        },
    },
    {
        "if": {
            "properties": {
                "marking_type": {"const": MarkingSchemeType.SECTION_LEVEL_STREAK}
            }
        },
        "then": {
            "properties": {
                "marking": {"$ref": "#/$def/section_marking_with_streak_object"},
            }
        },
    },
    {
        "if": {
            "properties": {
                "marking_type": {"const": MarkingSchemeType.VERDICT_LEVEL_STREAK}
            }
        },
        "then": {
            "properties": {
                "marking": {"$ref": "#/$def/section_marking_with_streak_object"},
            }
        },
        "else": {
            "properties": {
                "marking": {"$ref": "#/$def/section_marking_without_streak_object"},
            }
        },
    },
]

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

local_questions_and_answers_options = {
    "description": "This method allows setting questions and their answers within the evaluation file itself",
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
                                        {
                                            "$ref": "#/$def/marking_score_without_streak",
                                        },
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

common_evaluation_schema_properties = {
    "source_type": {"type": "string", "enum": ["csv", "image_and_csv", "local"]},
    "options": {"type": "object"},
    "marking_schemes": {
        "type": "object",
        "required": [DEFAULT_SECTION_KEY],
        "patternProperties": {
            f"^{DEFAULT_SECTION_KEY}$": {
                "$ref": "#/$def/section_marking_without_streak_object"
            },
            f"^(?!{DEFAULT_SECTION_KEY}$).*": {
                "description": "A section that defines custom marking for a subset of the questions",
                "additionalProperties": False,
                "required": ["marking", "questions"],
                "type": "object",
                "properties": {
                    "marking_type": {
                        "type": "string",
                        "enum": [*MARKING_SCHEME_TYPES_IN_ORDER],
                    },
                    "questions": {
                        "oneOf": [
                            {"$ref": "#/$def/field_string_type"},
                            {
                                "type": "array",
                                "items": {"$ref": "#/$def/field_string_type"},
                            },
                        ]
                    },
                    "marking": True,
                },
                "allOf": [*custom_section_marking_object_conditions],
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
            "should_export_explanation_csv": {
                "description": "Whether to export the explanation of evaluation results as a CSV file",
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
        "if": {"properties": {"source_type": {"const": "local"}}},
        "then": {
            "properties": {
                "options": {"$ref": "#/$def/local_questions_and_answers_options"}
            },
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
        "marking_score_without_streak": marking_score_without_streak,
        "marking_score_with_streak": marking_score_with_streak,
        "section_marking_without_streak_object": section_marking_without_streak_object,
        "section_marking_with_streak_object": section_marking_with_streak_object,
        "image_and_csv_options": image_and_csv_options,
        "local_questions_and_answers_options": local_questions_and_answers_options,
    },
    "title": "Evaluation Schema",
    "description": "The OMRChecker evaluation schema",
    "type": "object",
    "required": ["source_type", "options", "marking_schemes"],
    "additionalProperties": False,
    "properties": {
        "additionalProperties": False,
        # TODO: check if common_evaluation_schema_properties can be picked and overridden using a $ref
        **common_evaluation_schema_properties,
        "conditional_sets": {
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
                                "description": "Format string composed of the response variables to apply the regex on e.g. '{Roll}' or '{Roll}-{barcode}'",
                                "type": "string",
                            },
                            "matchRegex": {
                                "description": "The regex to match on the composed field string e.g. to match a suffix value: '.*-SET1'",
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
