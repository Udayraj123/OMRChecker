from dotmap import DotMap

from src.utils.constants import MATPLOTLIB_COLORS

DEFAULT_SECTION_KEY = "DEFAULT"
DEFAULT_SET_NAME = "DEFAULT_SET"

BONUS_SECTION_PREFIX = "BONUS"

Verdict = DotMap(
    {
        "ANSWER_MATCH": "answer-match",
        "NO_ANSWER_MATCH": "no-answer-match",
        "UNMARKED": "unmarked",
    },
    _dynamic=False,
)

VERDICTS_IN_ORDER = [
    Verdict.ANSWER_MATCH,
    Verdict.NO_ANSWER_MATCH,
    Verdict.UNMARKED,
]


SchemaVerdict = DotMap(
    {
        # Note: bonus not allowed as schema verdict.
        "CORRECT": "correct",
        "INCORRECT": "incorrect",
        "UNMARKED": "unmarked",
    },
    _dynamic=False,
)

MarkingSchemeType = DotMap(
    {
        "DEFAULT": "default",
        "VERDICT_LEVEL_STREAK": "verdict_level_streak",
        "SECTION_LEVEL_STREAK": "section_level_streak",
    },
    _dynamic=False,
)

MARKING_SCHEME_TYPES_IN_ORDER = [
    MarkingSchemeType.DEFAULT,
    MarkingSchemeType.VERDICT_LEVEL_STREAK,
    MarkingSchemeType.SECTION_LEVEL_STREAK,
]

SCHEMA_VERDICTS_IN_ORDER = [
    SchemaVerdict.CORRECT,
    SchemaVerdict.INCORRECT,
    SchemaVerdict.UNMARKED,
]

DEFAULT_SCORE_FORMAT_STRING = "Score: {score}"
DEFAULT_ANSWERS_SUMMARY_FORMAT_STRING = " ".join(
    [
        f"{schema_verdict.title()}: {{{schema_verdict}}}"
        for schema_verdict in SCHEMA_VERDICTS_IN_ORDER
    ]
)

VERDICT_TO_SCHEMA_VERDICT = {
    Verdict.ANSWER_MATCH: SchemaVerdict.CORRECT,
    Verdict.NO_ANSWER_MATCH: SchemaVerdict.INCORRECT,
    Verdict.UNMARKED: SchemaVerdict.UNMARKED,
}

AnswerType = DotMap(
    {
        # Standard answer type allows single correct answers. They can have multiple characters(multi-marked) as well.
        # Useful for any standard response e.g. 'A', '01', '99', 'AB', etc
        "STANDARD": "standard",
        # Multiple correct answer type covers multiple correct answers
        # Useful for ambiguous/bonus questions e.g. ['A', 'B'], ['1', '01'], ['A', 'B', 'AB'], etc
        "MULTIPLE_CORRECT": "multiple-correct",
        # Multiple correct weighted answer covers multiple answers with weights
        # Useful for partial marking e.g. [['A', 2], ['B', 0.5], ['AB', 2.5]], [['1', 0.5], ['01', 1]], etc
        "MULTIPLE_CORRECT_WEIGHTED": "multiple-correct-weighted",
    },
    _dynamic=False,
)

ALL_COMMON_DEFS = {
    "array_of_strings": {
        "type": "array",
        "items": {"type": "string"},
    },
    "field_string_type": {
        "type": "string",
        # TODO: underscore support is not there "q_11..2"
        "pattern": "^([^\\.]+|[^\\.\\d]+\\d+\\.{2,3}\\d+)$",
    },
    "positive_number": {"type": "number", "minimum": 0},
    "positive_integer": {"type": "integer", "minimum": 0},
    "two_integers": {
        "type": "array",
        "prefixItems": [
            {
                "type": "integer",
            },
            {
                "type": "integer",
            },
        ],
        "maxItems": 2,
        "minItems": 2,
    },
    "two_positive_integers": {
        "type": "array",
        "prefixItems": [
            {
                "$ref": "#/$def/positive_integer",
            },
            {
                "$ref": "#/$def/positive_integer",
            },
        ],
        "maxItems": 2,
        "minItems": 2,
    },
    "two_positive_numbers": {
        "type": "array",
        "prefixItems": [
            {
                "$ref": "#/$def/positive_number",
            },
            {
                "$ref": "#/$def/positive_number",
            },
        ],
        "maxItems": 2,
        "minItems": 2,
    },
    "zero_to_one_number": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
    },
    "matplotlib_color": {
        "oneOf": [
            {
                "type": "string",
                "description": "This should match with #rgb, #rgba, #rrggbb, and #rrggbbaa syntax",
                "pattern": "^#(?:(?:[\\da-fA-F]{3}){1,2}|(?:[\\da-fA-F]{4}){1,2})$",
            },
            {
                "type": "string",
                "description": "This should match with all colors supported by matplotlib. Ref: https://matplotlib.org/stable/gallery/color/named_colors.html",
                "enum": list(MATPLOTLIB_COLORS.keys()),
            },
        ]
    },
}

FIELD_STRING_REGEX_GROUPS = r"([^\.\d]+)(\d+)\.{2,3}(\d+)"


def load_common_defs(keys):
    # inject dependencies
    if "two_positive_integers" in keys:
        keys += ["positive_integer"]
    if "two_positive_numbers" in keys:
        keys += ["positive_number"]
    keys = set(keys)
    return {key: ALL_COMMON_DEFS[key] for key in keys}
