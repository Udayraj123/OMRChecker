from dotmap import DotMap

DEFAULT_SECTION_KEY = "DEFAULT"

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
        "CORRECT": "correct",
        "INCORRECT": "incorrect",
        "UNMARKED": "unmarked",
    },
    _dynamic=False,
)

SCHEMA_VERDICTS_IN_ORDER = [
    SchemaVerdict.CORRECT,
    SchemaVerdict.INCORRECT,
    SchemaVerdict.UNMARKED,
]

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


ARRAY_OF_STRINGS = {
    "type": "array",
    "items": {"type": "string"},
}

FIELD_STRING_TYPE = {
    "type": "string",
    "pattern": "^([^\\.]+|[^\\.\\d]+\\d+\\.{2,3}\\d+)$",
}

FIELD_STRING_REGEX_GROUPS = r"([^\.\d]+)(\d+)\.{2,3}(\d+)"
