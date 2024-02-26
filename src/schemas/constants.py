from dotmap import DotMap

DEFAULT_SECTION_KEY = "DEFAULT"

BONUS_SECTION_PREFIX = "BONUS"

Verdict = DotMap(
    {
        "CORRECT": "correct",
        "INCORRECT": "incorrect",
        "UNMARKED": "unmarked",
    },
    _dynamic=False,
)

VERDICTS_IN_ORDER = [
    Verdict.CORRECT,
    Verdict.INCORRECT,
    Verdict.UNMARKED,
]

ARRAY_OF_STRINGS = {
    "type": "array",
    "items": {"type": "string"},
}

FIELD_STRING_TYPE = {
    "type": "string",
    "pattern": "^([^\\.]+|[^\\.\\d]+\\d+\\.{2,3}\\d+)$",
}

FIELD_STRING_REGEX_GROUPS = r"([^\.\d]+)(\d+)\.{2,3}(\d+)"
