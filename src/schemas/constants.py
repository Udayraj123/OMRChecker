DEFAULT_SECTION_KEY = "DEFAULT"

BONUS_SECTION_PREFIX = "BONUS"

MARKING_VERDICT_TYPES = ["correct", "incorrect", "unmarked"]

ARRAY_OF_STRINGS = {
    "type": "array",
    "items": {"type": "string"},
}

FIELD_STRING_TYPE = {
    "type": "string",
    "pattern": "^([^\\.]+|[^\\.\\d]+\\d+\\.{2,3}\\d+)$",
}

FIELD_STRING_REGEX_GROUPS = r"([^\.\d]+)(\d+)\.{2,3}(\d+)"
