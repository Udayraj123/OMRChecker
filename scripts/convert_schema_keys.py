"""Script to convert JSON schema keys to camelCase.

This script systematically converts all snake_case keys in JSON schema files
to camelCase while preserving enum values and constant names.
"""

import re
from pathlib import Path


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def screaming_to_camel(name: str) -> str:
    """Convert SCREAMING_SNAKE_CASE to camelCase."""
    return snake_to_camel(name.lower())


def is_screaming_snake_case(text: str) -> bool:
    """Check if text is in SCREAMING_SNAKE_CASE format."""
    return bool(re.match(r"^[A-Z][A-Z0-9_]*$", text))


def is_snake_case(text: str) -> bool:
    """Check if text is in snake_case format."""
    return bool(re.match(r"^[a-z][a-z0-9_]*$", text))


def convert_schema_keys(content: str) -> str:
    """Convert all snake_case keys in schema to camelCase.

    This function handles property names in JSON schemas while preserving:
    - Enum values (SCREAMING_SNAKE_CASE like "BUBBLES_THRESHOLD", "DEFAULT")
    - String literals in descriptions
    - References to constants
    """
    lines = content.split("\n")
    result_lines = []

    for line in lines:
        # Match property definitions like: "snake_case_key": {
        property_match = re.match(r'^(\s*)"([a-z][a-z0-9_]+)":\s*(\{.*)$', line)
        if property_match:
            indent, key, rest = property_match.groups()
            camel_key = snake_to_camel(key)
            result_lines.append(f'{indent}"{camel_key}": {rest}')
            continue

        # Match SCREAMING_SNAKE property definitions in thresholding/config sections
        screaming_match = re.match(r'^(\s*)"([A-Z][A-Z0-9_]+)":\s*(\{.*)$', line)
        if screaming_match:
            indent, key, rest = screaming_match.groups()
            # Convert SCREAMING_SNAKE to camelCase for config values
            # But keep constants like output modes etc as is (handled contextually in description)
            if any(
                x in key
                for x in [
                    "THRESHOLD",
                    "JUMP",
                    "GAMMA",
                    "GAP",
                    "DELTA",
                    "MARGIN",
                    "SURPLUS",
                    "CONFIDENT",
                ]
            ):
                camel_key = screaming_to_camel(key)
                result_lines.append(f'{indent}"{camel_key}": {rest}')
                continue

        result_lines.append(line)

    return "\n".join(result_lines)


def main():
    """Convert schema files keys to camelCase."""
    schema_files = [
        Path(
            "/Users/udayraj.deshmukh/Personals/OMRChecker/src/schemas/config_schema.py"
        ),
        Path(
            "/Users/udayraj.deshmukh/Personals/OMRChecker/src/schemas/template_schema.py"
        ),
        Path(
            "/Users/udayraj.deshmukh/Personals/OMRChecker/src/schemas/evaluation_schema.py"
        ),
    ]

    for schema_file in schema_files:
        if not schema_file.exists():
            print(f"✗ File not found: {schema_file}")
            continue

        print(f"Converting {schema_file.name}...")
        content = schema_file.read_text()

        # Convert the keys
        converted_content = convert_schema_keys(content)

        # Write back
        schema_file.write_text(converted_content)
        print(f"✓ Converted {schema_file.name}")


if __name__ == "__main__":
    main()
