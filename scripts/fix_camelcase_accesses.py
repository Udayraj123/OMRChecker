"""Script to convert camelCase dictionary key accesses to snake_case."""

import re
from pathlib import Path


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


def convert_camelcase_dict_accesses(file_path: Path) -> None:
    """Convert all camelCase dictionary key accesses in a file to snake_case."""
    content = file_path.read_text()
    original_content = content

    # Pattern 1: Match dictionary key accesses like ["camelCaseKey"] or ['camelCaseKey']
    pattern1 = r'\[(["\'])([a-z][a-zA-Z]*[A-Z][a-zA-Z]*)\1\]'

    def replace_key1(match):
        quote = match.group(1)
        camel_key = match.group(2)
        snake_key = camel_to_snake(camel_key)
        return f"[{quote}{snake_key}{quote}]"

    content = re.sub(pattern1, replace_key1, content)

    # Pattern 2: Match .get("camelCaseKey") or .get('camelCaseKey')
    pattern2 = r'\.get\((["\'])([a-z][a-zA-Z]*[A-Z][a-zA-Z]*)\1'

    def replace_key2(match):
        quote = match.group(1)
        camel_key = match.group(2)
        snake_key = camel_to_snake(camel_key)
        return f".get({quote}{snake_key}{quote}"

    content = re.sub(pattern2, replace_key2, content)

    if content != original_content:
        file_path.write_text(content)
        print(f"Updated: {file_path}")
    else:
        print(f"No changes: {file_path}")


def main():
    """Convert all files with camelCase dict accesses."""
    files_to_fix = [
        "src/processors/image/CropOnDotLines.py",
        "src/processors/image/CropOnCustomMarkers.py",
        "src/processors/image/AutoRotate.py",
        "src/processors/evaluation/evaluation_config.py",
        "src/processors/alignment/template_alignment.py",
        "src/processors/image/CropOnPatchesCommon.py",
        "src/processors/image/CropPage.py",
        "src/processors/organization/processor.py",
    ]

    root = Path(__file__).parent.parent
    for file_path in files_to_fix:
        full_path = root / file_path
        if full_path.exists():
            convert_camelcase_dict_accesses(full_path)
        else:
            print(f"File not found: {full_path}")


if __name__ == "__main__":
    main()
