"""Script to comprehensively convert camelCase to snake_case in processor files."""

import re
from pathlib import Path


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


def convert_all_camelcase_keys(file_path: Path) -> None:
    """Convert ALL camelCase occurrences to snake_case in dict contexts."""
    content = file_path.read_text()
    original_content = content

    # Pattern 1: ["camelCaseKey"] or ['camelCaseKey']
    pattern1 = r'\[(["\'])([a-z][a-zA-Z]*[A-Z][a-zA-Z]*)\1\]'
    content = re.sub(
        pattern1,
        lambda m: f"[{m.group(1)}{camel_to_snake(m.group(2))}{m.group(1)}]",
        content,
    )

    # Pattern 2: .get("camelCaseKey") or .get('camelCaseKey')
    pattern2 = r'\.get\((["\'])([a-z][a-zA-Z]*[A-Z][a-zA-Z]*)\1'
    content = re.sub(
        pattern2,
        lambda m: f".get({m.group(1)}{camel_to_snake(m.group(2))}{m.group(1)}",
        content,
    )

    # Pattern 3: Dictionary literal keys: "camelCaseKey": (at start of line or after {, comma)
    pattern3 = r'([\{\,]\s*)(["\'])([a-z][a-zA-Z]*[A-Z][a-zA-Z]*)\2\s*:'
    content = re.sub(
        pattern3,
        lambda m: f"{m.group(1)}{m.group(2)}{camel_to_snake(m.group(3))}{m.group(2)}:",
        content,
    )

    if content != original_content:
        file_path.write_text(content)
        print(f"Updated: {file_path}")
        return True
    else:
        print(f"No changes: {file_path}")
        return False


def main():
    """Convert all processor files comprehensively."""
    files_to_fix = [
        "src/processors/image/crop_on_patches/dot_lines.py",
        "src/processors/image/crop_on_patches/custom_markers.py",
        "src/processors/image/AutoRotate.py",
        "src/processors/evaluation/evaluation_config.py",
        "src/processors/alignment/template_alignment.py",
        "src/processors/image/crop_on_patches/common.py",
        "src/processors/image/CropPage.py",
        "src/processors/organization/processor.py",
        "src/processors/image/WarpOnPointsCommon.py",
    ]

    root = Path(__file__).parent.parent
    for file_path in files_to_fix:
        full_path = root / file_path
        if full_path.exists():
            convert_all_camelcase_keys(full_path)
        else:
            print(f"File not found: {full_path}")


if __name__ == "__main__":
    main()
