"""Script to migrate all sample JSON files to camelCase keys.

This script systematically converts JSON files while:
- Converting snake_case keys to camelCase
- Converting SCREAMING_SNAKE_CASE config values to camelCase
- Keeping enum/constant values as SCREAMING_SNAKE_CASE (like "DEFAULT", "BUBBLES_THRESHOLD")
- Preserving formatting and structure
"""

import json
from pathlib import Path
from typing import Any


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def screaming_to_camel(name: str) -> str:
    """Convert SCREAMING_SNAKE_CASE to camelCase."""
    return snake_to_camel(name.lower())


# Enum-like keys that should stay as SCREAMING_SNAKE_CASE
ENUM_KEYS = {
    "DEFAULT",  # Default marking scheme key
    "BUBBLES_THRESHOLD",
    "OCR",
    "BARCODE_QR",  # Field detection types
    "FOUR_MARKERS",
    "ONE_LINE_TWO_DOTS",
    "TWO_DOTS_ONE_LINE",
    "TWO_LINES",
    "FOUR_DOTS",  # Marker types
    "QTYPE_MCQ4",
    "QTYPE_MCQ5",
    "QTYPE_INT",  # Built-in bubble field types
    "CUSTOM",
    "CENTERS",
    "INNER_WIDTHS",
    "INNER_HEIGHTS",
    "INNER_CORNERS",
    "OUTER_CORNERS",  # Selectors
}

# Keys within specific contexts that should stay as SCREAMING_SNAKE_CASE
CONSTANT_VALUE_KEYS = {
    "fieldDetectionType",  # Values like "BUBBLES_THRESHOLD"
    "bubbleFieldType",  # Values like "QTYPE_MCQ4"
    "type",  # Processor types like "FOUR_MARKERS"
    "zonePreset",  # Zone presets
    "selector",  # Selector types
}

# Config section SCREAMING_SNAKE keys that should become camelCase
CONFIG_SCREAMING_KEYS = {
    "GAMMA_LOW",
    "MIN_GAP_TWO_BUBBLES",
    "MIN_JUMP",
    "MIN_JUMP_STD",
    "CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY",
    "MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK",
    "GLOBAL_THRESHOLD_MARGIN",
    "JUMP_DELTA",
    "JUMP_DELTA_STD",
    "GLOBAL_PAGE_THRESHOLD",
    "GLOBAL_PAGE_THRESHOLD_STD",
}


def should_convert_key(key: str, parent_key: str = None) -> bool:
    """Determine if a key should be converted to camelCase."""
    # Don't convert if it's an enum/constant key
    if key in ENUM_KEYS:
        return False

    # Don't convert custom field type keys (they start with CUSTOM_)
    if key.startswith("CUSTOM_"):
        return False

    # Convert SCREAMING_SNAKE config keys
    if key in CONFIG_SCREAMING_KEYS:
        return True

    # Convert snake_case keys
    if "_" in key and key.islower():
        return True

    return False


def convert_dict_keys(data: Any, parent_key: str = None) -> Any:
    """Recursively convert dictionary keys based on context."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # Determine new key
            if should_convert_key(key, parent_key):
                if key in CONFIG_SCREAMING_KEYS:
                    new_key = screaming_to_camel(key)
                else:
                    new_key = snake_to_camel(key)
            else:
                new_key = key

            # Recursively process the value
            result[new_key] = convert_dict_keys(value, new_key)
        return result
    elif isinstance(data, list):
        return [convert_dict_keys(item, parent_key) for item in data]
    else:
        return data


def migrate_json_file(file_path: Path) -> bool:
    """Migrate a single JSON file to camelCase keys.

    Returns:
        True if file was modified, False otherwise
    """
    try:
        # Read the original file
        with open(file_path, "r", encoding="utf-8") as f:
            original_content = f.read()
            data = json.loads(original_content)

        # Convert keys
        converted_data = convert_dict_keys(data)

        # Write back with same formatting (2-space indent)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
            f.write("\n")  # Add trailing newline

        return True
    except json.JSONDecodeError as e:
        print(f"✗ JSON decode error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False


def main():
    """Migrate all sample JSON files."""
    base_dir = Path("/Users/udayraj.deshmukh/Personals/OMRChecker")
    samples_dir = base_dir / "samples"

    # Find all JSON files in samples directory
    json_files = list(samples_dir.rglob("*.json"))

    print(f"Found {len(json_files)} JSON files in {samples_dir}")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for json_file in sorted(json_files):
        relative_path = json_file.relative_to(base_dir)
        if migrate_json_file(json_file):
            print(f"✓ {relative_path}")
            success_count += 1
        else:
            error_count += 1

    print("=" * 60)
    print(f"Summary: {success_count} files converted, {error_count} errors")

    if error_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
