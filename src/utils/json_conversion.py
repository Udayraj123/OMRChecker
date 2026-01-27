"""JSON key conversion utilities for camelCase ↔ snake_case conversion.

This module provides utilities to convert between camelCase (used in JSON)
and snake_case (used in Python code), enabling a clean separation between
external API conventions and internal Python conventions.
"""

import re
from typing import Any


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case.

    Args:
        name: String in camelCase format

    Returns:
        String in snake_case format

    Examples:
        >>> camel_to_snake("showImageLevel")
        'show_image_level'
        >>> camel_to_snake("MLConfig")
        'ml_config'
        >>> camel_to_snake("globalPageThreshold")
        'global_page_threshold'
    """
    # Handle acronyms at the start (e.g., "MLConfig" -> "ml_config")
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    # Insert underscore before uppercase letters (e.g., "camelCase" -> "camel_Case")
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase.

    Args:
        name: String in snake_case format

    Returns:
        String in camelCase format

    Examples:
        >>> snake_to_camel("show_image_level")
        'showImageLevel'
        >>> snake_to_camel("ml_config")
        'mlConfig'
        >>> snake_to_camel("global_page_threshold")
        'globalPageThreshold'
    """
    components = name.split("_")
    # Keep first component as-is, capitalize the rest
    return components[0] + "".join(x.title() for x in components[1:])


def screaming_to_camel(name: str) -> str:
    """Convert SCREAMING_SNAKE_CASE to camelCase.

    Args:
        name: String in SCREAMING_SNAKE_CASE format

    Returns:
        String in camelCase format

    Examples:
        >>> screaming_to_camel("GLOBAL_PAGE_THRESHOLD")
        'globalPageThreshold'
        >>> screaming_to_camel("MIN_JUMP")
        'minJump'
    """
    return snake_to_camel(name.lower())


def validate_no_key_clash(data: dict[str, Any], path: str = "") -> None:
    """Validate that a dictionary has no keys that would clash after case conversion.

    This checks if both camelCase and snake_case versions of the same logical key exist,
    which would cause data loss or confusion during conversion.

    Args:
        data: Dictionary to validate
        path: Current path in nested structure (for error messages)

    Raises:
        ValueError: If clashing keys are found

    Examples:
        >>> validate_no_key_clash({"userName": "Alice", "user_name": "Bob"})
        Traceback (most recent call last):
        ...
        ValueError: Key clash detected: 'userName' and 'user_name' both convert to 'user_name'

        >>> validate_no_key_clash({"userName": "Alice", "email": "test@example.com"})
        # No error - keys don't clash
    """
    if not isinstance(data, dict):
        return

    # Build a mapping of converted keys to original keys
    snake_to_original = {}

    for key in data.keys():
        snake_key = camel_to_snake(key)

        if snake_key in snake_to_original:
            original_key = snake_to_original[snake_key]
            if original_key != key:
                prefix = f"at '{path}': " if path else ""
                raise ValueError(
                    f"{prefix}Key clash detected: '{original_key}' and '{key}' "
                    f"both convert to '{snake_key}'. Please use only one naming convention."
                )
        else:
            snake_to_original[snake_key] = key

    # Recursively validate nested structures
    for key, value in data.items():
        current_path = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            validate_no_key_clash(value, current_path)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    validate_no_key_clash(item, f"{current_path}[{i}]")


def convert_dict_keys_to_snake(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert dictionary keys from camelCase to snake_case.

    Args:
        data: Dictionary with camelCase keys

    Returns:
        Dictionary with snake_case keys
    """
    if not isinstance(data, dict):
        return data

    result = {}
    for key, value in data.items():
        # Convert the key
        snake_key = camel_to_snake(key)

        # Recursively process the value
        if isinstance(value, dict):
            result[snake_key] = convert_dict_keys_to_snake(value)
        elif isinstance(value, list):
            result[snake_key] = [
                convert_dict_keys_to_snake(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[snake_key] = value

    return result


def convert_dict_keys_to_camel(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert dictionary keys from snake_case to camelCase.

    Args:
        data: Dictionary with snake_case keys

    Returns:
        Dictionary with camelCase keys
    """
    if not isinstance(data, dict):
        return data

    result = {}
    for key, value in data.items():
        # Convert the key
        camel_key = snake_to_camel(key)

        # Recursively process the value
        if isinstance(value, dict):
            result[camel_key] = convert_dict_keys_to_camel(value)
        elif isinstance(value, list):
            result[camel_key] = [
                convert_dict_keys_to_camel(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[camel_key] = value

    return result
