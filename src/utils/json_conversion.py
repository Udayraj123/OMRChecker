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
