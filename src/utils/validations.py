"""
 OMRChecker
 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123
"""

import re

import jsonschema
from jsonschema import validate
from rich.table import Table

from src.schemas import SCHEMA_JSONS, SCHEMA_VALIDATORS
from src.utils.logger import console, logger


def to_camel_case(text: str) -> str:
    """Convert a string to camelCase format."""
    # Handle snake_case
    if "_" in text:
        components = text.split("_")
        return components[0].lower() + "".join(x.title() for x in components[1:])
    # Handle space-separated or kebab-case
    if " " in text or "-" in text:
        components = re.split(r"[\s\-]+", text)
        return components[0].lower() + "".join(x.title() for x in components[1:])
    # Already might be camelCase or single word
    if text and text[0].isupper():
        return text[0].lower() + text[1:]
    return text


def suggest_camel_case(key: str) -> str | None:
    """Suggest camelCase version if the key is not in camelCase format."""
    camel_version = to_camel_case(key)
    # Check if conversion changed the key (meaning it wasn't in camelCase)
    if camel_version != key:
        return camel_version
    return None


def get_camel_case_hint(key: str) -> str:
    """Get a hint about camelCase if the key appears to be incorrectly formatted."""
    suggestion = suggest_camel_case(key)
    if suggestion:
        return f" Did you mean '{suggestion}'?"
    return ""


def validate_evaluation_json(json_data, evaluation_path) -> None:
    logger.info(f"Loading evaluation.json: {evaluation_path}")
    try:
        validate(instance=json_data, schema=SCHEMA_JSONS["evaluation"])
    except jsonschema.exceptions.ValidationError as _err:
        table = Table(show_lines=True)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Error", style="magenta")

        errors = sorted(
            SCHEMA_VALIDATORS["evaluation"].iter_errors(json_data),
            key=lambda e: e.path,
        )
        for error in errors:
            key, validator, msg = parse_validation_error(error)
            if validator == "required":
                required_property = re.findall(r"'(.*?)'", msg)[0]
                hint = get_camel_case_hint(required_property)
                table.add_row(
                    f"{key}.{required_property}",
                    f"{msg}. Check spelling and use camelCase.{hint}",
                )
            elif validator == "additionalProperties":
                # Extract the invalid property name from the error message
                invalid_props = re.findall(r"'([^']+)'", msg)
                if invalid_props:
                    hints = [
                        f"'{p}' -> '{suggest_camel_case(p)}'"
                        for p in invalid_props
                        if suggest_camel_case(p)
                    ]
                    if hints:
                        msg = f"{msg}. Suggestions: {', '.join(hints)}"
                table.add_row(key, msg)
            else:
                if (
                    key in {"outputs_configuration", "marking_schemes"}
                    and len(error.path) > 1
                ):
                    path = ".".join(list(map(str, error.path)))
                    key = path
                    if "color" in path and validator == "oneOf":
                        color = re.findall(r"'(.*?)'", msg)[0]
                        msg = f"{color} is not a valid color."

                table.add_row(key, msg)
        console.print(table, justify="center")
        msg = f"Provided Evaluation JSON is Invalid: '{evaluation_path}'"
        raise Exception(msg) from None


def validate_template_json(json_data, template_path) -> None:
    logger.info(f"Loading template.json: {template_path}")
    try:
        validate(instance=json_data, schema=SCHEMA_JSONS["template"])
    except jsonschema.exceptions.ValidationError as _err:
        table = Table(show_lines=True)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Error", style="magenta")

        errors = sorted(
            SCHEMA_VALIDATORS["template"].iter_errors(json_data),
            key=lambda e: e.path,
        )
        min_path_length = 2
        for error in errors:
            key, validator, msg = parse_validation_error(error)

            # Print preProcessor name in case of options error
            if key == "preProcessors" and len(error.path) > min_path_length:
                pre_processor_json = json_data["preProcessors"][error.path[1]]
                pre_processor_name = pre_processor_json.get("name", "UNKNOWN")
                pre_processor_key = error.path[2]
                key = f"{key}.{pre_processor_name}.{pre_processor_key}"
            elif validator == "required":
                required_property = re.findall(r"'(.*?)'", msg)[0]
                key = f"{key}.{required_property}"
                hint = get_camel_case_hint(required_property)
                msg = f"{msg}. Check spelling and use camelCase.{hint}"
            elif validator == "additionalProperties":
                # Extract the invalid property name from the error message
                invalid_props = re.findall(r"'([^']+)'", msg)
                if invalid_props:
                    hints = [
                        f"'{p}' -> '{suggest_camel_case(p)}'"
                        for p in invalid_props
                        if suggest_camel_case(p)
                    ]
                    if hints:
                        msg = f"{msg}. Suggestions: {', '.join(hints)}"
            table.add_row(key, msg)
        console.print(table, justify="center")
        msg = f"Provided Template JSON is Invalid: '{template_path}'"
        raise Exception(msg) from None


def validate_config_json(json_data, config_path) -> None:
    logger.info(f"Loading config.json: {config_path}")
    try:
        validate(instance=json_data, schema=SCHEMA_JSONS["config"])
    except jsonschema.exceptions.ValidationError as _err:
        table = Table(show_lines=True)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Error", style="magenta")
        errors = sorted(
            SCHEMA_VALIDATORS["config"].iter_errors(json_data),
            key=lambda e: e.path,
        )
        for error in errors:
            key, validator, msg = parse_validation_error(error)

            if validator == "required":
                required_property = re.findall(r"'(.*?)'", msg)[0]
                hint = get_camel_case_hint(required_property)
                table.add_row(
                    f"{key}.{required_property}",
                    f"{msg}. Check spelling and use camelCase.{hint}",
                )
            elif validator == "additionalProperties":
                # Extract the invalid property name from the error message
                invalid_props = re.findall(r"'([^']+)'", msg)
                if invalid_props:
                    hints = [
                        f"'{p}' -> '{suggest_camel_case(p)}'"
                        for p in invalid_props
                        if suggest_camel_case(p)
                    ]
                    if hints:
                        msg = f"{msg}. Suggestions: {', '.join(hints)}"
                table.add_row(key, msg)
            else:
                table.add_row(key, msg)
        console.print(table, justify="center")
        msg = f"Provided config JSON is Invalid: '{config_path}'"
        raise Exception(msg) from None


def parse_validation_error(error) -> tuple[str, str, str]:
    return (
        (error.path[0] if len(error.path) > 0 else "$root"),
        error.validator,
        error.message,
    )
