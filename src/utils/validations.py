import re
from pathlib import Path

import jsonschema
from jsonschema import validate
from rich.table import Table

from src.exceptions import (
    ConfigValidationError,
    EvaluationValidationError,
    TemplateValidationError,
)
from src.schemas import SCHEMA_JSONS, SCHEMA_VALIDATORS
from src.utils.logger import console, logger


def validate_evaluation_json(json_data: dict, evaluation_path: Path) -> None:
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
        error_messages = []
        for error in errors:
            key, validator, msg = parse_validation_error(error)
            if validator == "required":
                required_property = re.findall(r"'(.*?)'", msg)[0]
                table.add_row(
                    f"{key}.{required_property}",
                    f"{msg}. Check for spelling errors and make sure it is in camelCase",
                )
                error_messages.append(f"{key}.{required_property}: {msg}")
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
                error_messages.append(f"{key}: {msg}")
        console.print(table, justify="center")
        raise EvaluationValidationError(
            evaluation_path, errors=error_messages
        ) from None


def validate_template_json(json_data: dict, template_path: Path) -> None:
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
        error_messages = []
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
                msg = (
                    f"{msg}. Check for spelling errors and make sure it is in camelCase"
                )
            table.add_row(key, msg)
            error_messages.append(f"{key}: {msg}")
        console.print(table, justify="center")
        raise TemplateValidationError(template_path, errors=error_messages) from None


def validate_config_json(json_data: dict, config_path: Path) -> None:
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
        error_messages = []
        for error in errors:
            key, validator, msg = parse_validation_error(error)

            if validator == "required":
                required_property = re.findall(r"'(.*?)'", msg)[0]
                table.add_row(
                    f"{key}.{required_property}",
                    f"{msg}. Check for spelling errors and make sure it is in camelCase",
                )
                error_messages.append(f"{key}.{required_property}: {msg}")
            elif (
                validator == "const"
                and len(error.path) >= 2
                and error.path[-2] == "processing"
                and error.path[-1] == "maxParallelWorkers"
                and json_data.get("outputs", {}).get("showImageLevel", 0) > 0
            ):
                # Custom message for showImageLevel > 0 with maxParallelWorkers > 1
                custom_msg = "When showImageLevel > 0 (interactive mode), maxParallelWorkers must be 1. Parallel processing is not compatible with interactive image display."
                table.add_row(
                    "processing.maxParallelWorkers",
                    custom_msg,
                )
                error_messages.append(f"processing.maxParallelWorkers: {custom_msg}")
            else:
                table.add_row(key, msg)
                error_messages.append(f"{key}: {msg}")
        console.print(table, justify="center")
        raise ConfigValidationError(config_path, errors=error_messages) from None


def parse_validation_error(
    error: jsonschema.exceptions.ValidationError,
) -> tuple[str, str, str]:
    return (
        (error.path[0] if len(error.path) > 0 else "$root"),
        error.validator,
        error.message,
    )
