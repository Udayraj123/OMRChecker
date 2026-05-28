"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""
import re

import jsonschema
from jsonschema import validate
from rich.table import Table

from src.logger import console, logger
from src.schemas import SCHEMA_JSONS, SCHEMA_VALIDATORS


def validate_evaluation_json(json_data, evaluation_path):
    logger.info(f"Loading evaluation.json: {evaluation_path}")
    try:
        validate(instance=json_data, schema=SCHEMA_JSONS["evaluation"])
    except jsonschema.exceptions.ValidationError as _err:  # NOQA
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
                requiredProperty = re.findall(r"'(.*?)'", msg)[0]
                table.add_row(
                    f"{key}.{requiredProperty}",
                    msg + ". Make sure the spelling of the key is correct",
                )
            else:
                table.add_row(key, msg)
        console.print(table, justify="center")
        raise Exception(
            f"Provided Evaluation JSON is Invalid: '{evaluation_path}'"
        ) from None


def validate_template_json(json_data, template_path):
    logger.info(f"Loading template.json: {template_path}")
    try:
        validate(instance=json_data, schema=SCHEMA_JSONS["template"])
    except jsonschema.exceptions.ValidationError as _err:  # NOQA
        table = Table(show_lines=True)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Error", style="magenta")

        errors = sorted(
            SCHEMA_VALIDATORS["template"].iter_errors(json_data),
            key=lambda e: e.path,
        )
        for error in errors:
            key, validator, msg = parse_validation_error(error, json_data=json_data)

            if validator == "required":
                requiredProperty = re.findall(r"'(.*?)'", msg)[0]
                table.add_row(
                    f"{key}.{requiredProperty}",
                    f"{msg}. Check for spelling errors and make sure it is in camelCase",
                )
            else:
                table.add_row(key, msg)
        console.print(table, justify="center")
        raise Exception(
            f"Provided Template JSON is Invalid: '{template_path}'"
        ) from None


def validate_config_json(json_data, config_path):
    logger.info(f"Loading config.json: {config_path}")
    try:
        validate(instance=json_data, schema=SCHEMA_JSONS["config"])
    except jsonschema.exceptions.ValidationError as _err:  # NOQA
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
                requiredProperty = re.findall(r"'(.*?)'", msg)[0]
                table.add_row(
                    f"{key}.{requiredProperty}",
                    f"{msg}. Check for spelling errors and make sure it is in camelCase",
                )
            else:
                table.add_row(key, msg)
        console.print(table, justify="center")
        raise Exception(f"Provided config JSON is Invalid: '{config_path}'") from None


def parse_validation_error(error, json_data=None):
    return (
        format_json_path(error.path, json_data=json_data),
        error.validator,
        error.message,
    )


def format_nested_path_tokens(path_tokens):
    nested_path_segments = []

    for path_token in path_tokens:
        if isinstance(path_token, int):
            if len(nested_path_segments) == 0:
                nested_path_segments.append(f"[{path_token}]")
            else:
                nested_path_segments[-1] = f"{nested_path_segments[-1]}[{path_token}]"
        else:
            nested_path_segments.append(str(path_token))

    return ".".join(nested_path_segments)


def format_json_path(path, json_data=None):
    path_tokens = list(path)

    if len(path_tokens) == 0:
        return "$root"

    if (
        json_data is not None
        and len(path_tokens) > 1
        and path_tokens[0] == "preProcessors"
        and isinstance(path_tokens[1], int)
    ):
        preprocessor_index = path_tokens[1]
        preprocessors = json_data.get("preProcessors", [])
        preprocessor_name = f"index_{preprocessor_index}"

        if isinstance(preprocessors, list) and 0 <= preprocessor_index < len(
            preprocessors
        ):
            preprocessor_name = preprocessors[preprocessor_index].get(
                "name", preprocessor_name
            )

        remaining_path = format_nested_path_tokens(path_tokens[2:])
        return (
            f"preProcessors.{preprocessor_name}.{remaining_path}"
            if remaining_path
            else f"preProcessors.{preprocessor_name}"
        )

    formatted_path = []
    for path_token in path_tokens:
        if isinstance(path_token, int):
            formatted_path.append(f"[{path_token}]")
        elif len(formatted_path) == 0:
            formatted_path.append(str(path_token))
        else:
            formatted_path.append(f".{path_token}")

    return "".join(formatted_path)
