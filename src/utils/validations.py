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
            key, validator, msg = parse_validation_error(error)

            # Print preProcessor name in case of options error
            if (
                len(error.path) > 2
                and error.path[0] == "preProcessors"
                and isinstance(error.path[1], int)
            ):
                preProcessorIndex = error.path[1]
                preProcessors = json_data.get("preProcessors", [])
                preProcessorName = f"index_{preProcessorIndex}"

                if (
                    isinstance(preProcessors, list)
                    and 0 <= preProcessorIndex < len(preProcessors)
                    and isinstance(preProcessors[preProcessorIndex], dict)
                ):
                    preProcessorName = preProcessors[preProcessorIndex].get(
                        "name", preProcessorName
                    )
                remainingTokens = list(error.path[2:])
                remainingSegments = []

                for remainingToken in remainingTokens:
                    if isinstance(remainingToken, int):
                        if len(remainingSegments) == 0:
                            remainingSegments.append(f"[{remainingToken}]")
                        else:
                            remainingSegments[-1] = (
                                f"{remainingSegments[-1]}[{remainingToken}]"
                            )
                    else:
                        remainingSegments.append(str(remainingToken))

                remainingPath = ".".join(remainingSegments)
                finalPath = (
                    f"preProcessors.{preProcessorName}.{remainingPath}"
                    if remainingPath
                    else f"preProcessors.{preProcessorName}"
                )
                table.add_row(key, msg)
                table.add_row(
                    finalPath, msg
                )
            elif validator == "required":
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


def parse_validation_error(error):
    return (format_json_path(error.path), error.validator, error.message)


def format_json_path(path):
    path_tokens = list(path)

    if len(path_tokens) == 0:
        return "$root"

    formatted_path = []
    for path_token in path_tokens:
        if isinstance(path_token, int):
            formatted_path.append(f"[{path_token}]")
        elif len(formatted_path) == 0:
            formatted_path.append(str(path_token))
        else:
            formatted_path.append(f".{path_token}")

    return "".join(formatted_path)
