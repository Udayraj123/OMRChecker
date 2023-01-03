"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""
import os
import re

import jsonschema
from jsonschema import Draft202012Validator, validate
from rich.console import Console
from rich.table import Table

from src.constants import SCHEMA_NAMES, SCHEMAS_PATH
from src.logger import logger
from src.utils.file import load_json

# Load schema files from src/schemas folder
SCHEMA_VALIDATORS, SCHEMA_JSONS = {}, {}
for schema_name in SCHEMA_NAMES:
    schema_path = os.path.join(SCHEMAS_PATH, f"{schema_name}-schema.json")
    execute_api_schema = load_json(schema_path)
    SCHEMA_JSONS[schema_name] = execute_api_schema
    SCHEMA_VALIDATORS[schema_name] = Draft202012Validator(execute_api_schema)


def parse_validation_error(error):
    return (
        (error.path[0] if len(error.path) > 0 else "root"),
        error.validator,
        error.message,
    )


def validate_evaluation_json(json_data, evaluation_path):
    logger.info("Validating evaluation.json...")
    try:
        validate(instance=json_data, schema=SCHEMA_JSONS[SCHEMA_NAMES.evaluation])
    except jsonschema.exceptions.ValidationError as _err:  # NOQA

        table = Table(show_lines=True)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Error", style="magenta")

        errors = sorted(
            SCHEMA_VALIDATORS[SCHEMA_NAMES.evaluation].iter_errors(json_data),
            key=lambda e: e.path,
        )
        for error in errors:
            key, validator, msg = parse_validation_error(error)
            if validator == "required":
                table.add_row(
                    re.findall(r"'(.*?)'", msg)[0],
                    msg + ". Make sure the spelling of the key is correct",
                )
            else:
                table.add_row(key, msg)
        console = Console()
        console.print(table)
        logger.critical(f"Provided Evaluation JSON is Invalid: {evaluation_path}")
        return False

    logger.info("Evaluation JSON validated successfully")
    return True


def validate_template_json(json_data, template_path):
    logger.info("Validating template.json...")
    try:
        validate(instance=json_data, schema=SCHEMA_JSONS[SCHEMA_NAMES.template])
    except jsonschema.exceptions.ValidationError as _err:  # NOQA

        table = Table(show_lines=True)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Error", style="magenta")

        errors = sorted(
            SCHEMA_VALIDATORS[SCHEMA_NAMES.template].iter_errors(json_data),
            key=lambda e: e.path,
        )
        for error in errors:
            key, validator, msg = parse_validation_error(error)

            # Print preProcessor name in case of options error
            if key == "preProcessors":
                preProcessorName = json_data["preProcessors"][error.path[1]]["name"]
                preProcessorKey = error.path[2]
                table.add_row(f"{key}.{preProcessorName}.{preProcessorKey}", msg)
            elif validator == "required":
                table.add_row(
                    re.findall(r"'(.*?)'", msg)[0],
                    msg
                    + ". Make sure the spelling of the key is correct and it is in camelCase",
                )
            else:
                table.add_row(key, msg)
        console = Console()
        console.print(table)
        logger.critical(f"Provided Template JSON is Invalid: {template_path}")
        return False

    logger.info("Template JSON validated successfully")
    return True


def validate_config_json(json_data, config_path):
    logger.info("Validating config.json...")
    try:
        validate(instance=json_data, schema=SCHEMA_JSONS[SCHEMA_NAMES.config])
    except jsonschema.exceptions.ValidationError as _err:  # NOQA
        table = Table(show_lines=True)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Error", style="magenta")
        errors = sorted(
            SCHEMA_VALIDATORS[SCHEMA_NAMES.config].iter_errors(json_data),
            key=lambda e: e.path,
        )
        for error in errors:
            key, validator, msg = parse_validation_error(error)

            if validator == "required":
                table.add_row(
                    re.findall(r"'(.*?)'", msg)[0],
                    msg
                    + ". Make sure the spelling of the key is correct and it is in camelCase",
                )
            else:
                table.add_row(key, msg)
        console = Console()
        console.print(table)
        logger.critical(f"Provided config JSON is Invalid: {config_path}")
        return False
    logger.info("Config JSON validated successfully")
    return True
