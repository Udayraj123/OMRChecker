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


def parse_validation_error(error):
    return (
        (error.path[0] if len(error.path) > 0 else "root key"),
        error.validator,
        error.message,
    )


def validate_evaluation_json(json_data, evaluation_path, template, curr_dir):
    logger.info("Validating evaluation.json...")
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
                table.add_row(
                    re.findall(r"'(.*?)'", msg)[0],
                    msg + ". Make sure the spelling of the key is correct",
                )
            else:
                table.add_row(key, msg)
        console.print(table, justify="center")
        logger.critical(f"Provided Evaluation JSON is Invalid: '{evaluation_path}'")
        return False

    # TODO: also validate these
    # - All mentioned qNos in sections should be present in template.json
    # - All ranges in questions_order should be exhaustive too
    # - All keys of sections should be present in keys of marking
    # - Sections should be mutually exclusive

    logger.info("Evaluation JSON validated successfully")
    return True


def validate_template_json(json_data, template_path):
    logger.info("Validating template.json...")
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
        console.print(table, justify="center")
        logger.critical(f"Provided Template JSON is Invalid: '{template_path}'")
        return False

    logger.info("Template JSON validated successfully")
    return True


def validate_config_json(json_data, config_path):
    logger.info("Validating config.json...")
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
                table.add_row(
                    re.findall(r"'(.*?)'", msg)[0],
                    msg
                    + ". Make sure the spelling of the key is correct and it is in camelCase",
                )
            else:
                table.add_row(key, msg)
        console.print(table, justify="center")
        logger.critical(f"Provided config JSON is Invalid: '{config_path}'")
        return False
    logger.info("Config JSON validated successfully")
    return True
