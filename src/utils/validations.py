"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""
import os
import re
from asyncio.log import logger

import jsonschema
from jsonschema import Draft202012Validator, validate
from rich.console import Console
from rich.table import Table

from src.constants import SCHEMA_NAMES, SCHEMAS_PATH
from src.utils.parsing import load_json

# Load schema files from src/schemas folder
SCHEMA_VALIDATORS, SCHEMA_JSONS = {}, {}
for schema_name in SCHEMA_NAMES:
    schema_path = os.path.join(SCHEMAS_PATH, f"{schema_name}-schema.json")
    execute_api_schema = load_json(schema_path)
    SCHEMA_JSONS[schema_name] = execute_api_schema
    SCHEMA_VALIDATORS[schema_name] = Draft202012Validator(execute_api_schema)


def validate_evaluation_json(json_data, evaluation_path):
    logger.info("Validating evaluation.json ...")
    try:
        validate(instance=json_data, schema=SCHEMA_JSONS[SCHEMA_NAMES.evaluation])
    except jsonschema.exceptions.ValidationError as err:

        table = Table(show_lines=True)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Error", style="magenta")

        errors = sorted(
            SCHEMA_VALIDATORS[SCHEMA_NAMES.evaluation].iter_errors(json_data),
            key=lambda e: e.path,
        )
        for error in errors:
            key, validator, msg = error.path[0], error.validator, error.message

            if validator == "required":
                table.add_row(
                    re.findall(r"'(.*?)'", msg)[0],
                    msg + ". Make sure the spelling of the key is correct",
                )
            else:
                table.add_row(key, msg)
        console = Console()
        console.print(table)
        err = f"Provided Evaluation JSON is Invalid: {evaluation_path}"
        return False, err

    message = "Evaluation JSON validated successfully..."
    return True, message


def validate_template_json(json_data, template_path):

    logger.info("Validating template.json ...")
    try:
        validate(instance=json_data, schema=SCHEMA_JSONS["template"])
    except jsonschema.exceptions.ValidationError as err:

        table = Table(show_lines=True)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Error", style="magenta")

        errors = sorted(
            SCHEMA_VALIDATORS["template"].iter_errors(json_data), key=lambda e: e.path
        )
        for error in errors:
            key, validator, msg = error.path[0], error.validator, error.message

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
        err = f"Provided Template JSON is Invalid: {template_path}"
        return False, err

    message = "Template JSON validated successfully..."
    return True, message
