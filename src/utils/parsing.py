import re
from copy import deepcopy
from fractions import Fraction

import numpy as np
from deepmerge import Merger
from dotmap import DotMap

from src.schemas.constants import FIELD_STRING_REGEX_GROUPS
from src.schemas.defaults import CONFIG_DEFAULTS, TEMPLATE_DEFAULTS
from src.schemas.defaults.evaluation import EVALUATION_CONFIG_DEFAULTS
from src.utils.constants import FIELD_LABEL_NUMBER_REGEX, SUPPORTED_PROCESSOR_NAMES
from src.utils.file import load_json
from src.utils.validations import (
    validate_config_json,
    validate_evaluation_json,
    validate_template_json,
)

OVERRIDE_MERGER = Merger(
    # pass in a list of tuples,with the
    # strategies you are looking to apply
    # to each type.
    [
        # (list, ["prepend"]),
        (dict, ["merge"])
    ],
    # next, choose the fallback strategies,
    # applied to all other types:
    ["override"],
    # finally, choose the strategies in
    # the case where the types conflict:
    ["override"],
)


def open_config_with_defaults(config_path):
    user_tuning_config = load_json(config_path)
    user_tuning_config = OVERRIDE_MERGER.merge(
        deepcopy(CONFIG_DEFAULTS), user_tuning_config
    )

    validate_config_json(user_tuning_config, config_path)

    # Broadcast the default boolean into preprocessor-wise boolean
    show_preprocessors_diff = user_tuning_config["outputs"]["show_preprocessors_diff"]
    if type(show_preprocessors_diff) == bool:
        user_tuning_config["outputs"]["show_preprocessors_diff"] = {
            processor_name: show_preprocessors_diff
            for processor_name in SUPPORTED_PROCESSOR_NAMES
        }

    # https://github.com/drgrib/dotmap/issues/74
    return DotMap(user_tuning_config, _dynamic=False)


def open_template_with_defaults(template_path):
    user_template = load_json(template_path)
    user_template = OVERRIDE_MERGER.merge(deepcopy(TEMPLATE_DEFAULTS), user_template)
    validate_template_json(user_template, template_path)
    return user_template


def open_evaluation_with_defaults(evaluation_path):
    user_evaluation_config = load_json(evaluation_path)
    user_evaluation_config = OVERRIDE_MERGER.merge(
        deepcopy(EVALUATION_CONFIG_DEFAULTS), user_evaluation_config
    )
    validate_evaluation_json(user_evaluation_config, evaluation_path)
    return user_evaluation_config


def parse_fields(key, fields):
    parsed_fields = []
    fields_set = set()
    for field_string in fields:
        fields_array = parse_field_string(field_string)
        current_set = set(fields_array)
        if not fields_set.isdisjoint(current_set):
            raise Exception(
                f"Given field string '{field_string}' has overlapping field(s) with other fields in '{key}': {fields}"
            )
        fields_set.update(current_set)
        parsed_fields.extend(fields_array)
    return parsed_fields


def parse_field_string(field_string):
    if "." in field_string:
        field_prefix, start, end = re.findall(FIELD_STRING_REGEX_GROUPS, field_string)[
            0
        ]
        start, end = int(start), int(end)
        if start >= end:
            raise Exception(
                f"Invalid range in fields string: '{field_string}', start: {start} is not less than end: {end}"
            )
        return [
            f"{field_prefix}{field_number}" for field_number in range(start, end + 1)
        ]
    else:
        return [field_string]


def custom_sort_output_columns(field_label):
    label_prefix, label_suffix = re.findall(FIELD_LABEL_NUMBER_REGEX, field_label)[0]
    return [label_prefix, int(label_suffix) if len(label_suffix) > 0 else 0, 0]


def parse_float_or_fraction(result):
    if type(result) == str and "/" in result:
        result = float(Fraction(result))
    else:
        result = float(result)
    return result


def default_dump(obj):
    return (
        bool(obj)
        if isinstance(obj, np.bool_)
        else (
            obj.to_json()
            if hasattr(obj, "to_json")
            else obj.__dict__
            if hasattr(obj, "__dict__")
            else obj
        )
    )
