import re
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from deepmerge import Merger

from src.utils.exceptions import OMRCheckerError
from src.schemas.constants import FIELD_STRING_REGEX_GROUPS
from src.schemas.models.config import Config
from src.schemas.models.evaluation import EvaluationConfig
from src.schemas.models.template import TemplateConfig
from src.utils.constants import FIELD_LABEL_NUMBER_REGEX, SUPPORTED_PROCESSOR_NAMES
from src.utils.file import load_json
from src.utils.validations import (
    validate_config_json,
    validate_evaluation_json,
    validate_template_json,
)
from src.utils.json_conversion import (
    convert_dict_keys_to_camel,
    validate_no_key_clash,
)


OVERRIDE_MERGER = Merger(
    # pass in a list of tuples,with the
    # strategies you are looking to apply
    # to each type.
    [(dict, ["merge"])],
    # next, choose the fallback strategies,
    # applied to all other types:
    ["override"],
    # finally, choose the strategies in
    # the case where the types conflict:
    ["override"],
)


def open_config_with_defaults(config_path: Path, args: dict[str, Any]) -> Config:
    """Load and merge configuration from file with defaults.

    Args:
        config_path: Path to the config.json file
        args: Command line arguments containing outputMode and debug flags

    Returns:
        Config dataclass instance with merged configuration

    Raises:
        ValueError: If the config JSON has clashing keys (both camelCase and snake_case)
    """
    output_mode = args["outputMode"]
    debug_mode = args["debug"]
    user_tuning_config = load_json(config_path)

    # Validate user config BEFORE merging with defaults
    validate_config_json(user_tuning_config, config_path)

    # Validate no key clashes before conversion
    try:
        validate_no_key_clash(user_tuning_config)
    except ValueError as e:
        raise ValueError(f"Invalid config JSON at {config_path}: {e}") from e

    defaults_from_args = {
        "outputs": {
            "output_mode": output_mode,
            "show_logs_by_type": {
                "debug": debug_mode,
            },
        }
    }
    # Note: precedence: file > args > dataclass defaults
    user_tuning_config = OVERRIDE_MERGER.merge(defaults_from_args, user_tuning_config)
    # Create default config instance and convert to dict for merging
    defaults_dict = Config().to_dict()
    user_tuning_config = OVERRIDE_MERGER.merge(
        deepcopy(defaults_dict), user_tuning_config
    )

    # Inject config path
    user_tuning_config["path"] = str(config_path)

    # Broadcast the default boolean into preprocessor-wise boolean
    show_preprocessors_diff = user_tuning_config["outputs"]["show_preprocessors_diff"]
    if isinstance(show_preprocessors_diff, bool):
        user_tuning_config["outputs"]["show_preprocessors_diff"] = dict.fromkeys(
            SUPPORTED_PROCESSOR_NAMES, show_preprocessors_diff
        )

    # Convert merged dict to Config dataclass
    return Config.from_dict(user_tuning_config)


def open_template_with_defaults(template_path: Path) -> "TemplateConfig":
    """Load and merge template configuration from file with defaults.

    Args:
        template_path: Path to the template.json file

    Returns:
        TemplateConfig dataclass instance with merged configuration

    Raises:
        ValueError: If the template JSON has clashing keys (both camelCase and snake_case)
    """
    user_template = load_json(template_path)

    # Validate user template BEFORE merging with defaults
    validate_template_json(user_template, template_path)

    # Validate no key clashes before conversion
    try:
        validate_no_key_clash(user_template)
    except ValueError as e:
        raise ValueError(f"Invalid template JSON at {template_path}: {e}") from e

    # Note: Keep user_template in camelCase for merging
    # Create default template instance and convert to dict for merging
    defaults_dict = TemplateConfig().to_dict()
    # Convert defaults dict back to camelCase for proper merging with user JSON
    defaults_camel = convert_dict_keys_to_camel(defaults_dict)

    merged_template = OVERRIDE_MERGER.merge(deepcopy(defaults_camel), user_template)

    # Convert merged dict to TemplateConfig dataclass (from_dict handles conversion)
    return TemplateConfig.from_dict(merged_template)


def open_evaluation_with_defaults(evaluation_path: Path) -> dict[str, Any]:
    """Load and merge evaluation configuration from file with defaults.

    Args:
        evaluation_path: Path to the evaluation.json file

    Returns:
        Dictionary representation of evaluation configuration

    Note:
        Returns dict for backward compatibility with existing code that expects
        dict-like access. Uses EvaluationConfig dataclass internally for validation.

    Raises:
        ValueError: If the evaluation JSON has clashing keys (both camelCase and snake_case)
    """

    user_evaluation_config = load_json(evaluation_path)

    # Validate user evaluation BEFORE merging with defaults
    validate_evaluation_json(user_evaluation_config, evaluation_path)

    # Validate no key clashes before conversion
    try:
        validate_no_key_clash(user_evaluation_config)
    except ValueError as e:
        raise ValueError(f"Invalid evaluation JSON at {evaluation_path}: {e}") from e

    # Use EvaluationConfig.from_dict to properly convert keys while preserving
    # user-defined marking scheme names and option values
    user_config = EvaluationConfig.from_dict(user_evaluation_config)

    # Create default evaluation instance for merging
    defaults = EvaluationConfig()

    # Merge user config with defaults
    defaults_dict = defaults.to_dict()
    user_dict = user_config.to_dict()
    merged_config = OVERRIDE_MERGER.merge(deepcopy(defaults_dict), user_dict)

    return merged_config


def parse_fields(key: str, fields: list[str]) -> list[str]:
    parsed_fields = []
    fields_set = set()
    for field_string in fields:
        fields_array = parse_field_string(field_string)
        current_set = set(fields_array)
        if not fields_set.isdisjoint(current_set):
            msg = f"Given field string '{field_string}' has overlapping field(s) with other fields in '{key}': {fields}"
            raise OMRCheckerError(
                msg,
                context={
                    "field_string": field_string,
                    "key": key,
                    "overlapping_fields": list(fields_set.intersection(current_set)),
                },
            )
        fields_set.update(current_set)
        parsed_fields.extend(fields_array)
    return parsed_fields


def parse_field_string(field_string) -> list[str]:
    if "." in field_string:
        field_prefix, start, end = re.findall(FIELD_STRING_REGEX_GROUPS, field_string)[
            0
        ]
        start, end = int(start), int(end)
        if start >= end:
            msg = f"Invalid range in fields string: '{field_string}', start: {start} is not less than end: {end}"
            raise OMRCheckerError(
                msg,
                context={
                    "field_string": field_string,
                    "start": start,
                    "end": end,
                },
            )
        return [
            f"{field_prefix}{field_number}" for field_number in range(start, end + 1)
        ]
    return [field_string]


def alphanumerical_sort_key(field_label: str) -> list[str | int]:
    label_prefix, label_suffix = re.findall(FIELD_LABEL_NUMBER_REGEX, field_label)[0]
    return [label_prefix, int(label_suffix) if len(label_suffix) > 0 else 0, 0]


def parse_float_or_fraction(result: str | float) -> float:
    if isinstance(result, str) and "/" in result:
        result = float(Fraction(result))
    else:
        result = float(result)
    return result


def default_dump(obj: object) -> bool | dict[str, Any] | str:
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


def table_to_df(table: object) -> pd.DataFrame:
    # ruff: noqa: SLF001
    data = {col.header: col._cells for col in table.columns}
    return pd.DataFrame(data)
