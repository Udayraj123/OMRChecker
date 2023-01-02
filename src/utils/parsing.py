from copy import deepcopy

from deepmerge import Merger
from dotmap import DotMap

from src.defaults import CONFIG_DEFAULTS, EVALUATION_DEFAULTS, TEMPLATE_DEFAULTS
from src.logger import logger
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


def get_concatenated_response(omr_response, template):
    concatenated_response = {}

    # TODO: get correct local/global emptyVal here for each question
    # symbol for absent response
    unmarked_symbol = ""

    # Multi-column/multi-row questions which need to be concatenated
    for q_no, resp_keys in template.concatenations.items():
        concatenated_response[q_no] = "".join(
            [omr_response.get(k, unmarked_symbol) for k in resp_keys]
        )

    # Single-column/single-row questions
    for q_no in template.singles:
        concatenated_response[q_no] = omr_response.get(q_no, unmarked_symbol)

    # Note: concatenations and singles together should be mutually exclusive
    # and should cover all questions in the template(exhaustive)
    # TODO: ^add a warning if omr_response has unused keys remaining
    return concatenated_response


def evaluate_concatenated_response(
    concatenated_response, evaluation_config, should_explain_scoring=False
):
    if should_explain_scoring:
        pass
    # todo: code for evaluation

    return 0


def open_config_with_defaults(config_path):
    user_tuning_config = load_json(config_path)
    user_tuning_config = OVERRIDE_MERGER.merge(
        deepcopy(CONFIG_DEFAULTS), user_tuning_config
    )
    is_valid = validate_config_json(user_tuning_config, config_path)

    if is_valid:
        return DotMap(user_tuning_config)
    else:
        logger.critical("\nExiting program")
        exit()


def open_template_with_defaults(template_path):
    user_template = load_json(template_path)
    user_template = OVERRIDE_MERGER.merge(deepcopy(TEMPLATE_DEFAULTS), user_template)
    is_valid = validate_template_json(user_template, template_path)

    if is_valid:
        return user_template
    else:
        logger.critical("\nExiting program")
        exit()


def open_evaluation_with_defaults(evaluation_path):
    user_evaluation_config = load_json(evaluation_path)
    user_evaluation_config = OVERRIDE_MERGER.merge(
        deepcopy(EVALUATION_DEFAULTS), user_evaluation_config
    )
    is_valid = validate_evaluation_json(user_evaluation_config, evaluation_path)

    if is_valid:
        return user_evaluation_config
    else:
        logger.critical("\nExiting program")
        exit()
