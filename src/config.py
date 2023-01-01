"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

from copy import deepcopy

from dotmap import DotMap

from src.defaults.config import CONFIG_DEFAULTS
from src.logger import logger
from src.utils.parsing import OVERRIDE_MERGER, load_json
from src.utils.validations import validate_config_json


def open_config_with_defaults(config_path):
    user_config = load_json(config_path)
    user_config = OVERRIDE_MERGER.merge(deepcopy(CONFIG_DEFAULTS), user_config)
    is_valid = validate_config_json(user_config, config_path)

    if is_valid:
        return DotMap(user_config)
    else:
        logger.critical("\nExiting program")
        exit()
