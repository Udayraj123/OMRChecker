"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

from dotmap import DotMap

from src.constants import CONFIG_DEFAULTS_PATH
from src.utils.parsing import OVERRIDE_MERGER
from src.utils.validations import load_json

CONFIG_DEFAULTS = DotMap(load_json(CONFIG_DEFAULTS_PATH))


def open_config_with_defaults(config_path):
    user_config = load_json(config_path)
    merged_dict = OVERRIDE_MERGER.merge(CONFIG_DEFAULTS, user_config)
    return DotMap(merged_dict)
