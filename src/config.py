"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

from dotmap import DotMap

from .constants import CONFIG_DEFAULTS_PATH
from .utils.file import load_json
from .utils.object import OVERRIDE_MERGER

CONFIG_DEFAULTS = DotMap(load_json(CONFIG_DEFAULTS_PATH))


def open_config_with_defaults(config_path):
    user_config = load_json(config_path)
    merged_dict = OVERRIDE_MERGER.merge(CONFIG_DEFAULTS, user_config)
    return DotMap(merged_dict)
