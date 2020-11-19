from .constants import CONFIG_DEFAULTS_PATH
from .utils.file import loadJson
from .utils.object import OVERRIDE_MERGER
from dotmap import DotMap

configDefaults = DotMap(loadJson(CONFIG_DEFAULTS_PATH))


def openConfigWithDefaults(configPath):
    user_config = loadJson(configPath)
    merged_dict = OVERRIDE_MERGER.merge(configDefaults, user_config)
    return DotMap(merged_dict)
