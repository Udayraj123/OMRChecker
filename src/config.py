import json
from deepmerge import always_merger
from .constants import CONFIG_DEFAULTS_PATH, TEMPLATE_DEFAULTS_PATH
from .utils import loadJson

templateDefaults = loadJson(TEMPLATE_DEFAULTS_PATH)

def openTemplateWithDefaults(templatePath):
    user_template = loadJson(templatePath)
    return always_merger.merge(templateDefaults, user_template)

configDefaults = loadJson(CONFIG_DEFAULTS_PATH)
def openConfigWithDefaults(configPath):
    user_config = loadJson(configPath)
    return always_merger.merge(configDefaults, user_config)
