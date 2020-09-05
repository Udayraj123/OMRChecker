"""

Designed and Developed by-
Udayraj Deshmukh 
https://github.com/Udayraj123

"""

"""
Constants
"""
EXTENSION_PATH="./extensions"
template_filename="template.json"
config_filename="config.json"

from deepmerge import always_merger
config = always_merger.merge(default_config, user_config)

ERROR_CODES = {
    "MULTI_BUBBLE_WARN": 1,
    "NO_MARKER_ERR": 2,
}

# Computations
CONFIDENT_JUMP = MIN_JUMP + CONFIDENT_SURPLUS


# Rather these are internal constants & not configs
# CLR_BLACK = rgb2tuple(CLR_BLACK)
TEXT_SIZE = 0.95
CLR_BLACK = (50, 150, 150)
CLR_WHITE = (250, 250, 250)
CLR_GRAY = (130, 130, 130)
# CLR_DARK_GRAY = (190,190,190)
CLR_DARK_GRAY = (100, 100, 100)

# Filepaths
class Paths:
    def __init__(self, output):
        self.output = output
        self.saveMarkedDir = f'{output}/CheckedOMRs/'
        self.resultDir = f'{output}/Results/'
        self.manualDir = f'{output}/Manual/'
        self.errorsDir = f'{self.manualDir}ErrorFiles/'
        self.badRollsDir = f'{self.manualDir}BadRollNosFiles/'
        self.multiMarkedDir = f'{self.manualDir}MultiMarkedFiles/'


