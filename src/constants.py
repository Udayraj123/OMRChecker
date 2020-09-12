"""

Designed and Developed by-
Udayraj Deshmukh 
https://github.com/Udayraj123

"""

"""
Constants
"""
EXTENSION_PATH="./extensions"
TEMPLATE_FILENAME="template.json"
CONFIG_FILENAME="config.json"

CONFIG_DEFAULTS_PATH = "./defaults/config.json"
TEMPLATE_DEFAULTS_PATH = "./defaults/template.json"

ERROR_CODES = {
    "MULTI_BUBBLE_WARN": 1,
    "NO_MARKER_ERR": 2,
}

# Rather these are internal constants & not configs
# CLR_BLACK = rgb2tuple(CLR_BLACK)
TEXT_SIZE = 0.95
CLR_BLACK = (50, 150, 150)
CLR_WHITE = (250, 250, 250)
CLR_GRAY = (130, 130, 130)
# CLR_DARK_GRAY = (190,190,190)
CLR_DARK_GRAY = (100, 100, 100)

# Filepaths - object is better
class Paths:
    def __init__(self, outputDir):
        self.OUTPUT_DIR = outputDir
        self.SAVE_MARKED_DIR = f'{self.OUTPUT_DIR}/CheckedOMRs/'
        self.RESULTS_DIR = f'{self.OUTPUT_DIR}/Results/'
        self.MANUAL_DIR = f'{self.OUTPUT_DIR}/Manual/'
        self.ERRORS_DIR = f'{self.MANUAL_DIR}ErrorFiles/'
        self.BAD_ROLLS_DIR = f'{self.MANUAL_DIR}BadRollNosFiles/'
        self.MULTI_MARKED_DIR = f'{self.MANUAL_DIR}MultiMarkedFiles/'


