"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

import os
from dotmap import DotMap

# Paths
CURRENT_MODULE_DIR = os.path.dirname(__file__)
CONFIG_DEFAULTS_PATH = os.path.join(CURRENT_MODULE_DIR, "defaults/config.json")
TEMPLATE_DEFAULTS_PATH = os.path.join(
    CURRENT_MODULE_DIR, "defaults/template.json")

# Filenames
TEMPLATE_FILENAME = "template.json"
CONFIG_FILENAME = "config.json"

#
ERROR_CODES = DotMap(
    {
        "MULTI_BUBBLE_WARN": 1,
        "NO_MARKER_ERR": 2,
    }
)

QTYPE_DATA = {
    "QTYPE_MED": {"vals": ["E", "H"], "orient": "V"},
    "QTYPE_ROLL": {"vals": range(10), "orient": "V"},
    "QTYPE_INT": {"vals": range(10), "orient": "V"},
    "QTYPE_INT_11": {"vals": range(11), "orient": "V"},
    "QTYPE_MCQ4": {"vals": ["A", "B", "C", "D"], "orient": "H"},
    "QTYPE_MCQ5": {"vals": ["A", "B", "C", "D", "E"], "orient": "H"},
    #
    # You can create and append custom question types here-
    #
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
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.save_marked_dir = f"{self.output_dir}/CheckedOMRs/"
        self.results_dir = f"{self.output_dir}/Results/"
        self.manual_dir = f"{self.output_dir}/Manual/"
        self.errors_dir = f"{self.manual_dir}ErrorFiles/"
        self.multi_marked_dir = f"{self.manual_dir}MultiMarkedFiles/"


class Stats:
    def __init__(self):
        self.files_moved = 0
        self.files_not_moved = 0
