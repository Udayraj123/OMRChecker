"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

from dotmap import DotMap

# Filenames
TEMPLATE_FILENAME = "template.json"
EVALUATION_FILENAME = "evaluation.json"
CONFIG_FILENAME = "config.json"

#
ERROR_CODES = DotMap(
    {
        "MULTI_BUBBLE_WARN": 1,
        "NO_MARKER_ERR": 2,
    },
    _dynamic=False,
)

QTYPE_DATA = {
    "QTYPE_ROLL": {"vals": range(10), "orient": "V"},
    "QTYPE_INT": {"vals": range(10), "orient": "V"},
    "QTYPE_INT_11": {"vals": range(11), "orient": "V"},
    "QTYPE_MCQ4": {"vals": ["A", "B", "C", "D"], "orient": "H"},
    "QTYPE_MCQ5": {"vals": ["A", "B", "C", "D", "E"], "orient": "H"},
    #
    # You can create and append custom question types here-
    #
}

# TODO: move to interaction.py
TEXT_SIZE = 0.95
CLR_BLACK = (50, 150, 150)
CLR_WHITE = (250, 250, 250)
CLR_GRAY = (130, 130, 130)
CLR_DARK_GRAY = (100, 100, 100)

# TODO: move to config.json
GLOBAL_PAGE_THRESHOLD_WHITE = 200
GLOBAL_PAGE_THRESHOLD_BLACK = 100
