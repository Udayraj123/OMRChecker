import matplotlib
from dotmap import DotMap

# Filenames
TEMPLATE_FILENAME = "template.json"
EVALUATION_FILENAME = "evaluation.json"
CONFIG_FILENAME = "config.json"

FIELD_LABEL_NUMBER_REGEX = r"([^\d]+)(\d*)"
#
ERROR_CODES = DotMap(
    {
        "MULTI_BUBBLE_WARN": 1,
        "NO_MARKER_ERR": 2,
    },
    _dynamic=False,
)

BUILTIN_FIELD_TYPES = {
    "QTYPE_INT": {
        "bubbleValues": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "direction": "vertical",
    },
    "QTYPE_INT_FROM_1": {
        "bubbleValues": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
        "direction": "vertical",
    },
    "QTYPE_MCQ4": {"bubbleValues": ["A", "B", "C", "D"], "direction": "horizontal"},
    "QTYPE_MCQ5": {
        "bubbleValues": ["A", "B", "C", "D", "E"],
        "direction": "horizontal",
    },
    #
    # You can create and append custom field types here-
    #
}

# TODO: Move TEXT_SIZE, etc into a class
TEXT_SIZE = 0.95
CLR_BLACK = (0, 0, 0)
CLR_DARK_GRAY = (100, 100, 100)
CLR_DARK_GREEN = (20, 255, 20)
CLR_NEAR_BLACK = (20, 20, 10)
CLR_GRAY = (130, 130, 130)
CLR_LIGHT_GRAY = (200, 200, 200)
CLR_GREEN = (100, 200, 100)
CLR_WHITE = (255, 255, 255)
MARKED_TEMPLATE_TRANSPARENCY = 0.65

MATPLOTLIB_COLORS = matplotlib.colors.get_named_colors_mapping()
