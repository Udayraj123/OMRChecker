"""
Constants for image processing operations across OMRChecker.
"""

# General Image Processing
DEFAULT_WHITE_COLOR = 255
DEFAULT_BLACK_COLOR = 0
DEFAULT_NORMALIZE_PARAMS = {"alpha": 0, "beta": 255}
DEFAULT_LINE_WIDTH = 2
DEFAULT_MARKER_LINE_WIDTH = 4
DEFAULT_CONTOUR_COLOR = (0, 255, 0)
DEFAULT_CONTOUR_LINE_WIDTH = 2
DEFAULT_CONTOUR_FILL_COLOR = (255, 255, 255)
DEFAULT_CONTOUR_FILL_WIDTH = 10
DEFAULT_BORDER_REMOVE = 5

DEFAULT_GAUSSIAN_BLUR_PARAMS_MARKER = {"kernel_size": (5, 5), "sigma_x": 0}

# CropPage constants
MIN_PAGE_AREA_THRESHOLD = 80000
MAX_COSINE_THRESHOLD = 0.35
DEFAULT_GAUSSIAN_BLUR_KERNEL = (3, 3)
PAGE_THRESHOLD_PARAMS = {"threshold_value": 200, "max_pixel_value": 255}
CANNY_PARAMS = {
    # lower_threshold: lower bound for Canny edge detection
    # upper_threshold: upper bound for Canny edge detection
    "lower_threshold": 185,
    "upper_threshold": 55,
}
APPROX_POLY_EPSILON_FACTOR = 0.025

# CropOnMarkers constants
QUADRANT_DIVISION = {"height_factor": 3, "width_factor": 2}
MARKER_RECTANGLE_COLOR = (150, 150, 150)
ERODE_RECT_COLOR = (50, 50, 50)
NORMAL_RECT_COLOR = (155, 155, 155)
EROSION_PARAMS = {"kernel_size": (5, 5), "iterations": 5}

# FeatureBasedAlignment constants
DEFAULT_MAX_FEATURES = 500
DEFAULT_GOOD_MATCH_PERCENT = 0.15

# Builtin processor constants
DEFAULT_MEDIAN_BLUR_KERNEL_SIZE = 5
DEFAULT_GAUSSIAN_BLUR_PARAMS = {"kernel_size": (3, 3), "sigma_x": 0}
