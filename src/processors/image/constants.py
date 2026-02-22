"""Image processing constants for OMRChecker.

This module contains all magic numbers and constants related to image processing,
thresholds, and computer vision operations. Extracting these values improves:
- Code readability and maintainability
- Easy parameter tuning
- Self-documenting code
"""

# ============================================================================
# Pixel and Color Values
# ============================================================================

# Standard pixel value ranges
PIXEL_VALUE_MAX = 255

# ============================================================================
# Thresholding Constants
# ============================================================================

# Page detection thresholds
THRESH_PAGE_TRUNCATE_HIGH = 210  # High truncate threshold for page detection
THRESH_PAGE_TRUNCATE_SECONDARY = 200  # Secondary truncate threshold

# Canny edge detection
CANNY_THRESHOLD_HIGH = 185  # High threshold for Canny edge detection
CANNY_THRESHOLD_LOW = 55  # Low threshold for Canny edge detection

# ============================================================================
# Contour and Shape Detection
# ============================================================================

# Minimum areas
MIN_PAGE_AREA = 80000  # Minimum area for valid page contour

# Approximation and simplification
APPROX_POLY_EPSILON_FACTOR = 0.025  # Epsilon factor for polygon approximation
CONTOUR_THICKNESS_STANDARD = 10  # Standard thickness for drawing contours


# ============================================================================
# Array and Collection Defaults
# ============================================================================

# Top N contours to consider
TOP_CONTOURS_COUNT = 5  # Number of top contours to analyze
