/**
 * Migrated from Python: src/processors/image/constants.py
 * Agent: Oz
 * Phase: Processor Image Migration
 *
 * Constants used by image processor utilities (page detection, etc.)
 */

export const PIXEL_VALUE_MAX = 255;

// Thresholds for page truncation
export const THRESH_PAGE_TRUNCATE_HIGH = 210;
export const THRESH_PAGE_TRUNCATE_SECONDARY = 200;

// Canny edge detection thresholds
export const CANNY_THRESHOLD_HIGH = 185;
export const CANNY_THRESHOLD_LOW = 55;

// Page area and contour constants
export const MIN_PAGE_AREA = 80000;
export const APPROX_POLY_EPSILON_FACTOR = 0.025;
export const CONTOUR_THICKNESS_STANDARD = 10;
export const TOP_CONTOURS_COUNT = 5;
