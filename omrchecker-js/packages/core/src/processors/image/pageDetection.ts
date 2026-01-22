/**
 * Page Detection Module
 *
 * TypeScript port of src/processors/image/page_detection.py
 *
 * Extracted from CropPage to provide focused, testable page boundary detection.
 *
 * This module handles:
 * - Edge detection using Canny
 * - Contour finding and filtering
 * - Page boundary identification
 * - Corner extraction
 */

import cv from '../../utils/opencv';
import { Logger } from '../../utils/logger';
import { ImageUtils } from '../../utils/ImageUtils';
import { MathUtils, type Point, type Rectangle } from '../../utils/math';
import { DrawingUtils } from '../../utils/drawing';

const logger = new Logger('PageDetection');

// Constants (from src/constants.py)
const APPROX_POLY_EPSILON_FACTOR = 0.02;
const CANNY_THRESHOLD_HIGH = 185;
const CANNY_THRESHOLD_LOW = 55;
const CONTOUR_THICKNESS_STANDARD = 4;
const MIN_PAGE_AREA = 80_000;
const PIXEL_VALUE_MAX = 255;
const THRESH_PAGE_TRUNCATE_HIGH = 205;
const THRESH_PAGE_TRUNCATE_SECONDARY = 205;
const TOP_CONTOURS_COUNT = 4;

// HSV white color range for colored Canny
const HSV_WHITE_LOW = new cv.Scalar(0, 0, 130);
const HSV_WHITE_HIGH = new cv.Scalar(255, 80, 255);
const CLR_WHITE: [number, number, number] = [255, 255, 255];

export class ImageProcessingError extends Error {
  filePath?: string;
  reason?: string;

  constructor(message: string, filePath?: string, reason?: string) {
    super(message);
    this.name = 'ImageProcessingError';
    this.filePath = filePath;
    this.reason = reason;
  }
}

/**
 * Prepare image for page detection.
 *
 * Applies truncation and normalization to enhance page boundaries.
 *
 * @param image - Grayscale input image
 * @returns Preprocessed image ready for edge detection
 */
export function preparePageImage(image: cv.Mat): cv.Mat {
  // Truncate high values to reduce noise
  const truncated = new cv.Mat();
  cv.threshold(image, truncated, THRESH_PAGE_TRUNCATE_HIGH, PIXEL_VALUE_MAX, cv.THRESH_TRUNC);

  // Normalize to full range
  const normalized = ImageUtils.normalize([truncated]) as cv.Mat;
  truncated.delete();

  return normalized;
}

/**
 * Apply Canny edge detection on color-masked image.
 *
 * Uses HSV color space to isolate white-ish regions (the page).
 *
 * @param image - Grayscale image
 * @param coloredImage - Original BGR color image
 * @returns Canny edge map
 */
export function applyColoredCanny(image: cv.Mat, coloredImage: cv.Mat): cv.Mat {
  // Convert to HSV for better color-based masking
  const hsv = new cv.Mat();
  cv.cvtColor(coloredImage, hsv, cv.COLOR_BGR2HSV);

  // Mask to select only white-ish zones (the page)
  const mask = new cv.Mat();
  const hsvWhiteLow = new cv.Mat(1, 1, cv.CV_8UC3, HSV_WHITE_LOW);
  const hsvWhiteHigh = new cv.Mat(1, 1, cv.CV_8UC3, HSV_WHITE_HIGH);
  cv.inRange(hsv, hsvWhiteLow, hsvWhiteHigh, mask);
  hsv.delete();
  hsvWhiteLow.delete();
  hsvWhiteHigh.delete();

  const maskResult = new cv.Mat();
  const noMask = new cv.Mat();  // Empty mask parameter
  cv.bitwise_and(image, image, maskResult, mask);
  mask.delete();
  noMask.delete();

  // Apply Canny edge detection
  const cannyEdge = new cv.Mat();
  cv.Canny(maskResult, cannyEdge, CANNY_THRESHOLD_HIGH, CANNY_THRESHOLD_LOW);
  maskResult.delete();

  return cannyEdge;
}

/**
 * Apply Canny edge detection on grayscale image.
 *
 * Optionally applies morphological closing to complete broken edges.
 *
 * @param image - Preprocessed grayscale image
 * @param morphKernel - Optional morphological kernel for closing operation
 * @returns Canny edge map
 */
export function applyGrayscaleCanny(image: cv.Mat, morphKernel?: cv.Mat): cv.Mat {
  // Second truncation threshold for cleaner edges
  const truncated = new cv.Mat();
  cv.threshold(
    image,
    truncated,
    THRESH_PAGE_TRUNCATE_SECONDARY,
    PIXEL_VALUE_MAX,
    cv.THRESH_TRUNC
  );

  const normalized = ImageUtils.normalize([truncated]) as cv.Mat;
  truncated.delete();

  // Close small holes to complete edges
  let closed: cv.Mat;
  if (morphKernel && morphKernel.rows > 1) {
    closed = new cv.Mat();
    cv.morphologyEx(normalized, closed, cv.MORPH_CLOSE, morphKernel);
    normalized.delete();
  } else {
    closed = normalized;
  }

  // Apply Canny edge detection
  const cannyEdge = new cv.Mat();
  cv.Canny(closed, cannyEdge, CANNY_THRESHOLD_HIGH, CANNY_THRESHOLD_LOW);
  closed.delete();

  return cannyEdge;
}

/**
 * Find and filter contours that could be the page boundary.
 *
 * @param cannyEdge - Canny edge map
 * @returns List of candidate contours, sorted by area (largest first)
 */
export function findPageContours(cannyEdge: cv.Mat): cv.MatVector {
  // Find all contours
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(cannyEdge, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
  hierarchy.delete();

  // Apply convex hull to resolve disordered curves from noise
  const hulledContours = new cv.MatVector();
  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i);
    const hull = new cv.Mat();
    cv.convexHull(contour, hull);
    hulledContours.push_back(hull);
  }

  // Sort by area and take top candidates
  const areas: Array<{ index: number; area: number }> = [];
  for (let i = 0; i < hulledContours.size(); i++) {
    const contour = hulledContours.get(i);
    areas.push({ index: i, area: cv.contourArea(contour) });
  }
  areas.sort((a, b) => b.area - a.area);

  const sortedContours = new cv.MatVector();
  const maxCount = Math.min(TOP_CONTOURS_COUNT, areas.length);
  for (let i = 0; i < maxCount; i++) {
    sortedContours.push_back(hulledContours.get(areas[i].index));
  }

  // Cleanup
  for (let i = 0; i < contours.size(); i++) {
    contours.get(i).delete();
  }
  for (let i = 0; i < hulledContours.size(); i++) {
    hulledContours.get(i).delete();
  }
  contours.delete();
  hulledContours.delete();

  return sortedContours;
}

/**
 * Extract the page rectangle from candidate contours.
 *
 * Finds the first contour that:
 * 1. Has area >= MIN_PAGE_AREA
 * 2. Can be approximated as a 4-sided polygon
 * 3. Forms a valid rectangle
 *
 * @param contours - List of candidate contours
 * @returns Tuple of [corners, full_contour] or [null, null] if not found
 */
export function extractPageRectangle(
  contours: cv.MatVector
): [Rectangle | null, cv.Mat | null] {
  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i);

    // Skip if too small
    if (cv.contourArea(contour) < MIN_PAGE_AREA) {
      continue;
    }

    // Approximate contour to polygon
    const perimeter = cv.arcLength(contour, true);
    const epsilon = APPROX_POLY_EPSILON_FACTOR * perimeter;
    const approx = new cv.Mat();
    cv.approxPolyDP(contour, approx, epsilon, true);

    // Check if it's a valid rectangle (4 corners)
    if (approx.rows === 4) {
      // Extract corners as Point array
      const corners: Point[] = [];
      for (let j = 0; j < 4; j++) {
        corners.push([approx.data32S[j * 2], approx.data32S[j * 2 + 1]]);
      }

      // Validate rectangle
      if (MathUtils.validateRect(corners)) {
        const [orderedCorners] = MathUtils.orderFourPoints(corners);
        const fullContour = contour.clone();
        approx.delete();
        return [orderedCorners, fullContour];
      }
    }

    approx.delete();
  }

  return [null, null];
}

/**
 * Find page boundary and extract corners.
 *
 * Main entry point for page detection. Combines all steps:
 * 1. Image preparation
 * 2. Edge detection (colored or grayscale)
 * 3. Contour finding
 * 4. Rectangle extraction
 *
 * @param image - Grayscale input image
 * @param coloredImage - Optional color image for colored Canny
 * @param useColoredCanny - Whether to use color-based edge detection
 * @param morphKernel - Optional morphological kernel
 * @param filePath - Optional file path for error messages
 * @param debugImage - Optional image to draw debug contours on
 * @returns Tuple of [corners, page_contour]
 * @throws ImageProcessingError if page boundary cannot be found
 */
export function findPageContourAndCorners(
  image: cv.Mat,
  options: {
    coloredImage?: cv.Mat;
    useColoredCanny?: boolean;
    morphKernel?: cv.Mat;
    filePath?: string;
    debugImage?: cv.Mat;
  } = {}
): [Rectangle, cv.Mat] {
  const { coloredImage, useColoredCanny = false, morphKernel, filePath, debugImage } = options;

  // Step 1: Prepare image
  const preparedImage = preparePageImage(image);

  // Step 2: Edge detection
  let cannyEdge: cv.Mat;
  if (useColoredCanny && coloredImage) {
    cannyEdge = applyColoredCanny(preparedImage, coloredImage);
  } else {
    cannyEdge = applyGrayscaleCanny(preparedImage, morphKernel);
  }
  preparedImage.delete();

  // Step 3: Find contours
  const contours = findPageContours(cannyEdge);

  // Step 4: Extract rectangle
  const [corners, pageContour] = extractPageRectangle(contours);

  // Cleanup contours
  for (let i = 0; i < contours.size(); i++) {
    contours.get(i).delete();
  }
  contours.delete();

  // Step 5: Draw debug visualization if requested
  if (corners && debugImage) {
    // Convert corners to Mat for drawing
    const approxPoints = cv.matFromArray(4, 1, cv.CV_32SC2, [
      corners[0][0],
      corners[0][1],
      corners[1][0],
      corners[1][1],
      corners[2][0],
      corners[2][1],
      corners[3][0],
      corners[3][1],
    ]);

    DrawingUtils.drawContour(
      cannyEdge,
      approxPoints,
      CLR_WHITE,
      CONTOUR_THICKNESS_STANDARD
    );
    DrawingUtils.drawContour(
      debugImage,
      approxPoints,
      CLR_WHITE,
      CONTOUR_THICKNESS_STANDARD
    );

    approxPoints.delete();
  }

  cannyEdge.delete();

  // Step 6: Error if not found
  if (!pageContour || !corners) {
    const fileStr = filePath || 'unknown';
    logger.error(`Paper boundary not found for: '${fileStr}'`);
    logger.warn(
      'Have you accidentally included CropPage preprocessor?\n' +
        `If no, increase processing dimensions from config. Current image size: [${image.rows}, ${image.cols}]`
    );

    throw new ImageProcessingError(
      'Paper boundary not found',
      filePath,
      `No valid rectangle found in candidates`
    );
  }

  return [corners, pageContour];
}

