/**
 * Migrated from Python: src/processors/image/page_detection.py
 * Agent: Oz
 * Phase: Processor Image Migration
 *
 * Page detection utilities for finding and extracting page boundaries
 * from images using edge detection and contour analysis.
 */

import cv from '@techstark/opencv-js';
import { ImageUtils } from '../../utils/image';
import { MathUtils, Point } from '../../utils/math';
import { DrawingUtils } from '../../utils/drawing';
import { ImageProcessingError } from '../../utils/exceptions';
import {
  THRESH_PAGE_TRUNCATE_HIGH,
  THRESH_PAGE_TRUNCATE_SECONDARY,
  CANNY_THRESHOLD_HIGH,
  CANNY_THRESHOLD_LOW,
  MIN_PAGE_AREA,
  APPROX_POLY_EPSILON_FACTOR,
  TOP_CONTOURS_COUNT,
  CONTOUR_THICKNESS_STANDARD,
} from './constants';

/** White color in BGR */
const CLR_WHITE: [number, number, number] = [255, 255, 255];

/** HSV lower bound for whitish region detection (H, S, V) */
const HSV_WHITE_LOW = [0, 0, 150];

/** HSV upper bound for whitish region detection (H, S, V) */
const HSV_WHITE_HIGH = [180, 60, 255];

export interface FindPageOptions {
  /** Optional colored image for HSV-based canny (BGR 3-channel) */
  coloredImage?: cv.Mat;
  /** Use colored canny instead of grayscale canny */
  useColoredCanny?: boolean;
  /** Optional morphology kernel for grayscale canny */
  morphKernel?: cv.Mat;
  /** File path for error context */
  filePath?: string;
  /** Debug image to draw found contours on */
  debugImage?: cv.Mat;
}

/**
 * Prepare the grayscale page image for edge detection.
 *
 * Truncates pixels above THRESH_PAGE_TRUNCATE_HIGH and normalizes
 * the result to the full 0–255 range.
 *
 * @param image - Grayscale input image (CV_8UC1)
 * @returns Prepared image (caller must delete)
 */
export function preparePageImage(image: cv.Mat): cv.Mat {
  const truncated = new cv.Mat();
  cv.threshold(image, truncated, THRESH_PAGE_TRUNCATE_HIGH, 255, cv.THRESH_TRUNC);
  const normalized = ImageUtils.normalizeSingle(truncated);
  truncated.delete();
  return normalized;
}

/**
 * Apply Canny edge detection using an HSV white-region mask.
 *
 * Converts the colored image to HSV, masks the whitish regions,
 * applies the mask to the prepared grayscale image, then runs Canny.
 *
 * @param image - Prepared grayscale image (CV_8UC1)
 * @param coloredImage - Original BGR color image (CV_8UC3)
 * @returns Canny edge image (caller must delete)
 */
export function applyColoredCanny(image: cv.Mat, coloredImage: cv.Mat): cv.Mat {
  const hsv = new cv.Mat();
  const hsvChannels = new cv.MatVector();
  const maskV = new cv.Mat();
  const maskS = new cv.Mat();
  const mask = new cv.Mat();
  const masked = new cv.Mat();
  const edges = new cv.Mat();

  try {
    cv.cvtColor(coloredImage, hsv, cv.COLOR_BGR2HSV);

    // Build whitish-region mask via HSV channel split:
    //   V >= HSV_WHITE_LOW[2] (150)  AND  S <= HSV_WHITE_HIGH[1] (60)
    cv.split(hsv, hsvChannels);
    const vChannel = hsvChannels.get(2);
    const sChannel = hsvChannels.get(1);
    cv.threshold(vChannel, maskV, HSV_WHITE_LOW[2] - 1, 255, cv.THRESH_BINARY);
    cv.threshold(sChannel, maskS, HSV_WHITE_HIGH[1], 255, cv.THRESH_BINARY_INV);
    cv.bitwise_and(maskV, maskS, mask);
    vChannel.delete();
    sChannel.delete();

    cv.bitwise_and(image, image, masked, mask);
    cv.Canny(masked, edges, CANNY_THRESHOLD_HIGH, CANNY_THRESHOLD_LOW);
    return edges.clone();
  } finally {
    hsv.delete();
    hsvChannels.delete();
    maskV.delete();
    maskS.delete();
    mask.delete();
    masked.delete();
    edges.delete();
  }
}

/**
 * Apply Canny edge detection on a grayscale image with optional morphological closing.
 *
 * Truncates at THRESH_PAGE_TRUNCATE_SECONDARY, normalizes, optionally applies
 * morphological closing, then runs Canny.
 *
 * @param image - Grayscale input image (CV_8UC1)
 * @param morphKernel - Optional morphology kernel (ignored if rows <= 1)
 * @returns Canny edge image (caller must delete)
 */
export function applyGrayscaleCanny(image: cv.Mat, morphKernel?: cv.Mat): cv.Mat {
  const truncated = new cv.Mat();
  cv.threshold(image, truncated, THRESH_PAGE_TRUNCATE_SECONDARY, 255, cv.THRESH_TRUNC);

  const normalized = ImageUtils.normalizeSingle(truncated);
  truncated.delete();

  let processed: cv.Mat;
  let ownedClosed: cv.Mat | null = null;

  if (morphKernel && morphKernel.rows > 1) {
    ownedClosed = new cv.Mat();
    cv.morphologyEx(normalized, ownedClosed, cv.MORPH_CLOSE, morphKernel);
    processed = ownedClosed;
  } else {
    processed = normalized;
  }

  const edges = new cv.Mat();
  try {
    cv.Canny(processed, edges, CANNY_THRESHOLD_HIGH, CANNY_THRESHOLD_LOW);
    return edges.clone();
  } finally {
    edges.delete();
    normalized.delete();
    if (ownedClosed) {
      ownedClosed.delete();
    }
  }
}

/**
 * Find the top candidate page contours from a Canny edge image.
 *
 * Extracts all contours, converts each to its convex hull, sorts by area
 * (descending), and returns the top TOP_CONTOURS_COUNT candidates.
 *
 * @param cannyEdge - Canny edge image (CV_8UC1)
 * @returns Array of contour Mats (caller must delete each)
 */
export function findPageContours(cannyEdge: cv.Mat): cv.Mat[] {
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();

  try {
    cv.findContours(cannyEdge, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
  } finally {
    hierarchy.delete();
  }

  // Convert each contour to its convex hull
  const hullList: cv.Mat[] = [];
  const size = contours.size();
  for (let i = 0; i < size; i++) {
    const hull = new cv.Mat();
    cv.convexHull(contours.get(i), hull);
    hullList.push(hull);
  }
  contours.delete();

  // Sort by contour area descending
  hullList.sort((a, b) => cv.contourArea(b) - cv.contourArea(a));

  // Keep only top candidates — delete the rest
  const topContours = hullList.slice(0, TOP_CONTOURS_COUNT);
  for (let i = TOP_CONTOURS_COUNT; i < hullList.length; i++) {
    hullList[i].delete();
  }

  return topContours;
}

/**
 * Extract points from an approxPolyDP result Mat.
 *
 * The Mat has shape (N, 1, 2) stored as int32. This reads data32S to get
 * the (x, y) pairs.
 *
 * @param approx - Result from cv.approxPolyDP (CV_32SC2)
 * @returns Array of [x, y] points
 */
function extractApproxPoints(approx: cv.Mat): Point[] {
  const n = approx.rows;
  const points: Point[] = [];
  for (let i = 0; i < n; i++) {
    const x = approx.data32S[i * 2];
    const y = approx.data32S[i * 2 + 1];
    points.push([x, y]);
  }
  return points;
}

/**
 * Extract a valid page rectangle from a list of candidate contours.
 *
 * Iterates over contours, skips those with area < MIN_PAGE_AREA, approximates
 * each as a polygon, and returns the first one that passes MathUtils.validateRect.
 *
 * @param contours - Candidate contour Mats (from findPageContours)
 * @returns Tuple of [corners (4x2 array), fullContour Mat] or [null, null] if not found.
 *          The returned fullContour Mat must be deleted by the caller.
 */
export function extractPageRectangle(contours: cv.Mat[]): [number[][] | null, cv.Mat | null] {
  for (const contour of contours) {
    const area = cv.contourArea(contour);
    if (area < MIN_PAGE_AREA) {
      continue;
    }

    const perimeter = cv.arcLength(contour, true);
    const epsilon = APPROX_POLY_EPSILON_FACTOR * perimeter;
    const approx = new cv.Mat();

    try {
      cv.approxPolyDP(contour, approx, epsilon, true);

      if (approx.rows !== 4) {
        continue;
      }

      const points = extractApproxPoints(approx);

      if (MathUtils.validateRect(points)) {
        // corners is a 4x2 array of [x, y]
        const corners: number[][] = points.map(([x, y]) => [x, y]);
        // full_contour: squeeze the contour data into an array of points
        // We clone the contour Mat so the caller owns it
        const fullContour = contour.clone();
        return [corners, fullContour];
      }
    } finally {
      approx.delete();
    }
  }

  return [null, null];
}

/**
 * Find the page contour and corner points in an image.
 *
 * Main entry point for page detection. Prepares the image, applies edge
 * detection, finds contours, extracts the page rectangle, optionally
 * draws debug contours, and raises ImageProcessingError if not found.
 *
 * @param image - Grayscale input image (CV_8UC1)
 * @param options - Optional configuration (colored canny, morph kernel, debug)
 * @returns Tuple of [corners (4x2 array of [x,y]), pageContour Mat].
 *          The returned pageContour Mat must be deleted by the caller.
 * @throws ImageProcessingError if no valid page rectangle is found
 */
export function findPageContourAndCorners(
  image: cv.Mat,
  options: FindPageOptions = {}
): [number[][], cv.Mat] {
  const { coloredImage, useColoredCanny = false, morphKernel, filePath, debugImage } = options;

  const prepared = preparePageImage(image);
  let cannyEdge: cv.Mat;

  try {
    if (useColoredCanny && coloredImage) {
      cannyEdge = applyColoredCanny(prepared, coloredImage);
    } else {
      cannyEdge = applyGrayscaleCanny(prepared, morphKernel);
    }
  } finally {
    prepared.delete();
  }

  let contours: cv.Mat[];
  try {
    contours = findPageContours(cannyEdge);
  } finally {
    cannyEdge.delete();
  }

  let corners: number[][] | null = null;
  let pageContour: cv.Mat | null = null;

  try {
    [corners, pageContour] = extractPageRectangle(contours);
  } finally {
    // Delete the candidate contours (we have our own clone in pageContour)
    for (const c of contours) {
      c.delete();
    }
  }

  if (corners !== null && pageContour !== null && debugImage) {
    // Draw found contour on debug image
    const approxPoints: Point[] = corners.map(([x, y]) => [x, y] as Point);
    DrawingUtils.drawContour(debugImage, approxPoints, CLR_WHITE, CONTOUR_THICKNESS_STANDARD);
  }

  if (pageContour === null) {
    throw new ImageProcessingError('Paper boundary not found', {
      filePath,
      reason: `No valid rectangle found in top contour candidates`,
    });
  }

  return [corners!, pageContour];
}
