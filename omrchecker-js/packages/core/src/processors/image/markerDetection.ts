/**
 * Marker Detection Module
 *
 * TypeScript port of src/processors/image/marker_detection.py
 *
 * Extracted from CropOnCustomMarkers to provide focused, testable
 * marker template matching algorithms.
 *
 * This module handles:
 * - Marker template preparation
 * - Multi-scale template matching
 * - Best match selection
 * - Marker corner extraction
 */

import cv from '../../utils/opencv';
import { Logger } from '../../utils/logger';
import { MathUtils } from '../../utils/math';

const logger = new Logger('MarkerDetection');

export interface ReferenceZone {
  origin: [number, number];
  dimensions: [number, number];
}

export interface MatchResult {
  position: [number, number] | null;
  optimalMarker: cv.Mat | null;
  confidence: number;
  scalePercent: number | null;
}

export interface DetectionResult {
  corners: number[][] | null;
}

/**
 * Extract and prepare marker template from reference image.
 *
 * Applies preprocessing to enhance marker features:
 * 1. Extract region of interest
 * 2. Resize if dimensions specified
 * 3. Gaussian blur to reduce noise
 * 4. Normalize to full range
 * 5. Optional erode-subtract to enhance edges
 *
 * @param referenceImage - Source image containing the marker
 * @param referenceZone - Object with 'origin' and 'dimensions' keys
 * @param markerDimensions - Optional [width, height] to resize marker
 * @param blurKernel - Gaussian blur kernel size
 * @param applyErodeSubtract - Whether to enhance edges with erosion
 * @returns Preprocessed marker template ready for matching
 */
export function prepareMarkerTemplate(
  referenceImage: cv.Mat,
  referenceZone: ReferenceZone,
  markerDimensions?: [number, number],
  blurKernel: [number, number] = [5, 5],
  applyErodeSubtract: boolean = true
): cv.Mat {
  const { origin, dimensions } = referenceZone;
  const [x, y] = origin;
  const [w, h] = dimensions;

  // Extract marker region
  const rect = new cv.Rect(x, y, w, h);
  let marker = referenceImage.roi(rect);

  try {
    // Resize if dimensions specified
    if (markerDimensions) {
      const [targetW, targetH] = markerDimensions;
      const resized = new cv.Mat();
      cv.resize(marker, resized, new cv.Size(targetW, targetH));
      marker.delete();
      marker = resized;
    }

    // Blur to reduce noise
    const blurred = new cv.Mat();
    const ksize = new cv.Size(blurKernel[0], blurKernel[1]);
    cv.GaussianBlur(marker, blurred, ksize, 0);
    marker.delete();
    marker = blurred;

    // Normalize to full range
    let normalized = new cv.Mat();
    cv.normalize(marker, normalized, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
    marker.delete();
    marker = normalized;

    // Optional edge enhancement
    if (applyErodeSubtract) {
      // Erode
      const eroded = new cv.Mat();
      const kernel = cv.Mat.ones(5, 5, cv.CV_8U);
      cv.erode(marker, eroded, kernel, new cv.Point(-1, -1), 5);
      kernel.delete();

      // Subtract
      const subtracted = new cv.Mat();
      cv.subtract(marker, eroded, subtracted);
      eroded.delete();
      marker.delete();
      marker = subtracted;

      // Re-normalize after edge enhancement
      normalized = new cv.Mat();
      cv.normalize(marker, normalized, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
      marker.delete();
      marker = normalized;
    }

    return marker;
  } catch (error) {
    marker.delete();
    throw error;
  }
}

/**
 * Perform multi-scale template matching to find best match.
 *
 * Tests marker at different scales within scaleRange to account for
 * size variations due to scanning/printing differences.
 *
 * @param patch - Image patch to search in
 * @param marker - Template marker to find
 * @param scaleRange - [minPercent, maxPercent] for scaling
 * @param scaleSteps - Number of scale increments to test
 * @returns Match result with position, marker, confidence, and scale
 */
export function multiScaleTemplateMatch(
  patch: cv.Mat,
  marker: cv.Mat,
  scaleRange: [number, number] = [85, 115],
  scaleSteps: number = 5
): MatchResult {
  const descentPerStep = (scaleRange[1] - scaleRange[0]) / scaleSteps;
  const markerHeight = marker.rows;
  const markerWidth = marker.cols;
  const patchHeight = patch.rows;
  const patchWidth = patch.cols;

  let bestPosition: [number, number] | null = null;
  let bestMarker: cv.Mat | null = null;
  let bestConfidence = 0.0;
  let bestScalePercent: number | null = null;

  // Test different scales
  for (
    let scalePercent = scaleRange[1];
    scalePercent > scaleRange[0];
    scalePercent -= descentPerStep
  ) {
    const scale = scalePercent / 100;
    if (scale <= 0.0) continue;

    // Rescale marker
    const scaledWidth = Math.round(markerWidth * scale);
    const scaledHeight = Math.round(markerHeight * scale);

    // Skip if rescaled marker is larger than patch
    if (scaledHeight > patchHeight || scaledWidth > patchWidth) {
      continue;
    }

    const scaledMarker = new cv.Mat();
    try {
      cv.resize(marker, scaledMarker, new cv.Size(scaledWidth, scaledHeight), 0, 0, cv.INTER_LINEAR);

      // Template matching
      const matchResult = new cv.Mat();
      cv.matchTemplate(patch, scaledMarker, matchResult, cv.TM_CCOEFF_NORMED);

      const result = cv.minMaxLoc(matchResult, new cv.Mat());
      const maxConfidence = result.maxVal;

      matchResult.delete();

      // Update best match if this is better
      if (maxConfidence > bestConfidence) {
        bestScalePercent = scalePercent;
        if (bestMarker) {
          bestMarker.delete();
        }
        bestMarker = scaledMarker.clone();
        bestConfidence = maxConfidence;
        bestPosition = [result.maxLoc.x, result.maxLoc.y];
      }

      scaledMarker.delete();
    } catch (error) {
      scaledMarker.delete();
      throw error;
    }
  }

  logger.debug(
    `Best match: scale=${bestScalePercent}%, confidence=${(bestConfidence * 100).toFixed(2)}%`
  );

  return {
    position: bestPosition,
    optimalMarker: bestMarker,
    confidence: bestConfidence,
    scalePercent: bestScalePercent,
  };
}

/**
 * Extract corner points of detected marker.
 *
 * @param position - [x, y] position of marker's top-left corner in patch
 * @param marker - The marker template (for dimensions)
 * @param zoneOffset - Offset to convert from patch coordinates to absolute
 * @returns 4x2 array of corner points in absolute coordinates
 */
export function extractMarkerCorners(
  position: [number, number],
  marker: cv.Mat,
  zoneOffset: [number, number] = [0, 0]
): number[][] {
  const h = marker.rows;
  const w = marker.cols;
  const [x, y] = position;

  // Get rectangle corners in patch coordinates
  const corners = MathUtils.getRectanglePoints(x, y, w, h);

  // Shift to absolute coordinates
  const absoluteCorners = MathUtils.shiftPointsFromOrigin(zoneOffset, corners);

  return absoluteCorners;
}

/**
 * Detect marker in a patch using multi-scale template matching.
 *
 * Main entry point for marker detection. Combines:
 * 1. Multi-scale template matching
 * 2. Confidence threshold check
 * 3. Corner extraction
 *
 * @param patch - Image patch to search in
 * @param marker - Template marker to find
 * @param zoneOffset - Offset for absolute coordinate conversion
 * @param scaleRange - [minPercent, maxPercent] for scaling
 * @param scaleSteps - Number of scale increments to test
 * @param minConfidence - Minimum confidence threshold (0.0 to 1.0)
 * @returns 4x2 array of corner points in absolute coordinates, or null if not found
 */
export function detectMarkerInPatch(
  patch: cv.Mat,
  marker: cv.Mat,
  zoneOffset: [number, number] = [0, 0],
  scaleRange: [number, number] = [85, 115],
  scaleSteps: number = 5,
  minConfidence: number = 0.3
): number[][] | null {
  // Perform multi-scale matching
  const matchResult = multiScaleTemplateMatch(patch, marker, scaleRange, scaleSteps);

  // Check if we found a valid match
  if (!matchResult.position || !matchResult.optimalMarker) {
    logger.warn('No marker match found in patch');
    if (matchResult.optimalMarker) {
      matchResult.optimalMarker.delete();
    }
    return null;
  }

  if (matchResult.confidence < minConfidence) {
    logger.warn(
      `Marker match confidence ${(matchResult.confidence * 100).toFixed(2)}% below threshold ${(
        minConfidence * 100
      ).toFixed(2)}%`
    );
    matchResult.optimalMarker.delete();
    return null;
  }

  // Extract corners
  const corners = extractMarkerCorners(
    matchResult.position,
    matchResult.optimalMarker,
    zoneOffset
  );

  matchResult.optimalMarker.delete();

  logger.debug(
    `Marker detected at [${matchResult.position}] with confidence ${(
      matchResult.confidence * 100
    ).toFixed(2)}% (scale=${matchResult.scalePercent}%)`
  );

  return corners;
}

/**
 * Validate detected marker corners.
 *
 * @param corners - 4x2 array of corner points
 * @param expectedAreaRange - Optional [minArea, maxArea] for validation
 * @returns True if corners are valid, false otherwise
 */
export function validateMarkerDetection(
  corners: number[][] | null,
  expectedAreaRange?: [number, number]
): boolean {
  if (!corners) {
    return false;
  }

  if (corners.length !== 4 || corners.some((c) => c.length !== 2)) {
    logger.warn(`Invalid corner shape: expected 4x2, got ${corners.length}`);
    return false;
  }

  // Optional area validation
  if (expectedAreaRange) {
    // Calculate area using cross product
    // Convert to cv.Mat for contourArea
    const mat = cv.matFromArray(corners.length, 1, cv.CV_32SC2, corners.flat());
    const area = cv.contourArea(mat);
    mat.delete();

    const [minArea, maxArea] = expectedAreaRange;

    if (area < minArea || area > maxArea) {
      logger.warn(`Marker area ${area} outside expected range [${minArea}, ${maxArea}]`);
      return false;
    }
  }

  return true;
}

