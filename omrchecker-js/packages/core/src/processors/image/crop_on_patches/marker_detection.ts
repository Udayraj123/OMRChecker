/**
 * Marker detection utilities for OMRChecker patch scanning.
 *
 * Migrated from Python: src/processors/image/crop_on_patches/marker_detection.py
 * Task: omr-6n4
 *
 * Provides functions for preparing marker templates, multi-scale template matching,
 * extracting marker corners, detecting markers in patches, and validating detections.
 */

import cv from '@techstark/opencv-js';
import { ImageUtils } from '../../../utils/image';
import { MathUtils } from '../../../utils/math';

/**
 * Reference zone definition with origin and dimensions.
 */
export interface ReferenceZone {
  origin: [number, number]; // [x, y]
  dimensions: [number, number]; // [width, height]
}

/**
 * Result of multi-scale template matching.
 */
export interface MultiScaleMatchResult {
  position: [number, number] | null; // [x, y] of top-left match location
  optimalMarker: cv.Mat | null; // scaled marker at best match (caller must delete)
  confidence: number; // best match confidence (TM_CCOEFF_NORMED score)
  optimalScalePercent: number | null; // scale percent that gave best match
}

/**
 * Prepare a marker template from a reference image and zone.
 *
 * Crops the ROI defined by referenceZone from referenceImage, optionally resizes it,
 * applies Gaussian blur, normalizes to [0, 255], and optionally applies erode-subtract
 * edge enhancement followed by a second normalization.
 *
 * Port of Python prepare_marker_template.
 *
 * @param referenceImage - Source grayscale image Mat (CV_8UC1)
 * @param referenceZone - Zone specifying origin [x, y] and dimensions [width, height]
 * @param markerDimensions - Optional [width, height] to resize the cropped ROI
 * @param blurKernel - [width, height] for GaussianBlur (default: [5, 5])
 * @param applyErodeSubtract - Whether to apply erode-subtract edge enhancement (default: true)
 * @returns Prepared marker Mat (caller must delete)
 */
export function prepareMarkerTemplate(
  referenceImage: cv.Mat,
  referenceZone: ReferenceZone,
  markerDimensions?: [number, number],
  blurKernel: [number, number] = [5, 5],
  applyErodeSubtract: boolean = true
): cv.Mat {
  const mats: cv.Mat[] = [];
  try {
    const [x, y] = referenceZone.origin;
    const [w, h] = referenceZone.dimensions;

    // Crop ROI using roi() and clone
    const roi = referenceImage.roi(new cv.Rect(x, y, w, h));
    let marker = roi.clone();
    mats.push(marker);
    // roi is a view, not a new Mat allocation; no need to delete separately

    // Resize if dimensions specified
    if (markerDimensions !== undefined) {
      const resized = ImageUtils.resizeSingle(marker, markerDimensions[0], markerDimensions[1]);
      mats.push(resized);
      const oldMarker = marker;
      marker = resized;
      // Remove oldMarker from mats list so it gets deleted now
      mats.splice(mats.indexOf(oldMarker), 1);
      oldMarker.delete();
    }

    // Gaussian blur
    const blurred = new cv.Mat();
    mats.push(blurred);
    cv.GaussianBlur(marker, blurred, new cv.Size(blurKernel[0], blurKernel[1]), 0);
    {
      const oldMarker = marker;
      marker = blurred;
      mats.splice(mats.indexOf(oldMarker), 1);
      oldMarker.delete();
    }

    // Normalize to full [0, 255] range
    const normalized = new cv.Mat();
    mats.push(normalized);
    cv.normalize(marker, normalized, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
    {
      const oldMarker = marker;
      marker = normalized;
      mats.splice(mats.indexOf(oldMarker), 1);
      oldMarker.delete();
    }

    // Optional erode-subtract for edge enhancement
    if (applyErodeSubtract) {
      const kernel = cv.Mat.ones(5, 5, cv.CV_8U);
      mats.push(kernel);

      const eroded = new cv.Mat();
      mats.push(eroded);
      cv.erode(marker, eroded, kernel, new cv.Point(-1, -1), 5);

      const subtracted = new cv.Mat();
      mats.push(subtracted);
      cv.subtract(marker, eroded, subtracted);

      const renormalized = new cv.Mat();
      mats.push(renormalized);
      cv.normalize(subtracted, renormalized, 0, 255, cv.NORM_MINMAX, cv.CV_8U);

      const oldMarker = marker;
      marker = renormalized;
      mats.splice(mats.indexOf(oldMarker), 1);
      oldMarker.delete();
    }

    // Remove marker from cleanup list - caller owns it
    mats.splice(mats.indexOf(marker), 1);
    return marker;
  } finally {
    mats.forEach(m => {
      try {
        m.delete();
      } catch (_) {
        // ignore
      }
    });
  }
}

/**
 * Perform multi-scale template matching of a marker against a patch.
 *
 * Iterates over scale percentages from scaleRange[1] down to scaleRange[0] in steps
 * of (scaleRange[1] - scaleRange[0]) / scaleSteps, resizes the marker at each scale,
 * and runs cv.matchTemplate (TM_CCOEFF_NORMED). Returns the position, scaled marker Mat,
 * confidence, and scale percent for the best match found.
 *
 * Port of Python multi_scale_template_match.
 *
 * @param patch - Grayscale patch image Mat to search in
 * @param marker - Marker template Mat to search for
 * @param scaleRange - [minScale, maxScale] as percentages (default: [85, 115])
 * @param scaleSteps - Number of scale steps to try (default: 5)
 * @returns MultiScaleMatchResult (caller must delete optimalMarker if not null)
 */
export function multiScaleTemplateMatch(
  patch: cv.Mat,
  marker: cv.Mat,
  scaleRange: [number, number] = [85, 115],
  scaleSteps: number = 5
): MultiScaleMatchResult {
  const descentPerStep = Math.floor((scaleRange[1] - scaleRange[0]) / scaleSteps);
  const markerH = marker.rows;
  const markerW = marker.cols;
  const patchH = patch.rows;
  const patchW = patch.cols;

  let bestPosition: [number, number] | null = null;
  let bestMarker: cv.Mat | null = null;
  let bestConfidence = 0.0;
  let bestScalePercent: number | null = null;

  // Iterate from scaleRange[1] down to scaleRange[0] (exclusive)
  for (
    let scalePercent = scaleRange[1];
    scalePercent > scaleRange[0];
    scalePercent -= descentPerStep
  ) {
    const scale = scalePercent / 100;
    if (scale <= 0) continue;

    const scaledW = Math.floor(markerW * scale);
    const scaledH = Math.floor(markerH * scale);

    const scaledMarker = ImageUtils.resizeSingle(marker, scaledW, scaledH);

    if (scaledH > patchH || scaledW > patchW) {
      scaledMarker.delete();
      continue;
    }

    const matchResult = new cv.Mat();
    cv.matchTemplate(patch, scaledMarker, matchResult, cv.TM_CCOEFF_NORMED);

    const minMax = cv.minMaxLoc(matchResult, new cv.Mat());
    const confidence = minMax.maxVal;
    matchResult.delete();

    if (confidence > bestConfidence) {
      // Delete previous best marker
      if (bestMarker !== null) {
        bestMarker.delete();
      }
      bestScalePercent = scalePercent;
      bestMarker = scaledMarker;
      bestConfidence = confidence;
      bestPosition = [minMax.maxLoc.x, minMax.maxLoc.y];
    } else {
      scaledMarker.delete();
    }
  }

  return {
    position: bestPosition,
    optimalMarker: bestMarker,
    confidence: bestConfidence,
    optimalScalePercent: bestScalePercent,
  };
}

/**
 * Extract the four corner points of a matched marker in absolute image coordinates.
 *
 * Given the top-left position of the match and the matched marker Mat, computes the
 * rectangle corners [tl, tr, br, bl] and shifts them by zoneOffset to get absolute
 * coordinates in the original image.
 *
 * Port of Python extract_marker_corners.
 *
 * @param position - [x, y] top-left position of the match in the patch
 * @param marker - Matched marker Mat (used for dimensions only)
 * @param zoneOffset - [x, y] offset of the patch in the original image (default: [0, 0])
 * @returns Array of 4 corner points [[x,y], ...] in [tl, tr, br, bl] order
 */
export function extractMarkerCorners(
  position: [number, number],
  marker: cv.Mat,
  zoneOffset: [number, number] = [0, 0]
): number[][] {
  const h = marker.rows;
  const w = marker.cols;
  const [x, y] = position;
  const corners = MathUtils.getRectanglePoints(x, y, w, h);
  const absoluteCorners = MathUtils.shiftPointsFromOrigin(zoneOffset, corners);
  return absoluteCorners as number[][];
}

/**
 * Detect a marker in a patch image using multi-scale template matching.
 *
 * Runs multiScaleTemplateMatch, checks confidence against minConfidence, and if
 * successful, extracts and returns the four corner points in absolute coordinates.
 * Returns null if no match found or confidence is below threshold.
 *
 * Port of Python detect_marker_in_patch.
 *
 * @param patch - Grayscale patch image Mat to search in
 * @param marker - Marker template Mat to search for
 * @param zoneOffset - [x, y] offset of the patch in the original image (default: [0, 0])
 * @param scaleRange - [minScale, maxScale] as percentages (default: [85, 115])
 * @param scaleSteps - Number of scale steps to try (default: 5)
 * @param minConfidence - Minimum confidence threshold (default: 0.3)
 * @returns Array of 4 corner points or null if detection failed
 */
export function detectMarkerInPatch(
  patch: cv.Mat,
  marker: cv.Mat,
  zoneOffset: [number, number] = [0, 0],
  scaleRange: [number, number] = [85, 115],
  scaleSteps: number = 5,
  minConfidence: number = 0.3
): number[][] | null {
  const { position, optimalMarker, confidence } = multiScaleTemplateMatch(
    patch,
    marker,
    scaleRange,
    scaleSteps
  );

  if (position === null || optimalMarker === null) {
    if (optimalMarker !== null) {
      optimalMarker.delete();
    }
    return null;
  }

  if (confidence < minConfidence) {
    optimalMarker.delete();
    return null;
  }

  const corners = extractMarkerCorners(position, optimalMarker, zoneOffset);
  optimalMarker.delete();
  return corners;
}

/**
 * Validate detected marker corners.
 *
 * Checks that corners is non-null, has exactly 4 points each with 2 coordinates,
 * and optionally verifies the contour area falls within an expected range.
 *
 * Port of Python validate_marker_detection.
 *
 * @param corners - Array of 4 corner points to validate, or null
 * @param expectedAreaRange - Optional [minArea, maxArea] for contour area validation
 * @returns true if corners are valid (and area is in range if specified), false otherwise
 */
export function validateMarkerDetection(
  corners: number[][] | null,
  expectedAreaRange?: [number, number]
): boolean {
  if (corners === null || corners === undefined) return false;
  if (corners.length !== 4) return false;
  if (!corners.every(c => Array.isArray(c) && c.length === 2)) return false;

  if (expectedAreaRange !== null && expectedAreaRange !== undefined) {
    // Calculate area using cv.contourArea with integer coordinates
    const flatData = corners.flat().map(Math.round);
    const contourMat = cv.matFromArray(4, 1, cv.CV_32SC2, flatData);
    const area = cv.contourArea(contourMat);
    contourMat.delete();

    const [minArea, maxArea] = expectedAreaRange;
    if (!(minArea <= area && area <= maxArea)) return false;
  }

  return true;
}
