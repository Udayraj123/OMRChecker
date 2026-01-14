/**
 * Dot and Line Detection Module
 *
 * TypeScript port of src/processors/image/dot_line_detection.py
 *
 * Extracted from CropOnDotLines to provide focused, testable detection algorithms.
 *
 * This module handles:
 * - Dot detection using morphological operations
 * - Line detection using morphological operations
 * - Edge detection using Canny
 * - Contour extraction and processing
 */

import * as cv from '@techstark/opencv-js';
import { Logger } from '../../utils/logger';
import { ImageUtils } from '../../utils/ImageUtils';
import { MathUtils, type Point, type Rectangle } from '../../utils/math';
import { ScannerType, EDGE_TYPES_IN_ORDER } from '../constants';

// Logger is defined for future use
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const logger = new Logger('DotLineDetection');

// Default Canny thresholds
const DEFAULT_CANNY_LOW = 55;
const DEFAULT_CANNY_HIGH = 185;

/**
 * Preprocess image zone for dot detection.
 *
 * Steps:
 * 1. Optional Gaussian blur
 * 2. White padding to prevent edge artifacts
 * 3. Morphological opening (erode then dilate)
 * 4. Thresholding and normalization
 *
 * @param zone - Image zone to process
 * @param dotKernel - Morphological structuring element for dots
 * @param dotThreshold - Threshold value for dot detection (darker = lower)
 * @param blurKernel - Optional Gaussian blur kernel size
 * @returns Preprocessed zone ready for contour detection
 */
export function preprocessDotZone(
  zone: cv.Mat,
  dotKernel: cv.Mat,
  dotThreshold: number = 150,
  blurKernel?: [number, number]
): cv.Mat {
  let processed = zone.clone();

  // Optional blur
  if (blurKernel) {
    const blurred = new cv.Mat();
    cv.GaussianBlur(processed, blurred, new cv.Size(blurKernel[0], blurKernel[1]), 0);
    processed.delete();
    processed = blurred;
  }

  // Add white padding to avoid dilations sticking to edges
  const kernelHeight = dotKernel.rows;
  const kernelWidth = dotKernel.cols;
  const [whitePadded, padRange] = ImageUtils.padImageFromCenter(
    processed,
    kernelWidth * 2,
    kernelHeight * 2,
    255
  );
  processed.delete();

  // Morphological opening: removes small noise while preserving dot shapes
  const morphed = new cv.Mat();
  cv.morphologyEx(whitePadded, morphed, cv.MORPH_OPEN, dotKernel, new cv.Point(-1, -1), 3);
  whitePadded.delete();

  // Threshold and normalize
  const thresholded = new cv.Mat();
  cv.threshold(morphed, thresholded, dotThreshold, 255, cv.THRESH_TRUNC);
  morphed.delete();

  const normalized = ImageUtils.normalize([thresholded]) as cv.Mat;
  thresholded.delete();

  // Remove white padding
  const roi = new cv.Rect(
    padRange[2],
    padRange[0],
    padRange[3] - padRange[2],
    padRange[1] - padRange[0]
  );
  const unpadded = normalized.roi(roi);
  const result = unpadded.clone();
  unpadded.delete();
  normalized.delete();

  return result;
}

/**
 * Preprocess image zone for line detection.
 *
 * Steps:
 * 1. Optional Gaussian blur
 * 2. Gamma adjustment to darken lines
 * 3. Thresholding and normalization
 * 4. White padding for morphology
 * 5. Morphological opening
 *
 * @param zone - Image zone to process
 * @param lineKernel - Morphological structuring element for lines
 * @param gammaLow - Gamma value for darkening (<1.0)
 * @param lineThreshold - Threshold value for line detection
 * @param blurKernel - Optional Gaussian blur kernel size
 * @returns Preprocessed zone ready for contour detection
 */
export function preprocessLineZone(
  zone: cv.Mat,
  lineKernel: cv.Mat,
  gammaLow: number,
  lineThreshold: number = 180,
  blurKernel?: [number, number]
): cv.Mat {
  let processed = zone.clone();

  // Optional blur
  if (blurKernel) {
    const blurred = new cv.Mat();
    cv.GaussianBlur(processed, blurred, new cv.Size(blurKernel[0], blurKernel[1]), 0);
    processed.delete();
    processed = blurred;
  }

  // Darken the image to make lines more prominent
  const darker = ImageUtils.adjustGamma(processed, gammaLow);
  processed.delete();

  // Threshold and normalize
  const thresholded = new cv.Mat();
  cv.threshold(darker, thresholded, lineThreshold, 255, cv.THRESH_TRUNC);
  darker.delete();

  const normalized = ImageUtils.normalize([thresholded]) as cv.Mat;
  thresholded.delete();

  // Add white padding for morphology
  const kernelHeight = lineKernel.rows;
  const kernelWidth = lineKernel.cols;
  const [whitePadded, padRange] = ImageUtils.padImageFromCenter(
    normalized,
    kernelWidth * 2,
    kernelHeight * 2,
    255
  );
  normalized.delete();

  // Threshold-normalize again after padding
  const whiteThresholded = new cv.Mat();
  cv.threshold(whitePadded, whiteThresholded, lineThreshold, 255, cv.THRESH_TRUNC);

  const whiteNormalized = ImageUtils.normalize([whiteThresholded]) as cv.Mat;
  whiteThresholded.delete();
  whitePadded.delete();

  // Morphological opening: removes small noise while preserving line shapes
  const lineMorphed = new cv.Mat();
  cv.morphologyEx(
    whiteNormalized,
    lineMorphed,
    cv.MORPH_OPEN,
    lineKernel,
    new cv.Point(-1, -1),
    3
  );
  whiteNormalized.delete();

  // Remove white padding
  const roi = new cv.Rect(
    padRange[2],
    padRange[0],
    padRange[3] - padRange[2],
    padRange[1] - padRange[0]
  );
  const unpadded = lineMorphed.roi(roi);
  const result = unpadded.clone();
  unpadded.delete();
  lineMorphed.delete();

  return result;
}

/**
 * Detect contours in zone using Canny edge detection.
 *
 * @param zone - Preprocessed image zone
 * @param cannyLow - Low threshold for Canny
 * @param cannyHigh - High threshold for Canny
 * @returns List of contours (sorted by area, largest first)
 */
export function detectContoursUsingCanny(
  zone: cv.Mat,
  cannyLow: number = DEFAULT_CANNY_LOW,
  cannyHigh: number = DEFAULT_CANNY_HIGH
): cv.MatVector {
  // Apply Canny edge detection
  const cannyEdges = new cv.Mat();
  cv.Canny(zone, cannyEdges, cannyHigh, cannyLow);

  // Find contours
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(cannyEdges, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
  cannyEdges.delete();
  hierarchy.delete();

  if (contours.size() === 0) {
    return contours;
  }

  // Sort by area (largest first)
  const areas: Array<{ index: number; area: number }> = [];
  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i);
    areas.push({ index: i, area: cv.contourArea(contour) });
  }
  areas.sort((a, b) => b.area - a.area);

  const sortedContours = new cv.MatVector();
  for (let i = 0; i < areas.length; i++) {
    sortedContours.push_back(contours.get(areas[i].index));
  }

  // Cleanup original contours
  for (let i = 0; i < contours.size(); i++) {
    contours.get(i).delete();
  }
  contours.delete();

  return sortedContours;
}

/**
 * Extract corner points and edge contours from detected contour.
 *
 * @param contour - Detected contour (largest from zone)
 * @param scannerType - Type of scanner (PATCH_DOT or PATCH_LINE)
 * @returns Tuple of [ordered_corners, edge_contours_map]
 */
export function extractPatchCornersAndEdges(
  contour: cv.Mat,
  scannerType: string
): [Rectangle, Record<string, Point[]>] {
  // Convert to convex hull
  const boundingHull = new cv.Mat();
  cv.convexHull(contour, boundingHull);

  let patchCorners: Rectangle;

  if (scannerType === ScannerType.PATCH_DOT) {
    // Use axis-aligned bounding rectangle for dots
    const rect = cv.boundingRect(boundingHull);
    patchCorners = MathUtils.getRectanglePoints(rect.x, rect.y, rect.width, rect.height);
  } else if (scannerType === ScannerType.PATCH_LINE) {
    // Use rotated rectangle for lines (handles slight rotations)
    const rotatedRect = cv.minAreaRect(boundingHull);
    const points = cv.RotatedRect.points(rotatedRect);
    patchCorners = [
      [Math.round(points[0].x), Math.round(points[0].y)],
      [Math.round(points[1].x), Math.round(points[1].y)],
      [Math.round(points[2].x), Math.round(points[2].y)],
      [Math.round(points[3].x), Math.round(points[3].y)],
    ];
  } else {
    boundingHull.delete();
    throw new Error(`Unsupported scanner type: ${scannerType}`);
  }

  boundingHull.delete();

  // Order corners
  const [orderedCorners] = MathUtils.orderFourPoints(patchCorners);

  // Create simplified edge contours map
  // For browser compatibility, we skip complex shapely-based splitting
  // and create a basic map with corners only
  const edgeContoursMap: Record<string, Point[]> = {};
  for (const edgeType of EDGE_TYPES_IN_ORDER) {
    edgeContoursMap[edgeType] = [];
  }

  return [orderedCorners, edgeContoursMap];
}

/**
 * Detect dot corners in image zone.
 *
 * Main entry point for dot detection. Combines:
 * 1. Preprocessing
 * 2. Contour detection
 * 3. Corner extraction
 *
 * @param zone - Image zone to search in
 * @param zoneOffset - Offset for absolute coordinates
 * @param dotKernel - Morphological kernel for dot detection
 * @param dotThreshold - Threshold value for dots
 * @param blurKernel - Optional Gaussian blur kernel
 * @returns 4x2 array of corner points in absolute coordinates, or null if not found
 */
export function detectDotCorners(
  zone: cv.Mat,
  zoneOffset: [number, number],
  dotKernel: cv.Mat,
  dotThreshold: number = 150,
  blurKernel?: [number, number]
): Rectangle | null {
  // Preprocess zone
  const preprocessed = preprocessDotZone(zone, dotKernel, dotThreshold, blurKernel);

  // Detect contours
  const contours = detectContoursUsingCanny(preprocessed);
  preprocessed.delete();

  if (contours.size() === 0) {
    contours.delete();
    return null;
  }

  // Extract corners from largest contour
  const largestContour = contours.get(0);
  const [corners] = extractPatchCornersAndEdges(largestContour, ScannerType.PATCH_DOT);

  // Cleanup
  for (let i = 0; i < contours.size(); i++) {
    contours.get(i).delete();
  }
  contours.delete();

  // Convert to absolute coordinates
  const absoluteCorners = MathUtils.shiftPointsFromOrigin(zoneOffset, corners);

  return absoluteCorners as Rectangle;
}

/**
 * Detect line corners and edge contours in image zone.
 *
 * Main entry point for line detection. Combines:
 * 1. Preprocessing
 * 2. Contour detection
 * 3. Corner and edge extraction
 *
 * @param zone - Image zone to search in
 * @param zoneOffset - Offset for absolute coordinates
 * @param lineKernel - Morphological kernel for line detection
 * @param gammaLow - Gamma value for darkening
 * @param lineThreshold - Threshold value for lines
 * @param blurKernel - Optional Gaussian blur kernel
 * @returns Tuple of [corners, edge_contours_map] or [null, null] if not found
 */
export function detectLineCornersAndEdges(
  zone: cv.Mat,
  zoneOffset: [number, number],
  lineKernel: cv.Mat,
  gammaLow: number,
  lineThreshold: number = 180,
  blurKernel?: [number, number]
): [Rectangle | null, Record<string, Point[]> | null] {
  // Preprocess zone
  const preprocessed = preprocessLineZone(zone, lineKernel, gammaLow, lineThreshold, blurKernel);

  // Detect contours
  const contours = detectContoursUsingCanny(preprocessed);
  preprocessed.delete();

  if (contours.size() === 0) {
    contours.delete();
    return [null, null];
  }

  // Extract corners and edges from largest contour
  const largestContour = contours.get(0);
  const [corners, edgeContoursMap] = extractPatchCornersAndEdges(
    largestContour,
    ScannerType.PATCH_LINE
  );

  // Cleanup
  for (let i = 0; i < contours.size(); i++) {
    contours.get(i).delete();
  }
  contours.delete();

  // Convert to absolute coordinates
  const absoluteCorners = MathUtils.shiftPointsFromOrigin(zoneOffset, corners);

  const shiftedEdgeContoursMap: Record<string, Point[]> = {};
  for (const edgeType of EDGE_TYPES_IN_ORDER) {
    shiftedEdgeContoursMap[edgeType] = MathUtils.shiftPointsFromOrigin(
      zoneOffset,
      edgeContoursMap[edgeType]
    );
  }

  return [absoluteCorners as Rectangle, shiftedEdgeContoursMap];
}

/**
 * Validate that blur kernel is smaller than zone.
 *
 * @param zoneShape - [height, width] of zone
 * @param blurKernel - [height, width] of blur kernel
 * @param zoneLabel - Optional label for error messages
 * @returns True if valid
 * @throws Error if kernel is too large
 */
export function validateBlurKernel(
  zoneShape: [number, number],
  blurKernel: [number, number],
  zoneLabel?: string
): boolean {
  const [zoneH, zoneW] = zoneShape;
  const [blurH, blurW] = blurKernel;

  if (!(zoneH > blurH && zoneW > blurW)) {
    const labelStr = zoneLabel ? ` '${zoneLabel}'` : '';
    throw new Error(
      `The zone${labelStr} is smaller than provided blur kernel: [${zoneH}, ${zoneW}] < [${blurH}, ${blurW}]`
    );
  }

  return true;
}

/**
 * Create morphological structuring element.
 *
 * @param shape - 'rect', 'ellipse', or 'cross'
 * @param size - [width, height] of element
 * @returns Structuring element as Mat
 */
export function createStructuringElement(shape: string, size: [number, number]): cv.Mat {
  const shapeMap: Record<string, number> = {
    rect: cv.MORPH_RECT,
    ellipse: cv.MORPH_ELLIPSE,
    cross: cv.MORPH_CROSS,
  };

  if (!(shape in shapeMap)) {
    throw new Error(`Unknown shape: ${shape}. Use ${Object.keys(shapeMap).join(', ')}`);
  }

  return cv.getStructuringElement(shapeMap[shape], new cv.Size(size[0], size[1]));
}

