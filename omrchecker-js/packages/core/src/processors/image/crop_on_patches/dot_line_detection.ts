/**
 * Dot and line detection utilities for OMRChecker patch scanning.
 *
 * Migrated from Python: src/processors/image/crop_on_patches/dot_line_detection.py
 * Task: omr-ni0
 *
 * Provides functions for preprocessing, detecting, and extracting corners/edges
 * from dot and line scanner patches using OpenCV.js morphological operations,
 * Canny edge detection, and convex hull analysis.
 */

import cv from '@techstark/opencv-js';
import { ScannerType, EDGE_TYPES_IN_ORDER } from '../../constants';
import { ImageUtils } from '../../../utils/image';
import { MathUtils } from '../../../utils/math';

/**
 * Preprocess a zone image to isolate a dot-shaped marker.
 *
 * Optionally blurs the zone, pads it symmetrically, applies morphological opening
 * with the dot kernel, TRUNC-thresholds, and normalizes. The result is cropped
 * back to the original zone dimensions.
 *
 * Port of Python preprocess_dot_zone.
 *
 * @param zone - Grayscale input zone Mat (CV_8UC1)
 * @param dotKernel - Structuring element for morphological opening
 * @param dotThreshold - Threshold value for THRESH_TRUNC (default: 150)
 * @param blurKernel - Optional [width, height] for GaussianBlur before processing
 * @returns Preprocessed zone Mat, same dimensions as input (caller must delete)
 */
export function preprocessDotZone(
  zone: cv.Mat,
  dotKernel: cv.Mat,
  dotThreshold: number = 150,
  blurKernel?: [number, number]
): cv.Mat {
  const mats: cv.Mat[] = [];
  try {
    let working: cv.Mat;

    if (blurKernel !== undefined) {
      const blurred = new cv.Mat();
      mats.push(blurred);
      cv.GaussianBlur(zone, blurred, new cv.Size(blurKernel[0], blurKernel[1]), 0);
      working = blurred;
    } else {
      working = zone;
    }

    const kernelHeight = dotKernel.rows;
    const kernelWidth = dotKernel.cols;

    const { paddedImage, padRange } = ImageUtils.padImageFromCenter(
      working,
      kernelWidth * 2,
      kernelHeight * 2,
      255
    );
    mats.push(paddedImage);

    const morphed = new cv.Mat();
    mats.push(morphed);
    cv.morphologyEx(
      paddedImage,
      morphed,
      cv.MORPH_OPEN,
      dotKernel,
      new cv.Point(-1, -1),
      3
    );

    const thresholded = new cv.Mat();
    mats.push(thresholded);
    cv.threshold(morphed, thresholded, dotThreshold, 255, cv.THRESH_TRUNC);

    const normalised = ImageUtils.normalizeSingle(thresholded);
    mats.push(normalised);

    // Crop back to original dimensions using padRange
    // padRange = [top, bottom, left, right] indices into padded image
    const [rowStart, rowEnd, colStart, colEnd] = padRange;
    const cropped = normalised
      .roi(new cv.Rect(colStart, rowStart, colEnd - colStart, rowEnd - rowStart))
      .clone();

    return cropped;
  } finally {
    mats.forEach(m => m.delete());
  }
}

/**
 * Preprocess a zone image to isolate a line-shaped marker.
 *
 * Optionally blurs, applies gamma correction to darken, TRUNC-thresholds,
 * normalizes, pads symmetrically, applies morphological opening, and crops
 * back to the original zone dimensions.
 *
 * Port of Python preprocess_line_zone.
 *
 * @param zone - Grayscale input zone Mat (CV_8UC1)
 * @param lineKernel - Structuring element for morphological opening
 * @param gammaLow - Gamma value for darkening (< 1.0 darkens the image)
 * @param lineThreshold - Threshold value for THRESH_TRUNC (default: 180)
 * @param blurKernel - Optional [width, height] for GaussianBlur before processing
 * @returns Preprocessed zone Mat, same dimensions as input (caller must delete)
 */
export function preprocessLineZone(
  zone: cv.Mat,
  lineKernel: cv.Mat,
  gammaLow: number,
  lineThreshold: number = 180,
  blurKernel?: [number, number]
): cv.Mat {
  const mats: cv.Mat[] = [];
  try {
    let working: cv.Mat;

    if (blurKernel !== undefined) {
      const blurred = new cv.Mat();
      mats.push(blurred);
      cv.GaussianBlur(zone, blurred, new cv.Size(blurKernel[0], blurKernel[1]), 0);
      working = blurred;
    } else {
      working = zone;
    }

    const darkerImage = ImageUtils.adjustGamma(working, gammaLow);
    mats.push(darkerImage);

    const thresholded = new cv.Mat();
    mats.push(thresholded);
    cv.threshold(darkerImage, thresholded, lineThreshold, 255, cv.THRESH_TRUNC);

    const normalised = ImageUtils.normalizeSingle(thresholded);
    mats.push(normalised);

    const kernelHeight = lineKernel.rows;
    const kernelWidth = lineKernel.cols;

    const { paddedImage, padRange } = ImageUtils.padImageFromCenter(
      normalised,
      kernelWidth * 2,
      kernelHeight * 2,
      255
    );
    mats.push(paddedImage);

    const whiteThresholded = new cv.Mat();
    mats.push(whiteThresholded);
    cv.threshold(paddedImage, whiteThresholded, lineThreshold, 255, cv.THRESH_TRUNC);

    const whiteNormalised = ImageUtils.normalizeSingle(whiteThresholded);
    mats.push(whiteNormalised);

    const lineMorphed = new cv.Mat();
    mats.push(lineMorphed);
    cv.morphologyEx(
      whiteNormalised,
      lineMorphed,
      cv.MORPH_OPEN,
      lineKernel,
      new cv.Point(-1, -1),
      3
    );

    // Crop back to original dimensions using padRange
    const [rowStart, rowEnd, colStart, colEnd] = padRange;
    const cropped = lineMorphed
      .roi(new cv.Rect(colStart, rowStart, colEnd - colStart, rowEnd - rowStart))
      .clone();

    return cropped;
  } finally {
    mats.forEach(m => m.delete());
  }
}

/**
 * Detect contours in a zone image using Canny edge detection.
 *
 * Applies Canny edge detection then finds all contours (RETR_LIST,
 * CHAIN_APPROX_SIMPLE). Contours are sorted by area descending.
 *
 * Port of Python detect_contours_using_canny.
 *
 * NOTE: In Python cv2.Canny(zone, canny_high, canny_low) passes high THEN low.
 * OpenCV.js cv.Canny(src, dst, threshold1, threshold2) uses the same order.
 * So we pass cannyHigh as threshold1 and cannyLow as threshold2.
 *
 * @param zone - Input grayscale image Mat
 * @param cannyLow - Lower Canny threshold (default: 55)
 * @param cannyHigh - Upper Canny threshold (default: 185)
 * @returns Array of contour Mats sorted by area descending (caller must delete each)
 */
export function detectContoursUsingCanny(
  zone: cv.Mat,
  cannyLow: number = 55,
  cannyHigh: number = 185
): cv.Mat[] {
  const edges = new cv.Mat();
  const contourVec = new cv.MatVector();
  const hierarchy = new cv.Mat();

  try {
    // Python: cv2.Canny(zone, canny_high, canny_low) — high then low in call args
    cv.Canny(zone, edges, cannyHigh, cannyLow);
    cv.findContours(edges, contourVec, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    const result: cv.Mat[] = [];
    const size = contourVec.size();
    for (let i = 0; i < size; i++) {
      result.push(contourVec.get(i).clone());
    }

    // Sort by contour area, largest first
    result.sort((a, b) => cv.contourArea(b) - cv.contourArea(a));
    return result;
  } finally {
    edges.delete();
    contourVec.delete();
    hierarchy.delete();
  }
}

/**
 * Extract ordered patch corners and edge contour maps from a detected contour.
 *
 * Builds the convex hull of the contour, computes patch corners based on scanner
 * type (bounding rect for DOT, minimum area rect for LINE), then splits the hull
 * boundary points into four edge groups.
 *
 * Port of Python extract_patch_corners_and_edges.
 *
 * @param contour - A single contour Mat from findContours (int32 data)
 * @param scannerType - One of ScannerType.PATCH_DOT or ScannerType.PATCH_LINE
 * @returns Object with corners (4×2 number array) and edgeContoursMap
 * @throws Error if scannerType is not supported
 */
export function extractPatchCornersAndEdges(
  contour: cv.Mat,
  scannerType: string
): {
  corners: [number, number][];
  edgeContoursMap: Record<string, [number, number][]>;
} {
  // Extract [x, y] points from the contour Mat (int32 pairs in data32S)
  const data = contour.data32S;
  const boundaryPoints: [number, number][] = [];
  for (let i = 0; i < data.length; i += 2) {
    boundaryPoints.push([data[i], data[i + 1]]);
  }

  if (boundaryPoints.length === 0) {
    throw new Error('Contour has no points');
  }

  // Build convex hull from contour points
  const hull = new cv.Mat();
  try {
    cv.convexHull(contour, hull, false, true);

    let patchCorners: [number, number][];

    if (scannerType === ScannerType.PATCH_DOT) {
      const rect = cv.boundingRect(hull);
      patchCorners = MathUtils.getRectanglePoints(rect.x, rect.y, rect.width, rect.height) as [
        number,
        number,
      ][];
    } else if (scannerType === ScannerType.PATCH_LINE) {
      const rotRect = cv.minAreaRect(hull);
      // In OpenCV.js 4.x, cv.boxPoints(rotRect) returns an array of {x, y} objects
      const boxPtsArray = (cv as any).boxPoints(rotRect) as Array<{ x: number; y: number }>;
      patchCorners = boxPtsArray.map(pt => [Math.round(pt.x), Math.round(pt.y)] as [number, number]);
    } else {
      throw new Error(`Unsupported scanner type: ${scannerType}`);
    }

    const { orderedCorners, edgeContoursMap } = ImageUtils.splitPatchContourOnCorners(
      patchCorners,
      boundaryPoints
    );

    return {
      corners: orderedCorners,
      edgeContoursMap,
    };
  } finally {
    hull.delete();
  }
}

/**
 * Detect dot corners in a zone image and return absolute coordinates.
 *
 * Preprocesses the zone for dot detection, finds contours via Canny, extracts
 * patch corners from the largest contour, and shifts them by the zone offset.
 *
 * Port of Python detect_dot_corners.
 *
 * @param zone - Grayscale input zone Mat
 * @param zoneOffset - [x, y] offset of the zone within the full image
 * @param dotKernel - Structuring element for morphological opening
 * @param dotThreshold - Threshold for THRESH_TRUNC (default: 150)
 * @param blurKernel - Optional [width, height] for GaussianBlur
 * @returns 4×2 array of corner coordinates, or null if none found
 */
export function detectDotCorners(
  zone: cv.Mat,
  zoneOffset: [number, number],
  dotKernel: cv.Mat,
  dotThreshold: number = 150,
  blurKernel?: [number, number]
): number[][] | null {
  const preprocessed = preprocessDotZone(zone, dotKernel, dotThreshold, blurKernel);
  const contours = detectContoursUsingCanny(preprocessed);
  preprocessed.delete();

  try {
    if (contours.length === 0) {
      return null;
    }

    const { corners } = extractPatchCornersAndEdges(contours[0], ScannerType.PATCH_DOT);

    if (corners === null || corners.length === 0) {
      return null;
    }

    const absoluteCorners = MathUtils.shiftPointsFromOrigin(zoneOffset, corners);
    return absoluteCorners;
  } finally {
    contours.forEach(c => c.delete());
  }
}

/**
 * Detect line corners and edge contour maps in a zone image.
 *
 * Preprocesses the zone for line detection, finds contours via Canny, extracts
 * patch corners and edge groups from the largest contour, and shifts them by
 * the zone offset.
 *
 * Port of Python detect_line_corners_and_edges.
 *
 * @param zone - Grayscale input zone Mat
 * @param zoneOffset - [x, y] offset of the zone within the full image
 * @param lineKernel - Structuring element for morphological opening
 * @param gammaLow - Gamma correction value (< 1.0 darkens)
 * @param lineThreshold - Threshold for THRESH_TRUNC (default: 180)
 * @param blurKernel - Optional [width, height] for GaussianBlur
 * @returns Object with corners (4×2 array or null) and edgeContoursMap (or null)
 */
export function detectLineCornersAndEdges(
  zone: cv.Mat,
  zoneOffset: [number, number],
  lineKernel: cv.Mat,
  gammaLow: number,
  lineThreshold: number = 180,
  blurKernel?: [number, number]
): {
  corners: number[][] | null;
  edgeContoursMap: Record<string, [number, number][]> | null;
} {
  const preprocessed = preprocessLineZone(zone, lineKernel, gammaLow, lineThreshold, blurKernel);
  const contours = detectContoursUsingCanny(preprocessed);
  preprocessed.delete();

  try {
    if (contours.length === 0) {
      return { corners: null, edgeContoursMap: null };
    }

    const { corners, edgeContoursMap } = extractPatchCornersAndEdges(
      contours[0],
      ScannerType.PATCH_LINE
    );

    if (corners === null || edgeContoursMap === null) {
      return { corners: null, edgeContoursMap: null };
    }

    const absoluteCorners = MathUtils.shiftPointsFromOrigin(zoneOffset, corners);

    const shiftedEdgeContoursMap: Record<string, [number, number][]> = {};
    for (const edgeType of EDGE_TYPES_IN_ORDER) {
      shiftedEdgeContoursMap[edgeType] = MathUtils.shiftPointsFromOrigin(
        zoneOffset,
        edgeContoursMap[edgeType]
      ) as [number, number][];
    }

    return { corners: absoluteCorners, edgeContoursMap: shiftedEdgeContoursMap };
  } finally {
    contours.forEach(c => c.delete());
  }
}

/**
 * Validate that a blur kernel is smaller than the zone dimensions.
 *
 * Raises an error if the zone is not strictly larger than the blur kernel
 * in both dimensions.
 *
 * Port of Python validate_blur_kernel.
 *
 * @param zoneShape - [height, width] of the zone image
 * @param blurKernel - [height, width] of the blur kernel
 * @param zoneLabel - Optional label for the zone used in error messages
 * @returns true if validation passes
 * @throws Error if zone is not larger than blur kernel
 */
export function validateBlurKernel(
  zoneShape: [number, number],
  blurKernel: [number, number],
  zoneLabel: string = ''
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
 * Create a morphological structuring element by shape name.
 *
 * Supports "rect", "ellipse", and "cross" shapes corresponding to
 * cv.MORPH_RECT, cv.MORPH_ELLIPSE, and cv.MORPH_CROSS.
 *
 * Port of Python create_structuring_element.
 *
 * @param shape - Shape name: "rect", "ellipse", or "cross"
 * @param size - [width, height] of the structuring element
 * @returns Structuring element Mat (caller must delete)
 * @throws Error if shape name is not recognized
 */
export function createStructuringElement(shape: string, size: [number, number]): cv.Mat {
  const shapeMap: Record<string, number> = {
    rect: cv.MORPH_RECT,
    ellipse: cv.MORPH_ELLIPSE,
    cross: cv.MORPH_CROSS,
  };

  if (!(shape in shapeMap)) {
    throw new Error(`Unknown shape: ${shape}. Use ${JSON.stringify(Object.keys(shapeMap))}`);
  }

  return cv.getStructuringElement(shapeMap[shape], new cv.Size(size[0], size[1]));
}
