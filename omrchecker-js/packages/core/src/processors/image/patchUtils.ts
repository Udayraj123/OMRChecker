/**
 * Utilities for patch-based scanning and point detection.
 *
 * TypeScript port of src/processors/image/patch_utils.py
 *
 * This module provides reusable utilities for:
 * - Selecting points from rectangles (corners, centers)
 * - Computing and drawing scan zones
 * - Managing edge contours from zone points
 * - Scan zone validation
 */

import cv from '../../utils/opencv';
import { PointArray } from './pointUtils';
import { ShapeUtils } from '../../utils/shapes';
import { DrawingUtils } from '../../utils/drawing';
import {
  EDGE_TYPES_IN_ORDER,
  TARGET_ENDPOINTS_FOR_EDGES,
  type EdgeTypeValue,
  type SelectorTypeValue,
  SelectorType,
} from '../constants';
import { createColor, type ColorTuple } from '../../utils/constants';

// Type for zone description
export interface ZoneDescription {
  label: string;
  origin?: [number, number];
  dimensions?: [number, number];
  margins?: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
  selector?: SelectorTypeValue;
  scannerType: string;
  maxPoints?: number;
}

/**
 * Select a specific point from a rectangle based on selector type.
 *
 * @param rectangle - Array of 4 corner points [tl, tr, br, bl]
 * @param pointsSelector - Selector type (e.g., "SELECT_CENTER", "SELECT_TOP_LEFT")
 * @returns Selected point as [x, y] or null if selector is invalid
 */
export function selectPointFromRectangle(
  rectangle: PointArray,
  pointsSelector: SelectorTypeValue
): PointArray[0] | null {
  if (rectangle.length !== 4) {
    return null;
  }

  const [tl, tr, br, bl] = rectangle;

  switch (pointsSelector) {
    case SelectorType.SELECT_TOP_LEFT:
      return tl;
    case SelectorType.SELECT_TOP_RIGHT:
      return tr;
    case SelectorType.SELECT_BOTTOM_RIGHT:
      return br;
    case SelectorType.SELECT_BOTTOM_LEFT:
      return bl;
    case SelectorType.SELECT_CENTER:
      return [Math.round((tl[0] + br[0]) / 2), Math.round((tl[1] + br[1]) / 2)];
    default:
      return null;
  }
}

/**
 * Extract image zone and compute zone boundaries.
 *
 * @param image - Source image to extract zone from
 * @param zoneDescription - Dictionary with zone origin, dimensions, and margins
 * @returns Tuple of [zone_image, zone_start_point, zone_end_point]
 */
export function computeScanZone(
  image: cv.Mat,
  zoneDescription: ZoneDescription
): [cv.Mat, [number, number], [number, number]] {
  const [zone, scanZoneRectangle] = ShapeUtils.extractImageFromZoneDescription(
    image,
    zoneDescription
  );

  const zoneStart: [number, number] = scanZoneRectangle[0];
  const zoneEnd: [number, number] = scanZoneRectangle[2];

  return [zone, zoneStart, zoneEnd];
}

/**
 * Build edge contours map from zone points.
 *
 * @param zonePresetPoints - Dictionary mapping zone preset names to point arrays
 * @returns Dictionary mapping EdgeType to list of contour points
 */
export function getEdgeContoursMapFromZonePoints(
  zonePresetPoints: Record<string, PointArray>
): Record<EdgeTypeValue, PointArray> {
  const edgeContoursMap: Record<EdgeTypeValue, PointArray> = {
    TOP: [],
    RIGHT: [],
    BOTTOM: [],
    LEFT: [],
  };

  for (const edgeType of EDGE_TYPES_IN_ORDER) {
    const endpoints = TARGET_ENDPOINTS_FOR_EDGES[edgeType];

    for (const [zonePreset, contourPointIndex] of endpoints) {
      if (zonePreset in zonePresetPoints) {
        const zonePoints = zonePresetPoints[zonePreset];

        if (contourPointIndex === 'ALL') {
          edgeContoursMap[edgeType].push(...zonePoints);
        } else {
          // Handle negative indices
          const index =
            contourPointIndex < 0 ? zonePoints.length + contourPointIndex : contourPointIndex;
          edgeContoursMap[edgeType].push(zonePoints[index]);
        }
      }
    }
  }

  return edgeContoursMap;
}

/**
 * Draw detected contours and alignment arrows for debugging.
 *
 * @param debugImage - Image to draw on
 * @param zoneControlPoints - List of detected control points
 * @param zoneDestinationPoints - List of target destination points
 */
export function drawZoneContoursAndAnchorShifts(
  debugImage: cv.Mat,
  zoneControlPoints: PointArray,
  zoneDestinationPoints: PointArray
): void {
  if (zoneControlPoints.length > 1) {
    if (zoneControlPoints.length === 2) {
      // Draw line if it's just two points
      DrawingUtils.drawContour(debugImage, zoneControlPoints);
    } else {
      // Draw convex hull of the found control points
      const pointsMat = cv.matFromArray(
        zoneControlPoints.length,
        1,
        cv.CV_32SC2,
        zoneControlPoints.flat()
      );
      const hull = new cv.Mat();
      cv.convexHull(pointsMat, hull, false, true);

      // Convert hull back to PointArray
      const hullPoints: PointArray = [];
      for (let i = 0; i < hull.rows; i++) {
        hullPoints.push([hull.intAt(i, 0), hull.intAt(i, 1)]);
      }

      DrawingUtils.drawContour(debugImage, hullPoints);

      pointsMat.delete();
      hull.delete();
    }
  }

  // Draw alignment arrows
  const CLR_DARK_GREEN: ColorTuple = createColor(0, 128, 0);
  DrawingUtils.drawArrows(debugImage, zoneControlPoints, zoneDestinationPoints, CLR_DARK_GREEN, 2);

  // Draw control point boxes
  for (const controlPoint of zoneControlPoints) {
    DrawingUtils.drawBox(
      debugImage,
      controlPoint,
      [20, 20],
      CLR_DARK_GREEN,
      'BOX_FILLED',
      1 / 12,
      1,
      true
    );
  }
}

/**
 * Draw scan zone boundaries on debug image.
 *
 * Draws two rectangles:
 * - Outer rectangle (green): includes margins
 * - Inner rectangle (black): actual scan zone without margins
 *
 * @param debugImage - Image to draw on
 * @param zoneDescription - Dictionary with zone origin, dimensions, and margins
 */
export function drawScanZone(debugImage: cv.Mat, zoneDescription: ZoneDescription): void {
  const scanZoneRectangle = ShapeUtils.computeScanZoneRectangle(zoneDescription, true);
  const scanZoneRectangleWithoutMargins = ShapeUtils.computeScanZoneRectangle(
    zoneDescription,
    false
  );

  const zoneStart = scanZoneRectangle[0];
  const zoneEnd = scanZoneRectangle[2];
  const zoneStartWithoutMargins = scanZoneRectangleWithoutMargins[0];
  const zoneEndWithoutMargins = scanZoneRectangleWithoutMargins[2];

  const CLR_DARK_GREEN: ColorTuple = createColor(0, 128, 0);
  const CLR_NEAR_BLACK: ColorTuple = createColor(20, 20, 20);

  DrawingUtils.drawBoxDiagonal(debugImage, zoneStart, zoneEnd, CLR_DARK_GREEN, 2);

  DrawingUtils.drawBoxDiagonal(
    debugImage,
    zoneStartWithoutMargins,
    zoneEndWithoutMargins,
    CLR_NEAR_BLACK,
    1
  );
}

