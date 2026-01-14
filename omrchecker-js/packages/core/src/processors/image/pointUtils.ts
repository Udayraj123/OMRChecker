/**
 * Point Parsing Utilities for Image Warping
 *
 * Extracted from WarpOnPointsCommon to handle parsing and validation
 * of control and destination points from various formats.
 */

import { logger } from '../../utils/logger';

export type Point = [number, number];
export type PointArray = Point[];

/**
 * Parse and validate point specifications for warping operations.
 *
 * Handles different formats:
 * - Direct arrays: [[x1,y1], [x2,y2], ...]
 * - String references: "template.dimensions", "page_dimensions"
 * - Tuples of arrays: [control_points, destination_points]
 */
export class PointParser {
  /**
   * Parse point specification into control and destination arrays.
   *
   * @param pointsSpec - Point specification in various formats
   * @param templateDimensions - Template [width, height] for reference
   * @param pageDimensions - Page [width, height] for reference
   * @param context - Additional context for resolving references
   * @returns Tuple of [controlPoints, destinationPoints]
   * @throws Error if pointsSpec format is invalid
   */
  static parsePoints(
    pointsSpec: PointArray | [PointArray, PointArray] | string,
    templateDimensions?: [number, number],
    pageDimensions?: [number, number],
    context?: Record<string, any>
  ): [PointArray, PointArray] {
    if (typeof pointsSpec === 'string') {
      return this.parseStringReference(
        pointsSpec,
        templateDimensions,
        pageDimensions,
        context
      );
    }

    // Check if it's a tuple of [control, destination]
    if (
      Array.isArray(pointsSpec) &&
      pointsSpec.length === 2 &&
      Array.isArray(pointsSpec[0]) &&
      Array.isArray(pointsSpec[1]) &&
      Array.isArray(pointsSpec[0][0])
    ) {
      const [control, dest] = pointsSpec as [PointArray, PointArray];
      return [control, dest];
    }

    // Single array - use as both control and destination
    if (Array.isArray(pointsSpec)) {
      const points = pointsSpec as PointArray;
      return [points, points.map((p) => [...p] as Point)];
    }

    throw new Error(
      `Invalid points specification type. Expected array, tuple, or string.`
    );
  }

  /**
   * Parse string reference to points.
   *
   * Supported references:
   * - "template.dimensions" -> corners of template
   * - "page_dimensions" -> corners of page
   */
  private static parseStringReference(
    reference: string,
    templateDimensions?: [number, number],
    pageDimensions?: [number, number],
    context?: Record<string, any>
  ): [PointArray, PointArray] {
    if (reference === 'template.dimensions') {
      if (!templateDimensions) {
        throw new Error('template.dimensions reference requires templateDimensions');
      }
      return this.createCornerPoints(templateDimensions);
    }

    if (reference === 'page_dimensions') {
      if (!pageDimensions) {
        throw new Error('page_dimensions reference requires pageDimensions');
      }
      return this.createCornerPoints(pageDimensions);
    }

    // Try to resolve from context
    if (context && reference in context) {
      const value = context[reference];
      return this.parsePoints(value, templateDimensions, pageDimensions, context);
    }

    throw new Error(`Unknown point reference: ${reference}`);
  }

  /**
   * Create corner points for a rectangle.
   *
   * @param dimensions - [width, height]
   * @returns Tuple of [corners, corners] for control and destination
   */
  private static createCornerPoints(
    dimensions: [number, number]
  ): [PointArray, PointArray] {
    const [w, h] = dimensions;
    const corners: PointArray = [
      [0, 0], // top-left
      [w - 1, 0], // top-right
      [w - 1, h - 1], // bottom-right
      [0, h - 1], // bottom-left
    ];

    return [corners, corners.map((p) => [...p] as Point)];
  }

  /**
   * Validate that point arrays are properly formed.
   *
   * @param controlPoints - Source points
   * @param destinationPoints - Target points
   * @param minPoints - Minimum number of required points
   * @throws Error if validation fails
   */
  static validatePoints(
    controlPoints: PointArray,
    destinationPoints: PointArray,
    minPoints: number = 4
  ): void {
    // Check that all points are 2D
    for (const point of controlPoints) {
      if (point.length !== 2) {
        throw new Error(`control_points must be Nx2 array, got point with length ${point.length}`);
      }
    }

    for (const point of destinationPoints) {
      if (point.length !== 2) {
        throw new Error(
          `destination_points must be Nx2 array, got point with length ${point.length}`
        );
      }
    }

    // Check count match
    if (controlPoints.length !== destinationPoints.length) {
      throw new Error(
        `Mismatch: ${controlPoints.length} control points vs ${destinationPoints.length} destination points`
      );
    }

    // Check minimum
    if (controlPoints.length < minPoints) {
      throw new Error(`At least ${minPoints} points required, got ${controlPoints.length}`);
    }

    logger.debug(`Validated ${controlPoints.length} point pairs`);
  }
}

/**
 * Calculate appropriate dimensions for warped images.
 *
 * Determines output image size based on destination points and
 * optional constraints.
 */
export class WarpedDimensionsCalculator {
  /**
   * Calculate warped dimensions from destination points.
   *
   * @param destinationPoints - Target points array
   * @param padding - Extra padding to add
   * @param maxDimension - Maximum width or height
   * @returns [width, height] tuple
   */
  static calculateFromPoints(
    destinationPoints: PointArray,
    padding: number = 0,
    maxDimension?: number
  ): [number, number] {
    // Find bounding box
    const xs = destinationPoints.map((p) => p[0]);
    const ys = destinationPoints.map((p) => p[1]);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    let width = Math.ceil(maxX - minX) + 1 + 2 * padding;
    let height = Math.ceil(maxY - minY) + 1 + 2 * padding;

    // Apply max dimension constraint if provided
    if (maxDimension !== undefined) {
      if (width > maxDimension || height > maxDimension) {
        const scale = maxDimension / Math.max(width, height);
        width = Math.floor(width * scale);
        height = Math.floor(height * scale);
        logger.debug(`Scaled dimensions to fit maxDimension=${maxDimension}`);
      }
    }

    logger.debug(`Calculated warped dimensions: ${width}x${height}`);
    return [width, height];
  }

  /**
   * Calculate warped dimensions from explicit dimensions.
   *
   * @param dimensions - [width, height]
   * @param scale - Scaling factor
   * @returns [scaledWidth, scaledHeight]
   */
  static calculateFromDimensions(
    dimensions: [number, number],
    scale: number = 1.0
  ): [number, number] {
    const [w, h] = dimensions;
    const scaledW = Math.floor(w * scale);
    const scaledH = Math.floor(h * scale);

    logger.debug(`Dimensions ${w}x${h} scaled by ${scale} -> ${scaledW}x${scaledH}`);
    return [scaledW, scaledH];
  }
}

/**
 * Order 4 points in consistent order: TL, TR, BR, BL.
 *
 * @param points - 4x2 array of points (in any order)
 * @returns Array ordered as [top-left, top-right, bottom-right, bottom-left]
 * @throws Error if not exactly 4 points
 */
export function orderFourPoints(points: PointArray): PointArray {
  if (points.length !== 4) {
    throw new Error(`orderFourPoints requires exactly 4 points, got ${points.length}`);
  }

  // Sort by y-coordinate to get top 2 and bottom 2
  const sortedByY = [...points].sort((a, b) => a[1] - b[1]);

  const topPoints = sortedByY.slice(0, 2);
  const bottomPoints = sortedByY.slice(2, 4);

  // Sort each pair by x-coordinate
  const [topLeft, topRight] = topPoints.sort((a, b) => a[0] - b[0]);
  const [bottomLeft, bottomRight] = bottomPoints.sort((a, b) => a[0] - b[0]);

  return [topLeft, topRight, bottomRight, bottomLeft];
}

/**
 * Compute Euclidean distances between corresponding points.
 *
 * @param points1 - First array of points
 * @param points2 - Second array of points
 * @returns Array of distances
 */
export function computePointDistances(points1: PointArray, points2: PointArray): number[] {
  if (points1.length !== points2.length) {
    throw new Error('Point arrays must have same length');
  }

  return points1.map((p1, i) => {
    const p2 = points2[i];
    const dx = p2[0] - p1[0];
    const dy = p2[1] - p1[1];
    return Math.sqrt(dx * dx + dy * dy);
  });
}

/**
 * Compute axis-aligned bounding box for points.
 *
 * @param points - Array of points
 * @returns [minX, minY, maxX, maxY] tuple
 */
export function computeBoundingBox(points: PointArray): [number, number, number, number] {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);

  const minX = Math.floor(Math.min(...xs));
  const maxX = Math.ceil(Math.max(...xs));
  const minY = Math.floor(Math.min(...ys));
  const maxY = Math.ceil(Math.max(...ys));

  return [minX, minY, maxX, maxY];
}

