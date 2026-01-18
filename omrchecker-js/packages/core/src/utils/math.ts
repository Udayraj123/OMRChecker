/**
 * Math utility functions for OMRChecker.
 *
 * TypeScript port of src/utils/math.py
 * A static-only class to hold common math utilities.
 */

/**
 * Point type - [x, y] coordinates
 */
export type Point = [number, number];

/**
 * Rectangle type - array of 4 points: [topLeft, topRight, bottomRight, bottomLeft]
 */
export type Rectangle = [Point, Point, Point, Point];

/**
 * Edge type enum
 */
export enum EdgeType {
  TOP = 'TOP',
  RIGHT = 'RIGHT',
  BOTTOM = 'BOTTOM',
  LEFT = 'LEFT',
}

/**
 * Math utilities for geometric operations and point manipulation.
 */
export class MathUtils {
  static readonly MAX_COSINE = 0.35;

  /**
   * Calculate Euclidean distance between two points.
   *
   * @param point1 - First point [x, y]
   * @param point2 - Second point [x, y]
   * @returns Distance between the points
   */
  static distance(point1: Point, point2: Point): number {
    const dx = point1[0] - point2[0];
    const dy = point1[1] - point2[1];
    return Math.hypot(dx, dy);
  }

  /**
   * Shift points from a new origin.
   *
   * @param newOrigin - The new origin point
   * @param listOfPoints - Array of points to shift
   * @returns Shifted points
   */
  static shiftPointsFromOrigin(newOrigin: Point, listOfPoints: Point[]): Point[] {
    return listOfPoints.map((point) => MathUtils.addPoints(newOrigin, point));
  }

  /**
   * Add two points together.
   *
   * @param newOrigin - First point
   * @param point - Second point
   * @returns Sum of the two points
   */
  static addPoints(newOrigin: Point, point: Point): Point {
    return [newOrigin[0] + point[0], newOrigin[1] + point[1]];
  }

  /**
   * Subtract new origin from a point.
   *
   * @param point - Point to subtract from
   * @param newOrigin - Point to subtract
   * @returns Difference of the two points
   */
  static subtractPoints(point: Point, newOrigin: Point): Point {
    return [point[0] - newOrigin[0], point[1] - newOrigin[1]];
  }

  /**
   * Shift points to a new origin.
   *
   * @param newOrigin - The new origin point
   * @param listOfPoints - Array of points to shift
   * @returns Shifted points
   */
  static shiftPointsToOrigin(newOrigin: Point, listOfPoints: Point[]): Point[] {
    return listOfPoints.map((point) => MathUtils.subtractPoints(point, newOrigin));
  }

  /**
   * Get a point on a line by a given ratio.
   *
   * @param edgeLine - Line defined by [start, end] points
   * @param lengthRatio - Ratio along the line (0 to 1)
   * @returns Point at the given ratio on the line
   */
  static getPointOnLineByRatio(edgeLine: [Point, Point], lengthRatio: number): Point {
    const [start, end] = edgeLine;
    return [
      start[0] + (end[0] - start[0]) * lengthRatio,
      start[1] + (end[1] - start[1]) * lengthRatio,
    ];
  }

  /**
   * Order four points in clockwise order: top-left, top-right, bottom-right, bottom-left.
   *
   * @param points - Array of 4 points
   * @returns Tuple of [ordered points, ordered indices]
   */
  static orderFourPoints(points: Point[]): [Rectangle, number[]] {
    // Calculate sum and diff for each point
    const sumOfPoints = points.map((p) => p[0] + p[1]);
    const diff = points.map((p) => p[0] - p[1]);

    // Find indices
    const minSumIdx = sumOfPoints.indexOf(Math.min(...sumOfPoints));
    const minDiffIdx = diff.indexOf(Math.min(...diff));
    const maxSumIdx = sumOfPoints.indexOf(Math.max(...sumOfPoints));
    const maxDiffIdx = diff.indexOf(Math.max(...diff));

    const orderedIndices = [minSumIdx, minDiffIdx, maxSumIdx, maxDiffIdx];
    const rect: Rectangle = [
      points[minSumIdx],
      points[minDiffIdx],
      points[maxSumIdx],
      points[maxDiffIdx],
    ];

    return [rect, orderedIndices];
  }

  /**
   * Convert points to tuple format with integer coordinates.
   *
   * @param points - Array of points
   * @returns Array of points as integer tuples
   */
  static getTuplePoints(points: Point[]): Point[] {
    return points.map((point) => [Math.floor(point[0]), Math.floor(point[1])]);
  }

  /**
   * Get bounding box of a set of points.
   *
   * @param points - Array of points
   * @returns Tuple of [bounding box as rectangle, dimensions as [width, height]]
   */
  static getBoundingBoxOfPoints(points: Point[]): [Rectangle, [number, number]] {
    const xCoords = points.map((p) => p[0]);
    const yCoords = points.map((p) => p[1]);

    const minX = Math.min(...xCoords);
    const minY = Math.min(...yCoords);
    const maxX = Math.max(...xCoords);
    const maxY = Math.max(...yCoords);

    const boundingBox: Rectangle = [
      [minX, minY],
      [maxX, minY],
      [maxX, maxY],
      [minX, maxY],
    ];

    const boxDimensions: [number, number] = [
      Math.floor(maxX - minX),
      Math.floor(maxY - minY),
    ];

    return [boundingBox, boxDimensions];
  }

  /**
   * Validate if approximation is a valid rectangle.
   *
   * @param approx - Approximated contour points
   * @returns True if valid rectangle
   */
  static validateRect(approx: Point[]): boolean {
    return approx.length === 4 && MathUtils.checkMaxCosine(approx);
  }

  /**
   * Get rectangle points from origin and dimensions.
   *
   * @param origin - Top-left corner [x, y]
   * @param dimensions - [width, height]
   * @returns Rectangle points in order: tl, tr, br, bl
   */
  static getRectanglePointsFromBox(origin: Point, dimensions: [number, number]): Rectangle {
    const [x, y] = origin;
    const [w, h] = dimensions;
    return MathUtils.getRectanglePoints(x, y, w, h);
  }

  /**
   * Get rectangle points from coordinates and dimensions.
   *
   * @param x - Top-left x coordinate
   * @param y - Top-left y coordinate
   * @param w - Width
   * @param h - Height
   * @returns Rectangle points in order: tl, tr, br, bl
   */
  static getRectanglePoints(x: number, y: number, w: number, h: number): Rectangle {
    return [
      [x, y],
      [x + w, y],
      [x + w, y + h],
      [x, y + h],
    ];
  }

  /**
   * Select an edge from a rectangle.
   *
   * @param rectangle - Rectangle with 4 points
   * @param edgeType - Type of edge to select
   * @returns Edge as [start, end] points
   */
  static selectEdgeFromRectangle(rectangle: Rectangle, edgeType: EdgeType): [Point, Point] {
    const [tl, tr, br, bl] = rectangle;

    switch (edgeType) {
      case EdgeType.TOP:
        return [tl, tr];
      case EdgeType.RIGHT:
        return [tr, br];
      case EdgeType.BOTTOM:
        return [br, bl];
      case EdgeType.LEFT:
        return [bl, tl];
      default:
        return [tl, tr];
    }
  }

  /**
   * Check if a point is contained within a rectangle.
   *
   * @param point - Point to check [x, y]
   * @param rect - Rectangle as 4-tuple [x1, y1, x2, y2]
   * @returns True if point is inside rectangle
   */
  static rectangleContains(point: Point, rect: [number, number, number, number]): boolean {
    const rectStart: Point = [rect[0], rect[1]];
    const rectEnd: Point = [rect[2], rect[3]];

    return !(
      point[0] < rectStart[0] ||
      point[1] < rectStart[1] ||
      point[0] > rectEnd[0] ||
      point[1] > rectEnd[1]
    );
  }

  /**
   * Check if the maximum cosine of angles is within acceptable range.
   *
   * @param approx - Array of 4 points
   * @returns True if quadrilateral is approximately a rectangle
   */
  static checkMaxCosine(approx: Point[]): boolean {
    let maxCosine = 0;
    let minCosine = 1.5;

    for (let i = 2; i < 5; i++) {
      const cosine = Math.abs(
        MathUtils.angle(approx[i % 4], approx[i - 2], approx[i - 1])
      );
      maxCosine = Math.max(cosine, maxCosine);
      minCosine = Math.min(cosine, minCosine);
    }

    if (maxCosine >= MathUtils.MAX_COSINE) {
      console.warn('Quadrilateral is not a rectangle.');
      return false;
    }

    return true;
  }

  /**
   * Calculate angle between three points.
   *
   * @param p1 - First point
   * @param p2 - Second point
   * @param p0 - Origin point
   * @returns Cosine of the angle
   */
  static angle(p1: Point, p2: Point, p0: Point): number {
    const dx1 = p1[0] - p0[0];
    const dy1 = p1[1] - p0[1];
    const dx2 = p2[0] - p0[0];
    const dy2 = p2[1] - p0[1];

    return (
      (dx1 * dx2 + dy1 * dy2) /
      Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10)
    );
  }

  /**
   * Check if three points are collinear.
   *
   * @param point1 - First point
   * @param point2 - Second point
   * @param point3 - Third point
   * @returns True if points are collinear
   */
  static checkCollinearPoints(point1: Point, point2: Point, point3: Point): boolean {
    const [x1, y1] = point1;
    const [x2, y2] = point2;
    const [x3, y3] = point3;

    return (y1 - y2) * (x1 - x3) === (y1 - y3) * (x1 - x2);
  }

  /**
   * Split an array into chunks of specified size.
   *
   * @param inputList - Array to split
   * @param chunkSize - Size of each chunk
   * @returns Generator yielding chunks
   */
  static *chunks<T>(inputList: T[], chunkSize: number): Generator<T[]> {
    const size = Math.max(1, chunkSize);
    for (let i = 0; i < inputList.length; i += size) {
      yield inputList.slice(i, i + size);
    }
  }

  /**
   * Convert any color format to BGR tuple.
   *
   * Supports hex colors (#RRGGBB), named colors, and RGB tuples.
   * Returns BGR tuple [B, G, R] for OpenCV compatibility.
   *
   * @param anyColor - Color in any format (hex string, named color, or RGB tuple)
   * @returns BGR tuple [B, G, R]
   */
  static toBgr(anyColor: string | [number, number, number]): [number, number, number] {
    let r: number, g: number, b: number;

    if (Array.isArray(anyColor)) {
      // Already RGB tuple
      [r, g, b] = anyColor;
    } else if (typeof anyColor === 'string') {
      // Hex color or named color
      if (anyColor.startsWith('#')) {
        // Hex color: #RRGGBB
        const hex = anyColor.slice(1);
        r = parseInt(hex.slice(0, 2), 16);
        g = parseInt(hex.slice(2, 4), 16);
        b = parseInt(hex.slice(4, 6), 16);
      } else {
        // Named color - simple mapping for common colors
        const namedColors: Record<string, [number, number, number]> = {
          black: [0, 0, 0],
          white: [255, 255, 255],
          red: [255, 0, 0],
          green: [0, 255, 0],
          blue: [0, 0, 255],
          yellow: [255, 255, 0],
          cyan: [0, 255, 255],
          magenta: [255, 0, 255],
        };
        const color = namedColors[anyColor.toLowerCase()] || [0, 0, 0];
        [r, g, b] = color;
      }
    } else {
      // Default to black
      [r, g, b] = [0, 0, 0];
    }

    // Return BGR tuple for OpenCV
    return [b, g, r];
  }
}

