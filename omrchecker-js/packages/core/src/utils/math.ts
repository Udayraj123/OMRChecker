/**
 * Migrated from Python: src/utils/math.py
 * Agent: Foundation-Alpha
 * Phase: 1
 *
 * Common math utilities for OMRChecker operations.
 * Pure functions for geometric calculations, point transformations, and array operations.
 */

// Type definitions
export type Point = [number, number];
export type Rectangle = [Point, Point, Point, Point]; // [tl, tr, br, bl]
export type Line = [Point, Point];

/**
 * Edge types for rectangle selection
 */
export enum EdgeType {
  TOP = 'TOP',
  RIGHT = 'RIGHT',
  BOTTOM = 'BOTTOM',
  LEFT = 'LEFT',
}

/**
 * Static utility class for mathematical operations
 */
export class MathUtils {
  /** Maximum cosine value for rectangle validation */
  static readonly MAX_COSINE = 0.35;

  /**
   * Calculate Euclidean distance between two points
   */
  static distance(point1: Point, point2: Point): number {
    const dx = point1[0] - point2[0];
    const dy = point1[1] - point2[1];
    return Math.hypot(dx, dy);
  }

  /**
   * Shift points from origin by adding offset
   */
  static shiftPointsFromOrigin(newOrigin: Point, listOfPoints: Point[]): Point[] {
    return listOfPoints.map(point => MathUtils.addPoints(newOrigin, point));
  }

  /**
   * Add two points (vector addition)
   */
  static addPoints(newOrigin: Point, point: Point): Point {
    return [
      newOrigin[0] + point[0],
      newOrigin[1] + point[1],
    ];
  }

  /**
   * Subtract two points (vector subtraction)
   */
  static subtractPoints(point: Point, newOrigin: Point): Point {
    return [
      point[0] - newOrigin[0],
      point[1] - newOrigin[1],
    ];
  }

  /**
   * Shift points to origin by subtracting offset
   */
  static shiftPointsToOrigin(newOrigin: Point, listOfPoints: Point[]): Point[] {
    return listOfPoints.map(point => MathUtils.subtractPoints(point, newOrigin));
  }

  /**
   * Get point on line by length ratio (linear interpolation)
   */
  static getPointOnLineByRatio(edgeLine: Line, lengthRatio: number): Point {
    const [start, end] = edgeLine;
    return [
      start[0] + (end[0] - start[0]) * lengthRatio,
      start[1] + (end[1] - start[1]) * lengthRatio,
    ];
  }

  /**
   * Order four points in rectangle order: [tl, tr, br, bl]
   * Based on sum and diff heuristics
   */
  static orderFourPoints(points: Point[]): { rect: Rectangle; orderedIndices: number[] } {
    if (points.length !== 4) {
      throw new Error(`Expected 4 points, got ${points.length}`);
    }

    // Calculate sum and diff for each point
    const sums = points.map(p => p[0] + p[1]);
    const diffs = points.map(p => p[1] - p[0]);

    // Find indices
    const minSumIdx = sums.indexOf(Math.min(...sums));      // top-left
    const minDiffIdx = diffs.indexOf(Math.min(...diffs));   // top-right
    const maxSumIdx = sums.indexOf(Math.max(...sums));      // bottom-right
    const maxDiffIdx = diffs.indexOf(Math.max(...diffs));   // bottom-left

    const orderedIndices = [minSumIdx, minDiffIdx, maxSumIdx, maxDiffIdx];
    const rect: Rectangle = [
      points[minSumIdx],
      points[minDiffIdx],
      points[maxSumIdx],
      points[maxDiffIdx],
    ];

    return { rect, orderedIndices };
  }

  /**
   * Convert points to integer tuples
   */
  static getTuplePoints(points: Point[]): Array<[number, number]> {
    return points.map(point => [
      Math.floor(point[0]),
      Math.floor(point[1]),
    ]);
  }

  /**
   * Get bounding box of points as rectangle
   * Returns ordered rectangle [tl, tr, br, bl] and dimensions [width, height]
   */
  static getBoundingBoxOfPoints(points: Point[]): {
    boundingBox: Rectangle;
    boxDimensions: [number, number];
  } {
    if (points.length === 0) {
      throw new Error('Cannot get bounding box of empty points array');
    }

    const xs = points.map(p => p[0]);
    const ys = points.map(p => p[1]);

    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const maxX = Math.max(...xs);
    const maxY = Math.max(...ys);

    const boundingBox: Rectangle = [
      [minX, minY], // top-left
      [maxX, minY], // top-right
      [maxX, maxY], // bottom-right
      [minX, maxY], // bottom-left
    ];

    const boxDimensions: [number, number] = [
      Math.floor(maxX - minX),
      Math.floor(maxY - minY),
    ];

    return { boundingBox, boxDimensions };
  }

  /**
   * Validate if 4 points form a valid rectangle
   */
  static validateRect(approx: Point[]): boolean {
    return approx.length === 4 && MathUtils.checkMaxCosine(approx);
  }

  /**
   * Get rectangle points from origin and dimensions
   */
  static getRectanglePointsFromBox(origin: Point, dimensions: [number, number]): Rectangle {
    const [x, y] = origin;
    const [w, h] = dimensions;
    return MathUtils.getRectanglePoints(x, y, w, h);
  }

  /**
   * Get rectangle points from x, y, width, height
   * Returns in order: [tl, tr, br, bl]
   */
  static getRectanglePoints(x: number, y: number, w: number, h: number): Rectangle {
    return [
      [x, y],         // top-left
      [x + w, y],     // top-right
      [x + w, y + h], // bottom-right
      [x, y + h],     // bottom-left
    ];
  }

  /**
   * Select edge from rectangle by type
   */
  static selectEdgeFromRectangle(rectangle: Rectangle, edgeType: EdgeType): Line {
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
        return [tl, tr]; // default to TOP
    }
  }

  /**
   * Check if point is inside rectangle
   * Rectangle format: [[x1, y1], [x2, y2]] or [x1, y1, x2, y2]
   */
  static rectangleContains(point: Point, rect: Point[] | number[]): boolean {
    let rectStart: Point;
    let rectEnd: Point;

    if (rect.length === 2 && Array.isArray(rect[0])) {
      // Format: [[x1, y1], [x2, y2]]
      [rectStart, rectEnd] = rect as [Point, Point];
    } else if (rect.length === 4 && typeof rect[0] === 'number') {
      // Format: [x1, y1, x2, y2]
      const nums = rect as number[];
      rectStart = [nums[0], nums[1]];
      rectEnd = [nums[2], nums[3]];
    } else {
      throw new Error('Invalid rectangle format');
    }

    return !(
      point[0] < rectStart[0] ||
      point[1] < rectStart[1] ||
      point[0] > rectEnd[0] ||
      point[1] > rectEnd[1]
    );
  }

  /**
   * Check if quadrilateral is close to rectangle based on cosine of angles
   * Assumes 4 points
   */
  static checkMaxCosine(approx: Point[]): boolean {
    if (approx.length !== 4) {
      return false;
    }

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
   * Calculate cosine of angle between three points
   * Used for rectangle validation
   */
  static angle(p1: Point, p2: Point, p0: Point): number {
    const dx1 = p1[0] - p0[0];
    const dy1 = p1[1] - p0[1];
    const dx2 = p2[0] - p0[0];
    const dy2 = p2[1] - p0[1];

    const dotProduct = dx1 * dx2 + dy1 * dy2;
    const magnitude = Math.sqrt(
      (dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10
    );

    return dotProduct / magnitude;
  }

  /**
   * Check if three points are collinear
   */
  static checkCollinearPoints(point1: Point, point2: Point, point3: Point): boolean {
    const [x1, y1] = point1;
    const [x2, y2] = point2;
    const [x3, y3] = point3;

    return (y1 - y2) * (x1 - x3) === (y1 - y3) * (x1 - x2);
  }

  /**
   * Convert named color to BGR tuple (for OpenCV compatibility)
   * Browser version: returns RGB, OpenCV.js handles BGR internally
   */
  static toBgr(anyColor: string): [number, number, number] {
    // In browser, we can use CSS color parsing
    const ctx = document.createElement('canvas').getContext('2d');
    if (!ctx) {
      throw new Error('Cannot create canvas context for color conversion');
    }

    ctx.fillStyle = anyColor;
    const computed = ctx.fillStyle;

    // Parse hex color
    const hex = computed.replace('#', '');
    const r = parseInt(hex.substr(0, 2), 16);
    const g = parseInt(hex.substr(2, 2), 16);
    const b = parseInt(hex.substr(4, 2), 16);

    // Return BGR for OpenCV
    return [b, g, r];
  }

  /**
   * Split array into chunks of specified size
   */
  static chunks<T>(inputList: T[], chunkSize: number): T[][] {
    const size = Math.max(1, chunkSize);
    const result: T[][] = [];

    for (let i = 0; i < inputList.length; i += size) {
      result.push(inputList.slice(i, i + size));
    }

    return result;
  }
}
