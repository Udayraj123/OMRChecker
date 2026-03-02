/**
 * Migrated from Python: src/processors/image/point_utils.py
 *
 * Point utility classes and functions for image warping and geometric
 * transformations. Pure math on number arrays — no OpenCV dependency.
 */

// A single 2D point as [x, y]
export type Point2D = [number, number];

// An array of 2D points, each row is [x, y]
export type PointArray = number[][];

// Accepted input formats for parsePoints
export type PointSpec = PointArray | [PointArray, PointArray] | string | number[][];

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Return indices that would sort `arr` in ascending order (numpy argsort).
 */
function argsort(arr: number[]): number[] {
  return arr.map((v, i) => [v, i] as [number, number]).sort(([a], [b]) => a - b).map(([, i]) => i);
}

/**
 * Ensure every element of `arr` is a number[] of length 2 (Nx2 check).
 * Returns true when the array is a valid Nx2 point array.
 */
function isNx2(arr: unknown): arr is number[][] {
  return (
    Array.isArray(arr) &&
    arr.every(
      (p: unknown) => Array.isArray(p) && (p as unknown[]).length === 2 && (p as unknown[]).every(v => typeof v === 'number')
    )
  );
}

/**
 * Deep-clone a number[][] so mutations to one copy don't affect the other.
 */
function clonePoints(points: number[][]): number[][] {
  return points.map(p => [...p]);
}

/**
 * Convert any point-like input to a number[][] of float values.
 * Mirrors Python's `_ensure_numpy_array`.
 */
function ensurePointArray(input: unknown): number[][] {
  if (!isNx2(input)) {
    throw new TypeError(`Cannot convert input to Nx2 point array: ${JSON.stringify(input)}`);
  }
  // Return values as-is (already numbers); no float32 truncation needed in JS.
  return input.map(p => [p[0], p[1]]);
}

// ---------------------------------------------------------------------------
// PointParser
// ---------------------------------------------------------------------------

/**
 * Options accepted by PointParser.parsePoints.
 */
export interface ParsePointsOptions {
  templateDimensions?: [number, number];
  pageDimensions?: [number, number];
  context?: Record<string, unknown>;
}

/**
 * Parses various point specifications into a (control, destination) pair of
 * Nx2 number arrays.
 *
 * Mirrors Python's `PointParser` class.
 */
export class PointParser {
  /**
   * Parse a points specification.
   *
   * @param pointsSpec - One of:
   *   - `number[][]`            — an Nx2 array; used as both control and dest
   *   - `[number[][], number[][]]` — a 2-tuple of (control, dest) arrays
   *   - `string`                — a named reference resolved via options
   * @param options - Optional resolution context for string references.
   * @returns Tuple `[controlPoints, destinationPoints]`, each Nx2.
   */
  static parsePoints(
    pointsSpec: PointSpec,
    options: ParsePointsOptions = {}
  ): [number[][], number[][]] {
    const { templateDimensions, pageDimensions, context } = options;

    // String reference
    if (typeof pointsSpec === 'string') {
      return PointParser._parseStringReference(pointsSpec, templateDimensions, pageDimensions, context);
    }

    // 2-element tuple [control, dest] — detect by checking that the outer
    // array has exactly 2 elements and each element is itself an Nx2 array.
    if (
      Array.isArray(pointsSpec) &&
      pointsSpec.length === 2 &&
      isNx2(pointsSpec[0]) &&
      isNx2(pointsSpec[1])
    ) {
      const [control, dest] = pointsSpec as [number[][], number[][]];
      return [ensurePointArray(control), ensurePointArray(dest)];
    }

    // Plain Nx2 array — used as both control and dest
    if (Array.isArray(pointsSpec) && isNx2(pointsSpec)) {
      const points = ensurePointArray(pointsSpec);
      return [points, clonePoints(points)];
    }

    throw new TypeError(
      `Invalid points specification type: ${typeof pointsSpec}. Expected list, tuple, string, or numpy array.`
    );
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private static _parseStringReference(
    reference: string,
    templateDimensions: [number, number] | undefined,
    pageDimensions: [number, number] | undefined,
    context: Record<string, unknown> | undefined
  ): [number[][], number[][]] {
    if (reference === 'template.dimensions') {
      if (templateDimensions == null) {
        throw new Error('template.dimensions reference requires template_dimensions');
      }
      return PointParser._createCornerPoints(templateDimensions);
    }

    if (reference === 'page_dimensions') {
      if (pageDimensions == null) {
        throw new Error('page_dimensions reference requires page_dimensions');
      }
      return PointParser._createCornerPoints(pageDimensions);
    }

    if (context != null && reference in context) {
      return PointParser.parsePoints(context[reference] as PointSpec, {
        templateDimensions,
        pageDimensions,
        context,
      });
    }

    throw new Error(`Unknown point reference: ${reference}`);
  }

  /**
   * Create four corner points for a rectangle of the given dimensions.
   * Order: [[0,0], [w-1,0], [w-1,h-1], [0,h-1]]
   */
  private static _createCornerPoints(
    dimensions: [number, number]
  ): [number[][], number[][]] {
    const [w, h] = dimensions;
    const corners: number[][] = [
      [0, 0],
      [w - 1, 0],
      [w - 1, h - 1],
      [0, h - 1],
    ];
    return [corners, clonePoints(corners)];
  }

  /**
   * Validate that control and destination point arrays are compatible.
   *
   * @param controlPoints  - Nx2 control points.
   * @param destPoints     - Nx2 destination points.
   * @param minPoints      - Minimum required number of points (default 4).
   * @throws Error if validation fails.
   */
  static validatePoints(
    controlPoints: number[][],
    destPoints: number[][],
    minPoints = 4
  ): void {
    if (!isNx2(controlPoints)) {
      throw new Error(
        `control_points must be Nx2 array, got shape [${controlPoints.length}]`
      );
    }
    if (!isNx2(destPoints)) {
      throw new Error(
        `destination_points must be Nx2 array, got shape [${destPoints.length}]`
      );
    }
    if (controlPoints.length !== destPoints.length) {
      throw new Error(
        `Mismatch: ${controlPoints.length} control points vs ${destPoints.length} destination points`
      );
    }
    if (controlPoints.length < minPoints) {
      throw new Error(
        `At least ${minPoints} points required, got ${controlPoints.length}`
      );
    }
  }
}

// ---------------------------------------------------------------------------
// WarpedDimensionsCalculator
// ---------------------------------------------------------------------------

/**
 * Computes output dimensions for warped images.
 *
 * Mirrors Python's `WarpedDimensionsCalculator` class.
 */
export class WarpedDimensionsCalculator {
  /**
   * Calculate output dimensions from a set of destination points.
   *
   * @param points       - Nx2 destination points.
   * @param padding      - Extra pixels added to each side (default 0).
   * @param maxDimension - Optional cap; if either dimension exceeds this value
   *                       both are scaled down proportionally.
   * @returns `[width, height]` as integers.
   */
  static calculateFromPoints(
    points: number[][],
    padding = 0,
    maxDimension?: number
  ): [number, number] {
    const xs = points.map(p => p[0]);
    const ys = points.map(p => p[1]);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    let width = Math.ceil(maxX - minX) + 1 + 2 * padding;
    let height = Math.ceil(maxY - minY) + 1 + 2 * padding;

    if (maxDimension != null) {
      const largest = Math.max(width, height);
      if (width > maxDimension || height > maxDimension) {
        const scale = maxDimension / largest;
        width = Math.floor(width * scale);
        height = Math.floor(height * scale);
      }
    }

    return [width, height];
  }

  /**
   * Scale a pair of dimensions by `scale`.
   *
   * @param dimensions - `[width, height]` input.
   * @param scale      - Scale factor (default 1.0).
   * @returns `[width, height]` after scaling, truncated to integers.
   */
  static calculateFromDimensions(
    dimensions: [number, number],
    scale = 1.0
  ): [number, number] {
    const [w, h] = dimensions;
    return [Math.floor(w * scale), Math.floor(h * scale)];
  }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/**
 * Order four points into [top-left, top-right, bottom-right, bottom-left].
 *
 * The algorithm mirrors Python: sort by y, take top-2 / bottom-2, then sort
 * each pair by x.
 *
 * @param points - Exactly 4 points (Nx2 where N=4).
 * @returns Ordered point array `[tl, tr, br, bl]`.
 * @throws Error if `points.length !== 4`.
 */
export function orderFourPoints(points: number[][]): number[][] {
  if (points.length !== 4) {
    throw new Error(`order_four_points requires exactly 4 points, got ${points.length}`);
  }

  // Sort by y (ascending)
  const ySortedIndices = argsort(points.map(p => p[1]));
  const sortedByY = ySortedIndices.map(i => points[i]);

  const topPoints = sortedByY.slice(0, 2);
  const bottomPoints = sortedByY.slice(2);

  // Within each pair, sort by x (ascending)
  const topXIndices = argsort(topPoints.map(p => p[0]));
  const [topLeft, topRight] = topXIndices.map(i => topPoints[i]);

  const bottomXIndices = argsort(bottomPoints.map(p => p[0]));
  const [bottomLeft, bottomRight] = bottomXIndices.map(i => bottomPoints[i]);

  return [topLeft, topRight, bottomRight, bottomLeft];
}

/**
 * Compute per-point Euclidean distances between two equal-length point arrays.
 *
 * @param points1 - First set of points (Nx2).
 * @param points2 - Second set of points (Nx2).
 * @returns Array of distances, one per point pair.
 * @throws Error if the arrays have different lengths.
 */
export function computePointDistances(
  points1: number[][],
  points2: number[][]
): number[] {
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
 * Compute the axis-aligned bounding box of a set of points.
 *
 * @param points - Nx2 point array.
 * @returns `[minX, minY, maxX, maxY]` with floor/ceil applied to match Python.
 */
export function computeBoundingBox(
  points: number[][]
): [number, number, number, number] {
  const xs = points.map(p => p[0]);
  const ys = points.map(p => p[1]);
  return [
    Math.floor(Math.min(...xs)),
    Math.floor(Math.min(...ys)),
    Math.ceil(Math.max(...xs)),
    Math.ceil(Math.max(...ys)),
  ];
}
