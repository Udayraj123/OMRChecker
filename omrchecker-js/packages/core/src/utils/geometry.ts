/**
 * Geometry utility functions for ML detection and alignment.
 *
 * TypeScript port of src/utils/geometry.py
 * Provides common geometric calculations used across ML processors
 * to maintain consistency and reduce code duplication.
 */

/**
 * Calculate Euclidean distance between two points.
 *
 * @param point1 - First point coordinates [x, y]
 * @param point2 - Second point coordinates [x, y]
 * @returns Euclidean distance between the points
 */
export function euclideanDistance(
  point1: number[],
  point2: number[]
): number {
  return Math.sqrt(
    point1.reduce((sum, val, idx) => sum + (val - point2[idx]) ** 2, 0)
  );
}

/**
 * Calculate magnitude (length) of a vector.
 *
 * @param vector - Vector coordinates (e.g., [dx, dy] for 2D)
 * @returns Vector magnitude
 */
export function vectorMagnitude(vector: number[]): number {
  return Math.sqrt(vector.reduce((sum, x) => sum + x ** 2, 0));
}

/**
 * Calculate center point of a bounding box.
 *
 * @param origin - Bounding box origin [x, y]
 * @param dimensions - Bounding box dimensions [width, height]
 * @returns Center point [x, y]
 */
export function bboxCenter(
  origin: number[],
  dimensions: number[]
): number[] {
  return [
    origin[0] + dimensions[0] / 2,
    origin[1] + dimensions[1] / 2,
  ];
}

