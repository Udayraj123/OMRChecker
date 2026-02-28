/**
 * Migrated from Python: src/utils/geometry.py
 * Agent: Foundation-Alpha
 * Phase: 1
 *
 * Geometry utility functions for ML detection and alignment.
 * Provides common geometric calculations used across ML processors
 * to maintain consistency and reduce code duplication.
 */

import { Point } from './math';

/**
 * Calculate Euclidean distance between two points
 * 
 * @param point1 - First point coordinates [x, y]
 * @param point2 - Second point coordinates [x, y]
 * @returns Euclidean distance between the points
 */
export function euclideanDistance(point1: Point, point2: Point): number {
  const dx = point1[0] - point2[0];
  const dy = point1[1] - point2[1];
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Calculate magnitude (length) of a vector
 * 
 * @param vector - Vector coordinates (e.g., [dx, dy] for 2D)
 * @returns Vector magnitude
 */
export function vectorMagnitude(vector: number[]): number {
  return Math.sqrt(vector.reduce((sum, x) => sum + x * x, 0));
}

/**
 * Calculate center point of a bounding box
 * 
 * @param origin - Bounding box origin [x, y]
 * @param dimensions - Bounding box dimensions [width, height]
 * @returns Center point [x, y]
 */
export function bboxCenter(
  origin: Point,
  dimensions: [number, number]
): Point {
  return [
    origin[0] + dimensions[0] / 2,
    origin[1] + dimensions[1] / 2,
  ];
}
