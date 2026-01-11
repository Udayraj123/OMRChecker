/**
 * Tests for geometry utility functions
 *
 * TypeScript port of src/tests/test_geometry.py
 */

import { describe, it, expect } from 'vitest';
import { euclideanDistance, vectorMagnitude, bboxCenter } from '../geometry';

describe('GeometryUtils', () => {
  describe('euclideanDistance', () => {
    it('should calculate Euclidean distance in 2D', () => {
      const point1 = [0, 0];
      const point2 = [3, 4];
      const distance = euclideanDistance(point1, point2);
      expect(distance).toBe(5.0);
    });

    it('should return zero for identical points', () => {
      const point = [10, 20];
      const distance = euclideanDistance(point, point);
      expect(distance).toBe(0.0);
    });

    it('should handle negative coordinates', () => {
      const point1 = [-3, -4];
      const point2 = [0, 0];
      const distance = euclideanDistance(point1, point2);
      expect(distance).toBe(5.0);
    });

    it('should handle floating point coordinates', () => {
      const point1 = [1.5, 2.5];
      const point2 = [4.5, 6.5];
      const distance = euclideanDistance(point1, point2);
      expect(distance).toBeCloseTo(5.0, 10);
    });
  });

  describe('vectorMagnitude', () => {
    it('should calculate magnitude of zero vector', () => {
      const vector = [0, 0];
      const magnitude = vectorMagnitude(vector);
      expect(magnitude).toBe(0.0);
    });

    it('should calculate magnitude of unit vector', () => {
      const vector = [1, 0];
      const magnitude = vectorMagnitude(vector);
      expect(magnitude).toBe(1.0);
    });

    it('should calculate vector magnitude in 2D', () => {
      const vector = [3, 4];
      const magnitude = vectorMagnitude(vector);
      expect(magnitude).toBe(5.0);
    });

    it('should handle negative components', () => {
      const vector = [-3, -4];
      const magnitude = vectorMagnitude(vector);
      expect(magnitude).toBe(5.0);
    });

    it('should calculate vector magnitude in 3D', () => {
      const vector = [1, 2, 2];
      const magnitude = vectorMagnitude(vector);
      expect(magnitude).toBe(3.0);
    });
  });

  describe('bboxCenter', () => {
    it('should calculate center for unit square at origin', () => {
      const origin = [0, 0];
      const dimensions = [2, 2];
      const center = bboxCenter(origin, dimensions);
      expect(center).toEqual([1.0, 1.0]);
    });

    it('should calculate center for offset bounding box', () => {
      const origin = [10, 20];
      const dimensions = [30, 40];
      const center = bboxCenter(origin, dimensions);
      expect(center).toEqual([25.0, 40.0]);
    });

    it('should handle floating point coordinates', () => {
      const origin = [1.5, 2.5];
      const dimensions = [3.0, 4.0];
      const center = bboxCenter(origin, dimensions);
      expect(center).toEqual([3.0, 4.5]);
    });

    it('should handle zero dimensions (point)', () => {
      const origin = [10, 20];
      const dimensions = [0, 0];
      const center = bboxCenter(origin, dimensions);
      expect(center).toEqual([10.0, 20.0]);
    });

    it('should calculate center for large bounding box', () => {
      const origin = [100, 200];
      const dimensions = [800, 600];
      const center = bboxCenter(origin, dimensions);
      expect(center).toEqual([500.0, 500.0]);
    });
  });

  describe('geometry consistency', () => {
    it('should maintain consistency between functions', () => {
      // Two boxes with known centers
      const origin1 = [0, 0];
      const dimensions1 = [10, 10];
      const center1 = bboxCenter(origin1, dimensions1);

      const origin2 = [20, 0];
      const dimensions2 = [10, 10];
      const center2 = bboxCenter(origin2, dimensions2);

      // Distance between centers should equal distance between origins + half widths
      const distance = euclideanDistance(center1, center2);
      const expectedDistance = 20.0; // Centers are at (5, 5) and (25, 5)
      expect(distance).toBeCloseTo(expectedDistance, 10);
    });

    it('should verify Pythagorean theorem', () => {
      // Right triangle with legs 3 and 4, hypotenuse should be 5
      const point1 = [0, 0];
      const point2 = [3, 0];
      const point3 = [0, 4];

      // Calculate all three sides
      const sideA = euclideanDistance(point1, point2);
      const sideB = euclideanDistance(point1, point3);
      const hypotenuse = euclideanDistance(point2, point3);

      // Verify Pythagorean theorem: a² + b² = c²
      expect(sideA ** 2 + sideB ** 2).toBeCloseTo(hypotenuse ** 2, 10);
      expect(hypotenuse).toBe(5.0);
    });
  });
});

