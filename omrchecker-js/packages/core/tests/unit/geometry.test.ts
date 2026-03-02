import { describe, it, expect } from 'vitest';
import { euclideanDistance, vectorMagnitude, bboxCenter } from '../../src/utils/geometry';
import { Point } from '../../src/utils/math';

describe('Geometry Utils', () => {
  describe('euclideanDistance', () => {
    it('should calculate distance between two points', () => {
      const result = euclideanDistance([0, 0], [3, 4]);
      expect(result).toBe(5);
    });

    it('should handle negative coordinates', () => {
      const result = euclideanDistance([-1, -1], [2, 3]);
      expect(result).toBeCloseTo(5, 5);
    });

    it('should return 0 for same point', () => {
      const result = euclideanDistance([10, 20], [10, 20]);
      expect(result).toBe(0);
    });

    it('should handle floating point coordinates', () => {
      const result = euclideanDistance([1.5, 2.5], [4.5, 6.5]);
      expect(result).toBe(5);
    });
  });

  describe('vectorMagnitude', () => {
    it('should calculate magnitude of 2D vector', () => {
      const result = vectorMagnitude([3, 4]);
      expect(result).toBe(5);
    });

    it('should calculate magnitude of 3D vector', () => {
      const result = vectorMagnitude([2, 3, 6]);
      expect(result).toBe(7);
    });

    it('should return 0 for zero vector', () => {
      const result = vectorMagnitude([0, 0, 0]);
      expect(result).toBe(0);
    });

    it('should handle negative components', () => {
      const result = vectorMagnitude([-3, 4]);
      expect(result).toBe(5);
    });

    it('should calculate magnitude of unit vector', () => {
      const result = vectorMagnitude([1, 0]);
      expect(result).toBe(1);
    });
  });

  describe('bboxCenter', () => {
    it('should calculate center for unit square at origin', () => {
      const origin: Point = [0, 0];
      const dimensions: [number, number] = [2, 2];
      const result = bboxCenter(origin, dimensions);
      expect(result).toEqual([1, 1]);
    });

    it('should handle non-zero origin', () => {
      const origin: Point = [10, 20];
      const dimensions: [number, number] = [30, 40];
      const result = bboxCenter(origin, dimensions);
      expect(result).toEqual([25, 40]);
    });

    it('should handle zero dimensions', () => {
      const origin: Point = [10, 20];
      const dimensions: [number, number] = [0, 0];
      const result = bboxCenter(origin, dimensions);
      expect(result).toEqual([10, 20]);
    });

    it('should handle fractional dimensions', () => {
      const origin: Point = [1.5, 2.5];
      const dimensions: [number, number] = [3, 4];
      const result = bboxCenter(origin, dimensions);
      expect(result).toEqual([3, 4.5]);
    });

    it('should handle large coordinates', () => {
      const origin: Point = [100, 200];
      const dimensions: [number, number] = [800, 600];
      const result = bboxCenter(origin, dimensions);
      expect(result).toEqual([500, 500]);
    });
  });

  describe('Geometry consistency', () => {
    it('should maintain consistency between geometry functions', () => {
      // Two boxes with known centers
      const origin1: Point = [0, 0];
      const dimensions1: [number, number] = [10, 10];
      const center1 = bboxCenter(origin1, dimensions1);

      const origin2: Point = [20, 0];
      const dimensions2: [number, number] = [10, 10];
      const center2 = bboxCenter(origin2, dimensions2);

      // Distance between centers should equal distance between origins + half widths
      const distance = euclideanDistance(center1, center2);
      const expectedDistance = 20; // Centers are at (5, 5) and (25, 5)
      expect(distance).toBeCloseTo(expectedDistance, 5);
    });

    it('should verify Pythagorean theorem', () => {
      // Right triangle with legs 3 and 4, hypotenuse should be 5
      const point1: Point = [0, 0];
      const point2: Point = [3, 0];
      const point3: Point = [0, 4];

      // Calculate all three sides
      const sideA = euclideanDistance(point1, point2);
      const sideB = euclideanDistance(point1, point3);
      const hypotenuse = euclideanDistance(point2, point3);

      // Verify Pythagorean theorem: a² + b² = c²
      expect(sideA ** 2 + sideB ** 2).toBeCloseTo(hypotenuse ** 2, 5);
      expect(hypotenuse).toBe(5);
    });
  });
});
