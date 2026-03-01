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

    it('should handle large distances', () => {
      const result = euclideanDistance([0, 0], [1000, 1000]);
      expect(result).toBeCloseTo(1414.213, 2);
    });

    it('should be symmetric', () => {
      const p1: Point = [10, 20];
      const p2: Point = [30, 40];
      expect(euclideanDistance(p1, p2)).toBe(euclideanDistance(p2, p1));
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

    it('should handle single element vector', () => {
      const result = vectorMagnitude([5]);
      expect(result).toBe(5);
    });

    it('should return 0 for zero vector', () => {
      const result = vectorMagnitude([0, 0, 0]);
      expect(result).toBe(0);
    });

    it('should handle negative components', () => {
      const result = vectorMagnitude([-3, 4]);
      expect(result).toBe(5);
    });

    it('should handle empty vector', () => {
      const result = vectorMagnitude([]);
      expect(result).toBe(0);
    });

    it('should handle floating point components', () => {
      const result = vectorMagnitude([1.5, 2.0]);
      expect(result).toBe(2.5);
    });

    it('should handle high dimensional vectors', () => {
      const result = vectorMagnitude([1, 1, 1, 1, 1]);
      expect(result).toBeCloseTo(Math.sqrt(5), 5);
    });
  });

  describe('bboxCenter', () => {
    it('should calculate center of bounding box', () => {
      const origin: Point = [0, 0];
      const dimensions: [number, number] = [100, 200];
      const result = bboxCenter(origin, dimensions);
      expect(result).toEqual([50, 100]);
    });

    it('should handle non-zero origin', () => {
      const origin: Point = [10, 20];
      const dimensions: [number, number] = [30, 40];
      const result = bboxCenter(origin, dimensions);
      expect(result).toEqual([25, 40]);
    });

    it('should handle negative origin', () => {
      const origin: Point = [-10, -10];
      const dimensions: [number, number] = [20, 20];
      const result = bboxCenter(origin, dimensions);
      expect(result).toEqual([0, 0]);
    });

    it('should handle zero dimensions', () => {
      const origin: Point = [50, 50];
      const dimensions: [number, number] = [0, 0];
      const result = bboxCenter(origin, dimensions);
      expect(result).toEqual([50, 50]);
    });

    it('should handle fractional dimensions', () => {
      const origin: Point = [0, 0];
      const dimensions: [number, number] = [10.5, 20.5];
      const result = bboxCenter(origin, dimensions);
      expect(result).toEqual([5.25, 10.25]);
    });

    it('should handle large coordinates', () => {
      const origin: Point = [1000, 2000];
      const dimensions: [number, number] = [500, 600];
      const result = bboxCenter(origin, dimensions);
      expect(result).toEqual([1250, 2300]);
    });
  });
});
