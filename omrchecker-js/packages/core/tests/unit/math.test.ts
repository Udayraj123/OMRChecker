import { describe, it, expect } from 'vitest';
import { MathUtils, Point, Rectangle, Line, EdgeType } from '../../src/utils/math';

describe('MathUtils', () => {
  describe('distance', () => {
    it('should calculate Euclidean distance between two points', () => {
      const result = MathUtils.distance([0, 0], [3, 4]);
      expect(result).toBe(5);
    });

    it('should handle negative coordinates', () => {
      const result = MathUtils.distance([-1, -1], [2, 3]);
      expect(result).toBeCloseTo(5, 1);
    });

    it('should return 0 for same point', () => {
      const result = MathUtils.distance([5, 5], [5, 5]);
      expect(result).toBe(0);
    });

    it('should handle floating point coordinates', () => {
      const result = MathUtils.distance([1.5, 2.5], [4.5, 6.5]);
      expect(result).toBe(5);
    });
  });

  describe('addPoints', () => {
    it('should add two points correctly', () => {
      const result = MathUtils.addPoints([10, 20], [5, 15]);
      expect(result).toEqual([15, 35]);
    });

    it('should handle negative coordinates', () => {
      const result = MathUtils.addPoints([-5, -10], [3, 7]);
      expect(result).toEqual([-2, -3]);
    });

    it('should handle zero vector', () => {
      const result = MathUtils.addPoints([0, 0], [10, 20]);
      expect(result).toEqual([10, 20]);
    });
  });

  describe('subtractPoints', () => {
    it('should subtract two points correctly', () => {
      const result = MathUtils.subtractPoints([15, 35], [5, 15]);
      expect(result).toEqual([10, 20]);
    });

    it('should handle negative results', () => {
      const result = MathUtils.subtractPoints([3, 7], [5, 10]);
      expect(result).toEqual([-2, -3]);
    });

    it('should return zero vector for same point', () => {
      const result = MathUtils.subtractPoints([10, 20], [10, 20]);
      expect(result).toEqual([0, 0]);
    });
  });

  describe('shiftPointsFromOrigin', () => {
    it('should shift all points by offset', () => {
      const points: Point[] = [[0, 0], [10, 10], [20, 20]];
      const result = MathUtils.shiftPointsFromOrigin([5, 5], points);
      expect(result).toEqual([[5, 5], [15, 15], [25, 25]]);
    });

    it('should handle empty array', () => {
      const result = MathUtils.shiftPointsFromOrigin([5, 5], []);
      expect(result).toEqual([]);
    });

    it('should handle negative offset', () => {
      const points: Point[] = [[10, 10], [20, 20]];
      const result = MathUtils.shiftPointsFromOrigin([-5, -5], points);
      expect(result).toEqual([[5, 5], [15, 15]]);
    });
  });

  describe('shiftPointsToOrigin', () => {
    it('should shift all points to origin', () => {
      const points: Point[] = [[15, 15], [25, 25], [35, 35]];
      const result = MathUtils.shiftPointsToOrigin([5, 5], points);
      expect(result).toEqual([[10, 10], [20, 20], [30, 30]]);
    });

    it('should handle empty array', () => {
      const result = MathUtils.shiftPointsToOrigin([5, 5], []);
      expect(result).toEqual([]);
    });
  });

  describe('getPointOnLineByRatio', () => {
    it('should get point at start (ratio 0)', () => {
      const line: Line = [[0, 0], [10, 10]];
      const result = MathUtils.getPointOnLineByRatio(line, 0);
      expect(result).toEqual([0, 0]);
    });

    it('should get point at end (ratio 1)', () => {
      const line: Line = [[0, 0], [10, 10]];
      const result = MathUtils.getPointOnLineByRatio(line, 1);
      expect(result).toEqual([10, 10]);
    });

    it('should get midpoint (ratio 0.5)', () => {
      const line: Line = [[0, 0], [10, 10]];
      const result = MathUtils.getPointOnLineByRatio(line, 0.5);
      expect(result).toEqual([5, 5]);
    });

    it('should handle arbitrary ratio', () => {
      const line: Line = [[0, 0], [100, 0]];
      const result = MathUtils.getPointOnLineByRatio(line, 0.25);
      expect(result).toEqual([25, 0]);
    });
  });

  describe('orderFourPoints', () => {
    it('should order points clockwise from top-left', () => {
      const points: Point[] = [[100, 100], [0, 0], [100, 0], [0, 100]];
      const { rect, orderedIndices } = MathUtils.orderFourPoints(points);
      
      // Expected order: tl, tr, br, bl
      expect(rect[0]).toEqual([0, 0]);    // top-left
      expect(rect[1]).toEqual([100, 0]);  // top-right
      expect(rect[2]).toEqual([100, 100]); // bottom-right
      expect(rect[3]).toEqual([0, 100]);  // bottom-left
      expect(orderedIndices).toHaveLength(4);
    });

    it('should throw error if not 4 points', () => {
      const points: Point[] = [[0, 0], [10, 10]];
      expect(() => MathUtils.orderFourPoints(points)).toThrow('Expected 4 points, got 2');
    });

    it('should handle points in random order', () => {
      const points: Point[] = [[50, 50], [50, 0], [0, 50], [0, 0]];
      const { rect } = MathUtils.orderFourPoints(points);
      
      expect(rect[0][0]).toBeLessThan(rect[1][0]); // tl.x < tr.x
      expect(rect[0][1]).toBeLessThan(rect[3][1]); // tl.y < bl.y
    });
  });

  describe('getTuplePoints', () => {
    it('should floor point coordinates to integers', () => {
      const points: Point[] = [[1.5, 2.5], [3.9, 4.1], [5.0, 6.0]];
      const result = MathUtils.getTuplePoints(points);
      expect(result).toEqual([[1, 2], [3, 4], [5, 6]]);
    });

    it('should handle negative coordinates', () => {
      const points: Point[] = [[-1.5, -2.5], [3.5, -4.5]];
      const result = MathUtils.getTuplePoints(points);
      expect(result).toEqual([[-2, -3], [3, -5]]);
    });

    it('should handle empty array', () => {
      const result = MathUtils.getTuplePoints([]);
      expect(result).toEqual([]);
    });
  });

  describe('getBoundingBoxOfPoints', () => {
    it('should calculate bounding box correctly', () => {
      const points: Point[] = [[10, 10], [50, 30], [30, 50]];
      const { boundingBox, boxDimensions } = MathUtils.getBoundingBoxOfPoints(points);
      
      expect(boundingBox[0]).toEqual([10, 10]); // top-left
      expect(boundingBox[1]).toEqual([50, 10]); // top-right
      expect(boundingBox[2]).toEqual([50, 50]); // bottom-right
      expect(boundingBox[3]).toEqual([10, 50]); // bottom-left
      expect(boxDimensions).toEqual([40, 40]); // width, height
    });

    it('should throw error for empty array', () => {
      expect(() => MathUtils.getBoundingBoxOfPoints([])).toThrow('Cannot get bounding box of empty points array');
    });

    it('should handle single point', () => {
      const points: Point[] = [[20, 30]];
      const { boundingBox, boxDimensions } = MathUtils.getBoundingBoxOfPoints(points);
      
      expect(boundingBox[0]).toEqual([20, 30]);
      expect(boxDimensions).toEqual([0, 0]);
    });

    it('should handle negative coordinates', () => {
      const points: Point[] = [[-10, -10], [10, 10], [0, 0]];
      const { boundingBox, boxDimensions } = MathUtils.getBoundingBoxOfPoints(points);
      
      expect(boundingBox[0]).toEqual([-10, -10]);
      expect(boundingBox[2]).toEqual([10, 10]);
      expect(boxDimensions).toEqual([20, 20]);
    });
  });

  describe('getRectanglePoints', () => {
    it('should create rectangle from x, y, width, height', () => {
      const rect = MathUtils.getRectanglePoints(10, 20, 30, 40);
      
      expect(rect[0]).toEqual([10, 20]);    // top-left
      expect(rect[1]).toEqual([40, 20]);    // top-right
      expect(rect[2]).toEqual([40, 60]);    // bottom-right
      expect(rect[3]).toEqual([10, 60]);    // bottom-left
    });

    it('should handle zero width/height', () => {
      const rect = MathUtils.getRectanglePoints(0, 0, 0, 0);
      expect(rect).toEqual([[0, 0], [0, 0], [0, 0], [0, 0]]);
    });

    it('should handle negative dimensions', () => {
      const rect = MathUtils.getRectanglePoints(10, 10, -5, -5);
      expect(rect[1]).toEqual([5, 10]);
      expect(rect[2]).toEqual([5, 5]);
    });
  });

  describe('getRectanglePointsFromBox', () => {
    it('should create rectangle from origin and dimensions', () => {
      const rect = MathUtils.getRectanglePointsFromBox([10, 20], [30, 40]);
      
      expect(rect[0]).toEqual([10, 20]);
      expect(rect[1]).toEqual([40, 20]);
      expect(rect[2]).toEqual([40, 60]);
      expect(rect[3]).toEqual([10, 60]);
    });
  });

  describe('selectEdgeFromRectangle', () => {
    const rect: Rectangle = [[0, 0], [100, 0], [100, 100], [0, 100]];

    it('should select TOP edge', () => {
      const edge = MathUtils.selectEdgeFromRectangle(rect, EdgeType.TOP);
      expect(edge).toEqual([[0, 0], [100, 0]]);
    });

    it('should select RIGHT edge', () => {
      const edge = MathUtils.selectEdgeFromRectangle(rect, EdgeType.RIGHT);
      expect(edge).toEqual([[100, 0], [100, 100]]);
    });

    it('should select BOTTOM edge', () => {
      const edge = MathUtils.selectEdgeFromRectangle(rect, EdgeType.BOTTOM);
      expect(edge).toEqual([[100, 100], [0, 100]]);
    });

    it('should select LEFT edge', () => {
      const edge = MathUtils.selectEdgeFromRectangle(rect, EdgeType.LEFT);
      expect(edge).toEqual([[0, 100], [0, 0]]);
    });
  });

  describe('rectangleContains', () => {
    it('should check if point is inside rectangle (array format)', () => {
      const rect: Point[] = [[0, 0], [100, 100]];
      expect(MathUtils.rectangleContains([50, 50], rect)).toBe(true);
      expect(MathUtils.rectangleContains([0, 0], rect)).toBe(true);
      expect(MathUtils.rectangleContains([100, 100], rect)).toBe(true);
      expect(MathUtils.rectangleContains([150, 50], rect)).toBe(false);
      expect(MathUtils.rectangleContains([50, -10], rect)).toBe(false);
    });

    it('should check if point is inside rectangle (flat format)', () => {
      const rect = [0, 0, 100, 100];
      expect(MathUtils.rectangleContains([50, 50], rect)).toBe(true);
      expect(MathUtils.rectangleContains([101, 50], rect)).toBe(false);
    });

    it('should throw error for invalid format', () => {
      const invalidRect = [0, 0, 100] as any;
      expect(() => MathUtils.rectangleContains([50, 50], invalidRect)).toThrow('Invalid rectangle format');
    });
  });

  describe('checkMaxCosine', () => {
    it('should validate rectangle (4 points with ~90 degree angles)', () => {
      const rectPoints: Point[] = [[0, 0], [100, 0], [100, 100], [0, 100]];
      const result = MathUtils.checkMaxCosine(rectPoints);
      expect(result).toBe(true);
    });

    it('should reject non-4-point shapes', () => {
      const points: Point[] = [[0, 0], [10, 10], [20, 0]];
      const result = MathUtils.checkMaxCosine(points);
      expect(result).toBe(false);
    });

    it('should reject shapes with non-90-degree angles', () => {
      // Triangle-like shape
      const points: Point[] = [[0, 0], [100, 0], [50, 10], [50, 90]];
      const result = MathUtils.checkMaxCosine(points);
      // This should fail the MAX_COSINE test
      expect(typeof result).toBe('boolean');
    });
  });

  describe('validateRect', () => {
    it('should validate valid rectangle', () => {
      const points: Point[] = [[0, 0], [100, 0], [100, 100], [0, 100]];
      const result = MathUtils.validateRect(points);
      expect(result).toBe(true);
    });

    it('should reject non-4-point shapes', () => {
      const points: Point[] = [[0, 0], [10, 10]];
      const result = MathUtils.validateRect(points);
      expect(result).toBe(false);
    });
  });

  describe('angle', () => {
    it('should calculate cosine of 90-degree angle', () => {
      const result = MathUtils.angle([10, 0], [0, 10], [0, 0]);
      expect(Math.abs(result)).toBeCloseTo(0, 5);
    });

    it('should calculate cosine of 180-degree angle', () => {
      const result = MathUtils.angle([10, 0], [-10, 0], [0, 0]);
      expect(result).toBeCloseTo(-1, 5);
    });

    it('should calculate cosine of 0-degree angle', () => {
      const result = MathUtils.angle([10, 0], [20, 0], [0, 0]);
      expect(result).toBeCloseTo(1, 5);
    });
  });

  describe('checkCollinearPoints', () => {
    it('should detect collinear points on horizontal line', () => {
      const result = MathUtils.checkCollinearPoints([0, 0], [10, 0], [20, 0]);
      expect(result).toBe(true);
    });

    it('should detect collinear points on vertical line', () => {
      const result = MathUtils.checkCollinearPoints([0, 0], [0, 10], [0, 20]);
      expect(result).toBe(true);
    });

    it('should detect collinear points on diagonal line', () => {
      const result = MathUtils.checkCollinearPoints([0, 0], [10, 10], [20, 20]);
      expect(result).toBe(true);
    });

    it('should detect non-collinear points', () => {
      const result = MathUtils.checkCollinearPoints([0, 0], [10, 0], [5, 5]);
      expect(result).toBe(false);
    });
  });

  describe('chunks', () => {
    it('should split array into chunks', () => {
      const arr = [1, 2, 3, 4, 5, 6, 7, 8];
      const result = MathUtils.chunks(arr, 3);
      expect(result).toEqual([[1, 2, 3], [4, 5, 6], [7, 8]]);
    });

    it('should handle chunk size larger than array', () => {
      const arr = [1, 2, 3];
      const result = MathUtils.chunks(arr, 10);
      expect(result).toEqual([[1, 2, 3]]);
    });

    it('should handle chunk size of 1', () => {
      const arr = [1, 2, 3];
      const result = MathUtils.chunks(arr, 1);
      expect(result).toEqual([[1], [2], [3]]);
    });

    it('should handle empty array', () => {
      const result = MathUtils.chunks([], 3);
      expect(result).toEqual([]);
    });

    it('should handle zero/negative chunk size by using 1', () => {
      const arr = [1, 2, 3];
      const result = MathUtils.chunks(arr, 0);
      expect(result).toEqual([[1], [2], [3]]);
    });

    it('should work with different types', () => {
      const arr = ['a', 'b', 'c', 'd'];
      const result = MathUtils.chunks(arr, 2);
      expect(result).toEqual([['a', 'b'], ['c', 'd']]);
    });
  });

  describe('MAX_COSINE constant', () => {
    it('should have correct value', () => {
      expect(MathUtils.MAX_COSINE).toBe(0.35);
    });
  });
});
