/**
 * Unit tests for pure math functions in image.ts
 *
 * Only tests functions that have no cv.Mat dependency (pure computation).
 * OpenCV-dependent functions (resizeSingle, normalizeSingle, etc.) are tested
 * in tests/browser/processors/image-utils.test.ts.
 *
 * Note: Python's test_image_utils.py only tests load_image which is excluded
 * from the browser port (file I/O incompatible with browser context).
 * These tests cover the TypeScript-specific pure math functions.
 */

import { describe, it, expect, vi } from 'vitest';

// image.ts imports @techstark/opencv-js at module level.
// The pure math functions under test (getCroppedWarpedRectanglePoints,
// splitPatchContourOnCorners) never call cv, so an empty mock is sufficient
// to prevent WASM loading from hanging the jsdom environment.
vi.mock('@techstark/opencv-js', () => ({ default: {} }));

import { ImageUtils } from '../../src/utils/image';

describe('ImageUtils (pure math)', () => {
  describe('getCroppedWarpedRectanglePoints', () => {
    it('should return axis-aligned destination points for a square input', () => {
      // Square: tl=(0,0), tr=(100,0), br=(100,100), bl=(0,100)
      const corners = [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
      ];
      const [warpedPoints, [maxWidth, maxHeight]] =
        ImageUtils.getCroppedWarpedRectanglePoints(corners);

      expect(maxWidth).toBe(100);
      expect(maxHeight).toBe(100);
      expect(warpedPoints).toEqual([
        [0, 0],
        [99, 0],
        [99, 99],
        [0, 99],
      ]);
    });

    it('should return correct dimensions for a rectangular input', () => {
      // Rectangle: tl=(0,0), tr=(200,0), br=(200,100), bl=(0,100)
      const corners = [
        [0, 0],
        [200, 0],
        [200, 100],
        [0, 100],
      ];
      const [warpedPoints, [maxWidth, maxHeight]] =
        ImageUtils.getCroppedWarpedRectanglePoints(corners);

      expect(maxWidth).toBe(200);
      expect(maxHeight).toBe(100);
      expect(warpedPoints[0]).toEqual([0, 0]);
      expect(warpedPoints[1]).toEqual([199, 0]);
      expect(warpedPoints[2]).toEqual([199, 99]);
      expect(warpedPoints[3]).toEqual([0, 99]);
    });

    it('should pick max width from top and bottom edges', () => {
      // Trapezoid: wider on top
      // tl=(0,0), tr=(300,0), br=(200,100), bl=(50,100)
      // top edge: dist(tr,tl) = 300
      // bottom edge: dist(br,bl) = 150
      // right edge: dist(tr,br) ≈ 141
      // left edge: dist(tl,bl) ≈ 111
      const corners = [
        [0, 0],     // tl
        [300, 0],   // tr
        [200, 100], // br
        [50, 100],  // bl
      ];
      const [, [maxWidth]] = ImageUtils.getCroppedWarpedRectanglePoints(corners);
      expect(maxWidth).toBe(300); // max of 300 and 150
    });

    it('should pick max height from left and right edges', () => {
      // tl=(0,0), tr=(100,0), br=(100,200), bl=(0,200)
      const corners = [
        [0, 0],
        [100, 0],
        [100, 200],
        [0, 200],
      ];
      const [, [, maxHeight]] = ImageUtils.getCroppedWarpedRectanglePoints(corners);
      expect(maxHeight).toBe(200);
    });

    it('should always return 4 destination points', () => {
      const corners = [[10, 10], [90, 5], [85, 95], [5, 90]];
      const [warpedPoints] = ImageUtils.getCroppedWarpedRectanglePoints(corners);
      expect(warpedPoints).toHaveLength(4);
    });

    it('destination points should start at origin [0,0]', () => {
      const corners = [[10, 10], [110, 10], [110, 60], [10, 60]];
      const [warpedPoints] = ImageUtils.getCroppedWarpedRectanglePoints(corners);
      expect(warpedPoints[0]).toEqual([0, 0]);
    });
  });

  describe('splitPatchContourOnCorners', () => {
    it('should assign points to the correct edge', () => {
      // Square patch: tl=(0,0), tr=(100,0), br=(100,100), bl=(0,100)
      const patchCorners: [number, number][] = [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
      ];
      // A point near the top edge
      const topPoint: [number, number] = [50, 2];
      // A point near the right edge
      const rightPoint: [number, number] = [98, 50];
      // A point near the bottom edge
      const bottomPoint: [number, number] = [50, 98];
      // A point near the left edge
      const leftPoint: [number, number] = [2, 50];

      const { edgeContoursMap } = ImageUtils.splitPatchContourOnCorners(
        patchCorners,
        [topPoint, rightPoint, bottomPoint, leftPoint]
      );

      // Each source point should be in the correct edge bucket
      expect(edgeContoursMap['TOP'].some(p => p[0] === 50 && p[1] === 2)).toBe(true);
      expect(edgeContoursMap['RIGHT'].some(p => p[0] === 98 && p[1] === 50)).toBe(true);
      expect(edgeContoursMap['BOTTOM'].some(p => p[0] === 50 && p[1] === 98)).toBe(true);
      expect(edgeContoursMap['LEFT'].some(p => p[0] === 2 && p[1] === 50)).toBe(true);
    });

    it('should include corner points at start and end of each edge', () => {
      const patchCorners: [number, number][] = [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
      ];
      const { orderedCorners, edgeContoursMap } = ImageUtils.splitPatchContourOnCorners(
        patchCorners,
        []
      );

      // Each edge should start and end with corner points
      // TOP: from orderedCorners[0] (tl) to orderedCorners[1] (tr)
      expect(edgeContoursMap['TOP'][0]).toEqual(orderedCorners[0]);
      expect(edgeContoursMap['TOP'][edgeContoursMap['TOP'].length - 1]).toEqual(orderedCorners[1]);
    });

    it('should return 4 ordered corner points', () => {
      const patchCorners: [number, number][] = [
        [0, 100],    // bl (unordered input)
        [100, 0],    // tr
        [0, 0],      // tl
        [100, 100],  // br
      ];
      const { orderedCorners } = ImageUtils.splitPatchContourOnCorners(patchCorners, []);
      expect(orderedCorners).toHaveLength(4);
    });
  });
});
