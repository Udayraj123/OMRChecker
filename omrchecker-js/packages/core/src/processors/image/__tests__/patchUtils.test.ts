/**
 * Tests for patchUtils - patch-based scanning and point detection utilities
 *
 * TypeScript port test coverage for src/processors/image/patch_utils.py
 */

import { describe, it, expect, beforeAll } from 'vitest';
import cv from '@techstark/opencv-js';
import {
  selectPointFromRectangle,
  computeScanZone,
  getEdgeContoursMapFromZonePoints,
  drawZoneContoursAndAnchorShifts,
  drawScanZone,
  type ZoneDescription,
} from '../patchUtils';
import { SelectorType, ZonePreset } from '../../constants';
import type { PointArray } from '../pointUtils';

describe('patchUtils', () => {
  beforeAll(() => {
    // Ensure OpenCV.js is loaded
    if (typeof cv === 'undefined') {
      throw new Error('OpenCV.js not loaded');
    }
  });

  describe('selectPointFromRectangle', () => {
    const rectangle: PointArray = [
      [10, 10], // top-left
      [50, 10], // top-right
      [50, 50], // bottom-right
      [10, 50], // bottom-left
    ];

    it('should select top-left point', () => {
      const point = selectPointFromRectangle(rectangle, SelectorType.SELECT_TOP_LEFT);
      expect(point).toEqual([10, 10]);
    });

    it('should select top-right point', () => {
      const point = selectPointFromRectangle(rectangle, SelectorType.SELECT_TOP_RIGHT);
      expect(point).toEqual([50, 10]);
    });

    it('should select bottom-right point', () => {
      const point = selectPointFromRectangle(rectangle, SelectorType.SELECT_BOTTOM_RIGHT);
      expect(point).toEqual([50, 50]);
    });

    it('should select bottom-left point', () => {
      const point = selectPointFromRectangle(rectangle, SelectorType.SELECT_BOTTOM_LEFT);
      expect(point).toEqual([10, 50]);
    });

    it('should select center point', () => {
      const point = selectPointFromRectangle(rectangle, SelectorType.SELECT_CENTER);
      expect(point).toEqual([30, 30]); // (10+50)/2, (10+50)/2
    });

    it('should return null for invalid selector', () => {
      const point = selectPointFromRectangle(rectangle, 'INVALID_SELECTOR' as any);
      expect(point).toBeNull();
    });

    it('should return null for invalid rectangle', () => {
      const invalidRect: PointArray = [[10, 10], [50, 10]];
      const point = selectPointFromRectangle(invalidRect, SelectorType.SELECT_CENTER);
      expect(point).toBeNull();
    });
  });

  describe('computeScanZone', () => {
    it('should extract zone and compute boundaries', () => {
      // Create a simple test image
      const testImage = new cv.Mat(100, 100, cv.CV_8UC1);
      testImage.setTo(new cv.Scalar(128));

      const zoneDescription: ZoneDescription = {
        label: 'testZone',
        origin: [10, 10],
        dimensions: [40, 40],
        scannerType: 'PATCH_DOT',
      };

      const [zone, zoneStart, zoneEnd] = computeScanZone(testImage, zoneDescription);

      // Verify zone is extracted
      expect(zone).toBeDefined();
      expect(zone.rows).toBe(40);
      expect(zone.cols).toBe(40);

      // Verify boundaries
      expect(zoneStart).toEqual([10, 10]);
      expect(zoneEnd).toEqual([50, 50]);

      // Cleanup
      testImage.delete();
      zone.delete();
    });
  });

  describe('getEdgeContoursMapFromZonePoints', () => {
    it('should build edge contours map from zone points', () => {
      const zonePresetPoints: Record<string, PointArray> = {
        [ZonePreset.topLeftDot]: [[10, 10]],
        [ZonePreset.topRightDot]: [[90, 10]],
        [ZonePreset.bottomRightDot]: [[90, 90]],
        [ZonePreset.bottomLeftDot]: [[10, 90]],
      };

      const edgeMap = getEdgeContoursMapFromZonePoints(zonePresetPoints);

      // Verify all edges have points
      expect(edgeMap.TOP).toBeDefined();
      expect(edgeMap.RIGHT).toBeDefined();
      expect(edgeMap.BOTTOM).toBeDefined();
      expect(edgeMap.LEFT).toBeDefined();

      // Top edge should include topLeftDot and topRightDot
      expect(edgeMap.TOP.length).toBeGreaterThan(0);
      expect(edgeMap.TOP).toContainEqual([10, 10]);
      expect(edgeMap.TOP).toContainEqual([90, 10]);
    });

    it('should handle ALL points from line zones', () => {
      const linePoints: PointArray = [
        [10, 50],
        [20, 50],
        [30, 50],
        [40, 50],
      ];

      const zonePresetPoints: Record<string, PointArray> = {
        [ZonePreset.leftLine]: linePoints,
      };

      const edgeMap = getEdgeContoursMapFromZonePoints(zonePresetPoints);

      // LEFT edge should include all points from leftLine
      expect(edgeMap.LEFT).toEqual(linePoints);
    });

    it('should handle negative indices', () => {
      const linePoints: PointArray = [
        [10, 10],
        [20, 20],
        [30, 30],
      ];

      const zonePresetPoints: Record<string, PointArray> = {
        [ZonePreset.leftLine]: linePoints,
      };

      const edgeMap = getEdgeContoursMapFromZonePoints(zonePresetPoints);

      // TOP edge should include last point of leftLine (negative index -1)
      expect(edgeMap.TOP).toContainEqual([30, 30]);
    });

    it('should return empty arrays for missing zones', () => {
      const zonePresetPoints: Record<string, PointArray> = {};

      const edgeMap = getEdgeContoursMapFromZonePoints(zonePresetPoints);

      expect(edgeMap.TOP).toEqual([]);
      expect(edgeMap.RIGHT).toEqual([]);
      expect(edgeMap.BOTTOM).toEqual([]);
      expect(edgeMap.LEFT).toEqual([]);
    });
  });

  describe('drawZoneContoursAndAnchorShifts', () => {
    it('should draw contours without errors', () => {
      const debugImage = new cv.Mat(200, 200, cv.CV_8UC3);
      debugImage.setTo(new cv.Scalar(255, 255, 255));

      const controlPoints: PointArray = [
        [50, 50],
        [150, 50],
        [150, 150],
        [50, 150],
      ];

      const destinationPoints: PointArray = [
        [60, 60],
        [140, 60],
        [140, 140],
        [60, 140],
      ];

      // Should not throw
      expect(() => {
        drawZoneContoursAndAnchorShifts(debugImage, controlPoints, destinationPoints);
      }).not.toThrow();

      // Cleanup
      debugImage.delete();
    });

    it('should handle two-point line', () => {
      const debugImage = new cv.Mat(200, 200, cv.CV_8UC3);
      debugImage.setTo(new cv.Scalar(255, 255, 255));

      const controlPoints: PointArray = [
        [50, 50],
        [150, 150],
      ];

      const destinationPoints: PointArray = [
        [60, 60],
        [140, 140],
      ];

      expect(() => {
        drawZoneContoursAndAnchorShifts(debugImage, controlPoints, destinationPoints);
      }).not.toThrow();

      debugImage.delete();
    });

    it('should handle single point', () => {
      const debugImage = new cv.Mat(200, 200, cv.CV_8UC3);
      debugImage.setTo(new cv.Scalar(255, 255, 255));

      const controlPoints: PointArray = [[100, 100]];
      const destinationPoints: PointArray = [[110, 110]];

      expect(() => {
        drawZoneContoursAndAnchorShifts(debugImage, controlPoints, destinationPoints);
      }).not.toThrow();

      debugImage.delete();
    });
  });

  describe('drawScanZone', () => {
    it('should draw scan zone with margins', () => {
      const debugImage = new cv.Mat(200, 200, cv.CV_8UC3);
      debugImage.setTo(new cv.Scalar(255, 255, 255));

      const zoneDescription: ZoneDescription = {
        label: 'testZone',
        origin: [50, 50],
        dimensions: [80, 80],
        margins: {
          top: 10,
          right: 10,
          bottom: 10,
          left: 10,
        },
        scannerType: 'PATCH_DOT',
      };

      // Should not throw
      expect(() => {
        drawScanZone(debugImage, zoneDescription);
      }).not.toThrow();

      // Cleanup
      debugImage.delete();
    });

    it('should handle zone without margins', () => {
      const debugImage = new cv.Mat(200, 200, cv.CV_8UC3);
      debugImage.setTo(new cv.Scalar(255, 255, 255));

      const zoneDescription: ZoneDescription = {
        label: 'testZone',
        origin: [50, 50],
        dimensions: [80, 80],
        scannerType: 'PATCH_DOT',
      };

      expect(() => {
        drawScanZone(debugImage, zoneDescription);
      }).not.toThrow();

      debugImage.delete();
    });
  });

  describe('Integration scenarios', () => {
    it('should work together for complete zone processing', () => {
      // Create test image
      const testImage = new cv.Mat(200, 200, cv.CV_8UC1);
      testImage.setTo(new cv.Scalar(128));

      // Define zone
      const zoneDescription: ZoneDescription = {
        label: 'testZone',
        origin: [50, 50],
        dimensions: [80, 80],
        margins: { top: 5, right: 5, bottom: 5, left: 5 },
        scannerType: 'PATCH_DOT',
      };

      // Extract zone
      const [zone, zoneStart, zoneEnd] = computeScanZone(testImage, zoneDescription);

      expect(zone.rows).toBe(80);
      expect(zone.cols).toBe(80);
      expect(zoneStart).toEqual([50, 50]);
      expect(zoneEnd).toEqual([130, 130]);

      // Create mock detected corners
      const corners: PointArray = [
        [zoneStart[0] + 10, zoneStart[1] + 10],
        [zoneEnd[0] - 10, zoneStart[1] + 10],
        [zoneEnd[0] - 10, zoneEnd[1] - 10],
        [zoneStart[0] + 10, zoneEnd[1] - 10],
      ];

      // Select center point
      const center = selectPointFromRectangle(corners, SelectorType.SELECT_CENTER);
      expect(center).toBeDefined();
      expect(center![0]).toBeGreaterThan(zoneStart[0]);
      expect(center![1]).toBeGreaterThan(zoneStart[1]);

      // Cleanup
      testImage.delete();
      zone.delete();
    });
  });
});

