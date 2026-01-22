/**
 * Tests for marker detection module.
 *
 * TypeScript port of tests/processors/image/test_marker_detection.py
 *
 * Tests cover:
 * - Marker template preparation
 * - Multi-scale template matching
 * - Corner extraction
 * - Full detection pipeline
 * - Edge cases and validation
 */

const cv = global.cv;
import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import {
  prepareMarkerTemplate,
  multiScaleTemplateMatch,
  extractMarkerCorners,
  detectMarkerInPatch,
  validateMarkerDetection,
  type ReferenceZone,
} from '../markerDetection';

describe('MarkerDetection', () => {
  let matsToCleanup: cv.Mat[] = [];

  beforeAll(async () => {
    // Ensure OpenCV is loaded
    if (typeof cv.Mat === 'undefined') {
      throw new Error('OpenCV.js not loaded');
    }
  });

  afterEach(() => {
    // Cleanup all mats created during tests
    matsToCleanup.forEach((mat) => {
      try {
        mat.delete();
      } catch (e) {
        // Already deleted
      }
    });
    matsToCleanup = [];
  });

  const track = (mat: cv.Mat): cv.Mat => {
    matsToCleanup.push(mat);
    return mat;
  };

  describe('prepareMarkerTemplate', () => {
    it('should extract basic marker from reference', () => {
      // Create a simple reference image
      const reference = track(new cv.Mat(100, 100, cv.CV_8UC1));
      reference.setTo(new cv.Scalar(0));
      // Draw white rectangle
      const rect = new cv.Rect(30, 20, 20, 20);
      const roi = reference.roi(rect);
      roi.setTo(new cv.Scalar(255));
      roi.delete();

      const referenceZone: ReferenceZone = {
        origin: [30, 20],
        dimensions: [20, 20],
      };

      const marker = prepareMarkerTemplate(
        reference,
        referenceZone,
        undefined,
        [3, 3],
        false
      );
      track(marker);

      expect(marker).toBeDefined();
      expect(marker.rows).toBe(20);
      expect(marker.cols).toBe(20);
      expect(marker.type()).toBe(cv.CV_8UC1);
    });

    it('should resize marker when dimensions specified', () => {
      const reference = track(new cv.Mat(100, 100, cv.CV_8UC1));
      reference.setTo(new cv.Scalar(128));

      const referenceZone: ReferenceZone = {
        origin: [10, 10],
        dimensions: [40, 30],
      };

      const marker = prepareMarkerTemplate(
        reference,
        referenceZone,
        [20, 20],
        [3, 3],
        false
      );
      track(marker);

      expect(marker.rows).toBe(20);
      expect(marker.cols).toBe(20);
    });

    it('should normalize marker output', () => {
      // Create reference with gradient
      const reference = track(new cv.Mat(60, 60, cv.CV_8UC1));
      reference.setTo(new cv.Scalar(0));

      // Create a gradient
      for (let i = 10; i < 50; i++) {
        for (let j = 10; j < 50; j++) {
          const value = Math.floor(((i - 10) / 40) * 255);
          reference.ucharPtr(i, j)[0] = value;
        }
      }

      const referenceZone: ReferenceZone = {
        origin: [10, 10],
        dimensions: [40, 40],
      };

      const marker = prepareMarkerTemplate(
        reference,
        referenceZone,
        undefined,
        [5, 5],
        false
      );
      track(marker);

      // Check normalization - should have max value at 255
      const result = cv.minMaxLoc(marker);
      expect(result.maxVal).toBe(255);
      expect(result.minVal).toBe(0);
    });
  });

  describe('multiScaleTemplateMatch', () => {
    const createTestPatchAndMarker = (scale: number = 1.0): [cv.Mat, cv.Mat] => {
      // Create marker (simple white square)
      const marker = track(new cv.Mat(20, 20, cv.CV_8UC1));
      marker.setTo(new cv.Scalar(0));
      const markerRect = new cv.Rect(5, 5, 10, 10);
      const markerRoi = marker.roi(markerRect);
      markerRoi.setTo(new cv.Scalar(255));
      markerRoi.delete();

      // Create patch
      const patch = track(new cv.Mat(100, 100, cv.CV_8UC1));
      patch.setTo(new cv.Scalar(0));

      // Embed scaled marker at position (30, 40)
      let markerScaled: cv.Mat;
      if (scale !== 1.0) {
        markerScaled = new cv.Mat();
        const size = new cv.Size(Math.round(20 * scale), Math.round(20 * scale));
        cv.resize(marker, markerScaled, size);
      } else {
        markerScaled = marker.clone();
      }

      const h = markerScaled.rows;
      const w = markerScaled.cols;
      const patchRect = new cv.Rect(30, 40, w, h);
      const patchRoi = patch.roi(patchRect);
      markerScaled.copyTo(patchRoi);
      patchRoi.delete();
      markerScaled.delete();

      return [patch, marker];
    };

    it('should find perfect match at exact scale', () => {
      const [patch, marker] = createTestPatchAndMarker(1.0);

      const result = multiScaleTemplateMatch(patch, marker, [90, 110], 5);

      expect(result.position).toBeDefined();
      expect(result.optimalMarker).toBeDefined();
      expect(result.confidence).toBeGreaterThan(0.9);
      expect(result.scalePercent).toBeGreaterThanOrEqual(95);
      expect(result.scalePercent).toBeLessThanOrEqual(105);

      if (result.optimalMarker) {
        result.optimalMarker.delete();
      }
    });

    it('should find scaled match', () => {
      const [patch, marker] = createTestPatchAndMarker(0.9);

      const result = multiScaleTemplateMatch(patch, marker, [80, 100], 5);

      expect(result.position).toBeDefined();
      expect(result.scalePercent).toBeGreaterThanOrEqual(85);
      expect(result.scalePercent).toBeLessThanOrEqual(95);

      if (result.optimalMarker) {
        result.optimalMarker.delete();
      }
    });

    it('should handle no match gracefully', () => {
      const patch = track(new cv.Mat(100, 100, cv.CV_8UC1));
      patch.setTo(new cv.Scalar(0));

      const marker = track(new cv.Mat(20, 20, cv.CV_8UC1));
      marker.setTo(new cv.Scalar(255));

      const result = multiScaleTemplateMatch(patch, marker, [90, 110], 5);

      // Should still return a result, but with low confidence
      expect(result.confidence).toBeLessThan(0.5);

      if (result.optimalMarker) {
        result.optimalMarker.delete();
      }
    });
  });

  describe('extractMarkerCorners', () => {
    it('should extract basic corners', () => {
      const marker = track(new cv.Mat(20, 30, cv.CV_8UC1));
      const position: [number, number] = [10, 15];

      const corners = extractMarkerCorners(position, marker, [0, 0]);

      expect(corners).toHaveLength(4);
      expect(corners[0]).toEqual([10, 15]); // top-left
      expect(corners[1]).toEqual([40, 15]); // top-right
      expect(corners[2]).toEqual([40, 35]); // bottom-right
      expect(corners[3]).toEqual([10, 35]); // bottom-left
    });

    it('should apply zone offset', () => {
      const marker = track(new cv.Mat(20, 20, cv.CV_8UC1));
      const position: [number, number] = [5, 10];
      const zoneOffset: [number, number] = [100, 200];

      const corners = extractMarkerCorners(position, marker, zoneOffset);

      // All corners should be shifted by offset
      expect(corners[0][0]).toBeGreaterThanOrEqual(100);
      expect(corners[0][1]).toBeGreaterThanOrEqual(200);
    });

    it('should handle different marker sizes', () => {
      const testCases: Array<[number, number]> = [
        [10, 10],
        [30, 20],
        [15, 40],
      ];

      testCases.forEach(([width, height]) => {
        const marker = track(new cv.Mat(height, width, cv.CV_8UC1));
        const position: [number, number] = [0, 0];

        const corners = extractMarkerCorners(position, marker, [0, 0]);

        // Check width and height
        expect(corners[1][0] - corners[0][0]).toBe(width);
        expect(corners[2][1] - corners[1][1]).toBe(height);
      });
    });
  });

  describe('detectMarkerInPatch', () => {
    const createTestScenario = (): [cv.Mat, cv.Mat] => {
      // Create marker
      const marker = track(new cv.Mat(25, 25, cv.CV_8UC1));
      marker.setTo(new cv.Scalar(0));
      const markerRect = new cv.Rect(5, 5, 15, 15);
      const markerRoi = marker.roi(markerRect);
      markerRoi.setTo(new cv.Scalar(255));
      markerRoi.delete();

      // Blur marker
      const markerBlurred = new cv.Mat();
      cv.GaussianBlur(marker, markerBlurred, new cv.Size(5, 5), 0);
      marker.delete();
      matsToCleanup = matsToCleanup.filter((m) => m !== marker);
      track(markerBlurred);

      // Create patch with marker embedded
      const patch = track(new cv.Mat(150, 150, cv.CV_8UC1));
      // Add some noise
      for (let i = 0; i < patch.rows; i++) {
        for (let j = 0; j < patch.cols; j++) {
          patch.ucharPtr(i, j)[0] = Math.floor(Math.random() * 50);
        }
      }

      // Embed marker at (60, 50)
      const patchRect = new cv.Rect(60, 50, markerBlurred.cols, markerBlurred.rows);
      const patchRoi = patch.roi(patchRect);
      markerBlurred.copyTo(patchRoi);
      patchRoi.delete();

      return [patch, markerBlurred];
    };

    it('should successfully detect marker', () => {
      const [patch, marker] = createTestScenario();

      const corners = detectMarkerInPatch(patch, marker, [0, 0], [90, 110], 5, 0.3);

      expect(corners).toBeDefined();
      expect(corners).toHaveLength(4);
      // Should be near position (60, 50)
      expect(corners![0][0]).toBeGreaterThanOrEqual(55);
      expect(corners![0][0]).toBeLessThanOrEqual(65);
      expect(corners![0][1]).toBeGreaterThanOrEqual(45);
      expect(corners![0][1]).toBeLessThanOrEqual(55);
    });

    it('should return null when confidence below threshold', () => {
      const patch = track(new cv.Mat(100, 100, cv.CV_8UC1));
      // Random noise
      for (let i = 0; i < patch.rows; i++) {
        for (let j = 0; j < patch.cols; j++) {
          patch.ucharPtr(i, j)[0] = Math.floor(Math.random() * 100);
        }
      }

      const marker = track(new cv.Mat(20, 20, cv.CV_8UC1));
      // Different random pattern
      for (let i = 0; i < marker.rows; i++) {
        for (let j = 0; j < marker.cols; j++) {
          marker.ucharPtr(i, j)[0] = Math.floor(Math.random() * 100) + 150;
        }
      }

      const corners = detectMarkerInPatch(patch, marker, [0, 0], [90, 110], 5, 0.95);

      // Should return null due to low confidence
      expect(corners).toBeNull();
    });

    it('should apply zone offset correctly', () => {
      const [patch, marker] = createTestScenario();
      const zoneOffset: [number, number] = [1000, 2000];

      const corners = detectMarkerInPatch(patch, marker, zoneOffset, [90, 110], 5, 0.3);

      expect(corners).toBeDefined();
      // Corners should be offset
      expect(corners![0][0]).toBeGreaterThanOrEqual(1000);
      expect(corners![0][1]).toBeGreaterThanOrEqual(2000);
    });
  });

  describe('validateMarkerDetection', () => {
    it('should validate valid corners', () => {
      const corners = [
        [10, 10],
        [30, 10],
        [30, 30],
        [10, 30],
      ];

      expect(validateMarkerDetection(corners)).toBe(true);
    });

    it('should reject null corners', () => {
      expect(validateMarkerDetection(null)).toBe(false);
    });

    it('should reject invalid shape', () => {
      const corners = [
        [10, 10],
        [20, 20],
      ]; // Only 2 points

      expect(validateMarkerDetection(corners)).toBe(false);
    });

    it('should validate area constraints', () => {
      // Square with area 400
      const corners = [
        [0, 0],
        [20, 0],
        [20, 20],
        [0, 20],
      ];

      // Should pass
      expect(validateMarkerDetection(corners, [300, 500])).toBe(true);

      // Should fail (too small)
      expect(validateMarkerDetection(corners, [500, 1000])).toBe(false);

      // Should fail (too large)
      expect(validateMarkerDetection(corners, [10, 100])).toBe(false);
    });

    it('should work without area validation', () => {
      const corners = [
        [0, 0],
        [20, 0],
        [20, 20],
        [0, 20],
      ];

      // Should pass (no area check)
      expect(validateMarkerDetection(corners, undefined)).toBe(true);
    });
  });
});

