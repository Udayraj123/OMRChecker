import { describe, it, expect, beforeEach, vi } from 'vitest';
import cv from '@techstark/opencv-js';
import { CropPage } from '../CropPage';
import { WarpMethod } from '../../constants';

describe('CropPage', () => {
  beforeEach(() => {
    // Ensure OpenCV is loaded
    expect(cv).toBeDefined();
  });

  describe('constructor', () => {
    it('should create with default options', () => {
      const saveImageOps = {
        tuningConfig: {
          outputs: { coloredOutputsEnabled: false },
        },
      };

      const processor = new CropPage({}, '/test', saveImageOps, [300, 400]);

      expect(processor.getClassName()).toBe('CropPage');
    });

    it('should apply default morphKernel', () => {
      const saveImageOps = {
        tuningConfig: {
          outputs: { coloredOutputsEnabled: false },
        },
      };

      const processor = new CropPage({}, '/test', saveImageOps, [300, 400]);

      // Check internal options
      expect((processor as any).options.morphKernel).toEqual([10, 10]);
    });

    it('should apply custom morphKernel', () => {
      const saveImageOps = {
        tuningConfig: {
          outputs: { coloredOutputsEnabled: false },
        },
      };

      const processor = new CropPage(
        { morphKernel: [5, 5] },
        '/test',
        saveImageOps,
        [300, 400]
      );

      expect((processor as any).options.morphKernel).toEqual([5, 5]);
    });

    it('should set enableCropping to true', () => {
      const saveImageOps = {
        tuningConfig: {
          outputs: { coloredOutputsEnabled: false },
        },
      };

      const processor = new CropPage({}, '/test', saveImageOps, [300, 400]);

      expect((processor as any).options.enableCropping).toBe(true);
    });

    it('should default warpMethod to PERSPECTIVE_TRANSFORM', () => {
      const saveImageOps = {
        tuningConfig: {
          outputs: { coloredOutputsEnabled: false },
        },
      };

      const processor = new CropPage({}, '/test', saveImageOps, [300, 400]);

      expect((processor as any).options.tuningOptions.warpMethod).toBe(
        WarpMethod.PERSPECTIVE_TRANSFORM
      );
    });

    it('should default useColoredCanny to false', () => {
      const saveImageOps = {
        tuningConfig: {
          outputs: { coloredOutputsEnabled: false },
        },
      };

      const processor = new CropPage({}, '/test', saveImageOps, [300, 400]);

      expect((processor as any).useColoredCanny).toBe(false);
    });

    it('should set useColoredCanny when provided', () => {
      const saveImageOps = {
        tuningConfig: {
          outputs: { coloredOutputsEnabled: true },
        },
      };

      const processor = new CropPage(
        { useColoredCanny: true },
        '/test',
        saveImageOps,
        [300, 400]
      );

      expect((processor as any).useColoredCanny).toBe(true);
    });

    it('should default maxPointsPerEdge to null', () => {
      const saveImageOps = {
        tuningConfig: {
          outputs: { coloredOutputsEnabled: false },
        },
      };

      const processor = new CropPage({}, '/test', saveImageOps, [300, 400]);

      expect((processor as any).options.maxPointsPerEdge).toBeNull();
    });

    it('should set maxPointsPerEdge when provided', () => {
      const saveImageOps = {
        tuningConfig: {
          outputs: { coloredOutputsEnabled: false },
        },
      };

      const processor = new CropPage(
        { maxPointsPerEdge: 20 },
        '/test',
        saveImageOps,
        [300, 400]
      );

      expect((processor as any).options.maxPointsPerEdge).toBe(20);
    });
  });

  describe('validation', () => {
    it('should validate and merge options correctly', () => {
      const saveImageOps = {
        tuningConfig: {
          outputs: { coloredOutputsEnabled: false },
        },
      };

      const processor = new CropPage(
        {
          morphKernel: [8, 8],
          useColoredCanny: true,
          tuningOptions: {
            customOption: 'test',
          },
        },
        '/test',
        saveImageOps,
        [300, 400]
      );

      const options = (processor as any).options;
      expect(options.morphKernel).toEqual([8, 8]);
      expect(options.useColoredCanny).toBe(true);
      expect(options.tuningOptions.customOption).toBe('test');
    });

    it('should preserve tuning options during merge', () => {
      const saveImageOps = {
        tuningConfig: {
          outputs: { coloredOutputsEnabled: false },
        },
      };

      const processor = new CropPage(
        {
          tuningOptions: {
            warpMethod: WarpMethod.HOMOGRAPHY,
            customFlag: true,
          },
        },
        '/test',
        saveImageOps,
        [300, 400]
      );

      const tuningOptions = (processor as any).options.tuningOptions;
      expect(tuningOptions.warpMethod).toBe(WarpMethod.HOMOGRAPHY);
      expect(tuningOptions.customFlag).toBe(true);
      expect(tuningOptions.normalizeConfig).toEqual([]);
      expect(tuningOptions.cannyConfig).toEqual([]);
    });
  });

  describe('inheritance', () => {
    it('should extend WarpOnPointsCommon', () => {
      const saveImageOps = {
        tuningConfig: {
          outputs: { coloredOutputsEnabled: false },
        },
      };

      const processor = new CropPage({}, '/test', saveImageOps, [300, 400]);

      // Check that it has parent's properties
      expect((processor as any).warpMethod).toBeDefined();
      expect((processor as any).warpMethodFlag).toBeDefined();
    });
  });

  describe('cleanup', () => {
    it('should cleanup morphKernel if it exists', () => {
      const saveImageOps = {
        tuningConfig: {
          outputs: { coloredOutputsEnabled: false },
        },
      };

      const processor = new CropPage(
        { morphKernel: [5, 5] },
        '/test',
        saveImageOps,
        [300, 400]
      );

      // Mock the delete method
      const morphKernel = (processor as any).morphKernel;
      if (morphKernel) {
        const deleteSpy = vi.fn();
        morphKernel.delete = deleteSpy;

        processor.cleanup();

        expect(deleteSpy).toHaveBeenCalled();
      }
    });

    it('should handle cleanup when morphKernel is undefined', () => {
      const saveImageOps = {
        tuningConfig: {
          outputs: { coloredOutputsEnabled: false },
        },
      };

      const processor = new CropPage({}, '/test', saveImageOps, [300, 400]);
      (processor as any).morphKernel = undefined;

      // Should not throw
      expect(() => processor.cleanup()).not.toThrow();
    });
  });
});

