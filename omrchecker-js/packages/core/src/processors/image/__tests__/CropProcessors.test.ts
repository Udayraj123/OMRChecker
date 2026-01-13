/**
 * Tests for CropPage and CropOnMarkers preprocessors
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as cv from '@techstark/opencv-js';
import { CropPage } from '../CropPage';
import { CropOnMarkers } from '../CropOnMarkers';
import { WarpMethod } from '../../constants';

// Mock OpenCV for headless testing
vi.mock('@techstark/opencv-js', () => ({
  Mat: class MockMat {
    rows = 0;
    cols = 0;
    constructor() {}
    delete() {}
  },
}));

describe('CropPage', () => {
  let mockSaveImageOps: any;

  beforeEach(() => {
    mockSaveImageOps = {
      appendSaveImage: vi.fn(),
      tuningConfig: {
        outputs: {
          coloredOutputsEnabled: false,
          showPreprocessorsDiff: {},
        },
      },
    };
  });

  it('should instantiate with default options', () => {
    const processor = new CropPage(
      {},
      '/test',
      mockSaveImageOps,
      [800, 1000]
    );

    expect(processor.getName()).toBe('CropPage');
    expect(processor.getClassName()).toBe('CropPage');
  });

  it('should instantiate with custom options', () => {
    const processor = new CropPage(
      {
        morphKernel: [5, 5],
        useColoredCanny: true,
        tuningOptions: {
          warpMethod: WarpMethod.HOMOGRAPHY,
        },
      },
      '/test',
      mockSaveImageOps,
      [800, 1000]
    );

    expect(processor.getName()).toBe('CropPage');
  });

  it('should return unchanged images (placeholder implementation)', () => {
    const processor = new CropPage(
      {},
      '/test',
      mockSaveImageOps,
      [800, 1000]
    );

    const mockImage = new cv.Mat() as any;
    const mockColoredImage = new cv.Mat() as any;
    const mockTemplate = { someProperty: 'value' };

    const [resultImage, resultColoredImage, resultTemplate] = processor.applyFilter(
      mockImage,
      mockColoredImage,
      mockTemplate,
      '/test/image.jpg'
    );

    expect(resultImage).toBe(mockImage);
    expect(resultColoredImage).toBe(mockColoredImage);
    expect(resultTemplate).toBe(mockTemplate);
  });
});

describe('CropOnMarkers', () => {
  let mockSaveImageOps: any;

  beforeEach(() => {
    mockSaveImageOps = {
      appendSaveImage: vi.fn(),
      tuningConfig: {
        outputs: {
          coloredOutputsEnabled: false,
          showPreprocessorsDiff: {},
        },
      },
    };
  });

  it('should instantiate with FOUR_MARKERS type', () => {
    const processor = new CropOnMarkers(
      {
        type: 'FOUR_MARKERS',
        referenceImage: '/markers/marker.png',
        markerDimensions: [50, 50],
      },
      '/test',
      mockSaveImageOps,
      [800, 1000]
    );

    expect(processor.getName()).toBe('CropOnMarkers');
    expect(processor.getClassName()).toBe('CropOnMarkers');
  });

  it('should instantiate with FOUR_DOTS type', () => {
    const processor = new CropOnMarkers(
      {
        type: 'FOUR_DOTS',
        defaultSelector: 'CENTERS',
      },
      '/test',
      mockSaveImageOps,
      [800, 1000]
    );

    expect(processor.getName()).toBe('CropOnMarkers');
  });

  it('should instantiate with TWO_LINES type', () => {
    const processor = new CropOnMarkers(
      {
        type: 'TWO_LINES',
        tuningOptions: {
          warpMethod: WarpMethod.PERSPECTIVE_TRANSFORM,
        },
      },
      '/test',
      mockSaveImageOps,
      [800, 1000]
    );

    expect(processor.getName()).toBe('CropOnMarkers');
  });

  it('should return unchanged images (placeholder implementation)', () => {
    const processor = new CropOnMarkers(
      {
        type: 'FOUR_MARKERS',
        referenceImage: '/markers/marker.png',
      },
      '/test',
      mockSaveImageOps,
      [800, 1000]
    );

    const mockImage = new cv.Mat() as any;
    const mockColoredImage = new cv.Mat() as any;
    const mockTemplate = { someProperty: 'value' };

    const [resultImage, resultColoredImage, resultTemplate] = processor.applyFilter(
      mockImage,
      mockColoredImage,
      mockTemplate,
      '/test/image.jpg'
    );

    expect(resultImage).toBe(mockImage);
    expect(resultColoredImage).toBe(mockColoredImage);
    expect(resultTemplate).toBe(mockTemplate);
  });

  it('should return empty excludeFiles list', () => {
    const processor = new CropOnMarkers(
      {
        type: 'FOUR_MARKERS',
        referenceImage: '/markers/marker.png',
      },
      '/test',
      mockSaveImageOps,
      [800, 1000]
    );

    expect(processor.excludeFiles()).toEqual([]);
  });
});

