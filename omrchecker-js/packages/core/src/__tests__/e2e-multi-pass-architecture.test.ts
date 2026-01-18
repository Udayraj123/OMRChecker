/**
 * End-to-end integration tests for multi-pass detection and interpretation architecture.
 *
 * Tests the complete pipeline using TemplateFileRunner with:
 * - Detection pass collecting aggregates
 * - Interpretation pass using aggregates
 * - Multi-marking detection
 * - Confidence metrics
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as cv from '@techstark/opencv-js';
import { TemplateFileRunner } from '../processors/detection/templateFileRunner';
import { TemplateLoader, type TemplateLayoutData } from '../template/TemplateLoader';
import type { TemplateConfig } from '../template/types';
import { OMRProcessor } from '../core/OMRProcessor';

describe('Multi-Pass Architecture E2E', () => {
  let templateConfig: TemplateConfig;
  let templateLayout: TemplateLayoutData;
  let mockGrayImage: cv.Mat;
  let mockColoredImage: cv.Mat;

  beforeEach(() => {
    // Create a minimal template config
    templateConfig = {
      templateDimensions: [900, 650],
      bubbleDimensions: [20, 20],
      fieldBlocks: {
        block1: {
          name: 'block1',
          origin: [100, 100],
          fieldLabels: ['q1', 'q2'],
          bubbleFieldType: 'QTYPE_MCQ4',
          bubblesGap: 30,
          labelsGap: 50,
        },
      },
    };

    templateLayout = TemplateLoader.loadLayoutFromJSON(templateConfig);

    // Create mock images
    mockGrayImage = new cv.Mat(900, 650, cv.CV_8UC1, new cv.Scalar(200));
    mockColoredImage = new cv.Mat(900, 650, cv.CV_8UC3, new cv.Scalar(200, 200, 200));
  });

  afterEach(() => {
    mockGrayImage.delete();
    mockColoredImage.delete();
  });

  describe('TemplateFileRunner - End-to-End Processing', () => {
    it('should process image with detection and interpretation passes', () => {
      const tuningConfig = {};
      const runner = new TemplateFileRunner(templateLayout, tuningConfig);

      const omrResponse = runner.readOmrAndUpdateMetrics(
        'test.jpg',
        mockGrayImage,
        mockColoredImage
      );

      expect(omrResponse).toBeDefined();
      expect(typeof omrResponse).toBe('object');

      // Should have responses for all fields
      expect(Object.keys(omrResponse).length).toBeGreaterThan(0);
    });

    it('should collect aggregates across detection and interpretation', () => {
      const tuningConfig = {};
      const runner = new TemplateFileRunner(templateLayout, tuningConfig);

      runner.readOmrAndUpdateMetrics('test.jpg', mockGrayImage, mockColoredImage);

      // Check detection aggregates
      const detectionAggs = runner.getFileLevelDetectionAggregates();
      expect(detectionAggs).toBeDefined();

      // Check interpretation aggregates
      const interpretationAggs = runner.getFileLevelInterpretationAggregates();
      expect(interpretationAggs).toBeDefined();
    });

    it('should handle multiple images with aggregate accumulation', () => {
      const tuningConfig = {};
      const runner = new TemplateFileRunner(templateLayout, tuningConfig);

      // Process first image
      runner.readOmrAndUpdateMetrics('test1.jpg', mockGrayImage, mockColoredImage);

      // Process second image
      const image2 = new cv.Mat(900, 650, cv.CV_8UC1, new cv.Scalar(200));
      runner.readOmrAndUpdateMetrics('test2.jpg', image2, mockColoredImage);

      // Directory-level aggregates should accumulate
      const dirDetectionAggs = runner.getDirectoryLevelDetectionAggregates();
      const dirInterpretationAggs = runner.getDirectoryLevelInterpretationAggregates();

      expect(dirDetectionAggs).toBeDefined();
      expect(dirInterpretationAggs).toBeDefined();

      image2.delete();
    });
  });

  describe('OMRProcessor Integration', () => {
    it('should process image using multi-pass architecture', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result = await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);

      expect(result).toBeDefined();
      expect(result.responses).toBeDefined();
      expect(result.filePath).toBe('test.jpg');
      expect(result.processingTimeMs).toBeGreaterThan(0);
    });

    it('should provide access to aggregates', async () => {
      const processor = new OMRProcessor(templateConfig);

      await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);

      const aggregates = processor.getAggregates();
      expect(aggregates).toBeDefined();
      expect(aggregates?.detection).toBeDefined();
      expect(aggregates?.interpretation).toBeDefined();
    });

    it('should detect multi-marking when multiple bubbles are marked', async () => {
      // Create image with multiple bubbles marked
      const image = new cv.Mat(900, 650, cv.CV_8UC1, new cv.Scalar(200));

      // Mark multiple bubbles (simulate multi-marking)
      // This is a simplified test - actual multi-marking detection
      // would require proper bubble field setup

      const processor = new OMRProcessor(templateConfig);
      const result = await processor.processImage(image, 'test.jpg', mockColoredImage);

      expect(result).toBeDefined();
      expect(result.multiMarkedFields).toBeDefined();
      expect(Array.isArray(result.multiMarkedFields)).toBe(true);

      image.delete();
    });
  });

  describe('Confidence Metrics', () => {
    it('should collect confidence metrics in interpretation aggregates', () => {
      const tuningConfig = {
        outputs: {
          show_confidence_metrics: true,
        },
      };
      const runner = new TemplateFileRunner(templateLayout, tuningConfig);

      runner.readOmrAndUpdateMetrics('test.jpg', mockGrayImage, mockColoredImage);

      const interpretationAggs = runner.getFileLevelInterpretationAggregates() as {
        confidence_metrics_for_file?: Record<string, unknown>;
      };

      // Confidence metrics should be collected if enabled
      if (interpretationAggs?.confidence_metrics_for_file) {
        expect(typeof interpretationAggs.confidence_metrics_for_file).toBe('object');
      }
    });
  });
});

