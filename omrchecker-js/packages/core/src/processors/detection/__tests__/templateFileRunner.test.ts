/**
 * Tests for TemplateFileRunner.
 */

import * as cv from '@techstark/opencv-js';
import { TemplateFileRunner } from '../templateFileRunner';
import { TemplateLoader, type TemplateLayoutData } from '../../../template/TemplateLoader';
import type { TemplateConfig } from '../../../template/types';

describe('TemplateFileRunner', () => {
  let templateConfig: TemplateConfig;
  let templateLayout: TemplateLayout;
  let runner: TemplateFileRunner;
  let mockGrayImage: cv.Mat;

  beforeEach(() => {
    // Create a minimal template config
    templateConfig = {
      templateDimensions: [900, 650],
      bubbleDimensions: [20, 20],
      fieldBlocks: {
        block1: {
          name: 'block1',
          origin: [100, 100],
          fieldLabels: ['q1'],
          bubbleFieldType: 'QTYPE_MCQ4',
          bubblesGap: 30,
          labelsGap: 50,
        },
      },
    };

    templateLayout = TemplateLoader.loadLayoutFromJSON(templateConfig);
    const tuningConfig = {};
    runner = new TemplateFileRunner(templateLayout, tuningConfig);

    // Create a mock grayscale image
    mockGrayImage = new cv.Mat(900, 650, cv.CV_8UC1);
  });

  afterEach(() => {
    mockGrayImage.delete();
  });

  describe('constructor', () => {
    it('should initialize with template layout', () => {
      expect(runner.template).toBe(templateLayout);
      expect(runner.allFields.length).toBeGreaterThan(0);
      expect(runner.allFieldDetectionTypes.length).toBeGreaterThan(0);
    });

    it('should initialize field detection type runners', () => {
      expect(Object.keys(runner['fieldDetectionTypeFileRunners']).length).toBeGreaterThan(0);
    });
  });

  describe('readOmrAndUpdateMetrics', () => {
    it('should run detection and interpretation passes', () => {
      const omrResponse = runner.readOmrAndUpdateMetrics(
        'test.jpg',
        mockGrayImage,
        mockGrayImage
      );

      expect(omrResponse).toBeDefined();
      expect(typeof omrResponse).toBe('object');
    });
  });

  describe('runFileLevelDetection', () => {
    it('should run detection for all fields', () => {
      expect(() => {
        runner.runFileLevelDetection('test.jpg', mockGrayImage, mockGrayImage);
      }).not.toThrow();
    });
  });

  describe('runFileLevelInterpretation', () => {
    it('should run interpretation for all fields', () => {
      // Run detection first
      runner.runFileLevelDetection('test.jpg', mockGrayImage, mockGrayImage);

      const omrResponse = runner.runFileLevelInterpretation(
        'test.jpg',
        mockGrayImage,
        mockGrayImage
      );

      expect(omrResponse).toBeDefined();
      expect(typeof omrResponse).toBe('object');
    });
  });

  describe('getFieldDetectionTypeFileRunner', () => {
    it('should return file runner for valid detection type', () => {
      const fileRunner = runner['getFieldDetectionTypeFileRunner']('BUBBLES_THRESHOLD');
      expect(fileRunner).toBeDefined();
      expect(fileRunner.fieldDetectionType).toBe('BUBBLES_THRESHOLD');
    });

    it('should throw error for invalid detection type', () => {
      expect(() => {
        runner['getFieldDetectionTypeFileRunner']('INVALID_TYPE');
      }).toThrow();
    });
  });
});

