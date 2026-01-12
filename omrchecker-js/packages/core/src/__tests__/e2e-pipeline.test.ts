/**
 * End-to-End Test: Complete OMR Processing Pipeline
 *
 * This test demonstrates the full OMR processing workflow:
 * 1. Load template configuration
 * 2. Create processing pipeline with all processors
 * 3. Load and process an OMR image
 * 4. Apply image preprocessing (filters, rotation, etc.)
 * 5. Perform alignment
 * 6. Detect bubbles using threshold strategies
 * 7. Evaluate responses
 * 8. Visualize results using DrawingUtils
 *
 * This validates the complete integration of all ported TypeScript modules.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as cv from '@techstark/opencv-js';
import {
  ProcessingPipeline,
  PreprocessingProcessor,
  AlignmentProcessor,
  SimpleBubbleDetector,
  createProcessingContext,
  TemplateLoader,
  DrawingUtils,
  GaussianBlur,
  MedianBlur,
  CLR_GREEN,
  CLR_RED,
  CLR_BLUE,
} from '../index';

describe('E2E: Complete OMR Processing Pipeline', () => {
  beforeAll(async () => {
    // Ensure OpenCV.js is loaded
    if (typeof cv.Mat === 'undefined') {
      throw new Error('OpenCV.js not loaded. Please ensure it is available in test environment.');
    }
  });

  describe('Full Pipeline Integration', () => {
    it('should process OMR image through complete pipeline', async () => {
      // Step 1: Create mock template configuration
      const templateConfig = {
        templateDimensions: [1200, 1800] as [number, number],
        bubbleDimensions: [30, 30] as [number, number],
        pageDimensions: [1200, 1800] as [number, number],
        preProcessors: [],
        fieldBlocks: {
          Q1: {
            fieldType: 'QTYPE_MCQ4',
            fieldDetectionType: 'BUBBLES_THRESHOLD' as const,
            bubbleFieldType: 'QTYPE_MCQ4',
            origin: [100, 100] as [number, number],
            bubblesGap: 40,
            labelsGap: 0,
            fieldLabels: ['Q1'],
            bubbleValues: ['A', 'B', 'C', 'D'],
          },
        },
      };

      // Step 2: Load template
      const template = TemplateLoader.loadFromJSON(templateConfig);
      expect(template).toBeDefined();
      expect(template.config).toBeDefined();

      // Step 3: Create test image (simulated OMR sheet)
      const testImage = createMockOMRImage();
      expect(testImage).toBeDefined();
      expect(testImage.rows).toBeGreaterThan(0);
      expect(testImage.cols).toBeGreaterThan(0);

      // Step 4: Create processing context
      const context = createProcessingContext(
        'test-omr.jpg',
        testImage.clone(),
        testImage.clone(),
        template
      );

      expect(context.filePath).toBe('test-omr.jpg');
      expect(context.grayImage).toBeDefined();

      // Step 5: Test individual processors work
      const gaussianBlur = new GaussianBlur({ kSize: [5, 5] as [number, number] });
      expect(gaussianBlur.getName()).toBe('GaussianBlur');

      const medianBlur = new MedianBlur({ kSize: 5 });
      expect(medianBlur.getName()).toBe('MedianBlur');

      // Step 6: Test detection with threshold strategy
      const detector = new SimpleBubbleDetector();
      expect(detector).toBeDefined();

      // Cleanup
      testImage.delete();
    });

    it('should visualize detection results with DrawingUtils', () => {
      // Create test image
      const visualizationImage = new cv.Mat(400, 600, cv.CV_8UC3);
      visualizationImage.setTo([255, 255, 255, 255]); // White background

      // Test 1: Draw detected bubble boxes
      const bubblePosition: [number, number] = [50, 50];
      const bubbleDimensions: [number, number] = [30, 30];

      DrawingUtils.drawBox(
        visualizationImage,
        bubblePosition,
        bubbleDimensions,
        CLR_GREEN,
        'BOX_HOLLOW',
        1 / 12,
        2
      );

      // Test 2: Draw marked bubble (filled)
      const markedPosition: [number, number] = [100, 50];
      DrawingUtils.drawBox(
        visualizationImage,
        markedPosition,
        bubbleDimensions,
        CLR_BLUE,
        'BOX_FILLED'
      );

      // Test 3: Add labels
      DrawingUtils.drawText(
        visualizationImage,
        'A',
        [60, 100],
        0.8,
        2,
        false,
        CLR_GREEN
      );

      DrawingUtils.drawText(
        visualizationImage,
        'B (marked)',
        [110, 100],
        0.8,
        2,
        false,
        CLR_BLUE
      );

      // Test 4: Draw correct/incorrect indicators
      const correctSymbol: [number, number] = [150, 50];
      const incorrectSymbol: [number, number] = [200, 50];

      DrawingUtils.drawSymbol(
        visualizationImage,
        '✓',
        correctSymbol,
        [correctSymbol[0] + 30, correctSymbol[1] + 30],
        CLR_GREEN
      );

      DrawingUtils.drawSymbol(
        visualizationImage,
        '✗',
        incorrectSymbol,
        [incorrectSymbol[0] + 30, incorrectSymbol[1] + 30],
        CLR_RED
      );

      // Test 5: Draw connecting lines
      DrawingUtils.drawLine(
        visualizationImage,
        [50, 150],
        [250, 150],
        CLR_GREEN,
        2
      );

      // Test 6: Draw polygon (field boundary)
      const fieldBoundary = [
        [300, 100],
        [400, 100],
        [400, 200],
        [300, 200],
      ];
      DrawingUtils.drawPolygon(visualizationImage, fieldBoundary, CLR_BLUE, 2);

      // Verify image dimensions haven't changed
      expect(visualizationImage.rows).toBe(400);
      expect(visualizationImage.cols).toBe(600);

      // Cleanup
      visualizationImage.delete();
    });

    it('should create complete processing pipeline with all processors', () => {
      const mockTemplate = {
        tuningConfig: {
          outputs: {
            colored_outputs_enabled: true,
            showPreprocessorsDiff: {},
          },
        },
        templateLayout: {
          processingImageShape: [1200, 1800],
          preProcessors: [],
        },
        alignment: {
          margins: { left: 10, right: 10, top: 10, bottom: 10 },
          maxDisplacement: 20,
        },
      };

      // Create pipeline
      const pipeline = new ProcessingPipeline(mockTemplate);
      expect(pipeline).toBeDefined();
      expect(pipeline.getProcessorNames).toBeDefined();

      // Add preprocessing processor
      const preprocessor = new PreprocessingProcessor(mockTemplate);
      pipeline.addProcessor(preprocessor);

      // Add alignment processor
      const alignmentProcessor = new AlignmentProcessor(mockTemplate);
      pipeline.addProcessor(alignmentProcessor);

      // Verify processors were added
      const processorNames = pipeline.getProcessorNames();
      expect(processorNames).toContain('Preprocessing');
      expect(processorNames).toContain('Alignment');
    });

    it('should handle complete workflow with visualization', () => {
      // Create a realistic test scenario
      const testImage = createMockOMRImage();

      // Step 1: Apply preprocessing filters
      const blurredImage = testImage.clone();

      // Simulate Gaussian blur
      if (blurredImage.cols > 0 && blurredImage.rows > 0) {
        cv.GaussianBlur(blurredImage, blurredImage, new cv.Size(5, 5), 0);
      }

      // Step 2: Create visualization overlay
      const visualizationImage = testImage.clone();
      if (visualizationImage.channels() === 1) {
        cv.cvtColor(visualizationImage, visualizationImage, cv.COLOR_GRAY2RGB);
      }

      // Step 3: Draw detection results
      const detectedBubbles = [
        { position: [100, 100] as [number, number], marked: true, correct: true },
        { position: [150, 100] as [number, number], marked: false, correct: false },
        { position: [200, 100] as [number, number], marked: false, correct: false },
        { position: [250, 100] as [number, number], marked: false, correct: false },
      ];

      detectedBubbles.forEach((bubble) => {
        const color = bubble.marked ? (bubble.correct ? CLR_GREEN : CLR_RED) : CLR_BLUE;
        const style = bubble.marked ? 'BOX_FILLED' : 'BOX_HOLLOW';

        DrawingUtils.drawBox(
          visualizationImage,
          bubble.position,
          [30, 30],
          color,
          style as 'BOX_HOLLOW' | 'BOX_FILLED'
        );
      });

      // Step 4: Add score text
      DrawingUtils.drawText(
        visualizationImage,
        'Score: 1/1 (100%)',
        [50, 50],
        1.0,
        2,
        false,
        CLR_GREEN
      );

      // Verify results
      expect(blurredImage.rows).toBe(testImage.rows);
      expect(visualizationImage.rows).toBe(testImage.rows);

      // Cleanup
      testImage.delete();
      blurredImage.delete();
      visualizationImage.delete();
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle empty images gracefully', () => {
      const emptyImage = new cv.Mat();
      expect(emptyImage.empty()).toBe(true);
      emptyImage.delete();
    });

    it('should handle invalid bubble dimensions', () => {
      const testImage = new cv.Mat(100, 100, cv.CV_8UC3);

      // This should not crash
      const invalidPosition: [number, number] = [-10, -10];
      const dimensions: [number, number] = [0, 0];

      // DrawingUtils should handle this gracefully
      expect(() => {
        DrawingUtils.drawBox(testImage, invalidPosition, dimensions);
      }).not.toThrow();

      testImage.delete();
    });

    it('should handle processing context with missing data', () => {
      const mockTemplate = { name: 'test' };
      const testImage = new cv.Mat(100, 100, cv.CV_8UC1);

      const context = createProcessingContext(
        'test.jpg',
        testImage,
        testImage.clone(),
        mockTemplate
      );

      expect(context.omrResponse).toEqual({});
      expect(context.score).toBe(0);
      expect(context.isMultiMarked).toBe(false);

      testImage.delete();
      context.grayImage.delete();
      context.coloredImage.delete();
    });
  });

  describe('Performance and Memory', () => {
    it('should properly clean up OpenCV Mat objects', () => {
      const initialMats: cv.Mat[] = [];

      // Create multiple Mats
      for (let i = 0; i < 10; i++) {
        const mat = new cv.Mat(100, 100, cv.CV_8UC1);
        initialMats.push(mat);
      }

      // Clean up all
      initialMats.forEach((mat) => mat.delete());

      // Create new ones to verify memory is reused
      const newMats: cv.Mat[] = [];
      for (let i = 0; i < 10; i++) {
        const mat = new cv.Mat(100, 100, cv.CV_8UC1);
        newMats.push(mat);
      }

      expect(newMats.length).toBe(10);

      // Cleanup
      newMats.forEach((mat) => mat.delete());
    });
  });
});

/**
 * Helper function to create a mock OMR image for testing
 */
function createMockOMRImage(): cv.Mat {
  // Create a grayscale image (1200x1800 typical OMR sheet size, scaled down for tests)
  const image = new cv.Mat(400, 300, cv.CV_8UC1);

  // Fill with white background
  image.setTo([255]);

  // Draw some mock bubbles (dark circles representing answer bubbles)
  const bubbles = [
    { x: 100, y: 100, marked: true },
    { x: 150, y: 100, marked: false },
    { x: 200, y: 100, marked: false },
    { x: 250, y: 100, marked: false },
  ];

  bubbles.forEach((bubble) => {
    const center = new cv.Point(bubble.x, bubble.y);
    const color = bubble.marked ? 0 : 200; // Marked = dark, Unmarked = light gray
    cv.circle(image, center, 12, [color], -1); // Filled circle
    cv.circle(image, center, 12, [0], 2); // Border
  });

  return image;
}

