/**
 * Unit tests for ProcessingContext and ProcessingPipeline.
 *
 * Translated from Python: src/tests/processors/__tests__/test_pipeline.py
 *
 * Tests translated (3 of 8 Python tests):
 *   - test_context_initialization
 *   - test_pipeline_processor_management
 *   - test_full_pipeline_execution
 *
 * Tests NOT translated (Python-specific classes not yet ported to TypeScript):
 *   - test_context_path_conversion    (no Path type in TS — not needed)
 *   - test_readomr_processor_flow     (ReadOMRProcessor not in TS)
 *   - test_alignment_with_reference_image (AlignmentProcessor not in TS)
 *   - test_pipeline_with_alignment_enabled (alignment auto-init not in TS)
 *   - test_pipeline_without_alignment_data (alignment auto-init not in TS)
 *
 * Python Pipeline.__init__ auto-adds PreprocessingCoordinator + ReadOMRProcessor
 * based on template config.  The TypeScript Pipeline starts with an empty
 * processors list and processors are added manually — constructor behaviour
 * differs by design for the browser port.
 */

import { vi, describe, it, expect } from 'vitest';

// Mock OpenCV.js before any imports.
// base.ts and Pipeline.ts import cv only for TypeScript type annotations (cv.Mat);
// the runtime code never calls cv directly, so an empty mock is sufficient.
vi.mock('@techstark/opencv-js', () => ({ default: {} }));

import {
  createProcessingContext,
  Processor,
  ProcessingContext,
} from '../../src/processors/base';
import { ProcessingPipeline } from '../../src/processors/Pipeline';

// ---------------------------------------------------------------------------
// Fixtures — mirrors Python's mock_template / mock_images pytest fixtures
// ---------------------------------------------------------------------------

function makeMockTemplate() {
  return {
    tuningConfig: {
      outputs: { coloredOutputsEnabled: true },
      alignment: { enabled: false },
    },
    alignment: {},
    templateDimensions: [1000, 800],
  };
}

/** Simulate np.zeros images as plain objects (cv.Mat is mocked away). */
function makeMockImages() {
  const grayImage = { rows: 1000, cols: 800 } as any;
  const coloredImage = { rows: 1000, cols: 800 } as any;
  return { grayImage, coloredImage };
}

// ---------------------------------------------------------------------------
// TestProcessingContext
// ---------------------------------------------------------------------------

describe('TestProcessingContext', () => {
  // test_context_initialization: context initialises correctly
  it('test_context_initialization', () => {
    const mockTemplate = makeMockTemplate();
    const { grayImage, coloredImage } = makeMockImages();

    const context = createProcessingContext(
      'test.jpg',
      grayImage,
      coloredImage,
      mockTemplate,
    );

    expect(context.filePath).toBe('test.jpg');
    expect(context.grayImage).toBe(grayImage);
    expect(context.coloredImage).toBe(coloredImage);
    expect(context.template).toBe(mockTemplate);
    expect(context.omrResponse).toEqual({});
    expect(context.isMultiMarked).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// TestProcessingPipeline
// ---------------------------------------------------------------------------

describe('TestProcessingPipeline', () => {
  // test_pipeline_processor_management: add / remove / list processors
  it('test_pipeline_processor_management', () => {
    const mockTemplate = makeMockTemplate();
    const pipeline = new ProcessingPipeline(mockTemplate);

    // TypeScript pipeline starts with an empty processor list.
    // (Python auto-adds "Preprocessing" + "ReadOMR" in __init__.)
    const initialNames = pipeline.getProcessorNames();
    expect(Array.isArray(initialNames)).toBe(true);

    // Add a custom processor
    class CustomProcessor extends Processor {
      getName() { return 'CustomProcessor'; }
      process(context: ProcessingContext) { return context; }
    }
    pipeline.addProcessor(new CustomProcessor());

    expect(pipeline.getProcessorNames()).toContain('CustomProcessor');

    // Remove the processor
    pipeline.removeProcessor('CustomProcessor');
    expect(pipeline.getProcessorNames()).not.toContain('CustomProcessor');
  });

  // test_full_pipeline_execution: processFile returns context with correct fields
  it('test_full_pipeline_execution', async () => {
    const mockTemplate = makeMockTemplate();
    const { grayImage, coloredImage } = makeMockImages();
    const pipeline = new ProcessingPipeline(mockTemplate);

    // Add a mock processor that simulates ReadOMR populating omrResponse
    class MockReadOMR extends Processor {
      getName() { return 'ReadOMR'; }
      process(context: ProcessingContext) {
        context.omrResponse = { Q1: 'A', Q2: 'B' };
        return context;
      }
    }
    pipeline.addProcessor(new MockReadOMR());

    const result = await pipeline.processFile('test.jpg', grayImage, coloredImage);

    expect(result.filePath).toBe('test.jpg');
    expect(result.omrResponse).toEqual({ Q1: 'A', Q2: 'B' });
    expect(result.isMultiMarked).toBe(false);
  });
});
