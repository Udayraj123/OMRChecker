/**
 * Migrated from Python: src/processors/base.py
 * Agent: Processor-Delta
 * Phase: 1
 *
 * Base classes for the unified processor architecture.
 * All processing steps inherit from Processor and operate on ProcessingContext.
 */

import cv from '@techstark/opencv-js';

/**
 * Context object that flows through all processors.
 * 
 * This encapsulates all the data that gets passed between processors,
 * making it easy to add new data without changing method signatures.
 */
export interface ProcessingContext {
  // Input data
  filePath: string;
  grayImage: cv.Mat;
  coloredImage: cv.Mat;
  template: any; // Template type (avoiding circular import)

  // Processing results (populated by processors)
  omrResponse: Record<string, string>;
  isMultiMarked: boolean;
  fieldIdToInterpretation: Record<string, any>;

  // Evaluation results (populated by EvaluationProcessor)
  score: number;
  evaluationMeta: Record<string, any> | null;
  evaluationConfigForResponse: any;
  defaultAnswersSummary: string;

  // Additional metadata
  metadata: Record<string, any>;
}

/**
 * Factory function to create a new ProcessingContext with default values
 * 
 * @param filePath - Path to the file being processed
 * @param grayImage - Grayscale version of the image
 * @param coloredImage - Color version of the image
 * @param template - Template configuration
 * @returns New ProcessingContext with defaults
 */
export function createProcessingContext(
  filePath: string,
  grayImage: cv.Mat,
  coloredImage: cv.Mat,
  template: any
): ProcessingContext {
  return {
    filePath,
    grayImage,
    coloredImage,
    template,
    omrResponse: {},
    isMultiMarked: false,
    fieldIdToInterpretation: {},
    score: 0.0,
    evaluationMeta: null,
    evaluationConfigForResponse: null,
    defaultAnswersSummary: '',
    metadata: {},
  };
}

/**
 * Abstract base class for all processors.
 * 
 * All processing steps (image preprocessing, alignment, detection, etc.)
 * inherit from this class and implement the same interface for consistency.
 */
export abstract class Processor {
  /**
   * Process the context and return updated context.
   * 
   * @param context - The processing context containing all current state
   * @returns Updated processing context with results from this processor
   */
  abstract process(context: ProcessingContext): ProcessingContext | Promise<ProcessingContext>;

  /**
   * Get a human-readable name for this processor.
   * 
   * @returns String name of the processor (e.g., "AutoRotate", "ReadOMR")
   */
  abstract getName(): string;
}
