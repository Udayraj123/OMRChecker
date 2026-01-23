/**
 * Base classes for the unified processor architecture.
 *
 * TypeScript port of src/processors/base.py
 * Maintains 1:1 correspondence with Python implementation.
 */

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
  template: any; // Template type (avoiding circular dependencies)

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
 * Create a new ProcessingContext with default values.
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

