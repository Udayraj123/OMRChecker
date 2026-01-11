/**
 * Shared type definitions for OMRChecker core library.
 *
 * TypeScript port of src/core/types.py
 * These types provide a clean interface between CLI and core processing logic.
 */

/**
 * Configuration for OMRProcessor.
 *
 * This mirrors the args dict but with a cleaner, typed interface.
 */
export interface ProcessorConfig {
  // Input/Output paths
  inputDir?: string;
  outputDir: string;

  // Processing modes
  debug: boolean;
  outputMode: string;
  setLayout: boolean;

  // ML-related settings (future support)
  mlModelPath?: string;
  fieldBlockModelPath?: string;
  useFieldBlockDetection: boolean;
  enableShiftDetection: boolean;
  fusionStrategy: string;

  // Training-related settings
  collectTrainingData: boolean;
  confidenceThreshold: number;
  trainingDataDir: string;

  // Execution mode
  mode: string; // 'process', 'auto-train', 'test-model', 'export-yolo'
  epochs: number;
}

/**
 * Create a ProcessorConfig with default values.
 */
export function createDefaultProcessorConfig(): ProcessorConfig {
  return {
    outputDir: 'outputs',
    debug: false,
    outputMode: 'default',
    setLayout: false,
    useFieldBlockDetection: false,
    enableShiftDetection: false,
    fusionStrategy: 'confidence_weighted',
    collectTrainingData: false,
    confidenceThreshold: 0.85,
    trainingDataDir: 'outputs/training_data',
    mode: 'process',
    epochs: 100,
  };
}

/**
 * Result of processing a single OMR sheet.
 *
 * This provides a clean, typed return value instead of relying on side effects.
 */
export interface OMRResult {
  // File information
  fileName: string;
  filePath: string;
  outputPath?: string;

  // Processing status
  status: 'success' | 'error' | 'multi_marked';
  error?: string;

  // OMR response
  omrResponse: Record<string, string>;
  rawOmrResponse: Record<string, any>;

  // Evaluation results
  score: number;
  evaluationMeta?: Record<string, any>;

  // Additional metadata
  isMultiMarked: boolean;
  processingTime: number;
  fieldInterpretations: Record<string, any>;
}

/**
 * Create a default OMRResult.
 */
export function createOMRResult(fileName: string, filePath: string): OMRResult {
  return {
    fileName,
    filePath,
    status: 'success',
    omrResponse: {},
    rawOmrResponse: {},
    score: 0.0,
    isMultiMarked: false,
    processingTime: 0.0,
    fieldInterpretations: {},
  };
}

/**
 * Result of processing a directory of OMR sheets.
 */
export interface DirectoryProcessingResult {
  totalFiles: number;
  successful: number;
  errors: number;
  multiMarked: number;
  results: OMRResult[];
  processingTime: number;
}

/**
 * Create a default DirectoryProcessingResult.
 */
export function createDirectoryProcessingResult(): DirectoryProcessingResult {
  return {
    totalFiles: 0,
    successful: 0,
    errors: 0,
    multiMarked: 0,
    results: [],
    processingTime: 0.0,
  };
}

/**
 * Add a single file result to directory processing result.
 */
export function addResult(
  dirResult: DirectoryProcessingResult,
  result: OMRResult
): DirectoryProcessingResult {
  const updated = { ...dirResult };
  updated.results = [...updated.results, result];
  updated.totalFiles += 1;

  if (result.status === 'success') {
    if (result.isMultiMarked) {
      updated.multiMarked += 1;
    } else {
      updated.successful += 1;
    }
  } else {
    updated.errors += 1;
  }

  return updated;
}

