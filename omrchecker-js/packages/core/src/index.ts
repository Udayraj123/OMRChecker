/**
 * OMRChecker Core Library
 *
 * TypeScript port of OMRChecker for browser use.
 * Maintains 1:1 correspondence with Python implementation.
 */

// Core types
export type {
  ProcessorConfig,
  OMRResult,
  DirectoryProcessingResult,
} from './core/types';

export {
  createDefaultProcessorConfig,
  createOMRResult,
  createDirectoryProcessingResult,
  addResult,
} from './core/types';

// Processor base classes
export type { ProcessingContext } from './processors/base';
export { Processor, createProcessingContext } from './processors/base';

// Exceptions
export {
  OMRCheckerError,
  InputError,
  InputFileNotFoundError,
  ConfigLoadError,
  ImageReadError,
  ImageProcessingError,
} from './core/exceptions';

// Utilities
export { MathUtils, EdgeType } from './utils/math';
export type { Point, Rectangle } from './utils/math';
export {
  euclideanDistance,
  vectorMagnitude,
  bboxCenter,
} from './utils/geometry';
export { Logger, logger } from './utils/logger';
export { PathUtils, loadJson } from './utils/file';
export { threadSafeCsvAppend } from './utils/csv';

// Constants
export * from './utils/constants';
export type { ColorTuple, BubbleFieldType } from './utils/constants';

// Schema validation
export { validateConfig } from './schemas/configSchema';
export { validateTemplate } from './schemas/templateSchema';
export type { ValidationResult } from './schemas/common';

// Version
export const VERSION = '1.0.0';


