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

// Utilities
export { MathUtils, EdgeType } from './utils/math';
export type { Point, Rectangle } from './utils/math';

// Version
export const VERSION = '1.0.0';

