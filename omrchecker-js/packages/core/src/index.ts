/**
 * OMRChecker Core Library - TypeScript Port
 *
 * Main entry point for the OMRChecker core functionality.
 */

// Core types and interfaces
export type { ProcessingContext } from './processors/base';
export { Processor, createProcessingContext } from './processors/base';

// Pipeline orchestration
export { ProcessingPipeline, type PipelineConfig } from './processors/Pipeline';

// Image filters
export { GaussianBlur, MedianBlur, Contrast } from './processors/image/filters';

// Utilities
export { Logger } from './utils/logger';
export * from './utils/geometry';
export * from './utils/math';
export * from './utils/file';

// Schemas
export * from './schemas/templateSchema';
export * from './schemas/configSchema';
