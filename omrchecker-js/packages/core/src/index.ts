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

// Image processors - Basic filters
export { GaussianBlur } from './processors/image/GaussianBlur';
export { MedianBlur } from './processors/image/MedianBlur';
export { Contrast } from './processors/image/Contrast';

// Image processors - Advanced
export { AutoRotate, type AutoRotateOptions } from './processors/image/AutoRotate';
export { Levels } from './processors/image/Levels';

// Threshold strategies
export {
  GlobalThreshold,
  type ThresholdConfig,
  type ThresholdResult,
} from './processors/threshold/GlobalThreshold';
export { LocalThreshold } from './processors/threshold/LocalThreshold';
export { AdaptiveThreshold } from './processors/threshold/AdaptiveThreshold';

// Detection
export {
  SimpleBubbleDetector,
  type BubbleLocation,
  type BubbleDetectionResult,
  type FieldDetectionResult,
} from './processors/detection/SimpleBubbleDetector';

// Utilities
export { Logger } from './utils/logger';
export { ImageUtils } from './utils/ImageUtils';
export * from './utils/geometry';
export * from './utils/math';
export * from './utils/file';

// Schemas
export * from './schemas/templateSchema';
export * from './schemas/configSchema';
