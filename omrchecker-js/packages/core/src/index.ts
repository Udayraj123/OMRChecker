/**
 * OMRChecker Core Library - TypeScript Port
 *
 * Main entry point for the OMRChecker core functionality.
 */

// Core OMR Processor
export {
  OMRProcessor,
  type OMRProcessorConfig,
  type OMRSheetResult,
} from './core/OMRProcessor';

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
export { CropPage } from './processors/image/CropPage';
export { CropOnMarkers, type CropOnMarkersOptions } from './processors/image/CropOnMarkers';

// Image processors - Base class
export {
  ImageTemplatePreprocessor,
  type ImagePreprocessorOptions,
  type SaveImageOps as SaveImageOpsInterface,
} from './processors/image/base';

// Image processors - Coordinator
export { PreprocessingProcessor } from './processors/image/coordinator';

// Alignment processors
export { AlignmentProcessor } from './processors/alignment/AlignmentProcessor';
export {
  applyTemplateAlignment,
  type AlignmentResult,
} from './processors/alignment/templateAlignment';

// Threshold strategies
export {
  GlobalThreshold,
  type ThresholdConfig,
  type ThresholdResult,
} from './processors/threshold/GlobalThreshold';
export { LocalThreshold } from './processors/threshold/LocalThreshold';
export { AdaptiveThreshold } from './processors/threshold/AdaptiveThreshold';

// Detection - Proper Python architecture mapping
export {
  // Base classes
  FieldDetection,
  TextDetection,
  // Bubbles threshold detection
  BubblesFieldDetection,
  // Detection result models
  ScanQuality,
  BubbleMeanValue,
  BubbleFieldDetectionResult,
  OCRFieldDetectionResult,
  BarcodeFieldDetectionResult,
  FileDetectionResults,
  type BubbleLocation,
} from './processors/detection';

// Evaluation - Core modules
export {
  EvaluationProcessor,
  AnswerMatcher,
  AnswerType,
  Verdict,
  SchemaVerdict,
  SectionMarkingScheme,
  MarkingSchemeType,
  EvaluationConfig,
  EvaluationConfigForSet,
  DEFAULT_SECTION_KEY,
  BONUS_SECTION_PREFIX,
  DEFAULT_SET_NAME,
} from './processors/evaluation';

// Evaluation - Types (still from processor for backward compatibility)
export type {
  EvaluationConfig as EvaluationConfigForResponse,
  QuestionMeta,
  EvaluationMetaDict,
  MarkingScheme,
} from './processors/evaluation/EvaluationProcessor';

// Evaluation - New types
export type {
  SectionSchemeConfig,
  EvaluationJSON,
  ConditionalSet,
  EvaluationOptions,
  OutputsConfiguration,
} from './processors/evaluation';

// Template
export {
  type TemplateConfig,
  type FieldBlock,
  type BubbleFieldType,
  type AlignmentConfig,
  type PreProcessorConfig,
  type ParsedTemplate,
  type ParsedField,
  type TuningConfig,
  BUILTIN_BUBBLE_FIELD_TYPES,
  DEFAULT_TEMPLATE_CONFIG,
} from './template/types';
export { TemplateLoader, type TemplateLayoutData } from './template/TemplateLoader';
export { TemplateLayout } from './template/TemplateLayout';
export { TemplateDrawing, TemplateDrawingUtils } from './template/TemplateDrawing';
export { templateConfigFromDict, templateConfigToDict } from './schemas/models/template';
export { TEMPLATE_DEFAULTS } from './schemas/defaults/template';
export { CONFIG_DEFAULTS } from './schemas/defaults/config';
export { EVALUATION_CONFIG_DEFAULTS } from './schemas/defaults/evaluation';
export { Config, FileGroupingConfig, type ThresholdingConfig, type OutputsConfig, type ProcessingConfig, type MLConfig, type ShiftDetectionConfig, type VisualizationConfig, type GroupingRule } from './schemas/models/config';
export { EvaluationConfig as EvaluationConfigModel, type DrawScoreConfig, type DrawAnswersSummaryConfig, type DrawQuestionVerdictsConfig, type DrawDetectedBubbleTextsConfig, type DrawAnswerGroupsConfig, type OutputsConfiguration as EvaluationOutputsConfiguration } from './schemas/models/evaluation';

// Utils
export { FilePatternResolver } from './utils/filePatternResolver';
export { dataclassToDict } from './utils/serialization';
export { SaveImageOps } from './utils/SaveImageOps';

// Layout classes
export * from './processors/layout/field';
export * from './processors/layout/fieldBlock';

// Utilities
export { Logger } from './utils/logger';
export { ImageUtils } from './utils/ImageUtils';
export * from './utils/geometry';
export * from './utils/file';
export { isObject, deepMerge, deepClone } from './utils/object';
// Export math utils without EdgeType to avoid conflict with processors/constants
export { MathUtils, type Point } from './utils/math';
export {
  DrawingUtils,
  CLR_BLACK,
  CLR_WHITE,
  CLR_GRAY,
  CLR_DARK_GRAY,
  CLR_GREEN,
  CLR_RED,
  CLR_BLUE,
  CLR_YELLOW,
  TEXT_SIZE,
  type BoxStyle,
  type BoxEdge,
} from './utils/drawing';
export { InteractionUtils } from './utils/InteractionUtils';
export {
  ImageSaverUtils,
  saveImage,
  appendSaveImage,
  requestDirectoryAccess,
  downloadAllStoredImages,
  clearStoredImages,
  getStoredImages,
  getStoredImageCount,
} from './utils/ImageSaver';

// Schemas
export * from './schemas/templateSchema';
export * from './schemas/configSchema';
export * from './schemas/evaluationSchema';
export * from './schemas/constants';

// Processor constants
export * from './processors/constants';
