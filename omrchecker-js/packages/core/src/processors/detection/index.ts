/**
 * Detection processors index.
 * Exports detection classes following Python architecture.
 */

// Base classes
export { FieldDetection, TextDetection } from './base';

// Bubbles threshold detection
export { BubblesFieldDetection } from './bubbles_threshold';

// Detection result models
export {
  ScanQuality,
  BubbleMeanValue,
  BubbleFieldDetectionResult,
  OCRFieldDetectionResult,
  BarcodeFieldDetectionResult,
  FileDetectionResults,
  type BubbleLocation,
} from './models';

// Template file runner
export { TemplateFileRunner } from './templateFileRunner';
