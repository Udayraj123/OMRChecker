/**
 * Processor manager for instantiating image preprocessors by name.
 *
 * TypeScript port of src/processors/manager.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import { OMRCheckerError } from '../../core/exceptions';
import { Processor } from '../base';
import { AutoRotate } from './AutoRotate';
import { Contrast } from './Contrast';
import { CropOnMarkers } from './CropOnMarkers';
import { CropPage } from './CropPage';
import { GaussianBlur } from './GaussianBlur';
import { Levels } from './Levels';
import { MedianBlur } from './MedianBlur';
import type { ImageTemplatePreprocessor } from './base';
import type { SaveImageOps } from './base';

/**
 * Supported processor names.
 * Must match the keys in PROCESSOR_MANAGER.
 */
export const SUPPORTED_PROCESSOR_NAMES = [
  'AutoRotate',
  'Contrast',
  'CropOnMarkers',
  'CropPage',
  'GaussianBlur',
  'Levels',
  'MedianBlur',
  // Note: FeatureBasedAlignment is not a preprocessor, it's an alignment processor
  // TODO: extract AlignOnMarkers preprocess from WarpOnPoints instead, or rename CropOnMarkers to something better with enableCropping support?
] as const;

/**
 * Processor factory function type.
 * Some processors extend ImageTemplatePreprocessor, others extend Processor directly.
 */
type ProcessorFactory = (
  options: any, // Use any to allow different option types per processor
  relativeDir: string,
  saveImageOps: SaveImageOps,
  defaultProcessingImageShape: [number, number]
) => ImageTemplatePreprocessor | Processor;

/**
 * Processor manager mapping processor names to their factory functions.
 *
 * Port of Python's PROCESSOR_MANAGER.
 * Note: Hard-coded mapping to support working export (similar to PyInstaller requirement in Python).
 *
 * Note: In Python, all processors extend ImageTemplatePreprocessor. In TypeScript,
 * some processors (AutoRotate, Contrast, GaussianBlur, Levels, MedianBlur) extend
 * Processor directly. We use factory functions to handle both cases.
 */
export const PROCESSOR_MANAGER: Record<string, ProcessorFactory> = {
  AutoRotate: (options, _relativeDir, _saveImageOps, _defaultProcessingImageShape) => {
    return new AutoRotate(options);
  },
  Contrast: (options, _relativeDir, _saveImageOps, _defaultProcessingImageShape) => {
    return new Contrast(options);
  },
  CropOnMarkers: (options, relativeDir, saveImageOps, defaultProcessingImageShape) => {
    return new CropOnMarkers(options, relativeDir, saveImageOps, defaultProcessingImageShape);
  },
  CropPage: (options, relativeDir, saveImageOps, defaultProcessingImageShape) => {
    return new CropPage(options, relativeDir, saveImageOps, defaultProcessingImageShape);
  },
  GaussianBlur: (options, _relativeDir, _saveImageOps, _defaultProcessingImageShape) => {
    return new GaussianBlur(options);
  },
  Levels: (options, _relativeDir, _saveImageOps, _defaultProcessingImageShape) => {
    return new Levels(options);
  },
  MedianBlur: (options, _relativeDir, _saveImageOps, _defaultProcessingImageShape) => {
    return new MedianBlur(options);
  },
  // TODO: extract AlignOnMarkers preprocess from WarpOnPoints instead, or rename CropOnMarkers to something better with enableCropping support?
};

/**
 * Validate that PROCESSOR_MANAGER keys match SUPPORTED_PROCESSOR_NAMES.
 *
 * Port of Python's validation logic.
 */
export function validateProcessorManager(): void {
  const managerKeys = new Set(Object.keys(PROCESSOR_MANAGER));
  const supportedNames = new Set(SUPPORTED_PROCESSOR_NAMES);

  if (managerKeys.size !== supportedNames.size ||
      ![...managerKeys].every(key => supportedNames.has(key as typeof SUPPORTED_PROCESSOR_NAMES[number]))) {
    throw new OMRCheckerError(
      `Processor keys mismatch: ${Array.from(managerKeys).join(', ')} != ${Array.from(supportedNames).join(', ')}`,
      {
        registered: Array.from(managerKeys),
        supported: Array.from(supportedNames),
      }
    );
  }
}

// Validate on module load (similar to Python)
validateProcessorManager();

