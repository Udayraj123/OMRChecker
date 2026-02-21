/**
 * OMRChecker.js - Browser-based OMR Detection
 * TypeScript port of OMRChecker Python library
 */

// Core exports
export { Template } from './core/template';
export { Field } from './core/field';
export { FieldBlock } from './core/field-block';
export { Config } from './core/config';

// Processor exports
export { processImage } from './processors/pipeline';
export { alignImage } from './processors/alignment';
export { detectBubbles } from './processors/detection';
export { evaluateResponse } from './processors/evaluation';

// Utility exports
export { logger } from './utils/logger/logger';
export { validateTemplate } from './utils/validation/template';
export { validateConfig } from './utils/validation/config';

// Type exports
export type * from './types';

// Default export
export { OMRChecker } from './omrchecker';
