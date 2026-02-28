/**
 * Custom exception hierarchy for OMRChecker.
 * 
 * This module defines all custom exceptions used throughout the application,
 * providing better error handling, debugging, and user feedback.
 */

export class OMRCheckerError extends Error {
  public context: Record<string, any>;

  constructor(message: string, context: Record<string, any> = {}) {
    super(message);
    this.name = 'OMRCheckerError';
    this.context = context;
    Object.setPrototypeOf(this, OMRCheckerError.prototype);
  }

  toString(): string {
    if (Object.keys(this.context).length > 0) {
      const contextStr = Object.entries(this.context)
        .map(([k, v]) => `${k}=${v}`)
        .join(', ');
      return `${this.message} (${contextStr})`;
    }
    return this.message;
  }
}

// ============================================================================
// Input/Output Exceptions
// ============================================================================

export class InputError extends OMRCheckerError {
  constructor(message: string, context: Record<string, any> = {}) {
    super(message, context);
    this.name = 'InputError';
  }
}

export class ImageReadError extends InputError {
  public path: string;
  public reason?: string;

  constructor(path: string, reason?: string) {
    const msg = reason ? `Unable to read image: '${path}' - ${reason}` : `Unable to read image: '${path}'`;
    super(msg, { path, reason });
    this.name = 'ImageReadError';
    this.path = path;
    this.reason = reason;
  }
}

// ============================================================================
// Validation Exceptions
// ============================================================================

export class ValidationError extends OMRCheckerError {
  constructor(message: string, context: Record<string, any> = {}) {
    super(message, context);
    this.name = 'ValidationError';
  }
}

export class SchemaValidationError extends ValidationError {
  public schemaName: string;
  public errors: string[];
  public dataPath?: string;

  constructor(schemaName: string, errors: string[], dataPath?: string) {
    const msg = dataPath
      ? `Schema validation failed for '${schemaName}' at '${dataPath}'`
      : `Schema validation failed for '${schemaName}'`;
    super(msg, { schema: schemaName, errors, dataPath });
    this.name = 'SchemaValidationError';
    this.schemaName = schemaName;
    this.errors = errors;
    this.dataPath = dataPath;
  }
}

// ============================================================================
// Processing Exceptions
// ============================================================================

export class ProcessingError extends OMRCheckerError {
  constructor(message: string, context: Record<string, any> = {}) {
    super(message, context);
    this.name = 'ProcessingError';
  }
}

export class ImageProcessingError extends ProcessingError {
  public operation?: string;
  public filePath?: string;
  public reason?: string;

  constructor(message: string, context: Record<string, any> = {}) {
    super(message, context);
    this.name = 'ImageProcessingError';
    this.operation = context.operation;
    this.filePath = context.filePath;
    this.reason = context.reason;
  }
}

export class AlignmentError extends ProcessingError {
  public filePath: string;
  public reason?: string;

  constructor(filePath: string, reason?: string) {
    const msg = reason ? `Image alignment failed for '${filePath}': ${reason}` : `Image alignment failed for '${filePath}'`;
    super(msg, { filePath, reason });
    this.name = 'AlignmentError';
    this.filePath = filePath;
    this.reason = reason;
  }
}

export class BubbleDetectionError extends ProcessingError {
  public filePath: string;
  public fieldId?: string;
  public reason?: string;

  constructor(filePath: string, fieldId?: string, reason?: string) {
    let msg = `Bubble detection failed for '${filePath}'`;
    if (fieldId) msg += ` at field '${fieldId}'`;
    if (reason) msg += `: ${reason}`;
    super(msg, { filePath, fieldId, reason });
    this.name = 'BubbleDetectionError';
    this.filePath = filePath;
    this.fieldId = fieldId;
    this.reason = reason;
  }
}

// ============================================================================
// Template Exceptions
// ============================================================================

export class TemplateError extends OMRCheckerError {
  constructor(message: string, context: Record<string, any> = {}) {
    super(message, context);
    this.name = 'TemplateError';
  }
}

export class PreprocessorError extends TemplateError {
  public preprocessorName: string;
  public filePath?: string;
  public reason?: string;

  constructor(preprocessorName: string, filePath?: string, reason?: string) {
    let msg = `Preprocessor '${preprocessorName}' failed`;
    if (filePath) msg += ` for '${filePath}'`;
    if (reason) msg += `: ${reason}`;
    super(msg, { preprocessor: preprocessorName, filePath, reason });
    this.name = 'PreprocessorError';
    this.preprocessorName = preprocessorName;
    this.filePath = filePath;
    this.reason = reason;
  }
}

export class FieldDefinitionError extends TemplateError {
  public fieldId: string;
  public reason: string;
  public templatePath?: string;

  constructor(fieldId: string, reason: string, templatePath?: string) {
    const msg = templatePath
      ? `Invalid field definition '${fieldId}': ${reason} in '${templatePath}'`
      : `Invalid field definition '${fieldId}': ${reason}`;
    super(msg, { fieldId, reason, templatePath });
    this.name = 'FieldDefinitionError';
    this.fieldId = fieldId;
    this.reason = reason;
    this.templatePath = templatePath;
  }
}

// ============================================================================
// Evaluation Exceptions
// ============================================================================

export class EvaluationError extends OMRCheckerError {
  constructor(message: string, context: Record<string, any> = {}) {
    super(message, context);
    this.name = 'EvaluationError';
  }
}

export class AnswerKeyError extends EvaluationError {
  public reason: string;
  public questionId?: string;

  constructor(reason: string, questionId?: string) {
    const msg = questionId ? `Answer key error: ${reason} (question: ${questionId})` : `Answer key error: ${reason}`;
    super(msg, { reason, questionId });
    this.name = 'AnswerKeyError';
    this.reason = reason;
    this.questionId = questionId;
  }
}

export class ScoringError extends EvaluationError {
  public reason: string;
  public filePath?: string;
  public questionId?: string;

  constructor(reason: string, filePath?: string, questionId?: string) {
    let msg = `Scoring failed: ${reason}`;
    if (filePath) msg += ` for '${filePath}'`;
    if (questionId) msg += ` at question '${questionId}'`;
    super(msg, { reason, filePath, questionId });
    this.name = 'ScoringError';
    this.reason = reason;
    this.filePath = filePath;
    this.questionId = questionId;
  }
}

// ============================================================================
// Configuration Exceptions
// ============================================================================

export class ConfigError extends OMRCheckerError {
  constructor(message: string, context: Record<string, any> = {}) {
    super(message, context);
    this.name = 'ConfigError';
  }
}

export class InvalidConfigValueError extends ConfigError {
  public key: string;
  public value: any;
  public reason: string;

  constructor(key: string, value: any, reason: string) {
    super(`Invalid configuration value for '${key}': ${value} - ${reason}`, {
      key,
      value: String(value),
      reason,
    });
    this.name = 'InvalidConfigValueError';
    this.key = key;
    this.value = value;
    this.reason = reason;
  }
}
