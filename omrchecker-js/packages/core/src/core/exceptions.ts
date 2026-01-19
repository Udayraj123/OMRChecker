/**
 * Custom exceptions for OMRChecker
 *
 * TypeScript port of src/exceptions.py (partial - only needed exceptions)
 */

export class OMRCheckerError extends Error {
  context: Record<string, unknown>;

  constructor(message: string, context?: Record<string, unknown>) {
    super(message);
    this.name = this.constructor.name;
    this.context = context || {};
    Object.setPrototypeOf(this, new.target.prototype);
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

export class InputError extends OMRCheckerError {}

export class InputFileNotFoundError extends InputError {
  path: string;
  fileType?: string;

  constructor(path: string, fileType?: string) {
    const fileDesc = fileType ? `${fileType} ` : '';
    super(`Input ${fileDesc}file not found: '${path}'`, {
      path,
      fileType,
    });
    this.path = path;
    this.fileType = fileType;
  }
}

export class TemplateError extends OMRCheckerError {}

export class FieldDefinitionError extends TemplateError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, context);
  }
}

export class ConfigError extends OMRCheckerError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, context);
  }
}

export class ConfigLoadError extends OMRCheckerError {
  path: string;

  constructor(path: string, reason: string) {
    super(`Failed to load configuration from '${path}': ${reason}`, {
      path,
      reason,
    });
    this.path = path;
  }
}

export class ImageReadError extends InputError {
  path: string;
  reason?: string;

  constructor(path: string, reason?: string) {
    let msg = `Unable to read image: '${path}'`;
    if (reason) {
      msg += ` - ${reason}`;
    }
    super(msg, { path, reason });
    this.path = path;
    this.reason = reason;
  }
}

export class ImageProcessingError extends OMRCheckerError {}

export class TemplateValidationError extends OMRCheckerError {
  templatePath?: string;

  constructor(templatePath: string, message: string, context?: Record<string, unknown>) {
    super(message, { ...context, templatePath });
    this.templatePath = templatePath;
  }
}

export class TemplateNotFoundError extends TemplateError {
  searchPath: string;

  constructor(searchPath: string) {
    super(`No template.json found in directory tree of '${searchPath}'`, { searchPath });
    this.searchPath = searchPath;
  }
}

export class TemplateLoadError extends TemplateError {
  path: string;
  reason: string;

  constructor(path: string, reason: string) {
    super(`Failed to load template '${path}': ${reason}`, { path, reason });
    this.path = path;
    this.reason = reason;
  }
}

// ============================================================================
// Evaluation Exceptions
// ============================================================================

export class EvaluationError extends OMRCheckerError {}

export class EvaluationConfigNotFoundError extends EvaluationError {
  searchPath: string;

  constructor(searchPath: string) {
    super(`No evaluation.json found at '${searchPath}'`, { searchPath });
    this.searchPath = searchPath;
  }
}

export class EvaluationConfigLoadError extends EvaluationError {
  path: string;
  reason: string;

  constructor(path: string, reason: string) {
    super(`Failed to load evaluation config '${path}': ${reason}`, { path, reason });
    this.path = path;
    this.reason = reason;
  }
}

export class AnswerKeyError extends EvaluationError {
  reason: string;
  questionId?: string;

  constructor(reason: string, questionId?: string) {
    let msg = `Answer key error: ${reason}`;
    if (questionId) {
      msg += ` (question: ${questionId})`;
    }
    super(msg, { reason, questionId });
    this.reason = reason;
    this.questionId = questionId;
  }
}

export class ScoringError extends EvaluationError {
  reason: string;
  filePath?: string;
  questionId?: string;

  constructor(reason: string, filePath?: string, questionId?: string) {
    let msg = `Scoring failed: ${reason}`;
    if (filePath) {
      msg += ` for '${filePath}'`;
    }
    if (questionId) {
      msg += ` at question '${questionId}'`;
    }
    super(msg, { reason, filePath, questionId });
    this.reason = reason;
    this.filePath = filePath;
    this.questionId = questionId;
  }
}

// ============================================================================
// Security Exceptions
// ============================================================================

export class SecurityError extends OMRCheckerError {}

export class PathTraversalError extends SecurityError {
  path: string;
  basePath?: string;

  constructor(path: string, basePath?: string) {
    const msg = basePath
      ? `Path traversal detected: '${path}' (base: '${basePath}')`
      : `Path traversal detected: '${path}'`;
    super(msg, { path, basePath });
    this.path = path;
    this.basePath = basePath;
  }
}

