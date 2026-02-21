/**
 * Base OMRChecker Error
 * Source: src/exceptions.py
 */
export class OMRCheckerError extends Error {
  constructor(message: string, public context?: Record<string, any>) {
    super(message);
    this.name = 'OMRCheckerError';
  }
}

/**
 * InputError
 * Source: exceptions.py
 */
export class InputError extends OMRCheckerError {
  constructor(message: string, public context?: Record<string, any>) {
    super(message);
    this.name = 'InputError';
  }
}

/**
 * OutputError
 * Source: exceptions.py
 */
export class OutputError extends OMRCheckerError {
  constructor(message: string, public context?: Record<string, any>) {
    super(message);
    this.name = 'OutputError';
  }
}

/**
 * ValidationError
 * Source: exceptions.py
 */
export class ValidationError extends OMRCheckerError {
  constructor(message: string, public context?: Record<string, any>) {
    super(message);
    this.name = 'ValidationError';
  }
}

/**
 * ProcessingError
 * Source: exceptions.py
 */
export class ProcessingError extends OMRCheckerError {
  constructor(message: string, public context?: Record<string, any>) {
    super(message);
    this.name = 'ProcessingError';
  }
}

/**
 * TemplateError
 * Source: exceptions.py
 */
export class TemplateError extends OMRCheckerError {
  constructor(message: string, public context?: Record<string, any>) {
    super(message);
    this.name = 'TemplateError';
  }
}

/**
 * EvaluationError
 * Source: exceptions.py
 */
export class EvaluationError extends OMRCheckerError {
  constructor(message: string, public context?: Record<string, any>) {
    super(message);
    this.name = 'EvaluationError';
  }
}

/**
 * SecurityError
 * Source: exceptions.py
 */
export class SecurityError extends OMRCheckerError {
  constructor(message: string, public context?: Record<string, any>) {
    super(message);
    this.name = 'SecurityError';
  }
}

/**
 * ConfigError
 * Source: exceptions.py
 */
export class ConfigError extends OMRCheckerError {
  constructor(message: string, public context?: Record<string, any>) {
    super(message);
    this.name = 'ConfigError';
  }
}

