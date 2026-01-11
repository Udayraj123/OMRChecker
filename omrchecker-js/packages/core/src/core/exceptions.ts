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

