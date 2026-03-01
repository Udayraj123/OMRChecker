import { describe, it, expect } from 'vitest';
import {
  OMRCheckerError,
  InputError,
  ImageReadError,
  ValidationError,
  SchemaValidationError,
  ProcessingError,
  ImageProcessingError,
  AlignmentError,
  BubbleDetectionError,
  TemplateError,
  PreprocessorError,
  FieldDefinitionError,
  EvaluationError,
  AnswerKeyError,
  ScoringError,
  ConfigError,
  InvalidConfigValueError,
} from '../../src/utils/exceptions';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Returns a plain object snapshot of all relevant Error properties so that
 * snapshot assertions are stable and readable.
 */
function snap(err: Error & { context?: Record<string, any> }) {
  return {
    name: err.name,
    message: err.message,
    context: (err as any).context,
  };
}

// ---------------------------------------------------------------------------
// OMRCheckerError — base class
// ---------------------------------------------------------------------------

describe('OMRCheckerError', () => {
  it('snapshot — message only', () => {
    const err = new OMRCheckerError('Something went wrong');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {},
        "message": "Something went wrong",
        "name": "OMRCheckerError",
      }
    `);
  });

  it('snapshot — message with context', () => {
    const err = new OMRCheckerError('Something went wrong', { file: 'test.png', code: 42 });
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "code": 42,
          "file": "test.png",
        },
        "message": "Something went wrong",
        "name": "OMRCheckerError",
      }
    `);
  });

  it('snapshot — toString() without context', () => {
    const err = new OMRCheckerError('Plain message');
    expect(err.toString()).toMatchInlineSnapshot(`"Plain message"`);
  });

  it('snapshot — toString() with context', () => {
    const err = new OMRCheckerError('Rich message', { key: 'value', num: 99 });
    expect(err.toString()).toMatchInlineSnapshot(`"Rich message (key=value, num=99)"`);
  });

  it('is instanceof Error', () => {
    const err = new OMRCheckerError('test');
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// InputError
// ---------------------------------------------------------------------------

describe('InputError', () => {
  it('snapshot — default context', () => {
    const err = new InputError('Bad input received');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {},
        "message": "Bad input received",
        "name": "InputError",
      }
    `);
  });

  it('snapshot — with context', () => {
    const err = new InputError('Bad input received', { source: 'stdin' });
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "source": "stdin",
        },
        "message": "Bad input received",
        "name": "InputError",
      }
    `);
  });

  it('is instanceof OMRCheckerError', () => {
    const err = new InputError('test');
    expect(err).toBeInstanceOf(OMRCheckerError);
    expect(err).toBeInstanceOf(InputError);
  });
});

// ---------------------------------------------------------------------------
// ImageReadError
// ---------------------------------------------------------------------------

describe('ImageReadError', () => {
  it('snapshot — path only', () => {
    const err = new ImageReadError('/data/sheet.png');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "path": "/data/sheet.png",
          "reason": undefined,
        },
        "message": "Unable to read image: '/data/sheet.png'",
        "name": "ImageReadError",
      }
    `);
  });

  it('snapshot — path and reason', () => {
    const err = new ImageReadError('/data/sheet.png', 'file not found');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "path": "/data/sheet.png",
          "reason": "file not found",
        },
        "message": "Unable to read image: '/data/sheet.png' - file not found",
        "name": "ImageReadError",
      }
    `);
  });

  it('exposes path and reason properties', () => {
    const err = new ImageReadError('/data/sheet.png', 'permission denied');
    expect(err.path).toBe('/data/sheet.png');
    expect(err.reason).toBe('permission denied');
  });

  it('is instanceof InputError and OMRCheckerError', () => {
    const err = new ImageReadError('/img.png');
    expect(err).toBeInstanceOf(InputError);
    expect(err).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// ValidationError
// ---------------------------------------------------------------------------

describe('ValidationError', () => {
  it('snapshot — message only', () => {
    const err = new ValidationError('Field value out of range');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {},
        "message": "Field value out of range",
        "name": "ValidationError",
      }
    `);
  });

  it('snapshot — message with context', () => {
    const err = new ValidationError('Field value out of range', { field: 'score', max: 100 });
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "field": "score",
          "max": 100,
        },
        "message": "Field value out of range",
        "name": "ValidationError",
      }
    `);
  });

  it('is instanceof OMRCheckerError', () => {
    expect(new ValidationError('test')).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// SchemaValidationError
// ---------------------------------------------------------------------------

describe('SchemaValidationError', () => {
  it('snapshot — without dataPath', () => {
    const err = new SchemaValidationError('AnswerKey', ['required field missing', 'invalid type']);
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "dataPath": undefined,
          "errors": [
            "required field missing",
            "invalid type",
          ],
          "schema": "AnswerKey",
        },
        "message": "Schema validation failed for 'AnswerKey'",
        "name": "SchemaValidationError",
      }
    `);
  });

  it('snapshot — with dataPath', () => {
    const err = new SchemaValidationError('Template', ['must be array'], '/questions');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "dataPath": "/questions",
          "errors": [
            "must be array",
          ],
          "schema": "Template",
        },
        "message": "Schema validation failed for 'Template' at '/questions'",
        "name": "SchemaValidationError",
      }
    `);
  });

  it('exposes schemaName, errors and dataPath properties', () => {
    const err = new SchemaValidationError('Config', ['err1', 'err2'], '/root');
    expect(err.schemaName).toBe('Config');
    expect(err.errors).toEqual(['err1', 'err2']);
    expect(err.dataPath).toBe('/root');
  });

  it('is instanceof ValidationError and OMRCheckerError', () => {
    const err = new SchemaValidationError('X', []);
    expect(err).toBeInstanceOf(ValidationError);
    expect(err).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// ProcessingError
// ---------------------------------------------------------------------------

describe('ProcessingError', () => {
  it('snapshot — message only', () => {
    const err = new ProcessingError('Processing pipeline failed');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {},
        "message": "Processing pipeline failed",
        "name": "ProcessingError",
      }
    `);
  });

  it('is instanceof OMRCheckerError', () => {
    expect(new ProcessingError('test')).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// ImageProcessingError
// ---------------------------------------------------------------------------

describe('ImageProcessingError', () => {
  it('snapshot — message only', () => {
    const err = new ImageProcessingError('Grayscale conversion failed');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {},
        "message": "Grayscale conversion failed",
        "name": "ImageProcessingError",
      }
    `);
  });

  it('snapshot — with full context', () => {
    const err = new ImageProcessingError('Threshold failed', {
      operation: 'threshold',
      filePath: '/imgs/q1.png',
      reason: 'NaN pixel values',
    });
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "filePath": "/imgs/q1.png",
          "operation": "threshold",
          "reason": "NaN pixel values",
        },
        "message": "Threshold failed",
        "name": "ImageProcessingError",
      }
    `);
  });

  it('maps context fields to instance properties', () => {
    const err = new ImageProcessingError('msg', {
      operation: 'blur',
      filePath: '/f.png',
      reason: 'out of memory',
    });
    expect(err.operation).toBe('blur');
    expect(err.filePath).toBe('/f.png');
    expect(err.reason).toBe('out of memory');
  });

  it('is instanceof ProcessingError and OMRCheckerError', () => {
    const err = new ImageProcessingError('test');
    expect(err).toBeInstanceOf(ProcessingError);
    expect(err).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// AlignmentError
// ---------------------------------------------------------------------------

describe('AlignmentError', () => {
  it('snapshot — path only', () => {
    const err = new AlignmentError('/data/scan.png');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "filePath": "/data/scan.png",
          "reason": undefined,
        },
        "message": "Image alignment failed for '/data/scan.png'",
        "name": "AlignmentError",
      }
    `);
  });

  it('snapshot — path and reason', () => {
    const err = new AlignmentError('/data/scan.png', 'no contours found');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "filePath": "/data/scan.png",
          "reason": "no contours found",
        },
        "message": "Image alignment failed for '/data/scan.png': no contours found",
        "name": "AlignmentError",
      }
    `);
  });

  it('exposes filePath and reason properties', () => {
    const err = new AlignmentError('/scan.png', 'warp failed');
    expect(err.filePath).toBe('/scan.png');
    expect(err.reason).toBe('warp failed');
  });

  it('is instanceof ProcessingError and OMRCheckerError', () => {
    const err = new AlignmentError('/x.png');
    expect(err).toBeInstanceOf(ProcessingError);
    expect(err).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// BubbleDetectionError
// ---------------------------------------------------------------------------

describe('BubbleDetectionError', () => {
  it('snapshot — path only', () => {
    const err = new BubbleDetectionError('/sheet.png');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "fieldId": undefined,
          "filePath": "/sheet.png",
          "reason": undefined,
        },
        "message": "Bubble detection failed for '/sheet.png'",
        "name": "BubbleDetectionError",
      }
    `);
  });

  it('snapshot — path and fieldId', () => {
    const err = new BubbleDetectionError('/sheet.png', 'Q1');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "fieldId": "Q1",
          "filePath": "/sheet.png",
          "reason": undefined,
        },
        "message": "Bubble detection failed for '/sheet.png' at field 'Q1'",
        "name": "BubbleDetectionError",
      }
    `);
  });

  it('snapshot — path, fieldId and reason', () => {
    const err = new BubbleDetectionError('/sheet.png', 'Q1', 'threshold too high');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "fieldId": "Q1",
          "filePath": "/sheet.png",
          "reason": "threshold too high",
        },
        "message": "Bubble detection failed for '/sheet.png' at field 'Q1': threshold too high",
        "name": "BubbleDetectionError",
      }
    `);
  });

  it('exposes filePath, fieldId and reason properties', () => {
    const err = new BubbleDetectionError('/img.png', 'Q5', 'low contrast');
    expect(err.filePath).toBe('/img.png');
    expect(err.fieldId).toBe('Q5');
    expect(err.reason).toBe('low contrast');
  });

  it('is instanceof ProcessingError and OMRCheckerError', () => {
    const err = new BubbleDetectionError('/img.png');
    expect(err).toBeInstanceOf(ProcessingError);
    expect(err).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// TemplateError
// ---------------------------------------------------------------------------

describe('TemplateError', () => {
  it('snapshot — message only', () => {
    const err = new TemplateError('Template parsing failed');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {},
        "message": "Template parsing failed",
        "name": "TemplateError",
      }
    `);
  });

  it('is instanceof OMRCheckerError', () => {
    expect(new TemplateError('test')).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// PreprocessorError
// ---------------------------------------------------------------------------

describe('PreprocessorError', () => {
  it('snapshot — name only', () => {
    const err = new PreprocessorError('CropPage');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "filePath": undefined,
          "preprocessor": "CropPage",
          "reason": undefined,
        },
        "message": "Preprocessor 'CropPage' failed",
        "name": "PreprocessorError",
      }
    `);
  });

  it('snapshot — name and filePath', () => {
    const err = new PreprocessorError('CropPage', '/sheet.png');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "filePath": "/sheet.png",
          "preprocessor": "CropPage",
          "reason": undefined,
        },
        "message": "Preprocessor 'CropPage' failed for '/sheet.png'",
        "name": "PreprocessorError",
      }
    `);
  });

  it('snapshot — name, filePath and reason', () => {
    const err = new PreprocessorError('CropPage', '/sheet.png', 'corners not detected');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "filePath": "/sheet.png",
          "preprocessor": "CropPage",
          "reason": "corners not detected",
        },
        "message": "Preprocessor 'CropPage' failed for '/sheet.png': corners not detected",
        "name": "PreprocessorError",
      }
    `);
  });

  it('exposes preprocessorName, filePath and reason properties', () => {
    const err = new PreprocessorError('Warp', '/img.png', 'bad transform');
    expect(err.preprocessorName).toBe('Warp');
    expect(err.filePath).toBe('/img.png');
    expect(err.reason).toBe('bad transform');
  });

  it('is instanceof TemplateError and OMRCheckerError', () => {
    const err = new PreprocessorError('X');
    expect(err).toBeInstanceOf(TemplateError);
    expect(err).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// FieldDefinitionError
// ---------------------------------------------------------------------------

describe('FieldDefinitionError', () => {
  it('snapshot — without templatePath', () => {
    const err = new FieldDefinitionError('Q1', 'bubbles count must be positive');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "fieldId": "Q1",
          "reason": "bubbles count must be positive",
          "templatePath": undefined,
        },
        "message": "Invalid field definition 'Q1': bubbles count must be positive",
        "name": "FieldDefinitionError",
      }
    `);
  });

  it('snapshot — with templatePath', () => {
    const err = new FieldDefinitionError('Q1', 'bubbles count must be positive', '/tmpl.json');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "fieldId": "Q1",
          "reason": "bubbles count must be positive",
          "templatePath": "/tmpl.json",
        },
        "message": "Invalid field definition 'Q1': bubbles count must be positive in '/tmpl.json'",
        "name": "FieldDefinitionError",
      }
    `);
  });

  it('exposes fieldId, reason and templatePath properties', () => {
    const err = new FieldDefinitionError('Q2', 'missing bounds', '/t.json');
    expect(err.fieldId).toBe('Q2');
    expect(err.reason).toBe('missing bounds');
    expect(err.templatePath).toBe('/t.json');
  });

  it('is instanceof TemplateError and OMRCheckerError', () => {
    const err = new FieldDefinitionError('Q1', 'bad def');
    expect(err).toBeInstanceOf(TemplateError);
    expect(err).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// EvaluationError
// ---------------------------------------------------------------------------

describe('EvaluationError', () => {
  it('snapshot — message only', () => {
    const err = new EvaluationError('Evaluation step failed');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {},
        "message": "Evaluation step failed",
        "name": "EvaluationError",
      }
    `);
  });

  it('is instanceof OMRCheckerError', () => {
    expect(new EvaluationError('test')).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// AnswerKeyError
// ---------------------------------------------------------------------------

describe('AnswerKeyError', () => {
  it('snapshot — reason only', () => {
    const err = new AnswerKeyError('duplicate question id');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "questionId": undefined,
          "reason": "duplicate question id",
        },
        "message": "Answer key error: duplicate question id",
        "name": "AnswerKeyError",
      }
    `);
  });

  it('snapshot — reason and questionId', () => {
    const err = new AnswerKeyError('duplicate question id', 'Q10');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "questionId": "Q10",
          "reason": "duplicate question id",
        },
        "message": "Answer key error: duplicate question id (question: Q10)",
        "name": "AnswerKeyError",
      }
    `);
  });

  it('exposes reason and questionId properties', () => {
    const err = new AnswerKeyError('missing answer', 'Q3');
    expect(err.reason).toBe('missing answer');
    expect(err.questionId).toBe('Q3');
  });

  it('is instanceof EvaluationError and OMRCheckerError', () => {
    const err = new AnswerKeyError('bad key');
    expect(err).toBeInstanceOf(EvaluationError);
    expect(err).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// ScoringError
// ---------------------------------------------------------------------------

describe('ScoringError', () => {
  it('snapshot — reason only', () => {
    const err = new ScoringError('negative marks config invalid');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "filePath": undefined,
          "questionId": undefined,
          "reason": "negative marks config invalid",
        },
        "message": "Scoring failed: negative marks config invalid",
        "name": "ScoringError",
      }
    `);
  });

  it('snapshot — reason and filePath', () => {
    const err = new ScoringError('negative marks config invalid', '/results/sheet1.png');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "filePath": "/results/sheet1.png",
          "questionId": undefined,
          "reason": "negative marks config invalid",
        },
        "message": "Scoring failed: negative marks config invalid for '/results/sheet1.png'",
        "name": "ScoringError",
      }
    `);
  });

  it('snapshot — reason, filePath and questionId', () => {
    const err = new ScoringError('negative marks config invalid', '/results/sheet1.png', 'Q7');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "filePath": "/results/sheet1.png",
          "questionId": "Q7",
          "reason": "negative marks config invalid",
        },
        "message": "Scoring failed: negative marks config invalid for '/results/sheet1.png' at question 'Q7'",
        "name": "ScoringError",
      }
    `);
  });

  it('exposes reason, filePath and questionId properties', () => {
    const err = new ScoringError('divide by zero', '/sheet.png', 'Q2');
    expect(err.reason).toBe('divide by zero');
    expect(err.filePath).toBe('/sheet.png');
    expect(err.questionId).toBe('Q2');
  });

  it('is instanceof EvaluationError and OMRCheckerError', () => {
    const err = new ScoringError('fail');
    expect(err).toBeInstanceOf(EvaluationError);
    expect(err).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// ConfigError
// ---------------------------------------------------------------------------

describe('ConfigError', () => {
  it('snapshot — message only', () => {
    const err = new ConfigError('Config file not found');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {},
        "message": "Config file not found",
        "name": "ConfigError",
      }
    `);
  });

  it('snapshot — message with context', () => {
    const err = new ConfigError('Config file not found', { path: '/etc/omrchecker.json' });
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "path": "/etc/omrchecker.json",
        },
        "message": "Config file not found",
        "name": "ConfigError",
      }
    `);
  });

  it('is instanceof OMRCheckerError', () => {
    expect(new ConfigError('test')).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// InvalidConfigValueError
// ---------------------------------------------------------------------------

describe('InvalidConfigValueError', () => {
  it('snapshot — string value', () => {
    const err = new InvalidConfigValueError('threshold', 'abc', 'must be a number');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "key": "threshold",
          "reason": "must be a number",
          "value": "abc",
        },
        "message": "Invalid configuration value for 'threshold': abc - must be a number",
        "name": "InvalidConfigValueError",
      }
    `);
  });

  it('snapshot — numeric value', () => {
    const err = new InvalidConfigValueError('zoom', -1, 'must be positive');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "key": "zoom",
          "reason": "must be positive",
          "value": "-1",
        },
        "message": "Invalid configuration value for 'zoom': -1 - must be positive",
        "name": "InvalidConfigValueError",
      }
    `);
  });

  it('snapshot — boolean value', () => {
    const err = new InvalidConfigValueError('debug', true, 'must be false in production');
    expect(snap(err)).toMatchInlineSnapshot(`
      {
        "context": {
          "key": "debug",
          "reason": "must be false in production",
          "value": "true",
        },
        "message": "Invalid configuration value for 'debug': true - must be false in production",
        "name": "InvalidConfigValueError",
      }
    `);
  });

  it('context.value is always stringified', () => {
    const err = new InvalidConfigValueError('size', 0, 'non-zero required');
    // value is stored as String(value) in context
    expect(err.context['value']).toBe('0');
    // but the raw property retains the original type
    expect(err.value).toBe(0);
  });

  it('exposes key, value and reason properties', () => {
    const err = new InvalidConfigValueError('mode', 'fast', 'unknown mode');
    expect(err.key).toBe('mode');
    expect(err.value).toBe('fast');
    expect(err.reason).toBe('unknown mode');
  });

  it('is instanceof ConfigError and OMRCheckerError', () => {
    const err = new InvalidConfigValueError('x', 1, 'bad');
    expect(err).toBeInstanceOf(ConfigError);
    expect(err).toBeInstanceOf(OMRCheckerError);
  });
});

// ---------------------------------------------------------------------------
// Cross-cutting: instanceof hierarchy smoke tests
// ---------------------------------------------------------------------------

describe('Exception hierarchy', () => {
  it('ImageReadError is instanceof InputError → OMRCheckerError → Error', () => {
    const err = new ImageReadError('/img.png');
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(OMRCheckerError);
    expect(err).toBeInstanceOf(InputError);
    expect(err).toBeInstanceOf(ImageReadError);
  });

  it('SchemaValidationError is instanceof ValidationError → OMRCheckerError → Error', () => {
    const err = new SchemaValidationError('S', []);
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(OMRCheckerError);
    expect(err).toBeInstanceOf(ValidationError);
    expect(err).toBeInstanceOf(SchemaValidationError);
  });

  it('AlignmentError is instanceof ProcessingError → OMRCheckerError → Error', () => {
    const err = new AlignmentError('/x.png');
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(OMRCheckerError);
    expect(err).toBeInstanceOf(ProcessingError);
    expect(err).toBeInstanceOf(AlignmentError);
  });

  it('FieldDefinitionError is instanceof TemplateError → OMRCheckerError → Error', () => {
    const err = new FieldDefinitionError('Q1', 'bad');
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(OMRCheckerError);
    expect(err).toBeInstanceOf(TemplateError);
    expect(err).toBeInstanceOf(FieldDefinitionError);
  });

  it('AnswerKeyError is instanceof EvaluationError → OMRCheckerError → Error', () => {
    const err = new AnswerKeyError('bad');
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(OMRCheckerError);
    expect(err).toBeInstanceOf(EvaluationError);
    expect(err).toBeInstanceOf(AnswerKeyError);
  });

  it('InvalidConfigValueError is instanceof ConfigError → OMRCheckerError → Error', () => {
    const err = new InvalidConfigValueError('k', 'v', 'r');
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(OMRCheckerError);
    expect(err).toBeInstanceOf(ConfigError);
    expect(err).toBeInstanceOf(InvalidConfigValueError);
  });

  it('errors can be caught by base class', () => {
    function throwAlignment() {
      throw new AlignmentError('/img.png', 'bad warp');
    }
    expect(throwAlignment).toThrowError(OMRCheckerError);
    expect(throwAlignment).toThrowError(ProcessingError);
    expect(throwAlignment).toThrowError(AlignmentError);
  });
});
