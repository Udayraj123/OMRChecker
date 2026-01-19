/**
 * Comprehensive tests for custom exception hierarchy.
 *
 * Tests all custom exceptions to ensure they:
 * 1. Store the correct context information
 * 2. Generate appropriate error messages
 * 3. Maintain the proper exception hierarchy
 * 4. Can be caught at different levels of specificity
 *
 * Ported from Python test_exceptions.py
 */

import { describe, it, expect } from 'vitest';
import {
  OMRCheckerError,
  InputError,
  InputFileNotFoundError,
  TemplateError,
  FieldDefinitionError,
  ConfigError,
  ConfigLoadError,
  ImageReadError,
  ImageProcessingError,
  TemplateValidationError,
} from '../exceptions';

describe('Exception Tests', () => {
  describe('Base Exception (OMRCheckerError)', () => {
    it('should create exception with message only', () => {
      const exc = new OMRCheckerError('Test error message');

      expect(exc.message).toBe('Test error message');
      expect(exc.context).toEqual({});
      expect(exc.toString()).toBe('Test error message');
    });

    it('should create exception with context information', () => {
      const context = { file: 'test.jpg', line: 42 };
      const exc = new OMRCheckerError('Test error', context);

      expect(exc.message).toBe('Test error');
      expect(exc.context).toEqual(context);
      const str = exc.toString();
      expect(str).toContain('file=test.jpg');
      expect(str).toContain('line=42');
    });

    it('should be catchable as Error', () => {
      expect(() => {
        throw new OMRCheckerError('Test');
      }).toThrow(Error);
    });

    it('should preserve context through toString', () => {
      const context = { operation: 'resize', dimensions: '800x600' };
      const exc = new OMRCheckerError('Processing failed', context);
      const msg = exc.toString();

      expect(msg).toContain('operation=resize');
      expect(msg).toContain('dimensions=800x600');
    });
  });

  describe('Input Exceptions', () => {
    it('should create InputFileNotFoundError without file type', () => {
      const path = '/path/to/missing.json';
      const exc = new InputFileNotFoundError(path);

      expect(exc).toBeInstanceOf(InputError);
      expect(exc).toBeInstanceOf(OMRCheckerError);
      expect(exc.path).toBe(path);
      expect(exc.fileType).toBeUndefined();
      expect(exc.message).toContain('missing.json');
      expect(exc.context.path).toBe(path);
    });

    it.each([
      ['template', true],
      ['config', true],
      ['evaluation', true],
    ])('should create InputFileNotFoundError with file type: %s', (fileType, expectedInStr) => {
      const path = '/path/to/missing.json';
      const exc = new InputFileNotFoundError(path, fileType);

      expect(exc).toBeInstanceOf(InputError);
      expect(exc.path).toBe(path);
      expect(exc.fileType).toBe(fileType);
      expect(exc.message).toContain(fileType);
      expect(exc.context.fileType).toBe(fileType);
    });

    it('should create ImageReadError without reason', () => {
      const path = '/path/to/corrupt.jpg';
      const exc = new ImageReadError(path);

      expect(exc).toBeInstanceOf(InputError);
      expect(exc.path).toBe(path);
      expect(exc.reason).toBeUndefined();
      expect(exc.message).toContain('corrupt.jpg');
    });

    it('should create ImageReadError with reason', () => {
      const path = '/path/to/corrupt.jpg';
      const reason = 'File is corrupted';
      const exc = new ImageReadError(path, reason);

      expect(exc).toBeInstanceOf(InputError);
      expect(exc.path).toBe(path);
      expect(exc.reason).toBe(reason);
      expect(exc.message).toContain(reason);
      expect(exc.context.reason).toBe(reason);
    });
  });

  describe('Template Exceptions', () => {
    it('should create FieldDefinitionError', () => {
      const message = 'Invalid bubble coordinates';
      const context = { field_id: 'q1_bubbles', template_path: '/templates/template.json' };
      const exc = new FieldDefinitionError(message, context);

      expect(exc).toBeInstanceOf(TemplateError);
      expect(exc).toBeInstanceOf(OMRCheckerError);
      expect(exc.message).toBe(message);
      expect(exc.context).toEqual(context);
      expect(exc.toString()).toContain('field_id=q1_bubbles');
    });

    it('should create TemplateValidationError', () => {
      const templatePath = '/path/to/template.json';
      const message = 'Invalid template JSON';
      const context = { errors: ['Missing field dimensions'] };
      const exc = new TemplateValidationError(templatePath, message, context);

      expect(exc).toBeInstanceOf(OMRCheckerError);
      expect(exc.templatePath).toBe(templatePath);
      expect(exc.message).toBe(message);
      expect(exc.context.templatePath).toBe(templatePath);
      expect(exc.context.errors).toEqual(['Missing field dimensions']);
    });
  });

  describe('Config Exceptions', () => {
    it('should create ConfigError', () => {
      const message = 'Invalid configuration';
      const context = { key: 'max_workers', value: -1 };
      const exc = new ConfigError(message, context);

      expect(exc).toBeInstanceOf(OMRCheckerError);
      expect(exc.message).toBe(message);
      expect(exc.context).toEqual(context);
    });

    it('should create ConfigLoadError', () => {
      const path = '/config/settings.json';
      const reason = 'Syntax error on line 5';
      const exc = new ConfigLoadError(path, reason);

      expect(exc).toBeInstanceOf(OMRCheckerError);
      expect(exc.path).toBe(path);
      expect(exc.message).toContain(path);
      expect(exc.message).toContain(reason);
      expect(exc.context.path).toBe(path);
      expect(exc.context.reason).toBe(reason);
    });
  });

  describe('Processing Exceptions', () => {
    it('should create ImageProcessingError', () => {
      const message = 'Image processing failed';
      const context = { operation: 'rotation', file_path: '/images/test.jpg' };
      const exc = new ImageProcessingError(message, context);

      expect(exc).toBeInstanceOf(OMRCheckerError);
      expect(exc.message).toBe(message);
      expect(exc.context).toEqual(context);
    });

    it('should create ImageProcessingError with minimal context', () => {
      const message = 'Image processing failed';
      const exc = new ImageProcessingError(message);

      expect(exc).toBeInstanceOf(OMRCheckerError);
      expect(exc.message).toBe(message);
      expect(exc.context).toEqual({});
    });
  });

  describe('Exception Hierarchy', () => {
    it.each([
      [InputFileNotFoundError, InputError, ['/missing']],
      [ImageReadError, InputError, ['/corrupt.jpg', 'Corrupted']],
      [FieldDefinitionError, TemplateError, ['Invalid field', { field_id: 'q1' }]],
      [ConfigError, OMRCheckerError, ['Invalid config']],
      [ImageProcessingError, OMRCheckerError, ['Processing failed']],
    ])(
      'should catch %s as %s',
      (ExceptionClass, CategoryClass, args) => {
        expect(() => {
          throw new (ExceptionClass as any)(...args);
        }).toThrow(CategoryClass);
      }
    );

    it('should catch all exceptions as OMRCheckerError', () => {
      const exceptions = [
        new InputFileNotFoundError('/test'),
        new ImageReadError('/test.jpg', 'Error'),
        new FieldDefinitionError('Test error', { field_id: 'q1' }),
        new ConfigError('Test error'),
        new ImageProcessingError('Test error'),
        new TemplateValidationError('/template.json', 'Test error'),
      ];

      exceptions.forEach((exc) => {
        expect(() => {
          throw exc;
        }).toThrow(OMRCheckerError);
      });
    });

    it('should preserve context through exception hierarchy', () => {
      const path = '/test/file.jpg';
      const reason = 'Test reason';
      const context = { operation: 'rotation', file_path: path, reason };
      const exc = new ImageProcessingError('Processing failed', context);

      // Context should be accessible
      expect(exc.context.operation).toBe('rotation');
      expect(exc.context.file_path).toBe(path);
      expect(exc.context.reason).toBe(reason);

      // Should still be catchable as base exception
      try {
        throw exc;
      } catch (caught) {
        expect(caught).toBeInstanceOf(OMRCheckerError);
        if (caught instanceof OMRCheckerError) {
          expect(caught.context.operation).toBe('rotation');
        }
      }
    });
  });

  describe('Exception Messages', () => {
    it('should contain relevant information in messages', () => {
      // Test with path and file type
      const path = '/path/to/file.jpg';
      const exc = new InputFileNotFoundError(path, 'template');
      const msg = exc.toString();

      expect(msg).toContain('file.jpg');
      expect(msg).toContain('template');
    });

    it('should work correctly without optional fields', () => {
      const exc = new ImageProcessingError('Processing failed');

      expect(exc.message).toBe('Processing failed');
      expect(exc.toString()).toBe('Processing failed');
      // Should not crash even without context
    });

    it('should include context in string representation', () => {
      const context = { operation: 'resize', dimensions: '800x600' };
      const exc = new OMRCheckerError('Processing failed', context);
      const msg = exc.toString();

      expect(msg).toContain('operation=resize');
      expect(msg).toContain('dimensions=800x600');
    });

    it('should format multiple context fields correctly', () => {
      const context = {
        file: 'test.jpg',
        line: 42,
        operation: 'crop',
      };
      const exc = new OMRCheckerError('Error occurred', context);
      const msg = exc.toString();

      expect(msg).toContain('file=test.jpg');
      expect(msg).toContain('line=42');
      expect(msg).toContain('operation=crop');
    });
  });

  describe('Exception Properties', () => {
    it('should set name property correctly', () => {
      const exc = new OMRCheckerError('Test');
      expect(exc.name).toBe('OMRCheckerError');

      const inputExc = new InputError('Test');
      expect(inputExc.name).toBe('InputError');

      const fileExc = new InputFileNotFoundError('/test');
      expect(fileExc.name).toBe('InputFileNotFoundError');
    });

    it('should maintain instanceof relationships', () => {
      const inputExc = new InputFileNotFoundError('/test');
      expect(inputExc).toBeInstanceOf(InputFileNotFoundError);
      expect(inputExc).toBeInstanceOf(InputError);
      expect(inputExc).toBeInstanceOf(OMRCheckerError);
      expect(inputExc).toBeInstanceOf(Error);

      const templateExc = new FieldDefinitionError('Test', {});
      expect(templateExc).toBeInstanceOf(FieldDefinitionError);
      expect(templateExc).toBeInstanceOf(TemplateError);
      expect(templateExc).toBeInstanceOf(OMRCheckerError);
      expect(templateExc).toBeInstanceOf(Error);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty context', () => {
      const exc = new OMRCheckerError('Test', {});
      expect(exc.context).toEqual({});
      expect(exc.toString()).toBe('Test');
    });

    it('should handle undefined context', () => {
      const exc = new OMRCheckerError('Test');
      expect(exc.context).toEqual({});
    });

    it('should handle null values in context', () => {
      const context = { value: null, other: 'test' };
      const exc = new OMRCheckerError('Test', context);
      expect(exc.context.value).toBeNull();
      expect(exc.context.other).toBe('test');
    });

    it('should handle complex context values', () => {
      const context = {
        array: [1, 2, 3],
        object: { nested: 'value' },
        number: 42,
        boolean: true,
      };
      const exc = new OMRCheckerError('Test', context);

      expect(exc.context.array).toEqual([1, 2, 3]);
      expect(exc.context.object).toEqual({ nested: 'value' });
      expect(exc.context.number).toBe(42);
      expect(exc.context.boolean).toBe(true);
    });
  });
});

