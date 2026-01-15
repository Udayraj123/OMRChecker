/**
 * Tests for object utilities
 */

import { describe, it, expect } from 'vitest';
import { isObject, deepMerge, deepClone } from '../object';

describe('Object Utils', () => {
  describe('isObject', () => {
    it('should return true for plain objects', () => {
      expect(isObject({})).toBe(true);
      expect(isObject({ a: 1 })).toBe(true);
      expect(isObject({ nested: { value: 2 } })).toBe(true);
    });

    it('should return false for arrays', () => {
      expect(isObject([])).toBe(false);
      expect(isObject([1, 2, 3])).toBe(false);
    });

    it('should return false for null', () => {
      expect(isObject(null)).toBe(false);
    });

    it('should return false for primitives', () => {
      expect(isObject(undefined)).toBe(false);
      expect(isObject(42)).toBe(false);
      expect(isObject('string')).toBe(false);
      expect(isObject(true)).toBe(false);
    });

    it('should return false for functions', () => {
      expect(isObject(() => {})).toBe(false);
      expect(isObject(function() {})).toBe(false);
    });
  });

  describe('deepMerge', () => {
    it('should merge shallow objects', () => {
      const target = { a: 1, b: 2 };
      const source = { b: 3, c: 4 };
      const result = deepMerge(target, source);

      expect(result).toEqual({ a: 1, b: 3, c: 4 });
    });

    it('should merge nested objects', () => {
      const target = { a: 1, nested: { x: 1, y: 2 } };
      const source = { nested: { y: 3, z: 4 }, b: 2 };
      const result = deepMerge(target, source);

      expect(result).toEqual({
        a: 1,
        nested: { x: 1, y: 3, z: 4 },
        b: 2,
      });
    });

    it('should override arrays (not merge)', () => {
      const target = { items: [1, 2, 3] };
      const source = { items: [4, 5] };
      const result = deepMerge(target, source);

      expect(result.items).toEqual([4, 5]);
      expect(result.items).not.toContain(1);
    });

    it('should handle deep nesting', () => {
      const target = {
        level1: {
          level2: {
            level3: {
              value: 'original',
            },
          },
        },
      };
      const source = {
        level1: {
          level2: {
            level3: {
              newValue: 'added',
            },
          },
        },
      };
      const result = deepMerge(target, source);

      expect(result).toEqual({
        level1: {
          level2: {
            level3: {
              value: 'original',
              newValue: 'added',
            },
          },
        },
      });
    });

    it('should not mutate original objects', () => {
      const target = { a: 1, nested: { x: 1 } };
      const source = { nested: { y: 2 } };
      const result = deepMerge(target, source);

      expect(target.nested).toEqual({ x: 1 });
      expect(result.nested).toEqual({ x: 1, y: 2 });
    });

    it('should handle empty objects', () => {
      expect(deepMerge({}, { a: 1 })).toEqual({ a: 1 });
      expect(deepMerge({ a: 1 }, {})).toEqual({ a: 1 });
      expect(deepMerge({}, {})).toEqual({});
    });

    it('should override non-object values', () => {
      const target = { a: { nested: true } };
      const source = { a: 'string' };
      const result = deepMerge(target, source);

      expect(result.a).toBe('string');
    });

    it('should add new properties at any level', () => {
      const target = { a: { b: 1 } };
      const source = { a: { c: 2 }, d: 3 };
      const result = deepMerge(target, source);

      expect(result).toEqual({ a: { b: 1, c: 2 }, d: 3 });
    });

    it('should handle null values', () => {
      const target = { a: 1 };
      const source = { a: null };
      const result = deepMerge(target, source);

      expect(result.a).toBeNull();
    });

    it('should handle undefined values', () => {
      const target = { a: 1 };
      const source = { a: undefined };
      const result = deepMerge(target, source);

      expect(result.a).toBeUndefined();
    });
  });

  describe('deepClone', () => {
    it('should clone primitives', () => {
      expect(deepClone(42)).toBe(42);
      expect(deepClone('string')).toBe('string');
      expect(deepClone(true)).toBe(true);
      expect(deepClone(null)).toBe(null);
      expect(deepClone(undefined)).toBe(undefined);
    });

    it('should clone shallow objects', () => {
      const original = { a: 1, b: 2 };
      const cloned = deepClone(original);

      expect(cloned).toEqual(original);
      expect(cloned).not.toBe(original);
    });

    it('should clone nested objects', () => {
      const original = { a: 1, nested: { x: 1, y: 2 } };
      const cloned = deepClone(original);

      expect(cloned).toEqual(original);
      expect(cloned).not.toBe(original);
      expect(cloned.nested).not.toBe(original.nested);
    });

    it('should clone arrays', () => {
      const original = [1, 2, 3];
      const cloned = deepClone(original);

      expect(cloned).toEqual(original);
      expect(cloned).not.toBe(original);
    });

    it('should clone arrays with objects', () => {
      const original = [{ a: 1 }, { b: 2 }];
      const cloned = deepClone(original);

      expect(cloned).toEqual(original);
      expect(cloned).not.toBe(original);
      expect(cloned[0]).not.toBe(original[0]);
    });

    it('should clone deeply nested structures', () => {
      const original = {
        level1: {
          level2: {
            level3: {
              array: [1, 2, { nested: true }],
            },
          },
        },
      };
      const cloned = deepClone(original);

      expect(cloned).toEqual(original);
      expect(cloned.level1.level2.level3.array[2]).not.toBe(
        original.level1.level2.level3.array[2]
      );
    });

    it('should not affect original when modifying clone', () => {
      const original = { a: 1, nested: { x: 1 } };
      const cloned = deepClone(original);

      cloned.nested.x = 999;

      expect(original.nested.x).toBe(1);
      expect(cloned.nested.x).toBe(999);
    });
  });
});

