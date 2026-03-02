/**
 * Translated from Python: src/tests/utils/__tests__/test_serialization.py
 * Tests for deepSerialize utility.
 */

import { describe, it, expect } from 'vitest';
import { deepSerialize } from '../../src/utils/serialization';

describe('deepSerialize', () => {
  it('test_simple_dataclass_serialization — plain object with value and name', () => {
    const config = { value: 42, name: 'hello' };
    const result = deepSerialize(config);

    expect(result).toEqual({ value: 42, name: 'hello' });
    expect(typeof result).toBe('object');
  });

  it('test_nested_dataclass_serialization — nested objects with path as string', () => {
    const config = {
      path: '/tmp/test.txt',
      nested: { value: 100, name: 'nested' },
      numbers: [],
      mapping: {},
      color: 'red',
      flag: true,
    };
    const result = deepSerialize(config) as Record<string, unknown>;

    expect(result['path']).toBe('/tmp/test.txt');
    expect(result['nested']).toEqual({ value: 100, name: 'nested' });
    expect(result['flag']).toBe(true);
  });

  it('test_path_serialization — string path value passes through as string', () => {
    const config = {
      path: '/home/user/file.json',
      nested: { value: 10, name: 'test' },
      numbers: [],
      mapping: {},
      color: 'red',
      flag: true,
    };
    const result = deepSerialize(config) as Record<string, unknown>;

    expect(result['path']).toBe('/home/user/file.json');
    expect(typeof result['path']).toBe('string');
  });

  it('test_enum_serialization — string color value passes through as string', () => {
    const config = {
      path: '/tmp',
      nested: { value: 10, name: 'test' },
      numbers: [],
      mapping: {},
      color: 'blue',
      flag: true,
    };
    const result = deepSerialize(config) as Record<string, unknown>;

    expect(result['color']).toBe('blue');
  });

  it('test_list_and_dict_serialization — arrays and nested objects serialize correctly', () => {
    const config = {
      path: '/tmp',
      nested: { value: 10, name: 'test' },
      numbers: [1, 2, 3, 4, 5],
      mapping: { key1: 'value1', key2: 'value2' },
      color: 'red',
      flag: true,
    };
    const result = deepSerialize(config) as Record<string, unknown>;

    expect(result['numbers']).toEqual([1, 2, 3, 4, 5]);
    expect(result['mapping']).toEqual({ key1: 'value1', key2: 'value2' });
  });

  it('test_nested_list_of_dataclasses — array of objects serializes correctly', () => {
    const container = {
      items: [
        { value: 1, name: 'first' },
        { value: 2, name: 'second' },
      ],
    };
    const result = deepSerialize(container) as Record<string, unknown>;

    expect(result['items']).toEqual([
      { value: 1, name: 'first' },
      { value: 2, name: 'second' },
    ]);
  });

  it('test_primitive_types — primitives pass through unchanged', () => {
    expect(deepSerialize('hello')).toBe('hello');
    expect(deepSerialize(42)).toBe(42);
    expect(deepSerialize(3.14)).toBe(3.14);
    expect(deepSerialize(true)).toBe(true);
    expect(deepSerialize(null)).toBe(null);
  });

  it('test_dict_with_nested_dataclasses — object with nested objects as values', () => {
    const wrapper = {
      configs: {
        config1: { value: 10, name: 'first' },
        config2: { value: 20, name: 'second' },
      },
    };
    const result = deepSerialize(wrapper) as Record<string, unknown>;
    const configs = result['configs'] as Record<string, unknown>;

    expect(configs['config1']).toEqual({ value: 10, name: 'first' });
    expect(configs['config2']).toEqual({ value: 20, name: 'second' });
  });
});
