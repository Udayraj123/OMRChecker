/**
 * Translated from Python: src/tests/utils/__tests__/test_file_pattern_resolver.py
 * 14 tests covering FilePatternResolver pattern formatting, path sanitization,
 * extension preservation, base directory, collision strategies, and batch resolution.
 */
import { describe, it, expect } from 'vitest';
import { FilePatternResolver } from '../../src/utils/file_pattern_resolver';

describe('FilePatternResolver', () => {
  // test_simple_pattern_formatting
  it('test_simple_pattern_formatting: substitutes all fields into pattern', () => {
    const resolver = new FilePatternResolver();
    const fields = { roll: '12345', score: '95' };
    const path = resolver.resolvePattern('roll_{roll}_score_{score}.jpg', fields, { collisionStrategy: 'overwrite' });
    expect(path).not.toBeNull();
    expect(path!.split('/').pop()).toBe('roll_12345_score_95.jpg');
  });

  // test_pattern_with_folders
  it('test_pattern_with_folders: resolves pattern with subdirectory segments', () => {
    const resolver = new FilePatternResolver();
    const fields = { booklet: 'A', batch: 'morning', roll: '12345' };
    const path = resolver.resolvePattern(
      'booklet_{booklet}/batch_{batch}/{roll}',
      fields,
      { originalPath: 'test.jpg', collisionStrategy: 'overwrite' }
    );
    expect(path).not.toBeNull();
    expect(path).toBe('booklet_A/batch_morning/12345.jpg');
  });

  // test_extension_preservation
  it('test_extension_preservation: inherits extension from originalPath when pattern has none', () => {
    const resolver = new FilePatternResolver();
    const fields = { roll: '12345' };
    const path = resolver.resolvePattern('student_{roll}', fields, {
      originalPath: 'original.png',
      collisionStrategy: 'overwrite',
    });
    expect(path).not.toBeNull();
    expect(path!.endsWith('.png')).toBe(true);
    expect(path!.split('/').pop()).toBe('student_12345.png');
  });

  // test_extension_in_pattern_overrides
  it('test_extension_in_pattern_overrides: pattern extension takes precedence over originalPath extension', () => {
    const resolver = new FilePatternResolver();
    const fields = { roll: '12345' };
    const path = resolver.resolvePattern('student_{roll}.jpg', fields, {
      originalPath: 'original.png',
      collisionStrategy: 'overwrite',
    });
    expect(path).not.toBeNull();
    expect(path!.endsWith('.jpg')).toBe(true);
    expect(path!.split('/').pop()).toBe('student_12345.jpg');
  });

  // test_base_directory
  it('test_base_directory: resolved path starts with baseDir', () => {
    const baseDir = '/tmp/organized';
    const resolver = new FilePatternResolver({ baseDir });
    const fields = { roll: '12345' };
    const path = resolver.resolvePattern('student_{roll}', fields, { collisionStrategy: 'overwrite' });
    expect(path).not.toBeNull();
    expect(path!.startsWith(baseDir)).toBe(true);
  });

  // test_path_sanitization
  it('test_path_sanitization: removes forbidden characters from field values', () => {
    const resolver = new FilePatternResolver();
    const fields = { name: 'John<>Doe', code: 'A|B*C' };
    const path = resolver.resolvePattern('{name}/{code}', fields, { collisionStrategy: 'overwrite' });
    expect(path).not.toBeNull();
    expect(path).not.toContain('<');
    expect(path).not.toContain('>');
    expect(path).not.toContain('|');
    expect(path).not.toContain('*');
  });

  // test_missing_field_returns_none
  it('test_missing_field_returns_none: returns null when pattern references an undefined field', () => {
    const resolver = new FilePatternResolver();
    const fields = { roll: '12345' };
    const path = resolver.resolvePattern('student_{name}', fields, { collisionStrategy: 'overwrite' });
    expect(path).toBeNull();
  });

  // test_collision_skip_strategy
  it('test_collision_skip_strategy: returns null when file exists and strategy is skip', () => {
    const existingFiles = new Set(['/tmp/organized/student_12345.jpg']);
    const resolver = new FilePatternResolver({
      baseDir: '/tmp/organized',
      existsCheck: (p) => existingFiles.has(p),
    });
    const path = resolver.resolvePattern('student_{roll}', { roll: '12345' }, {
      originalPath: 'test.jpg',
      collisionStrategy: 'skip',
    });
    expect(path).toBeNull();
  });

  // test_collision_increment_strategy
  it('test_collision_increment_strategy: returns incremented filename when base and _001 already exist', () => {
    const existingFiles = new Set([
      '/tmp/organized/student_12345.jpg',
      '/tmp/organized/student_12345_001.jpg',
    ]);
    const resolver = new FilePatternResolver({
      baseDir: '/tmp/organized',
      existsCheck: (p) => existingFiles.has(p),
    });
    const path = resolver.resolvePattern('student_{roll}', { roll: '12345' }, {
      originalPath: 'test.jpg',
      collisionStrategy: 'increment',
    });
    expect(path).not.toBeNull();
    expect(path!.split('/').pop()).toBe('student_12345_002.jpg');
  });

  // test_collision_overwrite_strategy
  it('test_collision_overwrite_strategy: returns same path even when file exists and strategy is overwrite', () => {
    const existingFiles = new Set(['/tmp/organized/student_12345.jpg']);
    const resolver = new FilePatternResolver({
      baseDir: '/tmp/organized',
      existsCheck: (p) => existingFiles.has(p),
    });
    const path = resolver.resolvePattern('student_{roll}', { roll: '12345' }, {
      originalPath: 'test.jpg',
      collisionStrategy: 'overwrite',
    });
    expect(path).not.toBeNull();
    expect(path!.split('/').pop()).toBe('student_12345.jpg');
  });

  // test_complex_pattern_with_multiple_fields
  it('test_complex_pattern_with_multiple_fields: resolves nested path with many substitution fields', () => {
    const resolver = new FilePatternResolver();
    const fields = {
      region: 'north',
      school: 'ABC',
      class: '10A',
      roll: '12345',
      name: 'John_Doe',
      score: '95',
    };
    const path = resolver.resolvePattern(
      'region_{region}/school_{school}/class_{class}/{name}_roll_{roll}_score_{score}',
      fields,
      { originalPath: 'test.jpg', collisionStrategy: 'overwrite' }
    );
    expect(path).not.toBeNull();
    expect(path).toBe('region_north/school_ABC/class_10A/John_Doe_roll_12345_score_95.jpg');
  });

  // test_empty_fields
  it('test_empty_fields: handles empty string field values without crashing', () => {
    const resolver = new FilePatternResolver();
    const fields = { roll: '', name: '' };
    const path = resolver.resolvePattern('{roll}/{name}', fields, { collisionStrategy: 'overwrite' });
    // Python asserts path is not None — empty sanitized parts are skipped but the call succeeds.
    expect(path).not.toBeNull();
  });

  // test_special_characters_in_field_values
  it('test_special_characters_in_field_values: sanitizes slash and backslash in field values', () => {
    const resolver = new FilePatternResolver();
    const fields = { name: 'John/Doe\\Test', code: 'A:B' };
    const path = resolver.resolvePattern('{name}_{code}', fields, { collisionStrategy: 'overwrite' });
    expect(path).not.toBeNull();
    // The final filename segment must not contain backslash
    const name = path!.split('/').pop()!;
    expect(name).not.toContain('\\');
  });

  // test_resolve_batch
  it('test_resolve_batch: resolves multiple patterns in one call', () => {
    const resolver = new FilePatternResolver();
    const patternsAndFields: Array<[string, Record<string, string>, string]> = [
      ['student_{roll}', { roll: '001' }, 'test1.jpg'],
      ['student_{roll}', { roll: '002' }, 'test2.jpg'],
      ['student_{roll}', { roll: '003' }, 'test3.jpg'],
    ];
    const results = resolver.resolveBatch(patternsAndFields, 'overwrite');
    expect(results.length).toBe(3);
    expect(results.every(([p]) => p !== null)).toBe(true);
    expect(results[0][0]!.split('/').pop()).toBe('student_001.jpg');
    expect(results[1][0]!.split('/').pop()).toBe('student_002.jpg');
    expect(results[2][0]!.split('/').pop()).toBe('student_003.jpg');
  });
});
