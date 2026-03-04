import { describe, it, expect } from 'vitest';
import { parseFieldString, parseFields } from '../../../src/template/parseFields';
import { OMRCheckerError } from '../../../src/utils/exceptions';

describe('parseFieldString', () => {
  it('returns a single-element array for a literal string', () => {
    expect(parseFieldString('Medium')).toEqual(['Medium']);
    expect(parseFieldString('q5_1')).toEqual(['q5_1']);
    expect(parseFieldString('Roll')).toEqual(['Roll']);
  });

  it('expands a simple numeric range', () => {
    expect(parseFieldString('roll1..9')).toEqual([
      'roll1', 'roll2', 'roll3', 'roll4', 'roll5',
      'roll6', 'roll7', 'roll8', 'roll9',
    ]);
  });

  it('expands a two-digit range', () => {
    expect(parseFieldString('q10..13')).toEqual(['q10', 'q11', 'q12', 'q13']);
  });

  it('expands a range of length 2', () => {
    expect(parseFieldString('q1..2')).toEqual(['q1', 'q2']);
  });

  it('expands range with prefix containing numbers', () => {
    // prefix must be non-digit characters only
    expect(parseFieldString('q1..4')).toEqual(['q1', 'q2', 'q3', 'q4']);
  });

  it('throws for a range where start >= end', () => {
    expect(() => parseFieldString('q5..3')).toThrow(OMRCheckerError);
    expect(() => parseFieldString('q5..5')).toThrow(OMRCheckerError);
  });

  it('throws for invalid range format with dots but bad pattern', () => {
    expect(() => parseFieldString('q..5')).toThrow(OMRCheckerError);
  });
});

describe('parseFields', () => {
  it('parses a list of literal field strings', () => {
    expect(parseFields('test', ['q1', 'q2', 'q3'])).toEqual(['q1', 'q2', 'q3']);
  });

  it('parses a list containing a range', () => {
    expect(parseFields('test', ['roll1..3'])).toEqual(['roll1', 'roll2', 'roll3']);
  });

  it('parses a mix of ranges and literals', () => {
    expect(parseFields('test', ['Medium', 'roll1..3'])).toEqual([
      'Medium', 'roll1', 'roll2', 'roll3',
    ]);
  });

  it('returns empty array for empty input', () => {
    expect(parseFields('test', [])).toEqual([]);
  });

  it('throws for overlapping field strings in the same list', () => {
    expect(() => parseFields('test', ['q1..3', 'q2..4'])).toThrow(OMRCheckerError);
  });

  it('throws for duplicate literal fields', () => {
    expect(() => parseFields('test', ['q1', 'q1'])).toThrow(OMRCheckerError);
  });

  it('parses the Roll block from sample template', () => {
    const result = parseFields('Roll', ['roll1..9']);
    expect(result).toHaveLength(9);
    expect(result[0]).toBe('roll1');
    expect(result[8]).toBe('roll9');
  });

  it('parses multi-label MCQ block', () => {
    const result = parseFields('MCQ_Block_Q1', ['q1..4']);
    expect(result).toEqual(['q1', 'q2', 'q3', 'q4']);
  });
});
