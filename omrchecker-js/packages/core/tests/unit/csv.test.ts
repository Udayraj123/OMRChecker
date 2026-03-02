import { describe, it, expect } from 'vitest';
import { formatCsvRow, appendCsvRow } from '../../src/utils/csv';

describe('CSV Utils', () => {
  it('test_thread_safe_csv_append_basic: formats a mixed row with quoted strings and unquoted numeric', () => {
    // Python: writes ["Alice", "30", "Engineer"], reads back Alice (str), 30 (int), Engineer (str)
    // QUOTE_NONNUMERIC: non-numeric fields are quoted, numeric-looking fields are unquoted
    expect(formatCsvRow(['Alice', '30', 'Engineer'])).toBe('"Alice",30,"Engineer"');

    // appendCsvRow adds exactly one entry to the rows array
    const rows: string[] = [];
    appendCsvRow(rows, ['Alice', '30', 'Engineer']);
    expect(rows.length).toBe(1);
  });

  it('test_thread_safe_csv_append_multiple: appending two rows produces two entries', () => {
    // Python: writes two rows, reads back 2 rows
    const rows: string[] = [];
    appendCsvRow(rows, ['Alice', '30', 'Engineer']);
    appendCsvRow(rows, ['Bob', '25', 'Designer']);
    expect(rows.length).toBe(2);
    expect(rows[0]).toContain('Alice');
    expect(rows[1]).toContain('Bob');
  });

  it('test_thread_safe_csv_append_numeric: all-numeric row has no quotes', () => {
    // Python: writes ["100", "200", "300"], pandas reads back as integers [100, 200, 300]
    // QUOTE_NONNUMERIC leaves numeric-looking values unquoted
    expect(formatCsvRow(['100', '200', '300'])).toBe('100,200,300');
  });

  it('test_thread_safe_csv_append_empty_line: empty array produces empty string and still appends', () => {
    // Python: writes [], file exists afterwards (row was recorded)
    expect(formatCsvRow([])).toBe('');

    const rows: string[] = [];
    appendCsvRow(rows, []);
    expect(rows.length).toBe(1);
    expect(rows[0]).toBe('');
  });
});
