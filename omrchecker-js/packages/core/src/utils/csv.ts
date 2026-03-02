/**
 * Migrated from Python: src/utils/csv.py
 * Format a row of values as a CSV line.
 * Non-numeric strings are quoted. Numeric strings are unquoted (matches QUOTE_NONNUMERIC).
 * Empty array produces empty string.
 */
export function formatCsvRow(dataLine: string[]): string {
  if (dataLine.length === 0) return '';
  return dataLine.map(v => {
    // Numeric-looking values: no quotes (matches pandas QUOTE_NONNUMERIC behavior)
    if (/^-?\d+(\.\d+)?$/.test(v)) return v;
    // Non-numeric: wrap in double quotes, escape internal quotes
    return `"${v.replace(/"/g, '""')}"`;
  }).join(',');
}

/**
 * Append a row to an in-memory CSV accumulator (array of lines).
 * Browser equivalent of thread_safe_csv_append — no file I/O.
 */
export function appendCsvRow(rows: string[], dataLine: string[]): void {
  rows.push(formatCsvRow(dataLine));
}
