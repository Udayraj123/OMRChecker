/**
 * CSV utility functions for thread-safe CSV operations
 *
 * TypeScript port of src/utils/csv.py using papaparse
 */

import { promises as fs } from 'fs';
import Papa from 'papaparse';
import { Mutex } from 'async-mutex';

// Thread-safe lock for CSV write operations
const csvWriteLock = new Mutex();

/**
 * Thread-safe function to append a line to a CSV file
 *
 * @param filePath - Path to the CSV file
 * @param dataLine - Array of values to append as a row
 */
export async function threadSafeCsvAppend(
  filePath: string,
  dataLine: (string | number)[]
): Promise<void> {
  await csvWriteLock.runExclusive(async () => {
    // Convert to CSV string
    const csvString = Papa.unparse([dataLine], {
      header: false,
      quotes: true, // QUOTE_NONNUMERIC equivalent
    });

    // Append to file
    await fs.appendFile(filePath, csvString + '\n');
  });
}

