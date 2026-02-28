/**
 * AUTO-GENERATED from Python source: src/utils/checksum.py
 * Migrated as part of Phase 0 test - Subagent delegation validation
 *
 * File checksum calculation utilities using Web Crypto API.
 * Browser-compatible implementation of Python's hashlib functionality.
 */

/**
 * Calculate the checksum of a file using the specified hashing algorithm.
 *
 * Note: Browser environment uses Web Crypto API instead of Python's hashlib.
 * Only supports SHA-1, SHA-256, SHA-384, and SHA-512 (not MD5 for security).
 *
 * @param fileData - File data as ArrayBuffer or Blob
 * @param algorithm - Hash algorithm to use (SHA-1, SHA-256, SHA-384, SHA-512)
 * @returns Promise resolving to hexadecimal string representation of the file's checksum
 * @throws {Error} If the algorithm is not supported by Web Crypto API
 */
export async function calculateFileChecksum(
  fileData: ArrayBuffer | Blob,
  algorithm: 'SHA-1' | 'SHA-256' | 'SHA-384' | 'SHA-512' = 'SHA-256'
): Promise<string> {
  // Convert Blob to ArrayBuffer if needed
  const buffer = fileData instanceof Blob
    ? await fileData.arrayBuffer()
    : fileData;

  try {
    // Use Web Crypto API for hashing
    const hashBuffer = await crypto.subtle.digest(algorithm, buffer);
    
    // Convert ArrayBuffer to hex string
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');
    
    return hashHex;
  } catch (error) {
    const message = `Unsupported hash algorithm: ${algorithm}`;
    throw new Error(message, { cause: error });
  }
}

/**
 * Calculate and print the checksum of a file.
 *
 * Note: In browser environment, this logs to console instead of printing.
 * Silently handles errors to match Python's contextlib.suppress behavior.
 *
 * @param fileData - File data as ArrayBuffer or Blob
 * @param algorithm - Hash algorithm to use (SHA-1, SHA-256, SHA-384, SHA-512)
 */
export async function printFileChecksum(
  fileData: ArrayBuffer | Blob,
  algorithm: 'SHA-1' | 'SHA-256' | 'SHA-384' | 'SHA-512' = 'SHA-256'
): Promise<void> {
  try {
    const checksum = await calculateFileChecksum(fileData, algorithm);
    console.log(`${algorithm}: ${checksum}`);
  } catch (error) {
    // Suppress errors to match Python's contextlib.suppress(FileNotFoundError, ValueError)
    // In browser context, there's no FileNotFoundError, but we catch all errors
  }
}
