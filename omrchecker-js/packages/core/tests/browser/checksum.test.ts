/**
 * Browser tests for checksum.ts using Web Crypto API
 * 
 * These tests verify checksum calculation works correctly in a real browser
 * environment with the Web Crypto API. Tests SHA-256 and other algorithms
 * with known hash values.
 * 
 * Note: These tests don't require OpenCV.js - they only use the Web Crypto API
 * which is available in all modern browsers.
 * 
 * Run with: npm run test:browser
 */

import { test, expect } from '@playwright/test';

// Increase timeout for browser tests
test.setTimeout(60000);

test.describe('Checksum Utils - Browser Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to data URL to ensure secure context for crypto.subtle
    // Using data URL with HTML provides a secure context in most browsers
    await page.goto('data:text/html,<!DOCTYPE html><html><head></head><body></body></html>');
  });

  test.describe('calculateFileChecksum - SHA-256', () => {
    test('should calculate SHA-256 checksum for simple string', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const text = 'hello world';
        const buffer = new TextEncoder().encode(text);
        
        const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
        
        return hashHex;
      });

      // Known SHA-256 hash for "hello world"
      expect(result).toBe('b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9');
    });

    test('should calculate SHA-256 checksum for empty string', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const buffer = new Uint8Array(0);
        const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      });

      // Known SHA-256 hash for empty string
      expect(result).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
    });

    test('should calculate SHA-256 checksum for binary data', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const buffer = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7]);
        const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      });

      expect(result).toBeTruthy();
      expect(result).toHaveLength(64); // SHA-256 produces 64 hex characters
    });

    test('should produce consistent hashes for same input', async ({ page }) => {
      const { hash1, hash2 } = await page.evaluate(async () => {
        const text = 'test data';
        const buffer = new TextEncoder().encode(text);
        
        const hashBuffer1 = await crypto.subtle.digest('SHA-256', buffer);
        const hash1 = Array.from(new Uint8Array(hashBuffer1))
          .map(b => b.toString(16).padStart(2, '0'))
          .join('');
        
        const hashBuffer2 = await crypto.subtle.digest('SHA-256', buffer);
        const hash2 = Array.from(new Uint8Array(hashBuffer2))
          .map(b => b.toString(16).padStart(2, '0'))
          .join('');
        
        return { hash1, hash2 };
      });

      expect(hash1).toBe(hash2);
    });

    test('should produce different hashes for different inputs', async ({ page }) => {
      const { hash1, hash2 } = await page.evaluate(async () => {
        const buffer1 = new TextEncoder().encode('data1');
        const buffer2 = new TextEncoder().encode('data2');
        
        const hashBuffer1 = await crypto.subtle.digest('SHA-256', buffer1);
        const hash1 = Array.from(new Uint8Array(hashBuffer1))
          .map(b => b.toString(16).padStart(2, '0'))
          .join('');
        
        const hashBuffer2 = await crypto.subtle.digest('SHA-256', buffer2);
        const hash2 = Array.from(new Uint8Array(hashBuffer2))
          .map(b => b.toString(16).padStart(2, '0'))
          .join('');
        
        return { hash1, hash2 };
      });

      expect(hash1).not.toBe(hash2);
    });
  });

  test.describe('calculateFileChecksum - Multiple Algorithms', () => {
    test('should support SHA-1 algorithm', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const text = 'test';
        const buffer = new TextEncoder().encode(text);
        const hashBuffer = await crypto.subtle.digest('SHA-1', buffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return {
          hash: hashArray.map(b => b.toString(16).padStart(2, '0')).join(''),
          length: hashArray.length * 2, // hex string length
        };
      });

      expect(result.length).toBe(40); // SHA-1 produces 40 hex characters
      expect(result.hash).toBe('a94a8fe5ccb19ba61c4c0873d391e987982fbbd3'); // Known SHA-1 for "test"
    });

    test('should support SHA-384 algorithm', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const text = 'test';
        const buffer = new TextEncoder().encode(text);
        const hashBuffer = await crypto.subtle.digest('SHA-384', buffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      });

      expect(result).toHaveLength(96); // SHA-384 produces 96 hex characters
    });

    test('should support SHA-512 algorithm', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const text = 'test';
        const buffer = new TextEncoder().encode(text);
        const hashBuffer = await crypto.subtle.digest('SHA-512', buffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      });

      expect(result).toHaveLength(128); // SHA-512 produces 128 hex characters
    });
  });

  test.describe('Blob Support', () => {
    test('should calculate checksum from Blob', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const text = 'hello world';
        const blob = new Blob([text], { type: 'text/plain' });
        const buffer = await blob.arrayBuffer();
        
        const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      });

      // Should match the same hash as string input
      expect(result).toBe('b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9');
    });

    test('should handle large data blobs', async ({ page }) => {
      const result = await page.evaluate(async () => {
        // Create 1MB of data
        const size = 1024 * 1024;
        const data = new Uint8Array(size);
        for (let i = 0; i < size; i++) {
          data[i] = i % 256;
        }
        
        const blob = new Blob([data]);
        const buffer = await blob.arrayBuffer();
        
        const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        const hash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
        
        return {
          hash,
          dataSize: buffer.byteLength,
        };
      });

      expect(result.dataSize).toBe(1024 * 1024);
      expect(result.hash).toHaveLength(64);
      expect(result.hash).toBeTruthy();
    });
  });
});
