import { describe, it, expect, beforeEach, vi } from 'vitest';
import { calculateFileChecksum, printFileChecksum } from '../../src/utils/checksum';

// Note: These tests require browser environment with Web Crypto API
// In Node.js/jsdom, crypto.subtle may not be fully available
// Tests are written to document expected behavior in browser

describe('Checksum Utils', () => {
  // Helper to create ArrayBuffer from string
  function stringToArrayBuffer(str: string): ArrayBuffer {
    const encoder = new TextEncoder();
    return encoder.encode(str).buffer;
  }

  // Helper to create Blob from string
  function stringToBlob(str: string): Blob {
    return new Blob([str], { type: 'text/plain' });
  }

  describe('calculateFileChecksum', () => {
    describe('with ArrayBuffer input', () => {
      it('should calculate SHA-256 checksum by default', async () => {
        const data = stringToArrayBuffer('hello world');
        const checksum = await calculateFileChecksum(data);
        
        // Known SHA-256 hash for "hello world"
        expect(checksum).toBe('b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9');
      });

      it('should calculate SHA-1 checksum', async () => {
        const data = stringToArrayBuffer('hello world');
        const checksum = await calculateFileChecksum(data, 'SHA-1');
        
        // Known SHA-1 hash for "hello world"
        expect(checksum).toBe('2aae6c35c94fcfb415dbe95f408b9ce91ee846ed');
      });

      it('should calculate SHA-384 checksum', async () => {
        const data = stringToArrayBuffer('hello world');
        const checksum = await calculateFileChecksum(data, 'SHA-384');
        
        // Known SHA-384 hash for "hello world"
        expect(checksum).toBe('fdbd8e75a67f29f701a4e040385e2e23986303ea10239211af907fcbb83578b3e417cb71ce646efd0819dd8c088de1bd');
      });

      it('should calculate SHA-512 checksum', async () => {
        const data = stringToArrayBuffer('hello world');
        const checksum = await calculateFileChecksum(data, 'SHA-512');
        
        // Known SHA-512 hash for "hello world"
        expect(checksum).toBe('309ecc489c12d6eb4cc40f50c902f2b4d0ed77ee511a7c7a9bcd3ca86d4cd86f989dd35bc5ff499670da34255b45b0cfd830e81f605dcf7dc5542e93ae9cd76f');
      });

      it('should handle empty data', async () => {
        const data = stringToArrayBuffer('');
        const checksum = await calculateFileChecksum(data);
        
        // Known SHA-256 hash for empty string
        expect(checksum).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
      });

      it('should handle binary data', async () => {
        const data = new Uint8Array([0, 1, 2, 3, 4, 5]).buffer;
        const checksum = await calculateFileChecksum(data);
        
        expect(checksum).toHaveLength(64); // SHA-256 produces 64 hex characters
        expect(checksum).toMatch(/^[0-9a-f]{64}$/);
      });

      it('should produce consistent results for same input', async () => {
        const data = stringToArrayBuffer('test data');
        const checksum1 = await calculateFileChecksum(data);
        const checksum2 = await calculateFileChecksum(data);
        
        expect(checksum1).toBe(checksum2);
      });

      it('should produce different results for different inputs', async () => {
        const data1 = stringToArrayBuffer('hello');
        const data2 = stringToArrayBuffer('world');
        
        const checksum1 = await calculateFileChecksum(data1);
        const checksum2 = await calculateFileChecksum(data2);
        
        expect(checksum1).not.toBe(checksum2);
      });

      it('should be case-sensitive', async () => {
        const data1 = stringToArrayBuffer('Hello');
        const data2 = stringToArrayBuffer('hello');
        
        const checksum1 = await calculateFileChecksum(data1);
        const checksum2 = await calculateFileChecksum(data2);
        
        expect(checksum1).not.toBe(checksum2);
      });

      it('should handle large data', async () => {
        // Create 1MB of data
        const largeData = new Uint8Array(1024 * 1024);
        for (let i = 0; i < largeData.length; i++) {
          largeData[i] = i % 256;
        }
        
        const checksum = await calculateFileChecksum(largeData.buffer);
        
        expect(checksum).toHaveLength(64);
        expect(checksum).toMatch(/^[0-9a-f]{64}$/);
      });
    });

    describe('with Blob input', () => {
      it('should calculate checksum from Blob', async () => {
        const blob = stringToBlob('hello world');
        const checksum = await calculateFileChecksum(blob);
        
        // Same as ArrayBuffer test
        expect(checksum).toBe('b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9');
      });

      it('should handle empty Blob', async () => {
        const blob = stringToBlob('');
        const checksum = await calculateFileChecksum(blob);
        
        expect(checksum).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
      });

      it('should work with different Blob types', async () => {
        const blob1 = new Blob(['test'], { type: 'text/plain' });
        const blob2 = new Blob(['test'], { type: 'application/json' });
        
        const checksum1 = await calculateFileChecksum(blob1);
        const checksum2 = await calculateFileChecksum(blob2);
        
        // Type doesn't affect content, so checksums should be equal
        expect(checksum1).toBe(checksum2);
      });
    });

    describe('error handling', () => {
      it('should throw error for unsupported algorithm', async () => {
        const data = stringToArrayBuffer('test');
        
        await expect(
          calculateFileChecksum(data, 'MD5' as any)
        ).rejects.toThrow('Unsupported hash algorithm');
      });

      it('should include cause in error', async () => {
        const data = stringToArrayBuffer('test');
        
        try {
          await calculateFileChecksum(data, 'INVALID' as any);
        } catch (error: any) {
          expect(error.message).toContain('Unsupported hash algorithm');
          expect(error.cause).toBeDefined();
        }
      });
    });

    describe('hex encoding', () => {
      it('should produce lowercase hex', async () => {
        const data = stringToArrayBuffer('test');
        const checksum = await calculateFileChecksum(data);
        
        expect(checksum).toMatch(/^[0-9a-f]+$/);
        expect(checksum).not.toMatch(/[A-F]/);
      });

      it('should pad hex values with leading zeros', async () => {
        const data = new Uint8Array([0, 1, 15, 16, 255]).buffer;
        const checksum = await calculateFileChecksum(data);
        
        // Checksum should be valid hex with even length
        expect(checksum.length % 2).toBe(0);
      });
    });
  });

  describe('printFileChecksum', () => {
    beforeEach(() => {
      // Clear console.log mock before each test
      vi.restoreAllMocks();
    });

    it('should log checksum to console', async () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      
      const data = stringToArrayBuffer('hello world');
      await printFileChecksum(data);
      
      expect(consoleSpy).toHaveBeenCalledOnce();
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('SHA-256:')
      );
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9')
      );
    });

    it('should use specified algorithm in log', async () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      
      const data = stringToArrayBuffer('test');
      await printFileChecksum(data, 'SHA-512');
      
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('SHA-512:')
      );
    });

    it('should suppress errors silently', async () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      
      const data = stringToArrayBuffer('test');
      
      // Should not throw even with invalid algorithm
      await expect(
        printFileChecksum(data, 'INVALID' as any)
      ).resolves.toBeUndefined();
      
      // Should not log anything on error
      expect(consoleSpy).not.toHaveBeenCalled();
    });

    it('should work with Blob input', async () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      
      const blob = stringToBlob('test data');
      await printFileChecksum(blob);
      
      expect(consoleSpy).toHaveBeenCalledOnce();
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('SHA-256:')
      );
    });
  });
});
