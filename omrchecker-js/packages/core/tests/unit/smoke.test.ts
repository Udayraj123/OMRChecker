import { describe, it, expect } from 'vitest';

describe('Smoke Test', () => {
  it('should pass a basic assertion', () => {
    expect(1 + 1).toBe(2);
  });

  it('should have test globals available', () => {
    expect(describe).toBeDefined();
    expect(it).toBeDefined();
    expect(expect).toBeDefined();
  });

  it('should be able to test async functions', async () => {
    const result = await Promise.resolve(42);
    expect(result).toBe(42);
  });
});
