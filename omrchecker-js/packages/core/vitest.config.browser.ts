import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    include: ['tests/browser/**/*.test.ts'],
    exclude: ['tests/unit/**', 'tests/visual/**'],
    testTimeout: 30000, // Browser tests are slower (30s)
    hookTimeout: 60000, // OpenCV.js loading takes time (60s)
    setupFiles: ['./tests/browser-setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'tests/',
        '**/*.test.ts',
        '**/*.spec.ts',
        'dist/',
        'vitest.config.ts',
        'vitest.config.browser.ts',
      ],
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 75,
        statements: 80,
      },
    },
  },
});
