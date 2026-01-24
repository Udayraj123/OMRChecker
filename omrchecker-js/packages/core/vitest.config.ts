import { coverageConfigDefaults, defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['src/__tests__/setup.ts'],
    include: ['src/**/__tests__/**/*.test.ts'],
    disableConsoleIntercept: true,
    silent: false,
    testTimeout: 10000, // 30 seconds per test
    hookTimeout: 10000, // 10 seconds for setup/teardown hooks
    teardownTimeout: 5000, // 5 seconds for teardown
    // pool: 'forks', // https://vitest.dev/guide/common-errors.html#segfaults-and-native-code-errors
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*.ts'],
      exclude: ['src/**/__tests__/**', 'src/**/*.test.ts', ...coverageConfigDefaults.exclude],
    },
  },
});

