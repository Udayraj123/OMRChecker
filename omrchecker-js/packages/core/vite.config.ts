import { defineConfig } from 'vite';
import dts from 'vite-plugin-dts';
import { resolve } from 'path';

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'OMRChecker',
      formats: ['es', 'umd'],
      fileName: (format) => `omrchecker.${format}.js`,
    },
    rollupOptions: {
      // Externalize dependencies that shouldn't be bundled
      external: [],
      output: {
        globals: {},
      },
    },
    sourcemap: true,
    target: 'es2020',
  },
  plugins: [
    dts({
      include: ['src'],
      exclude: ['**/*.test.ts'],
    }),
  ],
});

