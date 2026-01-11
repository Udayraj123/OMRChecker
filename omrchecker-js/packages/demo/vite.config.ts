import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      '@omrchecker/core': resolve(__dirname, '../core/src'),
    },
  },
  optimizeDeps: {
    exclude: ['@techstark/opencv-js'],
  },
  server: {
    port: 3000,
    open: true,
  },
});

