#!/bin/bash

# OMRChecker Migration Toolkit - Project Setup
# Creates omrchecker-js project structure with all necessary configuration

set -e

echo "🚀 OMRChecker Migration Toolkit - Project Setup"
echo "================================================"
echo ""

# Configuration
TARGET_DIR="../../omrchecker-js"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if target directory exists
if [ -d "$TARGET_DIR" ]; then
    echo "⚠️  Target directory $TARGET_DIR already exists"
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$TARGET_DIR"
        echo "✓ Deleted existing directory"
    else
        echo "❌ Aborted"
        exit 1
    fi
fi

echo "📁 Creating project structure..."

# Create directory structure
mkdir -p "$TARGET_DIR"/{src,tests,public}
mkdir -p "$TARGET_DIR"/src/{core,processors,utils,schemas,exceptions,workers,types}
mkdir -p "$TARGET_DIR"/src/processors/{alignment,detection,evaluation,preprocessing,threshold}
mkdir -p "$TARGET_DIR"/src/utils/{logger,validation,parsing}
mkdir -p "$TARGET_DIR"/tests/{core,processors,utils,schemas,fixtures}

echo "✓ Directory structure created"

# Generate package.json
echo "📝 Generating package.json..."
cat > "$TARGET_DIR/package.json" << 'EOF'
{
  "name": "omrchecker-js",
  "version": "1.0.0",
  "description": "OMR (Optical Mark Recognition) Checker for Browser - JavaScript port",
  "type": "module",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:ui": "vitest --ui",
    "test:coverage": "vitest --coverage",
    "lint": "eslint src --ext ts,tsx",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "opencv.js": "^1.2.1",
    "zod": "^3.23.8",
    "@zxing/library": "^0.21.3",
    "tesseract.js": "^5.1.1",
    "@tensorflow/tfjs": "^4.22.0",
    "lodash-es": "^4.17.21"
  },
  "devDependencies": {
    "@types/lodash-es": "^4.17.12",
    "@types/node": "^22.10.2",
    "@typescript-eslint/eslint-plugin": "^8.18.2",
    "@typescript-eslint/parser": "^8.18.2",
    "@vitejs/plugin-react": "^4.3.4",
    "@vitest/coverage-v8": "^2.1.8",
    "@vitest/ui": "^2.1.8",
    "eslint": "^9.17.0",
    "pixelmatch": "^6.0.0",
    "typescript": "^5.7.2",
    "vite": "^6.0.5",
    "vite-plugin-wasm": "^3.3.0",
    "vitest": "^2.1.8"
  },
  "keywords": [
    "omr",
    "optical-mark-recognition",
    "browser",
    "image-processing",
    "opencv",
    "education",
    "grading"
  ],
  "author": "OMRChecker Contributors",
  "license": "GPL-3.0",
  "repository": {
    "type": "git",
    "url": "https://github.com/Udayraj123/OMRChecker.git",
    "directory": "omrchecker-js"
  }
}
EOF

echo "✓ package.json created"

# Generate tsconfig.json
echo "📝 Generating tsconfig.json..."
cat > "$TARGET_DIR/tsconfig.json" << 'EOF'
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",

    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,

    /* Path aliases */
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@core/*": ["src/core/*"],
      "@processors/*": ["src/processors/*"],
      "@utils/*": ["src/utils/*"],
      "@types/*": ["src/types/*"],
      "@schemas/*": ["src/schemas/*"]
    },

    /* Output */
    "outDir": "./dist",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*", "tests/**/*"],
  "exclude": ["node_modules", "dist"]
}
EOF

echo "✓ tsconfig.json created"

# Generate vite.config.ts
echo "📝 Generating vite.config.ts..."
cat > "$TARGET_DIR/vite.config.ts" << 'EOF'
import { defineConfig } from 'vite';
import { resolve } from 'path';
import wasm from 'vite-plugin-wasm';

export default defineConfig({
  plugins: [wasm()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@core': resolve(__dirname, 'src/core'),
      '@processors': resolve(__dirname, 'src/processors'),
      '@utils': resolve(__dirname, 'src/utils'),
      '@types': resolve(__dirname, 'src/types'),
      '@schemas': resolve(__dirname, 'src/schemas'),
    },
  },
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'OMRChecker',
      fileName: (format) => `omrchecker.${format}.js`,
    },
    rollupOptions: {
      output: {
        assetFileNames: 'assets/[name][extname]',
        chunkFileNames: 'chunks/[name]-[hash].js',
      },
    },
    target: 'es2020',
    sourcemap: true,
  },
  optimizeDeps: {
    exclude: ['opencv.js'],
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
EOF

echo "✓ vite.config.ts created"

# Generate vitest.config.ts
echo "📝 Generating vitest.config.ts..."
cat > "$TARGET_DIR/vitest.config.ts" << 'EOF'
import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html', 'lcov'],
      include: ['src/**/*.ts'],
      exclude: [
        'src/**/*.test.ts',
        'src/**/*.spec.ts',
        'src/types/**',
        'node_modules/**',
      ],
    },
    testTimeout: 10000,
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@core': resolve(__dirname, 'src/core'),
      '@processors': resolve(__dirname, 'src/processors'),
      '@utils': resolve(__dirname, 'src/utils'),
      '@types': resolve(__dirname, 'src/types'),
      '@schemas': resolve(__dirname, 'src/schemas'),
    },
  },
});
EOF

echo "✓ vitest.config.ts created"

# Generate .gitignore
echo "📝 Generating .gitignore..."
cat > "$TARGET_DIR/.gitignore" << 'EOF'
# Dependencies
node_modules/
.pnp
.pnp.js

# Testing
coverage/
.nyc_output

# Build output
dist/
build/
*.tsbuildinfo

# Environment
.env
.env.local
.env.development
.env.test
.env.production

# Editor
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Temporary
.tmp/
tmp/
temp/

# Migration toolkit outputs
migration-log.json
migration-progress.json
EOF

echo "✓ .gitignore created"

# Generate test setup file
echo "📝 Generating test setup..."
cat > "$TARGET_DIR/tests/setup.ts" << 'EOF'
import { expect, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

// Cleanup after each test
afterEach(() => {
  cleanup();
});

// Custom matchers (extend as needed)
expect.extend({
  toBeWithinRange(received: number, floor: number, ceiling: number) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () => `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false,
      };
    }
  },
});

// Mock OpenCV.js (loaded dynamically in browser)
global.cv = {
  Mat: class Mat {},
  imread: () => ({}),
  imshow: () => {},
  // Add more mocks as needed
} as any;
EOF

echo "✓ Test setup created"

# Generate README for omrchecker-js
echo "📝 Generating README.md..."
cat > "$TARGET_DIR/README.md" << 'EOF'
# OMRChecker.js

Browser-based OMR (Optical Mark Recognition) Checker - TypeScript port of [OMRChecker](https://github.com/Udayraj123/OMRChecker).

## Features

- ✅ Client-side OMR processing (no server required)
- ✅ Template-based configuration
- ✅ Multiple detection types (bubbles, barcodes, OCR)
- ✅ Image alignment and preprocessing
- ✅ Parallel processing with Web Workers
- ✅ Batch evaluation and grading
- ✅ Debug visualization

## Quick Start

### Installation

```bash
npm install omrchecker-js
```

### Usage

```typescript
import { OMRChecker } from 'omrchecker-js';

// Load template
const template = await OMRChecker.loadTemplate('/path/to/template.json');

// Process image
const image = await loadImageFile(file);
const result = await OMRChecker.processImage(image, template);

console.log('Detected responses:', result.responses);
console.log('Score:', result.score);
```

## Development

```bash
# Install dependencies
npm install

# Run dev server
npm run dev

# Run tests
npm test

# Build for production
npm run build
```

## Browser Compatibility

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| OpenCV.js | ✅ | ✅ | ✅ | ✅ |
| Web Workers | ✅ | ✅ | ✅ | ✅ |
| File API | ✅ | ✅ | ✅ | ✅ |
| WASM | ✅ | ✅ | ✅ | ✅ |

Minimum versions: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

## Architecture

- **Core**: Template, Field, FieldBlock, Config
- **Processors**: Alignment, Detection, Threshold, Evaluation, Preprocessing
- **Utils**: Logger, Validation, Parsing, File handling
- **Workers**: Parallel image processing

See [migration documentation](../.agents/skills/omrchecker-migration-skill/) for detailed architecture.

## Migration from Python

This is a TypeScript port of the original Python OMRChecker. Key differences:

| Python | JavaScript/TypeScript |
|--------|-----------------------|
| OpenCV (cv2) | OpenCV.js (WASM) |
| NumPy | TypedArrays + ndarray.js |
| Pydantic | Zod |
| pytest | Vitest |
| ThreadPoolExecutor | Web Workers |
| Rich terminal | HTML/Canvas |

## Documentation

- [Migration Guide](../.agents/skills/omrchecker-migration-skill/README.md)
- [API Documentation](./docs/api.md)
- [Template Format](../.agents/skills/omrchecker-migration-skill/modules/integration/template-format/template-format.md)
- [Configuration](../.agents/skills/omrchecker-migration-skill/modules/foundation/configuration.md)

## License

GPL-3.0 (same as OMRChecker)

## Contributing

Contributions welcome! This is an automated migration from Python OMRChecker.

## Acknowledgments

Original Python OMRChecker by [Udayraj Deshmukh](https://github.com/Udayraj123)
EOF

echo "✓ README.md created"

# Generate src/index.ts entry point
echo "📝 Generating src/index.ts..."
cat > "$TARGET_DIR/src/index.ts" << 'EOF'
/**
 * OMRChecker.js - Browser-based OMR Detection
 * TypeScript port of OMRChecker Python library
 */

// Core exports
export { Template } from './core/template';
export { Field } from './core/field';
export { FieldBlock } from './core/field-block';
export { Config } from './core/config';

// Processor exports
export { processImage } from './processors/pipeline';
export { alignImage } from './processors/alignment';
export { detectBubbles } from './processors/detection';
export { evaluateResponse } from './processors/evaluation';

// Utility exports
export { logger } from './utils/logger/logger';
export { validateTemplate } from './utils/validation/template';
export { validateConfig } from './utils/validation/config';

// Type exports
export type * from './types';

// Default export
export { OMRChecker } from './omrchecker';
EOF

echo "✓ src/index.ts created"

# Generate placeholder OMRChecker class
echo "📝 Generating src/omrchecker.ts..."
cat > "$TARGET_DIR/src/omrchecker.ts" << 'EOF'
import { Template } from './core/template';
import { Config } from './core/config';
import { processImage } from './processors/pipeline';
import type { ProcessingResult } from './types';

/**
 * Main OMRChecker class - entry point for browser-based OMR processing
 */
export class OMRChecker {
  /**
   * Load a template from JSON
   */
  static async loadTemplate(templatePath: string, config?: Config): Promise<Template> {
    const response = await fetch(templatePath);
    const templateJson = await response.json();
    return new Template(templateJson, config);
  }

  /**
   * Process an image with a template
   */
  static async processImage(
    image: HTMLImageElement | ImageData,
    template: Template,
  ): Promise<ProcessingResult> {
    return processImage(image, template);
  }

  /**
   * Process multiple images in batch
   */
  static async processBatch(
    images: (HTMLImageElement | ImageData)[],
    template: Template,
    options?: { parallel?: boolean; maxWorkers?: number },
  ): Promise<ProcessingResult[]> {
    // TODO: Implement batch processing with Web Workers
    const results: ProcessingResult[] = [];
    for (const image of images) {
      results.push(await this.processImage(image, template));
    }
    return results;
  }
}
EOF

echo "✓ src/omrchecker.ts created"

echo ""
echo "✅ Project setup complete!"
echo ""
echo "Next steps:"
echo "  1. cd $TARGET_DIR"
echo "  2. npm install"
echo "  3. Run interface generator: node ../.agents/migration-toolkit/2-generate-interfaces.js"
echo "  4. Run test scaffolding: node ../.agents/migration-toolkit/3-generate-tests.js"
echo "  5. Run file migration: node ../.agents/migration-toolkit/4-migrate-files.js"
echo ""
echo "📚 Documentation: .agents/skills/omrchecker-migration-skill/"
echo ""
