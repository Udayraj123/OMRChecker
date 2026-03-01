# @omrchecker/core

TypeScript/JavaScript implementation of OMRChecker - Optical Mark Recognition for bubble sheets.

## Overview

This is a browser-compatible port of the Python OMRChecker library. It uses OpenCV.js for computer vision operations and runs entirely in the browser without requiring a server.

**Key Features:**
- Pure TypeScript with full type safety
- Browser-based processing (no server needed)
- OpenCV.js for image processing
- Web Crypto API for checksums
- Comprehensive memory leak detection
- Dual Python-TypeScript codebase (auto-synced)

## Installation

```bash
npm install @omrchecker/core
```

## Usage

```typescript
import { OMRChecker } from '@omrchecker/core';

// Load template and config
const template = await fetch('/path/to/template.json').then(r => r.json());
const config = await fetch('/path/to/config.json').then(r => r.json());

// Initialize OMRChecker
const checker = new OMRChecker(template, config);

// Process an image
const imageElement = document.getElementById('omr-image') as HTMLImageElement;
const result = await checker.process(imageElement);

console.log('Detected responses:', result.responses);
console.log('Score:', result.score);
```

## Development

### Setup

```bash
# Install dependencies
npm install

# Install Playwright browsers (for browser tests)
npx playwright install chromium
```

### Testing

```bash
# Run all tests (unit + browser)
npm run test:all

# Run unit tests only (fast, jsdom)
npm run test:unit

# Run browser tests only (Playwright)
npm run test:browser

# Watch mode
npm run test:watch

# Coverage
npm run coverage:all
```

### Building

```bash
# Type check
npm run typecheck

# Build
npm run build
```

## Project Structure

```
src/
├── utils/           # Utilities (math, array, drawing, checksum)
├── processors/      # Image processors (preprocessing, detection, evaluation)
├── schemas/         # Zod schemas for validation
└── index.ts         # Public API

tests/
├── unit/           # Unit tests (jsdom)
├── browser/        # Browser tests (Playwright + OpenCV.js)
│   ├── browser-setup.ts      # OpenCV.js loading utilities
│   ├── memory-utils.ts       # Memory leak detection
│   ├── opencv-smoke.test.ts  # OpenCV smoke tests
│   ├── drawing.test.ts       # Drawing utilities tests
│   ├── checksum.test.ts      # Checksum tests (Web Crypto)
│   └── memory-leak.test.ts   # Memory leak detection tests
└── README.md       # Testing documentation
```

## Browser Testing

This package includes comprehensive browser testing using Playwright to test OpenCV.js operations in real browsers:

- **OpenCV.js Integration**: Tests actual OpenCV operations (Mat creation, drawing, transformations)
- **Memory Leak Detection**: Automatic tracking of cv.Mat allocations to prevent memory leaks
- **Web Crypto API**: Tests checksum generation using native SubtleCrypto API
- **Real Browser Environment**: Uses Chromium via Playwright for authentic testing

See `tests/README.md` for detailed testing documentation.

## Python-TypeScript Parity

This codebase maintains parity with the Python implementation:

- **FILE_MAPPING.json**: Tracks Python ↔ TypeScript file correspondence
- **Auto-sync hooks**: Pre-commit hooks suggest TypeScript changes when Python files are modified
- **Validation**: Pre-commit validation ensures correspondence is maintained

When modifying TypeScript files, consider updating the corresponding Python files (and vice versa) to maintain parity.

## Contributing

1. Write tests first (TDD approach)
2. Ensure all tests pass: `npm run test:all`
3. Check types: `npm run typecheck`
4. Follow existing code patterns
5. Update documentation as needed

## License

MIT

## Related Projects

- [OMRChecker (Python)](https://github.com/Udayraj123/OMRChecker) - Original Python implementation
- [@omrchecker/web](../web) - Web UI for browser-based OMR processing

## Links

- [Documentation](https://github.com/Udayraj123/OMRChecker/tree/master/omrchecker-js)
- [Python Docs](https://github.com/Udayraj123/OMRChecker/blob/master/README.md)
- [Testing Strategy](../../docs/TESTING_STRATEGY.md)
