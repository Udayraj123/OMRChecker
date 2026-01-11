# OMRChecker TypeScript Port

This is the TypeScript/JavaScript port of OMRChecker, built as a browser-compatible library with React demo and E2E tests.

## Project Structure

```
omrchecker-js/
├── packages/
│   ├── core/               # Main OMRChecker library
│   ├── demo/               # React demo application
│   └── e2e/                # Playwright E2E tests
├── change-propagation-tool/  # Interactive Python ↔ TypeScript sync tool
└── pnpm-workspace.yaml     # pnpm workspace configuration
```

## Getting Started

### Prerequisites

- Node.js 18+ 
- pnpm 8+

### Installation

```bash
# Install pnpm globally if you haven't
npm install -g pnpm

# Install dependencies
pnpm install
```

### Development

```bash
# Run demo app in development mode
pnpm dev

# Run core library in watch mode
pnpm dev:core

# Run tests
pnpm test

# Run E2E tests
pnpm test:e2e

# Lint all packages
pnpm lint

# Format code
pnpm format

# Type check all packages
pnpm typecheck
```

### Building

```bash
# Build all packages
pnpm build

# Build only core library
pnpm build:core
```

## Packages

### @omrchecker/core

The main OMRChecker library ported to TypeScript. Provides core functionality for:
- Image preprocessing
- Template alignment
- OMR detection and interpretation
- Answer evaluation

### @omrchecker/demo

React-based demo application showcasing the OMRChecker library capabilities in the browser.

### @omrchecker/e2e

End-to-end tests using Playwright to ensure the library works correctly across different browsers.

## Change Propagation Tool

Interactive web UI for synchronizing Python and TypeScript code changes. See [DEPENDENCY_MAPPING.md](../DEPENDENCY_MAPPING.md) for details.

```bash
pnpm run change-tool
```

## Architecture

This TypeScript port maintains 1:1 correspondence with the Python codebase:

- **Module Structure**: Mirrors Python `src/` structure
- **Naming Conventions**: snake_case → camelCase for functions, PascalCase preserved for classes
- **Type System**: Python type hints → TypeScript types
- **OpenCV**: Uses opencv.js (WASM) instead of cv2

See [FILE_MAPPING.json](../FILE_MAPPING.json) for complete Python ↔ TypeScript mapping.

## Browser Compatibility

- Chrome/Edge 90+
- Firefox 89+
- Safari 14.1+

Requires WebAssembly and Web Workers support.

## Contributing

Please refer to [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:
- Code style and conventions
- Python-TypeScript synchronization workflow
- Testing requirements
- Pull request process

## License

Same as main OMRChecker project - see [LICENSE](../LICENSE).

