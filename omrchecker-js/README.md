# OMRChecker TypeScript Port

This is the TypeScript/JavaScript port of OMRChecker, built as a browser-compatible library with React demo and E2E tests.

## Project Structure

```
omrchecker-js/
├── packages/
│   ├── core/               # Main OMRChecker library
│   ├── demo/               # React demo application
│   └── e2e/                # Playwright E2E tests
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

## Workflow

### Development Workflow

1. Make changes to Python code in `src/`
2. Stage your changes: `git add src/...`
3. Commit: `git commit`
4. The pre-commit hook will automatically:
   - Auto-sync structural changes (classes/methods) to TypeScript
   - Stage the updated TypeScript files
   - Validate that all changes are synced
5. Review the auto-synced TypeScript code
6. Manually fix implementation details, types, and logic
7. Run tests: `pnpm test`
8. Stage additional changes if needed: `git add omrchecker-js/...`
9. Amend or create a new commit

### Manual Sync (without committing)

You can also run the auto-sync manually:

```bash
# Run auto-sync on staged Python files
uv run python scripts/sync_tool.py auto-sync

# Check sync status
uv run python scripts/sync_tool.py status

# Detect changes
uv run python scripts/sync_tool.py detect

# Generate TypeScript suggestions for a specific file
uv run python scripts/sync_tool.py suggest src/processors/image/CropPage.py
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

