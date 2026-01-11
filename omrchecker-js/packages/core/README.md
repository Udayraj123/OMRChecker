# @omrchecker/core

Core OMRChecker library ported to TypeScript for browser use.

## Features

- **Image Preprocessing**: Auto-rotation, cropping, filtering
- **Template Alignment**: Feature-based alignment using OpenCV.js
- **OMR Detection**: Bubble detection and interpretation
- **Answer Evaluation**: Scoring and evaluation logic
- **Browser Compatible**: Uses opencv.js (WebAssembly) for image processing

## Installation

```bash
pnpm add @omrchecker/core
```

## Usage

```typescript
import { OMRProcessor } from '@omrchecker/core';

// Initialize processor
const processor = new OMRProcessor(templateConfig);

// Process OMR image
const result = await processor.processImage(imageData);

console.log(result.omrResponse);
console.log(result.score);
```

## Development

```bash
# Build library
pnpm build

# Watch mode
pnpm dev

# Run tests
pnpm test

# Type check
pnpm typecheck
```

## Architecture

Follows the same modular architecture as the Python version:

- `core/`: Core types and processor
- `processors/`: Image processing, alignment, detection
- `utils/`: Utility functions
- `schemas/`: JSON schema validation

## Documentation

See main [README](../../README.md) and [DEPENDENCY_MAPPING.md](../../../DEPENDENCY_MAPPING.md) for details on Python ↔ TypeScript mapping.

