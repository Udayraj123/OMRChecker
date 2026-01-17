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

### Multi-Pass Detection & Interpretation Architecture

The TypeScript implementation uses a **multi-pass architecture** that matches the Python version exactly:

1. **Detection Pass**: Runs detection on all fields, collecting aggregates (bubble means, thresholds, etc.)
2. **Interpretation Pass**: Uses aggregates from detection pass to interpret fields and determine answers

This architecture enables:
- **Global threshold calculation** using file-level aggregates
- **Confidence metrics** for ML training
- **Multi-marking detection** across all fields
- **Aggregate collection** at field, file, and directory levels

#### Key Components

- **`FilePassAggregates`**: Base class managing aggregates at three levels (field, file, directory)
- **`FieldTypeDetectionPass`**: Abstract base for field type detection passes
- **`FieldTypeInterpretationPass`**: Abstract base for field type interpretation passes
- **`FileLevelRunner`**: Coordinates detection and interpretation passes
- **`TemplateFileRunner`**: Template-level runner managing all field type runners

#### Accessing Aggregates

For advanced use cases, you can access aggregates:

```typescript
const processor = new OMRProcessor(templateConfig);
await processor.processImage(image, 'test.jpg');

// Get aggregates for analysis
const aggregates = processor.getAggregates();
if (aggregates) {
  const detectionAggs = aggregates.detection;
  const interpretationAggs = aggregates.interpretation;
  // Use aggregates for custom analysis, ML training, etc.
}
```

#### Confidence Metrics

When enabled in tuning config, confidence metrics are collected:

```typescript
const templateConfig = {
  // ... template config
  tuningConfig: {
    outputs: {
      show_confidence_metrics: true,
    },
  },
};

const processor = new OMRProcessor(templateConfig);
const result = await processor.processImage(image, 'test.jpg');

// Confidence metrics available in aggregates
const aggregates = processor.getAggregates();
// Access field-level confidence metrics from interpretation aggregates
```

## Documentation

See main [README](../../README.md) and [DEPENDENCY_MAPPING.md](../../../DEPENDENCY_MAPPING.md) for details on Python ↔ TypeScript mapping.

