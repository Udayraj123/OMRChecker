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
