# OMRChecker Demo App

A beautiful, modern web application demonstrating browser-based OMR (Optical Mark Recognition) bubble detection using TypeScript and OpenCV.js.

## Features

✨ **Complete OMR Detection Pipeline**
- Template-based bubble detection
- Real-time image processing
- Confidence scoring
- Multi-mark detection

🎨 **Modern UI**
- Dark theme with gradient backgrounds
- Responsive design
- Interactive visualizations
- Real-time feedback

📊 **Results & Export**
- Detailed statistics
- Sortable results table
- CSV export
- Visual bubble overlay

## Getting Started

### Prerequisites

- Node.js 18+
- pnpm 8+

### Installation

```bash
# From the monorepo root
pnpm install

# Or from this package
cd packages/demo
pnpm install
```

### Running the Demo

```bash
# From the monorepo root
pnpm run dev

# Or from this package
pnpm run dev
```

The demo will open at `http://localhost:3000`

## Usage

1. **Upload Template**: Choose a `template.json` file that defines bubble locations
2. **Upload Image**: Choose an OMR sheet image (JPG, PNG, etc.)
3. **Detect**: Click "Detect Bubbles" to process the image
4. **Review**: See detection results with confidence scores
5. **Export**: Download results as CSV

## Example Template

See `/examples/simple-template.json` in the monorepo for a sample template with 10 multiple-choice questions.

## Technology Stack

- **TypeScript**: Type-safe application code
- **Vite**: Fast build tool and dev server
- **OpenCV.js**: Computer vision for bubble detection
- **@omrchecker/core**: TypeScript port of OMRChecker

## Architecture

```
User Input
    ↓
Template Loader (parse template.json)
    ↓
Image Loader (load & convert to grayscale)
    ↓
Bubble Detector (threshold-based detection)
    ↓
Visualization (draw bubbles on image)
    ↓
Results Display (statistics + table + CSV)
```

## Development

### Build

```bash
pnpm run build
```

### Type Check

```bash
pnpm run typecheck
```

### Lint

```bash
pnpm run lint
```

## Browser Compatibility

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

Requires WebAssembly support for OpenCV.js.

## License

MIT

