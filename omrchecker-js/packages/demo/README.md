# OMRChecker Demo App

A beautiful, modern web application demonstrating browser-based OMR (Optical Mark Recognition) bubble detection using TypeScript and OpenCV.js.

## Features

✨ **Complete OMR Processing Pipeline**
- Full `OMRProcessor` integration with preprocessing, alignment, and detection
- Template-based bubble detection
- Real-time image processing
- Confidence scoring
- Multi-mark detection
- Automatic evaluation/scoring (when answer key provided)
- Batch processing support

🎨 **Modern UI**
- Dark theme with gradient backgrounds
- Responsive design
- Interactive visualizations
- Real-time feedback

📊 **Results & Export**
- Detailed statistics
- Score display (when available)
- Sortable results table
- CSV export
- Visual bubble overlay

🚀 **Advanced Features**
- Image preprocessing pipeline
- Template alignment correction
- Folder upload with File System Access API
- Multi-file batch processing
- Auto-detection of `template.json` in folders

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
2. **Upload Image**: Choose an OMR sheet image (JPG, PNG, etc.) or select a folder
3. **Detect**: Click "Detect Bubbles" to process the image(s)
4. **Review**: See detection results with confidence scores and evaluation
5. **Export**: Download results as CSV

### Folder Upload

The demo supports batch processing through folder upload (Chrome/Edge only):
1. Click "📁 Or Select Folder"
2. Select a folder containing OMR images
3. If `template.json` exists in the folder, it will be auto-loaded
4. All images in the folder (including subfolders) will be processed

### Template Features

Your template can include:
- **Preprocessing**: Define image processing steps (crop, rotate, threshold)
- **Alignment**: Specify alignment markers for automatic correction
- **Evaluation**: Provide answer keys for automatic scoring
- **Custom Bubbles**: Define bubble locations, sizes, and labels

See the Python OMRChecker documentation for template format details.

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
OMRProcessor (orchestrator)
    ↓
Preprocessing Pipeline
    ├── Crop & Rotate
    ├── Noise Reduction
    └── Threshold Adjustment
    ↓
Alignment Processor (if configured)
    ├── Marker Detection
    └── Shift Calculation
    ↓
Bubble Detection
    ├── Field-level Processing
    ├── Threshold Calculation
    └── Multi-mark Detection
    ↓
Evaluation (if answer key provided)
    ├── Score Calculation
    └── Correctness Checking
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

