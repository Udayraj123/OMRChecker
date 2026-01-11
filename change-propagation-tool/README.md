# Change Propagation Tool

Interactive React-based tool for managing Python ↔ TypeScript code synchronization in OMRChecker.

## Features

- **Visual Dashboard**: See sync status at a glance
- **Filtering & Search**: Find files by status, phase, priority
- **Statistics**: Track overall progress
- **Real-time Updates**: Refresh to see latest changes

## Getting Started

### Installation

```bash
# From OMRChecker root
cd change-propagation-tool
pnpm install
```

### Development

```bash
pnpm dev
```

Opens at http://localhost:5174

### Building

```bash
pnpm build
```

## Usage

1. **Launch Tool**: `pnpm dev` or `pnpm run change-tool` from repo root
2. **View Status**: Dashboard shows all file mappings with sync status
3. **Filter Files**: Use dropdowns to filter by status/phase/priority
4. **Search**: Type to search Python or TypeScript file names
5. **Refresh**: Click refresh button to reload FILE_MAPPING.json

## Architecture

- **React 18** with TypeScript
- **Vite** for fast development
- **Tailwind CSS** for styling
- **React Router** for navigation
- **Monaco Editor** for code viewing/editing (planned)

## Integration

Reads `FILE_MAPPING.json` from parent directory to display sync status.
Can be launched automatically from pre-commit hooks when changes are detected.

