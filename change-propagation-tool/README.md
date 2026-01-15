# Change Propagation Tool

Interactive React-based tool for managing Python ↔ TypeScript code synchronization in OMRChecker.

## Features

### 📊 Progress Overview
- **Overall Progress**: Visual progress bar showing percentage of files synced
- **Phase Tracking**: See completion status for each development phase
- **Phase History**: View Phase 7, 8, 9 improvements with completion dates
- **Status Breakdown**: Quick stats on synced, partial, and not started files

### 🔍 Code Mapping Details
- **Class Mappings**: View all Python → TypeScript class mappings
- **Method Sync Status**: See which methods are synced vs pending
- **Function Mappings**: Track standalone function ports
- **Method Notes**: View phase-specific notes and changes
- **Progress Bars**: Visual indication of class/method completion

### 📋 File Management
- **Visual Dashboard**: See sync status at a glance
- **Filtering & Search**: Find files by status, phase, priority
- **Dual View Modes**: Switch between table and grid layouts
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

### 1. Launch Tool
```bash
pnpm dev
```
Or from repo root:
```bash
cd change-propagation-tool && pnpm dev
```

### 2. View Progress
- **Progress Overview**: See overall porting completion and recent phases
- **Phase Cards**: Expand to see detailed improvements for each phase
- **Statistics**: View breakdown by status, phase, and priority

### 3. Browse Mappings
- **Table View**: Compact list with all file mappings
- **Grid View**: Card-based layout for easier browsing
- **Filters**: Use dropdowns to filter by status/phase/priority
- **Search**: Type to search Python or TypeScript file names

### 4. View Code Details
- Click "View Details" on any file mapping
- See all classes with Python → TypeScript names
- Expand classes to view method mappings
- Check sync status for each method
- Read phase-specific notes and improvements

### 5. Refresh Data
- Click "Refresh" button in header
- Reloads `FILE_MAPPING.json` to show latest changes

## Features in Detail

### Progress Overview Component
Shows:
- Overall porting progress percentage
- Phase 1 (Core) completion
- Status breakdown (synced/partial/not started)
- Recent development phases (7, 8, 9) with:
  - Summary descriptions
  - Completion dates
  - Key improvements lists

### Code Mapping Detail Modal
Displays:
- File paths (Python, TypeScript, Test)
- Status, phase, and priority badges
- General notes about the file
- **Classes Section**:
  - Class name mappings
  - Sync status
  - Method count and progress
  - Expandable method list
  - Method-level sync status
  - Phase notes for methods
- **Functions Section**:
  - Function mappings
  - Sync status
  - Implementation notes

### View Modes
- **Table View**: Traditional list layout, compact and scannable
- **Grid View**: Card-based layout with more visual information

### Filtering Options
- **Status**: All, Synced, Partial, Not Started
- **Phase**: All, Phase 1, Phase 2, Future
- **Priority**: All, High, Medium, Low
- **Search**: Fuzzy search across file paths

## Architecture

- **React 18** with TypeScript
- **Vite** for fast development
- **Tailwind CSS** for styling
- **React Router** for navigation

## Components

### `App.tsx`
- Main application wrapper
- Data loading and error handling
- Header with refresh button

### `Dashboard.tsx`
- Main dashboard view
- File list with filtering
- View mode toggle
- Integrates ProgressOverview

### `ProgressOverview.tsx`
- Overall progress visualization
- Phase history cards
- Phase distribution stats

### `CodeMappingDetail.tsx`
- Detailed code mapping modal
- Class/method breakdown
- Function mappings
- Sync status indicators

### `mappingService.ts`
- Loads FILE_MAPPING.json
- Provides filtering methods
- Manages mapping data

## Integration

### Data Source
Reads `FILE_MAPPING.json` from parent directory. The JSON structure includes:

```json
{
  "version": "2.0",
  "metadata": {
    "phase7_completed": "...",
    "phase7_summary": "...",
    "phase7_improvements": { ... },
    "phase8_completed": "...",
    "phase8_summary": "...",
    "phase8_improvements": { ... },
    "phase9_completed": "...",
    "phase9_summary": "...",
    "phase9_improvements": { ... }
  },
  "mappings": [
    {
      "python": "src/...",
      "typescript": "omrchecker-js/...",
      "status": "synced|partial|not_started",
      "phase": 1,
      "priority": "high|medium|low",
      "classes": [
        {
          "python": "ClassName",
          "typescript": "ClassName",
          "synced": true,
          "methods": [
            {
              "python": "method_name",
              "typescript": "methodName",
              "synced": true,
              "notes": "Phase 9: ..."
            }
          ]
        }
      ],
      "functions": [...],
      "notes": "..."
    }
  ],
  "statistics": { ... }
}
```

### Auto-Launch
Can be launched automatically from pre-commit hooks when Python changes are detected.

## Future Enhancements

- [ ] Side-by-side code diff view (Monaco Editor)
- [ ] Git integration to show actual file diffs
- [ ] Auto-sync detection from git history
- [ ] Export reports (PDF, HTML)
- [ ] Phase timeline visualization
- [ ] Dependency graph visualization
- [ ] File change notifications
- [ ] Automated testing status

## Development

### Type Checking
```bash
pnpm run typecheck
```

### Linting
```bash
pnpm run lint
```

### Build
```bash
pnpm run build
```

## Contributing

When adding new features to FILE_MAPPING.json:
1. Update TypeScript types in `src/types/index.ts`
2. Add UI components as needed
3. Update this README
4. Test with `pnpm run typecheck`

