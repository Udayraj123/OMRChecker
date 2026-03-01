# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

OMRChecker evaluates OMR (Optical Mark Recognition) sheets using computer vision. It processes scanned documents or mobile camera images to detect marked responses on bubble sheets, similar to standardized test forms.

**Tech Stack:** Python 3.11+, OpenCV, uv package manager, pytest, ruff (linting/formatting)

**Key Features:**
- Classical image processing pipeline (no ML required for basic usage)
- Optional ML-based detection (YOLO, custom models)
- Supports low resolution, xeroxed, colored sheets at any angle
- Parallel TypeScript port in `omrchecker-js/` (browser-based)

## Development Commands

### Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Running OMRChecker
```bash
# Process default inputs directory
uv run main.py

# Process specific directory
uv run main.py -i inputs/sample1

# Process with custom output directory
uv run main.py -i inputs/sample1 -o outputs/results

# Set up template layout (interactive mode)
uv run main.py --setLayout -i inputs/sample1

# Debug mode
uv run main.py -i inputs/sample1 --debug
```

### Testing
```bash
# Run single quick test (used in pre-commit)
./scripts/run_single_test.sh

# Run all tests with coverage (used in pre-push, requires 70% coverage)
./scripts/run_all_tests_and_coverage.sh

# Run specific test
uv run pytest src/tests/path/to/test_file.py::test_function_name -vv

# Run tests in parallel (auto-detect CPUs)
uv run pytest -n auto

# Run tests with specific marker
uv run pytest -m "sample_1_mobile_camera"

# Keep output files for inspection
uv run pytest --keep-outputs
```

### TypeScript Testing (omrchecker-js)
```bash
# Navigate to TypeScript package
cd omrchecker-js/packages/core

# Run unit tests only (fast, jsdom)
npm run test:unit

# Run browser tests (Playwright + Chromium)
npm run test:browser

# Run all tests (unit + browser)
npm run test:all

# Watch mode
npm run test:watch              # Unit tests
npm run test:watch:browser      # Browser tests with UI

# Coverage
npm run coverage:unit
npm run coverage:browser
npm run coverage:all
```

**IMPORTANT for AI Agents**: 
- The `npm run test:browser` command uses Playwright and is configured with `{ open: 'never' }` to avoid waiting for user input
- DO NOT use `playwright test --ui` or any watch mode commands in automated workflows as they wait indefinitely
- Browser tests load OpenCV.js from CDN and may take 30-60 seconds to complete
- If tests hang, check `playwright.config.ts` reporter configuration
- **When writing browser tests, see `omrchecker-js/packages/core/tests/AGENT_TESTING_GUIDE.md` for critical rules and patterns**

### Code Quality
```bash
# Run ruff linter (with auto-fix)
uv run ruff check --fix

# Run ruff formatter
uv run ruff format

# Run pre-commit hooks manually
pre-commit run -a

# Type checking (pyright)
uv run pyright tools/
```

### Pre-commit Hooks
The project uses extensive pre-commit hooks that run automatically on `git commit` and `git push`:
- **On commit**: ruff check/format, image optimization, Python→TypeScript sync, single test
- **On push**: full test suite with coverage (must be ≥70%)

Important hooks:
- Python→TypeScript auto-sync and validation (maintains dual codebase)
- Image optimization (PNG→JPG conversion, resizing, compression)
- uv lock file synchronization

## Architecture Overview

### Core Pipeline
OMRChecker uses a **unified processor-based pipeline** where all processing stages implement the same `Processor` interface:

```
Input Image → Preprocessing → Alignment → Detection → Evaluation → Output
                   ↓              ↓           ↓           ↓
           (rotate, crop)  (feature-based) (bubbles)  (scoring)
```

**Main Entry Point:** `main.py` → `src/entry.py` → processes directories recursively

**Processing Flow:**
1. **Template Loading** (`src/processors/layout/template/`): Loads `template.json` defining OMR layout
2. **Preprocessing** (`src/processors/image/`): Rotation, cropping, filtering (Gaussian/Median blur, Levels, Contrast)
3. **Alignment** (`src/processors/image/alignment/`): Feature-based alignment (ORB/AKAZE), homography warping
4. **Detection** (`src/processors/detection/`): Multi-pass bubble detection with ML fallback
5. **Evaluation** (`src/processors/evaluation/`): Scoring based on answer keys from `evaluation.json`

### Directory Structure
```
src/
├── entry.py              # Main CLI entry point
├── processors/           # All processing stages
│   ├── base.py          # Base Processor interface
│   ├── pipeline.py      # ProcessingPipeline orchestration
│   ├── image/           # Preprocessing (rotation, crop, filters)
│   ├── detection/       # Bubble detection (traditional + ML)
│   ├── evaluation/      # Answer key matching & scoring
│   ├── layout/          # Template parsing & field definitions
│   └── experimental/    # Training data collection, file organization
├── schemas/             # Pydantic models for config/template validation
│   └── models/          # Config, Template, Evaluation models
├── utils/               # Image utils, file I/O, logging, CSV
└── tests/               # Pytest tests with snapshot testing
```

### Key Concepts

**Template (`template.json`)**: Defines OMR sheet layout - field blocks, bubble positions, pre-processors
**Config (`config.json`)**: Tuning parameters - thresholds, output settings, ML config
**Evaluation (`evaluation.json`)**: Answer keys, scoring rules, grouping

**Processor Interface**: All processors implement `process(context) -> context`
**ProcessingContext**: Carries state through pipeline (images, responses, metadata)
**TemplateFileRunner**: Manages multi-pass detection (initial detection → label stats → interpretation)

**Field Types**: `QTYPE_INT`, `QTYPE_MED`, `QTYPE_MCQX` (multi-mark allowed), `QTYPE_MCQ5` (max 5 options)
**Detection Types**: `BUBBLES_THRESHOLD`, `BUBBLES_ML` (ML fallback), `BARCODE`, `OCR`

### Python ↔ TypeScript Dual Codebase

This project maintains **parallel Python and TypeScript implementations**:
- **Python**: Full-featured CLI tool (this codebase)
- **TypeScript**: Browser-based demo in `omrchecker-js/packages/core/`

**Synchronization**:
- `FILE_MAPPING.json`: Tracks Python↔TypeScript file correspondence
- `CHANGE_PATTERNS.yaml`: Translation patterns (snake_case → camelCase, etc.)
- `scripts/auto_sync_python_to_ts.py`: Pre-commit hook for auto-sync
- `scripts/validate_code_correspondence.py`: Pre-commit validation

When editing Python files, the pre-commit hooks automatically:
1. Detect changes in `src/**/*.py`
2. Generate TypeScript suggestions using `generate_ts_suggestions.py`
3. Validate correspondence (fails commit if out of sync)

**Important:** If you modify Python code in `src/`, you may need to update the corresponding TypeScript files in `omrchecker-js/`.

## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

<!-- BEGIN BEADS INTEGRATION -->
## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Auto-syncs to JSONL for version control
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**

```bash
bd ready --json
```

**Create new issues:**

```bash
bd create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bd create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**

```bash
bd update bd-42 --status in_progress --json
bd update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task**: `bd --actor "[Agent-Name]" update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd --actor "[Agent-Name]" create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd --actor "[Agent-Name]" close <id> --reason "Done"`

**IMPORTANT**: Always use `--actor` flag with your agent name/role to ensure proper attribution in beads audit trail.

### Auto-Sync

bd automatically syncs with git:

- Exports to `.beads/issues.jsonl` after changes (5s debounce)
- Imports from JSONL when newer (e.g., after `git pull`)
- No manual export/import needed!

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

### Commit Message Standard

**REQUIRED FORMAT**: All commits must include beads issue ID:
```bash
<type>(<scope>): <summary> (<issue-id>)
```

**Examples**:
```bash
git commit -m "feat(auth): add OAuth2 login (omr-abc)"
git commit -m "test: add unit tests for math.ts (omr-z8z)"
```

**After committing, manually close issue**:
```bash
bd close <issue-id> --reason "Complete (commit $(git rev-parse --short HEAD))"
```

**Note**: Beads does NOT auto-update from commit messages. Always manually close issues with `bd close`.

For complete commit standard including multi-issue commits, agent patterns, and discovered work, see **docs/BEADS_AGENT_WORKFLOW.md**.

### Sub-Agent Task Assignment

When spawning sub-agents for parallel work:

1. **Create issues with agent role prefix**: `[Agent-Role]: Task description`
2. **Use --actor flag** to set agent as creator: `bd --actor "Agent-Name" create ...`
3. **Include detailed description** with `Agent:` field and success criteria
4. **Set dependencies** with `--deps blocked-by:parent-id` or `discovered-from:parent-id`
5. **Sub-agents claim** with `bd --actor "Agent-Name" update <id> --status in_progress`
6. **Sub-agents close** with `bd --actor "Agent-Name" close <id> --reason "..."`

**Example**:
```bash
bd --actor "Foundation-Alpha" create "Foundation-Alpha: Migrate math.py to math.ts" \
  --description="Agent: Foundation-Alpha
Responsibilities: Migrate all 19 methods with type coverage
Success Criteria: Compiles, 100% types, no 'any'
Estimated: 30-45 min" \
  -t task -p 1 --deps blocked-by:omr-setup --json
```

**Why --actor?** This ensures:
- `created_by` field shows "Foundation-Alpha" instead of lead agent name
- `owner` field shows agent identifier for accountability
- Clear audit trail of which agent created/updated which task

For complete sub-agent workflow, role naming conventions, and multi-agent coordination, see **docs/BEADS_AGENT_WORKFLOW.md**.

<!-- END BEADS INTEGRATION -->

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
