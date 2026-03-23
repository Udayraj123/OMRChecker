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

### TypeScript Testing Principles: "Translate, Don't Invent"

**CRITICAL RULE**: When migrating Python code to TypeScript, you MUST translate existing Python tests, not create new ones from scratch.

**Why This Matters**:
- Ensures behavioral parity between Python and TypeScript implementations
- Prevents test coverage drift and redundant testing
- Maintains single source of truth for business logic validation
- Avoids inventing test cases that may not reflect actual requirements

**Workflow for Test Migration**:

1. **Before writing ANY TypeScript test**, check if Python test exists:
   ```bash
   # For utils/math.ts, check for:
   find src/tests -name "*math*" -o -name "test_math.py"

   # For processors/image/GaussianBlur.ts, check for:
   find src/tests -name "*gaussian*" -o -name "*blur*" -o -name "*processor*"
   ```

2. **If Python test exists**: Translate it line-by-line to TypeScript
   - Match test names (e.g., `test_add_two_numbers()` → `describe('add', () => { it('adds two numbers', ...) }`)
   - Match test inputs and expected outputs exactly
   - Preserve edge cases and boundary conditions
   - Keep same assertion logic

3. **If Python test does NOT exist**: Check with project lead before writing tests
   - Python may have inline doctests or integration tests instead
   - The function may be untested intentionally (e.g., simple wrappers)
   - Adding tests without Python equivalent creates maintenance burden

4. **Exception: Processor/Integration Tests**
   - Python has NO unit tests for individual processors
   - Python tests entire pipeline end-to-end with sample images
   - TypeScript CAN add processor unit tests (they're environment-specific)
   - Browser tests for processors are ACCEPTABLE and encouraged

**Audit Process**:

Before completing migration, audit test coverage:

```bash
cd omrchecker-js/packages/core

# Check test files against Python tests
ls tests/unit/*.test.ts | while read f; do
  echo "\nTypeScript: $f"
  basename="$(basename $f .test.ts)"
  echo "Python equivalent: src/tests/test_${basename}.py"
  ls ../../src/tests/test_${basename}.py 2>/dev/null || echo "  ⚠️  NO PYTHON TEST FOUND"
done
```

**Reference**: See `omrchecker-js/packages/core/tests/TEST_COVERAGE_AUDIT.md` for current coverage analysis.

**Examples**:

✅ **CORRECT**: Translating Python test
```python
# Python: src/tests/test_drawing.py
def test_draw_contour():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    contour = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
    result = draw_contour(img, contour, (255, 0, 0), 2)
    assert result.shape == (100, 100, 3)
```

```typescript
// TypeScript: tests/unit/drawing.test.ts
it('draws contour on image', () => {
  const img = cv.Mat.zeros(100, 100, cv.CV_8UC3);
  const contour = cv.matFromArray(4, 1, cv.CV_32SC2, [10,10, 90,10, 90,90, 10,90]);
  const result = drawContour(img, contour, [255, 0, 0], 2);
  expect(result.rows).toBe(100);
  expect(result.cols).toBe(100);
});
```

❌ **INCORRECT**: Inventing new tests
```typescript
// TypeScript: tests/unit/math.test.ts
// NO Python equivalent exists - these 59 tests were invented!
it('handles negative infinity', () => { /* ... */ });
it('handles subnormal numbers', () => { /* ... */ });
it('validates IEEE 754 compliance', () => { /* ... */ });
```

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

## Pixel Agents Visualizer (optional)

The **Pixel Agents** VS Code extension shows bd activity as animated pixel-art characters in a virtual office. When it's running, exporting `PIXEL_AGENTS_*` env vars causes your `bd update` calls to automatically push live status to the visualizer — no extra commands needed.

### Setup (do this once per session)

**First, check the server is live:**
```bash
curl -sf http://localhost:3001/health >/dev/null 2>&1 \
  && echo "✓ Pixel Agents server live" \
  || { echo "⚠️  Pixel Agents server not live — skipping hook setup"; return 0 2>/dev/null || true; }
```

**If live, export your session variables:**
```bash
# Replace with your actual agent name (matches the name prefix in your bd issue titles)
# and the shared session ID for this run (ask the coordinator or use the default below)
export PIXEL_AGENTS_SESSION=omr-run
export PIXEL_AGENTS_AGENT_ID=Schema-Validator   # e.g. Foundation-Alpha, Processor-Delta …
```

After this, every `bd update <id> --status in_progress` or `bd close <id>` automatically
updates the visualizer. No other changes to your workflow.

**Optional — tell the visualizer who your coordinator is (for task-assignment animations):**
```bash
export PIXEL_AGENTS_COORDINATOR_ID=coordinator
```

### How the hook maps bd titles → agent names

If your agent name is embedded as a title prefix (`Schema-Validator: Write schema tests`),
the hook extracts it automatically even without `PIXEL_AGENTS_AGENT_ID`. The explicit env var
always takes priority if set.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **OMRChecker** (3133 symbols, 8841 relationships, 242 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/OMRChecker/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/OMRChecker/context` | Codebase overview, check index freshness |
| `gitnexus://repo/OMRChecker/clusters` | All functional areas |
| `gitnexus://repo/OMRChecker/processes` | All execution flows |
| `gitnexus://repo/OMRChecker/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
