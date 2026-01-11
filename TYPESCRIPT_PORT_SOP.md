# Standard Operating Procedure: TypeScript Port Maintenance

## Overview
This document defines the standard operating procedure for maintaining the TypeScript port of OMRChecker, ensuring 1:1 correspondence with the Python codebase.

## Core Principles

### 1. 1:1 File Mapping
- **One Python file = One TypeScript file** (whenever possible)
- **One Python class = One TypeScript class** in its own file
- Each processor should have its own dedicated file and test file
- Maintain clear correspondence for easy review and maintenance

### 2. Naming Conventions
- Python: `ClassNameInCamelCase.py` (e.g., `GaussianBlur.py`)
- TypeScript: `ClassNameInCamelCase.ts` (e.g., `GaussianBlur.ts`)
- Tests: `ClassNameInCamelCase.test.ts` (e.g., `GaussianBlur.test.ts`)
- Class names may differ slightly (e.g., `GaussianBlurProcessor` in Python → `GaussianBlur` in TypeScript)

### 3. Directory Structure
```
Python:
src/processors/
├── image/
│   ├── GaussianBlur.py
│   ├── MedianBlur.py
│   └── AutoRotate.py
└── threshold/
    └── strategies.py  (contains multiple strategies)

TypeScript:
omrchecker-js/packages/core/src/processors/
├── image/
│   ├── GaussianBlur.ts
│   ├── MedianBlur.ts
│   ├── AutoRotate.ts
│   └── __tests__/
│       ├── GaussianBlur.test.ts
│       ├── MedianBlur.test.ts
│       └── AutoRotate.test.ts
└── threshold/
    ├── GlobalThreshold.ts     (1 class per file)
    ├── LocalThreshold.ts
    ├── AdaptiveThreshold.ts
    └── __tests__/
        ├── GlobalThreshold.test.ts
        ├── LocalThreshold.test.ts
        └── AdaptiveThreshold.test.ts
```

## Required Steps When Porting/Updating Code

### Step 1: Create/Update TypeScript File
1. Create the TypeScript file with the same name as the Python file
2. Port the class/function with equivalent TypeScript implementation
3. Maintain method signatures and behavior as closely as possible
4. Add JSDoc comments referencing the Python source

Example:
```typescript
/**
 * Gaussian Blur filter processor.
 *
 * TypeScript port of src/processors/image/GaussianBlur.py
 * Maintains 1:1 correspondence with Python implementation.
 */
export class GaussianBlur extends Processor {
  // ... implementation
}
```

### Step 2: Create/Update Test File
1. Create a corresponding test file in `__tests__/` directory
2. Port test cases from Python tests (if they exist)
3. Add TypeScript-specific tests as needed
4. Ensure comprehensive coverage (aim for 100% for new code)

### Step 3: Update Exports in index.ts
1. Add/update export statements in `packages/core/src/index.ts`
2. Export individual classes, not barrel exports of combined files

Example:
```typescript
// Good - Individual exports
export { GaussianBlur } from './processors/image/GaussianBlur';
export { MedianBlur } from './processors/image/MedianBlur';

// Bad - Barrel export
export * from './processors/image/filters';
```

### Step 4: **CRITICAL - Update FILE_MAPPING.json**
This is a **MANDATORY** step that must not be skipped!

1. Open `/Users/udayraj.deshmukh/Personals/OMRChecker/FILE_MAPPING.json`
2. Find or create the mapping entry for the file
3. Update the following fields:

```json
{
  "python": "src/processors/image/GaussianBlur.py",
  "typescript": "omrchecker-js/packages/core/src/processors/image/GaussianBlur.ts",
  "status": "synced",  // Update from "not_started" or "partial"
  "phase": 1,
  "priority": "medium",
  "lastSyncedCommit": null,
  "lastPythonChange": null,
  "lastTypescriptChange": "2026-01-12T00:00:00Z",  // Update with current date
  "testFile": "omrchecker-js/packages/core/src/processors/image/__tests__/GaussianBlur.test.ts",
  "classes": [
    {
      "python": "GaussianBlurProcessor",
      "typescript": "GaussianBlur",
      "synced": true,  // Update to true when complete
      "methods": [
        {
          "python": "apply_on_image",
          "typescript": "process",
          "synced": true  // Update to true when ported
        }
      ]
    }
  ],
  "notes": "1:1 file mapping. Basic Gaussian blur filter."
}
```

4. Update the statistics section at the end of FILE_MAPPING.json:
```json
"statistics": {
  "total": 39,        // Increment if adding new mapping
  "synced": 9,        // Increment when marking as synced
  "partial": 3,       // Adjust accordingly
  "not_started": 27,  // Decrement when starting/completing
  "phase1": 31,
  "phase2": 4,
  "future": 3
}
```

### Step 5: Run Linter and Tests
1. Run TypeScript linter: `cd omrchecker-js && npm run lint`
2. Run TypeScript tests: `cd omrchecker-js/packages/core && npm test`
3. Run Python linter: `cd /path/to/OMRChecker && uv run ruff check .`
4. Run Python tests: `uv run pytest`

### Step 6: Verify Pre-commit Hooks
1. Ensure pre-commit hooks pass
2. The hook validates Python-TypeScript correspondence using FILE_MAPPING.json
3. Files marked as `phase: "future"` are automatically skipped

## Special Cases

### When Python has Multiple Classes in One File
If Python has multiple classes in one file (e.g., `strategies.py` with `GlobalThresholdStrategy`, `LocalThresholdStrategy`, `AdaptiveThresholdStrategy`):

1. **Split into separate TypeScript files** for better 1:1 class mapping
2. Create one file per class
3. Create one test file per class
4. Add **multiple entries** in FILE_MAPPING.json, all pointing to the same Python file but different TypeScript files

Example:
```json
{
  "python": "src/processors/threshold/strategies.py",
  "typescript": "omrchecker-js/packages/core/src/processors/threshold/GlobalThreshold.ts",
  "notes": "1:1 class mapping. In Python, all strategies are in strategies.py. In TypeScript, each strategy has its own file."
},
{
  "python": "src/processors/threshold/strategies.py",
  "typescript": "omrchecker-js/packages/core/src/processors/threshold/LocalThreshold.ts",
  "notes": "1:1 class mapping. Local threshold for per-question thresholding."
}
```

### When Adding Brand New Functionality
1. Add to both Python and TypeScript simultaneously if possible
2. If only adding to one codebase, add entry to FILE_MAPPING.json with:
   - `status: "not_started"` for the missing side
   - `phase: "future"` if not planned for immediate porting
3. Document in the `notes` field

## Automation Tools

### 1. File Mapping Generator
Run this to regenerate the mapping (updates paths, not status):
```bash
cd /Users/udayraj.deshmukh/Personals/OMRChecker
python code-browser/generate-mapping.py
```

### 2. Pre-commit Hook
The pre-commit hook at `scripts/hooks/validate_code_correspondence.py` automatically:
- Validates Python-TypeScript correspondence
- Checks FILE_MAPPING.json for consistency
- Skips files marked as `phase: "future"`

### 3. Code Browser
Use the code browser to review 1:1 mappings:
```bash
cd /Users/udayraj.deshmukh/Personals/OMRChecker/code-browser
python server.py
```

## Checklist for Every Port/Update

- [ ] Created/Updated TypeScript file with same name as Python
- [ ] Created/Updated corresponding test file
- [ ] Updated exports in `index.ts`
- [ ] **Updated FILE_MAPPING.json with correct status**
- [ ] **Updated statistics in FILE_MAPPING.json**
- [ ] Added JSDoc comments referencing Python source
- [ ] Ran TypeScript linter (no errors)
- [ ] Ran TypeScript tests (all passing)
- [ ] Ran Python linter (no new errors)
- [ ] Verified pre-commit hooks pass
- [ ] Documented any deviations in `notes` field

## Common Mistakes to Avoid

1. ❌ **Forgetting to update FILE_MAPPING.json** - This is the most critical mistake!
2. ❌ Creating combined/barrel export files (e.g., `filters.ts` with multiple processors)
3. ❌ Not creating corresponding test files
4. ❌ Not updating the statistics section in FILE_MAPPING.json
5. ❌ Leaving `status: "not_started"` after porting is complete
6. ❌ Not adding `testFile` field to FILE_MAPPING.json entries

## FILE_MAPPING.json Status Values

- **`not_started`**: TypeScript file doesn't exist or is empty
- **`partial`**: TypeScript file exists but is incomplete (missing methods/features)
- **`synced`**: TypeScript file is complete and matches Python functionality
- **`diverged`**: TypeScript and Python implementations have intentionally different approaches (document in `notes`)

## Priority Levels

- **`high`**: Core functionality, required for MVP
- **`medium`**: Important features, should be ported in Phase 1-2
- **`low`**: Nice-to-have features, can be deferred

## Phase Definitions

- **`1`**: Core Pipeline & Basic Processors
- **`2`**: Advanced Processors & Visualization
- **`future`**: ML Models & Advanced Features (not blocking)

## Questions?

If unsure about any step, refer to:
1. `TYPESCRIPT_1TO1_MAPPING.md` - Visual mapping table
2. `FILE_MAPPING.json` - Authoritative source of truth
3. Existing ported files as examples (e.g., `GaussianBlur.ts`)

---

**Remember: FILE_MAPPING.json is the single source of truth for tracking port progress. Always update it!**

