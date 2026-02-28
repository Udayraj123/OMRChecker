# Python to TypeScript Migration Skill

**Version**: 1.0.0  
**Type**: Migration Execution  
**Target**: OMRChecker TypeScript Port  
**Status**: Production Ready

---

## Purpose

This skill provides a standardized, step-by-step workflow for migrating Python files to TypeScript. It ensures consistency across multiple subagents working in parallel and maintains high quality standards.

---

## When to Use This Skill

**Use this skill when:**
- You are assigned a Python file to migrate to TypeScript
- You need to perform batch migration of multiple files
- You are working on Phase 1-11 of the migration plan
- You need to validate an existing migration

**Do NOT use for:**
- Files listed in `.ts-migration-exclude`
- Experimental features (src/processors/experimental/)
- ML training code
- CLI-only utilities

---

## Prerequisites

Before starting migration, verify:
1. ✅ Python file exists and is not excluded
2. ✅ You have your phase/agent assignment
3. ✅ Dependencies for this file have been migrated first
4. ✅ TypeScript project structure exists (omrchecker-js/)

---

## Migration Workflow

### Step 0: Get and Claim Task (Recommended for Parallel Work)

**Using migration_tasks.py orchestrator:**

```bash
# Get next available task
TASK_JSON=$(uv run scripts/migration_tasks.py next)
TASK_ID=$(echo $TASK_JSON | jq -r '.id')
PYTHON_FILE=$(echo $TASK_JSON | jq -r '.python_file')
TS_FILE=$(echo $TASK_JSON | jq -r '.typescript_file')

# Claim the task (prevents duplicate work)
uv run scripts/migration_tasks.py claim $TASK_ID --agent $(whoami)

# Or use your agent name
uv run scripts/migration_tasks.py claim $TASK_ID --agent Foundation-Alpha
```

**Alternative: Check assigned task list**
- See `.agents/SUBAGENT_TASKS.md` for your agent's assignments
- Pick next unclaimed task from your list

**Expected outcome**: Task claimed, other agents won't pick it up

---

### Step 1: Pre-Migration Checks

```bash
# Check if file is excluded
grep -F "<python_file>" .ts-migration-exclude

# Verify Python file exists
test -f <python_file> && echo "✅ File exists" || echo "❌ File not found"

# Find TypeScript target path from FILE_MAPPING.json
python3 -c "
import json
with open('FILE_MAPPING.json') as f:
    data = json.load(f)
for m in data['mappings']:
    if m['python'] == '<python_file>':
        print(m['typescript'])
        break
"
```

**Expected outcome**: File exists, not excluded, target path identified

---

### Step 2: Generate TypeScript Code

```bash
# Run migration script
uv run scripts/generate_ts_suggestions.py \\
  --file <python_file> \\
  --output <typescript_file>
```

**What this generates:**
- TypeScript class with proper structure
- Import statements from base classes
- Property declarations from `__init__`
- JSDoc comments from docstrings
- Header with migration notes
- TODO markers for manual review

**Example output**:
```typescript
/**
 * AUTO-GENERATED from Python source: src/utils/math.py
 * ...manual review notes...
 */

import { BaseClass } from './baseClass';

export class MathUtils extends BaseClass {
  private privateProperty: number;
  public publicProperty: string;
  
  constructor(...) {
    // ...
  }
  
  static distance(point1: [number, number], point2: [number, number]): number {
    // TODO: Implement
  }
}
```

---

### Step 3: Manual Review and Enhancement

Open the generated TypeScript file and complete these tasks:

#### 3.1 Fix Import Statements
- [ ] Verify all imports are correct
- [ ] Convert Python module paths to TypeScript relative paths
- [ ] Add OpenCV.js import if needed: `import cv from '@techstark/opencv-js'`
- [ ] Remove unused imports
- [ ] Add missing dependencies

**Example fixes:**
```typescript
// Generated (may be wrong):
import { BaseClass } from '../BaseClass';

// Fixed:
import { BaseClass } from './base';
import cv from '@techstark/opencv-js';
```

#### 3.2 Enhance Type Annotations
- [ ] Replace `any` with proper types where possible
- [ ] Add generic type parameters
- [ ] Use union types for optional values: `string | null`
- [ ] Add `readonly` for immutable properties
- [ ] Use `const` assertions where appropriate

**Example enhancements:**
```typescript
// Generated:
process(data: any): any {

// Enhanced:
process<T extends ImageData>(data: T): ProcessedResult<T> {
```

#### 3.3 Add OpenCV Memory Management
For files using OpenCV (cv.Mat):
- [ ] Wrap Mat operations in try/finally
- [ ] Call `.delete()` on all cv.Mat instances
- [ ] Use helper functions for cleanup

**Pattern:**
```typescript
processImage(input: cv.Mat): cv.Mat {
  const temp = new cv.Mat();
  const result = new cv.Mat();
  
  try {
    cv.GaussianBlur(input, temp, new cv.Size(5, 5), 0);
    cv.threshold(temp, result, 128, 255, cv.THRESH_BINARY);
    return result.clone();
  } finally {
    temp.delete();
    result.delete();
  }
}
```

#### 3.4 Review Property Declarations
- [ ] Check visibility modifiers (private/public/protected)
- [ ] Verify property types are correct
- [ ] Ensure constructor parameters match properties
- [ ] Add `readonly` for constants

#### 3.5 Address All TODO Comments
- [ ] Implement method bodies (copy logic from Python)
- [ ] Fix type annotations marked with TODO
- [ ] Resolve import TODOs
- [ ] Remove or complete initialization TODOs

---

### Step 4: Add Tests (if Python has tests)

If `src/tests/test_<module>.py` exists:

```bash
# Generate test stub
uv run scripts/generate_test_stub.py \\
  --python-test src/tests/test_<module>.py \\
  --output omrchecker-js/packages/core/src/__tests__/<module>.test.ts
```

Manual test migration:
- [ ] Convert pytest fixtures to Vitest/Jest equivalents
- [ ] Map `assert` to `expect().toBe/toEqual/etc`
- [ ] Update file paths for browser environment
- [ ] Mock OpenCV operations where needed
- [ ] Ensure all test cases covered

---

### Step 5: Validate Migration

```bash
# Run validation script
uv run scripts/validate_ts_migration.py \\
  --python-file <python_file> \\
  --typescript-file <typescript_file>
```

**Expected output:**
```
✅ TypeScript file exists
✅ Class structure matches (1 classes)
✅ Method counts match
⚠️  Type annotations: 8 'any' types (acceptable)
✅ Imports present (2 modules)
Score: 80.0%
```

**Fix any issues before proceeding:**
- ❌ Score < 70%: Fix critical issues
- ⚠️ Score 70-90%: Address warnings
- ✅ Score > 90%: Excellent!

---

### Step 6: TypeScript Compilation

```bash
# Navigate to TypeScript project
cd omrchecker-js/packages/core

# Run type checker
npm run typecheck
```

**Fix ALL compilation errors:**
- Type mismatches
- Missing imports
- Undefined references
- Syntax errors

**No `any` types without explicit justification!**

---

### Step 7: Update FILE_MAPPING.json

```bash
# Update mapping status
python3 -c "
import json
from datetime import datetime, UTC

with open('FILE_MAPPING.json') as f:
    data = json.load(f)

for m in data['mappings']:
    if m['python'] == '<python_file>':
        m['status'] = 'synced'
        m['lastTypescriptChange'] = datetime.now(UTC).isoformat()
        break

with open('FILE_MAPPING.json', 'w') as f:
    json.dump(data, f, indent=2)
"
```

---

### Step 8: Commit Changes

```bash
# Stage files
git add <typescript_file>
git add <test_file>  # if created
git add FILE_MAPPING.json

# Commit with standard message
git commit -m "feat(ts-migrate): migrate <module> module

Migrated from: <python_file>
Target: <typescript_file>
Migration completeness: <score>%
Phase: <phase_number>

Manual enhancements:
- Fixed import paths
- Enhanced type annotations
- Added OpenCV memory management
- Implemented method bodies

Known issues: <if any>

Co-Authored-By: Oz <oz-agent@warp.dev>"
```

---

### Step 9: Push and Report Progress

```bash
# Get validation score for tracking
SCORE=$(uv run scripts/validate_ts_migration.py \
  --python-file <python_file> \
  --typescript-file <typescript_file> \
  --json | jq -r '.score')

# Mark task as complete in orchestrator
uv run scripts/migration_tasks.py complete <task_id> --score $SCORE

# Push to remote
git push origin <branch-name>

# Check overall progress
uv run scripts/migration_tasks.py progress
```

**What this does**:
- Records completion in `.migration-tasks.jsonl`
- Makes task unavailable to other agents
- Tracks quality score for metrics
- Shows team progress

---

## Common Patterns

### Pattern 1: Snake Case → Camel Case
```python
# Python
def process_omr_response(self, response_data):
```
```typescript
// TypeScript
processOmrResponse(responseData: ResponseData): void {
```

### Pattern 2: Type Conversions
```python
# Python
def process(data: list[int]) -> dict[str, Any]:
```
```typescript
// TypeScript
process(data: number[]): Record<string, any> {
```

### Pattern 3: Property Declarations
```python
# Python
def __init__(self, threshold: int = 128):
    self.threshold = threshold
    self._private_value = 0
```
```typescript
// TypeScript
private _privateValue: number;
public threshold: number;

constructor(threshold: number = 128) {
  this.threshold = threshold;
  this._privateValue = 0;
}
```

### Pattern 4: Abstract Methods
```python
# Python
from abc import abstractmethod

@abstractmethod
def process(self) -> None:
    pass
```
```typescript
// TypeScript
abstract process(): void;
```

### Pattern 5: Static Methods
```python
# Python
@staticmethod
def helper(value: int) -> int:
    return value * 2
```
```typescript
// TypeScript
static helper(value: number): number {
  return value * 2;
}
```

---

## Quality Checklist

Before marking migration as complete:

- [ ] TypeScript file exists and compiles
- [ ] Validation score ≥ 70%
- [ ] All imports are correct
- [ ] Type annotations are meaningful (minimal `any`)
- [ ] OpenCV memory management added where needed
- [ ] Properties properly declared
- [ ] All TODO comments addressed
- [ ] Tests migrated (if applicable)
- [ ] FILE_MAPPING.json updated
- [ ] Changes committed with proper message
- [ ] Pushed to remote

---

## Troubleshooting

### Issue: Import paths are wrong
**Solution**: Use relative paths from TypeScript file location
```typescript
// If in: omrchecker-js/packages/core/src/utils/math.ts
// Importing from: omrchecker-js/packages/core/src/utils/geometry.ts
import { Point } from './geometry';

// Importing from: omrchecker-js/packages/core/src/processors/base.ts
import { Processor } from '../processors/base';
```

### Issue: Too many `any` types
**Solution**: Look up types in Python source:
- Function parameters with type hints → use those types
- Return values → infer from implementation
- Properties → check __init__ assignments

### Issue: Validation fails
**Solution**: Check specific failure:
- Class structure mismatch → ensure all classes present
- Method count mismatch → check for missing/extra methods
- No imports → add base class imports

### Issue: TypeScript compilation errors
**Solution**: Common fixes:
- `Cannot find module` → fix import path
- `Type X is not assignable to Y` → fix type annotation
- `Property X does not exist` → add property declaration

---

## Batch Migration

For migrating multiple files at once:

```bash
# Create task file (python|typescript|phase|priority per line)
cat > phase1-tasks.txt << 'EOF'
src/utils/math.py|omrchecker-js/packages/core/src/utils/math.ts|1|high
src/utils/geometry.py|omrchecker-js/packages/core/src/utils/geometry.ts|1|high
EOF

# Run batch migration
uv run scripts/batch_migrate.py \\
  --task-file phase1-tasks.txt

# Or migrate entire phase from FILE_MAPPING.json
uv run scripts/batch_migrate.py \\
  --from-mapping \\
  --phase 1 \\
  --status not_started
```

---

## Notes for Subagents

1. **Consistency is key**: Follow this workflow exactly for all migrations
2. **Quality over speed**: Take time to fix types and add proper error handling
3. **Communicate issues**: If you encounter problems, report them clearly
4. **Update progress**: Keep FILE_MAPPING.json and progress files current
5. **Test your changes**: Ensure TypeScript compiles before committing
6. **Document decisions**: Add comments explaining non-obvious choices

---

## Success Metrics

A successful migration has:
- ✅ Validation score ≥ 80%
- ✅ < 5 `any` types per file
- ✅ All imports resolve correctly
- ✅ TypeScript compiles with zero errors
- ✅ Proper memory management for OpenCV
- ✅ Complete test coverage (if tests exist)

---

## References

- **Validation Report**: `docs/SCRIPT_VALIDATION_REPORT.md`
- **Migration Plan**: TypeScript Migration Plan (Plan ID: 925703ad-51f4-46c8-80a5-308508fd06c3)
- **Pattern Library**: `CHANGE_PATTERNS.yaml`
- **File Mapping**: `FILE_MAPPING.json`
- **Exclusions**: `.ts-migration-exclude`

---

**Last Updated**: 2026-02-28  
**Maintainer**: Migration Team
