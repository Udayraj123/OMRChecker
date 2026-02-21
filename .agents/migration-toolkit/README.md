# OMRChecker Migration Toolkit

Automated tools to accelerate Python → TypeScript migration using the documented migration skill.

## Overview

This toolkit provides 4 automated generators that translate Python OMRChecker to TypeScript for the browser:

1. **Project Setup** - Initialize omrchecker-js with proper structure and configuration
2. **Interface Generator** - Convert Python dataclasses/Pydantic to TypeScript interfaces
3. **Test Scaffolding** - Translate pytest structure to Vitest
4. **File Migration** - Automated Python → TypeScript translation using Claude API

## Automation Level

- **Automated (70-80%)**: Project setup, interfaces, type definitions, test structure, utilities, **OpenCV.js operations** (API nearly identical!)
- **Manual (20-30%)**: Memory management polish, Web Workers, Canvas debug UI, ML models (optional, skipped)

**Expected Timeline**:
- Toolkit execution: ~30 minutes
- Manual refinement: 3-5 days (mostly memory cleanup and Web Workers)
- **Total**: 1 week vs 6-8 weeks manual

**Key Insight**: OpenCV.js API is nearly identical to Python cv2 - the migration script handles ~90% automatically!

## Quick Start

```bash
# 1. Setup project structure
cd .agents/migration-toolkit
./1-setup-project.sh

# 2. Generate TypeScript interfaces
node 2-generate-interfaces.js

# 3. Generate test scaffolding
node 3-generate-tests.js

# 4. Migrate Python files (requires Claude API key)
export ANTHROPIC_API_KEY="your-key-here"
node 4-migrate-files.js
```

## Tool Details

### 1. Project Setup (`1-setup-project.sh`)

Creates omrchecker-js project structure:

```
omrchecker-js/
├── src/
│   ├── core/              # Core entities (Template, Field, etc.)
│   ├── processors/        # Image processing pipeline
│   ├── utils/             # Utilities (logger, validation, etc.)
│   ├── schemas/           # JSON schemas and validation
│   └── workers/           # Web Workers for parallel processing
├── tests/                 # Vitest test structure
├── public/                # Static assets
├── package.json
├── tsconfig.json
├── vite.config.ts
└── vitest.config.ts
```

Generates:
- `package.json` with all browser dependencies
- `tsconfig.json` for TypeScript configuration
- `vite.config.ts` for build configuration
- `vitest.config.ts` for testing
- `.gitignore` with appropriate rules

### 2. Interface Generator (`2-generate-interfaces.js`)

Scans Python codebase and generates TypeScript interfaces:

**Input**: Python dataclasses, Pydantic models, exception types
**Output**: TypeScript interfaces in `omrchecker-js/src/types/`

Example transformation:
```python
# Python (src/schemas/models/template.py)
@dataclass
class Template:
    path: Path
    field_blocks: dict[str, FieldBlock]
    pre_processors: list[str]
    tuning_config: Config
```

```typescript
// TypeScript (omrchecker-js/src/types/template.ts)
export interface Template {
  path: string;
  fieldBlocks: Record<string, FieldBlock>;
  preProcessors: string[];
  tuningConfig: Config;
}
```

Handles:
- Dataclass → interface conversion
- Pydantic models → Zod schemas
- Exception classes → Error classes
- Type mapping (Path → string, dict → Record, etc.)

### 3. Test Scaffolding (`3-generate-tests.js`)

Translates pytest structure to Vitest:

**Input**: `src/tests/` pytest structure
**Output**: `omrchecker-js/tests/` Vitest structure

Creates:
- Test file structure mirroring Python tests
- Vitest setup files (`vitest.config.ts`, `setup.ts`)
- Fixture equivalents in JavaScript
- Image snapshot testing setup (pixelmatch)

Example:
```python
# Python (src/tests/processors/test_alignment.py)
def test_sift_alignment(mock_template, minimal_args):
    result = align_image(image, template, "SIFT")
    assert result is not None
```

```typescript
// TypeScript (omrchecker-js/tests/processors/alignment.test.ts)
import { describe, it, expect } from 'vitest';
import { alignImage } from '@/processors/alignment';

describe('SIFT Alignment', () => {
  it('should align image using SIFT', () => {
    const result = alignImage(image, template, 'SIFT');
    expect(result).not.toBeNull();
  });
});
```

### 4. File Migration (`4-migrate-files.js`)

Uses Claude API to translate Python files to TypeScript:

**Features**:
- File-by-file translation with migration skill context
- Parallel processing (configurable workers)
- Resume capability (tracks completed files)
- Progress tracking with detailed logs
- Error handling and retry logic

**Configuration** (`migration-config.json`):
```json
{
  "anthropicApiKey": "env:ANTHROPIC_API_KEY",
  "model": "claude-sonnet-4.5",
  "maxConcurrent": 5,
  "sourceRoot": "../../src",
  "targetRoot": "../../omrchecker-js/src",
  "skipPatterns": [
    "*/tests/*",
    "*/__pycache__/*",
    "*.pyc"
  ],
  "migrationSkillPath": "../skills/omrchecker-migration-skill"
}
```

**Usage**:
```bash
# Migrate all files
node 4-migrate-files.js

# Migrate specific directory
node 4-migrate-files.js --dir processors/alignment

# Resume after interruption
node 4-migrate-files.js --resume

# Dry run (show what would be migrated)
node 4-migrate-files.js --dry-run
```

**Output**:
- TypeScript files in `omrchecker-js/src/`
- Migration log: `migration-log.json`
- Progress tracking: `migration-progress.json`

## Migration Strategy

The toolkit uses a **hybrid approach**:

### ✅ Automated (Generators)
- Project structure and configuration
- TypeScript interfaces and types
- Error classes (29 exception types)
- Configuration management (Zod schemas)
- Validation patterns (JSON Schema → Zod)
- Logging utilities (console-based)
- Test structure and fixtures
- Basic utility functions
- **OpenCV.js operations** (~90% identical to cv2, auto-migrated!)
- File handling (Browser File API wrappers)

### 🔧 Manual Polish Required (3-5 days)
- **Memory management** - Add .delete() calls for OpenCV.js Mat objects
- **Web Workers** - Parallel processing for image pipeline (optional optimization)
- **Canvas rendering** - Debug visualization system
- **Error handling** - try/finally blocks for cleanup
- **Performance optimization** - Memory limits, large image handling
- **Browser compatibility** - Test in Chrome, Firefox, Safari

### ⏭️ Skipped (Optional Features)
- **ML models** - YOLO, OCR, Barcode detection (dummy placeholders created)
- Can be added later if needed with TensorFlow.js, Tesseract.js, @zxing/library

## Migration Skill Integration

All generators reference the migration skill at `.agents/skills/omrchecker-migration-skill/`:

- **Boundaries**: `core/boundaries.md` - What to migrate, what to skip
- **Foundation**: `modules/foundation/` - Error handling, logging, testing patterns
- **Domain**: `modules/domain/` - Core entities, flows, business logic
- **Technical**: `modules/technical/` - OpenCV, NumPy, filesystem patterns
- **Migration Context**: `modules/migration/` - Library mappings, browser adaptations

Generators use this documentation to:
1. Understand Python code structure
2. Apply correct browser equivalents
3. Skip browser-incompatible features
4. Preserve business logic and edge cases

## Expected Output

After running all 4 tools:

```
omrchecker-js/
├── src/
│   ├── types/           # 50+ TypeScript interfaces (automated)
│   ├── core/            # Template, Field, Config, etc. (automated scaffold)
│   ├── processors/      # Pipeline processors (automated scaffold)
│   ├── utils/           # Logger, validation, etc. (80% automated)
│   ├── schemas/         # Zod schemas (automated)
│   ├── exceptions/      # Error classes (automated)
│   └── workers/         # Web Worker scaffold (manual)
├── tests/               # Vitest structure (automated)
├── 172 .ts files        # From 172 .py files (70% ready, 30% needs refinement)
└── Configuration files  # package.json, tsconfig, etc. (100% automated)
```

## Validation Checklist

After running toolkit, validate:

- [ ] All TypeScript files compile without errors
- [ ] All interfaces exported and imported correctly
- [ ] Vitest tests run (may fail, structure should be correct)
- [ ] OpenCV.js integration loads correctly
- [ ] Web Workers communicate properly
- [ ] File API handles image uploads
- [ ] Zod validation matches Python JSON Schema
- [ ] Error handling preserves all 29 exception types
- [ ] Logger levels configurable like Python version
- [ ] Memory usage reasonable for browser (test with 10MB+ images)

## Troubleshooting

### Generator Fails
- Check Node.js version (18+ required)
- Verify migration skill path is correct
- Review migration-log.json for errors

### Claude API Errors
- Verify ANTHROPIC_API_KEY is set
- Check API rate limits (reduce maxConcurrent)
- Review migration-progress.json for resume point

### TypeScript Compilation Errors
- Expected: Complex types may need manual refinement
- Focus on interface correctness first
- Implement stubs for complex logic

### Test Failures
- Expected: Tests scaffold structure, not implementation
- Implement missing functions first
- Add fixtures as needed

## Next Steps

1. **Run toolkit** (30 minutes)
2. **Review generated code** (1-2 hours)
3. **Implement complex logic** (1-2 weeks):
   - OpenCV.js operations
   - Web Workers pipeline
   - ML model integration
   - Canvas debug system
4. **Test with sample OMR sheets** (2-3 days)
5. **UI integration** (if needed, 3-5 days)
6. **Performance optimization** (2-3 days)

## Support

- Migration skill documentation: `.agents/skills/omrchecker-migration-skill/`
- Python codebase: `../../src/`
- Browser compatibility: `modules/migration/compatibility.md`
- Technical patterns: `modules/technical/`

## License

Same as OMRChecker (GPLv3)
