# OMRChecker - Agent-Ready Repository

**Repository**: OMRChecker
**Type**: Computer Vision / Image Processing / OMR Detection
**Language**: Python → JavaScript (Migration in Progress)
**Status**: Agent-Ready (Migration Documentation Phase)

---

## Repository Overview

OMRChecker is an open-source Optical Mark Recognition (OMR) system that evaluates OMR sheets fast and accurately using classical image processing with OpenCV. It supports scanned documents and mobile phone images, processing at 200+ OMRs per minute with ~100% accuracy on scanned sheets and ~90% on mobile images.

**Key Capabilities**:
- Bubble detection (threshold-based + ML)
- Barcode detection
- OCR (text recognition)
- Automatic rotation and alignment
- Template-driven layout configuration
- Evaluation and scoring
- Parallel processing
- Rich visual debugging

---

## Skills Index

### Migration Skills

#### 1. omrchecker-migration-skill
**Path**: `.agents/skills/omrchecker-migration-skill/`
**Type**: Migration Documentation
**Status**: ✅ COMPLETE (100% - 88/88 tasks)
**Purpose**: Documents the complete Python OMRChecker codebase for zero-edge-case-loss migration to JavaScript (browser/client-side).

**Trigger When**:
- User mentions "migrate", "port to javascript", "browser version", "omrchecker-js"
- User asks how any Python OMRChecker feature works
- User needs to implement a specific feature in JavaScript
- User asks about bubble detection, alignment, preprocessing, evaluation logic
- User queries "how does Python handle X?"

**Capabilities**:
- 88 documentation tasks covering all aspects of Python codebase (85 completed)
- Progressive disclosure (load only what's needed)
- Code references for validation
- SKIP documentation for features not needed in browser
- Browser adaptation guidance
- Complete TypeScript/JavaScript migration patterns
- OpenCV.js, TensorFlow.js, ONNX Runtime Web integration
- Web Workers, IndexedDB, Canvas API browser patterns

**Module Coverage** (✅ = Complete):
- ✅ Foundation (5/5): error handling, logging, testing, validation, configuration
- ✅ Domain - Entities (5/5): Template, FieldBlock, Field, ProcessingContext, Config
- ✅ Domain - Pipeline (8/8): main pipeline, preprocessing (AutoRotate, CropOnMarkers, CropPage, filters, warping), alignment (SIFT, Phase Correlation, Template Matching, Piecewise Affine, K-Nearest)
- ✅ Domain - Detection (14/14): ReadOMR orchestrator, bubble detection (threshold strategy, detection pass, interpretation pass, statistics, drawing, interpretation logic), barcode, OCR, ML bubble detector, ML field block detector, shift detection, detection fusion, STN module
- ✅ Domain - Thresholding (3/3): strategy pattern, global threshold, local threshold, adaptive threshold
- ✅ Domain - Evaluation (5/5): config, section marking, answer matcher, meta, config for set
- ✅ Domain - Utilities (8/8): image, geometry, drawing, file, CSV, math, parsing, serialization
- ✅ Domain - Visualization (4/4): file organization, workflow session, workflow tracker, HTML exporter
- ✅ Domain - Training (2/2): SKIP - training data collector, YOLO exporter (server-side only)
- ✅ Technical (10/10): OpenCV operations, NumPy arrays, Pydantic schemas, filesystem, concurrency, caching, state management, metrics, debugging, error recovery
- ✅ Integration (5/5): CLI interface, file I/O, template format, config format, evaluation format
- ✅ Migration Context (5/5): Python→JS mappings, browser adaptations, ML model migration, performance, compatibility matrix

**Entry Point**: `.agents/skills/omrchecker-migration-skill/SKILL.md`

---

## How to Use These Skills

### For Agents

When a user asks about OMRChecker:

1. **Check trigger patterns** in Skills Index above
2. **Load skill SKILL.md** to understand structure
3. **Use progressive disclosure**:
   - Always start with CONTEXT.md + core/quick-ref.md
   - Load specific modules only when needed
   - Avoid loading entire skill at once
4. **Reference code**: All behaviors linked to source code locations
5. **Check SKIP items**: Don't waste time on features marked SKIP

### For Developers

When implementing JavaScript OMRChecker:

1. **Find your feature** in the Skills Index above
2. **Navigate to skill path** (e.g., `.agents/skills/omrchecker-migration-skill/`)
3. **Read SKILL.md** for orientation and load rules
4. **Review CONTEXT.md** for migration context
5. **Locate relevant module** (domain/preprocessing/, domain/detection/, etc.)
6. **Read flows.md and constraints.md** for that module
7. **Check code references** to validate against current Python code
8. **Review browser-adaptations.md** for JavaScript guidance
9. **Implement with zero edge case loss**

---

## Migration Progress

**Current Phase**: ✅ COMPLETED
**Tasks Completed**: 88 / 88 (100%)
**Last Updated**: 2026-02-21

**Breakdown by Category**:
| Category | Completed | Total | % |
|----------|-----------|-------|---|
| Foundation | 5 | 5 | 100% |
| Domain | 55 | 55 | 100% |
| Technical | 10 | 10 | 100% |
| Integration | 5 | 5 | 100% |
| Migration Context | 5 | 5 | 100% |
| Documentation Setup | 5 | 5 | 100% |
| Finalization | 3 | 3 | 100% |
| **TOTAL** | **88** | **88** | **100%** |

**All Tasks Complete**:
✅ AGENTS.md updated with complete skill index
✅ Validation pass completed (155 docs, all code refs validated)
✅ Skill packaged and ready for use

**Checklist**: See `.agents/skills/omrchecker-migration-skill/BUILD_CHECKLIST.md`

---

## Repository Structure

```
OMRChecker/
├── src/                        # Python source code (legacy)
│   ├── cli/                    # CLI interface
│   ├── processors/             # Core processing modules
│   │   ├── alignment/          # Image alignment
│   │   ├── detection/          # Bubble/barcode/OCR detection
│   │   ├── evaluation/         # Scoring and evaluation
│   │   ├── image/              # Image preprocessing
│   │   └── template/           # Template management
│   ├── schemas/                # Pydantic schemas
│   └── utils/                  # Utility functions
│
├── omrchecker-js/              # JavaScript port (in progress)
│
├── .agents/                    # Agent-ready documentation
│   ├── AGENTS.md               # This file
│   └── skills/                 # Skills directory
│       └── omrchecker-migration-skill/  # Migration documentation
│
├── main.py                     # Entry point (Python)
├── README.md                   # User documentation
└── docs/                       # Additional documentation
```

---

## Quick Start

### Using the Migration Skill

**Scenario**: You want to implement bubble detection in JavaScript

```plaintext
1. Ask: "How does bubble detection work in Python OMRChecker?"
2. Agent loads: .agents/skills/omrchecker-migration-skill/modules/domain/detection/bubbles-threshold/
3. You read: concept.md, flows.md, constraints.md
4. Check: modules/migration/browser-adaptations.md for OpenCV.js guidance
5. Implement: JavaScript version with all edge cases covered
```

**Scenario**: You want to understand template alignment strategies

```plaintext
1. Ask: "What alignment strategies does OMRChecker use?"
2. Agent loads: .agents/skills/omrchecker-migration-skill/modules/domain/alignment/
3. You read: concept.md for overview, then flows.md for each strategy
4. Deep dive: sift/, phase-correlation/, template-matching/ subdirectories
5. Implement: Choose appropriate strategy for browser environment
```

---

## Development Workflow

### Resumable Migration

The migration documentation process is designed to be resumable:

1. **Check progress**: Read BUILD_CHECKLIST.md
2. **Pick next task**: Find first "pending" task
3. **Work on task**: Document the specific module
4. **Update checklist**: Mark task as complete with summary
5. **Clear context**: Avoid context overflow by working incrementally
6. **Resume later**: Next agent picks up from checklist

### Spawning Sub-Agents

For parallel migration work, spawn sub-agents per module:

```plaintext
Agent 1: Document preprocessing modules (Task 3.1-3.8)
Agent 2: Document alignment modules (Task 3.9-3.13)
Agent 3: Document detection modules (Task 4.1-4.14)
```

Each sub-agent:
- Loads only its assigned modules
- Updates BUILD_CHECKLIST.md with progress
- Works independently to avoid context conflicts

---

## Contact & Support

**Repository**: https://github.com/Udayraj123/OMRChecker
**Discord**: https://discord.gg/qFv2Vqf
**Documentation**: https://github.com/Udayraj123/OMRChecker/wiki

---

## Skill Maintenance

**Update Triggers**:
- Code references become stale (Python code changes)
- New features added to Python version
- Edge cases discovered during JavaScript implementation
- Browser compatibility issues require documentation updates

**Validation**:
- Code references point to existing source files
- All flows documented with test coverage
- SKIP items clearly justified
- Migration guidance tested in browser environment

---

**Last Updated**: 2026-02-20
**Agent-Ready Version**: 1.0.0
