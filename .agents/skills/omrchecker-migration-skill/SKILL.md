# OMRChecker Migration Skill

**Version**: 1.0.0
**Type**: Migration Documentation
**Legacy System**: Python OMRChecker
**Target System**: JavaScript OMRChecker (Browser/Client-Side)
**Status**: In Progress

---

## Purpose

This skill documents the complete Python OMRChecker codebase for zero-edge-case-loss migration to JavaScript (browser/client-side). It preserves all domain knowledge, business logic, edge cases, and implementation details needed to build a functionally equivalent browser-based OMR checker.

---

## When to Use This Skill

**Trigger Patterns** (for automatic agent invocation):
- User mentions "migrate omrchecker", "port to javascript", "browser version"
- User asks about OMR detection algorithms, bubble detection logic
- User needs to understand image preprocessing, alignment strategies
- User wants to know evaluation/scoring logic
- User asks "how does Python OMRChecker handle X?" where X is any feature

**Use Cases**:
- Implementing JavaScript version of any OMRChecker feature
- Understanding edge cases in bubble detection
- Clarifying image preprocessing flows
- Mapping Python OpenCV to OpenCV.js equivalents
- Answering "how should I implement X in JavaScript?"

---

## Progressive Disclosure Rules

**Load Priority**:
1. **Always load first**: CONTEXT.md, core/quick-ref.md
2. **For specific feature questions**: Load relevant domain module (e.g., modules/domain/detection/bubbles-threshold/)
3. **For implementation details**: Load technical module (e.g., modules/technical/opencv/)
4. **For integration questions**: Load integration module (e.g., modules/integration/template-format/)
5. **For migration questions**: Load modules/migration/ files

**Context Management**:
- Start with CONTEXT.md + quick-ref.md (lightweight overview)
- Load specific modules only when needed for current question
- Each module is self-contained with code references
- Use code references to validate current state vs documentation

---

## Module Structure

```
.agents/skills/omrchecker-migration-skill/
├── SKILL.md                    # This file (orchestrator)
├── CONTEXT.md                  # Migration context and discovery answers
├── BUILD_CHECKLIST.md          # Progress tracking, resumable workflow
│
├── core/
│   ├── boundaries.md           # What OMRChecker does/doesn't do
│   └── quick-ref.md            # Quick reference for common operations
│
├── modules/
│   ├── foundation/             # Error handling, logging, testing (5 files)
│   ├── domain/                 # Business logic entities (45+ files)
│   │   ├── template/           # Template entity and flows
│   │   ├── field-block/        # Field block entity
│   │   ├── field/              # Field types (bubble, barcode, OCR)
│   │   ├── pipeline/           # Main processing pipeline
│   │   ├── preprocessing/      # Image preprocessing flows
│   │   ├── alignment/          # Image alignment strategies
│   │   ├── detection/          # Detection systems (bubble, barcode, OCR, ML)
│   │   ├── threshold/          # Thresholding strategies
│   │   ├── evaluation/         # Scoring and evaluation
│   │   ├── utils/              # Utility functions
│   │   ├── organization/       # File organization
│   │   └── visualization/      # Workflow tracking and visualization
│   ├── technical/              # Implementation patterns (10 files)
│   │   ├── opencv/             # OpenCV operations
│   │   ├── numpy/              # NumPy array operations
│   │   ├── schemas/            # Pydantic validation
│   │   ├── filesystem/         # File I/O patterns
│   │   ├── concurrency/        # Threading patterns
│   │   ├── caching/            # Caching strategies
│   │   ├── state/              # State management
│   │   ├── metrics/            # Metrics and statistics
│   │   ├── debugging/          # Debug system
│   │   └── error-recovery/     # Error recovery patterns
│   ├── integration/            # External interfaces (5 files)
│   │   ├── cli/                # CLI interface (SKIP for browser)
│   │   ├── file-io/            # File input/output
│   │   ├── template-format/    # template.json specification
│   │   ├── config-format/      # config.json specification
│   │   └── evaluation-format/  # evaluation.json specification
│   └── migration/              # Migration context (5 files)
│       ├── MIGRATION_CONTEXT.md
│       ├── browser-adaptations.md
│       ├── ml-model-migration.md
│       ├── performance.md
│       └── compatibility.md
```

---

## How to Query This Skill

### Example Queries and Load Patterns

**Q: "How does bubble detection work?"**
- Load: modules/domain/detection/bubbles-threshold/concept.md, flows.md
- Additional: modules/domain/threshold/ for thresholding strategies

**Q: "What edge cases exist for rotated images?"**
- Load: modules/domain/preprocessing/auto-rotate/flows.md, constraints.md
- Additional: modules/technical/opencv/opencv-operations.md for rotation details

**Q: "How should I implement template alignment in JavaScript?"**
- Load: modules/domain/alignment/concept.md, flows.md
- Additional: modules/domain/alignment/sift/, phase-correlation/, template-matching/
- Additional: modules/migration/browser-adaptations.md for OpenCV.js patterns

**Q: "What's the template.json format?"**
- Load: modules/integration/template-format/template-format.md

**Q: "How is multi-mark detection handled?"**
- Load: modules/domain/detection/bubbles-threshold/interpretation-pass/flows.md
- Additional: modules/domain/detection/bubbles-threshold/interpretation.md

**Q: "What SKIP items shouldn't be migrated?"**
- Load: modules/domain/training/SKIP.md
- Additional: modules/integration/cli/ (CLI-specific features)

---

## Guarantees

1. **Zero Edge Case Loss**: All flows documented, including edge cases
2. **Code References**: Every behavior linked to source code location
3. **AS-IS Documentation**: Documents what Python code DOES, not what it should do
4. **Browser Adaptation Notes**: Clear guidance for JavaScript/browser equivalents
5. **SKIP Documentation**: Explicitly marks features not needed in browser

---

## Status & Progress

See BUILD_CHECKLIST.md for current progress.

**Completed**: 0 / 75 tasks (0%)

**Current Phase**: Phase 0 (Documentation Setup)

**Last Updated**: 2026-02-20

---

## For Developers

When implementing JavaScript OMRChecker:

1. **Start here**: Read CONTEXT.md and core/quick-ref.md
2. **Find your feature**: Locate relevant domain module
3. **Check code references**: Validate against current Python code
4. **Review constraints**: Understand all edge cases and business rules
5. **Check migration notes**: Review browser adaptation guidance
6. **Implement**: Build JavaScript equivalent with zero edge case loss
7. **Validate**: Ensure all flows from Python version are covered

---

## Skill Metadata

**Author**: Migration Builder Agent
**Created**: 2026-02-20
**Legacy System**: Python 3.11+, ~39K LOC
**Target System**: Browser JavaScript (ES2020+), OpenCV.js, Canvas API
**Documentation Type**: AS-IS migration documentation
**Maintenance**: Auto-detect staleness via code references
