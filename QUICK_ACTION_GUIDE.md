# Quick Action Guide - What to Do Next

## ✅ What Just Happened

**CropOnDotLines** refactored: **528 → 357 lines (-32%, 171 lines eliminated)**

- Used extracted `dot_line_detection` module
- Removed 105-line detection method
- Simplified to thin wrappers
- 0 lint errors, ready to commit

---

## 🎯 Choose Your Path

### Path A: Keep Refactoring Python 🐍
```bash
# Continue with CropOnPatchesCommon.py refactoring
# (This will increase Python/TypeScript drift)
```

### Path B: Sync TypeScript Now ⚡ **(RECOMMENDED)**
```bash
# 1. Commit current work
git add src/processors/image/CropOnDotLines.py FILE_MAPPING.json REFACTORING_*.md
git commit -m "refactor: extract detection from CropOnDotLines (-32% LOC)"

# 2. Update TypeScript
# Edit: omrchecker-js/packages/core/src/processors/image/CropOnDotLines.ts
# - Import: detect_dot_corners, detect_line_corners_and_edges
# - Remove: inline detection code (like Python refactoring)
# - Fix: TypeScript compilation errors

# 3. Commit TypeScript sync
git add omrchecker-js/
git commit -m "sync: update CropOnDotLines.ts to match Python"
```

**Why B?** Git pre-commit hooks expect Python ↔ TypeScript sync. Next Python commit may fail validation.

---

## 📊 Current Status

**Python Refactoring**:
- Phase 1 & 2: ✅ 100% done
- Phase 3: 🔄 60% done (3/5 files)
- Next file: CropOnPatchesCommon.py (378 lines)

**TypeScript Port**:
- Overall: 71% synced (29/41 files)
- Phase 1: 81% complete
- **Needs sync**: CropOnDotLines, CropOnCustomMarkers

**Files Ready to Commit**:
```
Modified:   src/processors/image/CropOnDotLines.py
Modified:   FILE_MAPPING.json
Untracked:  REFACTORING_*.md (3 files)
```

---

## 📚 Documentation Created

1. **REFACTORING_PHASE3_SUMMARY.md**: Complete phase 3 report
2. **REFACTORING_DECISION_GUIDE.md**: Detailed options analysis
3. **REFACTORING_SESSION_COMPLETE_2026-01-14.md**: Full session summary

---

## 🎯 Recommendation

**Do Path B (Sync TypeScript)** to maintain 1:1 mapping and avoid git hook issues. Takes ~3-4 hours.

After TypeScript sync, you can resume Python refactoring with confidence.

---

**Date**: January 14, 2026
**Status**: CropOnDotLines refactoring complete, awaiting commit + decision

