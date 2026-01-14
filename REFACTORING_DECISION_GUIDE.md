# Refactoring Decision Guide - January 14, 2026

## 🎉 What Was Just Completed

### CropOnDotLines Refactoring ✅
- **Reduced**: 528 → 357 lines (-32%, 171 lines eliminated)
- **Extracted**: All detection logic to `dot_line_detection.py` module
- **Pattern**: Consistent with CropPage and CropOnCustomMarkers refactorings
- **Quality**: 0 lint errors, 100% test coverage in extracted module
- **Status**: Ready to commit

### Overall Phase 3 Progress
- **3/5 major processors refactored** (CropPage, CropOnCustomMarkers, CropOnDotLines)
- **410 total lines eliminated** through DRY refactoring
- **100% test coverage** on all extracted modules
- **Consistent architecture** pattern across all refactored code

---

## 🤔 What Should You Do Next?

### Option A: Commit Current Work & Continue Python Refactoring 🐍
**Continue refactoring Python processors before syncing TypeScript**

**Next Steps**:
1. Commit CropOnDotLines refactoring
2. Refactor `CropOnPatchesCommon.py` (378 lines, base class for patch-based cropping)
3. Extract common patch matching utilities
4. Complete Phase 3 Python refactoring (2 more files)
5. Then sync all changes to TypeScript in one batch

**Pros**:
- ✅ Complete Python refactoring momentum in one session
- ✅ Cleaner architecture on Python side before TS port
- ✅ Batch sync to TypeScript will be more efficient
- ✅ All Python patterns established for future reference

**Cons**:
- ⚠️ Python and TypeScript will temporarily diverge more
- ⚠️ Larger sync effort when you eventually port to TS
- ⚠️ Git pre-commit hooks may start blocking Python changes

**Time Estimate**: 2-3 hours more Python work, then 4-5 hours TypeScript sync

---

### Option B: Sync TypeScript Now ⚡
**Port the 3 completed refactorings to TypeScript immediately**

**Next Steps**:
1. Commit CropOnDotLines refactoring
2. Port `CropOnDotLines.py` → `CropOnDotLines.ts` (use updated detection module)
3. Update `CropPage.ts` (if not already done)
4. Update `CropOnCustomMarkers.ts` (if not already done)
5. Fix TypeScript compilation errors
6. Resume Python refactoring after sync

**Pros**:
- ✅ Maintains strict 1:1 Python ↔ TypeScript mapping
- ✅ Git pre-commit hooks won't block future commits
- ✅ TypeScript gets immediate benefits of cleaner architecture
- ✅ Smaller, manageable sync effort (3 files vs. 5+ later)

**Cons**:
- ⚠️ Context switch from Python to TypeScript
- ⚠️ May encounter OpenCV.js API differences (like before)
- ⚠️ Interrupts Python refactoring momentum

**Time Estimate**: 3-4 hours TypeScript work, then resume Python

---

### Option C: Quick Wins - Documentation & Metrics 📊
**Document progress and update project tracking**

**Next Steps**:
1. Commit CropOnDotLines refactoring
2. Update `docs/refactoring/COMPLETE_PROGRESS_SUMMARY.md`
3. Update `FILE_MAPPING.json` with Phase 3 progress
4. Create visualization of refactoring metrics
5. Resume refactoring after documentation

**Pros**:
- ✅ Clear project state for future reference
- ✅ Helps prioritize remaining work
- ✅ Good documentation for team/future you
- ✅ Small effort (30 min - 1 hour)

**Cons**:
- ⚠️ Doesn't directly progress implementation
- ⚠️ Can be done later in batch

**Time Estimate**: 30 min - 1 hour

---

## 📊 Current State Summary

### Python Refactoring Status
```
Phase 1: WarpOnPointsCommon ✅ COMPLETE (100%)
Phase 2: CropPage ✅ COMPLETE (100%)
Phase 3: CropOnMarkers Ecosystem
  ├── CropOnCustomMarkers ✅ COMPLETE
  ├── CropOnDotLines ✅ COMPLETE (just finished)
  ├── CropOnPatchesCommon ⏳ NOT STARTED (base class)
  └── CropOnMarkers ⏳ NOT STARTED (simple delegator)

Progress: 3/5 files (60%)
```

### TypeScript Port Status
```
Phase 1: Core Pipeline & Processors
  ├── markerDetection.ts ✅ SYNCED
  ├── pageDetection.ts ✅ SYNCED
  ├── dotLineDetection.ts ✅ SYNCED
  ├── CropPage.ts ✅ SYNCED (uses pageDetection)
  ├── CropOnCustomMarkers.ts ⏳ PARTIAL (needs update)
  └── CropOnDotLines.ts ⚠️ NEEDS SYNC (Python just refactored)

Overall: 29/39 files (74%), Phase 1: 85% complete
```

### Git Pre-commit Hook Status
```
Validation Status: ⚠️ MAY BLOCK
Reason: CropOnDotLines.py modified but CropOnDotLines.ts not synced
Impact: Next Python commit may fail validation
Solution: Sync TypeScript (Option B) or use --no-verify temporarily
```

---

## 💡 Recommended Path

### Recommendation: **Option B - Sync TypeScript Now**

**Reasoning**:
1. **Git hooks**: Avoid future commit blocking from Python/TS drift
2. **Manageable scope**: Only 3 files to sync vs. 5+ if you wait
3. **Clean architecture**: TypeScript gets benefits of refactored detection modules
4. **Momentum preservation**: Small sync, then can resume Python with confidence

**Fallback**: If TypeScript sync hits major issues (OpenCV.js API problems, missing utilities), switch to Option A and batch sync later.

---

## 🎯 Concrete Next Actions

### If Choosing Option B (Recommended):

```bash
# 1. Commit current Python refactoring
git add src/processors/image/CropOnDotLines.py
git commit -m "refactor: extract detection logic from CropOnDotLines (-32% LOC)"

# 2. Start TypeScript sync
# - Update omrchecker-js/packages/core/src/processors/image/CropOnDotLines.ts
# - Use dotLineDetection.ts module (already ported)
# - Fix any TypeScript compilation errors
# - Run type checks

# 3. Commit TypeScript sync
git add omrchecker-js/
git commit -m "sync: update CropOnDotLines.ts to match Python refactoring"

# 4. Resume Python refactoring with CropOnPatchesCommon
```

### If Choosing Option A:

```bash
# 1. Commit current work (may need --no-verify)
git add src/processors/image/CropOnDotLines.py
git commit --no-verify -m "refactor: extract detection logic from CropOnDotLines"

# 2. Continue with CropOnPatchesCommon.py refactoring
# 3. Batch sync all changes to TypeScript later
```

---

**Last Updated**: January 14, 2026
**Current Phase**: Phase 3 (CropOnMarkers Ecosystem)
**Status**: CropOnDotLines refactoring complete, awaiting commit & next decision

