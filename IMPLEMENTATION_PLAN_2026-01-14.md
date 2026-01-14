# Implementation Plan - Python Then TypeScript

## 🎯 Strategy: Complete Python, Then Full TypeScript Port

**Your Decision**: Resume Python refactoring → Complete Phase 3 → Port full TypeScript infrastructure

This is the **optimal approach** for:
- ✅ Maintaining momentum
- ✅ Clean separation of work
- ✅ Proper TypeScript architecture
- ✅ Efficient batch porting

---

## 📋 Phase 3: Complete Python Refactoring

### File 1: CropOnPatchesCommon.py (Current Target) 🎯
**Status**: Analyzing for refactoring opportunities
**Size**: 378 lines
**Priority**: Medium (base class, used by CropOnDotLines & CropOnCustomMarkers)

**Potential Extractions**:
- Patch scanning utilities
- Point selection strategies
- Zone validation logic
- Common scan zone processing

**Estimated Time**: 2-3 hours
**Expected Reduction**: ~100-150 lines (25-40%)

### File 2: CropOnMarkers.py (Quick Review)
**Status**: Not started
**Size**: 33 lines
**Priority**: Low (simple delegator)

**Analysis Needed**: Check if it's just a delegator or needs work
**Estimated Time**: 15-30 minutes

### Phase 3 Completion Target
- **Files remaining**: 2/5 (40%)
- **Time estimate**: 2.5-3.5 hours total
- **Expected total reduction**: 500+ lines across Phase 3

---

## 📋 Phase 4: Full TypeScript Infrastructure Port

### Step 1: Utility Modules (~700 lines, 2-3 hours)

**1.1. warpStrategies.py → warpStrategies.ts**
- **Size**: ~400 lines
- **Priority**: HIGH (blocks WarpOnPointsCommon)
- **Content**: 4 warp strategies (Strategy Pattern)
  - PerspectiveTransform
  - HomographyTransform
  - AffineFourPoint
  - AffineMinAreaRect
- **Complexity**: Medium (OpenCV.js API translations needed)

**1.2. point_utils.py → pointUtils.ts**
- **Size**: ~300 lines
- **Priority**: HIGH (blocks WarpOnPointsCommon)
- **Content**: Point manipulation utilities
  - Point ordering/sorting
  - Distance calculations
  - Rectangle operations
- **Complexity**: Low (mostly math, no CV dependencies)

### Step 2: Base Classes (~894 lines, 3-4 hours)

**2.1. WarpOnPointsCommon.py → WarpOnPointsCommon.ts**
- **Size**: ~516 lines
- **Priority**: CRITICAL (blocks all crop processors)
- **Content**: Point-based warping/transformation base class
  - Uses warp_strategies and point_utils
  - Core warping logic
  - Point validation
- **Complexity**: High (core infrastructure)

**2.2. CropOnPatchesCommon.py → CropOnPatchesCommon.ts**
- **Size**: ~378 lines (may be reduced after refactoring)
- **Priority**: HIGH (blocks CropOnDotLines, CropOnCustomMarkers)
- **Content**: Patch-based detection base class
  - Scan zone management
  - Point selection
  - Zone validation
- **Complexity**: Medium-High

### Step 3: Concrete Processors (~750 lines, 2-3 hours)

**3.1. Update CropPage.ts**
- **Status**: Exists, verify with real WarpOnPointsCommon base
- **Size**: ~136 lines (already refactored in Python)
- **Priority**: HIGH (validate base class works)

**3.2. CropOnDotLines.ts**
- **Status**: New file (CREATE)
- **Size**: ~357 lines (refactored Python)
- **Priority**: HIGH (recently refactored)
- **Uses**: dotLineDetection.ts (already ported)

**3.3. CropOnCustomMarkers.ts**
- **Status**: May exist, needs update
- **Size**: ~340 lines (refactored Python)
- **Priority**: HIGH (recently refactored)
- **Uses**: markerDetection.ts (already ported)

**3.4. CropOnMarkers.ts**
- **Status**: Exists?
- **Size**: ~33 lines
- **Priority**: LOW (simple delegator)

---

## 📊 Time Estimates

### Python Refactoring (Phase 3)
| Task | Time | Status |
|------|------|--------|
| Analyze CropOnPatchesCommon | 30 min | In Progress |
| Refactor CropOnPatchesCommon | 2-3 hours | Pending |
| Review CropOnMarkers | 15-30 min | Pending |
| **Total Phase 3** | **2.5-3.5 hours** | **60% Done** |

### TypeScript Port (Phase 4)
| Task | Time | Lines |
|------|------|-------|
| warpStrategies.ts | 2-3 hours | 400 |
| pointUtils.ts | 1 hour | 300 |
| WarpOnPointsCommon.ts | 2-3 hours | 516 |
| CropOnPatchesCommon.ts | 2-3 hours | 378 |
| Crop processors | 2-3 hours | 750 |
| Testing & fixes | 1-2 hours | - |
| **Total Phase 4** | **10-15 hours** | **~2,344 lines** |

### Grand Total
- **Python**: 2.5-3.5 hours remaining
- **TypeScript**: 10-15 hours
- **Overall**: 12.5-18.5 hours to complete both

---

## 🎯 Milestones

### Milestone 1: Python Phase 3 Complete ✅
- All image processors refactored
- ~500+ lines reduced
- 100% test coverage maintained
- Ready for TypeScript batch port

### Milestone 2: TypeScript Utilities ✅
- warpStrategies.ts complete
- pointUtils.ts complete
- Foundation ready for base classes

### Milestone 3: TypeScript Base Classes ✅
- WarpOnPointsCommon.ts complete
- CropOnPatchesCommon.ts complete
- All crop processors can extend proper base

### Milestone 4: TypeScript Processors ✅
- All crop processors ported
- 1:1 Python ↔ TypeScript mapping
- Full test coverage
- **PROJECT PHASE 1 COMPLETE** 🎉

---

## 🚀 Starting Now: CropOnPatchesCommon Analysis

Let's begin by analyzing `CropOnPatchesCommon.py` to identify refactoring opportunities...

---

**Plan Created**: January 14, 2026
**Status**: Starting Python Phase 3 refactoring
**Next**: Analyze CropOnPatchesCommon.py

