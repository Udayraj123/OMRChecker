# TypeScript Port Continuation Plan

**Date**: January 14, 2026
**Current Status**: Phase 3 Complete, Starting Phase 4

## Overview

Phase 3 successfully implemented the core processor hierarchy. Phase 4 focuses on completing the utility layer and finishing pending implementations.

## Current TODOs Analysis (27 instances)

### Category 1: Utility Functions (Priority 1 - HIGH)

#### 1.1 MathUtils - ✅ MOSTLY COMPLETE
**Status**: Has all required functions!
- ✅ `orderFourPoints()` - Line 112
- ✅ `getRectanglePointsFromBox()` - Line 191
- ✅ `getRectanglePoints()` - Line 206
- ✅ `selectEdgeFromRectangle()` - Line 222
- ✅ `shiftPointsFromOrigin()` - Line 54

**Action**: No work needed!

#### 1.2 ShapeUtils - ❌ NEEDS CREATION
**Missing Functions**:
- `extractImageFromZoneDescription(image, zoneDescription)` → Used in `CropOnPatchesCommon.ts:390`
- `computeScanZoneRectangle(zoneDescription, includeMargins)` → Used in `CropOnPatchesCommon.ts`

**Location**: `/omrchecker-js/packages/core/src/utils/shapes.ts`

**Effort**: 2-3 hours

#### 1.3 ImageUtils Extensions - ⚠️ PARTIAL
**Missing Functions**:
- `getControlDestinationPointsFromContour(contour, destinationLine, maxPoints)` → Used in `CropOnDotLines.ts:424`
- `normalize()` with erode-subtract support → Used in `CropOnCustomMarkers.ts:451`

**Location**: `/omrchecker-js/packages/core/src/utils/ImageUtils.ts`

**Effort**: 2-3 hours

#### 1.4 DrawingUtils - ❌ NEEDS IMPLEMENTATION
**Missing Functions**:
- `drawBoxDiagonal()` → Used for scan zone visualization
- `drawContour()` → Used for debug visualization
- `drawMatches()` → Used for showing match lines

**Location**: Already exists at `/omrchecker-js/packages/core/src/utils/drawing.ts`

**Action**: Check existing implementation and add missing functions

**Effort**: 2-3 hours

---

### Category 2: Configuration & Constants (Priority 2 - MEDIUM)

#### 2.1 WarpMethod Enum - ⚠️ DUPLICATE
**Issue**: `WarpMethod` defined in both:
- `WarpOnPointsCommon.ts:24` (local enum)
- `processors/constants.ts:87` (exported const)

**Action**: Remove local enum, import from constants

**Effort**: 15 minutes

#### 2.2 TARGET_ENDPOINTS_FOR_EDGES - ❌ MISSING
**Issue**: Needed for edge contours map in `CropOnPatchesCommon.ts:314`

**Action**: Add to `processors/constants.ts`

**Effort**: 30 minutes

---

### Category 3: Feature Implementations (Priority 3 - MEDIUM)

#### 3.1 CropPage Warping - ❌ INCOMPLETE
**Status**: Detection complete, warping TODO (Line 105)

**Needs**:
1. Calculate destination corners
2. Apply perspective transform or warp method
3. Handle edge-based warping for HOMOGRAPHY/REMAP

**Effort**: 3-4 hours

#### 3.2 CropOnCustomMarkers Image Loading - ❌ INCOMPLETE
**Status**: Warns and skips (Line 232)

**Needs**:
1. Implement actual image loading using `ImageUtils.loadImage()`
2. Handle file path resolution
3. Cache loaded images

**Effort**: 2-3 hours

#### 3.3 GridDataRemap Strategy - ❌ INCOMPLETE
**Status**: Falls back to perspective transform (Line 285)

**Needs**:
- JavaScript interpolation library or custom implementation
- Grid-based warping

**Effort**: 4-6 hours (Low priority - advanced feature)

#### 3.4 DocRefineRectify Strategy - ❌ INCOMPLETE
**Status**: Not implemented (Line 336)

**Needs**:
- Port rectify helper from Python
- Scanline-based warping

**Effort**: 4-6 hours (Low priority - specialized feature)

---

### Category 4: Debug & Visualization (Priority 4 - LOW)

#### 4.1 InteractionUtils.show() - ❌ MISSING
**Used in**: Multiple locations for interactive debugging

**Action**: Implement browser-compatible visualization (canvas overlay)

**Effort**: 3-4 hours

#### 4.2 appendSaveImage() - ❌ MISSING
**Used in**: `WarpOnPointsCommon.ts:512` for saving debug images

**Action**: Implement with download or display functionality

**Effort**: 2-3 hours

---

## Implementation Priority

### Phase 4A: Core Utilities (Week 1)
**Goal**: Complete essential utility functions

1. ✅ **ShapeUtils.ts** (2-3 hours)
   - `extractImageFromZoneDescription()`
   - `computeScanZoneRectangle()`

2. ✅ **ImageUtils extensions** (2-3 hours)
   - `getControlDestinationPointsFromContour()`
   - Enhanced `normalize()` with erode-subtract

3. ✅ **DrawingUtils additions** (2-3 hours)
   - Check existing functions
   - Add missing debug visualization methods

4. ✅ **Constants cleanup** (45 min)
   - Remove WarpMethod duplicate
   - Add TARGET_ENDPOINTS_FOR_EDGES

**Total Effort**: ~8-10 hours

---

### Phase 4B: Feature Completion (Week 2)
**Goal**: Complete processor implementations

5. ✅ **CropPage warping** (3-4 hours)
   - Implement full warping pipeline
   - Test with page detection

6. ✅ **CropOnCustomMarkers image loading** (2-3 hours)
   - Real image loading
   - File path resolution
   - Caching

**Total Effort**: ~6-7 hours

---

### Phase 4C: Advanced Features (Week 3+)
**Goal**: Optional enhancements

7. ⚠️ **Debug visualization** (3-4 hours)
   - InteractionUtils.show() for canvas
   - appendSaveImage() for downloads

8. ⚠️ **Advanced strategies** (8-12 hours)
   - GridDataRemap with interpolation
   - DocRefineRectify with scanlines

**Total Effort**: ~11-16 hours (OPTIONAL)

---

## Success Criteria

### Phase 4A Complete When:
- ✅ All shape manipulation utilities work
- ✅ Image processing utilities complete
- ✅ No critical TODOs remain in base classes
- ✅ Debug drawing works (basic level)

### Phase 4B Complete When:
- ✅ CropPage fully functional with warping
- ✅ CropOnCustomMarkers loads real images
- ✅ All HIGH priority processors working
- ✅ Can process real OMR sheets end-to-end

### Phase 4C Complete When:
- ✅ Interactive debugging works in browser
- ✅ All warping strategies implemented
- ✅ Demo application shows all features

---

## Next Actions

### Immediate (Today)
1. Create `ShapeUtils.ts` with zone utilities
2. Extend `ImageUtils.ts` with contour functions
3. Fix WarpMethod enum duplication
4. Add TARGET_ENDPOINTS_FOR_EDGES constant

### This Week
5. Complete CropPage warping
6. Implement real image loading
7. Test with sample OMR sheets

### Next Week
8. Add debug visualization
9. Create demo application
10. Documentation and examples

---

## Files to Create/Modify

### New Files:
- `utils/shapes.ts` ✨

### Files to Modify:
- `utils/ImageUtils.ts` (add 2 functions)
- `utils/drawing.ts` (check & add missing)
- `processors/constants.ts` (add constant)
- `processors/image/WarpOnPointsCommon.ts` (remove local enum)
- `processors/image/CropPage.ts` (implement warping)
- `processors/image/CropOnCustomMarkers.ts` (add image loading)

---

## Estimated Timeline

- **Phase 4A**: 8-10 hours (1-2 days)
- **Phase 4B**: 6-7 hours (1 day)
- **Phase 4C**: 11-16 hours (2-3 days) - OPTIONAL

**Total Core Work**: ~15-17 hours over 2-3 days
**With Optional**: ~26-33 hours over 4-6 days

---

## Status: READY TO PROCEED 🚀

All planning complete. Ready to start implementation!

