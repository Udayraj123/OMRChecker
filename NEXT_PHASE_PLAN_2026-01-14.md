# 🎯 Next Phase Implementation Plan
**Date:** January 14, 2026
**Status:** Python refactoring in progress, TypeScript port 72% Phase 1 complete

---

## 📊 Current State Summary

### Python Codebase
✅ **Completed:**
- Bubble detection refactoring (57% code reduction)
- Strategy patterns for threshold calculation
- Pydantic models for type safety
- Repository patterns implemented

🔄 **In Progress:**
- Image processor refactoring (CropPage, CropOnCustomMarkers, WarpOnPointsCommon)
- New extraction modules: marker_detection.py, dot_line_detection.py, page_detection.py
- DRY improvements across processors

### TypeScript Port (feature/js-port branch)
✅ **Completed (26/36 files - 72%):**
- Core pipeline architecture
- Base processor system
- Image preprocessors (5 filters)
- Threshold strategies (3 strategies)
- Utility modules (ImageUtils, geometry, math, logger, csv, file, drawing)
- Schema validation (config, template, evaluation)
- Template system (loader, types)
- Detection processors (SimpleBubbleDetector)
- Evaluation processor

⏳ **Partial (7 files):**
- CropPage (placeholder only, needs sync with Python refactoring)
- CropOnMarkers (not started)
- Alignment system (missing SIFT features)
- Detection system (SimpleBubbleDetector is basic version)

❌ **Not Started (3 files):**
- Barcode processor (phase 2)
- ML detector (future)
- OCR processor (future)

---

## 🚨 Immediate Blocker: Sync Issue

**Problem:** CropPage.py was refactored but CropPage.ts not updated, blocking commit.

**Solutions:**

### Option A: Skip Sync Temporarily (Fastest)
```bash
git commit --no-verify -m "refactor: extract marker/page detection modules"
```
**Pros:** Unblocks you immediately
**Cons:** Creates temporary drift between Python/TS
**When to use:** You want to finish Python refactoring first, then sync TS in batch

### Option B: Update TypeScript Now (Maintains Sync)
Port the new Python changes to TypeScript:
- Create `markerDetection.ts` (I see you started this - untracked file)
- Create `pageDetection.ts`
- Update `CropPage.ts` to use new modules
- Commit everything together

**Pros:** Keeps codebases in perfect sync
**Cons:** Takes more time now
**When to use:** You want to maintain strict 1:1 parity

---

## 🎯 Recommended Path Forward

### **Phase A: Complete Python Refactoring** (1-2 days)
**Goal:** Stabilize Python codebase before syncing to TypeScript

1. ✅ Extract detection modules (marker, dot/line, page) - DONE
2. 🔄 Complete CropPage/CropOnCustomMarkers refactoring - IN PROGRESS
3. ⏳ Test all samples to ensure no regressions
4. ⏳ Update documentation
5. ⏳ Commit all Python changes (use `--no-verify` if needed)

**Deliverable:** Clean, refactored Python codebase with DRY modules

---

### **Phase B: Sync TypeScript Port** (2-3 days)
**Goal:** Bring TypeScript up to date with Python refactoring

#### B1. Port New Detection Modules (4-6 hours)
Files to create:
- `markerDetection.ts` (you started this ✅)
- `pageDetection.ts` (port find_page_contour_and_corners)
- `dotLineDetection.ts`

**Approach:**
- Port Python functions 1:1
- Use OpenCV.js equivalents
- Write tests for each module
- Update FILE_MAPPING.json

#### B2. Update CropPage.ts (2-3 hours)
- Replace placeholder with real implementation
- Use new pageDetection module
- Implement perspective transform
- Add tests

#### B3. Port CropOnMarkers.ts (3-4 hours)
- Use new markerDetection module
- Implement marker-based cropping
- Add tests

**Deliverable:** TypeScript fully synced with Python refactoring

---

### **Phase C: Complete Bubble Detection System** (3-4 days)
**Goal:** Get a working end-to-end demo

#### C1. Enhance SimpleBubbleDetector
Current state: Basic threshold-based detection
Needed:
- Multiple threshold passes
- Confidence scoring
- Multi-mark detection
- Empty/multiple answer handling

#### C2. Port ReadOMRProcessor Fully
Current: Simplified version
Needed:
- Template file runner integration
- Field block iteration
- Response aggregation
- CSV output generation

#### C3. Create Demo Application
**Features:**
- Upload OMR image
- Select/upload template
- Process and show results
- Visualize detected bubbles
- Export answers to CSV

**Tech Stack:**
- React + Vite
- OpenCV.js for processing
- Canvas for visualization

**Deliverable:** Working browser-based OMR checker demo

---

### **Phase D: Advanced Features** (Future)
Lower priority items to tackle after core system works:

1. **Better Alignment** (1 week)
   - Port ORB feature matching (SIFT not available in browser)
   - Implement alternative alignment methods
   - Or: Simple template matching for basic cases

2. **Barcode Detection** (2-3 days)
   - Port using @zxing/library
   - Integrate with detection pipeline

3. **ML-based Detection** (2-3 weeks)
   - Export YOLO model to ONNX
   - Use ONNX Runtime Web
   - Fallback to threshold detection

4. **OCR Support** (1-2 weeks)
   - Integrate Tesseract.js
   - Add text field types to templates

---

## 📋 Detailed Action Items

### This Week (Jan 14-20)

#### Day 1-2: Finish Python Refactoring
- [ ] Complete CropPage refactoring
- [ ] Complete CropOnCustomMarkers refactoring
- [ ] Test with all sample templates
- [ ] Fix any regressions
- [ ] Commit Python changes (use `--no-verify`)
- [ ] Update Python documentation

#### Day 3-4: Sync TypeScript
- [ ] Port markerDetection module (finish started file)
- [ ] Port pageDetection module
- [ ] Port dotLineDetection module
- [ ] Update CropPage.ts
- [ ] Write tests for all new modules
- [ ] Update FILE_MAPPING.json

#### Day 5: Enhanced Detection
- [ ] Improve SimpleBubbleDetector
- [ ] Add confidence scoring
- [ ] Handle edge cases (multiple marks, no marks)
- [ ] Add more tests

### Next Week (Jan 21-27)

#### Full Detection System
- [ ] Port full ReadOMRProcessor
- [ ] Implement template file runner
- [ ] Add field block processing
- [ ] Implement response aggregation
- [ ] Add CSV export

#### Demo Application
- [ ] Set up React project structure
- [ ] Create image upload component
- [ ] Create template selector
- [ ] Add processing pipeline integration
- [ ] Create results visualization
- [ ] Add export functionality

---

## 🎯 Success Criteria

### Phase A Complete When:
- [ ] All Python processors refactored and working
- [ ] All samples process without errors
- [ ] Code passes all tests and linting
- [ ] Documentation updated

### Phase B Complete When:
- [ ] TypeScript matches Python functionality
- [ ] All new modules ported and tested
- [ ] FILE_MAPPING.json shows 90%+ sync
- [ ] No sync warnings on commit

### Phase C Complete When:
- [ ] Demo app can process OMR images
- [ ] Bubble detection accuracy matches Python
- [ ] Results can be exported
- [ ] Basic error handling in place

---

## 📊 Updated FILE_MAPPING.json Stats

```json
{
  "total": 36,
  "synced": 26,      // 72% complete
  "partial": 7,      // 19% in progress
  "not_started": 3,  // 8% remaining
  "phase1": 31,
  "phase2": 3,
  "future": 2
}
```

**Target for end of Phase B:** 33/36 synced (92%)

---

## 🚀 Quick Commands

### Commit Python changes (bypass sync check):
```bash
git commit --no-verify -m "refactor: extract detection modules and improve DRY"
```

### Run Python tests:
```bash
uv run pytest tests/processors/image/
```

### Run TypeScript tests:
```bash
cd omrchecker-js
pnpm test
```

### Check sync status:
```bash
uv run python scripts/hooks/validate_code_correspondence.py
```

### Update FILE_MAPPING:
```bash
# Edit FILE_MAPPING.json manually
# Or use change propagation tool:
cd change-propagation-tool && pnpm run dev
```

---

## 💡 Recommendations

### For This Session:

**If you have 1-2 hours:**
→ Choose Option A (commit with --no-verify) and continue Python refactoring

**If you have 3-4 hours:**
→ Choose Option B (sync TypeScript now) to maintain strict parity

**If you have 6+ hours:**
→ Complete Phase B1-B2 (port all new modules + update CropPage)

### Overall Strategy:

1. **Short term:** Complete Python refactoring → sync TypeScript
2. **Medium term:** Enhance detection system → build demo
3. **Long term:** Add ML models, OCR, advanced alignment

The key is: **Python is the source of truth.** Stabilize it first, then sync TypeScript.

---

## 📝 Notes

- You have untracked `markerDetection.ts` - good start! This suggests you were thinking Option B
- The marker_detection.py Python module is new and well-structured
- Python refactoring is introducing better separation of concerns
- TypeScript should mirror this improved architecture

**My recommendation:** Finish the Python refactoring session, commit with `--no-verify`, then do a focused TypeScript sync session next.

---

**Status:** Ready to proceed with your choice of Option A or B
**Next Update:** After Python refactoring complete or TypeScript sync done

