# OMRChecker Demo - Testing Guide

## 🚀 Demo is Running!

**URL**: http://localhost:3001/

The demo is now live and ready to test!

---

## 🧪 Test Scenarios

### ✅ Test 1: Auto-Template Detection (Main Feature)

**Setup:**
```bash
# Create a test folder structure
mkdir -p ~/omr-test/scans
cd ~/omr-test

# Use the example template
cp /Users/udayraj.deshmukh/Personals/OMRChecker/omrchecker-js/examples/simple-template.json ./template.json

# Add some test images (you'll need actual OMR sheet images)
# Or use any sample images for testing
```

**Steps:**
1. Open http://localhost:3001/
2. Wait for "✅ OpenCV loaded successfully"
3. Click "📁 Or Select Folder"
4. Select the `~/omr-test` folder
5. **Expected**:
   - ✅ "📄 template.json (auto-detected)"
   - ✅ "10 fields | Found in: omr-test"
   - ✅ Images found message

**Result**: Template automatically loaded! ✨

---

### ✅ Test 2: Manual Template Upload

**Steps:**
1. Open http://localhost:3001/
2. Click "Choose template.json"
3. Select: `/Users/udayraj.deshmukh/Personals/OMRChecker/omrchecker-js/examples/simple-template.json`
4. **Expected**: "✅ Template loaded: 10 fields"

---

### ✅ Test 3: Single Image Detection

**Steps:**
1. Load template (auto or manual)
2. Click "Choose image file(s)"
3. Select a single OMR sheet image
4. Click "🔍 Detect Bubbles"
5. **Expected**:
   - Processing spinner
   - Results visualization
   - Statistics cards
   - Results table

---

### ✅ Test 4: Multiple Files

**Steps:**
1. Load template
2. Click "Choose image file(s)"
3. Hold Ctrl/Cmd and select multiple images
4. Click "🔍 Detect Bubbles"
5. **Expected**:
   - "Processing 1/N: filename"
   - Batch results with file headers
   - Aggregate statistics

---

### ✅ Test 5: Folder Upload (Chrome/Edge)

**Steps:**
1. Open in Chrome or Edge
2. Click "📁 Or Select Folder"
3. Select folder with images (and optionally template.json)
4. **Expected**:
   - Template auto-loaded (if present)
   - All images from subdirectories
   - Ready to detect

---

### ✅ Test 6: CSV Export

**Steps:**
1. After detection completes
2. Click "📊 Export CSV"
3. **Expected**:
   - CSV file downloads
   - Contains all results
   - One row per image (batch) or per question (single)

---

### ✅ Test 7: Reset

**Steps:**
1. After loading and detecting
2. Click "🔄 Reset"
3. **Expected**:
   - All files cleared
   - Canvases cleared
   - Ready for new session

---

## 🎨 UI Features to Check

### Header
- [x] Title displays: "📝 OMRChecker Demo"
- [x] Status bar shows OpenCV status
- [x] Green success message when loaded

### Upload Section
- [x] Template upload area
- [x] Image upload area (multiple)
- [x] Folder button visible
- [x] File info displays after selection

### Buttons
- [x] Detect button disabled until template + images loaded
- [x] Export button disabled until detection complete
- [x] Reset button always enabled

### Preview Section
- [x] Original image canvas
- [x] Detection result canvas
- [x] Bubble overlay visualization
- [x] Green circles for marked bubbles
- [x] Gray circles for unmarked

### Results Section
- [x] Statistics cards (Total, Answered, etc.)
- [x] Results table with all fields
- [x] Color-coded rows (green=answered, yellow=multi-marked, red=empty)
- [x] Status badges

### Loading Overlay
- [x] Spinner animation
- [x] Progress messages
- [x] Backdrop blur effect

---

## 🔍 What to Look For

### ✅ Success Indicators
- Green checkmarks in status messages
- Template auto-detected message
- Images loaded count
- Detection complete message
- Smooth animations

### ⚠️ Warning Messages
- "No template.json found" (if not in folder)
- "Multi-marked detected"
- Yellow status badges

### ❌ Error Handling
- Invalid template JSON
- No images found
- Detection failures
- Browser compatibility messages

---

## 🌐 Browser Testing

### Chrome/Edge (Full Support)
- [x] OpenCV loading
- [x] File upload
- [x] Folder upload (File System Access API)
- [x] Batch processing
- [x] CSV export

### Firefox (No Folder Upload)
- [x] OpenCV loading
- [x] File upload (single/multiple)
- [x] Batch processing
- [x] CSV export
- [ ] Folder upload (not supported - shows alert)

### Safari (No Folder Upload)
- [x] OpenCV loading
- [x] File upload (single/multiple)
- [x] Batch processing
- [x] CSV export
- [ ] Folder upload (not supported - shows alert)

---

## 📊 Performance Checks

### Load Times
- OpenCV: ~2 seconds
- Template: <100ms
- Image: ~200ms
- Detection: ~300ms per image

### Memory
- Check browser DevTools → Memory
- Should clean up cv.Mat objects
- No memory leaks after multiple detections

### Batch Processing
- 10 images: ~5 seconds
- 100 images: ~50 seconds
- Progress updates smooth

---

## 🐛 Known Issues / Edge Cases

### Browser Security
- ⚠️ Cannot access parent directories in folder mode
- ✅ Workaround: Select parent folder or upload manually

### File Types
- ✅ Supported: JPG, PNG, BMP, TIF, WEBP
- ❌ Not supported: PDF, HEIC (need conversion)

### Template Validation
- Currently basic validation
- Invalid JSON shows error
- Missing required fields caught

---

## 🎯 Quick Test Checklist

**5-Minute Smoke Test:**
- [ ] Open http://localhost:3001/
- [ ] Wait for OpenCV to load (green message)
- [ ] Upload template.json (or use folder with template)
- [ ] Upload an image or folder
- [ ] Click Detect
- [ ] See visualization with green/gray circles
- [ ] Check statistics cards
- [ ] Export CSV
- [ ] Verify CSV contains results
- [ ] Click Reset
- [ ] Everything cleared

**If all pass → Demo is working!** ✅

---

## 📝 Sample Test Data

### Using Example Template
```bash
# Template location:
/Users/udayraj.deshmukh/Personals/OMRChecker/omrchecker-js/examples/simple-template.json

# This template defines:
- 10 multiple-choice questions (q1-q10)
- 4 options each (A, B, C, D)
- Bubble dimensions: 40x40px
- Template size: 1000x800px
```

### Sample Folder Structure
```
test-folder/
├── template.json       ← Auto-detected
├── scans/
│   ├── student1.jpg
│   ├── student2.jpg
│   └── student3.jpg
└── backup/
    └── test.jpg
```

---

## 🎉 What You Should See

### Initial Load
```
📝 OMRChecker Demo
TypeScript-powered OMR bubble detection in the browser
✅ OpenCV loaded successfully
```

### After Folder Selection (with template)
```
Template:
📄 template.json (auto-detected)
10 fields | Found in: test-folder

Images:
4 images from folder
Root: test-folder

[🔍 Detect Bubbles] ← Now enabled!
```

### During Detection
```
🔄 Processing 2/4: scans/student2.jpg
```

### After Detection
```
Statistics:
Total: 40 (4 images)
Answered: 35
Unanswered: 3
Multi-marked: 2
Avg Confidence: 87.5%

Results table shows all 40 results grouped by image
```

---

## 🚀 Next Steps After Testing

If everything works:
1. ✅ OpenCV loads properly
2. ✅ Template auto-detection works
3. ✅ Folder upload works (Chrome/Edge)
4. ✅ Batch processing works
5. ✅ Visualization displays correctly
6. ✅ CSV export works

**Demo is production-ready!** 🎊

---

## 💡 Tips for Best Results

1. **Image Quality**: Use clear, high-contrast images
2. **Template Accuracy**: Ensure bubble locations match actual sheets
3. **Browser**: Use Chrome/Edge for full features
4. **File Organization**: Keep template.json with images
5. **Batch Size**: 10-50 images per batch for best performance

---

## 🔗 Useful Links

- **Demo**: http://localhost:3001/
- **Template Example**: `/omrchecker-js/examples/simple-template.json`
- **Documentation**: See `TYPESCRIPT_PORT_PHASE6_DEMO_APP.md`
- **Fixes**: See `DEMO_FIXES_OPENCV_FOLDER.md`
- **Auto-Template**: See `AUTO_TEMPLATE_DETECTION.md`

---

**Happy Testing!** 🧪✨

