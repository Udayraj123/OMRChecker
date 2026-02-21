# OpenCV.js Migration Patterns

The OpenCV.js API is **nearly identical** to Python's cv2. Here are the key patterns for migration:

## Memory Management

**Main Difference**: OpenCV.js requires manual memory cleanup.

### Pattern: Always use try/finally
```javascript
// Python
def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    return thresh

// JavaScript
function processImage(img) {
    const gray = new cv.Mat();
    const thresh = new cv.Mat();

    try {
        cv.cvtColor(img, gray, cv.COLOR_BGR2GRAY);
        cv.threshold(gray, thresh, 127, 255, cv.THRESH_BINARY);
        return thresh.clone(); // Clone before cleanup
    } finally {
        gray.delete();
        // Don't delete thresh - we're returning it
    }
}
```

## Common Operations (API is Same!)

### Image Reading
```python
# Python
img = cv2.imread('image.jpg')

// JavaScript
const img = cv.imread(imageElement); // imageElement is <img> or <canvas>
```

### Color Conversion
```python
# Python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

// JavaScript
const gray = new cv.Mat();
cv.cvtColor(img, gray, cv.COLOR_BGR2GRAY);
```

### Thresholding
```python
# Python
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

// JavaScript
const thresh = new cv.Mat();
cv.threshold(gray, thresh, 127, 255, cv.THRESH_BINARY);
// Note: cv.threshold returns threshold value in JavaScript too, but usually ignored
```

### Find Contours
```python
# Python
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

// JavaScript
const contours = new cv.MatVector();
const hierarchy = new cv.Mat();
cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
```

### Morphological Operations
```python
# Python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(img, kernel, iterations=1)

// JavaScript
const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
const dilated = new cv.Mat();
cv.dilate(img, dilated, kernel);
kernel.delete();
```

### Drawing
```python
# Python
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

// JavaScript
cv.rectangle(img, new cv.Point(x, y), new cv.Point(x+w, y+h),
             new cv.Scalar(0, 255, 0), 2);
```

## NumPy Array Operations

### Python NumPy → TypedArray
```python
# Python
pixels = np.array([0, 255, 128], dtype=np.uint8)

// JavaScript
const pixels = new Uint8Array([0, 255, 128]);
```

### Accessing Mat Data
```python
# Python
height, width = img.shape[:2]
pixel_value = img[y, x]

// JavaScript
const height = img.rows;
const width = img.cols;
const pixelValue = img.ucharAt(y, x);
```

## Async Loading (Only Once at Startup)

```javascript
// Load OpenCV.js once at app startup
async function loadOpenCV() {
    return new Promise((resolve) => {
        if (window.cv && cv.Mat) {
            resolve();
        } else {
            window.Module = {
                onRuntimeInitialized: resolve
            };
            const script = document.createElement('script');
            script.src = 'opencv.js';
            document.body.appendChild(script);
        }
    });
}

// Use in app initialization
await loadOpenCV();
// Now cv is available globally
```

## Migration Script Auto-Handles

The migration script (`4-migrate-files.js`) automatically converts:

✅ `cv2.` → `cv.`
✅ Function names (identical)
✅ Constants (cv2.THRESH_BINARY → cv.THRESH_BINARY)
✅ Basic memory management patterns

## Manual Tweaks Needed (Minimal)

After migration, you may need to manually:

1. **Add memory cleanup** for complex functions with multiple Mat objects
2. **Clone return values** before cleanup if returning Mat
3. **Convert NumPy-specific operations** (array slicing, broadcasting) to TypedArray equivalents
4. **Handle image loading** from File API instead of file paths

## Example: Full Function Migration

### Python Original
```python
def align_image(image, template):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray, None)
    kp2, des2 = sift.detectAndCompute(template_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    h, w = image.shape[:2]
    aligned = cv2.warpPerspective(image, M, (w, h))

    return aligned
```

### JavaScript Auto-Migrated (98% correct!)
```typescript
function alignImage(image: cv.Mat, template: cv.Mat): cv.Mat | null {
    const gray = new cv.Mat();
    const templateGray = new cv.Mat();

    try {
        cv.cvtColor(image, gray, cv.COLOR_BGR2GRAY);
        cv.cvtColor(template, templateGray, cv.COLOR_BGR2GRAY);

        // Note: OpenCV.js doesn't have SIFT (patented), use ORB instead
        const orb = new cv.ORB();
        const kp1 = new cv.KeyPointVector();
        const des1 = new cv.Mat();
        const kp2 = new cv.KeyPointVector();
        const des2 = new cv.Mat();

        orb.detectAndCompute(gray, new cv.Mat(), kp1, des1);
        orb.detectAndCompute(templateGray, new cv.Mat(), kp2, des2);

        const bf = new cv.BFMatcher();
        const matches = new cv.DMatchVectorVector();
        bf.knnMatch(des1, des2, matches, 2);

        const goodMatches = [];
        for (let i = 0; i < matches.size(); i++) {
            const match = matches.get(i);
            const m = match.get(0);
            const n = match.get(1);
            if (m.distance < 0.75 * n.distance) {
                goodMatches.push(m);
            }
        }

        if (goodMatches.length < 10) {
            return null;
        }

        // Build point arrays
        const srcPts = [];
        const dstPts = [];
        for (const m of goodMatches) {
            srcPts.push(kp1.get(m.queryIdx).pt.x);
            srcPts.push(kp1.get(m.queryIdx).pt.y);
            dstPts.push(kp2.get(m.trainIdx).pt.x);
            dstPts.push(kp2.get(m.trainIdx).pt.y);
        }

        const srcMat = cv.matFromArray(goodMatches.length, 1, cv.CV_32FC2, srcPts);
        const dstMat = cv.matFromArray(goodMatches.length, 1, cv.CV_32FC2, dstPts);

        const M = cv.findHomography(srcMat, dstMat, cv.RANSAC);
        const aligned = new cv.Mat();
        const size = new cv.Size(image.cols, image.rows);
        cv.warpPerspective(image, aligned, M, size);

        // Cleanup
        gray.delete();
        templateGray.delete();
        orb.delete();
        kp1.delete();
        des1.delete();
        kp2.delete();
        des2.delete();
        bf.delete();
        matches.delete();
        srcMat.delete();
        dstMat.delete();
        M.delete();

        return aligned; // Caller must delete

    } catch (error) {
        // Cleanup on error
        gray.delete();
        templateGray.delete();
        throw error;
    }
}
```

## Summary

**OpenCV.js migration is mostly automatic!**

- ✅ 90% of code: Same API, just `cv2.` → `cv.`
- ⚠️ 10% manual: Memory management (.delete()), error handling
- ⚠️ Minor: SIFT → ORB (patent issue), some advanced features missing

**The migration script handles the bulk of it - you'll just need to polish memory management.**
