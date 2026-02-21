# Image Filters - Flows

**Module**: Domain / Preprocessing / Filters
**Python Reference**: `src/processors/image/GaussianBlur.py`, `src/processors/image/Contrast.py`, `src/processors/image/Levels.py`
**Last Updated**: 2026-02-20

---

## GaussianBlur

**Purpose**: Reduce noise in scanned images

**Configuration**:
```json
{
  "name": "GaussianBlur",
  "options": {
    "kSize": [3, 3],
    "sigmaX": 0
  }
}
```

**Algorithm**: `cv2.GaussianBlur(image, kSize, sigmaX)`
**Effect**: Smooths image, reduces high-frequency noise

**Browser**:
```typescript
const blurred = new cv.Mat();
cv.GaussianBlur(image, blurred, new cv.Size(3, 3), 0);
```

---

## Contrast

**Purpose**: Adjust image contrast for better bubble detection

**Configuration**:
```json
{
  "name": "Contrast",
  "options": {
    "alpha": 1.5,
    "beta": 0
  }
}
```

**Algorithm**: `output = alpha * input + beta`
**Effect**: Increases contrast (alpha > 1), adjusts brightness (beta)

**Browser**:
```typescript
const contrasted = new cv.Mat();
image.convertTo(contrasted, -1, alpha, beta);
```

---

## Levels

**Purpose**: Adjust brightness/darkness levels (histogram adjustment)

**Configuration**:
```json
{
  "name": "Levels",
  "options": {
    "low": 50,
    "high": 200,
    "gamma": 1.0
  }
}
```

**Algorithm**:
1. Map [low, high] to [0, 255]
2. Apply gamma correction

**Browser**:
```typescript
// Normalize to [low, high]
const normalized = new cv.Mat();
cv.normalize(image, normalized, low, high, cv.NORM_MINMAX);

// Apply gamma if needed
if (gamma !== 1.0) {
    const lookupTable = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
        lookupTable[i] = Math.pow(i / 255, gamma) * 255;
    }
    cv.LUT(normalized, lookupTable, normalized);
}
```

---

## Summary

**Filters**: Enhance image quality for better detection
**Types**: GaussianBlur (noise), Contrast (contrast/brightness), Levels (histogram)
**Browser**: Use OpenCV.js filter functions
**Effect**: Improve bubble detection accuracy
