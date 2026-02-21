# Debug & Visualization System in OMRChecker

**Module**: modules/technical/debugging/
**Created**: 2026-02-20

## Python Debug System

**Save Image Operations**: Configured debug levels control intermediate image saving
**Image Stacking**: Visual debug output with hstack/vstack

## Browser Debugging

### Canvas-based Visualization

```javascript
function debugDrawBubbles(canvas, bubbles, interpretations) {
  const ctx = canvas.getContext('2d');

  bubbles.forEach((bubble, idx) => {
    const isMarked = interpretations[idx].isAttempted;
    ctx.strokeStyle = isMarked ? 'green' : 'red';
    ctx.strokeRect(bubble.x, bubble.y, bubble.width, bubble.height);
  });
}
```

### Debug Levels

```javascript
const DebugLevel = {
  NONE: 0,
  ERRORS: 1,
  WARNINGS: 2,
  INFO: 3,
  VERBOSE: 4
};

function debugLog(level, message, data) {
  if (level <= currentDebugLevel) {
    console.log(`[${DebugLevel[level]}] ${message}`, data);
  }
}
```

### Download Debug Images

```javascript
function downloadDebugImage(canvas, filename) {
  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  });
}
```

**Recommendation**: Use canvas overlays for visual debugging, console for logs.
