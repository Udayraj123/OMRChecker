# Metrics & Statistics in OMRChecker

**Module**: modules/technical/metrics/
**Created**: 2026-02-20

## Python Metrics

**Bubble Statistics**:
- Mean bubble value
- Standard deviation
- Percentile thresholds
- Confidence scores

**Performance Metrics**:
- Processing time per image
- Detection accuracy

## Browser Metrics

### Performance API

```javascript
// Measure processing time
performance.mark('processing-start');
await processImage(image);
performance.mark('processing-end');

performance.measure('processing', 'processing-start', 'processing-end');
const measure = performance.getEntriesByName('processing')[0];
console.log(`Processing took ${measure.duration}ms`);
```

### Statistics Calculations

```javascript
function calculateBubbleStats(bubbleValues) {
  const mean = bubbleValues.reduce((a, b) => a + b, 0) / bubbleValues.length;

  const variance = bubbleValues.reduce((sum, val) =>
    sum + Math.pow(val - mean, 2), 0
  ) / bubbleValues.length;

  const std = Math.sqrt(variance);

  return { mean, std };
}
```

### Analytics Integration

```javascript
// Google Analytics
gtag('event', 'omr_processing', {
  processing_time: duration,
  image_count: count,
  success: true
});
```

**Recommendation**: Use Performance API for timing, custom functions for statistics.
