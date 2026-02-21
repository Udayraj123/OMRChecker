# Error Recovery Patterns in OMRChecker

**Module**: modules/technical/error-recovery/
**Created**: 2026-02-20

## Python Error Recovery

**Graceful Degradation**: Fallback to simpler algorithms on failure
**Error File Handling**: Move failed images to error directory

## Browser Error Recovery

### Fallback Strategies

```javascript
async function detectBubbles(image, config) {
  try {
    // Try ML-based detection
    if (config.useML && await isModelLoaded()) {
      return await mlBubbleDetection(image);
    }
  } catch (error) {
    console.warn('ML detection failed, falling back to threshold', error);
  }

  // Fallback to threshold-based detection
  return thresholdBubbleDetection(image);
}
```

### Error Boundaries (React)

```jsx
class OMRErrorBoundary extends React.Component {
  state = { hasError: false };

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('OMR Processing Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return <ErrorFallbackUI />;
    }
    return this.props.children;
  }
}
```

### Retry Logic

```javascript
async function processWithRetry(fn, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
}
```

**Recommendation**: Always provide fallbacks for ML operations, use error boundaries in UI.
