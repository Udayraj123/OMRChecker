# Concurrency Patterns in OMRChecker

**Module**: modules/technical/concurrency/
**Created**: 2026-02-20

## Python: ThreadPoolExecutor

**Usage in OMRChecker**:
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_image, image_files)
```

## Browser: Web Workers

**Main Thread**:
```javascript
const worker = new Worker('omr-worker.js');

worker.postMessage({
  type: 'process',
  imageData: imageData,
  template: template
});

worker.onmessage = (e) => {
  const result = e.data.result;
  displayResult(result);
};
```

**Worker Thread** (`omr-worker.js`):
```javascript
self.onmessage = async (e) => {
  const { type, imageData, template } = e.data;

  if (type === 'process') {
    const result = await processImage(imageData, template);
    self.postMessage({ type: 'result', result });
  }
};
```

**Key Differences**:
- Python: Shared memory (GIL limitations)
- Browser: Message passing (structured clone)
- Use `Transferable` objects for zero-copy image transfer

**Recommendation**: Use Web Workers for heavy image processing to keep UI responsive.
