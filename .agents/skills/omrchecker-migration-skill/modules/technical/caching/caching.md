# Caching Strategies in OMRChecker

**Module**: modules/technical/caching/
**Created**: 2026-02-20

## Python Caching

**Template Caching**: Load template once, reuse for multiple images
**Image Caching**: Cache preprocessed images

## Browser Caching

### 1. Memory Cache (Map/Object)

```javascript
const templateCache = new Map();

async function getTemplate(templateId) {
  if (templateCache.has(templateId)) {
    return templateCache.get(templateId);
  }

  const template = await loadTemplate(templateId);
  templateCache.set(templateId, template);
  return template;
}
```

### 2. IndexedDB (Persistent)

```javascript
// Store template
const db = await openDB('omr-cache', 1, {
  upgrade(db) {
    db.createObjectStore('templates');
  }
});

await db.put('templates', template, templateId);
const cached = await db.get('templates', templateId);
```

### 3. Cache API (Service Worker)

```javascript
// Cache OpenCV.js and models
const cache = await caches.open('omr-v1');
await cache.addAll([
  '/opencv.js',
  '/models/yolo.json'
]);
```

**Recommendation**: Use Map for session cache, IndexedDB for persistent templates.
