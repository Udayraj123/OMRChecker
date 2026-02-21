# CLI Interface in OMRChecker

**Module**: modules/integration/cli/
**Created**: 2026-02-20

## Python CLI

**Entry Point**: `main.py`
**Arguments**: Template path, input directory, output directory, tuning config

**Example**:
```bash
python main.py \
  --template template.json \
  --input inputs/ \
  --output outputs/ \
  --config config.json
```

## Browser Equivalent: Web UI

### File Input

```html
<input type="file" id="imageInput" multiple accept="image/*" />
<input type="file" id="templateInput" accept=".json" />
<button onclick="processImages()">Process</button>
```

### JavaScript Handler

```javascript
async function processImages() {
  const imageFiles = document.getElementById('imageInput').files;
  const templateFile = document.getElementById('templateInput').files[0];

  const template = JSON.parse(await readFile(templateFile));

  for (const imageFile of imageFiles) {
    const result = await processImage(imageFile, template);
    displayResult(result);
  }
}
```

### Progressive Web App (PWA)

```javascript
// Service worker for offline processing
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then(response =>
      response || fetch(event.request)
    )
  );
});
```

**Browser Interface Options**:
1. **Web UI**: Drag-and-drop files, visual feedback
2. **Electron**: Desktop app with file system access
3. **Browser Extension**: Process images from web pages

**Recommendation**: Build web UI with drag-and-drop for best UX.
