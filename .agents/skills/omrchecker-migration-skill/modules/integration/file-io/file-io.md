# File Input/Output in OMRChecker

**Module**: modules/integration/file-io/
**Created**: 2026-02-20

## Input Formats

**Images**: JPG, PNG, TIFF
**Template**: JSON
**Config**: JSON
**Evaluation**: JSON

## Output Formats

**CSV**: Detection results
**Images**: Marked/debug images
**JSON**: Detailed results

## Browser File I/O

### Reading Files

```javascript
// Image
const imageFile = event.target.files[0];
const imageUrl = URL.createObjectURL(imageFile);
const img = new Image();
img.src = imageUrl;

// JSON
const jsonFile = event.target.files[0];
const jsonText = await jsonFile.text();
const data = JSON.parse(jsonText);
```

### Writing Files (Downloads)

```javascript
// CSV
function downloadCSV(results, filename) {
  const csv = results.map(r =>
    `${r.fileName},${r.fieldLabel},${r.value}`
  ).join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  downloadBlob(blob, filename);
}

// JSON
function downloadJSON(data, filename) {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  downloadBlob(blob, filename);
}

// Image
function downloadImage(canvas, filename) {
  canvas.toBlob(blob => downloadBlob(blob, filename), 'image/jpeg');
}
```

### Batch Processing

```javascript
async function processBatch(files, template) {
  const results = [];

  for (const file of files) {
    const result = await processImage(file, template);
    results.push(result);
  }

  downloadCSV(results, 'results.csv');
}
```

**Key Differences**:
- Python: Direct file system access
- Browser: User must select files, downloads for output
- Use IndexedDB for temporary storage between processing steps
