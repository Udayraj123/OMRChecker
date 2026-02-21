# File System Patterns in OMRChecker

**Module**: modules/technical/filesystem/
**Created**: 2026-02-20

## Browser Migration: File API + IndexedDB

**Python File I/O** → **Browser Alternatives**:

| Python | Browser | Use Case |
|--------|---------|----------|
| `open(file, 'r')` | `FileReader` API | Read user files |
| `cv2.imread()` | `Canvas` + `Image` | Load images |
| `json.load()` | `JSON.parse()` | Parse JSON |
| File system paths | `File` objects | Handle file references |
| `os.path.join()` | Template literals | Path construction |
| Persistent storage | `IndexedDB` | Store templates/configs |

**Key Pattern**: Browser can't write to disk directly. Use downloads or IndexedDB.

**Example - Read File**:
```javascript
async function readFile(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.readAsText(file);
  });
}
```

**Example - Save Result**:
```javascript
function downloadFile(content, filename) {
  const blob = new Blob([content], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
```
