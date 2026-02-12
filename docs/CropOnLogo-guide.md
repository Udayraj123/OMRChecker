# CropOnLogo guide

Use **CropOnLogo** when your OMR sheets have a fixed logo that appears in the same place. The pre-processor finds the logo with template matching and translates the image so the logo sits at a known position, giving a stable reference for the bubble grid.

## When to use CropOnLogo

- Your sheets always show the same logo (e.g. company or exam header).
- You do **not** have corner/edge markers (in that case use **CropOnMarkers**).
- You want alignment based on that logo instead of markers or manual cropping.

## Logo image requirements

| Requirement   | Details |
|---------------|--------|
| **Content**   | Exact crop of the logo as it appears on the scanned sheets: same design, no extra borders. It is matched as-is. |
| **Scale**     | Same apparent size as on the resized sheet. Sheets are resized to `processing_width` (default 666 px) before matching. If the logo is e.g. 1/10 of the sheet width, the logo file should be ~66 px wide, or use `sheetToLogoWidthRatio` in options. |
| **Format**    | Any format OpenCV reads (PNG, JPG, etc.). Loaded in grayscale. |
| **Orientation** | Same as in the scans (no rotation). |

## Configuration

Add CropOnLogo to the `preProcessors` array in your `template.json`.

### Minimal

```json
{"name": "CropOnLogo", "options": {"relativePath": "logo.png"}}
```

Place `logo.png` in the same directory as `template.json`.

### All options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `relativePath` | string | `"logo.png"` | Path to the logo image relative to the template directory. |
| `expected_origin` | `[x, y]` | `[0, 0]` | Pixel position where the logo’s top-left corner should end up after alignment. |
| `min_matching_threshold` | number (0–1) | `0.4` | Minimum template-match score to accept. Lower = more tolerant; raise if you get false matches. |
| `sheetToLogoWidthRatio` | number | — | Ratio (sheet_width / logo_width) on the resized sheet. Set when the logo file’s size doesn’t match the apparent size on the sheet. |

### Example

```json
{
  "preProcessors": [
    {
      "name": "CropOnLogo",
      "options": {
        "relativePath": "logo.jpg",
        "expected_origin": [50, 30],
        "min_matching_threshold": 0.35
      }
    },
    {"name": "GaussianBlur", "options": {"kSize": [3, 3], "sigmaX": 0}}
  ]
}
```

## How it works

1. The sheet is resized to the processing width and normalized (grayscale).
2. The logo image is loaded, optionally resized with `sheetToLogoWidthRatio`, blurred and normalized.
3. Template matching finds the logo position; if the score is below `min_matching_threshold`, processing fails for that image.
4. A translation is applied so the logo’s top-left corner moves to `expected_origin`. New border areas are filled with white; image dimensions stay the same.

## Troubleshooting

- **"Logo not found"**  
  Check that `relativePath` is correct and the file exists next to `template.json`.

- **Match score too low**  
  Ensure the logo file is an exact crop of what appears on the sheet (same scale and orientation). Try lowering `min_matching_threshold` slightly or set `sheetToLogoWidthRatio` if the logo size on the sheet is different.

- **Wrong alignment**  
  Adjust `expected_origin` so it matches the position you use in your template grid. Use `--setLayout` to inspect the aligned result.
