# Config JSON Format

**Module**: modules/integration/config-format/
**Created**: 2026-02-20
**Reference**: `src/schemas/config_schema.py`

## Config Structure

```json
{
  "dimensions": {
    "processing_width": 1000,
    "processing_height": 1414
  },
  "alignment": {
    "max_matching_variation": 10
  },
  "thresholding": {
    "threshold_circle": 200,
    "min_matching_threshold": 0.3
  },
  "outputs": {
    "save_image_level": 2,
    "colored_outputs_enabled": true
  }
}
```

## Sections

**dimensions**: Processing image dimensions
**alignment**: SIFT/template matching parameters
**thresholding**: Bubble detection thresholds
**outputs**: Debug output configuration

## Browser Config

```typescript
const ConfigSchema = z.object({
  dimensions: z.object({
    processing_width: z.number().default(1000),
    processing_height: z.number().default(1414)
  }),
  alignment: z.object({
    max_matching_variation: z.number().default(10)
  }),
  thresholding: z.object({
    threshold_circle: z.number().default(200),
    min_matching_threshold: z.number().default(0.3)
  }),
  outputs: z.object({
    save_image_level: z.number().default(2),
    colored_outputs_enabled: z.boolean().default(true)
  })
});
```

**Storage**: Use localStorage for user config persistence

```javascript
function saveConfig(config) {
  localStorage.setItem('omr-config', JSON.stringify(config));
}

function loadConfig() {
  const stored = localStorage.getItem('omr-config');
  return stored ? JSON.parse(stored) : getDefaultConfig();
}
```
