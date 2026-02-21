# Template JSON Format

**Module**: modules/integration/template-format/
**Created**: 2026-02-20
**Reference**: `src/schemas/template_schema.py`

## Template Structure

```json
{
  "pageDimensions": [1240, 1754],
  "bubbleDimensions": [32, 32],
  "preProcessors": [
    {
      "name": "CropPage",
      "options": {
        "maxPoints": 10
      }
    }
  ],
  "fieldBlocks": [
    {
      "name": "MCQ_Block_1",
      "origin": [100, 100],
      "dimensions": [400, 600],
      "fields": [
        {
          "fieldLabel": "Q1",
          "fieldType": "QTYPE_MCQ4",
          "bubbleValues": ["A", "B", "C", "D"]
        }
      ]
    }
  ]
}
```

## Key Fields

**pageDimensions**: [width, height] - Reference page size
**bubbleDimensions**: [width, height] - Default bubble size
**preProcessors**: Array of image preprocessing steps
**fieldBlocks**: Array of detection regions
**fields**: Array of answer fields within blocks

## Field Types

- `QTYPE_MCQ4`: 4 options (A, B, C, D)
- `QTYPE_MCQ5`: 5 options (A, B, C, D, E)
- `QTYPE_INT`: Integer input
- `CUSTOM`: Custom bubble values

## Browser Validation

```typescript
import { z } from 'zod';

const TemplateSchema = z.object({
  pageDimensions: z.tuple([z.number(), z.number()]),
  bubbleDimensions: z.tuple([z.number(), z.number()]),
  preProcessors: z.array(z.object({
    name: z.string(),
    options: z.record(z.any()).optional()
  })),
  fieldBlocks: z.array(z.object({
    name: z.string(),
    origin: z.tuple([z.number(), z.number()]),
    dimensions: z.tuple([z.number(), z.number()]),
    fields: z.array(z.object({
      fieldLabel: z.string(),
      fieldType: z.string(),
      bubbleValues: z.array(z.string())
    }))
  }))
});

type Template = z.infer<typeof TemplateSchema>;
```

**See Also**: `modules/domain/template/` for detailed entity documentation
