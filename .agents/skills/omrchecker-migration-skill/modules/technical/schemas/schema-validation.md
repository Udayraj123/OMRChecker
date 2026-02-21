# Pydantic Schema System in OMRChecker

**Module**: modules/technical/schemas/
**Status**: Complete
**Created**: 2026-02-20

---

## Overview

OMRChecker uses Pydantic for schema validation, data parsing, and serialization. Key schemas include Template, Config, Evaluation, and field-level models.

**Key Files**:
- `src/schemas/models/template.py` - Template schema
- `src/schemas/models/config.py` - Config schema
- `src/schemas/models/evaluation.py` - Evaluation schema
- `src/schemas/defaults/` - Default values

---

## Browser Migration: Zod

**Installation**:
```bash
npm install zod
```

**Python Pydantic** → **Browser Zod** mapping:

| Pydantic | Zod | Example |
|----------|-----|---------|
| `str` | `z.string()` | Field names |
| `int` | `z.number().int()` | Bubble dimensions |
| `float` | `z.number()` | Confidence scores |
| `bool` | `z.boolean()` | Enable flags |
| `List[str]` | `z.array(z.string())` | Field labels |
| `Dict[str, Any]` | `z.record(z.any())` | Custom data |
| `Optional[int]` | `z.number().optional()` | Optional fields |
| `Union[str, int]` | `z.union([z.string(), z.number()])` | Multiple types |
| `Literal["a", "b"]` | `z.enum(["a", "b"])` | Fixed values |
| `BaseModel` | `z.object({...})` | Nested schemas |

---

## Example: Template Schema

**Python (Pydantic)**:
```python
from pydantic import BaseModel, Field

class BubbleDimensions(BaseModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)

class FieldConfig(BaseModel):
    field_label: str
    field_type: str
    bubble_dimensions: Optional[BubbleDimensions]
```

**Browser (Zod)**:
```typescript
import { z } from 'zod';

const BubbleDimensionsSchema = z.object({
  width: z.number().int().min(1),
  height: z.number().int().min(1)
});

const FieldConfigSchema = z.object({
  field_label: z.string(),
  field_type: z.string(),
  bubble_dimensions: BubbleDimensionsSchema.optional()
});

type FieldConfig = z.infer<typeof FieldConfigSchema>;
```

---

## Validation Patterns

### 1. Parse and Validate

**Python**:
```python
template = TemplateSchema.parse_obj(json_data)
```

**Browser**:
```typescript
const result = TemplateSchema.safeParse(jsonData);

if (result.success) {
  const template = result.data;
} else {
  console.error(result.error.issues);
}
```

### 2. Custom Validators

**Python**:
```python
from pydantic import validator

class FieldBlock(BaseModel):
    @validator('bubble_dimensions')
    def validate_dimensions(cls, v):
        if v and (v.width <= 0 or v.height <= 0):
            raise ValueError('Dimensions must be positive')
        return v
```

**Browser**:
```typescript
const FieldBlockSchema = z.object({
  bubble_dimensions: z.object({
    width: z.number(),
    height: z.number()
  }).refine(
    (dims) => dims.width > 0 && dims.height > 0,
    { message: 'Dimensions must be positive' }
  )
});
```

---

## Default Values

**Python**:
```python
class Config(BaseModel):
    threshold: int = 200
    save_image_level: int = 0
```

**Browser**:
```typescript
const ConfigSchema = z.object({
  threshold: z.number().default(200),
  save_image_level: z.number().default(0)
});
```

---

## Summary

**Migration Path**: Pydantic → Zod for TypeScript/JavaScript
**Key Features**: Type safety, validation, default values, error messages
**Recommendation**: Use Zod for full type safety and runtime validation
