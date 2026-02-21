# Validation Patterns

**Module**: Foundation
**Python Reference**: `src/utils/validations.py`, `src/schemas/`
**Last Updated**: 2026-02-20

---

## Overview

OMRChecker uses **JSON Schema validation** (jsonschema library) with **Pydantic** models for structured data validation.

**Key Components**:
1. JSON Schema definitions (`src/schemas/*_schema.py`)
2. Validation utilities (`src/utils/validations.py`)
3. Pydantic dataclass models (`src/schemas/models/`)
4. Custom exception classes for validation errors

---

## Validation Functions

**Code Reference**: `src/utils/validations.py`

### validate_template_json
```python
def validate_template_json(json_data: dict, template_path: Path):
    validate(instance=json_data, schema=SCHEMA_JSONS["template"])
    # On error: display Rich table + raise TemplateValidationError
```

### validate_config_json
```python
def validate_config_json(json_data: dict, config_path: Path):
    validate(instance=json_data, schema=SCHEMA_JSONS["config"])
    # On error: raise ConfigValidationError
```

### validate_evaluation_json
```python
def validate_evaluation_json(json_data: dict, evaluation_path: Path):
    validate(instance=json_data, schema=SCHEMA_JSONS["evaluation"])
    # On error: raise EvaluationValidationError
```

---

## Schema System

### JSON Schemas
**Location**: `src/schemas/`
- `template_schema.py` - Template JSON structure
- `config_schema.py` - Config JSON structure
- `evaluation_schema.py` - Evaluation JSON structure
- `constants.py` - Common schema definitions

### Pydantic Models
**Location**: `src/schemas/models/`
- `template.py` - Template dataclasses
- `config.py` - Config dataclasses
- `evaluation.py` - Evaluation dataclasses

**Pattern**: JSON Schema for validation → Pydantic for type safety

---

## Validation Error Display

### Rich Table Format
```python
table = Table(show_lines=True)
table.add_column("Key", style="cyan")
table.add_column("Error", style="magenta")

for error in errors:
    key, validator, msg = parse_validation_error(error)
    table.add_row(key, msg)

console.print(table)
```

### Error Parsing
```python
def parse_validation_error(error):
    key = ".".join(map(str, error.path))
    validator = error.validator
    msg = error.message
    return key, validator, msg
```

---

## Common Validation Rules

### Required Fields
```json
{
  "required": ["templateDimensions", "bubbleDimensions", "fieldBlocks"]
}
```

### Type Validation
```json
{
  "type": "object",
  "properties": {
    "templateDimensions": {"type": "array", "items": {"type": "number"}}
  }
}
```

### Enum Validation
```json
{
  "fieldDetectionType": {
    "enum": ["BUBBLES_THRESHOLD", "BARCODE", "OCR"]
  }
}
```

### Pattern Validation
```json
{
  "pattern": "^[A-Za-z0-9_]+$"
}
```

---

## Browser Migration

### Zod Validation (TypeScript)
```typescript
import { z } from 'zod';

const templateSchema = z.object({
  templateDimensions: z.tuple([z.number(), z.number()]),
  bubbleDimensions: z.tuple([z.number(), z.number()]),
  fieldBlocks: z.record(z.object({
    fieldDetectionType: z.enum(['BUBBLES_THRESHOLD', 'BARCODE', 'OCR']),
    // ...
  })),
});

// Validation
try {
  const template = templateSchema.parse(jsonData);
} catch (e) {
  if (e instanceof z.ZodError) {
    throw new TemplateValidationError(path, e.errors.map(err => err.message));
  }
}
```

### JSON Schema (JavaScript)
```javascript
import Ajv from 'ajv';

const ajv = new Ajv();
const validate = ajv.compile(templateSchema);

if (!validate(jsonData)) {
  throw new TemplateValidationError(path, validate.errors);
}
```

---

## Summary

**Validation**: JSON Schema + jsonschema library
**Type Safety**: Pydantic dataclasses
**Error Display**: Rich tables with formatted errors
**Browser**: Use Zod (TypeScript) or Ajv (JavaScript)

**Key Pattern**: Validate early, fail fast with clear error messages
