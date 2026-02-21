# FieldBlock Entity - Concept

**Module**: Domain / FieldBlock
**Python Reference**: `src/processors/layout/field_block/base.py`
**Last Updated**: 2026-02-20

---

## Overview

A FieldBlock represents a logical grouping of related fields (questions) on an OMR sheet. It defines the spatial arrangement, bubble dimensions, field detection strategy, and layout parameters for a group of fields.

**Key Responsibilities**:
1. **Spatial Layout**: Define origin, direction, gaps, and dimensions
2. **Field Generation**: Create Field objects for each field label
3. **Detection Strategy**: Specify detection type (bubbles, OCR, barcode)
4. **Shift Management**: Track position shifts during preprocessing
5. **Drawing Coordination**: Provide drawing utilities for visualization

---

## FieldBlock Architecture

### FieldBlock Class

**Code Reference**: `src/processors/layout/field_block/base.py:18-172`

**Attributes**:
- `name` (str): Field block identifier (e.g., "q1-10", "Roll")
- `origin` (list[int]): Top-left position [x, y]
- `shifts` (list[int]): Current position shift [dx, dy]
- `direction` (str): Layout direction ("vertical" or "horizontal")
- `empty_value` (str): Value for unanswered fields
- `field_detection_type` (str): Detection strategy
- `fields` (list[Field]): Generated field objects
- `parsed_field_labels` (list[str]): Parsed field names
- `labels_gap` (int): Vertical gap between fields
- `drawing` (FieldBlockDrawing): Drawing utilities

**Type-Specific Attributes** (BUBBLES_THRESHOLD):
- `bubble_dimensions` (list[int]): Bubble size [width, height]
- `bubble_values` (list[str]): Possible values (e.g., ["A", "B", "C", "D"])
- `bubbles_gap` (int): Horizontal gap between bubbles
- `bubble_field_type` (str): Bubble field type name
- `alignment` (dict): Alignment configuration

**Type-Specific Attributes** (OCR):
- `scan_zone` (list[int]): OCR scan area [x, y, width, height]

**Type-Specific Attributes** (BARCODE_QR):
- `scan_zone` (list[int]): Barcode scan area [x, y, width, height]

---

## Field Detection Types

### 1. BUBBLES_THRESHOLD

**Code Reference**: `src/processors/layout/field_block/base.py:115-144`

**Purpose**: Traditional bubble detection using threshold-based image processing

**Required Properties**:
- `bubble_dimensions`: Bubble size
- `bubble_values`: Answer choices
- `bubbles_gap`: Gap between bubbles
- `bubble_field_type`: Reference to custom or builtin bubble type

**Example**:
```json
{
  "origin": [200, 100],
  "fieldLabels": ["q1", "q2", "q3"],
  "labelsGap": 60,
  "bubbleValues": ["A", "B", "C", "D"],
  "bubblesGap": 40,
  "direction": "vertical",
  "fieldDetectionType": "BUBBLES_THRESHOLD",
  "bubbleFieldType": "StandardBubble",
  "bubbleDimensions": [32, 32]
}
```

**Field Class**: BubbleField

---

### 2. OCR

**Code Reference**: `src/processors/layout/field_block/base.py:146-153`

**Purpose**: Optical Character Recognition for handwritten/printed text

**Required Properties**:
- `scan_zone`: Rectangular area to scan [x, y, width, height]

**Example**:
```json
{
  "origin": [100, 500],
  "fieldLabels": ["Name"],
  "fieldDetectionType": "OCR",
  "scanZone": [100, 500, 300, 50]
}
```

**Field Class**: OCRField

**OCR Engines**: EasyOCR, Tesseract (configurable in tuning config)

---

### 3. BARCODE_QR

**Code Reference**: `src/processors/layout/field_block/base.py:155-162`

**Purpose**: Barcode and QR code detection and decoding

**Required Properties**:
- `scan_zone`: Rectangular area to scan [x, y, width, height]

**Example**:
```json
{
  "origin": [50, 50],
  "fieldLabels": ["StudentID"],
  "fieldDetectionType": "BARCODE_QR",
  "scanZone": [50, 50, 200, 200]
}
```

**Field Class**: BarcodeField

**Supported Formats**: QR, Code128, Code39, EAN13, etc. (via PyZbar)

---

## Field Generation

**Code Reference**: `src/processors/layout/field_block/base.py:164-172`

```python
def generate_fields(self) -> None:
    field_class = self.field_detection_type_to_field_class[self.field_detection_type]

    for field_label in self.parsed_field_labels:
        field = field_class(
            direction=self.direction,
            empty_value=self.empty_value,
            field_block=self,
            field_detection_type=self.field_detection_type,
            field_label=field_label,
            origin=self.compute_field_origin(field_label),
        )
        self.fields.append(field)
```

**Field Classes**:
- `BUBBLES_THRESHOLD` → BubbleField
- `OCR` → OCRField
- `BARCODE_QR` → BarcodeField

**Origin Calculation**: Based on field index, labels_gap, and direction

---

## Shift Management

### Why Shifts?

**Reason**: Preprocessors can move field blocks during image transformation

**Examples**:
- CropOnMarkers: Crop based on markers, shifts entire layout
- Alignment: Feature-based alignment shifts fields to match reference

### Shift Operations

**Code Reference**: `src/processors/layout/field_block/base.py:40-48`

```python
def reset_all_shifts(self) -> None:
    self.shifts = [0, 0]
    for field in self.fields:
        field.reset_all_shifts()

def get_shifted_origin(self):
    origin, shifts = self.origin, self.shifts
    return [origin[0] + shifts[0], origin[1] + shifts[1]]
```

**Shift Flow**:
1. Template copied before processing each file
2. Preprocessor modifies shifts (e.g., `field_block.shifts = [10, -5]`)
3. Shifted origin used for bubble detection
4. Shifts reset before next file

---

## Bubble Field Types

### Built-in Types

**Code Reference**: `src/utils/constants.py`

**StandardBubble**: Default bubble layout
```json
{
  "origin": [0, 0],
  "bubblesGap": 0,
  "labelsGap": 0,
  "fieldLabels": ["STANDARD"]
}
```

### Custom Types

**Purpose**: Reusable bubble layout templates

**Example** (in template.json):
```json
{
  "customBubbleFieldTypes": {
    "WideSpaced": {
      "origin": [0, 0],
      "bubblesGap": 60,
      "labelsGap": 80,
      "fieldLabels": ["WIDE"]
    }
  }
}
```

**Usage in Field Block**:
```json
{
  "bubbleFieldType": "WideSpaced"
}
```

**Validation**: Custom types must be defined before use (checked during template load)

---

## Field Label Parsing

**Code Reference**: `src/utils/parsing.py:parse_fields()`

**Purpose**: Parse field label patterns into individual field names

**Patterns**:
1. **Range**: `"q1-10"` → `["q1", "q2", ..., "q10"]`
2. **List**: `["Roll", "Name"]` → `["Roll", "Name"]`
3. **Mixed**: `["q1-5", "Bonus"]` → `["q1", "q2", "q3", "q4", "q5", "Bonus"]`

**Example**:
```python
parse_fields("FieldBlock q1-10", ["q1-10"])
# Returns: ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]

parse_fields("FieldBlock Roll", ["Roll1", "Roll2", "Roll3"])
# Returns: ["Roll1", "Roll2", "Roll3"]
```

---

## Serialization

**Code Reference**: `src/processors/layout/field_block/base.py:51-64`

```python
def to_json(self):
    return {
        "bubble_dimensions": self.bubble_dimensions,
        "bounding_box_dimensions": self.bounding_box_dimensions,
        "empty_value": self.empty_value,
        "fields": [field.to_json() for field in self.fields],
        "name": self.name,
        "bounding_box_origin": self.bounding_box_origin,
    }
```

**Purpose**: Export template structure for visualization and debugging

**Use Case**: Image metrics export (JavaScript object for HTML viewer)

---

## Browser Migration Notes

### TypeScript FieldBlock Class

```typescript
interface FieldBlockConfig {
    origin: [number, number];
    fieldLabels: string[];
    labelsGap?: number;
    direction: 'vertical' | 'horizontal';
    fieldDetectionType: 'BUBBLES_THRESHOLD' | 'OCR' | 'BARCODE_QR';
    emptyValue?: string;

    // BUBBLES_THRESHOLD specific
    bubbleValues?: string[];
    bubblesGap?: number;
    bubbleDimensions?: [number, number];
    bubbleFieldType?: string;

    // OCR/BARCODE specific
    scanZone?: [number, number, number, number];
}

class FieldBlock {
    name: string;
    origin: [number, number];
    shifts: [number, number] = [0, 0];
    fields: Field[] = [];

    constructor(name: string, config: FieldBlockConfig, offset: [number, number]) {
        this.name = name;
        this.origin = [config.origin[0] + offset[0], config.origin[1] + offset[1]];
        this.setupFieldBlock(config);
        this.generateFields();
    }

    resetAllShifts(): void {
        this.shifts = [0, 0];
        this.fields.forEach(field => field.resetAllShifts());
    }

    getShiftedOrigin(): [number, number] {
        return [
            this.origin[0] + this.shifts[0],
            this.origin[1] + this.shifts[1]
        ];
    }

    private generateFields(): void {
        const FieldClass = FIELD_DETECTION_TYPE_TO_CLASS[this.fieldDetectionType];

        this.parsedFieldLabels.forEach(label => {
            const field = new FieldClass({
                fieldBlock: this,
                fieldLabel: label,
                origin: this.computeFieldOrigin(label)
            });
            this.fields.push(field);
        });
    }
}
```

### Field Label Parsing in TypeScript

```typescript
function parseFields(description: string, fieldLabels: string[]): string[] {
    const parsed: string[] = [];

    for (const label of fieldLabels) {
        // Check for range pattern: "q1-10"
        const rangeMatch = label.match(/^(\w+?)(\d+)-(\d+)$/);
        if (rangeMatch) {
            const [, prefix, start, end] = rangeMatch;
            for (let i = parseInt(start); i <= parseInt(end); i++) {
                parsed.push(`${prefix}${i}`);
            }
        } else {
            parsed.push(label);
        }
    }

    return parsed;
}
```

### Zod Schema for Validation

```typescript
import { z } from 'zod';

const BubblesFieldBlockSchema = z.object({
    origin: z.tuple([z.number(), z.number()]),
    fieldLabels: z.array(z.string()),
    labelsGap: z.number().optional(),
    bubbleValues: z.array(z.string()),
    bubblesGap: z.number(),
    bubbleDimensions: z.tuple([z.number(), z.number()]),
    direction: z.enum(['vertical', 'horizontal']),
    fieldDetectionType: z.literal('BUBBLES_THRESHOLD'),
    bubbleFieldType: z.string(),
    emptyValue: z.string().optional(),
});

const OCRFieldBlockSchema = z.object({
    origin: z.tuple([z.number(), z.number()]),
    fieldLabels: z.array(z.string()),
    scanZone: z.tuple([z.number(), z.number(), z.number(), z.number()]),
    fieldDetectionType: z.literal('OCR'),
});

const FieldBlockSchema = z.union([
    BubblesFieldBlockSchema,
    OCRFieldBlockSchema,
    // ... BarcodeFieldBlockSchema
]);
```

---

## Summary

**FieldBlock Entity**: Logical grouping of related fields
**Detection Types**: BUBBLES_THRESHOLD, OCR, BARCODE_QR
**Field Generation**: Automatic from field label patterns
**Shift Management**: Track position changes during preprocessing
**Custom Types**: Reusable bubble field type templates

**Browser Migration**:
- Use TypeScript interfaces for type safety
- Validate with Zod schemas
- Implement same shift management logic
- Parse field label ranges consistently

**Key Takeaway**: FieldBlock is the bridge between template configuration (JSON) and runtime field objects. Browser version should maintain same structure and validation logic.
