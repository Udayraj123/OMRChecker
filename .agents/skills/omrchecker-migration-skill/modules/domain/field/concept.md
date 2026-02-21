# Field Entity - Concept

**Module**: Domain / Field
**Python Reference**: `src/processors/layout/field/base.py`, `src/processors/layout/field/bubble_field.py`
**Last Updated**: 2026-02-20

---

## Overview

A Field represents a single question or data field on an OMR sheet. It contains one or more ScanBoxes (bubbles, OCR zones, barcode zones) and manages field-level detection and interpretation logic.

**Key Responsibilities**:
1. **ScanBox Generation**: Create scan boxes based on field type and layout
2. **Field Identity**: Maintain field label and unique ID
3. **Detection Coordination**: Coordinate detection across scan boxes
4. **Shift Management**: Propagate shifts to scan boxes
5. **Drawing Support**: Provide drawing utilities for visualization

---

## Field Architecture

### Field Base Class

**Code Reference**: `src/processors/layout/field/base.py:7-68`

**Attributes**:
- `field_label` (str): Field identifier (e.g., "q1", "Roll", "Name")
- `id` (str): Unique identifier (e.g., "q1-10::q5")
- `name` (str): Display name (same as field_label)
- `origin` (list[int]): Top-left position [x, y]
- `direction` (str): Layout direction ("vertical" or "horizontal")
- `empty_value` (str): Value when field is unanswered
- `field_block` (FieldBlock): Parent field block reference
- `field_detection_type` (str): Detection strategy
- `scan_boxes` (list[ScanBox]): Scan box objects (bubbles, zones)
- `drawing` (FieldDrawing): Drawing utilities

---

## Field Types

### 1. BubbleField

**Code Reference**: `src/processors/layout/field/bubble_field.py`

**Purpose**: Field with multiple choice bubbles

**ScanBox Generation**: Creates one scan box per bubble value

**Example**:
- Field: "q1"
- Bubble values: ["A", "B", "C", "D"]
- Scan boxes: 4 (one for each answer choice)

**Layout Calculation**:
```python
for i, bubble_value in enumerate(bubble_values):
    bubble_origin = [
        origin[0] + i * (bubble_width + bubbles_gap),  # Horizontal spacing
        origin[1]
    ]
    scan_box = ScanBox(i, self, bubble_origin, bubble_dimensions, margins)
    self.scan_boxes.append(scan_box)
```

**Detection**: Threshold-based bubble detection (darkness measurement)

---

### 2. OCRField

**Code Reference**: `src/processors/layout/field/ocr_field.py`

**Purpose**: Field for text recognition

**ScanBox Generation**: Single scan box covering the entire OCR zone

**Example**:
- Field: "Name"
- Scan zone: [100, 500, 300, 50]
- Scan boxes: 1 (entire zone)

**Detection**: EasyOCR or Tesseract OCR engine

---

### 3. BarcodeField

**Code Reference**: `src/processors/layout/field/barcode_field.py`

**Purpose**: Field for barcode/QR code reading

**ScanBox Generation**: Single scan box covering barcode area

**Example**:
- Field: "StudentID"
- Scan zone: [50, 50, 200, 200]
- Scan boxes: 1 (entire barcode area)

**Detection**: PyZbar barcode decoder (Python) / zxing-js (Browser)

---

## ScanBox Entity

**Code Reference**: `src/processors/layout/field/base.py:71-122`

**Purpose**: The smallest unit in template layout (individual bubble, OCR zone, or barcode area)

**Attributes**:
- `field_index` (int): Index within field (0, 1, 2, ...)
- `name` (str): Unique name (e.g., "q1_0", "q1_1", "q1_2", "q1_3")
- `field_label` (str): Parent field label
- `field_detection_type` (str): Detection type
- `origin` (list[int]): Top-left position [x, y]
- `x` (int): Rounded x position
- `y` (int): Rounded y position
- `dimensions` (list[int]): Size [width, height]
- `margins` (dict): Margins {top, bottom, left, right}
- `shifts` (list[int]): Position shift [dx, dy]
- `field` (Field): Parent field reference

---

## Field vs FieldBlock vs ScanBox

**Hierarchy**:
```
Template
└── FieldBlock (e.g., "q1-10")
    └── Field (e.g., "q5")
        └── ScanBox (e.g., "q5_0", "q5_1", "q5_2", "q5_3")
```

**Responsibilities**:
- **FieldBlock**: Layout strategy for group of fields
- **Field**: Single question logic
- **ScanBox**: Individual bubble/zone detection

**Example** (Multiple Choice):
```
FieldBlock: "q1-10"
  ├── Field: "q1" (label: "q1")
  │   ├── ScanBox: "q1_0" (bubble A)
  │   ├── ScanBox: "q1_1" (bubble B)
  │   ├── ScanBox: "q1_2" (bubble C)
  │   └── ScanBox: "q1_3" (bubble D)
  ├── Field: "q2" (label: "q2")
  │   ├── ScanBox: "q2_0" (bubble A)
  │   └── ... (B, C, D)
  └── ... (q3-q10)
```

---

## Field ID Format

**Pattern**: `{field_block_name}::{field_label}`

**Examples**:
- `"q1-10::q5"` (field "q5" in block "q1-10")
- `"Roll::Roll"` (field "Roll" in block "Roll")
- `"Names::FirstName"` (field "FirstName" in block "Names")

**Usage**: Unique identification for field-level operations

---

## Shift Propagation

**Code Reference**: `src/processors/layout/field/base.py:49-52`

```python
def reset_all_shifts(self) -> None:
    for bubble in self.scan_boxes:
        bubble.reset_shifts()
```

**Flow**:
1. Template → FieldBlock shifts reset
2. FieldBlock → Field shifts reset (propagates)
3. Field → ScanBox shifts reset

**Why Needed**: Preprocessors shift FieldBlocks, which affects all Fields and ScanBoxes

---

## Shifted Position Calculation

**Code Reference**: `src/processors/layout/field/base.py:101-107`

```python
def get_shifted_position(self, shifts=None):
    if shifts is None:
        shifts = self.field.field_block.shifts
    return [
        self.x + self.shifts[0] + shifts[0],
        self.y + self.shifts[1] + shifts[1],
    ]
```

**Position Hierarchy**:
1. Original position: `(x, y)`
2. ScanBox shift: `(x + scan_box.shifts[0], y + scan_box.shifts[1])`
3. FieldBlock shift: `(x + scan_box.shifts[0] + field_block.shifts[0], ...)`

**Use Case**: Detect bubbles at shifted positions after preprocessing

---

## Serialization

**Code Reference**: `src/processors/layout/field/base.py:58-68`

```python
def to_json(self):
    return {
        "id": self.id,
        "field_label": self.field_label,
        "field_detection_type": self.field_detection_type,
        "direction": self.direction,
        "scan_boxes": [box.to_json() for box in self.scan_boxes],
    }
```

**Purpose**: Export for visualization and debugging

**Output Format**: JSON object with nested scan boxes

---

## Browser Migration Notes

### TypeScript Field Classes

```typescript
abstract class Field {
    fieldLabel: string;
    id: string;
    name: string;
    origin: [number, number];
    direction: 'vertical' | 'horizontal';
    emptyValue: string;
    fieldBlock: FieldBlock;
    fieldDetectionType: string;
    scanBoxes: ScanBox[] = [];

    constructor(config: FieldConfig) {
        this.fieldLabel = config.fieldLabel;
        this.id = `${config.fieldBlock.name}::${config.fieldLabel}`;
        this.name = config.fieldLabel;
        this.origin = config.origin;
        this.direction = config.direction;
        this.emptyValue = config.emptyValue;
        this.fieldBlock = config.fieldBlock;
        this.fieldDetectionType = config.fieldDetectionType;

        this.setupScanBoxes();
    }

    abstract setupScanBoxes(): void;

    resetAllShifts(): void {
        this.scanBoxes.forEach(box => box.resetShifts());
    }

    toJSON(): any {
        return {
            id: this.id,
            fieldLabel: this.fieldLabel,
            fieldDetectionType: this.fieldDetectionType,
            direction: this.direction,
            scanBoxes: this.scanBoxes.map(box => box.toJSON())
        };
    }
}

class BubbleField extends Field {
    setupScanBoxes(): void {
        const { bubbleValues, bubblesGap, bubbleDimensions } = this.fieldBlock;

        bubbleValues.forEach((value, i) => {
            const bubbleOrigin: [number, number] = [
                this.origin[0] + i * (bubbleDimensions[0] + bubblesGap),
                this.origin[1]
            ];

            const scanBox = new ScanBox({
                fieldIndex: i,
                field: this,
                origin: bubbleOrigin,
                dimensions: bubbleDimensions,
                margins: { top: 0, bottom: 0, left: 0, right: 0 }
            });

            this.scanBoxes.push(scanBox);
        });
    }
}

class OCRField extends Field {
    setupScanBoxes(): void {
        const scanZone = this.fieldBlock.scanZone;
        const scanBox = new ScanBox({
            fieldIndex: 0,
            field: this,
            origin: [scanZone[0], scanZone[1]],
            dimensions: [scanZone[2], scanZone[3]],
            margins: { top: 0, bottom: 0, left: 0, right: 0 }
        });
        this.scanBoxes.push(scanBox);
    }
}
```

### ScanBox TypeScript

```typescript
interface ScanBoxConfig {
    fieldIndex: number;
    field: Field;
    origin: [number, number];
    dimensions: [number, number];
    margins: { top: number; bottom: number; left: number; right: number };
}

class ScanBox {
    fieldIndex: number;
    name: string;
    fieldLabel: string;
    fieldDetectionType: string;
    origin: [number, number];
    x: number;
    y: number;
    dimensions: [number, number];
    margins: { top: number; bottom: number; left: number; right: number };
    shifts: [number, number] = [0, 0];
    field: Field;

    constructor(config: ScanBoxConfig) {
        this.fieldIndex = config.fieldIndex;
        this.field = config.field;
        this.fieldLabel = config.field.fieldLabel;
        this.fieldDetectionType = config.field.fieldDetectionType;
        this.origin = config.origin;
        this.x = Math.round(config.origin[0]);
        this.y = Math.round(config.origin[1]);
        this.dimensions = config.dimensions;
        this.margins = config.margins;
        this.name = `${this.fieldLabel}_${this.fieldIndex}`;
    }

    resetShifts(): void {
        this.shifts = [0, 0];
    }

    getShiftedPosition(shifts?: [number, number]): [number, number] {
        const fieldBlockShifts = shifts || this.field.fieldBlock.shifts;
        return [
            this.x + this.shifts[0] + fieldBlockShifts[0],
            this.y + this.shifts[1] + fieldBlockShifts[1]
        ];
    }

    toJSON(): any {
        return {
            fieldLabel: this.fieldLabel,
            fieldDetectionType: this.fieldDetectionType,
            name: this.name,
            x: this.x,
            y: this.y,
            origin: this.origin
        };
    }
}
```

### Factory Pattern for Field Creation

```typescript
const FIELD_DETECTION_TYPE_TO_CLASS = {
    'BUBBLES_THRESHOLD': BubbleField,
    'OCR': OCRField,
    'BARCODE_QR': BarcodeField
};

function createField(config: FieldConfig): Field {
    const FieldClass = FIELD_DETECTION_TYPE_TO_CLASS[config.fieldDetectionType];
    if (!FieldClass) {
        throw new Error(`Unknown field detection type: ${config.fieldDetectionType}`);
    }
    return new FieldClass(config);
}
```

---

## Summary

**Field Entity**: Single question/data field on OMR sheet
**Types**: BubbleField, OCRField, BarcodeField
**ScanBox**: Smallest unit (individual bubble, OCR zone, barcode area)
**Hierarchy**: Template → FieldBlock → Field → ScanBox
**Shift Management**: Hierarchical shift propagation from FieldBlock to ScanBox

**Browser Migration**:
- Use TypeScript abstract class for Field base
- Implement concrete classes for each detection type
- Use factory pattern for field creation
- Maintain same shift calculation logic
- Preserve serialization format

**Key Takeaway**: Field bridges FieldBlock layout and ScanBox-level detection. Browser version should maintain same hierarchy and shift propagation logic.
