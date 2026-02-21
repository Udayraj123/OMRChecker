# Barcode Detection - Decision Points

**Module**: Barcode Detection
**Python Reference**: `src/processors/detection/barcode/`
**Focus**: Configuration choices, format selection, threshold logic

---

## Decision 1: Library Selection - PyZBar

### Question
Which barcode decoding library should be used for Python implementation?

### Options Considered

| Library | Pros | Cons | Decision |
|---------|------|------|----------|
| **PyZBar** | - Fast (C library)<br>- Supports 1D/2D barcodes<br>- Stable, mature<br>- No ML required | - Requires system dependencies<br>- Limited format support vs. ZXing | **SELECTED** |
| ZBar (direct) | - Maximum performance | - Complex C integration<br>- No Python bindings | Rejected |
| OpenCV Barcode | - Already in dependencies<br>- Integrated with CV pipeline | - Limited format support<br>- Newer, less stable | Rejected |
| ZXing-CPP | - More formats than ZBar<br>- Better QR support | - Harder to install<br>- Python bindings immature | Future option |
| ML-based (custom) | - Handles damaged barcodes<br>- Custom formats | - Training required<br>- Slow inference<br>- Overkill for standard barcodes | Rejected |

### Decision: PyZBar

**Rationale**:
1. **Performance**: C library provides 10-50ms decode times
2. **Reliability**: Mature library with extensive barcode format support
3. **Ease of Use**: Simple Python wrapper with minimal configuration
4. **OMR Use Case**: Standard barcodes (QR, CODE-128) are well-supported
5. **No Training**: Works out-of-the-box without ML models

**Implementation**:
```python
from pyzbar.pyzbar import decode

# Lazy loading to avoid import errors if not installed
class PyZBar:
    decode_barcode = None

    @staticmethod
    def initialize():
        from pyzbar.pyzbar import decode as decode_barcode
        PyZBar.decode_barcode = decode_barcode
```

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get install libzbar0

# macOS
brew install zbar

# Python package
pip install pyzbar
```

---

## Decision 2: Single vs. Multiple Barcodes Per Field

### Question
Should a single barcode field support detecting multiple barcodes?

### Options

#### Option A: Single Barcode Only (CURRENT)
```python
text_detection = PyZBar.get_single_text_detection(
    image_zone,
    confidence_threshold=0.8
)
# Returns: TextDetection or None
```

**Pros**:
- Simple, clear semantics (one field = one value)
- Faster (stop after first detection)
- Easier error handling (no ambiguity)

**Cons**:
- Can't handle fields with multiple barcodes
- Ignores secondary barcodes (potential data loss)

#### Option B: Multiple Barcodes with Concatenation (FUTURE)
```python
text_detections = PyZBar.get_all_text_detections(
    image_zone,
    confidence_threshold=0.8
)
# Returns: List[TextDetection]

field_value = separator.join([d.detected_text for d in text_detections])
# Example: "ABC,DEF,GHI"
```

**Pros**:
- Supports complex fields (e.g., multiple serial numbers)
- Captures all data in scan zone
- Flexible concatenation options

**Cons**:
- Complex configuration (separator, max count)
- Order dependency (which barcode first?)
- Higher chance of false positives

### Decision: Single Barcode (with Future Multi-Barcode Support)

**Current Implementation**: `get_single_text_detection()` only
- Most OMR use cases: 1 barcode per field (student ID, roll number)
- Simpler to configure and debug
- Better performance (early exit)

**Future Enhancement**: Add configuration option
```json
{
    "fieldBlocks": {
        "RollNumber": {
            "fieldDetectionType": "BARCODE_QR",
            "barcodeConfig": {
                "maxBarcodes": 1,  // Default
                "concatenationEnabled": false
            }
        },
        "SerialNumbers": {
            "fieldDetectionType": "BARCODE_QR",
            "barcodeConfig": {
                "maxBarcodes": 3,
                "concatenationEnabled": true,
                "concatenationSeparator": ","
            }
        }
    }
}
```

---

## Decision 3: Confidence Threshold - Hardcoded vs. Configurable

### Question
Should confidence threshold be hardcoded or configurable per field/template?

### Current Implementation: Hardcoded
```python
text_detection = PyZBar.get_single_text_detection(
    image_zone,
    confidence_threshold=0.8  # Hardcoded 80%
)
```

### Options

#### Option A: Hardcoded Threshold (CURRENT)
```python
CONFIDENCE_THRESHOLD = 0.8  # 80% minimum
```

**Pros**:
- Simple, no configuration needed
- Consistent behavior across all fields
- Good default for most use cases

**Cons**:
- Can't adjust for low-quality scans
- Can't be stricter for critical fields
- No per-field customization

#### Option B: Global Tuning Config
```json
{
    "tuningConfig": {
        "detection": {
            "barcode": {
                "confidenceThreshold": 0.8
            }
        }
    }
}
```

**Pros**:
- Single source of truth
- Easy to adjust for all fields
- Documented in config

**Cons**:
- All fields share same threshold
- Can't prioritize critical fields

#### Option C: Per-Field Config
```json
{
    "fieldBlocks": {
        "RollNumber": {
            "fieldDetectionType": "BARCODE_QR",
            "barcodeConfig": {
                "confidenceThreshold": 0.9  // Strict: critical field
            }
        },
        "OptionalID": {
            "fieldDetectionType": "BARCODE_QR",
            "barcodeConfig": {
                "confidenceThreshold": 0.7  // Lenient: optional field
            }
        }
    }
}
```

**Pros**:
- Maximum flexibility
- Can prioritize critical fields
- Handles varying scan quality per region

**Cons**:
- Complex configuration
- Easy to misconfigure
- Harder to debug (different thresholds)

### Decision: Hardcoded with Future Tuning Config

**Current**: `0.8` hardcoded in `get_single_text_detection()`
- Balances reliability (few false positives) with tolerance (accepts slightly damaged barcodes)
- Tested across various scan qualities
- Sufficient for most OMR sheets

**Future Enhancement**: Add to TuningConfig
```python
# src/schemas/models/config.py
class BarcodeDetectionConfig:
    confidence_threshold: float = 0.8
    retry_with_preprocessing: bool = True

# Usage
threshold = tuning_config.detection.barcode.confidence_threshold
text_detection = PyZBar.get_single_text_detection(
    image_zone, confidence_threshold=threshold
)
```

---

## Decision 4: Coordinate System - Relative vs. Absolute

### Question
Should barcode bounding boxes use scan-zone-relative or image-absolute coordinates?

### Challenge
```
Scan Zone Definition:
    origin: [100, 50]
    dimensions: [300, 80]

Barcode detected in scan zone at:
    relative_bbox: [(10, 5), (290, 5), (290, 75), (10, 75)]

Question: Store relative or absolute coordinates?
```

### Options

#### Option A: Relative Coordinates
```python
detection = BarcodeDetection(
    bounding_box=[(10, 5), (290, 5), (290, 75), (10, 75)],
    scan_zone_origin=[100, 50]  # Store origin separately
)
```

**Pros**:
- Smaller numbers (easier to debug)
- Independent of scan zone position
- Can recalculate absolute on demand

**Cons**:
- Requires scan zone context for drawing
- More complex drawing code
- Error-prone (forgot to add origin?)

#### Option B: Absolute Coordinates (SELECTED)
```python
detection = BarcodeDetection(
    bounding_box=[(110, 55), (390, 55), (390, 125), (110, 125)],
    scan_zone_rectangle=[(100,50), (400,50), (400,130), (100,130)]
)
```

**Pros**:
- Direct use in drawing (no transformation)
- Consistent with bubble detection
- Easy to visualize on full image

**Cons**:
- Larger numbers
- Tightly coupled to scan zone position

### Decision: Absolute Coordinates

**Rationale**:
1. **Consistency**: Bubble detection uses absolute coordinates
2. **Simplicity**: Drawing code directly uses stored coordinates
3. **Integration**: DetectionRepository stores absolute coordinates
4. **Standard Practice**: Most CV libraries use absolute coordinates

**Implementation**:
```python
@staticmethod
def from_scan_zone_detection(scan_zone_rectangle, text_detection: TextDetection):
    zone_start = scan_zone_rectangle[0]  # Top-left corner

    # Shift from relative to absolute
    absolute_bounding_box = MathUtils.shift_points_from_origin(
        zone_start, text_detection.bounding_box
    )
    absolute_rotated_rectangle = MathUtils.shift_points_from_origin(
        zone_start, text_detection.rotated_rectangle
    )

    return BarcodeDetection(
        scan_zone_rectangle,
        text_detection.detected_text,
        absolute_bounding_box,
        absolute_rotated_rectangle,
        text_detection.confident_score
    )
```

---

## Decision 5: Empty Value Handling

### Question
What should be returned when no barcode is detected?

### Options

#### Option A: Empty String `""` (DEFAULT)
```python
if len(marked_interpretations) == 0:
    return ""  # Empty string
```

**Pros**:
- Standard behavior (null value)
- Easy to check (`if not value`)
- JSON-friendly

**Cons**:
- Can't distinguish "not attempted" from "empty barcode"
- CSV output has blank cell

#### Option B: Null/None
```python
if len(marked_interpretations) == 0:
    return None
```

**Pros**:
- Explicit "no value" semantics
- Type-safe (Optional[str])

**Cons**:
- JSON serialization issues
- CSV output has "null" string

#### Option C: Configurable Empty Value (SELECTED)
```json
{
    "fieldBlocks": {
        "RollNumber": {
            "fieldDetectionType": "BARCODE_QR",
            "emptyValue": ""  // Default: empty string
        },
        "OptionalID": {
            "fieldDetectionType": "BARCODE_QR",
            "emptyValue": "NOT_SCANNED"  // Custom value
        }
    }
}
```

**Pros**:
- Flexible per field
- Can distinguish different empty cases
- Clear in CSV/JSON output

**Cons**:
- More configuration
- Inconsistent across fields (if misused)

### Decision: Configurable Empty Value (Default: `""`)

**Rationale**:
1. **Flexibility**: Different fields have different "empty" semantics
2. **Clarity**: "NOT_SCANNED" is clearer than "" in reports
3. **Backward Compatibility**: Default "" matches existing behavior
4. **CSV Readability**: Custom values make missing data explicit

**Implementation**:
```python
class BarcodeFieldInterpretation(FieldInterpretation):
    def __init__(self, tuning_config, field, ...):
        self.empty_value = field.empty_value  # From template JSON

    def get_field_interpretation_string(self):
        if len(marked_interpretations) == 0:
            return self.empty_value
        return "".join(marked_interpretations)
```

**Template JSON**:
```json
{
    "fieldBlocks": {
        "RollNumber": {
            "emptyValue": "",  // Default
            ...
        }
    }
}
```

---

## Decision 6: Text Postprocessing - When and How

### Question
Should detected barcode text be cleaned/filtered before returning?

### Raw PyZBar Output
```python
result.data = b'  STU-2024-12345\x00\n  '  # Bytes with whitespace/control chars
```

### Processing Options

#### Option A: No Processing (Raw)
```python
detected_text = result.data.decode('utf-8')
# Returns: "  STU-2024-12345\x00\n  "
```

**Pros**: Preserves original data
**Cons**: Unusable (whitespace, control chars)

#### Option B: Minimal Processing
```python
detected_text = result.data.decode('utf-8').strip()
# Returns: "STU-2024-12345\x00"
```

**Pros**: Removes leading/trailing whitespace
**Cons**: Still has control characters

#### Option C: Full Processing (SELECTED)
```python
# 1. Strip whitespace
stripped_text = text.strip()

# 2. Remove non-ASCII characters
printable_text = "".join([c for c in stripped_text if ord(c) < 128])

# 3. Collapse multiple spaces (optional)
if clear_whitespace:
    cleaned_text = re.sub(r'\s{2,}', ' ', printable_text)

# Returns: "STU-2024-12345"
```

**Pros**:
- Clean, usable output
- Handles barcode scanner artifacts
- Consistent text format

**Cons**:
- May remove intentional characters
- Non-ASCII data lost (e.g., UTF-8 names)

### Decision: Full Processing with Options

**Current Implementation**:
```python
def postprocess_text(text, clear_whitespace=True, max_length=None, charset=None):
    # 1. Strip
    stripped_text = text.strip()

    # 2. Remove non-ASCII
    printable_text = "".join([c for c in stripped_text if ord(c) < 128])

    # 3. Collapse whitespace
    if clear_whitespace:
        cleaned_text = re.sub(r'\s{2,}', ' ', printable_text)

    # 4. Filter by charset (optional)
    if charset is not None:
        cleaned_text = "".join([c for c in cleaned_text if c in charset])

    # 5. Clip to max length (optional)
    if max_length is not None and len(cleaned_text) > max_length:
        cleaned_text = cleaned_text[:max_length]

    return cleaned_text
```

**Rationale**:
1. **Strip**: Always remove leading/trailing whitespace
2. **ASCII-only**: OMR sheets typically use ASCII (student IDs, roll numbers)
3. **Whitespace Collapse**: Handles scanner artifacts (double spaces)
4. **Optional Filters**: Charset/length for specific use cases

**Future Enhancement**: Configurable processing
```json
{
    "barcodeConfig": {
        "textProcessing": {
            "stripWhitespace": true,
            "asciiOnly": true,
            "collapseSpaces": true,
            "charset": "0123456789-",  // Optional: numeric + hyphen only
            "maxLength": 50
        }
    }
}
```

---

## Decision 7: Format Selection - Auto-detect vs. Configured

### Question
Should PyZBar auto-detect all barcode formats, or limit to specific formats?

### Current Implementation: Auto-detect All Formats
```python
# PyZBar automatically tries all supported formats
results = decode(image)
# Detects: QR, CODE-128, EAN-13, etc.
```

### Options

#### Option A: Auto-detect All (CURRENT)
```python
results = decode(image)  # No format hints
```

**Pros**:
- Works with any barcode type
- No configuration needed
- Maximum flexibility

**Cons**:
- Slower (tries all formats)
- May detect wrong format (ambiguity)
- Can't prioritize specific formats

#### Option B: Format Allowlist
```python
# Configuration
{
    "barcodeConfig": {
        "allowedFormats": ["QRCODE", "CODE128"]
    }
}

# Implementation
results = decode(image, symbols=[ZBarSymbol.QRCODE, ZBarSymbol.CODE128])
```

**Pros**:
- Faster (fewer formats to try)
- Prevents unwanted formats
- More predictable

**Cons**:
- Requires configuration
- May miss valid barcodes
- Format names must match PyZBar

#### Option C: Format Priority
```python
# Configuration
{
    "barcodeConfig": {
        "formatPriority": ["QRCODE", "CODE128", "EAN13"]
    }
}

# Implementation
for format_type in format_priority:
    results = decode(image, symbols=[format_type])
    if results:
        return results[0]
```

**Pros**:
- Tries preferred formats first
- Falls back to others if needed
- Balances speed and flexibility

**Cons**:
- More complex logic
- Multiple decode attempts (slower)

### Decision: Auto-detect All (with Future Format Filtering)

**Current**: Auto-detect all formats
- Simple, works for all OMR use cases
- Performance is acceptable (~10-50ms)
- Most OMR sheets use QR or CODE-128 (fast to detect)

**Future Enhancement**: Add format hints to config
```python
# src/processors/detection/barcode/detection.py
def run_detection(self, field, gray_image, colored_image):
    # TODO: access field config to determine which lib to use + lib level config
    # if self.library == BARCODE_LIBS.PYZBAR:
    #     allowed_formats = field.barcode_config.get("allowedFormats")
    #     text_detection = PyZBar.get_single_text_detection(
    #         image_zone,
    #         confidence_threshold=0.8,
    #         allowed_formats=allowed_formats  # Future
    #     )
```

**Rationale**:
1. **YAGNI**: Most users don't need format filtering
2. **Performance**: Auto-detect is fast enough for OMR
3. **Simplicity**: No configuration needed
4. **Future-proof**: Easy to add format hints later

---

## Decision 8: Multi-Marking Detection

### Question
How should multiple barcodes in a single scan zone be handled?

### Current Implementation: Flag as Multi-Marked
```python
marked_interpretations = [
    interpretation.get_value()
    for interpretation in self.interpretations
    if interpretation.is_attempted
]
self.is_multi_marked = len(marked_interpretations) > 1
```

### Options

#### Option A: Return First Only
```python
if marked_interpretations:
    return marked_interpretations[0]  # Ignore others
```

**Pros**: Simple, predictable
**Cons**: Data loss (ignores secondary barcodes)

#### Option B: Concatenate All
```python
return "".join(marked_interpretations)
# Example: "ABC" + "DEF" = "ABCDEF"
```

**Pros**: Captures all data
**Cons**: Ambiguous (where does one end?)

#### Option C: Flag as Error (CURRENT)
```python
self.is_multi_marked = True  # Flag for review
return "".join(marked_interpretations)  # But still return data
```

**Pros**:
- Preserves data
- Alerts user to ambiguity
- Can be handled in evaluation

**Cons**:
- Requires manual review
- May not be an error (intentional multi-barcode)

### Decision: Flag as Multi-Marked + Concatenate

**Rationale**:
1. **Preserve Data**: Don't lose information
2. **Alert User**: Multi-marking is unusual, flag for review
3. **Flexible Handling**: Downstream can decide (accept/reject)
4. **Consistent with Bubbles**: Bubble detection also flags multi-marking

**Implementation**:
```python
self.is_multi_marked = len(marked_interpretations) > 1

# Return concatenated value
field_value = "".join(marked_interpretations)

# Downstream handling
if is_multi_marked:
    logger.warning(f"Multiple barcodes detected in field {field_label}")
    # Evaluation can mark as incorrect or partial credit
```

---

## Browser Migration Decisions

### Decision: @zxing/library vs. Alternatives

**Options**:
- **@zxing/library** (ZXing JavaScript port)
- **jsQR** (QR-only, lighter)
- **quagga2** (1D barcodes only)
- **Native Barcode Detection API** (experimental)

**Decision**: @zxing/library

**Rationale**:
1. **Comprehensive**: Supports 1D and 2D barcodes
2. **Mature**: Actively maintained, production-ready
3. **PyZBar Equivalent**: Closest feature parity
4. **TypeScript**: Full type definitions
5. **Performance**: Acceptable with Web Workers

**Trade-offs**:
- **Size**: Larger bundle (~400KB) vs. jsQR (~50KB)
- **No Quality Score**: ZXing returns success/failure, not confidence
- **Async Only**: All methods return Promises

---

## Summary Table

| Decision | Current Choice | Future Enhancement | Rationale |
|----------|---------------|-------------------|-----------|
| Library | PyZBar | ZXing-CPP for more formats | Performance, stability |
| Barcode Count | Single only | Multi-barcode with config | Simplicity for common case |
| Confidence | 0.8 hardcoded | TuningConfig option | Good default, future flexibility |
| Coordinates | Absolute | - | Consistency with bubbles |
| Empty Value | Configurable (default "") | - | Flexibility per field |
| Text Processing | Full cleanup | Configurable filters | Clean output, handle artifacts |
| Format Selection | Auto-detect all | Format hints in config | Works for all, fast enough |
| Multi-Marking | Flag + concatenate | Configuration per field | Preserve data, alert user |
| Browser Library | @zxing/library | Native API when stable | Comprehensive support |

**Key Principles**:
1. **Start Simple**: Hardcoded defaults for common cases
2. **Future-Proof**: Clear paths to add configuration
3. **Preserve Data**: Don't lose information (multi-marking, text)
4. **Consistency**: Match bubble detection patterns where possible
5. **Browser-First**: Design with migration in mind
