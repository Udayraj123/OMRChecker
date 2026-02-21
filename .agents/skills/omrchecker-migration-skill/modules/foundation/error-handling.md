# Error Handling Patterns

**Module**: Foundation
**Python Reference**: `src/exceptions.py`
**Last Updated**: 2026-02-20

---

## Overview

OMRChecker uses a comprehensive custom exception hierarchy for structured error handling. All exceptions inherit from a base `OMRCheckerError` class, allowing unified error handling and rich context information.

**Key Principles**:
1. **Hierarchical exceptions**: Base exception per category (Input, Processing, Template, etc.)
2. **Rich context**: Every exception includes message + context dictionary
3. **Typed exceptions**: Specific exception types for specific error scenarios
4. **Path-aware**: Most exceptions track file paths for debugging
5. **Reason tracking**: Optional reason field for detailed error information

---

## Exception Hierarchy

```
OMRCheckerError (base)
├── InputError
│   ├── InputDirectoryNotFoundError
│   ├── InputFileNotFoundError
│   └── ImageReadError
├── OutputError
│   ├── OutputDirectoryError
│   └── FileWriteError
├── ValidationError
│   ├── TemplateValidationError
│   ├── ConfigValidationError
│   ├── EvaluationValidationError
│   └── SchemaValidationError
├── ProcessingError
│   ├── MarkerDetectionError
│   ├── ImageProcessingError
│   ├── AlignmentError
│   ├── BubbleDetectionError
│   ├── OCRError
│   └── BarcodeDetectionError
├── TemplateError
│   ├── TemplateNotFoundError
│   ├── TemplateLoadError
│   ├── PreprocessorError
│   └── FieldDefinitionError
├── EvaluationError
│   ├── EvaluationConfigNotFoundError
│   ├── EvaluationConfigLoadError
│   ├── AnswerKeyError
│   └── ScoringError
├── SecurityError
│   ├── PathTraversalError
│   └── FileSizeLimitError
└── ConfigError
    ├── ConfigNotFoundError
    ├── ConfigLoadError
    └── InvalidConfigValueError
```

**Total Exception Types**: 29 (1 base + 28 specific)

---

## Base Exception Class

### OMRCheckerError

**Code Reference**: `src/exceptions.py:11-38`

**Purpose**: Root exception for all OMRChecker errors

**Attributes**:
- `message` (str): Human-readable error message
- `context` (dict): Additional context information

**Methods**:
- `__init__(message, context=None)`: Initialize with message and optional context
- `__str__()`: Format message with context for display

**Usage Pattern**:
```python
raise OMRCheckerError(
    "Something went wrong",
    context={"detail1": value1, "detail2": value2}
)
```

**String Representation**:
```python
# Without context
str(exception) → "Something went wrong"

# With context
str(exception) → "Something went wrong (detail1=value1, detail2=value2)"
```

---

## Exception Categories

### 1. Input/Output Exceptions

#### InputDirectoryNotFoundError
**Code Reference**: `src/exceptions.py:50-62`

**When Raised**: Specified input directory does not exist

**Attributes**:
- `path` (Path): Directory path that was not found

**Usage**:
```python
if not input_dir.exists():
    raise InputDirectoryNotFoundError(input_dir)
```

**Browser Migration**: Use File API error handling for missing directories

---

#### InputFileNotFoundError
**Code Reference**: `src/exceptions.py:65-81`

**When Raised**: Expected input file not found

**Attributes**:
- `path` (Path): File path that was not found
- `file_type` (str | None): Optional file type description

**Usage**:
```python
if not template_path.exists():
    raise InputFileNotFoundError(template_path, file_type="template")
```

---

#### ImageReadError
**Code Reference**: `src/exceptions.py:84-99`

**When Raised**: Image file cannot be read or decoded

**Attributes**:
- `path` (Path): Image file path
- `reason` (str | None): Why the image couldn't be read

**Usage**:
```python
image = cv2.imread(str(file_path))
if image is None:
    raise ImageReadError(file_path, reason="File is corrupt or not an image")
```

**Browser Migration**: Handle Image/Canvas loading errors

---

#### OutputDirectoryError
**Code Reference**: `src/exceptions.py:106-121`

**When Raised**: Issues with output directory operations

**Attributes**:
- `path` (Path): Output directory path
- `reason` (str): Reason for error

**Browser Migration**: Not applicable (browser uses downloads/blobs)

---

#### FileWriteError
**Code Reference**: `src/exceptions.py:124-139`

**When Raised**: File cannot be written

**Attributes**:
- `path` (Path): File path
- `reason` (str | None): Reason for failure

**Browser Migration**: Handle Blob/download errors

---

### 2. Validation Exceptions

#### TemplateValidationError
**Code Reference**: `src/exceptions.py:151-173`

**When Raised**: Template JSON fails validation

**Attributes**:
- `path` (Path): Template file path
- `errors` (list[str]): List of validation error messages
- `reason` (str | None): General reason for validation failure

**Usage**:
```python
errors = validate_template_schema(template_data)
if errors:
    raise TemplateValidationError(
        template_path,
        errors=errors,
        reason="Schema validation failed"
    )
```

**Browser Migration**: Use Zod/JSON Schema validator, display errors in UI

---

#### ConfigValidationError
**Code Reference**: `src/exceptions.py:176-198`

**When Raised**: Config JSON fails validation

**Similar to TemplateValidationError**, but for config files

---

#### EvaluationValidationError
**Code Reference**: `src/exceptions.py:201-223`

**When Raised**: Evaluation JSON fails validation

**Similar to TemplateValidationError**, but for evaluation files

---

#### SchemaValidationError
**Code Reference**: `src/exceptions.py:226-252`

**When Raised**: Data fails schema validation (generic)

**Attributes**:
- `schema_name` (str): Name of schema that failed
- `errors` (list[str]): Validation error messages
- `data_path` (Path | None): Optional data file path

**Usage**:
```python
try:
    validate(data, schema)
except jsonschema.ValidationError as e:
    raise SchemaValidationError(
        "ConfigSchema",
        errors=[str(e)],
        data_path=config_path
    )
```

---

### 3. Processing Exceptions

#### MarkerDetectionError
**Code Reference**: `src/exceptions.py:264-279`

**When Raised**: Markers cannot be detected on OMR sheet

**Attributes**:
- `file_path` (Path): OMR image file
- `reason` (str | None): Reason for detection failure

**Common Reasons**:
- Markers not visible
- Poor image quality
- Incorrect marker type in config
- Image too dark/light

**Usage**:
```python
markers = detect_markers(image)
if markers is None or len(markers) < 4:
    raise MarkerDetectionError(
        file_path,
        reason=f"Expected 4 markers, found {len(markers) if markers else 0}"
    )
```

**Error Code**: `NO_MARKER_ERR`

**File Movement**: Moved to errors/ directory

---

#### ImageProcessingError
**Code Reference**: `src/exceptions.py:282-310`

**When Raised**: Image processing operations fail

**Attributes**:
- `operation` (str): Name of failed operation
- `file_path` (Path | None): Image being processed
- `reason` (str | None): Failure reason

**Common Operations**:
- "rotation"
- "cropping"
- "warping"
- "thresholding"
- "filtering"

**Usage**:
```python
try:
    warped = cv2.warpPerspective(image, matrix, dimensions)
except cv2.error as e:
    raise ImageProcessingError(
        "warping",
        file_path=file_path,
        reason=str(e)
    )
```

---

#### AlignmentError
**Code Reference**: `src/exceptions.py:313-328`

**When Raised**: Image alignment fails

**Attributes**:
- `file_path` (Path): Image file
- `reason` (str | None): Alignment failure reason

**Common Reasons**:
- Insufficient feature matches
- No homography found
- RANSAC failure
- Template mismatch

**Usage**:
```python
if len(good_matches) < MIN_MATCHES:
    raise AlignmentError(
        file_path,
        reason=f"Insufficient matches: {len(good_matches)} < {MIN_MATCHES}"
    )
```

---

#### BubbleDetectionError
**Code Reference**: `src/exceptions.py:331-359`

**When Raised**: Bubble detection fails

**Attributes**:
- `file_path` (Path): OMR image
- `field_id` (str | None): Field where detection failed
- `reason` (str | None): Failure reason

**Common Reasons**:
- No bubbles found in field
- Invalid bubble dimensions
- Threshold failure

---

#### OCRError
**Code Reference**: `src/exceptions.py:362-390`

**When Raised**: OCR processing fails

**Attributes**:
- `file_path` (Path): Image
- `field_id` (str | None): Field where OCR failed
- `reason` (str | None): Failure reason

**Common Reasons**:
- Text not detected
- Low confidence
- OCR library error

---

#### BarcodeDetectionError
**Code Reference**: `src/exceptions.py:393-421`

**When Raised**: Barcode detection fails

**Attributes**:
- `file_path` (Path): Image
- `field_id` (str | None): Field where barcode detection failed
- `reason` (str | None): Failure reason

**Common Reasons**:
- Barcode not detected
- Unsupported barcode format
- Low quality image

---

### 4. Template Exceptions

#### TemplateNotFoundError
**Code Reference**: `src/exceptions.py:433-446`

**When Raised**: template.json not found in directory tree

**Attributes**:
- `search_path` (Path): Directory searched

**Usage**:
```python
if not template_path.exists():
    raise TemplateNotFoundError(current_directory)
```

---

#### TemplateLoadError
**Code Reference**: `src/exceptions.py:449-464`

**When Raised**: Template file cannot be loaded or parsed

**Attributes**:
- `path` (Path): Template file
- `reason` (str): Load failure reason

**Common Reasons**:
- Invalid JSON syntax
- File encoding issues
- Permission denied

---

#### PreprocessorError
**Code Reference**: `src/exceptions.py:467-498`

**When Raised**: Preprocessor operation fails

**Attributes**:
- `preprocessor_name` (str): Name of failed preprocessor
- `file_path` (Path | None): File being processed
- `reason` (str | None): Failure reason

**Common Preprocessor Names**:
- "AutoRotate"
- "CropOnMarkers"
- "CropPage"
- "GaussianBlur"
- "Contrast"
- "FeatureBasedAlignment"

**Usage**:
```python
try:
    filtered_image = apply_gaussian_blur(image, kernel_size)
except Exception as e:
    raise PreprocessorError(
        "GaussianBlur",
        file_path=file_path,
        reason=str(e)
    )
```

---

#### FieldDefinitionError
**Code Reference**: `src/exceptions.py:501-527`

**When Raised**: Issue with field definition in template

**Attributes**:
- `field_id` (str): Problematic field ID
- `reason` (str): Issue description
- `template_path` (Path | None): Template file

**Common Reasons**:
- Invalid field coordinates
- Overlapping fields
- Missing required properties
- Invalid bubble values

---

### 5. Evaluation Exceptions

#### EvaluationConfigNotFoundError
**Code Reference**: `src/exceptions.py:539-552`

**When Raised**: evaluation.json expected but not found

---

#### EvaluationConfigLoadError
**Code Reference**: `src/exceptions.py:555-570`

**When Raised**: Evaluation config cannot be loaded

---

#### AnswerKeyError
**Code Reference**: `src/exceptions.py:573-588`

**When Raised**: Issue with answer key

**Attributes**:
- `reason` (str): Issue description
- `question_id` (str | None): Problematic question

**Common Reasons**:
- Missing answer for question
- Invalid answer format
- Duplicate question IDs

---

#### ScoringError
**Code Reference**: `src/exceptions.py:591-622`

**When Raised**: Score calculation fails

**Attributes**:
- `reason` (str): Scoring issue
- `file_path` (Path | None): OMR file being scored
- `question_id` (str | None): Problematic question

---

### 6. Security Exceptions

#### PathTraversalError
**Code Reference**: `src/exceptions.py:634-655`

**When Raised**: Path traversal attempt detected

**Attributes**:
- `path` (Path): Suspicious path
- `base_path` (Path | None): Protected base path

**Usage**:
```python
if not path.is_relative_to(base_path):
    raise PathTraversalError(path, base_path)
```

**Browser Migration**: Not applicable (no file system access)

---

#### FileSizeLimitError
**Code Reference**: `src/exceptions.py:658-675`

**When Raised**: File exceeds size limits

**Attributes**:
- `path` (Path): Oversized file
- `size` (int): Actual size in bytes
- `limit` (int): Maximum allowed size

**Browser Migration**: Check File.size before processing

---

### 7. Configuration Exceptions

#### ConfigNotFoundError
**Code Reference**: `src/exceptions.py:687-699`

**When Raised**: Required config file not found

---

#### ConfigLoadError
**Code Reference**: `src/exceptions.py:702-717`

**When Raised**: Config file cannot be loaded

---

#### InvalidConfigValueError
**Code Reference**: `src/exceptions.py:720-737`

**When Raised**: Config value is invalid

**Attributes**:
- `key` (str): Configuration key
- `value` (object): Invalid value
- `reason` (str): Why invalid

**Usage**:
```python
if max_workers < 1 or max_workers > 8:
    raise InvalidConfigValueError(
        "max_parallel_workers",
        max_workers,
        "Must be between 1 and 8"
    )
```

---

## Error Handling Patterns

### 1. Try-Except with Specific Exception

**Pattern**: Catch specific exception types

```python
try:
    template = Template(template_path, config, args)
except TemplateValidationError as e:
    logger.error(f"Template validation failed: {e}")
    logger.error(f"Errors: {e.errors}")
    sys.exit(1)
except TemplateNotFoundError as e:
    logger.error(f"Template not found: {e}")
    sys.exit(1)
```

**Code References**:
- `src/entry.py:29-228` (process_directory_wise)
- `main.py:177-215` (main function)

---

### 2. Catch-All with Base Exception

**Pattern**: Catch all OMRChecker errors

```python
try:
    result = process_single_file(file_info)
except OMRCheckerError as e:
    logger.error(f"Processing failed: {e}")
    logger.debug(f"Context: {e.context}")
    result["status"] = "error"
    result["error"] = str(e)
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

**Code References**: `src/entry.py:524-528`

---

### 3. Graceful Degradation

**Pattern**: Log error but continue processing

```python
try:
    aligned_image = align_image(image, template)
except AlignmentError as e:
    logger.warning(f"Alignment failed: {e}")
    logger.warning("Continuing without alignment")
    aligned_image = image  # Fallback to original
```

**Code References**: `src/processors/alignment/processor.py`

---

### 4. Error File Handling

**Pattern**: Move error files to separate directory

```python
if gray_image is None:
    output_file_path = template.get_errors_dir().joinpath(file_name)
    if check_and_move(ERROR_CODES.NO_MARKER_ERR, file_path, output_file_path):
        error_file_line = [file_name, file_path, output_file_path, "NA", ...]
        thread_safe_csv_append(template.get_errors_file(), error_file_line)
    result["status"] = "error"
    result["error"] = "NO_MARKER_ERR"
    return result
```

**Code References**: `src/entry.py:345-368`

---

### 5. Multi-Marked File Handling

**Pattern**: Flag multi-marked files separately

```python
if is_multi_marked:
    logger.info(f"Found multi-marked file: '{file_id}'")
    output_file_path = template.get_multi_marked_dir().joinpath(file_name)
    if check_and_move(ERROR_CODES.MULTI_BUBBLE_WARN, file_path, output_file_path):
        mm_line = [file_name, posix_file_path, output_file_path, "NA", ...]
        thread_safe_csv_append(template.get_multi_marked_file(), mm_line)
```

**Code References**: `src/entry.py:500-522`

---

## Error Codes

**Legacy Error Codes** (not exception-based):

Defined in `src/utils/constants.py`:
```python
ERROR_CODES = DotMap({
    "NO_MARKER_ERR": 1,
    "MULTI_BUBBLE_WARN": 2,
})
```

**Usage**: File classification for errors vs multi-marked

**Browser Migration**: Use enum or constants object

---

## Browser Migration Notes

### JavaScript Equivalents

**Python Exception Hierarchy** → **JavaScript Error Classes**

```javascript
class OMRCheckerError extends Error {
    constructor(message, context = {}) {
        super(message);
        this.name = this.constructor.name;
        this.context = context;
    }

    toString() {
        if (Object.keys(this.context).length > 0) {
            const contextStr = Object.entries(this.context)
                .map(([k, v]) => `${k}=${v}`)
                .join(", ");
            return `${this.message} (${contextStr})`;
        }
        return this.message;
    }
}

class ImageReadError extends OMRCheckerError {
    constructor(path, reason = null) {
        super(
            `Unable to read image: '${path}'${reason ? ` - ${reason}` : ''}`,
            { path, reason }
        );
        this.path = path;
        this.reason = reason;
    }
}

// ... other error classes
```

### Browser-Specific Patterns

**File API Errors**:
```javascript
try {
    const file = await loadFile(filePath);
} catch (e) {
    if (e.name === 'NotFoundError') {
        throw new InputFileNotFoundError(filePath);
    }
    throw new ImageReadError(filePath, e.message);
}
```

**Canvas/Image Loading Errors**:
```javascript
const img = new Image();
img.onerror = () => {
    throw new ImageReadError(filePath, 'Failed to decode image');
};
img.src = URL.createObjectURL(blob);
```

**Validation Errors with Zod**:
```javascript
try {
    const template = templateSchema.parse(templateData);
} catch (e) {
    if (e instanceof z.ZodError) {
        throw new TemplateValidationError(
            templatePath,
            e.errors.map(err => err.message),
            'Schema validation failed'
        );
    }
}
```

---

## Summary

**Exception Types**: 29 (1 base + 28 specific)
**Categories**: 7 (Input/Output, Validation, Processing, Template, Evaluation, Security, Config)
**Context-Rich**: All exceptions carry context dictionary
**Path-Aware**: Most exceptions track file paths
**Hierarchical**: Easy to catch by category or specific type

**Browser Migration**:
- Implement JavaScript Error classes with same hierarchy
- Use try-catch with specific error types
- Display errors in UI (not terminal)
- No file movement (use download/blob instead)
- Validate with Zod/JSON Schema

**Key Takeaway**: OMRChecker's error handling is comprehensive and structured. Browser version should maintain same hierarchy and context-richness for debugging.
