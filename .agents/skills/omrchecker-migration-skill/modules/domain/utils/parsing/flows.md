# Parsing Utils - Flows

## Overview

The parsing utilities handle JSON configuration loading, merging with defaults, type conversions, and field string parsing. This module is critical for the configuration system that supports hierarchical defaults and flexible field definitions.

**Python Reference**: `src/utils/parsing.py`, `src/utils/json_conversion.py`, `src/utils/file.py`

---

## Core Flows

### 1. Config Loading Flow

**Function**: `open_config_with_defaults(config_path: Path, args: dict) -> Config`

This is the main entry point for loading and merging configuration files.

#### Flow Steps

```
1. Load JSON from file
   └─> load_json(config_path)

2. Validate user config (BEFORE merging)
   └─> validate_config_json(user_config, path)
   └─> validate_no_key_clash(user_config)

3. Create defaults from CLI args
   └─> Build defaults_from_args dict with outputMode & debug

4. Merge in order: args → user config
   └─> OVERRIDE_MERGER.merge(defaults_from_args, user_tuning_config)

5. Merge with system defaults: CONFIG_DEFAULTS → merged config
   └─> OVERRIDE_MERGER.merge(CONFIG_DEFAULTS.to_dict(), user_tuning_config)

6. Inject config path
   └─> user_tuning_config["path"] = str(config_path)

7. Broadcast preprocessor diff setting
   └─> Convert show_preprocessors_diff: bool → dict[processor_name, bool]

8. Convert to dataclass
   └─> Config.from_dict(user_tuning_config)
```

#### Code Example

```python
def open_config_with_defaults(config_path: Path, args: dict[str, Any]) -> Config:
    output_mode = args["outputMode"]
    debug_mode = args["debug"]
    user_tuning_config = load_json(config_path)

    # Validate BEFORE merging
    validate_config_json(user_tuning_config, config_path)
    validate_no_key_clash(user_tuning_config)

    # Merge: args → user → defaults
    defaults_from_args = {
        "outputs": {
            "output_mode": output_mode,
            "show_logs_by_type": {"debug": debug_mode},
        }
    }
    user_tuning_config = OVERRIDE_MERGER.merge(defaults_from_args, user_tuning_config)
    defaults_dict = CONFIG_DEFAULTS.to_dict()
    user_tuning_config = OVERRIDE_MERGER.merge(deepcopy(defaults_dict), user_tuning_config)

    # Inject path
    user_tuning_config["path"] = str(config_path)

    # Broadcast boolean to processor-wise dict
    show_preprocessors_diff = user_tuning_config["outputs"]["show_preprocessors_diff"]
    if isinstance(show_preprocessors_diff, bool):
        user_tuning_config["outputs"]["show_preprocessors_diff"] = dict.fromkeys(
            SUPPORTED_PROCESSOR_NAMES, show_preprocessors_diff
        )

    return Config.from_dict(user_tuning_config)
```

**Merge Precedence**: `file > args > CONFIG_DEFAULTS`

---

### 2. Template Loading Flow

**Function**: `open_template_with_defaults(template_path: Path) -> TemplateConfig`

#### Flow Steps

```
1. Load JSON from file
   └─> load_json(template_path)

2. Validate user template (BEFORE merging)
   └─> validate_template_json(user_template, path)
   └─> validate_no_key_clash(user_template)

3. Convert defaults to camelCase
   └─> TEMPLATE_DEFAULTS.to_dict()
   └─> convert_dict_keys_to_camel(defaults_dict)

4. Merge: defaults → user template
   └─> OVERRIDE_MERGER.merge(defaults_camel, user_template)

5. Convert to dataclass
   └─> TemplateConfig.from_dict(merged_template)
       └─> Handles camelCase → snake_case conversion internally
```

#### Code Example

```python
def open_template_with_defaults(template_path: Path) -> "TemplateConfig":
    user_template = load_json(template_path)

    # Validate BEFORE merging
    validate_template_json(user_template, template_path)
    validate_no_key_clash(user_template)

    # Convert defaults to camelCase for proper merging
    defaults_dict = TEMPLATE_DEFAULTS.to_dict()
    defaults_camel = convert_dict_keys_to_camel(defaults_dict)

    # Merge
    merged_template = OVERRIDE_MERGER.merge(deepcopy(defaults_camel), user_template)

    # Convert to dataclass (from_dict handles camelCase → snake_case)
    return TemplateConfig.from_dict(merged_template)
```

**Key Difference**: Template JSON uses camelCase, so defaults are converted to camelCase before merging.

---

### 3. Evaluation Loading Flow

**Function**: `open_evaluation_with_defaults(evaluation_path: Path) -> dict[str, Any]`

#### Flow Steps

```
1. Load JSON from file
   └─> load_json(evaluation_path)

2. Validate user evaluation (BEFORE merging)
   └─> validate_evaluation_json(user_evaluation, path)
   └─> validate_no_key_clash(user_evaluation)

3. Convert to snake_case
   └─> convert_dict_keys_to_snake(user_evaluation)

4. Merge with defaults
   └─> EVALUATION_CONFIG_DEFAULTS.to_dict()
   └─> OVERRIDE_MERGER.merge(defaults_dict, user_evaluation)

5. Return as dict (for backward compatibility)
```

#### Code Example

```python
def open_evaluation_with_defaults(evaluation_path: Path) -> dict[str, Any]:
    user_evaluation_config = load_json(evaluation_path)

    # Validate BEFORE merging
    validate_evaluation_json(user_evaluation_config, evaluation_path)
    validate_no_key_clash(user_evaluation_config)

    # Convert camelCase → snake_case
    user_evaluation_config = convert_dict_keys_to_snake(user_evaluation_config)

    # Merge with defaults
    defaults_dict = EVALUATION_CONFIG_DEFAULTS.to_dict()
    user_evaluation_config = OVERRIDE_MERGER.merge(
        deepcopy(defaults_dict), user_evaluation_config
    )

    return user_evaluation_config
```

**Note**: Returns dict for backward compatibility, not a dataclass.

---

### 4. JSON Loading Flow

**Function**: `load_json(path: str | Path, **rest) -> dict[str, Any]`

Low-level JSON file loader with error handling.

#### Flow Steps

```
1. Check file exists
   └─> If not: raise InputFileNotFoundError

2. Open and parse JSON
   └─> Path.open(path)
   └─> json.load(f, **rest)

3. Handle decode errors
   └─> catch JSONDecodeError
   └─> log critical error
   └─> raise ConfigLoadError
```

#### Code Example

```python
def load_json(path, **rest) -> dict[str, Any]:
    if not Path(path).exists():
        raise InputFileNotFoundError(Path(path), "JSON")
    try:
        with Path.open(path) as f:
            loaded = json.load(f, **rest)
    except json.decoder.JSONDecodeError as error:
        logger.critical(f"Error when loading json file at: '{path}'", error)
        raise ConfigLoadError(Path(path), f"Invalid JSON format: {error}") from None

    return loaded
```

---

### 5. Deep Merge Strategy

**Configuration**: `OVERRIDE_MERGER`

The deep merge strategy determines how nested configurations are combined.

#### Merger Configuration

```python
from deepmerge import Merger

OVERRIDE_MERGER = Merger(
    # Strategies for each type
    [(dict, ["merge"])],      # Dicts: merge nested keys
    # Fallback for other types
    ["override"],             # Lists, strings, numbers: override
    # Type conflict resolution
    ["override"],             # When types conflict: override
)
```

#### Merge Behavior Examples

```python
# Dict merging (nested merge)
base = {"a": {"x": 1, "y": 2}, "b": 3}
user = {"a": {"y": 20, "z": 30}}
result = OVERRIDE_MERGER.merge(base, user)
# → {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}

# List override (not merge)
base = {"items": [1, 2, 3]}
user = {"items": [4, 5]}
result = OVERRIDE_MERGER.merge(base, user)
# → {"items": [4, 5]}  # Lists replace entirely

# Type conflict (override)
base = {"value": [1, 2, 3]}
user = {"value": "string"}
result = OVERRIDE_MERGER.merge(base, user)
# → {"value": "string"}  # Type change: user wins
```

#### Merge Order

**Important**: `OVERRIDE_MERGER.merge(base, user)` means **user overrides base**.

```python
# Config merging order
OVERRIDE_MERGER.merge(CONFIG_DEFAULTS, user_config)
# → user_config values override defaults

# Multi-level merging
OVERRIDE_MERGER.merge(defaults_from_args, user_tuning_config)
# → user file overrides args
OVERRIDE_MERGER.merge(CONFIG_DEFAULTS, merged_config)
# → merged overrides system defaults
```

---

### 6. Field String Parsing Flow

**Function**: `parse_field_string(field_string: str) -> list[str]`

Parses field shorthand notation into individual field names.

#### Syntax

- **Single field**: `"q1"` → `["q1"]`
- **Range (inclusive)**: `"q1..5"` → `["q1", "q2", "q3", "q4", "q5"]`
- **Range (triple dot)**: `"q1...5"` → `["q1", "q2", "q3", "q4", "q5"]` (same as `..`)

#### Flow Steps

```
1. Check for range notation
   └─> if "." in field_string:

2. Extract components using regex
   └─> FIELD_STRING_REGEX_GROUPS: r"([^\.\d]+)(\d+)\.{2,3}(\d+)"
   └─> Captures: (prefix, start, end)
   └─> Example: "q1..5" → ("q", "1", "5")

3. Validate range
   └─> start < end (else raise OMRCheckerError)

4. Generate field list
   └─> [f"{prefix}{num}" for num in range(start, end + 1)]

5. Return single field if no range
   └─> return [field_string]
```

#### Code Example

```python
import re
from src.schemas.constants import FIELD_STRING_REGEX_GROUPS

def parse_field_string(field_string) -> list[str]:
    if "." in field_string:
        # Extract: prefix, start, end
        field_prefix, start, end = re.findall(FIELD_STRING_REGEX_GROUPS, field_string)[0]
        start, end = int(start), int(end)

        # Validate
        if start >= end:
            raise OMRCheckerError(
                f"Invalid range in fields string: '{field_string}', start: {start} is not less than end: {end}",
                context={"field_string": field_string, "start": start, "end": end}
            )

        # Generate range
        return [f"{field_prefix}{field_number}" for field_number in range(start, end + 1)]

    return [field_string]
```

#### Examples

```python
parse_field_string("q1")       # → ["q1"]
parse_field_string("q1..5")    # → ["q1", "q2", "q3", "q4", "q5"]
parse_field_string("ans1..10") # → ["ans1", "ans2", ..., "ans10"]
parse_field_string("q1..1")    # → Error: start >= end
```

**Regex Pattern**: `FIELD_STRING_REGEX_GROUPS = r"([^\.\d]+)(\d+)\.{2,3}(\d+)"`
- `([^\.\d]+)`: Field prefix (non-digit, non-dot characters)
- `(\d+)`: Start number
- `\.{2,3}`: Two or three dots
- `(\d+)`: End number

---

### 7. Field Array Parsing Flow

**Function**: `parse_fields(key: str, fields: list[str]) -> list[str]`

Parses an array of field strings and checks for overlaps.

#### Flow Steps

```
1. Initialize tracking
   └─> parsed_fields = []
   └─> fields_set = set()  # Track seen fields

2. For each field string:
   a. Parse field string
      └─> fields_array = parse_field_string(field_string)

   b. Check for overlaps
      └─> current_set = set(fields_array)
      └─> if overlap: raise OMRCheckerError

   c. Update tracking
      └─> fields_set.update(current_set)
      └─> parsed_fields.extend(fields_array)

3. Return flattened list
```

#### Code Example

```python
def parse_fields(key: str, fields: list[str]) -> list[str]:
    parsed_fields = []
    fields_set = set()

    for field_string in fields:
        fields_array = parse_field_string(field_string)
        current_set = set(fields_array)

        # Check for overlap
        if not fields_set.isdisjoint(current_set):
            raise OMRCheckerError(
                f"Given field string '{field_string}' has overlapping field(s) with other fields in '{key}': {fields}",
                context={
                    "field_string": field_string,
                    "key": key,
                    "overlapping_fields": list(fields_set.intersection(current_set)),
                }
            )

        fields_set.update(current_set)
        parsed_fields.extend(fields_array)

    return parsed_fields
```

#### Examples

```python
# Valid: no overlap
parse_fields("section1", ["q1..5", "q6..10"])
# → ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]

# Error: overlap detected
parse_fields("section1", ["q1..5", "q3..7"])
# → Error: "q3..7" overlaps with ["q3", "q4", "q5"]

# Valid: different prefixes
parse_fields("mixed", ["q1..3", "ans1..3"])
# → ["q1", "q2", "q3", "ans1", "ans2", "ans3"]
```

---

### 8. Case Conversion Flows

#### a. camelCase → snake_case

**Function**: `camel_to_snake(name: str) -> str`

```python
import re

def camel_to_snake(name: str) -> str:
    # Handle acronyms at start: "MLConfig" → "ml_config"
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)

    # Insert underscore before uppercase: "camelCase" → "camel_Case"
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)

    return name.lower()
```

**Examples**:
```python
camel_to_snake("showImageLevel")      # → "show_image_level"
camel_to_snake("MLConfig")            # → "ml_config"
camel_to_snake("globalPageThreshold") # → "global_page_threshold"
```

#### b. snake_case → camelCase

**Function**: `snake_to_camel(name: str) -> str`

```python
def snake_to_camel(name: str) -> str:
    components = name.split("_")
    # Keep first component lowercase, capitalize rest
    return components[0] + "".join(x.title() for x in components[1:])
```

**Examples**:
```python
snake_to_camel("show_image_level")      # → "showImageLevel"
snake_to_camel("ml_config")             # → "mlConfig"
snake_to_camel("global_page_threshold") # → "globalPageThreshold"
```

#### c. Recursive Dict Conversion

**Functions**:
- `convert_dict_keys_to_snake(data: dict) -> dict`
- `convert_dict_keys_to_camel(data: dict) -> dict`

```python
def convert_dict_keys_to_snake(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        return data

    result = {}
    for key, value in data.items():
        snake_key = camel_to_snake(key)

        # Recursively process nested structures
        if isinstance(value, dict):
            result[snake_key] = convert_dict_keys_to_snake(value)
        elif isinstance(value, list):
            result[snake_key] = [
                convert_dict_keys_to_snake(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[snake_key] = value

    return result
```

**Examples**:
```python
# Nested dict conversion
data = {
    "showImageLevel": 3,
    "debugConfig": {
        "saveImages": True,
        "maxLevel": 5
    }
}

convert_dict_keys_to_snake(data)
# → {
#     "show_image_level": 3,
#     "debug_config": {
#         "save_images": True,
#         "max_level": 5
#     }
# }
```

---

### 9. Key Clash Validation Flow

**Function**: `validate_no_key_clash(data: dict, path: str = "") -> None`

Ensures a JSON doesn't have both camelCase and snake_case versions of the same key.

#### Flow Steps

```
1. Build conversion map
   └─> For each key: map snake_case version to original key

2. Check for clashes
   └─> If snake_case already seen with different original:
       └─> Raise ValueError with context

3. Recursively validate nested structures
   └─> Validate nested dicts
   └─> Validate dicts in lists
```

#### Code Example

```python
def validate_no_key_clash(data: dict[str, Any], path: str = "") -> None:
    if not isinstance(data, dict):
        return

    # Build mapping: snake_case → original_key
    snake_to_original = {}

    for key in data.keys():
        snake_key = camel_to_snake(key)

        if snake_key in snake_to_original:
            original_key = snake_to_original[snake_key]
            if original_key != key:
                prefix = f"at '{path}': " if path else ""
                raise ValueError(
                    f"{prefix}Key clash detected: '{original_key}' and '{key}' "
                    f"both convert to '{snake_key}'. Please use only one naming convention."
                )
        else:
            snake_to_original[snake_key] = key

    # Recursively validate nested structures
    for key, value in data.items():
        current_path = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            validate_no_key_clash(value, current_path)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    validate_no_key_clash(item, f"{current_path}[{i}]")
```

#### Examples

```python
# Valid: no clash
validate_no_key_clash({"userName": "Alice", "email": "test@example.com"})
# → No error

# Error: clash detected
validate_no_key_clash({"userName": "Alice", "user_name": "Bob"})
# → ValueError: Key clash detected: 'userName' and 'user_name' both convert to 'user_name'

# Error: nested clash
validate_no_key_clash({
    "config": {
        "showImageLevel": 3,
        "show_image_level": 5
    }
})
# → ValueError: at 'config': Key clash detected...
```

---

### 10. Type Conversion Flows

#### a. Float or Fraction Parsing

**Function**: `parse_float_or_fraction(result: str | float) -> float`

Handles fractional string notation in evaluation configs.

```python
from fractions import Fraction

def parse_float_or_fraction(result: str | float) -> float:
    if isinstance(result, str) and "/" in result:
        result = float(Fraction(result))
    else:
        result = float(result)
    return result
```

**Examples**:
```python
parse_float_or_fraction("1/2")    # → 0.5
parse_float_or_fraction("3/4")    # → 0.75
parse_float_or_fraction(0.5)      # → 0.5
parse_float_or_fraction("0.5")    # → 0.5
```

#### b. Alphanumerical Sort Key

**Function**: `alphanumerical_sort_key(field_label: str) -> list[str | int]`

Creates sort keys for field labels with numeric suffixes.

```python
import re
from src.utils.constants import FIELD_LABEL_NUMBER_REGEX

def alphanumerical_sort_key(field_label: str) -> list[str | int]:
    # FIELD_LABEL_NUMBER_REGEX = r"([^\d]+)(\d*)"
    label_prefix, label_suffix = re.findall(FIELD_LABEL_NUMBER_REGEX, field_label)[0]
    return [label_prefix, int(label_suffix) if len(label_suffix) > 0 else 0, 0]
```

**Examples**:
```python
fields = ["q10", "q2", "q1", "ans1", "q20"]
sorted(fields, key=alphanumerical_sort_key)
# → ["ans1", "q1", "q2", "q10", "q20"]  # Numerical sort, not lexical

# Without this function:
sorted(fields)
# → ["ans1", "q1", "q10", "q2", "q20"]  # Lexical sort (wrong order)
```

#### c. Default JSON Serialization

**Function**: `default_dump(obj: object) -> bool | dict | str`

Custom JSON encoder for non-serializable objects.

```python
import numpy as np

def default_dump(obj: object) -> bool | dict[str, Any] | str:
    return (
        bool(obj)
        if isinstance(obj, np.bool_)
        else (
            obj.to_json()
            if hasattr(obj, "to_json")
            else obj.__dict__
            if hasattr(obj, "__dict__")
            else obj
        )
    )
```

**Usage**:
```python
import json

data = {
    "value": np.bool_(True),
    "config": Config(...),
    "custom": CustomObject(...)
}

json.dumps(data, default=default_dump)
```

**Handling Order**:
1. `np.bool_` → `bool`
2. Objects with `to_json()` method → call `to_json()`
3. Objects with `__dict__` → serialize `__dict__`
4. Other → return as-is

---

## Key Design Patterns

### 1. Validation Before Merge

```python
# Always validate BEFORE merging with defaults
validate_config_json(user_config, path)
validate_no_key_clash(user_config)

# Then merge
merged = OVERRIDE_MERGER.merge(defaults, user_config)
```

**Reason**: Catch user errors before contaminating with defaults.

### 2. Deep Copy Defaults

```python
# Always deep copy before merging
merged = OVERRIDE_MERGER.merge(deepcopy(defaults_dict), user_config)
```

**Reason**: Prevent mutation of default config objects.

### 3. Precedence Chain

```
System Defaults → CLI Args → User File
   (lowest)                    (highest)
```

```python
# Step 1: Merge args → user
config = OVERRIDE_MERGER.merge(args_defaults, user_config)

# Step 2: Merge defaults → merged
config = OVERRIDE_MERGER.merge(system_defaults, config)
```

### 4. Case Convention Handling

- **Config JSON**: Uses snake_case or camelCase (both accepted, converted internally)
- **Template JSON**: Uses camelCase (external API convention)
- **Evaluation JSON**: Uses camelCase (converted to snake_case internally)
- **Python Code**: Always uses snake_case

```python
# Template: keep camelCase until final conversion
defaults_camel = convert_dict_keys_to_camel(TEMPLATE_DEFAULTS.to_dict())
merged = OVERRIDE_MERGER.merge(defaults_camel, user_template)
# TemplateConfig.from_dict() handles conversion to snake_case

# Evaluation: convert early
user_evaluation = convert_dict_keys_to_snake(user_evaluation)
merged = OVERRIDE_MERGER.merge(defaults, user_evaluation)
```

---

## Error Handling

### 1. File Not Found

```python
if not Path(path).exists():
    raise InputFileNotFoundError(Path(path), "JSON")
```

### 2. Invalid JSON Syntax

```python
try:
    loaded = json.load(f)
except json.decoder.JSONDecodeError as error:
    logger.critical(f"Error when loading json file at: '{path}'", error)
    raise ConfigLoadError(Path(path), f"Invalid JSON format: {error}") from None
```

### 3. Key Clash

```python
validate_no_key_clash(user_config)
# Raises: ValueError with detailed path and conflicting keys
```

### 4. Invalid Field Range

```python
if start >= end:
    raise OMRCheckerError(
        f"Invalid range: '{field_string}', start: {start} >= end: {end}",
        context={"field_string": field_string, "start": start, "end": end}
    )
```

### 5. Overlapping Fields

```python
if not fields_set.isdisjoint(current_set):
    raise OMRCheckerError(
        f"Overlapping fields detected",
        context={
            "field_string": field_string,
            "overlapping_fields": list(fields_set.intersection(current_set))
        }
    )
```

---

## Complete Example: Config Loading

```python
# User's config.json (partial)
{
    "dimensions": {
        "processingHeight": 1200
    },
    "outputs": {
        "showImageLevel": 2
    }
}

# CLI args
args = {
    "outputMode": "default",
    "debug": True
}

# Loading flow
config = open_config_with_defaults(Path("config.json"), args)

# Internal steps:
# 1. Load JSON
# 2. Validate JSON schema
# 3. Validate no key clash
# 4. Create args defaults:
#    {
#      "outputs": {
#        "output_mode": "default",
#        "show_logs_by_type": {"debug": True}
#      }
#    }
# 5. Merge: args → user
#    {
#      "dimensions": {"processingHeight": 1200},
#      "outputs": {
#        "output_mode": "default",
#        "show_logs_by_type": {"debug": True},
#        "showImageLevel": 2
#      }
#    }
# 6. Merge: defaults → merged
#    (All missing keys filled from CONFIG_DEFAULTS)
# 7. Inject path: config["path"] = "config.json"
# 8. Broadcast show_preprocessors_diff
# 9. Convert to Config dataclass

# Result: fully populated Config object with all defaults
```

---

## Related Modules

- **Validation**: `src/utils/validations.py` - JSON schema validation
- **Defaults**: `src/schemas/defaults/` - Default config values
- **Models**: `src/schemas/models/` - Dataclass definitions
- **File Utils**: `src/utils/file.py` - JSON loading
- **Constants**: `src/schemas/constants.py` - Regex patterns, field types

---

## Testing Considerations

### Unit Tests

```python
# Test field string parsing
assert parse_field_string("q1") == ["q1"]
assert parse_field_string("q1..5") == ["q1", "q2", "q3", "q4", "q5"]

# Test case conversion
assert camel_to_snake("showImageLevel") == "show_image_level"
assert snake_to_camel("show_image_level") == "showImageLevel"

# Test key clash validation
with pytest.raises(ValueError):
    validate_no_key_clash({"userName": "A", "user_name": "B"})

# Test merge behavior
result = OVERRIDE_MERGER.merge({"a": {"x": 1}}, {"a": {"y": 2}})
assert result == {"a": {"x": 1, "y": 2}}
```

### Integration Tests

```python
# Test config loading with real files
config = open_config_with_defaults(Path("samples/config.json"), args)
assert isinstance(config, Config)
assert config.path == "samples/config.json"

# Test template loading
template = open_template_with_defaults(Path("samples/template.json"))
assert isinstance(template, TemplateConfig)
assert len(template.field_blocks) > 0
```

---

## Performance Notes

- **Deep Copy**: `deepcopy()` is expensive but necessary to prevent mutation
- **Regex**: Field string parsing uses compiled regex for efficiency
- **Dict Merging**: Recursive merging can be slow for deeply nested configs
- **Validation**: Schema validation runs on every load (consider caching for repeated loads)

**Optimization**: For batch processing, cache loaded configs and reuse:

```python
# Cache templates by path
_template_cache = {}

def get_cached_template(path: Path) -> TemplateConfig:
    if path not in _template_cache:
        _template_cache[path] = open_template_with_defaults(path)
    return _template_cache[path]
```
