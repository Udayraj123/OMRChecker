# Parsing Utils - Constraints

## Overview

This document covers performance constraints, browser migration considerations, and limitations for the parsing utilities.

**Python Reference**: `src/utils/parsing.py`, `src/utils/json_conversion.py`

---

## Performance Constraints

### 1. Deep Copy Overhead

**Issue**: Deep copying large config dictionaries is expensive.

```python
# Deep copy happens on every config load
defaults_dict = CONFIG_DEFAULTS.to_dict()
user_tuning_config = OVERRIDE_MERGER.merge(
    deepcopy(defaults_dict),  # ← Expensive operation
    user_tuning_config
)
```

**Constraints**:
- **Time Complexity**: O(n) where n = total number of nested keys/values
- **Memory**: Creates complete copy of defaults dictionary
- **Impact**: Noticeable delay for large configs (100+ nested keys)

**Mitigation**:
```python
# Cache deep-copied defaults (singleton pattern)
_cached_config_defaults = None

def get_config_defaults() -> dict:
    global _cached_config_defaults
    if _cached_config_defaults is None:
        _cached_config_defaults = deepcopy(CONFIG_DEFAULTS.to_dict())
    return _cached_config_defaults

# Use cached version
user_tuning_config = OVERRIDE_MERGER.merge(
    get_config_defaults(),  # ← Only deep copied once
    user_tuning_config
)
```

**Browser Note**: JavaScript has no built-in deep copy. Use `structuredClone()` (modern browsers) or lodash.

```javascript
// Modern browsers (Chrome 98+, Firefox 94+, Safari 15.4+)
const defaultsCopy = structuredClone(CONFIG_DEFAULTS);

// Alternative: lodash
import cloneDeep from 'lodash/cloneDeep';
const defaultsCopy = cloneDeep(CONFIG_DEFAULTS);
```

---

### 2. Regex Performance

**Issue**: Field string parsing uses regex on every field.

```python
FIELD_STRING_REGEX_GROUPS = r"([^\.\d]+)(\d+)\.{2,3}(\d+)"

def parse_field_string(field_string) -> list[str]:
    if "." in field_string:
        # Regex called for every range field
        field_prefix, start, end = re.findall(FIELD_STRING_REGEX_GROUPS, field_string)[0]
        # ...
```

**Constraints**:
- **Regex Compilation**: Pattern is compiled on import (fast)
- **Matching**: O(m) where m = field string length
- **Typical Load**: 10-100 field strings per template

**Optimization**: Pre-compile regex patterns.

```python
import re

# Pre-compile at module level
_FIELD_RANGE_PATTERN = re.compile(r"([^\.\d]+)(\d+)\.{2,3}(\d+)")

def parse_field_string(field_string) -> list[str]:
    if "." in field_string:
        match = _FIELD_RANGE_PATTERN.match(field_string)
        if match:
            field_prefix, start, end = match.groups()
            # ...
```

**Browser Migration**:
```javascript
// JavaScript regex is similar, auto-compiled
const FIELD_RANGE_PATTERN = /([^\.\d]+)(\d+)\.{2,3}(\d+)/;

function parseFieldString(fieldString) {
    if (fieldString.includes('.')) {
        const match = fieldString.match(FIELD_RANGE_PATTERN);
        if (match) {
            const [, prefix, start, end] = match;
            // ...
        }
    }
    return [fieldString];
}
```

---

### 3. Recursive Dict Conversion

**Issue**: Case conversion traverses entire dict tree.

```python
def convert_dict_keys_to_snake(data: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in data.items():
        snake_key = camel_to_snake(key)  # ← Called for every key

        if isinstance(value, dict):
            result[snake_key] = convert_dict_keys_to_snake(value)  # ← Recursion
        elif isinstance(value, list):
            result[snake_key] = [
                convert_dict_keys_to_snake(item) if isinstance(item, dict) else item
                for item in value
            ]
        # ...
```

**Constraints**:
- **Time Complexity**: O(k) where k = total keys in nested structure
- **Space Complexity**: O(k) for new dict creation
- **Call Stack**: O(d) where d = max nesting depth
- **Typical Scale**: 50-200 keys for template, 20-50 for config

**Risk**: Stack overflow for deeply nested structures (>1000 levels).

**Browser Migration**:
```javascript
function convertDictKeysToSnake(data) {
    if (typeof data !== 'object' || data === null) return data;

    const result = {};
    for (const [key, value] of Object.entries(data)) {
        const snakeKey = camelToSnake(key);

        if (Array.isArray(value)) {
            result[snakeKey] = value.map(item =>
                typeof item === 'object' ? convertDictKeysToSnake(item) : item
            );
        } else if (typeof value === 'object' && value !== null) {
            result[snakeKey] = convertDictKeysToSnake(value);
        } else {
            result[snakeKey] = value;
        }
    }
    return result;
}
```

---

### 4. Validation Overhead

**Issue**: Validation runs on every config load.

```python
# Validate BEFORE merging
validate_config_json(user_tuning_config, config_path)
validate_no_key_clash(user_tuning_config)
```

**Constraints**:
- **Schema Validation**: O(n) where n = number of schema rules
- **Key Clash Check**: O(k) where k = number of keys
- **Total Time**: ~10-50ms for typical configs

**Trade-off**: Validation is expensive but critical for user experience.

**Optimization**: Skip validation in production if configs are pre-validated.

```python
def open_config_with_defaults(
    config_path: Path,
    args: dict,
    skip_validation: bool = False  # ← Production flag
) -> Config:
    user_tuning_config = load_json(config_path)

    if not skip_validation:
        validate_config_json(user_tuning_config, config_path)
        validate_no_key_clash(user_tuning_config)

    # ... rest of loading
```

**Browser Note**: JSON Schema validation in browser is slower. Use lazy validation.

```javascript
// Validate only when user edits, not on every load
let validatedConfig = null;

function loadConfig(configData, { forceValidate = false } = {}) {
    if (!validatedConfig || forceValidate) {
        validateConfigJSON(configData);  // ← Only when needed
        validatedConfig = configData;
    }
    return mergeWithDefaults(configData);
}
```

---

### 5. Merge Strategy Performance

**Issue**: Deep merge traverses nested dicts multiple times.

```python
# Multiple merge operations
config = OVERRIDE_MERGER.merge(defaults_from_args, user_tuning_config)
config = OVERRIDE_MERGER.merge(deepcopy(CONFIG_DEFAULTS.to_dict()), config)
```

**Constraints**:
- **Per Merge**: O(n) where n = total keys in both dicts
- **Nested Merge**: Recursively merges all nested levels
- **Double Merge**: Config loading does 2-3 merges per load

**Optimization**: Single-pass merge with priority levels.

```python
def merge_multi_level(defaults, args, user):
    """Merge three levels in single pass."""
    # Custom merger that handles priority in one go
    result = deepcopy(defaults)

    # Apply args layer
    for key, value in args.items():
        if isinstance(value, dict) and key in result:
            result[key].update(value)
        else:
            result[key] = value

    # Apply user layer
    for key, value in user.items():
        if isinstance(value, dict) and key in result:
            result[key].update(value)
        else:
            result[key] = value

    return result
```

---

## Browser Migration Constraints

### 1. JSON Parsing API Differences

#### Python

```python
import json

# Standard library
with open(path) as f:
    data = json.load(f)

# Custom decoder
data = json.load(f, object_hook=custom_decoder)
```

#### Browser

```javascript
// Native JSON API (synchronous, in-memory only)
const data = JSON.parse(jsonString);

// File API (async)
async function loadJSON(file) {
    const text = await file.text();
    return JSON.parse(text);
}

// Fetch API (async)
async function loadJSON(url) {
    const response = await fetch(url);
    return await response.json();
}
```

**Constraints**:
- **No File System**: Browser cannot directly access `config.json` on disk
- **Must Be Async**: All file loading is async in browser
- **CORS Restrictions**: Cross-origin JSON files may be blocked
- **Memory Limits**: Large JSON files (>50MB) may cause memory issues

**Migration Pattern**:

```javascript
// Browser: Use File API for user uploads
async function openConfigWithDefaults(file, args) {
    const text = await file.text();
    let userConfig = JSON.parse(text);

    // Validate
    validateConfigJSON(userConfig);
    validateNoKeyClash(userConfig);

    // Merge (synchronous)
    const defaultsFromArgs = { /* ... */ };
    userConfig = merge(defaultsFromArgs, userConfig);
    userConfig = merge(structuredClone(CONFIG_DEFAULTS), userConfig);

    return Config.fromDict(userConfig);
}

// Usage
const fileInput = document.querySelector('input[type="file"]');
fileInput.addEventListener('change', async (e) => {
    const config = await openConfigWithDefaults(e.target.files[0], args);
});
```

---

### 2. Deep Merge Library

#### Python

```python
from deepmerge import Merger

OVERRIDE_MERGER = Merger(
    [(dict, ["merge"])],
    ["override"],
    ["override"],
)
```

#### Browser Options

**Option 1: deepmerge-ts** (TypeScript)

```typescript
import { deepmerge } from 'deepmerge-ts';

const merged = deepmerge(base, user);
```

**Option 2: lodash.merge**

```javascript
import merge from 'lodash/merge';

const merged = merge({}, base, user);  // Note: mutates first arg
```

**Option 3: Custom Implementation**

```javascript
function overrideMerge(base, user) {
    if (typeof user !== 'object' || user === null) return user;
    if (typeof base !== 'object' || base === null) return user;
    if (Array.isArray(user)) return user;  // Arrays override

    const result = { ...base };
    for (const [key, value] of Object.entries(user)) {
        if (typeof value === 'object' && !Array.isArray(value) && value !== null) {
            result[key] = overrideMerge(result[key] || {}, value);
        } else {
            result[key] = value;
        }
    }
    return result;
}
```

**Constraints**:
- **No Standard Library**: Must use external library or custom implementation
- **Bundle Size**: deepmerge-ts (~2KB), lodash.merge (~10KB)
- **Behavior Differences**: Ensure merge behavior matches Python's `deepmerge`

**Recommendation**: Use custom implementation (smallest bundle, exact control).

---

### 3. Fraction Parsing

#### Python

```python
from fractions import Fraction

def parse_float_or_fraction(result: str | float) -> float:
    if isinstance(result, str) and "/" in result:
        result = float(Fraction(result))  # ← Built-in library
    else:
        result = float(result)
    return result
```

#### Browser

**No built-in Fraction library**. Options:

**Option 1: fraction.js** library

```javascript
import Fraction from 'fraction.js';

function parseFloatOrFraction(result) {
    if (typeof result === 'string' && result.includes('/')) {
        return new Fraction(result).valueOf();
    }
    return parseFloat(result);
}
```

**Option 2: Custom parser** (lightweight)

```javascript
function parseFloatOrFraction(result) {
    if (typeof result === 'string' && result.includes('/')) {
        const [num, denom] = result.split('/').map(s => parseFloat(s.trim()));
        if (denom === 0) throw new Error('Division by zero');
        return num / denom;
    }
    return parseFloat(result);
}
```

**Constraints**:
- **Bundle Size**: fraction.js (~8KB gzipped)
- **Precision**: Custom parser may have floating-point errors
- **Edge Cases**: Handle "0/0", "1/0", " 1 / 2 " (spaces)

**Recommendation**: Use custom parser for simple fractions, fraction.js for advanced use cases.

---

### 4. Regular Expressions

**Good News**: JavaScript regex is very similar to Python.

#### Python

```python
import re

FIELD_STRING_REGEX_GROUPS = r"([^\.\d]+)(\d+)\.{2,3}(\d+)"
match = re.findall(FIELD_STRING_REGEX_GROUPS, field_string)
```

#### JavaScript

```javascript
const FIELD_STRING_REGEX_GROUPS = /([^\.\d]+)(\d+)\.{2,3}(\d+)/;
const match = field_string.match(FIELD_STRING_REGEX_GROUPS);
```

**Constraints**:
- **API Differences**: `re.findall()` → `String.match()` or `RegExp.exec()`
- **Global Flag**: Python `re.findall()` finds all matches; JS needs `/g` flag
- **Named Groups**: Syntax differs (`(?P<name>...)` in Python vs `(?<name>...)` in JS)

**Migration Pattern**:

```javascript
// Python: re.findall(pattern, string)
// JavaScript equivalent
function findAll(pattern, string) {
    const matches = [];
    const regex = new RegExp(pattern, 'g');
    let match;
    while ((match = regex.exec(string)) !== null) {
        matches.push(match);
    }
    return matches;
}

// Or use String.matchAll() (modern browsers)
const matches = [...field_string.matchAll(FIELD_STRING_REGEX_GROUPS)];
```

---

### 5. Error Handling

#### Python

```python
# Custom exceptions with context
raise OMRCheckerError(
    "Invalid range",
    context={"field_string": field_string, "start": start, "end": end}
)
```

#### JavaScript

```javascript
// Custom error class
class OMRCheckerError extends Error {
    constructor(message, context = {}) {
        super(message);
        this.name = 'OMRCheckerError';
        this.context = context;
    }
}

// Usage
throw new OMRCheckerError(
    'Invalid range',
    { fieldString, start, end }
);
```

**Constraints**:
- **Stack Traces**: JavaScript stack traces differ from Python
- **Error Display**: Rich console formatting not available in browser console
- **Error Recovery**: Browser needs user-friendly error messages (not stack dumps)

**Browser Pattern**: Show user-friendly errors in UI.

```javascript
try {
    const config = await openConfigWithDefaults(file, args);
} catch (error) {
    if (error instanceof OMRCheckerError) {
        // Show friendly message to user
        showErrorToast(error.message, error.context);
    } else {
        // Unexpected error - log to console
        console.error('Unexpected error:', error);
        showErrorToast('An unexpected error occurred');
    }
}
```

---

## Memory Constraints

### 1. Large Config Files

**Python**: Can handle multi-MB JSON files easily.

**Browser**:
- **Typical Limit**: 50-100MB per file before browser lags
- **Memory Multiplier**: JSON parsing creates 2-3x memory usage (string + parsed object)
- **Mobile Browsers**: More restrictive (10-20MB safe limit)

**Mitigation**: Validate config size before parsing.

```javascript
async function loadJSON(file) {
    // Check size before loading
    const MAX_SIZE = 10 * 1024 * 1024;  // 10MB
    if (file.size > MAX_SIZE) {
        throw new Error(`Config file too large: ${file.size} bytes (max ${MAX_SIZE})`);
    }

    const text = await file.text();
    return JSON.parse(text);
}
```

---

### 2. Template Caching

**Python**: Cache templates in memory for batch processing.

```python
_template_cache = {}

def get_cached_template(path: Path) -> TemplateConfig:
    if path not in _template_cache:
        _template_cache[path] = open_template_with_defaults(path)
    return _template_cache[path]
```

**Browser**: Use IndexedDB for persistent caching across sessions.

```javascript
// IndexedDB wrapper
class TemplateCache {
    async get(key) {
        const db = await this.openDB();
        return await db.get('templates', key);
    }

    async set(key, value) {
        const db = await this.openDB();
        await db.put('templates', value, key);
    }

    // IndexedDB setup code...
}

const cache = new TemplateCache();

async function getCachedTemplate(path) {
    let template = await cache.get(path);
    if (!template) {
        template = await openTemplateWithDefaults(path);
        await cache.set(path, template);
    }
    return template;
}
```

**Constraints**:
- **IndexedDB Limits**: 50MB-unlimited (browser-dependent)
- **Async API**: All operations are async
- **Quota Management**: Browser may clear cache without warning

---

## Validation Constraints

### 1. JSON Schema Validation

**Python**: Uses `jsonschema` library (full JSON Schema Draft 7 support).

**Browser Options**:

**Option 1: Ajv** (most complete)

```javascript
import Ajv from 'ajv';

const ajv = new Ajv();
const validate = ajv.compile(configSchema);

if (!validate(data)) {
    throw new Error(ajv.errorsText(validate.errors));
}
```

**Option 2: Zod** (TypeScript, runtime validation)

```javascript
import { z } from 'zod';

const ConfigSchema = z.object({
    dimensions: z.object({
        processingHeight: z.number().positive()
    }),
    // ...
});

const config = ConfigSchema.parse(data);  // Throws on error
```

**Constraints**:
- **Bundle Size**: Ajv (~40KB), Zod (~12KB)
- **Performance**: Ajv is faster for large schemas
- **TypeScript**: Zod has better TypeScript integration
- **Error Messages**: Zod has more user-friendly errors

**Recommendation**: Use Zod for TypeScript projects, Ajv for JavaScript.

---

### 2. Key Clash Validation

**Python**: Recursive dict traversal.

**JavaScript**: Same approach works.

```javascript
function validateNoKeyClash(data, path = '') {
    if (typeof data !== 'object' || data === null) return;

    const snakeToOriginal = new Map();

    for (const key of Object.keys(data)) {
        const snakeKey = camelToSnake(key);

        if (snakeToOriginal.has(snakeKey)) {
            const originalKey = snakeToOriginal.get(snakeKey);
            if (originalKey !== key) {
                const prefix = path ? `at '${path}': ` : '';
                throw new Error(
                    `${prefix}Key clash: '${originalKey}' and '${key}' ` +
                    `both convert to '${snakeKey}'`
                );
            }
        } else {
            snakeToOriginal.set(snakeKey, key);
        }
    }

    // Recursively validate
    for (const [key, value] of Object.entries(data)) {
        const currentPath = path ? `${path}.${key}` : key;

        if (typeof value === 'object' && value !== null) {
            if (Array.isArray(value)) {
                value.forEach((item, i) => {
                    if (typeof item === 'object') {
                        validateNoKeyClash(item, `${currentPath}[${i}]`);
                    }
                });
            } else {
                validateNoKeyClash(value, currentPath);
            }
        }
    }
}
```

**No constraints** - works identically in browser.

---

## Type System Constraints

### 1. Python Type Hints vs TypeScript

**Python**:

```python
from typing import Any

def load_json(path: str | Path, **rest) -> dict[str, Any]:
    # ...
```

**TypeScript**:

```typescript
type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

function loadJSON(path: string): Promise<Record<string, JsonValue>> {
    // ...
}
```

**Constraints**:
- **Runtime Validation**: Python uses runtime checks; TypeScript is compile-time only
- **Union Types**: `str | Path` in Python → `string | Path` in TypeScript
- **Generic Constraints**: Different syntax for generics

**Migration**: Use Zod for runtime type validation in TypeScript.

```typescript
import { z } from 'zod';

const JsonValueSchema: z.ZodType<JsonValue> = z.lazy(() =>
    z.union([
        z.string(),
        z.number(),
        z.boolean(),
        z.null(),
        z.array(JsonValueSchema),
        z.record(JsonValueSchema),
    ])
);

async function loadJSON(path: string): Promise<Record<string, JsonValue>> {
    const response = await fetch(path);
    const data = await response.json();
    return z.record(JsonValueSchema).parse(data);
}
```

---

## String Processing Constraints

### 1. Case Conversion Performance

**Python**:

```python
def camel_to_snake(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()
```

**JavaScript**:

```javascript
function camelToSnake(name) {
    return name
        .replace(/([A-Z]+)([A-Z][a-z])/g, '$1_$2')
        .replace(/([a-z\d])([A-Z])/g, '$1_$2')
        .toLowerCase();
}
```

**Constraints**:
- **Regex Performance**: Similar in both languages
- **String Immutability**: Both create new strings (no mutation)
- **Edge Cases**: Same edge cases apply (acronyms, numbers)

**No significant constraints** for browser migration.

---

## File System Constraints

### 1. Path Operations

**Python**:

```python
from pathlib import Path

config_path = Path("configs/config.json")
if not config_path.exists():
    raise InputFileNotFoundError(config_path, "JSON")
```

**Browser**: No file system access.

**Migration**: Use File API or fetch API.

```javascript
// Option 1: File upload (user selects file)
async function loadConfigFromFile(file) {
    // File object has name, size, type properties
    const text = await file.text();
    return JSON.parse(text);
}

// Option 2: Fetch from URL (if hosted)
async function loadConfigFromURL(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Config not found: ${url}`);
    }
    return await response.json();
}

// Option 3: IndexedDB (previously saved)
async function loadConfigFromDB(key) {
    const db = await openDB('omr-configs');
    const config = await db.get('configs', key);
    if (!config) {
        throw new Error(`Config not found: ${key}`);
    }
    return config;
}
```

**Constraints**:
- **No Direct File Access**: User must explicitly select files
- **No Automatic Discovery**: Cannot scan directories for config files
- **Async Only**: All file operations are async

---

## Recommended Browser Architecture

```javascript
// config-loader.js - Browser adaptation

class ConfigLoader {
    constructor() {
        this.cache = new Map();
    }

    // Load from File object (user upload)
    async loadFromFile(file) {
        const text = await file.text();
        return this.parseAndValidate(text, file.name);
    }

    // Load from URL (hosted config)
    async loadFromURL(url) {
        const cacheKey = `url:${url}`;
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        const response = await fetch(url);
        const text = await response.text();
        const config = this.parseAndValidate(text, url);

        this.cache.set(cacheKey, config);
        return config;
    }

    // Load from localStorage (quick access)
    loadFromStorage(key) {
        const text = localStorage.getItem(key);
        if (!text) throw new Error(`Config not found: ${key}`);
        return this.parseAndValidate(text, key);
    }

    // Parse and validate JSON
    parseAndValidate(text, source) {
        let data;
        try {
            data = JSON.parse(text);
        } catch (error) {
            throw new ConfigLoadError(source, `Invalid JSON: ${error.message}`);
        }

        // Validate
        validateConfigJSON(data);
        validateNoKeyClash(data);

        return data;
    }

    // Merge with defaults (like Python version)
    mergeWithDefaults(userConfig, args) {
        const defaultsFromArgs = {
            outputs: {
                outputMode: args.outputMode,
                showLogsByType: { debug: args.debug }
            }
        };

        let config = overrideMerge(defaultsFromArgs, userConfig);
        config = overrideMerge(structuredClone(CONFIG_DEFAULTS), config);

        // Inject metadata
        config.loadedAt = new Date().toISOString();

        return config;
    }
}

// Usage
const loader = new ConfigLoader();

// From file upload
const config = await loader.loadFromFile(file);
const merged = loader.mergeWithDefaults(config, args);

// From hosted URL
const config2 = await loader.loadFromURL('/configs/default.json');
```

---

## Summary of Key Constraints

| Constraint | Python | Browser | Migration Strategy |
|------------|--------|---------|-------------------|
| **Deep Copy** | `deepcopy()` | `structuredClone()` or lodash | Use modern `structuredClone()` |
| **JSON Parsing** | Sync, file-based | Async, File API/fetch | Convert to async functions |
| **Deep Merge** | `deepmerge` library | Custom or lodash | Implement custom merger |
| **Regex** | `re` module | Built-in `RegExp` | Direct translation |
| **Fractions** | `fractions.Fraction` | fraction.js or custom | Use custom parser |
| **Validation** | `jsonschema` | Ajv or Zod | Use Zod for TypeScript |
| **Error Handling** | Rich formatting | Plain console | Custom UI error display |
| **File System** | pathlib | File API/IndexedDB | User-selected files only |
| **Caching** | In-memory dict | Map/IndexedDB | Use IndexedDB for persistence |

---

## Performance Recommendations

1. **Cache defaults**: Deep copy CONFIG_DEFAULTS once, reuse
2. **Lazy validation**: Skip validation for trusted configs
3. **Batch processing**: Load multiple configs in parallel
4. **IndexedDB**: Cache templates and configs across sessions
5. **Web Workers**: Offload JSON parsing for large files
6. **Regex**: Use compiled patterns (automatic in JS)
7. **Avoid mutations**: Use `structuredClone()` to prevent bugs

---

## Browser Compatibility

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| `structuredClone()` | 98+ | 94+ | 15.4+ | 98+ |
| `File.text()` | 76+ | 69+ | 14+ | 79+ |
| `matchAll()` | 73+ | 67+ | 13+ | 79+ |
| IndexedDB | All | All | All | All |
| JSON API | All | All | All | All |

**Polyfills**: Use `core-js` for older browsers.

```javascript
import 'core-js/stable/structured-clone';  // Polyfill for structuredClone
```
