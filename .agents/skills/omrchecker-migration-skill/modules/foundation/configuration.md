# Configuration Management

**Module**: Foundation
**Python Reference**: `src/schemas/defaults/config.py`, `src/utils/parsing.py`
**Last Updated**: 2026-02-20

---

## Overview

OMRChecker uses **hierarchical configuration** with deep merging: defaults → global config → local config.

**Config Structure**:
- **Thresholding**: Bubble detection thresholds
- **Outputs**: Visualization, logging, file organization
- **Processing**: Parallel workers
- **ML**: Machine learning model settings

---

## Configuration Hierarchy

### 1. Defaults
**Code**: `src/schemas/defaults/config.py:CONFIG_DEFAULTS`
```python
CONFIG_DEFAULTS = Config(
    thresholding=ThresholdingConfig(...),
    outputs=OutputsConfig(...),
    processing=ProcessingConfig(max_parallel_workers=1),
    ml=MLConfig(enabled=False),
)
```

### 2. Global Config
**Location**: `config.json` in project root
**Scope**: Applies to all directories

### 3. Local Config
**Location**: `config.json` in specific input directory
**Scope**: Overrides global for that directory only

---

## Config Merging

**Code Reference**: `src/utils/parsing.py:open_config_with_defaults`

```python
def open_config_with_defaults(config_path, args):
    # 1. Start with defaults
    config = CONFIG_DEFAULTS

    # 2. Deep merge with loaded config
    config_dict = json.load(config_path)
    merged = deepmerge.merge(defaults_dict, config_dict)

    # 3. Convert back to Pydantic model
    return Config.from_dict(merged)
```

**Deep Merge**: Nested objects merged recursively

---

## Config Sections

### ThresholdingConfig
```python
@dataclass
class ThresholdingConfig:
    gamma_low: float = 0.7
    min_gap_two_bubbles: int = 30
    min_jump: int = 25
    global_page_threshold: int = 200
    # ... 10 parameters
```

### OutputsConfig
```python
@dataclass
class OutputsConfig:
    show_image_level: int = 0  # 0-6
    show_logs_by_type: dict = field(default_factory=lambda: {
        "critical": True, "error": True, "warning": True,
        "info": True, "debug": False
    })
    save_detections: bool = True
    colored_outputs_enabled: bool = False
    filter_out_multimarked_files: bool = False
    file_grouping: FileGroupingConfig = field(default_factory=FileGroupingConfig)
```

### ProcessingConfig
```python
@dataclass
class ProcessingConfig:
    max_parallel_workers: int = 1  # 1-8 recommended
```

### MLConfig
```python
@dataclass
class MLConfig:
    enabled: bool = False
    model_path: str | None = None
    field_block_confidence_threshold: float = 0.5
    bubble_confidence_threshold: float = 0.5
```

---

## Dynamic Configuration

### Runtime Log Level Changes
```python
# Set log levels from config
logger.set_log_levels(tuning_config.outputs.show_logs_by_type)

# Reset after processing
logger.reset_log_levels()
```

**Code Reference**: `src/entry.py:89, 635`

---

## Browser Migration

### Local Storage Config
```javascript
class ConfigManager {
    constructor() {
        this.defaults = CONFIG_DEFAULTS;
        this.loadConfig();
    }

    loadConfig() {
        const stored = localStorage.getItem('omr_config');
        if (stored) {
            this.config = this.merge(this.defaults, JSON.parse(stored));
        } else {
            this.config = { ...this.defaults };
        }
    }

    saveConfig() {
        localStorage.setItem('omr_config', JSON.stringify(this.config));
    }

    merge(defaults, overrides) {
        // Deep merge implementation
        return _.merge({}, defaults, overrides);
    }
}
```

### UI Configuration
```javascript
// Expose config as reactive state (React/Vue)
const [config, setConfig] = useState(CONFIG_DEFAULTS);

function updateThreshold(value) {
    setConfig(prev => ({
        ...prev,
        thresholding: {
            ...prev.thresholding,
            global_page_threshold: value
        }
    }));
}
```

---

## Summary

**Hierarchy**: defaults → global → local (deep merge)
**Sections**: Thresholding, Outputs, Processing, ML
**Dynamic**: Runtime log level control
**Validation**: JSON Schema + Pydantic
**Browser**: localStorage + reactive state management

**Key Pattern**: Sensible defaults with progressive overrides
