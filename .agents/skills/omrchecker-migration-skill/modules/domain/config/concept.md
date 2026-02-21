# Config Entity - Concept

**Module**: Domain / Config
**Python Reference**: `src/schemas/models/config.py`, `src/schemas/defaults/config.py`
**Last Updated**: 2026-02-20

---

## Overview

The Config entity (also called TuningConfig) contains all runtime configuration settings for OMR processing. It controls thresholds, output settings, processing parameters, ML model settings, and visualization options.

**Key Responsibilities**:
1. **Threshold Configuration**: Bubble detection thresholds and scoring
2. **Output Settings**: File organization, CSV format, visualization
3. **Processing Settings**: Image processing parameters, concurrency
4. **ML Configuration**: ML model paths, confidence thresholds
5. **Logging Control**: Log level filtering and debug modes

---

## Config Architecture

### TuningConfig Dataclass

**Code Reference**: `src/schemas/models/config.py`

**Structure**:
```
TuningConfig
├── thresholds (ThresholdConfig)
├── outputs (OutputsConfig)
├── processing (ProcessingConfig)
├── ml (MLConfig)
└── (other settings)
```

**Loading**:
```python
# Load from config.json with defaults
tuning_config = open_config_with_defaults()

# Access settings
threshold_values = tuning_config.thresholds.bubble_values
output_mode = tuning_config.outputs.output_mode
show_image_level = tuning_config.outputs.show_image_level
```

---

## Config Sections

### 1. Thresholds

**Purpose**: Control bubble detection and scoring thresholds

**Fields**:
- `bubble_values` (dict): Bubble darkness thresholds
  - `min_threshold`: Minimum darkness to consider bubble marked
  - `max_threshold`: Maximum darkness for valid bubble
- `marking_threshold_mode` (str): Threshold calculation mode
  - `"GLOBAL"`: Single threshold for all bubbles
  - `"LOCAL"`: Per-field adaptive threshold
  - `"FIELD_LEVEL"`: Per-field custom threshold
- `multi_bubble_threshold` (float): Threshold for multi-mark detection

**Example** (defaults):
```python
thresholds = {
    "bubble_values": {
        "min_threshold": 0.25,
        "max_threshold": 0.95
    },
    "marking_threshold_mode": "GLOBAL",
    "multi_bubble_threshold": 0.15
}
```

---

### 2. Outputs

**Purpose**: Control output generation and visualization

**Fields**:
- `output_mode` (str): Output organization mode
  - `"default"`: Standard output structure
  - `"set_layout"`: Organize by question set
- `colored_outputs_enabled` (bool): Generate colored debug images
- `show_image_level` (int): Debug image verbosity (0-6)
  - `0`: No images
  - `1-2`: Key stages only
  - `3-4`: Detailed stages
  - `5-6`: All intermediate steps
- `show_logs_by_type` (dict): Log level filtering
  - `{"critical": True, "error": True, "warning": True, "info": True, "debug": False}`
- `save_image_level` (int): Save debug images to disk (0-6)
- `show_preprocessors_diff` (dict): Show before/after for preprocessors
  - `{"AutoRotate": True, "CropOnMarkers": False, ...}`

**Example**:
```json
{
  "outputs": {
    "outputMode": "default",
    "coloredOutputsEnabled": true,
    "showImageLevel": 2,
    "saveImageLevel": 0,
    "showLogsByType": {
      "debug": true
    },
    "showPreprocessorsDiff": {
      "AutoRotate": true,
      "CropOnMarkers": true
    }
  }
}
```

---

### 3. Processing

**Purpose**: Control image processing parameters

**Fields**:
- `max_workers` (int): Number of parallel worker threads
- `enable_cropping` (bool): Enable automatic page cropping
- `enable_alignment` (bool): Enable feature-based alignment
- `alignment_mode` (str): Alignment algorithm
  - `"sift"`: SIFT feature matching
  - `"phase_correlation"`: Phase correlation
  - `"template_matching"`: Template matching

**Example**:
```json
{
  "processing": {
    "maxWorkers": 4,
    "enableCropping": true,
    "enableAlignment": true,
    "alignmentMode": "sift"
  }
}
```

---

### 4. ML (Machine Learning)

**Purpose**: Configure ML model settings

**Fields**:
- `bubble_detection` (dict): ML bubble detection settings
  - `enabled`: Enable ML-based bubble detection
  - `confidence_threshold`: Minimum confidence for ML detections
  - `fallback_to_threshold`: Use traditional detection if ML fails
- `field_block_detection` (dict): Field block detection settings
  - `confidence_threshold`: Minimum confidence for field block detection
- `shift_detection` (dict): Shift correction settings
  - `enabled`: Enable ML-based shift detection
  - `model_path`: Path to shift detection model

**Example**:
```json
{
  "ml": {
    "bubbleDetection": {
      "enabled": true,
      "confidenceThreshold": 0.5,
      "fallbackToThreshold": true
    },
    "fieldBlockDetection": {
      "confidenceThreshold": 0.7
    },
    "shiftDetection": {
      "enabled": false
    }
  }
}
```

---

## Config Hierarchy

**Loading Priority** (highest to lowest):
1. **Local config.json**: Directory-specific settings
2. **Global config.json**: Project-wide settings
3. **Default config**: Built-in defaults

**Code Reference**: `src/utils/parsing.py:open_config_with_defaults()`

```python
def open_config_with_defaults():
    # 1. Load default config
    config = DEFAULT_CONFIG.copy()

    # 2. Merge with global config (if exists)
    global_config_path = Path("config.json")
    if global_config_path.exists():
        global_config = json.loads(global_config_path.read_text())
        config = deep_merge(config, global_config)

    # 3. Merge with local config (if exists)
    local_config_path = Path("local_config.json")
    if local_config_path.exists():
        local_config = json.loads(local_config_path.read_text())
        config = deep_merge(config, local_config)

    return TuningConfig.from_dict(config)
```

---

## Default Config

**Code Reference**: `src/schemas/defaults/config.py:21-84`

```python
DEFAULT_CONFIG = {
    "thresholds": {
        "bubble_values": {
            "min_threshold": 0.25,
            "max_threshold": 0.95
        },
        "marking_threshold_mode": "GLOBAL",
        "multi_bubble_threshold": 0.15
    },
    "outputs": {
        "output_mode": "default",
        "colored_outputs_enabled": True,
        "show_image_level": 0,
        "save_image_level": 0,
        "show_logs_by_type": {
            "critical": True,
            "error": True,
            "warning": True,
            "info": True,
            "debug": False
        },
        "show_preprocessors_diff": {}
    },
    "processing": {
        "max_workers": 4,
        "enable_cropping": False,
        "enable_alignment": True
    },
    "ml": {
        "bubble_detection": {
            "enabled": False,
            "confidence_threshold": 0.5,
            "fallback_to_threshold": True
        },
        "field_block_detection": {
            "confidence_threshold": 0.7
        },
        "shift_detection": {
            "enabled": False
        }
    }
}
```

---

## Config Usage Patterns

### Pattern 1: Access Configuration

```python
# Access threshold settings
min_threshold = tuning_config.thresholds.bubble_values["min_threshold"]
threshold_mode = tuning_config.thresholds.marking_threshold_mode

# Access output settings
show_images = tuning_config.outputs.show_image_level >= 3
colored_enabled = tuning_config.outputs.colored_outputs_enabled

# Access ML settings
ml_enabled = tuning_config.ml.bubble_detection["enabled"]
confidence = tuning_config.ml.bubble_detection["confidence_threshold"]
```

---

### Pattern 2: Conditional Processing

```python
# Show debug images based on config
if tuning_config.outputs.show_image_level >= 4:
    InteractionUtils.show("Debug Image", image)

# Save debug images based on config
if tuning_config.outputs.save_image_level >= 2:
    save_image_ops.save_image(image, "debug.png")

# Enable colored outputs
if tuning_config.outputs.colored_outputs_enabled:
    colored_image = process_colored_image(colored_image)
```

---

### Pattern 3: Dynamic Log Levels

```python
# Set log levels from config
logger.set_log_levels(tuning_config.outputs.show_logs_by_type)

# Later: reset to defaults
logger.reset_log_levels()
```

---

## Config File Examples

### Minimal config.json

```json
{
  "outputs": {
    "showImageLevel": 2
  }
}
```

### Development config.json

```json
{
  "outputs": {
    "showImageLevel": 5,
    "saveImageLevel": 3,
    "coloredOutputsEnabled": true,
    "showLogsByType": {
      "debug": true
    },
    "showPreprocessorsDiff": {
      "AutoRotate": true,
      "CropOnMarkers": true,
      "GaussianBlur": true
    }
  },
  "processing": {
    "maxWorkers": 1
  }
}
```

### Production config.json

```json
{
  "thresholds": {
    "bubbleValues": {
      "minThreshold": 0.3,
      "maxThreshold": 0.9
    }
  },
  "outputs": {
    "showImageLevel": 0,
    "saveImageLevel": 0,
    "coloredOutputsEnabled": false,
    "showLogsByType": {
      "info": true,
      "debug": false
    }
  },
  "processing": {
    "maxWorkers": 8,
    "enableAlignment": true
  },
  "ml": {
    "bubbleDetection": {
      "enabled": true,
      "confidenceThreshold": 0.6,
      "fallbackToThreshold": true
    }
  }
}
```

---

## Browser Migration Notes

### TypeScript Config Interface

```typescript
interface TuningConfig {
    thresholds: {
        bubbleValues: {
            minThreshold: number;
            maxThreshold: number;
        };
        markingThresholdMode: 'GLOBAL' | 'LOCAL' | 'FIELD_LEVEL';
        multiBubbleThreshold: number;
    };
    outputs: {
        outputMode: 'default' | 'set_layout';
        coloredOutputsEnabled: boolean;
        showImageLevel: number;
        saveImageLevel: number;
        showLogsByType: Record<string, boolean>;
        showPreprocessorsDiff: Record<string, boolean>;
    };
    processing: {
        maxWorkers: number;
        enableCropping: boolean;
        enableAlignment: boolean;
        alignmentMode?: 'sift' | 'phase_correlation' | 'template_matching';
    };
    ml: {
        bubbleDetection: {
            enabled: boolean;
            confidenceThreshold: number;
            fallbackToThreshold: boolean;
        };
        fieldBlockDetection: {
            confidenceThreshold: number;
        };
        shiftDetection: {
            enabled: boolean;
            modelPath?: string;
        };
    };
}
```

### Config Loading in Browser

```typescript
// Load from JSON file (fetch or File API)
async function loadConfig(configPath?: string): Promise<TuningConfig> {
    // Start with defaults
    let config = DEFAULT_CONFIG;

    // Load global config (if provided)
    if (configPath) {
        const response = await fetch(configPath);
        const userConfig = await response.json();
        config = deepMerge(config, userConfig);
    }

    // Validate with Zod
    return TuningConfigSchema.parse(config);
}

// Zod schema
const TuningConfigSchema = z.object({
    thresholds: z.object({
        bubbleValues: z.object({
            minThreshold: z.number().min(0).max(1),
            maxThreshold: z.number().min(0).max(1)
        }),
        markingThresholdMode: z.enum(['GLOBAL', 'LOCAL', 'FIELD_LEVEL']),
        multiBubbleThreshold: z.number()
    }),
    outputs: z.object({
        outputMode: z.enum(['default', 'set_layout']),
        coloredOutputsEnabled: z.boolean(),
        showImageLevel: z.number().int().min(0).max(6),
        saveImageLevel: z.number().int().min(0).max(6),
        showLogsByType: z.record(z.boolean()),
        showPreprocessorsDiff: z.record(z.boolean())
    }),
    processing: z.object({
        maxWorkers: z.number().int().min(1).max(16),
        enableCropping: z.boolean(),
        enableAlignment: z.boolean(),
        alignmentMode: z.enum(['sift', 'phase_correlation', 'template_matching']).optional()
    }),
    ml: z.object({
        bubbleDetection: z.object({
            enabled: z.boolean(),
            confidenceThreshold: z.number().min(0).max(1),
            fallbackToThreshold: z.boolean()
        }),
        fieldBlockDetection: z.object({
            confidenceThreshold: z.number().min(0).max(1)
        }),
        shiftDetection: z.object({
            enabled: z.boolean(),
            modelPath: z.string().optional()
        })
    })
});
```

### Browser-Specific Adaptations

```typescript
// Browser: No parallel workers (use Web Workers instead)
const processingConfig = {
    maxWorkers: 1, // Main thread only
    enableWebWorkers: true, // Use Web Workers for heavy tasks
    workerPoolSize: 4
};

// Browser: Show images in UI instead of cv2.imshow
function showDebugImage(title: string, mat: cv.Mat): void {
    if (tuningConfig.outputs.showImageLevel >= 3) {
        const canvas = document.createElement('canvas');
        cv.imshow(canvas, mat);
        displayInUI(title, canvas);
    }
}

// Browser: Download instead of save to disk
function saveDebugImage(filename: string, mat: cv.Mat): void {
    if (tuningConfig.outputs.saveImageLevel >= 2) {
        const canvas = document.createElement('canvas');
        cv.imshow(canvas, mat);
        canvas.toBlob(blob => {
            downloadBlob(blob, filename);
        });
    }
}
```

---

## Summary

**Config Entity**: Runtime configuration for OMR processing
**Sections**: Thresholds, Outputs, Processing, ML
**Hierarchy**: Default → Global → Local (merge strategy)
**Usage**: Control thresholds, visualization, logging, ML models
**Validation**: Type-safe dataclass with defaults

**Browser Migration**:
- Use TypeScript interfaces for type safety
- Validate with Zod schemas
- Adapt worker settings for Web Workers
- Replace file I/O with downloads/blobs
- Show images in UI instead of cv2.imshow

**Key Takeaway**: Config is the single source of truth for runtime settings. Browser version should maintain same structure and validation, but adapt I/O operations for browser environment.
