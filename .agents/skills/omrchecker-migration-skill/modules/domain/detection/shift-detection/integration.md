# Shift Detection - Integration

## Overview

This document describes how the Shift Detection system integrates with other components of the OMRChecker pipeline, including data flow, API contracts, and browser migration patterns.

## Pipeline Integration

### Position in Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Processing Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│ 1. Preprocessing (rotate, crop, filters, warp)             │
│    ↓                                                        │
│ 2. Alignment (SIFT/phase correlation)                      │
│    ↓                                                        │
│ 3. ML Field Block Detection ← Populates ml_block_alignments│
│    ↓                                                        │
│ 4. Shift Detection ← Validates and applies shifts          │ ◄── THIS MODULE
│    ↓                                                        │
│ 5. Detection/Interpretation (bubbles, OCR, barcode)        │
│    ↓                                                        │
│ 6. Training Data Collection (optional)                     │
│    ↓                                                        │
│ 7. Output (CSV, JSON, debug images)                        │
└─────────────────────────────────────────────────────────────┘
```

### Integration Points

```python
# main.py or entry.py
def create_processing_pipeline(template, tuning_config):
    """Create processing pipeline with shift detection."""
    processors = []

    # Stage 1: Preprocessing
    processors.append(PreprocessingCoordinator(template, tuning_config))

    # Stage 2: Alignment
    if tuning_config.alignment.method != AlignmentMethod.NONE:
        processors.append(AlignmentProcessor(template, tuning_config.alignment))

    # Stage 3: ML Field Block Detection (optional)
    if tuning_config.ml.field_block_detection.enabled:
        processors.append(MLFieldBlockDetector(
            model_path=tuning_config.ml.field_block_detection.model_path,
            confidence_threshold=tuning_config.ml.field_block_detection.confidence_threshold
        ))

    # Stage 4: Shift Detection (depends on Stage 3)
    if tuning_config.ml.shift_detection.enabled:
        processors.append(ShiftDetectionProcessor(
            template=template,
            shift_config=tuning_config.ml.shift_detection
        ))

    # Stage 5: Traditional Detection
    # (Not added here - handled by TemplateFileRunner internally)

    # Stage 6: Training Data Collection (optional)
    if tuning_config.ml.training.enabled:
        processors.append(TrainingDataCollector(
            output_dir=tuning_config.ml.training.output_dir
        ))

    return ProcessingPipeline(processors)
```

---

## Data Flow Between Components

### Input from MLFieldBlockDetector

**MLFieldBlockDetector Output**:
```python
class MLFieldBlockDetector(Processor):
    def process(self, context: ProcessingContext) -> ProcessingContext:
        # Detect field blocks using YOLO
        detections = self.model.predict(context.gray_image, conf=self.confidence_threshold)

        # Match detections to template blocks
        ml_alignments = {}
        for detection in detections:
            template_block = self._match_to_template(detection)
            shift = self._calculate_shift(detection, template_block)

            ml_alignments[template_block.name] = {
                "shift": [shift_x, shift_y],
                "confidence": detection.confidence,
                "bbox": detection.bbox,
                "expected_bbox": template_block.bounding_box
            }

        # Populate context metadata
        context.metadata["ml_block_alignments"] = ml_alignments

        return context
```

**ShiftDetectionProcessor Input**:
```python
class ShiftDetectionProcessor(Processor):
    def process(self, context: ProcessingContext) -> ProcessingContext:
        # Read ML alignments from context
        ml_alignments = context.metadata.get("ml_block_alignments", {})

        if not ml_alignments or not self.shift_config.enabled:
            return context  # Skip if no alignments or disabled

        # Continue with shift validation and detection...
```

### Output to Detection/Interpretation

**ShiftDetectionProcessor Output**:
```python
class ShiftDetectionProcessor(Processor):
    def process(self, context: ProcessingContext) -> ProcessingContext:
        # ... validation and dual detection ...

        # Update context with final results
        context.field_id_to_interpretation = shifted_results["interpretations"]
        context.omr_response = shifted_results["omr_response"]
        context.is_multi_marked = shifted_results["is_multi_marked"]

        # Store shift detection metadata for debugging/audit
        context.metadata["shift_detection"] = {
            "applied_shifts": validated_shifts,
            "comparison": comparison_meta,
            "confidence_adjustments": comparison_meta["confidence_reductions"]
        }

        return context
```

**Downstream Consumers**:
```python
# CSV Output
def write_results_to_csv(context: ProcessingContext):
    # Use context.omr_response (already includes shifted detections)
    writer.writerow(context.omr_response)

    # Optionally log confidence adjustments
    if "shift_detection" in context.metadata:
        adjustments = context.metadata["shift_detection"]["confidence_adjustments"]
        for field_id, adj in adjustments.items():
            logger.debug(f"{field_id}: confidence {adj['original']:.2f} → {adj['final']:.2f}")

# Visualization
def generate_debug_image(context: ProcessingContext):
    # Use context.field_id_to_interpretation for annotations
    for field_id, interp in context.field_id_to_interpretation.items():
        confidence = interp["confidence"]
        response = interp["response"]

        # Highlight low-confidence fields (likely due to shift mismatch)
        color = RED if confidence < 0.6 else GREEN
        draw_field_response(image, field_id, response, color)
```

---

## API Contracts

### Processor Interface

```python
class Processor(ABC):
    """Base processor interface."""

    @abstractmethod
    def get_name(self) -> str:
        """Get processor name for logging."""
        pass

    @abstractmethod
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Process the context and return updated context."""
        pass
```

**ShiftDetectionProcessor Implementation**:
```python
class ShiftDetectionProcessor(Processor):
    def get_name(self) -> str:
        return "ShiftDetection"

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Apply shifts and validate detection results.

        Args:
            context: Processing context with:
                - gray_image: Grayscale image for detection
                - colored_image: Color image for visualization
                - metadata.ml_block_alignments: ML-detected shifts (optional)
                - template: Template with field blocks

        Returns:
            Updated context with:
                - field_id_to_interpretation: Final interpretations
                - omr_response: Final OMR responses
                - is_multi_marked: Multi-mark flags
                - metadata.shift_detection: Validation metadata
        """
        # Implementation...
```

### Context Contract

**Required Context Fields**:
```python
@dataclass
class ProcessingContext:
    # Input (REQUIRED)
    file_path: str
    gray_image: np.ndarray  # REQUIRED for detection
    colored_image: np.ndarray  # REQUIRED for visualization
    template: Template  # REQUIRED for field blocks

    # Input from MLFieldBlockDetector (OPTIONAL)
    metadata: dict  # Expected: metadata["ml_block_alignments"]

    # Output (POPULATED by ShiftDetectionProcessor)
    field_id_to_interpretation: dict[str, dict]
    omr_response: dict[str, str]
    is_multi_marked: dict[str, bool]
```

**Metadata Schema**:
```python
# Input metadata (from MLFieldBlockDetector)
context.metadata["ml_block_alignments"] = {
    "MCQBlock1a1": {
        "shift": [float, float],      # REQUIRED: [dx, dy] in pixels
        "confidence": float,           # OPTIONAL: ML confidence score
        "bbox": [int, int, int, int],  # OPTIONAL: Detected [x, y, w, h]
        "expected_bbox": [int, int, int, int]  # OPTIONAL: Template [x, y, w, h]
    }
}

# Output metadata (from ShiftDetectionProcessor)
context.metadata["shift_detection"] = {
    "applied_shifts": {
        "MCQBlock1a1": {"dx": float, "dy": float}
    },
    "comparison": {
        "bubble_mismatches": [
            {
                "field_id": str,
                "count": int,
                "bubbles": [
                    {"index": int, "shifted": int, "baseline": int}
                ]
            }
        ],
        "field_mismatches": [
            {
                "field_id": str,
                "shifted_response": str,
                "baseline_response": str
            }
        ],
        "confidence_reductions": {
            "field_id": {
                "original": float,
                "reduction": float,
                "final": float,
                "reason": str
            }
        }
    },
    "confidence_adjustments": dict  # Alias to comparison.confidence_reductions
}
```

### Template Contract

**Required Template Methods**:
```python
class Template:
    field_blocks: list[FieldBlock]  # REQUIRED

class FieldBlock:
    name: str  # REQUIRED: Unique block name
    shifts: list[float, float]  # REQUIRED: [dx, dy] shift state
    origin: list[int, int]  # REQUIRED: Block origin position

    def reset_all_shifts(self) -> None:
        """Reset shifts to [0, 0]. REQUIRED method."""
        pass

    def get_shifted_origin(self) -> list[int, int]:
        """Get origin + shifts. REQUIRED method."""
        pass
```

---

## Configuration Integration

### Configuration Schema

```python
# src/schemas/models/config.py
@dataclass
class TuningConfig:
    thresholding: ThresholdingConfig
    outputs: OutputsConfig
    processing: ProcessingConfig
    ml: MLConfig  # Contains shift_detection

@dataclass
class MLConfig:
    field_block_detection: FieldBlockDetectionConfig
    shift_detection: ShiftDetectionConfig  # ◄── Shift detection config
    training: TrainingConfig

@dataclass
class ShiftDetectionConfig:
    enabled: bool = False
    global_max_shift_pixels: int = 50
    per_block_max_shift_pixels: dict[str, int] = field(default_factory=dict)
    confidence_reduction_min: float = 0.1
    confidence_reduction_max: float = 0.5
    bubble_mismatch_threshold: int = 3
    field_mismatch_threshold: int = 1
```

### JSON Configuration

**config.json**:
```json
{
  "ml": {
    "fieldBlockDetection": {
      "enabled": true,
      "modelPath": "models/field_block_detector.pt",
      "confidenceThreshold": 0.7
    },
    "shiftDetection": {
      "enabled": true,
      "globalMaxShiftPixels": 50,
      "perBlockMaxShiftPixels": {
        "StudentID": 20,
        "MCQBlock1a1": 50,
        "Handwritten": 80
      },
      "confidenceReductionMin": 0.1,
      "confidenceReductionMax": 0.5,
      "bubbleMismatchThreshold": 3,
      "fieldMismatchThreshold": 1
    }
  }
}
```

**Loading Configuration**:
```python
from src.schemas.models.config import TuningConfig

# Load from JSON
with open("config.json") as f:
    config_data = json.load(f)

tuning_config = TuningConfig.from_dict(config_data)

# Access shift detection config
shift_config = tuning_config.ml.shift_detection
if shift_config.enabled:
    processor = ShiftDetectionProcessor(template, shift_config)
```

---

## Integration with TemplateFileRunner

### Detection Execution

**TemplateFileRunner Role**:
```python
class TemplateFileRunner(FileLevelRunner):
    """Executes traditional OMR detection for a template."""

    def read_omr_and_update_metrics(
        self,
        file_path: str,
        gray_image: np.ndarray,
        colored_image: np.ndarray
    ) -> tuple[dict, dict, dict]:
        """Run detection and interpretation.

        Returns:
            (omr_response, is_multi_marked, interpretations)
        """
        # Detection pass: Extract bubbles from image
        detection_results = self.detection_pass.detect(
            gray_image, colored_image, self.all_fields
        )

        # Interpretation pass: Threshold and interpret bubbles
        interpretation_results = self.interpretation_pass.interpret(
            detection_results, self.all_fields
        )

        return (omr_response, is_multi_marked, interpretations)
```

**ShiftDetectionProcessor Usage**:
```python
class ShiftDetectionProcessor(Processor):
    def __init__(self, template, shift_config):
        self.template = template
        self.shift_config = shift_config
        self.template_file_runner = TemplateFileRunner(template)  # ◄── Reuse runner

    def _run_traditional_detection(self, context: ProcessingContext) -> dict:
        """Run traditional detection using TemplateFileRunner."""
        omr_response, is_multi_marked, interpretations = (
            self.template_file_runner.read_omr_and_update_metrics(
                context.file_path,
                context.gray_image,
                context.colored_image
            )
        )

        return {
            "interpretations": interpretations,
            "omr_response": omr_response,
            "is_multi_marked": is_multi_marked
        }
```

---

## Browser Migration Integration

### TypeScript Type Definitions

```typescript
// types/core.ts
export interface ProcessingContext {
  filePath: string;
  grayImage: ImageData;
  coloredImage: ImageData;
  template: Template;
  metadata: {
    mlBlockAlignments?: MLBlockAlignments;
    shiftDetection?: ShiftDetectionMetadata;
  };
  fieldIdToInterpretation: Record<string, FieldInterpretation>;
  omrResponse: Record<string, string>;
  isMultiMarked: Record<string, boolean>;
}

export interface MLBlockAlignments {
  [blockName: string]: {
    shift: [number, number];
    confidence?: number;
    bbox?: [number, number, number, number];
    expectedBbox?: [number, number, number, number];
  };
}

export interface ShiftDetectionMetadata {
  appliedShifts: Record<string, { dx: number; dy: number }>;
  comparison: ComparisonMetadata;
  confidenceAdjustments: Record<string, ConfidenceAdjustment>;
}

export interface ShiftDetectionConfig {
  enabled: boolean;
  globalMaxShiftPixels: number;
  perBlockMaxShiftPixels: Record<string, number>;
  confidenceReductionMin: number;
  confidenceReductionMax: number;
  bubbleMismatchThreshold: number;
  fieldMismatchThreshold: number;
}
```

### Processor Implementation

```typescript
// processors/shift-detection.ts
import { Processor, ProcessingContext } from '../types/core';
import { ShiftDetectionConfig } from '../types/config';
import { TemplateFileRunner } from './detection/template-file-runner';

export class ShiftDetectionProcessor implements Processor {
  private config: ShiftDetectionConfig;
  private templateFileRunner: TemplateFileRunner;

  constructor(template: Template, config: ShiftDetectionConfig) {
    this.config = config;
    this.templateFileRunner = new TemplateFileRunner(template);
  }

  getName(): string {
    return 'ShiftDetection';
  }

  async process(context: ProcessingContext): Promise<ProcessingContext> {
    const mlAlignments = context.metadata.mlBlockAlignments;

    if (!mlAlignments || !this.config.enabled) {
      return context;
    }

    const validatedShifts = this.validateShifts(mlAlignments);

    if (Object.keys(validatedShifts).length === 0) {
      return context;
    }

    // Run dual detection (can be parallelized with Web Workers)
    const [shiftedResults, baselineResults] = await Promise.all([
      this.runDetectionWithShifts(context, validatedShifts),
      this.runDetectionWithoutShifts(context)
    ]);

    const [finalResults, comparison] = this.compareAndAdjustConfidence(
      shiftedResults,
      baselineResults,
      validatedShifts
    );

    // Update context
    context.fieldIdToInterpretation = finalResults.interpretations;
    context.omrResponse = finalResults.omrResponse;
    context.isMultiMarked = finalResults.isMultiMarked;
    context.metadata.shiftDetection = {
      appliedShifts: validatedShifts,
      comparison,
      confidenceAdjustments: comparison.confidenceReductions
    };

    return context;
  }

  private validateShifts(
    mlAlignments: MLBlockAlignments
  ): Record<string, { dx: number; dy: number }> {
    const validated: Record<string, { dx: number; dy: number }> = {};

    for (const [blockName, alignmentData] of Object.entries(mlAlignments)) {
      const [dx, dy] = alignmentData.shift;

      const maxShift =
        this.config.perBlockMaxShiftPixels[blockName] ??
        this.config.globalMaxShiftPixels;

      const magnitude = Math.sqrt(dx * dx + dy * dy);

      if (magnitude <= maxShift) {
        validated[blockName] = { dx, dy };
      }
    }

    return validated;
  }

  // ... other methods ...
}
```

### Web Worker Integration

```typescript
// workers/detection-worker.ts
import { ProcessingContext } from '../types/core';
import { TemplateFileRunner } from '../processors/detection/template-file-runner';

// Worker message handler
self.onmessage = async (e: MessageEvent) => {
  const { context, shifts } = e.data;

  // Apply shifts to template (if provided)
  if (shifts) {
    for (const [blockName, shift] of Object.entries(shifts)) {
      const block = findBlockByName(blockName);
      if (block) {
        block.shifts = [shift.dx, shift.dy];
      }
    }
  }

  // Run detection
  const runner = new TemplateFileRunner(context.template);
  const results = await runner.readOMRAndUpdateMetrics(
    context.filePath,
    context.grayImage,
    context.coloredImage
  );

  // Reset shifts
  if (shifts) {
    for (const block of context.template.fieldBlocks) {
      block.resetAllShifts();
    }
  }

  // Send results back to main thread
  self.postMessage(results);
};
```

**Using Web Workers**:
```typescript
// processors/shift-detection.ts (browser)
private async runParallelDetection(
  context: ProcessingContext,
  shifts: ValidatedShifts
): Promise<[DetectionResults, DetectionResults]> {
  const worker1 = new Worker('/workers/detection-worker.js');
  const worker2 = new Worker('/workers/detection-worker.js');

  const shiftedPromise = new Promise<DetectionResults>((resolve) => {
    worker1.onmessage = (e) => {
      worker1.terminate();
      resolve(e.data);
    };
    worker1.postMessage({ context, shifts });
  });

  const baselinePromise = new Promise<DetectionResults>((resolve) => {
    worker2.onmessage = (e) => {
      worker2.terminate();
      resolve(e.data);
    };
    worker2.postMessage({ context, shifts: null });
  });

  return Promise.all([shiftedPromise, baselinePromise]);
}
```

---

## Testing Integration

### Unit Testing

```python
# tests/test_shift_detection.py
import pytest
from src.processors.detection.shift_detection_processor import ShiftDetectionProcessor
from src.schemas.models.config import ShiftDetectionConfig
from src.processors.base import ProcessingContext

def test_shift_detection_integration():
    """Test integration with pipeline."""
    # Create processor
    config = ShiftDetectionConfig(enabled=True, global_max_shift_pixels=50)
    processor = ShiftDetectionProcessor(template, config)

    # Create context with ML alignments
    context = ProcessingContext(
        file_path="test.jpg",
        gray_image=gray_img,
        colored_image=color_img,
        template=template,
        metadata={
            "ml_block_alignments": {
                "MCQBlock1a1": {"shift": [20, 30]}
            }
        }
    )

    # Process
    result = processor.process(context)

    # Verify output
    assert "shift_detection" in result.metadata
    assert "MCQBlock1a1" in result.metadata["shift_detection"]["applied_shifts"]
    assert result.field_id_to_interpretation is not None
```

### Integration Testing

```python
# tests/test_pipeline_integration.py
def test_full_pipeline_with_shift_detection():
    """Test full pipeline with ML field block detection and shift detection."""
    # Setup
    pipeline = create_processing_pipeline(template, tuning_config)

    # Process image
    context = pipeline.process_image("test.jpg")

    # Verify shift detection ran
    assert "shift_detection" in context.metadata

    # Verify final results
    assert context.omr_response is not None
    assert all(isinstance(conf, float) for conf in
               [interp["confidence"] for interp in context.field_id_to_interpretation.values()])
```

---

## Debugging and Monitoring

### Logging Integration

```python
from src.utils.logger import logger

class ShiftDetectionProcessor(Processor):
    def process(self, context: ProcessingContext) -> ProcessingContext:
        logger.info(f"Shift detection enabled with {len(ml_alignments)} potential shifts")

        validated_shifts = self._validate_shifts(ml_alignments)
        logger.info(f"Applying {len(validated_shifts)} validated shifts")

        # Log rejected shifts
        for block_name, shift_data in ml_alignments.items():
            if block_name not in validated_shifts:
                logger.warning(f"Shift for {block_name} rejected")

        # Log mismatches
        if self.stats["mismatches_detected"] > 0:
            logger.info(
                f"Shift detection complete: {self.stats['mismatches_detected']} mismatches detected"
            )
```

### Metrics Collection

```python
class ShiftDetectionProcessor(Processor):
    def __init__(self, template, shift_config):
        # ...
        self.stats = {
            "shifts_applied": 0,
            "shifts_rejected": 0,
            "mismatches_detected": 0,
            "confidence_reductions": []
        }

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            **self.stats,
            "avg_confidence_reduction": (
                sum(self.stats["confidence_reductions"]) /
                len(self.stats["confidence_reductions"])
                if self.stats["confidence_reductions"] else 0.0
            )
        }
```

### Browser Monitoring

```typescript
class ShiftDetectionProcessor implements Processor {
  private stats = {
    shiftsApplied: 0,
    shiftsRejected: 0,
    mismatchesDetected: 0,
    confidenceReductions: [] as number[]
  };

  async process(context: ProcessingContext): Promise<ProcessingContext> {
    const startTime = performance.now();

    // ... processing ...

    const duration = performance.now() - startTime;

    // Emit metrics
    this.emit('metrics', {
      processorName: 'ShiftDetection',
      duration,
      stats: this.stats
    });

    return context;
  }
}
```

---

## Error Handling Integration

### Error Propagation

```python
from src.utils.exceptions import ProcessorException

class ShiftDetectionProcessor(Processor):
    def process(self, context: ProcessingContext) -> ProcessingContext:
        try:
            # ... processing ...
            return context

        except KeyError as e:
            raise ProcessorException(
                f"Missing required data in context: {e}",
                processor_name=self.get_name(),
                context=context
            ) from e

        except Exception as e:
            logger.error(f"Shift detection failed: {e}", exc_info=True)
            # Return context unchanged (graceful degradation)
            return context
```

### Browser Error Handling

```typescript
class ShiftDetectionProcessor implements Processor {
  async process(context: ProcessingContext): Promise<ProcessingContext> {
    try {
      // ... processing ...
      return context;
    } catch (error) {
      console.error('Shift detection failed:', error);

      // Emit error event
      this.emit('error', {
        processorName: 'ShiftDetection',
        error,
        context
      });

      // Graceful degradation: return context unchanged
      return context;
    }
  }
}
```

---

## Summary

### Key Integration Points

1. **Pipeline**: Stage 4 after ML field block detection
2. **Input**: `context.metadata.ml_block_alignments` from MLFieldBlockDetector
3. **Output**: `context.field_id_to_interpretation` with adjusted confidence
4. **Template**: Mutates `field_block.shifts` temporarily during detection
5. **Configuration**: `tuning_config.ml.shift_detection` in config.json
6. **Detection**: Reuses `TemplateFileRunner` for traditional OMR detection

### Browser Migration Checklist

- ✓ TypeScript type definitions
- ✓ Processor interface implementation
- ✓ Web Worker support for parallel detection
- ✓ Event-based metrics and error handling
- ✓ Performance monitoring with `performance.now()`
- ✓ Graceful degradation on errors
- ✓ Memory management (null references after comparison)

### Testing Strategy

- Unit tests: Shift validation, comparison, confidence calculation
- Integration tests: Full pipeline with ML detection
- Performance tests: Processing time, memory usage
- Browser tests: Web Worker communication, parallel detection
