# ML Bubble Detector - Integration

**Module**: Domain - Detection - ML Bubble
**Python Reference**: `src/processors/detection/ml_bubble_detector.py`
**Last Updated**: 2026-02-21

---

## Overview

This document describes how the ML Bubble Detector integrates with the broader OMRChecker pipeline, including ReadOMR processor, field block detection, and hybrid fallback strategies.

---

## Integration Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────┐
│ Processing Pipeline                                         │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│ │Preprocessing│─>│ Alignment   │─>│ Field Block Detection│ │
│ └─────────────┘  └─────────────┘  └──────────┬───────────┘ │
│                                               │             │
│                              ┌────────────────┴──────────┐  │
│                              │                           │  │
│                    ┌─────────▼────────┐      ┌──────────▼──────┐
│                    │ ML Field Block   │      │ Shift Detection │
│                    │ Detector (Stage1)│      │ Processor       │
│                    └─────────┬────────┘      └──────────┬──────┘
│                              │                           │      │
│                    ┌─────────▼────────┐                  │      │
│                    │ ML Bubble        │                  │      │
│                    │ Detector (Stage2)│<─────────────────┘      │
│                    └─────────┬────────┘                         │
│                              │                                  │
│                    ┌─────────▼────────┐                         │
│                    │ ReadOMR          │                         │
│                    │ (Detection +     │                         │
│                    │  Interpretation) │                         │
│                    └─────────┬────────┘                         │
│                              │                                  │
│                    ┌─────────▼────────┐                         │
│                    │ Evaluation       │                         │
│                    └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration Point 1: ML Field Block Detector (Stage 1)

### Data Flow

**Stage 1 Output** → **Stage 2 Input**

```python
# Stage 1: ML Field Block Detector
# Sets: context.metadata["ml_detected_blocks"]

context.metadata["ml_detected_blocks"] = [
    {
        "class_id": 0,
        "class_name": "mcq_block_4",
        "confidence": 0.92,
        "bbox_xyxy": [100, 200, 400, 350],
        "bbox_origin": [100, 200],
        "bbox_dimensions": [300, 150]
    },
    # ... more blocks
]

# Stage 2: ML Bubble Detector
# Reads: context.metadata["ml_detected_blocks"]
# Adds: block["ml_bubbles"] and block["ml_bubbles_count"]

ml_blocks = context.metadata.get("ml_detected_blocks", [])
for block in ml_blocks:
    # Detect bubbles within block
    block["ml_bubbles"] = detect_bubbles_in_block(block)
    block["ml_bubbles_count"] = len(block["ml_bubbles"])
```

### Contract

**Required Fields from Stage 1**:
```python
{
    "bbox_xyxy": [x1, y1, x2, y2],  # REQUIRED: Bounding box for cropping
    "class_name": str,               # REQUIRED: For logging
    "confidence": float,             # OPTIONAL: For quality filtering
}
```

**Added Fields by Stage 2**:
```python
{
    # Original Stage 1 fields...
    "ml_bubbles": [                 # NEW: Bubble detections
        {
            "class_id": int,
            "class_name": str,
            "confidence": float,
            "bbox_xyxy": [x1, y1, x2, y2],
            "state": "filled" | "empty"
        }
    ],
    "ml_bubbles_count": int        # NEW: Count for quick access
}
```

### Error Handling

```python
# Stage 2 gracefully handles missing Stage 1 data
ml_blocks = context.metadata.get("ml_detected_blocks", [])

if not ml_blocks:
    logger.debug("No ML-detected blocks, skipping ML bubble detection")
    return context  # Skip Stage 2 if Stage 1 didn't run
```

---

## Integration Point 2: ReadOMR Processor (Hybrid Fallback)

### Hybrid Detection Strategy

ReadOMR coordinates traditional threshold-based detection with ML fallback.

```python
class ReadOMRProcessor(Processor):
    def __init__(self, template, ml_model_path: str | None = None):
        self.template = template
        self.template_file_runner = TemplateFileRunner(template)

        # Optional ML fallback
        self.ml_detector = None
        self.hybrid_strategy = None
        if ml_model_path:
            self._initialize_ml_fallback(ml_model_path)

    def _initialize_ml_fallback(self, ml_model_path: str | Path):
        """Initialize ML fallback detector."""
        from src.processors.detection.ml_detector import (
            HybridDetectionStrategy,
            MLBubbleDetector,
        )

        self.ml_detector = MLBubbleDetector(ml_model_path)

        # Get confidence threshold from config
        confidence_threshold = (
            getattr(self.tuning_config.ml, "confidence_threshold", 0.75)
            if hasattr(self.tuning_config, "ml")
            else 0.75
        )

        self.hybrid_strategy = HybridDetectionStrategy(
            self.ml_detector,
            confidence_threshold=confidence_threshold
        )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Execute OMR detection and interpretation."""

        # 1. Run traditional threshold-based detection
        raw_omr_response = self.template_file_runner.read_omr_and_update_metrics(
            file_path, gray_image, colored_image
        )

        # Extract interpretation metrics
        field_id_to_interpretation = ...

        # Update context
        context.field_id_to_interpretation = field_id_to_interpretation

        # 2. Check for low-confidence fields and use ML fallback if needed
        if self.hybrid_strategy and self.hybrid_strategy.should_use_ml_fallback(context):
            logger.info("Using ML fallback for low-confidence fields")

            # Enable ML detector
            self.ml_detector.enable_for_low_confidence()

            # Run ML detection
            context = self.ml_detector.process(context)

            # Disable ML detector
            self.ml_detector.disable()

            # Update statistics
            self.hybrid_strategy.stats["ml_fallback_used"] += 1

        return context
```

### Confidence Evaluation

```python
class HybridDetectionStrategy:
    def should_use_ml_fallback(self, context: ProcessingContext) -> bool:
        """Determine if ML fallback should be used."""

        if self.ml_detector is None or self.ml_detector.model is None:
            return False

        # Check for low confidence fields
        field_id_to_interpretation = context.field_id_to_interpretation
        low_confidence = self.identify_low_confidence_fields(
            field_id_to_interpretation
        )

        if low_confidence:
            logger.info(
                f"Found {len(low_confidence)} low-confidence fields, "
                f"triggering ML fallback"
            )
            return True

        return False

    def identify_low_confidence_fields(
        self, field_id_to_interpretation: dict
    ) -> list[tuple[str, float]]:
        """Identify fields with low confidence scores."""

        low_confidence_fields = []

        for field_id, interpretation in field_id_to_interpretation.items():
            # Get confidence metrics
            confidence_metrics = getattr(
                interpretation, "field_level_confidence_metrics", {}
            )
            confidence_score = confidence_metrics.get("overall_confidence_score", 1.0)

            # Check threshold (default: 0.75)
            if confidence_score < self.confidence_threshold:
                low_confidence_fields.append((field_id, confidence_score))
                logger.debug(
                    f"Low confidence field: {field_id} "
                    f"(confidence: {confidence_score:.3f})"
                )
            else:
                self.stats["high_confidence_fields"] += 1

        return low_confidence_fields
```

### Statistics Tracking

```python
def finish_processing_directory(self):
    """Finish processing directory and get aggregated results."""
    results = self.template_file_runner.finish_processing_directory()

    # Log ML fallback statistics if enabled
    if self.hybrid_strategy:
        stats = self.hybrid_strategy.get_statistics()
        logger.info("=" * 60)
        logger.info("ML Fallback Statistics")
        logger.info("=" * 60)
        logger.info(f"Total fields processed: {stats['total_fields']}")
        logger.info(f"High confidence fields: {stats['high_confidence_fields']}")
        logger.info(f"Low confidence fields: {stats['low_confidence_fields']}")
        logger.info(f"ML fallback used: {stats['ml_fallback_used']} times")
        logger.info("=" * 60)

    return results
```

---

## Integration Point 3: Shift Detection Processor

### Using ML Bubbles for Shift Calculation

Shift Detection can use ML-detected bubble positions to calculate field shifts.

```python
class ShiftDetectionProcessor(Processor):
    def process(self, context: ProcessingContext) -> ProcessingContext:
        # Get ML-detected blocks
        ml_blocks = context.metadata.get("ml_detected_blocks", [])

        if not ml_blocks:
            logger.debug("No ML blocks, using traditional shift detection")
            return self._traditional_shift_detection(context)

        # Calculate shifts using ML bubble positions
        for block in ml_blocks:
            ml_bubbles = block.get("ml_bubbles", [])

            if not ml_bubbles:
                continue

            # Find corresponding template field
            field = self._find_matching_field(block, context.template)

            if field:
                # Calculate shift by comparing ML positions to template positions
                shift = self._calculate_shift(ml_bubbles, field)
                field.apply_shift(shift)

        return context
```

### Shift Calculation Algorithm

```python
def _calculate_shift(
    self, ml_bubbles: list[dict], field: Field
) -> tuple[int, int]:
    """Calculate field shift from ML bubble positions."""

    # Get expected bubble positions from template
    expected_positions = [
        bubble.position for bubble in field.scan_boxes
    ]

    # Get actual bubble positions from ML
    actual_positions = [
        bubble["bbox_origin"] for bubble in ml_bubbles
    ]

    # Match bubbles (nearest neighbor)
    matches = self._match_bubbles(expected_positions, actual_positions)

    # Calculate median shift (robust to outliers)
    shifts = [
        (actual[0] - expected[0], actual[1] - expected[1])
        for expected, actual in matches
    ]

    shift_x = int(np.median([s[0] for s in shifts]))
    shift_y = int(np.median([s[1] for s in shifts]))

    return (shift_x, shift_y)
```

---

## Integration Point 4: Pipeline Registration

### Pipeline Configuration

```python
# In src/processors/pipeline.py

from src.processors.detection.ml_bubble_detector import MLBubbleDetector
from src.processors.detection.ml_field_block_detector import MLFieldBlockDetector

class ProcessingPipeline:
    def __init__(self, template, tuning_config):
        self.stages = []

        # ... preprocessing, alignment stages ...

        # Optional: ML Field Block Detection (Stage 1)
        if tuning_config.ml.enable_field_block_detection:
            ml_field_block_detector = MLFieldBlockDetector(
                model_path=tuning_config.ml.field_block_model_path,
                confidence_threshold=tuning_config.ml.field_block_confidence
            )
            self.stages.append(ml_field_block_detector)

        # Optional: ML Bubble Detection (Stage 2)
        if tuning_config.ml.enable_bubble_detection:
            ml_bubble_detector = MLBubbleDetector(
                model_path=tuning_config.ml.bubble_model_path,
                confidence_threshold=tuning_config.ml.bubble_confidence
            )
            self.stages.append(ml_bubble_detector)

        # Optional: Shift Detection (uses ML bubble positions if available)
        if tuning_config.enable_shift_detection:
            shift_detector = ShiftDetectionProcessor()
            self.stages.append(shift_detector)

        # Main detection (with optional ML fallback)
        read_omr = ReadOMRProcessor(
            template,
            ml_model_path=tuning_config.ml.bubble_model_path if tuning_config.ml.enable_fallback else None
        )
        self.stages.append(read_omr)
```

### Configuration Schema

```python
# In src/schemas/models/config.py

@dataclass
class MLConfig:
    """ML detection configuration."""

    # Field Block Detection (Stage 1)
    enable_field_block_detection: bool = False
    field_block_model_path: str = "models/field_block_detector.pt"
    field_block_confidence: float = 0.7

    # Bubble Detection (Stage 2)
    enable_bubble_detection: bool = False
    bubble_model_path: str = "models/bubble_detector.pt"
    bubble_confidence: float = 0.7

    # Hybrid Fallback
    enable_fallback: bool = False
    fallback_confidence_threshold: float = 0.75

@dataclass
class TuningConfig:
    """Main tuning configuration."""

    thresholds: ThresholdConfig
    outputs: OutputConfig
    processing: ProcessingConfig
    ml: MLConfig  # NEW: ML configuration section
```

### Example Configuration

```json
{
  "ml": {
    "enable_field_block_detection": true,
    "field_block_model_path": "models/field_block_detector_v8n.pt",
    "field_block_confidence": 0.7,

    "enable_bubble_detection": true,
    "bubble_model_path": "models/bubble_detector_v8n.pt",
    "bubble_confidence": 0.7,

    "enable_fallback": true,
    "fallback_confidence_threshold": 0.75
  }
}
```

---

## Integration Point 5: Detection Fusion

### Merging ML and Traditional Results

Detection Fusion combines ML and traditional detection results for final answer.

```python
class DetectionFusion:
    """Fuses ML-based and traditional bubble detection results."""

    def fuse_detections(
        self,
        traditional_response: dict,
        ml_blocks_response: list[dict],
        field_id_to_interpretation: dict,
        confidence_threshold: float = 0.85,
    ) -> tuple[dict, list[dict]]:
        """Fuse traditional and ML detection results."""

        fused_response = {}
        discrepancies = []

        for field_id, trad_interp in field_id_to_interpretation.items():
            # Get ML interpretation for same field
            ml_interp = self._find_ml_interpretation(field_id, ml_blocks_response)

            if ml_interp is None:
                # No ML detection, use traditional
                fused_response[field_id] = trad_interp
                continue

            # Both available - apply fusion strategy
            fused, discrepancy = self._confidence_weighted_fusion(
                field_id,
                trad_interp,
                ml_interp,
                confidence_threshold,
            )

            fused_response[field_id] = fused
            if discrepancy:
                discrepancies.append(discrepancy)

        return fused_response, discrepancies
```

### Fusion Strategies

```python
def _confidence_weighted_fusion(
    self,
    field_id: str,
    trad_interp,
    ml_interp: dict,
    confidence_threshold: float,
) -> tuple[any, dict | None]:
    """Confidence-weighted fusion strategy."""

    # Get confidence scores
    trad_confidence = getattr(trad_interp, "overall_confidence_score", 0.0)
    ml_confidence = ml_interp.get("confidence", 0.0)

    # Check agreement
    trad_response = getattr(trad_interp, "matched_bubbles", [])
    ml_response = ml_interp.get("detected_bubbles", [])
    responses_agree = self._responses_agree(trad_response, ml_response)

    # Decision matrix
    if trad_confidence > confidence_threshold and ml_confidence > confidence_threshold:
        # Both high confidence
        if responses_agree:
            return trad_interp, None  # Agreement - use traditional
        else:
            # Disagreement - flag for review, use higher confidence
            discrepancy = {
                "field_id": field_id,
                "traditional": {"response": trad_response, "confidence": trad_confidence},
                "ml": {"response": ml_response, "confidence": ml_confidence},
                "reason": "high_confidence_disagreement",
            }
            if trad_confidence > ml_confidence:
                return trad_interp, discrepancy
            else:
                return self._create_ml_interpretation(ml_interp), discrepancy

    elif trad_confidence < 0.6 and ml_confidence > confidence_threshold:
        # Low traditional, high ML - use ML
        return self._create_ml_interpretation(ml_interp), None

    else:
        # Default to traditional
        return trad_interp, None
```

---

## Browser Integration

### TensorFlow.js Integration

```typescript
// ml-bubble-detector.ts

import * as tf from '@tensorflow/tfjs';
import { ProcessingContext } from './processing-context';

export class MLBubbleDetector {
  private model: tf.GraphModel | null = null;
  private confidenceThreshold: number = 0.7;

  async loadModel(modelUrl: string): Promise<void> {
    // Load TensorFlow.js model
    this.model = await tf.loadGraphModel(modelUrl);
    console.log('ML Bubble Detector model loaded');
  }

  async process(context: ProcessingContext): Promise<ProcessingContext> {
    if (!this.model) {
      console.debug('ML bubble detector not initialized');
      return context;
    }

    // Get ML-detected blocks from context
    const mlBlocks = context.metadata.mlDetectedBlocks || [];

    if (mlBlocks.length === 0) {
      console.debug('No ML-detected blocks, skipping ML bubble detection');
      return context;
    }

    // Process each field block
    for (const block of mlBlocks) {
      // Crop block region
      const blockCrop = this.cropBlockRegion(
        context.grayImage,
        block.bboxXyxy
      );

      if (!blockCrop || blockCrop.width === 0) {
        console.warn(`Empty crop for block ${block.className}`);
        continue;
      }

      // Run bubble detection
      const bubbles = await this.detectBubbles(blockCrop, block.bboxXyxy);

      // Store results
      block.mlBubbles = bubbles;
      block.mlBubblesCount = bubbles.length;

      console.debug(
        `Detected ${bubbles.length} bubbles in block ${block.className}`
      );
    }

    console.info(`ML bubble detection complete for ${mlBlocks.length} blocks`);
    return context;
  }

  private async detectBubbles(
    blockCrop: ImageData,
    blockBbox: [number, number, number, number]
  ): Promise<BubbleDetection[]> {
    // Preprocess
    const tensor = tf.browser.fromPixels(blockCrop)
      .resizeBilinear([640, 640])
      .div(255.0)
      .expandDims(0);

    // Inference
    const predictions = this.model!.predict(tensor) as tf.Tensor;

    // Post-process
    const bubbles = await this.postProcess(predictions, blockBbox);

    // Cleanup
    tensor.dispose();
    predictions.dispose();

    return bubbles;
  }

  private cropBlockRegion(
    image: ImageData,
    bbox: [number, number, number, number]
  ): ImageData | null {
    const [x1, y1, x2, y2] = bbox;

    // Bounds checking
    const clampedX1 = Math.max(0, x1);
    const clampedY1 = Math.max(0, y1);
    const clampedX2 = Math.min(image.width, x2);
    const clampedY2 = Math.min(image.height, y2);

    const width = clampedX2 - clampedX1;
    const height = clampedY2 - clampedY1;

    if (width <= 0 || height <= 0) {
      return null;
    }

    // Crop using canvas
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = image.width;
    tempCanvas.height = image.height;
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCtx.putImageData(image, 0, 0);

    ctx.drawImage(
      tempCanvas,
      clampedX1, clampedY1, width, height,
      0, 0, width, height
    );

    return ctx.getImageData(0, 0, width, height);
  }

  private async postProcess(
    predictions: tf.Tensor,
    blockBbox: [number, number, number, number]
  ): Promise<BubbleDetection[]> {
    const [blockX1, blockY1] = blockBbox;
    const data = await predictions.data();

    const bubbles: BubbleDetection[] = [];

    // Parse YOLO output (assuming [x1, y1, x2, y2, conf, classId])
    for (let i = 0; i < data.length; i += 6) {
      const confidence = data[i + 4];
      if (confidence < this.confidenceThreshold) continue;

      const [x1Crop, y1Crop, x2Crop, y2Crop] = [
        data[i], data[i + 1], data[i + 2], data[i + 3]
      ];
      const classId = Math.round(data[i + 5]);

      // Map to full image coordinates
      bubbles.push({
        classId,
        className: this.getClassName(classId),
        confidence,
        bboxXyxy: [
          blockX1 + x1Crop,
          blockY1 + y1Crop,
          blockX1 + x2Crop,
          blockY1 + y2Crop
        ],
        state: classId === 1 ? 'filled' : 'empty'
      });
    }

    return bubbles;
  }

  private getClassName(classId: number): string {
    const CLASS_NAMES: Record<number, string> = {
      0: 'bubble_empty',
      1: 'bubble_filled',
      2: 'bubble_partial'
    };
    return CLASS_NAMES[classId] || 'unknown';
  }
}
```

### Pipeline Integration (Browser)

```typescript
// processing-pipeline.ts

import { MLFieldBlockDetector } from './ml-field-block-detector';
import { MLBubbleDetector } from './ml-bubble-detector';
import { ReadOMRProcessor } from './read-omr-processor';

export class ProcessingPipeline {
  private stages: Processor[] = [];

  async initialize(config: TuningConfig): Promise<void> {
    // ML Field Block Detection (Stage 1)
    if (config.ml.enableFieldBlockDetection) {
      const mlFieldBlockDetector = new MLFieldBlockDetector();
      await mlFieldBlockDetector.loadModel(config.ml.fieldBlockModelPath);
      this.stages.push(mlFieldBlockDetector);
    }

    // ML Bubble Detection (Stage 2)
    if (config.ml.enableBubbleDetection) {
      const mlBubbleDetector = new MLBubbleDetector();
      await mlBubbleDetector.loadModel(config.ml.bubbleModelPath);
      this.stages.push(mlBubbleDetector);
    }

    // Main detection (with optional ML fallback)
    const readOMR = new ReadOMRProcessor(
      template,
      config.ml.enableFallback ? config.ml.bubbleModelPath : null
    );
    this.stages.push(readOMR);
  }

  async process(context: ProcessingContext): Promise<ProcessingContext> {
    for (const stage of this.stages) {
      context = await stage.process(context);
    }
    return context;
  }
}
```

---

## Testing Integration

### Unit Tests

```python
# test_ml_bubble_detector.py

def test_ml_bubble_detector_integration_with_stage1():
    """Test ML bubble detector with field block input."""

    # Mock Stage 1 output
    context = ProcessingContext(
        gray_image=test_image,
        metadata={
            "ml_detected_blocks": [
                {
                    "class_name": "mcq_block_4",
                    "bbox_xyxy": [100, 200, 400, 350],
                    "confidence": 0.92
                }
            ]
        }
    )

    # Run Stage 2
    detector = MLBubbleDetector("models/bubble_detector.pt")
    result = detector.process(context)

    # Verify Stage 2 output
    blocks = result.metadata["ml_detected_blocks"]
    assert "ml_bubbles" in blocks[0]
    assert blocks[0]["ml_bubbles_count"] > 0
```

### Integration Tests

```python
def test_pipeline_with_ml_detection():
    """Test full pipeline with ML detection enabled."""

    config = TuningConfig(
        ml=MLConfig(
            enable_field_block_detection=True,
            enable_bubble_detection=True,
            field_block_model_path="models/field_block.pt",
            bubble_model_path="models/bubble.pt"
        )
    )

    pipeline = ProcessingPipeline(template, config)
    context = pipeline.process_image("test.png")

    # Verify ML detections were used
    assert "ml_detected_blocks" in context.metadata
    assert context.metadata["ml_detected_blocks"][0]["ml_bubbles_count"] > 0
```

---

## Summary

ML Bubble Detector integrates with:

1. **ML Field Block Detector (Stage 1)**: Consumes field block bounding boxes for cropping
2. **ReadOMR Processor**: Provides ML fallback for low-confidence traditional detections
3. **Shift Detection**: Uses bubble positions to calculate field shifts
4. **Pipeline**: Registered as optional processor stage
5. **Detection Fusion**: Merges ML and traditional results with confidence weighting

**Key Integration Patterns**:
- Context metadata for data passing (`ml_detected_blocks`)
- Graceful degradation (skip if Stage 1 missing)
- Optional/on-demand execution (enabled via config)
- Hybrid strategy (ML as fallback, not primary)
- Browser compatibility (TensorFlow.js/ONNX Runtime Web)

**Configuration-Driven**:
- Enable/disable via `ml.enable_bubble_detection`
- Model paths configurable
- Confidence thresholds tunable
- Fallback behavior configurable

**Next Steps**: See related documentation for complete system understanding.
