# Detection Fusion - Integration Guide

**Status**: Advanced Detection System
**Python Reference**: `src/processors/detection/fusion/detection_fusion.py`

---

## Overview

This document explains how Detection Fusion integrates with the ReadOMR detection pipeline, shift detection system, and ML detection components. It covers integration patterns, data flow, and usage examples.

---

## ReadOMR Pipeline Integration

### Integration Architecture

```
ProcessingPipeline
    ↓
Preprocessing → Alignment → ShiftDetection
    ↓
ReadOMRProcessor
    ├─> Traditional Detection (always runs)
    │   └─> TemplateFileRunner
    │       ├─> BubblesThresholdFileRunner
    │       ├─> OCRFileRunner
    │       └─> BarcodeFileRunner
    │
    ├─> ML Detection (optional)
    │   └─> MLFieldBlockDetector
    │       └─> ML blocks with bubble detections
    │
    └─> Detection Fusion (optional)
        └─> DetectionFusion.fuse_detections()
            ├─> Merge traditional + ML
            ├─> Apply fusion strategy
            └─> Return (fused_response, discrepancies)
```

---

### Integration Point 1: ReadOMR Processor

**Current State**: Fusion is **not integrated** in the main pipeline.

**Future Integration**:

```python
# src/processors/detection/processor.py

from src.processors.detection.fusion.detection_fusion import DetectionFusion

class ReadOMRProcessor(Processor):
    def __init__(
        self,
        template,
        ml_model_path: str | Path | None = None,
        enable_fusion: bool = False,  # New parameter
        fusion_strategy: str = "confidence_weighted",
    ):
        self.template = template
        self.template_file_runner = TemplateFileRunner(template)

        # ML fallback (existing)
        self.ml_detector = None
        self.hybrid_strategy = None
        if ml_model_path:
            self._initialize_ml_fallback(ml_model_path)

        # Detection fusion (new)
        self.enable_fusion = enable_fusion
        self.fusion = None
        if enable_fusion:
            self.fusion = DetectionFusion(
                fusion_strategy=fusion_strategy,
                discrepancy_threshold=0.3,
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Execute OMR detection with optional fusion."""

        # Traditional detection (always runs)
        raw_omr_response = self.template_file_runner.read_omr_and_update_metrics(
            file_path, gray_image, colored_image
        )

        # Extract interpretation metrics
        field_id_to_interpretation = ...

        # ML detection (if enabled)
        ml_blocks_response = []
        if self.ml_detector:
            ml_context = self.ml_detector.process(context)
            ml_blocks_response = ml_context.metadata.get("ml_detected_blocks", [])

        # Detection fusion (if enabled)
        if self.fusion and ml_blocks_response:
            fused_response, discrepancies = self.fusion.fuse_detections(
                traditional_response={},  # Unused currently
                ml_blocks_response=ml_blocks_response,
                field_id_to_interpretation=field_id_to_interpretation,
                confidence_threshold=0.85,
            )

            # Use fused response instead of traditional
            context.field_id_to_interpretation = fused_response

            # Store discrepancies for review
            context.metadata["fusion_discrepancies"] = discrepancies

            logger.info(
                f"Fusion complete: {len(fused_response)} fields, "
                f"{len(discrepancies)} discrepancies"
            )
        else:
            # No fusion - use traditional results
            context.field_id_to_interpretation = field_id_to_interpretation

        return context
```

---

### Integration Point 2: Shift Detection Processor

**Shift Detection** runs ML-based field block detection and applies shifts to field positions.

**Integration Flow**:

```
ShiftDetectionProcessor
    ↓
1. Run MLFieldBlockDetector
    └─> Detects field blocks with ML
    └─> Computes alignment adjustments
    └─> Stores in context.metadata["ml_detected_blocks"]
    ↓
2. Apply shifts to field positions
    └─> Updates template field origins
    ↓
3. Run ReadOMR detection with shifted fields
    └─> Traditional detection uses shifted positions
    └─> ML blocks already detected (reused)
    ↓
4. Detection Fusion (optional)
    └─> Fuse shifted traditional + original ML detections
    └─> Return unified response
```

**Code Integration**:

```python
# src/processors/detection/shift_detection_processor.py

class ShiftDetectionProcessor(Processor):
    def __init__(
        self,
        template,
        ml_field_block_model_path: str,
        enable_fusion: bool = True,  # New parameter
        fusion_strategy: str = "confidence_weighted",
    ):
        self.ml_field_block_detector = MLFieldBlockDetector(
            model_path=ml_field_block_model_path
        )

        self.template_file_runner = TemplateFileRunner(template)

        # Detection fusion (new)
        self.enable_fusion = enable_fusion
        self.fusion = None
        if enable_fusion:
            self.fusion = DetectionFusion(
                fusion_strategy=fusion_strategy,
                discrepancy_threshold=0.3,
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # Step 1: ML field block detection
        context = self.ml_field_block_detector.process(context)
        ml_blocks = context.metadata.get("ml_detected_blocks", [])

        # Step 2: Apply shifts to field positions
        self._apply_shifts_to_template(ml_blocks)

        # Step 3: Traditional detection with shifted positions
        raw_omr_response = self.template_file_runner.read_omr_and_update_metrics(
            file_path, gray_image, colored_image
        )

        field_id_to_interpretation = ...

        # Step 4: Detection fusion (optional)
        if self.fusion and ml_blocks:
            fused_response, discrepancies = self.fusion.fuse_detections(
                traditional_response={},
                ml_blocks_response=ml_blocks,
                field_id_to_interpretation=field_id_to_interpretation,
                confidence_threshold=0.85,
            )

            context.field_id_to_interpretation = fused_response
            context.metadata["fusion_discrepancies"] = discrepancies
        else:
            context.field_id_to_interpretation = field_id_to_interpretation

        # Mark shift detection as complete
        context.metadata["shift_detection"] = {
            "applied": True,
            "blocks_detected": len(ml_blocks),
            "fusion_enabled": self.enable_fusion,
        }

        return context
```

---

## ML Field Block Detector Integration

### ML Block Response Format

**MLFieldBlockDetector Output**:

```python
ml_blocks_response = [
    {
        "class_id": 0,
        "class_name": "field_block_mcq",
        "confidence": 0.92,
        "bbox_origin": [100, 200],  # [x, y]
        "bbox_dimensions": [400, 150],  # [width, height]
        "bbox_xyxy": [100, 200, 500, 350],  # [x1, y1, x2, y2]
        # Future: Add bubble-level detections
        "fields": [
            {
                "field_id": "Q1",
                "bubbles": ["A"],
                "confidence": 0.88
            }
        ]
    },
    {
        "class_id": 0,
        "class_name": "field_block_mcq",
        "confidence": 0.89,
        "bbox_origin": [100, 400],
        "bbox_dimensions": [400, 150],
        "bbox_xyxy": [100, 400, 500, 550],
        "fields": [
            {
                "field_id": "Q2",
                "bubbles": ["B"],
                "confidence": 0.91
            }
        ]
    }
]
```

**Fusion Requirements**:
- Each ML block must include field-level detections (currently missing)
- Field IDs must match template field IDs
- Bubble detections must be in same format as traditional detections

---

### Future ML Block Enhancement

**Required for Fusion**:

```python
# src/processors/detection/ml_field_block_detector.py

class MLFieldBlockDetector(Processor):
    def process(self, context: ProcessingContext) -> ProcessingContext:
        # Step 1: Detect field blocks (existing)
        detected_blocks = self._parse_block_detections(results, image.shape)

        # Step 2: Run bubble-level detection per block (NEW)
        for block in detected_blocks:
            # Extract block ROI
            roi = self._extract_roi(context.gray_image, block["bbox_xyxy"])

            # Run ML bubble detector on ROI
            bubble_detections = self._detect_bubbles_in_block(roi, block)

            # Add to block
            block["fields"] = bubble_detections
            # Example:
            # {
            #   "field_id": "Q1",
            #   "bubbles": ["A"],
            #   "confidence": 0.88
            # }

        context.metadata["ml_detected_blocks"] = detected_blocks
        return context

    def _detect_bubbles_in_block(self, roi, block):
        """Run ML bubble detection on block ROI.

        Returns list of field detections.
        """
        # Use MLBubbleDetector or YOLO bubble model
        # Map detections to field IDs
        # Return structured field detections
        pass
```

---

## Template Configuration Integration

### Fusion Configuration in Template

**Template Config** (`template.json`):

```json
{
  "templateDimensions": [1200, 1600],
  "tuningConfig": {
    "thresholding": {
      "strategy": "global",
      "looseness": 4
    },
    "ml": {
      "enable_ml_fallback": true,
      "ml_model_path": "models/bubble_detector.pt",
      "confidence_threshold": 0.75,
      "enable_fusion": true,
      "fusion_strategy": "confidence_weighted",
      "fusion_discrepancy_threshold": 0.3
    }
  },
  "fieldBlocks": [...]
}
```

**Usage**:

```python
# Extract fusion config from template
tuning_config = template.tuning_config
ml_config = tuning_config.ml

enable_fusion = getattr(ml_config, "enable_fusion", False)
fusion_strategy = getattr(ml_config, "fusion_strategy", "confidence_weighted")
discrepancy_threshold = getattr(ml_config, "fusion_discrepancy_threshold", 0.3)

# Initialize ReadOMR with fusion
processor = ReadOMRProcessor(
    template=template,
    ml_model_path=getattr(ml_config, "ml_model_path", None),
    enable_fusion=enable_fusion,
    fusion_strategy=fusion_strategy,
)
```

---

## Context Integration

### ProcessingContext Fields

**New Fields for Fusion**:

```python
# src/processors/base.py

@dataclass
class ProcessingContext:
    # Existing fields
    file_path: str
    gray_image: np.ndarray
    colored_image: np.ndarray
    template: Template
    omr_response: dict
    field_id_to_interpretation: dict

    # Fusion-specific fields (in metadata)
    metadata: dict = field(default_factory=dict)
    # metadata["ml_detected_blocks"]: ML block detections
    # metadata["fusion_discrepancies"]: Flagged discrepancies
    # metadata["fusion_stats"]: Fusion statistics
```

**Fusion Stats Example**:

```python
context.metadata["fusion_stats"] = {
    "total_fields": 50,
    "ml_available": 48,  # 2 fields missing ML detection
    "fused_fields": 48,
    "traditional_used": 30,
    "ml_used": 18,
    "discrepancies_flagged": 5,
    "strategy": "confidence_weighted",
}
```

---

## Discrepancy Handling Integration

### Storing Discrepancies

**Context Storage**:

```python
# After fusion
context.metadata["fusion_discrepancies"] = [
    {
        "field_id": "Q5",
        "traditional": {"response": ["C"], "confidence": 0.89},
        "ml": {"response": ["D"], "confidence": 0.86},
        "reason": "high_confidence_disagreement",
    },
    {
        "field_id": "Q12",
        "traditional": {"response": ["A"], "confidence": 0.91},
        "ml": {"response": ["B"], "confidence": 0.88},
        "reason": "high_confidence_disagreement",
    }
]
```

---

### Discrepancy Output Integration

**CSV Export with Discrepancies**:

```python
# src/processors/output/csv_writer.py

def write_csv_with_discrepancies(context: ProcessingContext):
    """Write CSV with discrepancy flags."""

    # Standard CSV columns
    csv_row = {
        "file_path": context.file_path,
        "Q1": context.omr_response["Q1"],
        "Q2": context.omr_response["Q2"],
        # ...
    }

    # Add discrepancy flags
    discrepancies = context.metadata.get("fusion_discrepancies", [])
    discrepancy_fields = {d["field_id"] for d in discrepancies}

    for field_id in discrepancy_fields:
        csv_row[f"{field_id}_NEEDS_REVIEW"] = "YES"

    # Write to CSV
    csv_writer.writerow(csv_row)
```

**CSV Output**:

```csv
file_path,Q1,Q2,Q3,Q4,Q5,Q5_NEEDS_REVIEW,Q12,Q12_NEEDS_REVIEW
sheet1.jpg,A,B,C,D,C,YES,A,YES
sheet2.jpg,A,B,C,D,A,,B,
```

---

### HTML Report Integration

**Visualization Output**:

```python
# src/processors/visualization/html_exporter.py

def export_discrepancies_report(context: ProcessingContext):
    """Generate HTML report for discrepancies."""

    discrepancies = context.metadata.get("fusion_discrepancies", [])

    html = "<html><body>"
    html += "<h1>Fusion Discrepancies Report</h1>"

    for disc in discrepancies:
        html += f"<div class='discrepancy'>"
        html += f"<h2>Field: {disc['field_id']}</h2>"
        html += f"<p><strong>Reason:</strong> {disc['reason']}</p>"
        html += f"<p><strong>Traditional:</strong> {disc['traditional']['response']} "
        html += f"(confidence: {disc['traditional']['confidence']:.2f})</p>"
        html += f"<p><strong>ML:</strong> {disc['ml']['response']} "
        html += f"(confidence: {disc['ml']['confidence']:.2f})</p>"
        html += f"</div>"

    html += "</body></html>"

    # Write to file
    with open("fusion_discrepancies.html", "w") as f:
        f.write(html)
```

---

## Usage Examples

### Example 1: Basic Fusion Integration

```python
from src.processors.detection.fusion.detection_fusion import DetectionFusion

# Initialize fusion
fusion = DetectionFusion(
    fusion_strategy="confidence_weighted",
    discrepancy_threshold=0.3,
)

# Get detection results
field_id_to_interpretation = {
    "Q1": BubblesFieldInterpretation(...),
    "Q2": BubblesFieldInterpretation(...),
}

ml_blocks_response = [...]  # From MLFieldBlockDetector

# Fuse detections
fused_response, discrepancies = fusion.fuse_detections(
    traditional_response={},
    ml_blocks_response=ml_blocks_response,
    field_id_to_interpretation=field_id_to_interpretation,
    confidence_threshold=0.85,
)

# Use fused results
for field_id, interpretation in fused_response.items():
    print(f"{field_id}: {interpretation.get_field_interpretation_string()}")

# Review discrepancies
for disc in discrepancies:
    print(f"⚠️  {disc['field_id']}: {disc['reason']}")
```

---

### Example 2: ML Fallback Strategy

```python
# High-speed processing with ML safety net
fusion = DetectionFusion(
    fusion_strategy="ml_fallback"
)

fused_response, discrepancies = fusion.fuse_detections(
    traditional_response={},
    ml_blocks_response=ml_blocks_response,
    field_id_to_interpretation=field_id_to_interpretation,
)

# ML only used when traditional confidence < 0.6
# No discrepancies flagged (faster processing)
assert len(discrepancies) == 0
```

---

### Example 3: Traditional Primary (ML Evaluation)

```python
# Evaluate ML model accuracy
fusion = DetectionFusion(
    fusion_strategy="traditional_primary"
)

fused_response, discrepancies = fusion.fuse_detections(
    traditional_response={},
    ml_blocks_response=ml_blocks_response,
    field_id_to_interpretation=field_id_to_interpretation,
)

# Results always use traditional
# But discrepancies show ML differences
print(f"ML disagreed on {len(discrepancies)} fields:")
for disc in discrepancies:
    print(f"  {disc['field_id']}: Traditional={disc['traditional']}, ML={disc['ml']}")

# Use for ML model improvement
```

---

### Example 4: Pipeline Integration

```python
# Full pipeline with fusion
from src.processors.pipeline import ProcessingPipeline

# Configure template with fusion
template_config = {
    "tuningConfig": {
        "ml": {
            "enable_fusion": True,
            "fusion_strategy": "confidence_weighted",
            "fusion_discrepancy_threshold": 0.3,
        }
    }
}

# Create pipeline
pipeline = ProcessingPipeline(template)

# Process OMR sheet
context = pipeline.process_file("sheet1.jpg")

# Check fusion results
if "fusion_discrepancies" in context.metadata:
    discrepancies = context.metadata["fusion_discrepancies"]
    print(f"Fusion flagged {len(discrepancies)} discrepancies")

    # Export for review
    export_discrepancies_report(context)
```

---

## Browser Integration

### TypeScript Implementation

```typescript
// Browser-compatible fusion class
import type {
  FieldInterpretation,
  MLBlock,
  Discrepancy
} from './types';

class DetectionFusion {
  private fusionStrategies = {
    confidence_weighted: this.confidenceWeightedFusion.bind(this),
    ml_fallback: this.mlFallbackFusion.bind(this),
    traditional_primary: this.traditionalPrimaryFusion.bind(this),
  };

  constructor(
    private fusionStrategy: keyof typeof this.fusionStrategies = "confidence_weighted",
    private discrepancyThreshold: number = 0.3
  ) {}

  fuseDetections(
    traditionalResponse: Record<string, any>,
    mlBlocksResponse: MLBlock[],
    fieldIdToInterpretation: Record<string, FieldInterpretation>,
    confidenceThreshold: number = 0.85
  ): [Record<string, any>, Discrepancy[]] {
    const fusedResponse: Record<string, any> = {};
    const discrepancies: Discrepancy[] = [];

    for (const [fieldId, tradInterp] of Object.entries(fieldIdToInterpretation)) {
      const mlInterp = this.findMLInterpretation(fieldId, mlBlocksResponse);

      if (!mlInterp) {
        fusedResponse[fieldId] = tradInterp;
        continue;
      }

      const strategyMethod = this.fusionStrategies[this.fusionStrategy];
      const [fused, discrepancy] = strategyMethod(
        fieldId,
        tradInterp,
        mlInterp,
        confidenceThreshold
      );

      fusedResponse[fieldId] = fused;
      if (discrepancy) {
        discrepancies.push(discrepancy);
      }
    }

    console.log(
      `Fusion complete: ${Object.keys(fusedResponse).length} fields, ` +
      `${discrepancies.length} discrepancies`
    );

    return [fusedResponse, discrepancies];
  }

  // Strategy methods...
  private confidenceWeightedFusion(...) { ... }
  private mlFallbackFusion(...) { ... }
  private traditionalPrimaryFusion(...) { ... }
}

export { DetectionFusion };
```

---

### React Component Integration

```typescript
import { DetectionFusion } from './fusion';

function OMRProcessor() {
  const [fusedResponse, setFusedResponse] = useState({});
  const [discrepancies, setDiscrepancies] = useState([]);

  async function processOMR(imageFile: File) {
    // Run traditional detection
    const traditionalResult = await runTraditionalDetection(imageFile);

    // Run ML detection
    const mlBlocks = await runMLDetection(imageFile);

    // Fuse results
    const fusion = new DetectionFusion("confidence_weighted");
    const [fused, discs] = fusion.fuseDetections(
      {},
      mlBlocks,
      traditionalResult.fieldIdToInterpretation,
      0.85
    );

    setFusedResponse(fused);
    setDiscrepancies(discs);
  }

  return (
    <div>
      <h2>OMR Results</h2>
      {/* Display fused results */}

      {discrepancies.length > 0 && (
        <div className="discrepancies">
          <h3>⚠️  {discrepancies.length} Fields Need Review</h3>
          {discrepancies.map(disc => (
            <DiscrepancyCard key={disc.fieldId} discrepancy={disc} />
          ))}
        </div>
      )}
    </div>
  );
}
```

---

## Testing Integration

### Unit Tests

```python
# tests/processors/detection/fusion/test_detection_fusion.py

import pytest
from src.processors.detection.fusion.detection_fusion import DetectionFusion

def test_confidence_weighted_fusion_agreement():
    """Test confidence-weighted fusion with high confidence agreement."""
    fusion = DetectionFusion(fusion_strategy="confidence_weighted")

    # Mock interpretations
    trad_interp = create_mock_interpretation(
        matched_bubbles=["A"],
        confidence=0.88
    )
    ml_interp = {
        "detected_bubbles": ["A"],
        "confidence": 0.92
    }

    # Run fusion
    fused, discrepancy = fusion._confidence_weighted_fusion(
        "Q1", trad_interp, ml_interp, 0.85
    )

    # Assert: Uses traditional (agreement)
    assert fused == trad_interp
    assert discrepancy is None

def test_ml_fallback_low_confidence():
    """Test ML fallback with low traditional confidence."""
    fusion = DetectionFusion(fusion_strategy="ml_fallback")

    trad_interp = create_mock_interpretation(confidence=0.52)
    ml_interp = {"detected_bubbles": ["B"], "confidence": 0.91}

    fused, discrepancy = fusion._ml_fallback_fusion(
        "Q2", trad_interp, ml_interp, 0.85
    )

    # Assert: Uses ML (low traditional confidence)
    assert fused != trad_interp  # ML interpretation created
    assert discrepancy is None
```

---

### Integration Tests

```python
def test_full_fusion_pipeline():
    """Test complete fusion pipeline with real detections."""

    # Run traditional detection
    traditional_result = run_traditional_detection(test_image)

    # Run ML detection
    ml_blocks = run_ml_detection(test_image)

    # Fuse
    fusion = DetectionFusion("confidence_weighted")
    fused_response, discrepancies = fusion.fuse_detections(
        traditional_response={},
        ml_blocks_response=ml_blocks,
        field_id_to_interpretation=traditional_result,
        confidence_threshold=0.85
    )

    # Assert
    assert len(fused_response) == len(traditional_result)
    assert all(field_id in fused_response for field_id in traditional_result)
```

---

## Monitoring & Logging Integration

### Fusion Statistics Logging

```python
# Log fusion statistics per file
logger.info(f"Fusion Stats for {context.file_path}:")
logger.info(f"  Total fields: {fusion_stats['total_fields']}")
logger.info(f"  ML available: {fusion_stats['ml_available']}")
logger.info(f"  Traditional used: {fusion_stats['traditional_used']}")
logger.info(f"  ML used: {fusion_stats['ml_used']}")
logger.info(f"  Discrepancies: {fusion_stats['discrepancies_flagged']}")

# Directory-level aggregates
logger.info("=" * 60)
logger.info("Directory-Level Fusion Summary")
logger.info("=" * 60)
logger.info(f"Total files processed: {total_files}")
logger.info(f"Average discrepancies per file: {avg_discrepancies:.2f}")
logger.info(f"ML usage rate: {ml_usage_rate:.1f}%")
logger.info("=" * 60)
```

---

## Next Steps

1. Implement field-to-ML-block mapping (`_find_ml_interpretation`)
2. Implement agreement check (`_responses_agree`)
3. Integrate fusion into ReadOMR processor
4. Add fusion configuration to template schema
5. Create discrepancy review UI (HTML export)
6. Add fusion statistics to CSV output

---

## Related Modules

- **ReadOMR Processor** (`../concept.md`) - Main detection orchestrator
- **ML Field Block Detector** (`../ml-field-block/`) - ML block detection
- **Shift Detection** (`../shift-detection/`) - ML-based alignment
- **ProcessingContext** (`../../processing-context/`) - Data flow container
- **Template** (`../../template/`) - Configuration and fields
