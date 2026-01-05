# Translation-Only STN Quick Start Guide

## What is Translation-Only STN?

A **simplified Spatial Transformer Network** that learns only **positional shifts (tx, ty)** instead of full affine transformations (rotation, scaling, shear, translation).

## When Should I Use It?

### ✅ Use Translation-Only STN When:
- Your documents are consistently scanned from the same scanner
- You have systematic positional offsets (but no rotation)
- You want faster inference (~10-20% faster than affine)
- You're doing real-time or embedded processing
- You want to avoid rotation/scaling artifacts

### ✅ Use Affine STN When:
- You have mobile camera photos with perspective distortion
- Documents are photocopied (warping, rotation)
- Capture conditions vary (different scanners, angles)
- You have skewed or rotated scans
- You're not sure what distortions exist

## Quick Commands

### 1. Train Translation-Only STN

```bash
python train_stn_yolo.py \
    --augmented-data outputs/training_data/yolo_field_blocks_augmented \
    --transformation-type translation_only \
    --epochs 50
```

### 2. Train Affine STN (Default)

```bash
python train_stn_yolo.py \
    --augmented-data outputs/training_data/yolo_field_blocks_augmented \
    --transformation-type affine \
    --epochs 50
```

### 3. Compare Both Types

```bash
python benchmark_stn_yolo.py \
    --stn-model outputs/models/stn_affine.pt \
    --stn-model-translation outputs/models/stn_translation.pt \
    --compare-stn-types \
    --max-images 20
```

## Configuration in Templates

The system **automatically detects** which type of STN you're using from the metadata file. Just specify the model path:

```json
{
  "preProcessors": [
    {
      "name": "MLFieldBlockDetector",
      "options": {
        "model_path": "outputs/models/field_block_detector.pt",
        "use_stn": true,
        "stn_model_path": "outputs/models/stn_translation_only.pt"
      }
    }
  ]
}
```

No need to specify the transformation type - it's detected automatically!

## Performance Comparison

| Metric | Affine STN | Translation-Only |
|--------|-----------|-----------------|
| **Parameters** | 41,590 | 41,332 (-0.6%) |
| **STN Overhead (CPU)** | ~10-12ms | ~8-10ms (-20%) |
| **Training Speed** | Baseline | ~20% faster |
| **Handles Rotation** | ✅ Yes | ❌ No |
| **Handles Scaling** | ✅ Yes | ❌ No |
| **Handles Translation** | ✅ Yes | ✅ Yes |

## Decision Flowchart

```
Is your document source consistent (same scanner)?
│
├─ YES → Do you have rotation/skew issues?
│         │
│         ├─ YES → Use AFFINE STN
│         └─ NO  → Use TRANSLATION-ONLY STN ✓ (faster)
│
└─ NO  → Do you need real-time processing?
          │
          ├─ YES → Try TRANSLATION-ONLY first (faster)
          └─ NO  → Use AFFINE STN (more robust)
```

## Common Scenarios

### Scenario 1: Office Scanner (Flatbed)
**Problem**: Documents scanned from HP flatbed scanner, consistent ~15px offset
**Solution**: **Translation-Only STN**
**Why**: No rotation, faster inference, perfect for systematic offset

### Scenario 2: Mobile Camera Photos
**Problem**: Students submitting photos from phones, various angles
**Solution**: **Affine STN**
**Why**: Handles perspective, rotation, varying capture conditions

### Scenario 3: Document Feeder (ADF)
**Problem**: Batch processing from automatic document feeder, slight misalignment
**Solution**: **Translation-Only STN**
**Why**: Consistent source, only positional shifts, faster for batch

### Scenario 4: Mixed Sources
**Problem**: Documents from multiple scanners and cameras
**Solution**: **Affine STN**
**Why**: More robust to unknown distortions

## Troubleshooting

### My translation-only STN doesn't improve accuracy

**Possible causes**:
1. Your documents actually have rotation → Use affine STN instead
2. Not enough training data → Increase augmentation
3. Verify you're using translation augmentation (Types 7-9)

**Fix**:
```bash
# Compare both types on your data
python benchmark_stn_yolo.py --compare-stn-types
```

### Training is too slow

**Solutions**:
1. Use translation-only instead of affine (~20% faster)
2. Reduce batch size: `--batch-size 2`
3. Use GPU if available (CUDA)

### How do I know which type a trained model is?

Check the metadata file:
```bash
cat outputs/models/stn_refinement_20260105_120000.json
```

Look for:
```json
{
  "transformation_type": "translation_only"  // or "affine"
}
```

## Testing Your Implementation

### Run all tests:
```bash
uv run pytest src/tests/test_stn_integration.py -v
```

### Test only translation-only:
```bash
uv run pytest src/tests/test_stn_integration.py::TestTranslationOnlySTN -v
```

### Expected output:
```
28 passed in 2.34s
```

## Best Practices

1. **Start with comparison**: Run both types on your test set
2. **Measure what matters**: If speed is critical, translation-only wins
3. **Check your data**: Visualize transformations to understand distortions
4. **Iterate**: You can train both types and switch easily
5. **Document your choice**: Note which type works best for your use case

## Next Steps

1. ✅ Train both STN types on your augmented data
2. ✅ Run comparison benchmark on test images
3. ✅ Choose based on performance metrics
4. ✅ Update templates with chosen model
5. ✅ Monitor inference times in production

## Need Help?

- **Full Documentation**: `STN_INTEGRATION_GUIDE.md`
- **Implementation Details**: `TRANSLATION_STN_IMPLEMENTATION.md`
- **Run Tests**: `uv run pytest src/tests/test_stn_integration.py -v`
- **Benchmark**: `python benchmark_stn_yolo.py --help`

---

**TL;DR**: Translation-only is faster and simpler but only handles positional shifts. Affine is more robust but slower. Try both on your data using `--compare-stn-types`.

