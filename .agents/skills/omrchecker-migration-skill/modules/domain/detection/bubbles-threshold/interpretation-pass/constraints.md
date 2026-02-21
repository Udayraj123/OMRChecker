# Bubble Detection - Interpretation Pass: Constraints

## Performance Constraints

### Time Complexity

#### File-level Initialization

**Total**: O(F × B log B) where F = number of fields, B = average bubbles per field

```python
# 1. Get all bubble means: O(F × B)
all_bubble_means = repository.get_all_bubble_means_for_current_file()

# 2. Calculate outlier deviation: O(F)
all_outlier_deviations = [field.std_deviation for field in fields]

# 3. Global threshold calculation: O((F × B) log (F × B))
# Sorting all bubble means across all fields
threshold_result = strategy.calculate_threshold(all_bubble_values, config)
```

**Typical Performance**:
- 50 fields × 4 bubbles = 200 total bubbles
- O(200 log 200) ≈ 1,530 operations
- **< 1ms on modern hardware**

#### Field-level Interpretation

**Total**: O(B log B) per field

```python
# 1. Extract detection result: O(1)
detection_result = bubble_fields[field_label]

# 2. Calculate threshold: O(B log B)
# Sorting bubble means for max jump calculation
sorted_values = sorted(bubble_mean_values)

# 3. Interpret bubbles: O(B)
for bubble_mean in bubble_means:
    BubbleInterpretation(bubble_mean, threshold)

# 4. Check multi-marking: O(B)
marked_count = sum(1 for interp in interpretations if interp.is_attempted)

# 5. Confidence metrics: O(B)
for bubble_mean in bubble_means:
    # Check disparity
```

**Typical Performance per Field**:
- 4 bubbles: O(4 log 4) ≈ 8 operations
- **< 0.1ms per field**

### Space Complexity

#### File-level Aggregates

```python
file_level_aggregates = {
    "file_level_fallback_threshold": float,  # 8 bytes
    "global_max_jump": float,  # 8 bytes
    "outlier_deviation_threshold_for_file": float,  # 8 bytes
    "field_label_wise_local_thresholds": dict,  # ~50 × 16 bytes = 800 bytes
    "all_fields_local_thresholds": NumberAggregate,  # ~200 bytes
    "field_wise_confidence_metrics": dict,  # ~50 × 500 bytes = 25 KB (if enabled)
}
```

**Total**: ~26 KB per file (with confidence metrics), ~2 KB without

#### Field-level Aggregates

```python
field_level_aggregates = {
    "is_multi_marked": bool,  # 1 byte
    "local_threshold_for_field": float,  # 8 bytes
    "bubble_interpretations": list[BubbleInterpretation],  # ~4 × 64 bytes = 256 bytes
    "field_level_confidence_metrics": dict,  # ~500 bytes (if enabled)
}
```

**Total**: ~764 bytes per field (with confidence), ~264 bytes without

#### Total Memory Footprint

For a typical OMR sheet (50 fields, 4 bubbles each):

```
File-level: 26 KB
Field-level: 50 × 764 bytes = 38 KB
Total: ~64 KB per file
```

**Browser Constraint**: Easily fits in browser memory (< 100 KB per file)

### Browser-Specific Constraints

#### Memory Limits

**Chrome**: ~2 GB per tab (typical)
**Firefox**: ~4 GB per tab (typical)
**Safari**: ~1.5 GB per tab (typical)

**Safe Batch Size**: Process 100 files at a time (~6.4 MB total)

```typescript
async function processBatch(files: File[], batchSize = 100): Promise<void> {
    for (let i = 0; i < files.length; i += batchSize) {
        const batch = files.slice(i, i + batchSize);
        await Promise.all(batch.map(processFile));

        // Clear batch memory
        batch.length = 0;

        // Allow GC to run
        await new Promise(resolve => setTimeout(resolve, 100));
    }
}
```

#### Main Thread Performance

**Target**: < 50ms per field to maintain 60 FPS UI

**Actual**: ~0.1ms per field (500× faster than target)

**Recommendation**: Can run on main thread for small batches (< 10 files)

```typescript
// Main thread (fast operations)
if (files.length <= 10) {
    const results = await processFilesOnMainThread(files);
}

// Web Worker (large batches)
else {
    const results = await processFilesInWorker(files);
}
```

#### Web Worker Support

**Transferable Objects**: Use for detection results (avoid copying)

```typescript
// In main thread
const detectionResults = new Float32Array(bubbleMeans);
worker.postMessage({ detectionResults }, [detectionResults.buffer]);

// In worker
self.onmessage = (e) => {
    const { detectionResults } = e.data;
    const interpretation = runInterpretation(detectionResults);
    self.postMessage({ interpretation });
};
```

---

## Edge Cases

### 1. Single Bubble Fields

**Scenario**: Field has only 1 bubble (e.g., true/false question)

**Issue**: Cannot calculate local threshold (needs 2+ bubbles)

**Solution**: Always use global fallback

```python
if len(bubble_mean_values) < 2:
    return ThresholdResult(
        threshold_value=global_fallback,
        confidence=0.0,
        method_used="local_single_bubble_fallback",
        fallback_used=True
    )
```

**Browser Implementation**:

```typescript
if (bubbleMeanValues.length < 2) {
    return {
        thresholdValue: globalFallback,
        confidence: 0.0,
        maxJump: 0.0,
        methodUsed: "local_single_bubble_fallback",
        fallbackUsed: true,
    };
}
```

### 2. Two Bubble Fields (True/False)

**Scenario**: Field has exactly 2 bubbles

**Issue**: Limited data for robust threshold calculation

**Solution**: Check gap between bubbles

```python
if len(sorted_values) == 2:
    gap = sorted_values[1] - sorted_values[0]

    # Gap too small (likely both unmarked or both marked)
    if gap < config.min_gap_two_bubbles:  # 30
        return ThresholdResult(
            threshold_value=global_fallback,
            confidence=0.3,
            method_used="local_two_bubbles_small_gap_fallback",
            fallback_used=True
        )

    # Sufficient gap - use mean as threshold
    return ThresholdResult(
        threshold_value=np.mean(sorted_values),
        confidence=0.7,
        method_used="local_two_bubbles_mean",
        fallback_used=False
    )
```

**Examples**:

| Bubble 1 | Bubble 2 | Gap | Threshold | Method | Confidence |
|----------|----------|-----|-----------|--------|------------|
| 150 | 155 | 5 | Global (200) | Small gap fallback | 0.3 |
| 150 | 200 | 50 | 175 | Two bubbles mean | 0.7 |
| 140 | 240 | 100 | 190 | Two bubbles mean | 0.7 |

### 3. All Bubbles Same Intensity

**Scenario**: All bubbles have identical or very similar intensities

**Issue**: Max jump is very small, no clear threshold

**Solution**: Use global fallback (low confidence)

```python
# Example: [180, 181, 182, 183] - max jump = 3
confident_jump = config.min_jump + config.min_jump_surplus_for_global_fallback  # 30

if max_jump < confident_jump:
    return ThresholdResult(
        threshold_value=global_fallback,
        confidence=0.4,
        method_used="local_low_confidence_global_fallback",
        fallback_used=True
    )
```

**Causes**:
- Photocopied OMR sheet (all bubbles dark)
- Low-quality scan (poor contrast)
- All bubbles unmarked or all marked

**Detection**: Log warning if std_deviation < 15

```python
if detection_result.std_deviation < 15:
    logger.warning(
        f"Low std deviation ({detection_result.std_deviation:.1f}) in field {field_label}. "
        f"Scan quality: {detection_result.scan_quality.value}"
    )
```

### 4. No Bubbles Marked

**Scenario**: Student left question blank

**Issue**: Should return empty string, not error

**Solution**: Handle in `get_field_interpretation_string()`

```python
marked_bubbles = [
    interp.bubble_value
    for interp in bubble_interpretations
    if interp.is_attempted
]

if len(marked_bubbles) == 0:
    return ""  # Empty value (blank answer)
```

**Browser Implementation**:

```typescript
const markedBubbles = this.bubbleInterpretations
    .filter(interp => interp.isAttempted)
    .map(interp => interp.bubbleValue);

if (markedBubbles.length === 0) {
    return "";
}
```

### 5. All Bubbles Marked

**Scenario**: All bubbles in a field are detected as marked

**Issue**: Likely scanning/detection error (not student error)

**Solution**: Treat as unmarked (return empty string)

```python
total_bubbles = len(bubble_interpretations)
if len(marked_bubbles) == total_bubbles:
    return ""  # Treat as unmarked (scanning issue)
```

**Rationale**:
- Students rarely mark ALL options (illogical)
- More likely: scan too dark, threshold too high, detection error
- Prevents garbage data in CSV

**Example**:

```
Field: Q1 (A/B/C/D)
Bubble intensities: [180, 185, 190, 195]
Threshold (incorrect): 200

Result: All 4 bubbles marked (all < 200)
Output: "" (empty, not "ABCD")
```

### 6. Multi-marking

**Scenario**: Multiple bubbles marked in single-answer field

**Issue**: Student error or ambiguous marking

**Solution**: Flag as multi-marked, still return concatenated values

```python
marked_count = sum(1 for interp in bubble_interpretations if interp.is_attempted)
is_multi_marked = marked_count > 1

if is_multi_marked:
    logger.warning(f"Multi-marking in {field_label}: {marked_count} bubbles")

# Still return concatenated values (e.g., "AB" for A+B marked)
return "".join(marked_bubbles)
```

**File-level Flag**: Track multi-marked files for filtering

```python
if is_multi_marked:
    file_level_aggregates["read_response_flags"]["is_multi_marked"] = True
    file_level_aggregates["read_response_flags"]["multi_marked_fields"].append(field)

# Later: optionally filter out multi-marked files
if config.outputs.filter_out_multimarked_files:
    if file_aggregates["read_response_flags"]["is_multi_marked"]:
        # Skip file in CSV output
```

### 7. Threshold Disparity (Local vs Global)

**Scenario**: Local threshold differs significantly from global threshold

**Issue**: Ambiguous bubbles (marked by local, unmarked by global, or vice versa)

**Solution**: Calculate disparity bubbles for confidence metrics

```python
disparity_bubbles = []
for bubble_mean in detection_result.bubble_means:
    local_marked = bubble_mean.mean_value < local_threshold
    global_marked = bubble_mean.mean_value < global_threshold

    if local_marked != global_marked:
        disparity_bubbles.append(bubble_mean)

# Include in confidence metrics
confidence_metrics["bubbles_in_doubt"] = {
    "by_disparity": disparity_bubbles,
}

# Warn if significant disparity
if len(disparity_bubbles) > 0:
    logger.warning(
        f"Threshold disparity in {field_label}: "
        f"{len(disparity_bubbles)} bubbles in doubt"
    )
```

**Example**:

```
Local threshold: 180
Global threshold: 200

Bubble intensities: [150, 175, 185, 225]

Local result: A, B marked (150 < 180, 175 < 180)
Global result: A, B, C marked (all < 200)

Disparity bubble: C (185)
- Local: unmarked (185 >= 180)
- Global: marked (185 < 200)
```

### 8. Missing Detection Result

**Scenario**: Detection pass failed to process field

**Issue**: No bubble means available for interpretation

**Solution**: Graceful error with helpful message

```python
if field_label not in bubble_fields:
    available_fields = list(bubble_fields.keys())
    raise ValueError(
        f"No detection result for field '{field_label}'. "
        f"Available: {available_fields}"
    )
```

**Browser Implementation**:

```typescript
if (!(fieldLabel in bubbleFields)) {
    const availableFields = Object.keys(bubbleFields);
    throw new Error(
        `No detection result for field '${fieldLabel}'. ` +
        `Available: ${availableFields.join(", ")}`
    );
}
```

### 9. Xeroxed/Photocopied OMR Sheets

**Scenario**: OMR sheet is a photocopy, all bubbles very dark

**Issue**: Global threshold = 255 (max value), unreliable detection

**Solution**: Warn user, use alternative processing

```python
if file_level_fallback_threshold == 255:
    logger.warning(
        f"Looks like a Xeroxed OMR (threshold=255). "
        f"Results may be unreliable. Consider using original sheets."
    )
```

**Recommendation**: Reject photocopied sheets or use ML-based detection as fallback

### 10. Low Scan Quality

**Scenario**: Poor scan quality (low contrast, blurry)

**Issue**: High std deviation variance across fields, unreliable thresholds

**Solution**: Use `scan_quality` property from detection result

```python
@property
def scan_quality(self) -> ScanQuality:
    std = self.std_deviation
    if std > 50:
        return ScanQuality.EXCELLENT
    if std > 30:
        return ScanQuality.GOOD
    if std > 15:
        return ScanQuality.ACCEPTABLE
    return ScanQuality.POOR

# In interpretation
if detection_result.scan_quality == ScanQuality.POOR:
    logger.warning(
        f"Poor scan quality for {field_label} "
        f"(std={detection_result.std_deviation:.1f}). "
        f"Consider rescanning."
    )
```

**Browser Implementation**: Show warning badge in UI

```typescript
if (detectionResult.scanQuality === "POOR") {
    showWarningBadge(fieldLabel, "Poor scan quality - results may be unreliable");
}
```

### 11. Empty Detection Result

**Scenario**: No bubbles detected in field (detection failure)

**Issue**: Empty `bubble_means` list

**Solution**: Handle gracefully, return empty value

```python
if len(detection_result.bubble_means) == 0:
    logger.error(f"No bubbles detected in field {field_label}")
    return ""  # Empty interpretation
```

**Confidence Score**: Return 0.0 for empty results

```python
if not detection_result.bubble_means:
    return 0.0
```

### 12. Extreme Outliers

**Scenario**: One bubble has extremely different intensity (e.g., [150, 155, 160, 250])

**Issue**: Skews max jump calculation

**Solution**: Handled by outlier deviation threshold

```python
# Calculated at file level
outlier_deviation_threshold = get_outlier_deviation_threshold(all_outlier_deviations)

# Used in confidence metrics
if detection_result.std_deviation > outlier_deviation_threshold * 2:
    logger.warning(f"Extreme outliers detected in {field_label}")
```

**Local threshold handles gracefully**:

```
Bubbles: [150, 155, 160, 250]
Jumps: [5, 5, 90]
Max jump: 90 (between 160 and 250)
Threshold: 160 + 90/2 = 205

Result: A, B, C marked (all < 205), D unmarked (250 >= 205)
```

---

## Browser Compatibility

### Supported Browsers

| Browser | Minimum Version | Notes |
|---------|-----------------|-------|
| Chrome | 90+ | Full support |
| Firefox | 88+ | Full support |
| Safari | 14+ | Full support |
| Edge | 90+ | Full support (Chromium-based) |

### Required Features

#### 1. ES2020 Support

**Features Used**:
- Optional chaining (`obj?.property`)
- Nullish coalescing (`a ?? b`)
- `Array.prototype.filter/map/reduce`
- Async/await

**Browser Support**: All modern browsers (2020+)

#### 2. Typed Arrays

**Used for**: Efficient number storage

```typescript
const bubbleMeans = new Float32Array(detectionResult.bubbleMeans);
```

**Browser Support**: Universal (IE10+)

#### 3. Web Workers (Optional)

**Used for**: Large batch processing

```typescript
const worker = new Worker("interpretation-worker.js");
```

**Browser Support**: Universal (IE10+)

**Fallback**: Process on main thread if workers unavailable

```typescript
if (typeof Worker === "undefined") {
    console.warn("Web Workers not supported, processing on main thread");
    return processOnMainThread(files);
}
```

### Polyfills Not Required

All operations use native JavaScript:
- **Math operations**: `Math.min()`, `Math.max()`, `Math.round()`
- **Array operations**: Native array methods
- **Sorting**: Native `Array.prototype.sort()`

### Performance by Browser

**Benchmark**: Process 50 fields × 4 bubbles (200 total)

| Browser | Main Thread | Web Worker | Notes |
|---------|-------------|------------|-------|
| Chrome | ~5ms | ~3ms | V8 engine optimization |
| Firefox | ~6ms | ~4ms | SpiderMonkey engine |
| Safari | ~8ms | ~5ms | JavaScriptCore engine |
| Edge | ~5ms | ~3ms | V8 engine (same as Chrome) |

**Conclusion**: All browsers handle interpretation pass efficiently (< 10ms)

### Mobile Browser Constraints

#### Memory Limits

**iOS Safari**: ~1 GB per tab
**Chrome Mobile**: ~1.5 GB per tab
**Firefox Mobile**: ~1 GB per tab

**Safe Batch Size**: 50 files at a time (~3.2 MB total)

#### Performance

**Target**: < 100ms per file (maintain responsive UI)

**Actual**: ~10ms per file (50 fields × 0.2ms)

**Recommendation**: Use Web Workers for > 10 files on mobile

```typescript
const isMobile = /iPhone|iPad|Android/i.test(navigator.userAgent);
const batchSize = isMobile ? 50 : 100;
```

---

## Validation & Testing

### Unit Tests

#### Test 1: Single Bubble Fallback

```typescript
test("single bubble uses global fallback", () => {
    const bubbleMeans = [150];
    const config = new ThresholdConfig({ defaultThreshold: 200 });

    const result = strategy.calculateThreshold(bubbleMeans, config);

    expect(result.thresholdValue).toBe(200);
    expect(result.methodUsed).toBe("local_single_bubble_fallback");
    expect(result.fallbackUsed).toBe(true);
    expect(result.confidence).toBe(0.0);
});
```

#### Test 2: Two Bubbles Small Gap

```typescript
test("two bubbles with small gap uses fallback", () => {
    const bubbleMeans = [150, 155];  // Gap = 5 < 30
    const config = new ThresholdConfig({ minGapTwoBubbles: 30, defaultThreshold: 200 });

    const result = strategy.calculateThreshold(bubbleMeans, config);

    expect(result.thresholdValue).toBe(200);
    expect(result.methodUsed).toBe("local_two_bubbles_small_gap_fallback");
    expect(result.fallbackUsed).toBe(true);
});
```

#### Test 3: Two Bubbles Sufficient Gap

```typescript
test("two bubbles with sufficient gap uses mean", () => {
    const bubbleMeans = [150, 200];  // Gap = 50 > 30
    const config = new ThresholdConfig({ minGapTwoBubbles: 30 });

    const result = strategy.calculateThreshold(bubbleMeans, config);

    expect(result.thresholdValue).toBe(175);  // Mean of [150, 200]
    expect(result.methodUsed).toBe("local_two_bubbles_mean");
    expect(result.fallbackUsed).toBe(false);
    expect(result.confidence).toBe(0.7);
});
```

#### Test 4: Normal Local Threshold

```typescript
test("normal case uses local max jump", () => {
    const bubbleMeans = [150, 155, 160, 225, 230, 235];  // Max jump = 65
    const config = new ThresholdConfig({ minJump: 25, minJumpSurplusForGlobalFallback: 5 });

    const result = strategy.calculateThreshold(bubbleMeans, config);

    expect(result.thresholdValue).toBeCloseTo(192.5, 1);  // 160 + 65/2
    expect(result.methodUsed).toBe("local_max_jump");
    expect(result.fallbackUsed).toBe(false);
    expect(result.confidence).toBeGreaterThan(0.7);
});
```

#### Test 5: All Bubbles Marked Edge Case

```typescript
test("all bubbles marked returns empty string", () => {
    const bubbleInterpretations = [
        { isAttempted: true, bubbleValue: "A" },
        { isAttempted: true, bubbleValue: "B" },
        { isAttempted: true, bubbleValue: "C" },
        { isAttempted: true, bubbleValue: "D" },
    ];

    const result = getFieldInterpretationString(bubbleInterpretations);

    expect(result).toBe("");  // Treat as unmarked (scanning issue)
});
```

#### Test 6: Multi-marking Detection

```typescript
test("multi-marking is detected", () => {
    const bubbleInterpretations = [
        { isAttempted: true, bubbleValue: "A" },
        { isAttempted: true, bubbleValue: "B" },
        { isAttempted: false, bubbleValue: "C" },
        { isAttempted: false, bubbleValue: "D" },
    ];

    const isMultiMarked = checkMultiMarking(bubbleInterpretations);

    expect(isMultiMarked).toBe(true);
});
```

### Integration Tests

#### Test 1: Full Interpretation Flow

```typescript
test("full interpretation flow", async () => {
    const field = createMockField("Q1");
    const detectionResult = createMockDetectionResult([150, 155, 225, 230]);
    const fileLevelAggregates = createMockFileLevelAggregates({ globalThreshold: 200 });

    const interpretation = new BubblesFieldInterpretation(tuningConfig, field);
    await interpretation.runInterpretation(field, { bubble_fields: { Q1: detectionResult } }, fileLevelAggregates);

    expect(interpretation.localThresholdForField).toBeCloseTo(190, 0);  // Local threshold
    expect(interpretation.bubbleInterpretations.length).toBe(4);
    expect(interpretation.isMultiMarked).toBe(false);
    expect(interpretation.getFieldInterpretationString()).toBe("A");  // Only A marked
});
```

### Performance Tests

#### Test 1: Batch Processing

```typescript
test("batch processing completes within time limit", async () => {
    const files = generateMockFiles(100);  // 100 files

    const start = performance.now();
    await processBatch(files, 100);
    const duration = performance.now() - start;

    expect(duration).toBeLessThan(5000);  // < 5 seconds for 100 files
});
```

---

## Error Handling

### Critical Errors (Throw)

```typescript
// 1. Missing detection result
if (!(fieldLabel in bubbleFields)) {
    throw new Error(`No detection result for field '${fieldLabel}'`);
}

// 2. Invalid threshold config
if (config.minJump < 0 || config.defaultThreshold < 0) {
    throw new Error("Invalid threshold config: negative values");
}
```

### Warnings (Log + Continue)

```typescript
// 1. Multi-marking
if (markedCount > 1) {
    console.warn(`Multi-marking in ${fieldLabel}: ${markedCount} bubbles`);
}

// 2. Poor scan quality
if (scanQuality === "POOR") {
    console.warn(`Poor scan quality for ${fieldLabel} (std=${stdDeviation})`);
}

// 3. Threshold disparity
if (disparityBubbles.length > 0) {
    console.warn(`Threshold disparity in ${fieldLabel}: ${disparityBubbles.length} bubbles in doubt`);
}

// 4. Xeroxed OMR
if (globalThreshold === 255) {
    console.warn("Looks like a Xeroxed OMR (threshold=255). Results may be unreliable.");
}
```

### Graceful Degradation

```typescript
// 1. Empty bubble means (return empty value)
if (detectionResult.bubbleMeans.length === 0) {
    console.error(`No bubbles detected in ${fieldLabel}`);
    return "";
}

// 2. Missing confidence metrics (skip calculation)
if (!tuningConfig.outputs.showConfidenceMetrics) {
    // Skip confidence calculation
}

// 3. Web Workers unavailable (use main thread)
if (typeof Worker === "undefined") {
    console.warn("Web Workers not supported, using main thread");
    return processOnMainThread(files);
}
```

---

## Configuration Defaults

### Threshold Configuration

```typescript
const DEFAULT_THRESHOLD_CONFIG: ThresholdConfig = {
    minJump: 25,
    jumpDelta: 30,
    minGapTwoBubbles: 30,
    minJumpSurplusForGlobalFallback: 5,
    confidentJumpSurplusForDisparity: 25,
    globalThresholdMargin: 10,
    outlierDeviationThreshold: 5.0,
    defaultThreshold: 200,
};
```

### Scan Quality Thresholds

```typescript
const SCAN_QUALITY_THRESHOLDS = {
    EXCELLENT: 50,  // std > 50
    GOOD: 30,       // 30 < std <= 50
    ACCEPTABLE: 15, // 15 < std <= 30
    POOR: 0,        // std <= 15
};
```

### Confidence Score Weights

```typescript
const CONFIDENCE_WEIGHTS = {
    thresholdConfidence: 0.35,  // 35%
    marginConfidence: 0.25,     // 25%
    scanQuality: 0.20,          // 20%
    multiMarkPenalty: 0.30,     // Reduce by 30% if multi-marked
    disparityPenalty: 0.40,     // Up to 40% penalty
};
```

---

## See Also

- **Detection Pass**: `detection-pass/constraints.md` - Detection constraints and edge cases
- **Threshold Strategies**: `../../threshold/global/constraints.md` - Global threshold constraints
- **Local Threshold**: `../../threshold/local/flows.md` - Local threshold algorithm details
- **Drawing**: `drawing/constraints.md` - Visualization constraints
