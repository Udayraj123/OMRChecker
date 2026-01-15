/**
 * DEMO REFACTORING PLAN - 1:1 Python Mapping
 *
 * ## Problem
 * The demo was built around SimpleBubbleDetector which returned a structure with:
 * - result.detectedAnswer
 * - result.bubbles (with isMarked, confidence)
 * - result.threshold
 * - result.isMultiMarked
 *
 * BubbleFieldDetectionResult (proper Python mapping) has:
 * - bubbleMeans: BubbleMeanValue[] (mean intensity values)
 * - stdDeviation, jumps, scanQuality (auto-calculated properties)
 * - NO detectedAnswer (calculated by OMRProcessor + threshold strategy)
 * - NO threshold (applied by threshold strategy separately)
 * - NO isMultiMarked (determined after thresholding)
 *
 * ## Solution
 * The demo should work directly with OMRSheetResult which has all the needed info:
 * - responses: Record<string, string | null> - detected answers
 * - multiMarkedFields: string[] - which fields are multi-marked
 * - emptyFields: string[] - which fields are empty
 * - fieldResults: Record<string, BubbleFieldDetectionResult> - raw detection data
 * - statistics: pre-calculated stats
 * - score/maxScore: evaluation results
 *
 * ## Changes Needed
 * 1. ✅ Remove SimpleBubbleDetector
 * 2. ✅ Update OMRProcessor to use BubblesFieldDetection + threshold strategy
 * 3. ⏳ Update demo displayResults() to use OMRSheetResult structure
 * 4. ⏳ Update demo displayBatchResults() similarly
 * 5. ⏳ Update demo visualizeResults() to show scan quality instead of confidence
 * 6. ⏳ Remove all references to result.detectedAnswer, result.bubbles, etc.
 *
 * ## Display Mapping
 * Old SimpleBubbleDetector → New Proper Architecture
 * - result.detectedAnswer → sheetResult.responses[fieldId]
 * - result.isMultiMarked → sheetResult.multiMarkedFields.includes(fieldId)
 * - result.bubbles.find(b => b.isMarked).confidence → fieldResult.stdDeviation (scan quality)
 * - result.threshold → N/A (internal to threshold strategy)
 *
 * ## Files to Update
 * - omrchecker-js/packages/demo/src/main.ts - all display functions
 * - No HTML changes needed (structure same, just data source changes)
 */

