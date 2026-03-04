/**
 * BubbleReader — TypeScript port of Python bubble detection & interpretation.
 *
 * Migrated from:
 *   src/processors/detection/bubbles_threshold/detection.py   (pixel mean per scan box)
 *   src/processors/detection/threshold/local_threshold.py     (LocalThresholdStrategy)
 *   src/processors/detection/bubbles_threshold/interpretation.py (marked bubble determination)
 *
 * Scope: BUBBLES_THRESHOLD field detection only.
 * Excluded: confidence metrics, drawing, multi-mark logging, alignment.
 *
 * bd issue: omr-11b
 */

import cv from '@techstark/opencv-js';
import { Template } from '../template/Template';

// ── ThresholdConfig ───────────────────────────────────────────────────────────

/**
 * Configuration for the local threshold strategy.
 *
 * Ported from Python: src/processors/detection/threshold/threshold_result.py::ThresholdConfig
 */
export interface ThresholdConfig {
  /** Minimum jump size between sorted bubble means to be considered significant. Default: 30. */
  minJump: number;
  /** Minimum gap between 2-bubble means to use midpoint threshold. Default: 20. */
  minGapTwoBubbles: number;
  /**
   * Extra surplus above minJump needed for a high-confidence local threshold
   * (prevents global fallback when the jump is marginal). Default: 10.
   */
  minJumpSurplusForGlobalFallback: number;
  /** Global fallback threshold value used when local confidence is low. Default: 127.5. */
  globalFallbackThreshold: number;
}

// ── BubbleReaderConfig ────────────────────────────────────────────────────────

/**
 * User-facing configuration for BubbleReader (all fields optional, with defaults).
 */
export interface BubbleReaderConfig {
  minJump?: number;
  minGapTwoBubbles?: number;
  minJumpSurplusForGlobalFallback?: number;
  globalFallbackThreshold?: number;
}

// ── OMRResponse ───────────────────────────────────────────────────────────────

/**
 * Map from field label to detected bubble value(s).
 *
 * Each value is:
 *   - The marked bubble value (e.g. "A", "3") for single-mark MCQ/INT fields
 *   - Concatenated values (e.g. "AB") for multi-answer fields
 *   - The field's emptyValue (e.g. "") when no bubble is marked or all are marked
 */
export type OMRResponse = Record<string, string>;

// ── BubbleReader ──────────────────────────────────────────────────────────────

/**
 * Reads bubble pixel means from a grayscale image and determines which bubbles
 * are marked using a local threshold strategy (with global fallback).
 *
 * Usage:
 * ```ts
 * const reader = new BubbleReader();
 * const response = reader.readBubbles(grayMat, template);
 * // response: { Medium: 'E', roll1: '5', q1: 'B', ... }
 * ```
 */
export class BubbleReader {
  private readonly config: Required<ThresholdConfig>;

  constructor(userConfig: BubbleReaderConfig = {}) {
    this.config = {
      minJump: userConfig.minJump ?? 30,
      minGapTwoBubbles: userConfig.minGapTwoBubbles ?? 20,
      minJumpSurplusForGlobalFallback: userConfig.minJumpSurplusForGlobalFallback ?? 10,
      globalFallbackThreshold: userConfig.globalFallbackThreshold ?? 127.5,
    };
  }

  /**
   * Process all fields in a template against a pre-processed grayscale image.
   *
   * Ported from Python:
   *   BubblesFieldDetection.run_detection  (pixel means)
   *   LocalThresholdStrategy.calculate_threshold
   *   BubblesFieldInterpretation.get_field_interpretation_string
   *
   * @param grayImage - Pre-processed grayscale cv.Mat (must not be deleted during call)
   * @param template  - Parsed Template (provides allFields with scanBoxes)
   * @returns OMRResponse mapping each fieldLabel → detected value string
   */
  readBubbles(grayImage: cv.Mat, template: Template): OMRResponse {
    const response: OMRResponse = {};
    const { globalFallbackThreshold } = this.config;

    for (const field of template.allFields) {
      const bubbleMeans: number[] = [];

      // ── Step 1: Read mean intensity per scan box ────────────────────────────
      // Port of BubblesFieldDetection.read_bubble_mean_value
      for (const scanBox of field.scanBoxes) {
        const [w, h] = scanBox.bubbleDimensions;

        // Use shifted position (accounts for any alignment shifts applied earlier)
        const [shiftedX, shiftedY] = scanBox.getShiftedPosition();
        const x = Math.round(shiftedX);
        const y = Math.round(shiftedY);

        // Clamp to image bounds to avoid out-of-bounds cv.Rect
        const safeX = Math.max(0, Math.min(x, grayImage.cols - 1));
        const safeY = Math.max(0, Math.min(y, grayImage.rows - 1));
        const safeW = Math.min(w, grayImage.cols - safeX);
        const safeH = Math.min(h, grayImage.rows - safeY);

        if (safeW <= 0 || safeH <= 0) {
          // Scan box entirely out of bounds — treat as fully white (unmarked)
          bubbleMeans.push(255);
          continue;
        }

        const roi = grayImage.roi(new cv.Rect(safeX, safeY, safeW, safeH));
        try {
          // cv.mean returns [mean_blue_or_gray, mean_green, mean_red, mean_alpha]
          const meanVal = cv.mean(roi)[0];
          bubbleMeans.push(meanVal);
        } finally {
          roi.delete();
        }
      }

      // ── Step 2: Calculate local threshold ──────────────────────────────────
      // Port of LocalThresholdStrategy.calculate_threshold
      const threshold = this.localThreshold(bubbleMeans, globalFallbackThreshold);

      // ── Step 3: Interpret marked bubbles ────────────────────────────────────
      // Port of BubblesFieldInterpretation.get_field_interpretation_string
      const markedBubbles = field.scanBoxes.filter((_sb, i) => bubbleMeans[i] < threshold);

      let fieldValue: string;
      if (markedBubbles.length === 0 || markedBubbles.length === field.scanBoxes.length) {
        // No marks or all marks → treat as empty (scanning artifact or no response)
        fieldValue = field.emptyValue;
      } else {
        // Concatenate bubble values of all marked scan boxes
        fieldValue = markedBubbles.map(sb => sb.bubbleValue).join('');
      }

      response[field.fieldLabel] = fieldValue;
    }

    return response;
  }

  /**
   * Calculate the local threshold for a set of bubble mean values.
   *
   * Falls back to the global threshold when local confidence is too low.
   *
   * Ported from Python: LocalThresholdStrategy.calculate_threshold
   *
   * Algorithm:
   *   - 0 or 1 bubbles  → globalFallback
   *   - 2 bubbles       → midpoint if gap ≥ minGapTwoBubbles, else globalFallback
   *   - 3+ bubbles      → midpoint of largest inter-bubble jump;
   *                       use globalFallback when maxJump < minJump + minJumpSurplusForGlobalFallback
   *
   * @param bubbleMeans    - Mean pixel intensity per bubble (0–255)
   * @param globalFallback - Fallback threshold to use when local confidence is low
   * @returns Threshold value (bubble mean < threshold → bubble is marked)
   */
  private localThreshold(bubbleMeans: number[], globalFallback: number): number {
    if (bubbleMeans.length < 2) {
      return globalFallback;
    }

    const sorted = [...bubbleMeans].sort((a, b) => a - b);

    // Special case: exactly 2 bubbles
    if (sorted.length === 2) {
      const gap = sorted[1] - sorted[0];
      if (gap < this.config.minGapTwoBubbles) {
        return globalFallback;
      }
      // Midpoint between the two values
      return (sorted[0] + sorted[1]) / 2;
    }

    // 3+ bubbles: find the largest jump across sorted pairs
    // Python: for i in range(1, len-1): jump = sorted[i+1] - sorted[i-1]
    let maxJump = 0;
    let localThreshold = globalFallback;

    for (let i = 1; i < sorted.length - 1; i++) {
      const jump = sorted[i + 1] - sorted[i - 1];
      if (jump > maxJump) {
        maxJump = jump;
        localThreshold = sorted[i - 1] + jump / 2;
      }
    }

    // Use global fallback if the jump is not large enough to be confident
    const confidentJump = this.config.minJump + this.config.minJumpSurplusForGlobalFallback;
    if (maxJump < confidentJump) {
      return globalFallback;
    }

    return localThreshold;
  }
}
