/**
 * Bubble field detection using typed models.
 *
 * Port of Python's src/processors/detection/bubbles_threshold/detection.py
 * Refactored to use BubbleFieldDetectionResult instead of nested dictionaries.
 */

import cv from '../../../utils/opencv';
import { FieldDetection } from '../base';
import {
  BubbleFieldDetectionResult,
  BubbleMeanValue,
} from '../models';
import type { Field } from '../../layout/field/base';
import { BubblesScanBox } from '../../layout/field/bubbleField';

/**
 * Detects bubble values and returns strongly-typed result.
 *
 * Replaces dictionary-based aggregates with BubbleFieldDetectionResult.
 * Extends FieldDetection base class following Python architecture.
 */
export class BubblesFieldDetection extends FieldDetection {
  // Result is set by runDetection() which is called from super()
  // Declared without initializer to avoid overwriting the value set by super()
  declare public result: BubbleFieldDetectionResult | null;

  constructor(
    field: Field,
    grayImage: cv.Mat,
    coloredImage?: cv.Mat
  ) {
    super(field, grayImage, coloredImage);
  }

  /**
   * Run detection and create typed result.
   * Implements abstract method from FieldDetection.
   */
  protected runDetection(field: Field, grayImage: cv.Mat, _coloredImage: cv.Mat): void {
    const bubbleMeans: BubbleMeanValue[] = [];

    // field.scanBoxes are BubblesScanBox instances
    for (const unitBubble of field.scanBoxes) {
      // TODO: cross/check mark detection support (#167)
      const bubbleMeanValue = BubblesFieldDetection.readBubbleMeanValue(
        unitBubble as BubblesScanBox,
        grayImage
      );
      bubbleMeans.push(bubbleMeanValue);
    }

    // Create strongly-typed result
    // Properties like stdDeviation and scanQuality are auto-calculated
    this.result = new BubbleFieldDetectionResult(
      field.id,
      field.fieldLabel,
      bubbleMeans
    );
  }

  /**
   * Read mean intensity value for a single bubble.
   *
   * @param unitBubble - Bubble scan box (BubblesScanBox instance)
   * @param grayImage - Grayscale image
   * @returns BubbleMeanValue with mean intensity
   */
  static readBubbleMeanValue(
    unitBubble: BubblesScanBox,
    grayImage: cv.Mat
  ): BubbleMeanValue {
    const [boxW, boxH] = unitBubble.bubbleDimensions;

    const [x, y] = unitBubble.getShiftedPosition();

    // Extract the bubble region
    const rect = new cv.Rect(x, y, boxW, boxH);
    const roi = grayImage.roi(rect);

    const mean = cv.mean(roi);
    const meanValue = mean[0];

    // Clean up
    roi.delete();

    // Note: unitBubble is BubblesScanBox, but BubbleMeanValue expects BubbleLocation interface
    // We'll pass the BubblesScanBox and it will be compatible
    return new BubbleMeanValue(
      meanValue,
      {
        x,
        y,
        width: boxW,
        height: boxH,
        label: unitBubble.bubbleValue,
      },
      [x, y]
    );
  }

  /**
   * Get the detection result.
   */
  getResult(): BubbleFieldDetectionResult {
    if (!this.result) {
      throw new Error('Detection not run');
    }
    return this.result;
  }
}

