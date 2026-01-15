/**
 * Bubble field detection using typed models.
 *
 * Port of Python's src/processors/detection/bubbles_threshold/detection.py
 * Refactored to use BubbleFieldDetectionResult instead of nested dictionaries.
 */

import * as cv from '@techstark/opencv-js';
import { FieldDetection } from '../base';
import {
  BubbleFieldDetectionResult,
  BubbleMeanValue,
  type BubbleLocation,
} from '../models';

/**
 * Detects bubble values and returns strongly-typed result.
 *
 * Replaces dictionary-based aggregates with BubbleFieldDetectionResult.
 * Extends FieldDetection base class following Python architecture.
 */
export class BubblesFieldDetection extends FieldDetection {
  public result: BubbleFieldDetectionResult | null = null;
  public fieldBubbleMeans: BubbleMeanValue[] = []; // Backward compatibility

  constructor(
    private fieldId: string,
    private fieldLabel: string,
    private bubbles: BubbleLocation[],
    grayImage: cv.Mat,
    coloredImage?: cv.Mat
  ) {
    // Call parent constructor which will trigger runDetection
    super({ id: fieldId, field_label: fieldLabel, bubbles }, grayImage, coloredImage);
  }

  /**
   * Run detection and create typed result.
   * Implements abstract method from FieldDetection.
   */
  protected runDetection(_field: any, grayImage: any, _coloredImage: any): void {
    const bubbleMeans: BubbleMeanValue[] = [];

    for (const bubble of this.bubbles) {
      // TODO: cross/check mark detection support (#167)
      const bubbleMeanValue = BubblesFieldDetection.readBubbleMeanValue(
        bubble,
        grayImage
      );
      bubbleMeans.push(bubbleMeanValue);
    }

    // Create strongly-typed result
    // Properties like stdDeviation and scanQuality are auto-calculated
    this.result = new BubbleFieldDetectionResult(
      this.fieldId,
      this.fieldLabel,
      bubbleMeans
    );

    // Keep backward compatibility
    this.fieldBubbleMeans = bubbleMeans;
  }

  /**
   * Read mean intensity value for a single bubble.
   *
   * @param bubble - Bubble location and dimensions
   * @param grayImage - Grayscale image
   * @returns BubbleMeanValue with mean intensity
   */
  static readBubbleMeanValue(
    bubble: BubbleLocation,
    grayImage: cv.Mat
  ): BubbleMeanValue {
    const { x, y, width, height } = bubble;

    // Extract the bubble region
    const rect = new cv.Rect(x, y, width, height);
    const roi = grayImage.roi(rect);

    // Calculate mean value
    const mean = cv.mean(roi);
    const meanValue = mean[0];

    // Clean up
    roi.delete();

    return new BubbleMeanValue(meanValue, bubble, [x, y]);
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

