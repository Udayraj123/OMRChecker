/**
 * Simple Bubble Detector using threshold-based detection.
 *
 * TypeScript port - Simplified version of bubble detection logic
 * Uses threshold strategies already ported from Python.
 *
 * This is an incremental approach - starting simple and building up
 * to the full ReadOMRProcessor complexity.
 */

import * as cv from '@techstark/opencv-js';
import { Logger } from '../../utils/logger';
import { GlobalThreshold, ThresholdConfig } from '../threshold/GlobalThreshold';

const logger = new Logger('SimpleBubbleDetector');

/**
 * Represents a bubble/scan box location and its detection result
 */
export interface BubbleLocation {
  /** X coordinate of bubble center */
  x: number;
  /** Y coordinate of bubble center */
  y: number;
  /** Width of scan region */
  width: number;
  /** Height of scan region */
  height: number;
  /** Label/value for this bubble (e.g., "A", "B", "1", "2") */
  label: string;
}

/**
 * Result of bubble detection
 */
export interface BubbleDetectionResult {
  /** Label of the bubble */
  label: string;
  /** Mean pixel intensity (0-255, lower = darker = more filled) */
  meanValue: number;
  /** Whether this bubble is marked (below threshold) */
  isMarked: boolean;
  /** Confidence score (0-1) */
  confidence: number;
}

/**
 * Result for a field (group of bubbles, like a multiple choice question)
 */
export interface FieldDetectionResult {
  /** Field identifier */
  fieldId: string;
  /** All bubble detection results */
  bubbles: BubbleDetectionResult[];
  /** Detected answer (label of marked bubble) */
  detectedAnswer: string | null;
  /** Threshold value used */
  threshold: number;
  /** Whether multiple bubbles were marked */
  isMultiMarked: boolean;
}

/**
 * Simplified bubble detector using threshold-based detection.
 *
 * This detector:
 * 1. Extracts bubble regions from the image
 * 2. Calculates mean pixel intensity for each bubble
 * 3. Determines threshold using GlobalThreshold strategy
 * 4. Marks bubbles below threshold as filled
 * 5. Returns detection results
 */
export class SimpleBubbleDetector {
  private thresholdStrategy: GlobalThreshold;
  private config: ThresholdConfig;

  constructor(config?: Partial<ThresholdConfig>) {
    this.config = {
      defaultThreshold: 200,
      minJump: 30,
      ...config,
    };
    this.thresholdStrategy = new GlobalThreshold();
  }

  /**
   * Detect bubbles in a field (e.g., one multiple choice question).
   *
   * @param image - Grayscale image
   * @param bubbles - Array of bubble locations to check
   * @param fieldId - Identifier for this field
   * @returns Detection result for the field
   */
  detectField(
    image: cv.Mat,
    bubbles: BubbleLocation[],
    fieldId: string
  ): FieldDetectionResult {
    logger.debug(`Detecting field: ${fieldId} with ${bubbles.length} bubbles`);

    // Step 1: Calculate mean intensity for each bubble
    const bubbleMeans: number[] = [];
    const bubbleResults: BubbleDetectionResult[] = [];

    for (const bubble of bubbles) {
      const meanValue = this.extractBubbleMean(image, bubble);
      bubbleMeans.push(meanValue);

      bubbleResults.push({
        label: bubble.label,
        meanValue,
        isMarked: false, // Will be determined after threshold calculation
        confidence: 0, // Will be calculated after threshold
      });
    }

    // Step 2: Calculate threshold for this field
    const thresholdResult = this.thresholdStrategy.calculateThreshold(
      bubbleMeans,
      this.config
    );

    const threshold = thresholdResult.thresholdValue;
    logger.debug(
      `Field ${fieldId}: threshold=${threshold.toFixed(1)}, confidence=${thresholdResult.confidence.toFixed(2)}`
    );

    // Step 3: Mark bubbles below threshold
    const markedBubbles: BubbleDetectionResult[] = [];

    for (let i = 0; i < bubbleResults.length; i++) {
      const result = bubbleResults[i];
      result.isMarked = result.meanValue < threshold;

      // Calculate confidence based on distance from threshold
      const distanceFromThreshold = Math.abs(result.meanValue - threshold);
      result.confidence = Math.min(1.0, distanceFromThreshold / 50); // Normalize to 0-1

      if (result.isMarked) {
        markedBubbles.push(result);
      }
    }

    // Step 4: Determine detected answer
    let detectedAnswer: string | null = null;
    const isMultiMarked = markedBubbles.length > 1;

    if (markedBubbles.length === 1) {
      detectedAnswer = markedBubbles[0].label;
    } else if (markedBubbles.length > 1) {
      // Multi-marked - take the darkest one
      const darkest = markedBubbles.reduce((min, curr) =>
        curr.meanValue < min.meanValue ? curr : min
      );
      detectedAnswer = darkest.label;
      logger.warn(
        `Field ${fieldId}: Multi-marked detected (${markedBubbles.length} bubbles marked)`
      );
    }

    return {
      fieldId,
      bubbles: bubbleResults,
      detectedAnswer,
      threshold,
      isMultiMarked,
    };
  }

  /**
   * Extract mean pixel intensity from a bubble region.
   *
   * @param image - Grayscale image
   * @param bubble - Bubble location
   * @returns Mean pixel intensity (0-255)
   */
  private extractBubbleMean(image: cv.Mat, bubble: BubbleLocation): number {
    try {
      // Ensure coordinates are within image bounds
      const x = Math.max(0, Math.floor(bubble.x));
      const y = Math.max(0, Math.floor(bubble.y));
      const width = Math.min(bubble.width, image.cols - x);
      const height = Math.min(bubble.height, image.rows - y);

      if (width <= 0 || height <= 0) {
        logger.warn(
          `Invalid bubble dimensions at (${x}, ${y}): ${width}x${height}`
        );
        return 255; // Return white (unmarked)
      }

      // Extract ROI (Region of Interest)
      const roi = image.roi(new cv.Rect(x, y, width, height));

      // Calculate mean
      const mean = cv.mean(roi);
      const meanValue = mean[0]; // First channel (grayscale)

      roi.delete(); // Clean up

      return meanValue;
    } catch (error) {
      logger.error(`Error extracting bubble mean: ${error}`);
      return 255; // Return white (unmarked) on error
    }
  }

  /**
   * Detect multiple fields at once (e.g., entire test).
   *
   * @param image - Grayscale image
   * @param fields - Map of field IDs to bubble locations
   * @returns Map of field IDs to detection results
   */
  detectMultipleFields(
    image: cv.Mat,
    fields: Map<string, BubbleLocation[]>
  ): Map<string, FieldDetectionResult> {
    const results = new Map<string, FieldDetectionResult>();

    for (const [fieldId, bubbles] of fields.entries()) {
      const result = this.detectField(image, bubbles, fieldId);
      results.set(fieldId, result);
    }

    return results;
  }

  /**
   * Get summary statistics for detection results.
   *
   * @param results - Detection results
   * @returns Summary statistics
   */
  getDetectionStats(results: Map<string, FieldDetectionResult>): {
    totalFields: number;
    answeredFields: number;
    multiMarkedFields: number;
    unansweredFields: number;
    avgConfidence: number;
  } {
    let answeredFields = 0;
    let multiMarkedFields = 0;
    let totalConfidence = 0;
    let confidenceCount = 0;

    for (const result of results.values()) {
      if (result.detectedAnswer) {
        answeredFields++;
      }
      if (result.isMultiMarked) {
        multiMarkedFields++;
      }

      // Calculate average confidence of marked bubbles
      for (const bubble of result.bubbles) {
        if (bubble.isMarked) {
          totalConfidence += bubble.confidence;
          confidenceCount++;
        }
      }
    }

    return {
      totalFields: results.size,
      answeredFields,
      multiMarkedFields,
      unansweredFields: results.size - answeredFields,
      avgConfidence: confidenceCount > 0 ? totalConfidence / confidenceCount : 0,
    };
  }
}

