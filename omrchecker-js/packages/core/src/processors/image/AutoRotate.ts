/**
 * AutoRotate processor - Automatic image orientation detection.
 *
 * TypeScript port of src/processors/image/AutoRotate.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import * as cv from '@techstark/opencv-js';
import { Processor, ProcessingContext } from '../base';
import { Logger } from '../../utils/logger';

const logger = new Logger('AutoRotate');

/**
 * Interface for AutoRotate options
 */
export interface AutoRotateOptions {
  /**
   * Path to reference image for template matching
   */
  referenceImagePath: string;

  /**
   * Optional marker dimensions for resizing reference
   */
  markerDimensions?: { width: number; height: number };

  /**
   * Threshold configuration for rotation matching
   */
  threshold?: {
    value: number;
    passthrough: boolean;
  };
}

/**
 * AutoRotate processor - Detects and corrects image orientation.
 *
 * Uses template matching to find the best rotation (0°, 90°, 180°, 270°)
 * that matches a reference image.
 */
export class AutoRotate extends Processor {
  private referenceImage: cv.Mat;
  private resizedReference: cv.Mat;
  private options: AutoRotateOptions;

  constructor(options: AutoRotateOptions) {
    super();
    this.options = options;

    // Reference image loading would happen here
    // For now, we'll assume it's provided or loaded externally
    // In a real implementation, you'd use fetch/FileReader to load the image
    this.referenceImage = new cv.Mat(); // Placeholder
    this.resizedReference = this.referenceImage;
  }

  getName(): string {
    return 'AutoRotate';
  }

  /**
   * Set the reference image for template matching
   */
  setReferenceImage(image: cv.Mat): void {
    this.referenceImage = image;

    if (this.options.markerDimensions) {
      const { width, height } = this.options.markerDimensions;
      this.resizedReference = new cv.Mat();
      cv.resize(
        this.referenceImage,
        this.resizedReference,
        new cv.Size(width, height),
        0,
        0,
        cv.INTER_AREA
      );
    } else {
      this.resizedReference = this.referenceImage;
    }
  }

  process(context: ProcessingContext): ProcessingContext {
    try {
      let bestVal = -1;
      let bestRotation: cv.RotateFlags | null = null;

      // Try all 4 rotations
      const rotations: (cv.RotateFlags | null)[] = [
        null, // No rotation
        cv.ROTATE_90_CLOCKWISE,
        cv.ROTATE_180,
        cv.ROTATE_90_COUNTERCLOCKWISE,
      ];

      for (const rotation of rotations) {
        let rotatedImg: cv.Mat;

        if (rotation === null) {
          rotatedImg = context.grayImage;
        } else {
          rotatedImg = new cv.Mat();
          cv.rotate(context.grayImage, rotatedImg, rotation);
        }

        // Template matching
        const result = new cv.Mat();
        const mask = new cv.Mat();

        cv.matchTemplate(rotatedImg, this.resizedReference, result, cv.TM_CCOEFF_NORMED, mask);

        const minMax = cv.minMaxLoc(result);
        const maxVal = minMax.maxVal;

        // Cleanup
        result.delete();
        mask.delete();
        if (rotation !== null) {
          rotatedImg.delete();
        }

        if (maxVal > bestVal) {
          bestVal = maxVal;
          bestRotation = rotation;
        }
      }

      // Check threshold if provided
      if (this.options.threshold && this.options.threshold.value > bestVal) {
        if (this.options.threshold.passthrough) {
          logger.warn(
            'AutoRotate score below threshold. Continuing due to passthrough flag.'
          );
        } else {
          logger.error('AutoRotate score below threshold.');
          throw new Error(
            `AutoRotate score (${bestVal.toFixed(3)}) below threshold (${this.options.threshold.value})`
          );
        }
      }

      logger.info(
        `AutoRotate applied with rotation ${bestRotation === null ? 'none' : bestRotation} (score: ${bestVal.toFixed(3)})`
      );

      // Apply best rotation if needed
      if (bestRotation !== null) {
        const rotatedGray = new cv.Mat();
        const rotatedColor = new cv.Mat();

        cv.rotate(context.grayImage, rotatedGray, bestRotation);
        cv.rotate(context.coloredImage, rotatedColor, bestRotation);

        // Replace images
        context.grayImage.delete();
        context.coloredImage.delete();
        context.grayImage = rotatedGray;
        context.coloredImage = rotatedColor;
      }

      return context;
    } catch (error) {
      logger.error(`Error in AutoRotate: ${error}`);
      throw error;
    }
  }
}

