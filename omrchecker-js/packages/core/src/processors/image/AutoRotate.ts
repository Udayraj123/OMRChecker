/**
 * Migrated from Python: src/processors/image/AutoRotate.py
 * Agent: Oz
 * Phase: Filter Processors
 *
 * Automatic rotation detection and correction using template matching.
 */

import cv from '@techstark/opencv-js';
import { ProcessingContext, Processor } from '../base';

export interface AutoRotateOptions {
  referenceImage: cv.Mat;
  markerDimensions?: [number, number];
  threshold?: {
    value: number;
    passthrough: boolean;
  };
}

/**
 * Automatically detects and corrects document orientation.
 * 
 * Uses template matching to compare the input image against a reference marker
 * in all four rotations (0°, 90°, 180°, 270°) and selects the best match.
 * 
 * Common use cases:
 * - Correct images scanned upside-down or sideways
 * - Handle mobile camera captures at any angle
 * - Pre-processing before alignment or detection
 * 
 * Configuration:
 * - referenceImage: Reference marker to match against (Mat)
 * - markerDimensions: Optional resize dimensions for reference [width, height]
 * - threshold: Optional minimum match score requirement
 *   - value: Minimum correlation coefficient (0-1)
 *   - passthrough: If true, continue even if below threshold (with warning)
 */
export class AutoRotate extends Processor {
  private referenceImage: cv.Mat;
  private resizedReference: cv.Mat;
  private threshold?: {
    value: number;
    passthrough: boolean;
  };

  constructor(options: AutoRotateOptions) {
    super();
    this.referenceImage = options.referenceImage;
    this.threshold = options.threshold;

    // Resize reference if marker dimensions specified
    if (options.markerDimensions) {
      const [width, height] = options.markerDimensions;
      this.resizedReference = new cv.Mat();
      cv.resize(
        this.referenceImage,
        this.resizedReference,
        new cv.Size(width, height),
        0,
        0,
        cv.INTER_LINEAR
      );
    } else {
      this.resizedReference = this.referenceImage;
    }
  }

  getName(): string {
    return 'AutoRotate';
  }

  /**
   * Rotate image while maintaining original shape.
   */
  private rotateKeepingShape(image: cv.Mat, rotationCode: number): cv.Mat {
    const originalHeight = image.rows;
    const originalWidth = image.cols;

    // Rotate image
    const rotated = new cv.Mat();
    cv.rotate(image, rotated, rotationCode);

    // If dimensions changed, resize back to original shape
    if (rotated.rows !== originalHeight || rotated.cols !== originalWidth) {
      const resized = new cv.Mat();
      cv.resize(
        rotated,
        resized,
        new cv.Size(originalWidth, originalHeight),
        0,
        0,
        cv.INTER_LINEAR
      );
      rotated.delete();
      return resized;
    }

    return rotated;
  }

  process(context: ProcessingContext): ProcessingContext {
    const { grayImage, coloredImage } = context;

    // Try all 4 rotations
    const rotations = [
      { code: null, label: 'No Rotation' },
      { code: cv.ROTATE_90_CLOCKWISE, label: '90°' },
      { code: cv.ROTATE_180, label: '180°' },
      { code: cv.ROTATE_90_COUNTERCLOCKWISE, label: '270°' },
    ];

    let bestVal = -1;
    let bestRotation: number | null = null;

    // Template matching for each rotation
    for (const { code, label } of rotations) {
      let rotatedImg: cv.Mat;
      let needsCleanup = false;

      if (code === null) {
        rotatedImg = grayImage;
      } else {
        rotatedImg = this.rotateKeepingShape(grayImage, code);
        needsCleanup = true;
      }

      // Perform template matching
      const result = new cv.Mat();
      cv.matchTemplate(
        rotatedImg,
        this.resizedReference,
        result,
        cv.TM_CCOEFF_NORMED
      );

      const minMax = cv.minMaxLoc(result, new cv.Mat());
      const maxVal = minMax.maxVal;

      console.debug(
        `AutoRotate template matching: ${label} = ${maxVal.toFixed(4)}`
      );

      result.delete();
      if (needsCleanup) {
        rotatedImg.delete();
      }

      if (maxVal > bestVal) {
        bestVal = maxVal;
        bestRotation = code;
      }
    }

    // Check threshold if specified
    if (this.threshold && this.threshold.value > bestVal) {
      const message = `AutoRotate score ${bestVal.toFixed(4)} below threshold ${this.threshold.value}`;
      
      if (this.threshold.passthrough) {
        console.warn(`${message}. Continuing due to passthrough flag.`);
      } else {
        throw new Error(
          `${message}. Adjust threshold or check reference marker and input image.`
        );
      }
    }

    console.info(
      `AutoRotate applied: rotation=${bestRotation === null ? 'none' : rotations.find(r => r.code === bestRotation)?.label}, score=${bestVal.toFixed(4)}`
    );

    // Apply best rotation if needed
    if (bestRotation !== null) {
      const rotatedGray = this.rotateKeepingShape(grayImage, bestRotation);
      grayImage.delete();
      context.grayImage = rotatedGray;

      // Rotate colored image if present
      if (coloredImage && !coloredImage.empty()) {
        const rotatedColor = this.rotateKeepingShape(
          coloredImage,
          bestRotation
        );
        coloredImage.delete();
        context.coloredImage = rotatedColor;
      }
    }

    return context;
  }

  /**
   * Clean up allocated resources.
   */
  cleanup(): void {
    if (this.resizedReference !== this.referenceImage) {
      this.resizedReference.delete();
    }
  }
}
