/**
 * Migrated from Python: src/processors/image/MedianBlur.py
 * Agent: Oz
 * Phase: Filter Processors
 *
 * Median blur filter for noise reduction while preserving edges.
 */

import cv from '@techstark/opencv-js';
import { ProcessingContext, Processor } from '../base';

export interface MedianBlurOptions {
  kSize?: number;
}

/**
 * Applies median blur to reduce salt-and-pepper noise while preserving edges.
 * 
 * Median blur is particularly effective for:
 * - Removing salt-and-pepper noise
 * - Preserving sharp edges (better than Gaussian blur)
 * - Pre-processing before bubble detection
 * 
 * Configuration:
 * - kSize: Aperture linear size (must be odd and greater than 1). Default: 5
 */
export class MedianBlur extends Processor {
  private kSize: number;

  constructor(options: MedianBlurOptions = {}) {
    super();
    this.kSize = options.kSize ?? 5;
  }

  getName(): string {
    return 'MedianBlur';
  }

  process(context: ProcessingContext): ProcessingContext {
    const { grayImage } = context;

    // Apply median blur
    const blurred = new cv.Mat();
    try {
      cv.medianBlur(grayImage, blurred, this.kSize);

      // Replace gray image with blurred version
      grayImage.delete();
      context.grayImage = blurred;
    } catch (error) {
      // Clean up on error
      blurred.delete();
      throw error;
    }

    return context;
  }
}
