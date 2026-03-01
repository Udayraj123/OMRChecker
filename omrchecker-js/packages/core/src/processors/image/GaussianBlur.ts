/**
 * Migrated from Python: src/processors/image/GaussianBlur.py
 * Agent: Oz
 * Phase: Filter Processors
 *
 * Gaussian blur filter for image smoothing and noise reduction.
 */

import cv from '@techstark/opencv-js';
import { ProcessingContext, Processor } from '../base';

export interface GaussianBlurOptions {
  kSize?: [number, number];
  sigmaX?: number;
}

/**
 * Applies Gaussian blur to smooth images and reduce noise.
 * 
 * Common use cases:
 * - Reduce image noise before thresholding
 * - Smooth edges for better contour detection
 * - Pre-processing step for template matching
 * 
 * Configuration:
 * - kSize: Gaussian kernel size (must be positive and odd). Default: [3, 3]
 * - sigmaX: Gaussian kernel standard deviation in X direction. Default: 0 (auto-calculated)
 */
export class GaussianBlur extends Processor {
  private kSize: cv.Size;
  private sigmaX: number;

  constructor(options: GaussianBlurOptions = {}) {
    super();
    const kSizeArray = options.kSize || [3, 3];
    this.kSize = new cv.Size(kSizeArray[0], kSizeArray[1]);
    this.sigmaX = options.sigmaX ?? 0;
  }

  getName(): string {
    return 'GaussianBlur';
  }

  process(context: ProcessingContext): ProcessingContext {
    const { grayImage } = context;

    // Apply Gaussian blur
    const blurred = new cv.Mat();
    try {
      cv.GaussianBlur(grayImage, blurred, this.kSize, this.sigmaX);

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
