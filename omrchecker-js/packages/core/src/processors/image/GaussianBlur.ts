/**
 * Gaussian Blur filter processor.
 *
 * TypeScript port of src/processors/image/GaussianBlur.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import * as cv from '@techstark/opencv-js';
import { Processor, ProcessingContext } from '../base';
import { Logger } from '../../utils/logger';

const logger = new Logger('GaussianBlur');

/**
 * Gaussian Blur filter processor.
 *
 * Applies Gaussian blur to reduce noise and detail.
 */
export class GaussianBlur extends Processor {
  private kSize: [number, number];
  private sigmaX: number;

  constructor(options: { kSize?: [number, number]; sigmaX?: number } = {}) {
    super();
    this.kSize = options.kSize || [3, 3];
    this.sigmaX = options.sigmaX || 0;
  }

  getName(): string {
    return 'GaussianBlur';
  }

  process(context: ProcessingContext): ProcessingContext {
    try {
      const blurred = new cv.Mat();
      const kSize = new cv.Size(this.kSize[0], this.kSize[1]);

      cv.GaussianBlur(context.grayImage, blurred, kSize, this.sigmaX);

      // Release old image and replace with blurred
      context.grayImage.delete();
      context.grayImage = blurred;

      logger.debug(`Applied Gaussian blur (kSize=${this.kSize}, sigmaX=${this.sigmaX})`);

      return context;
    } catch (error) {
      logger.error(`Error in GaussianBlur: ${error}`);
      throw error;
    }
  }
}

