/**
 * Median Blur filter processor.
 *
 * TypeScript port of src/processors/image/MedianBlur.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import cv from '../../utils/opencv';
import { Processor, ProcessingContext } from '../base';
import { Logger } from '../../utils/logger';

const logger = new Logger('MedianBlur');

/**
 * Median Blur filter processor.
 *
 * Applies median blur to reduce salt-and-pepper noise.
 */
export class MedianBlur extends Processor {
  private kSize: number;

  constructor(options: { kSize?: number } = {}) {
    super();
    this.kSize = options.kSize || 5;
  }

  getName(): string {
    return 'MedianBlur';
  }

  process(context: ProcessingContext): ProcessingContext {
    try {
      const blurred = new cv.Mat();

      cv.medianBlur(context.grayImage, blurred, this.kSize);

      // Release old image and replace with blurred
      context.grayImage.delete();
      context.grayImage = blurred;

      logger.debug(`Applied median blur (kSize=${this.kSize})`);

      return context;
    } catch (error) {
      logger.error(`Error in MedianBlur: ${error}`);
      throw error;
    }
  }
}

