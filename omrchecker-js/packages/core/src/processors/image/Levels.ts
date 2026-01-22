/**
 * Levels adjustment processor.
 *
 * TypeScript port of src/processors/image/Levels.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import cv from '../../utils/opencv';
import { Processor, ProcessingContext } from '../base';
import { Logger } from '../../utils/logger';

const logger = new Logger('Levels');

/**
 * Levels adjustment processor.
 *
 * Adjusts the input and output levels of an image using a gamma curve.
 */
export class Levels extends Processor {
  private gamma: cv.Mat;
  private low: number;
  private high: number;
  private gammaValue: number;

  constructor(
    options: {
      low?: number;
      high?: number;
      gamma?: number;
    } = {}
  ) {
    super();
    this.low = options.low || 0;
    this.high = options.high || 1;
    this.gammaValue = options.gamma || 1.0;

    // Create gamma lookup table
    this.gamma = this.createGammaLUT(this.low, this.high, this.gammaValue);
  }

  getName(): string {
    return 'Levels';
  }

  private createGammaLUT(low: number, high: number, gamma: number): cv.Mat {
    const lowScaled = 255 * low;
    const highScaled = 255 * high;
    const invGamma = 1.0 / gamma;

    const lut = new Uint8Array(256);

    for (let i = 0; i < 256; i++) {
      let value: number;

      if (i <= lowScaled) {
        value = 0;
      } else if (i >= highScaled) {
        value = 255;
      } else {
        const normalized = (i - lowScaled) / (highScaled - lowScaled);
        value = Math.pow(normalized, invGamma) * 255;
      }

      lut[i] = Math.round(value);
    }

    // Create OpenCV Mat from the LUT
    const lutMat = new cv.Mat(1, 256, cv.CV_8UC1);
    lutMat.data.set(lut);

    return lutMat;
  }

  process(context: ProcessingContext): ProcessingContext {
    try {
      const adjusted = new cv.Mat();

      // Apply lookup table
      cv.LUT(context.grayImage, this.gamma, adjusted);

      // Release old image and replace with adjusted
      context.grayImage.delete();
      context.grayImage = adjusted;

      logger.debug(
        `Applied levels adjustment (low=${this.low}, high=${this.high}, gamma=${this.gammaValue})`
      );

      return context;
    } catch (error) {
      logger.error(`Error in Levels: ${error}`);
      throw error;
    }
  }

  /**
   * Cleanup method to free the gamma LUT
   */
  cleanup(): void {
    if (this.gamma) {
      this.gamma.delete();
    }
  }
}

