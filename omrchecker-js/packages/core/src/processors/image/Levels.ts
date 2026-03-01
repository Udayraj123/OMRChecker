/**
 * Migrated from Python: src/processors/image/Levels.py
 * Agent: Oz
 * Phase: Filter Processors
 *
 * Levels adjustment processor using lookup tables.
 */

import cv from '@techstark/opencv-js';
import { ProcessingContext, Processor } from '../base';

export interface LevelsOptions {
  low?: number; // Input low level (0-1). Default: 0
  high?: number; // Input high level (0-1). Default: 1
  gamma?: number; // Gamma correction. Default: 1.0
}

/**
 * Adjusts image levels (similar to Photoshop levels adjustment).
 * 
 * This processor maps input pixel values to output values using a lookup table:
 * - Values below 'low' are mapped to black (0)
 * - Values above 'high' are mapped to white (255)
 * - Values in between are gamma-corrected and stretched
 * 
 * Use cases:
 * - Increase contrast by narrowing the input range
 * - Correct for under/over-exposed images
 * - Apply gamma correction to brighten/darken midtones
 * 
 * Configuration:
 * - low: Input low level as fraction (0-1). Default: 0
 * - high: Input high level as fraction (0-1). Default: 1
 * - gamma: Gamma correction factor. Default: 1.0
 *   - gamma < 1: brightens midtones
 *   - gamma > 1: darkens midtones
 */
export class Levels extends Processor {
  private gamma: Uint8Array;

  constructor(options: LevelsOptions = {}) {
    super();
    const low = options.low ?? 0;
    const high = options.high ?? 1;
    const gamma = options.gamma ?? 1.0;

    // Build lookup table
    this.gamma = this.buildLookupTable(low, high, gamma);
  }

  getName(): string {
    return 'Levels';
  }

  /**
   * Build lookup table for level adjustment.
   */
  private buildLookupTable(
    low: number,
    high: number,
    gamma: number
  ): Uint8Array {
    const lut = new Uint8Array(256);
    const invGamma = 1.0 / gamma;
    const lowPixel = Math.floor(255 * low);
    const highPixel = Math.floor(255 * high);

    for (let i = 0; i < 256; i++) {
      lut[i] = this.outputLevel(i, lowPixel, highPixel, invGamma);
    }

    return lut;
  }

  /**
   * Calculate output level for a given input value.
   */
  private outputLevel(
    value: number,
    low: number,
    high: number,
    invGamma: number
  ): number {
    if (value <= low) {
      return 0;
    }
    if (value >= high) {
      return 255;
    }
    const normalized = (value - low) / (high - low);
    return Math.round(Math.pow(normalized, invGamma) * 255);
  }

  process(context: ProcessingContext): ProcessingContext {
    const { grayImage } = context;

    // Create lookup table Mat
    const lut = cv.matFromArray(256, 1, cv.CV_8U, Array.from(this.gamma));

    // Apply lookup table
    const adjusted = new cv.Mat();
    try {
      cv.LUT(grayImage, lut, adjusted);

      // Replace gray image with adjusted version
      grayImage.delete();
      context.grayImage = adjusted;
    } catch (error) {
      // Clean up on error
      adjusted.delete();
      throw error;
    } finally {
      lut.delete();
    }

    return context;
  }
}
