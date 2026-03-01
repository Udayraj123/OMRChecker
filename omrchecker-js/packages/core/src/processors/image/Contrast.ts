/**
 * Migrated from Python: src/processors/image/Contrast.py
 * Agent: Oz
 * Phase: Filter Processors
 *
 * Contrast adjustment processor with automatic and manual modes.
 */

import cv from '@techstark/opencv-js';
import { ProcessingContext, Processor } from '../base';

export interface ContrastOptions {
  mode?: 'auto' | 'manual';
  clipPercentage?: number; // For auto mode
  alpha?: number; // For manual mode
  beta?: number; // For manual mode
}

/**
 * Adjusts image contrast using automatic or manual methods.
 * 
 * Auto mode:
 * - Calculates optimal contrast based on histogram
 * - Clips specified percentage from both ends
 * - Good for varying lighting conditions
 * 
 * Manual mode:
 * - Uses alpha (contrast) and beta (brightness) parameters
 * - Formula: output = alpha * input + beta
 * - More control but requires tuning
 * 
 * Configuration:
 * - mode: 'auto' or 'manual'. Default: 'manual'
 * - clipPercentage: Percentage to clip from histogram (auto mode). Default: 1
 * - alpha: Contrast multiplier (manual mode). Default: 1.75
 * - beta: Brightness offset (manual mode). Default: 0
 */
export class Contrast extends Processor {
  private mode: 'auto' | 'manual';
  private clipPercentage: number;
  private alpha: number;
  private beta: number;

  constructor(options: ContrastOptions = {}) {
    super();
    this.mode = options.mode ?? 'manual';
    this.clipPercentage = options.clipPercentage ?? 1;
    this.alpha = options.alpha ?? 1.75;
    this.beta = options.beta ?? 0;
  }

  getName(): string {
    return 'Contrast';
  }

  /**
   * Automatic brightness and contrast optimization with histogram clipping.
   */
  private automaticBrightnessAndContrast(
    image: cv.Mat,
    clipPercentage: number
  ): { result: cv.Mat; alpha: number; beta: number } {
    // Calculate grayscale histogram
    const hist = new cv.Mat();
    const mask = new cv.Mat();
    const matVec = new cv.MatVector();
    matVec.push_back(image);
    cv.calcHist(
      matVec,
      [0],
      mask,
      hist,
      [256],
      [0, 256]
    );

    // Calculate cumulative distribution
    const accumulator: number[] = [];
    accumulator.push(hist.data32F[0]);
    for (let i = 1; i < 256; i++) {
      accumulator.push(accumulator[i - 1] + hist.data32F[i]);
    }

    // Locate points to clip
    const maximum = accumulator[255];
    const clipValue = (clipPercentage * maximum) / 100.0 / 2.0;

    // Locate left cut
    let minimumGray = 0;
    while (accumulator[minimumGray] < clipValue) {
      minimumGray++;
    }

    // Locate right cut
    let maximumGray = 255;
    while (accumulator[maximumGray] >= maximum - clipValue) {
      maximumGray--;
    }

    // Calculate alpha and beta values
    const alpha = 255 / (maximumGray - minimumGray);
    const beta = -minimumGray * alpha;

    // Apply transformation
    const result = new cv.Mat();
    cv.convertScaleAbs(image, result, alpha, beta);

    // Clean up
    hist.delete();
    mask.delete();
    matVec.delete();

    return { result, alpha, beta };
  }

  process(context: ProcessingContext): ProcessingContext {
    const { grayImage } = context;

    let adjusted: cv.Mat;

    if (this.mode === 'auto') {
      const { result } = this.automaticBrightnessAndContrast(
        grayImage,
        this.clipPercentage
      );
      adjusted = result;
    } else {
      // Manual mode
      adjusted = new cv.Mat();
      cv.convertScaleAbs(grayImage, adjusted, this.alpha, this.beta);
    }

    // Replace gray image with adjusted version
    grayImage.delete();
    context.grayImage = adjusted;

    return context;
  }
}
