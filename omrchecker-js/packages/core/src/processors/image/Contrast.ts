/**
 * Contrast adjustment processor.
 *
 * TypeScript port of src/processors/image/Contrast.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import * as cv from '@techstark/opencv-js';
import { Processor, ProcessingContext } from '../base';
import { Logger } from '../../utils/logger';

const logger = new Logger('Contrast');

/**
 * Automatic brightness and contrast optimization.
 *
 * @param image - Input grayscale image
 * @param clipHistPercent - Percentage of histogram to clip (default: 1)
 * @returns Tuple of [processed image, alpha, beta]
 */
function automaticBrightnessAndContrast(
  image: cv.Mat,
  clipHistPercent: number = 1
): [cv.Mat, number, number] {
  // Calculate grayscale histogram
  const hist = new cv.Mat();
  const histSize = [256];
  const ranges = [0, 256];

  cv.calcHist(
    new cv.MatVector([image]),
    [0], // channels
    new cv.Mat(), // mask
    hist,
    histSize,
    ranges
  );

  // Calculate cumulative distribution
  const accumulator: number[] = [];
  accumulator.push(hist.data32F[0]);

  for (let i = 1; i < histSize[0]; i++) {
    accumulator.push(accumulator[i - 1] + hist.data32F[i]);
  }

  // Locate points to clip
  const maximum = accumulator[accumulator.length - 1];
  let clipAmount = (clipHistPercent * maximum) / 100.0 / 2.0;

  // Locate left cut
  let minimumGray = 0;
  while (accumulator[minimumGray] < clipAmount) {
    minimumGray++;
  }

  // Locate right cut
  let maximumGray = histSize[0] - 1;
  while (accumulator[maximumGray] >= maximum - clipAmount) {
    maximumGray--;
  }

  // Calculate alpha and beta values
  const alpha = 255 / (maximumGray - minimumGray);
  const beta = -minimumGray * alpha;

  // Apply contrast adjustment
  const result = new cv.Mat();
  image.convertTo(result, -1, alpha, beta);

  // Cleanup
  hist.delete();

  return [result, alpha, beta];
}

/**
 * Contrast adjustment processor.
 *
 * Adjusts image contrast using either manual or automatic mode.
 */
export class Contrast extends Processor {
  private clipHistPercent: number;
  private alpha: number;
  private beta: number;
  private mode: 'manual' | 'auto';

  constructor(
    options: {
      clipPercentage?: number;
      alpha?: number;
      beta?: number;
      mode?: 'manual' | 'auto';
    } = {}
  ) {
    super();
    this.clipHistPercent = options.clipPercentage || 1;
    this.alpha = options.alpha || 1.75;
    this.beta = options.beta || 0;
    this.mode = options.mode || 'manual';
  }

  getName(): string {
    return 'Contrast';
  }

  process(context: ProcessingContext): ProcessingContext {
    try {
      let adjusted: cv.Mat;

      if (this.mode === 'auto') {
        const [result, alpha, beta] = automaticBrightnessAndContrast(
          context.grayImage,
          this.clipHistPercent
        );
        adjusted = result;

        logger.debug(
          `Applied auto contrast (alpha=${alpha.toFixed(2)}, beta=${beta.toFixed(2)})`
        );
      } else {
        adjusted = new cv.Mat();
        context.grayImage.convertTo(adjusted, -1, this.alpha, this.beta);

        logger.debug(`Applied manual contrast (alpha=${this.alpha}, beta=${this.beta})`);
      }

      // Release old image and replace with adjusted
      context.grayImage.delete();
      context.grayImage = adjusted;

      return context;
    } catch (error) {
      logger.error(`Error in Contrast: ${error}`);
      throw error;
    }
  }
}

