/**
 * CropPage - Automatic page detection and cropping preprocessor.
 *
 * TypeScript port of src/processors/image/CropPage.py
 *
 * This preprocessor automatically detects the page boundary using edge detection
 * and crops/warps the image to remove margins and correct perspective distortion.
 *
 * Uses the extracted pageDetection module for clean separation of concerns.
 */

import type * as cv from '@techstark/opencv-js';
import { ImageTemplatePreprocessor } from './base';
import { Logger } from '../../utils/logger';
import { WarpMethod, type WarpMethodValue } from '../constants';
import { findPageContourAndCorners } from './pageDetection';

const logger = new Logger('CropPage');

export interface CropPageOptions {
  morphKernel?: [number, number];
  useColoredCanny?: boolean;
  maxPointsPerEdge?: number | null;
  tuningOptions?: {
    warpMethod?: WarpMethodValue;
  };
}

/**
 * Preprocessor for automatic page detection and cropping.
 *
 * Detects the page boundary in the image and applies perspective correction
 * to produce a properly aligned rectangular output.
 */
export class CropPage extends ImageTemplatePreprocessor {
  private morphKernel?: cv.Mat;
  private useColoredCanny: boolean;
  // Reserved for future warping implementation
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private warpMethod: WarpMethodValue;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private maxPointsPerEdge?: number | null;

  constructor(
    options: CropPageOptions,
    relativeDir: string,
    saveImageOps: any,
    defaultProcessingImageShape: [number, number]
  ) {
    const remappedOptions = {
      ...options,
      morphKernel: options.morphKernel || [10, 10],
      useColoredCanny: options.useColoredCanny || false,
      maxPointsPerEdge: options.maxPointsPerEdge || null,
      enableCropping: true,
      processingImageShape: defaultProcessingImageShape,
      tuningOptions: {
        warpMethod: options.tuningOptions?.warpMethod || WarpMethod.PERSPECTIVE_TRANSFORM,
        normalizeConfig: [],
        cannyConfig: [],
      },
    };

    super(remappedOptions, relativeDir, saveImageOps, defaultProcessingImageShape);

    this.useColoredCanny = remappedOptions.useColoredCanny;
    this.warpMethod = remappedOptions.tuningOptions.warpMethod;
    this.maxPointsPerEdge = remappedOptions.maxPointsPerEdge;

    // Create morphological kernel if specified
    if (remappedOptions.morphKernel) {
      const cv = (globalThis as any).cv;
      if (cv) {
        this.morphKernel = cv.getStructuringElement(
          cv.MORPH_RECT,
          new cv.Size(remappedOptions.morphKernel[0], remappedOptions.morphKernel[1])
        );
      }
    }
  }

  getClassName(): string {
    return 'CropPage';
  }

  applyFilter(
    image: cv.Mat,
    coloredImage: cv.Mat,
    template: any,
    filePath: string
  ): [cv.Mat, cv.Mat, any] {
    logger.debug(`Applying ${this.getName()} filter`);

    try {
      // Use extracted page detection module
      const [corners, _pageContour] = findPageContourAndCorners(image, {
        coloredImage: coloredImage,
        useColoredCanny: this.useColoredCanny,
        morphKernel: this.morphKernel,
        filePath: filePath,
      });

      logger.info(`Found page corners: ${JSON.stringify(corners)}`);

      // TODO: Implement warping logic
      // For now, return images unchanged
      // Full implementation needs:
      // 1. Calculate destination corners
      // 2. Apply perspective transform or other warp method
      // 3. Handle edge-based warping for HOMOGRAPHY/REMAP methods

      logger.warn(
        'CropPage: Warping not yet implemented - returning image unchanged. ' +
          'Full implementation requires perspective transform and edge-based warping.'
      );

      return [image, coloredImage, template];
    } catch (error) {
      logger.error(`CropPage failed: ${error}`);
      // Return images unchanged on error
      return [image, coloredImage, template];
    }
  }

  cleanup(): void {
    if (this.morphKernel) {
      this.morphKernel.delete();
    }
  }
}
