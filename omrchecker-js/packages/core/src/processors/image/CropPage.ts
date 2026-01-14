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

import * as cv from '@techstark/opencv-js';
import { ImageTemplatePreprocessor } from './base';
import { Logger } from '../../utils/logger';
import { WarpMethod, type WarpMethodValue } from '../constants';
import { findPageContourAndCorners } from './pageDetection';
import { ImageUtils } from '../../utils/ImageUtils';
import { MathUtils } from '../../utils/math';

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
  private warpMethod: WarpMethodValue;
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
      // Step 1: Use extracted page detection module
      const [corners, _pageContour] = findPageContourAndCorners(image, {
        coloredImage: coloredImage,
        useColoredCanny: this.useColoredCanny,
        morphKernel: this.morphKernel,
        filePath: filePath,
      });

      if (!corners || corners.length !== 4) {
        logger.warn(`Invalid page corners detected: ${corners?.length || 0} points`);
        return [image, coloredImage, template];
      }

      logger.info(`Found page corners: ${JSON.stringify(corners)}`);

      // Step 2: Order corners consistently (TL, TR, BR, BL)
      const [orderedCorners] = MathUtils.orderFourPoints(corners);

      // Step 3: Calculate destination corners and dimensions
      const [destinationMat, dimensions] =
        ImageUtils.getCroppedWarpedRectanglePoints(orderedCorners);

      const [width, height] = dimensions;
      logger.debug(`Warping to dimensions: ${width}x${height}`);

      // Step 4: Create source points matrix
      const sourcePointsFlat = orderedCorners.flat();
      const sourceMat = cv.matFromArray(4, 1, cv.CV_32FC2, sourcePointsFlat);

      // Step 5: Apply perspective transform
      const warpedGray = this.applyWarpTransform(
        image,
        sourceMat,
        destinationMat,
        width,
        height
      );

      // Step 6: Apply to colored image if available
      let warpedColored = coloredImage;
      if (coloredImage && !coloredImage.empty()) {
        warpedColored = this.applyWarpTransform(
          coloredImage,
          sourceMat,
          destinationMat,
          width,
          height
        );
      }

      // Cleanup temporary matrices
      sourceMat.delete();
      destinationMat.delete();

      logger.info(`Successfully warped page to ${width}x${height}`);
      return [warpedGray, warpedColored, template];
    } catch (error) {
      logger.error(`CropPage failed: ${error}`);
      // Return images unchanged on error
      return [image, coloredImage, template];
    }
  }

  /**
   * Apply perspective transformation to an image.
   *
   * @param image - Source image
   * @param sourceMat - Source points (4x1 CV_32FC2)
   * @param destinationMat - Destination points (4x1 CV_32FC2)
   * @param width - Output width
   * @param height - Output height
   * @returns Warped image
   */
  private applyWarpTransform(
    image: cv.Mat,
    sourceMat: cv.Mat,
    destinationMat: cv.Mat,
    width: number,
    height: number
  ): cv.Mat {
    // Compute perspective transformation matrix
    const transformMatrix = cv.getPerspectiveTransform(sourceMat, destinationMat);

    // Apply warp
    const warped = new cv.Mat();
    cv.warpPerspective(
      image,
      warped,
      transformMatrix,
      new cv.Size(width, height),
      cv.INTER_LINEAR
    );

    // Cleanup
    transformMatrix.delete();

    return warped;
  }

  cleanup(): void {
    if (this.morphKernel) {
      this.morphKernel.delete();
    }
  }
}
