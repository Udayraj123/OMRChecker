/**
 * CropPage - Automatic page detection and cropping preprocessor.
 *
 * TypeScript port of src/processors/image/CropPage.py
 *
 * This preprocessor automatically detects the page boundary using edge detection
 * and crops/warps the image to remove margins and correct perspective distortion.
 *
 * Extends WarpOnPointsCommon to leverage the parent's warping infrastructure.
 */

import { WarpOnPointsCommon } from './WarpOnPointsCommon';
import { PointArray } from './pointUtils';
import { logger } from '../../utils/logger';
import { WarpMethod } from '../constants';
import { findPageContourAndCorners } from './pageDetection';
import { ImageUtils } from '../../utils/ImageUtils';
import cv from '../../utils/opencv';

/**
 * Preprocessor for automatic page detection and cropping.
 *
 * Detects the page boundary in the image and applies perspective correction
 * to produce a properly aligned rectangular output.
 */
export class CropPage extends WarpOnPointsCommon {
  protected static override readonly __isInternalPreprocessor = true;

  private morphKernel?: cv.Mat;
  private useColoredCanny: boolean;

  constructor(
    options: any,
    relativeDir: string,
    saveImageOps: any,
    defaultProcessingImageShape: [number, number]
  ) {
    // Python can call self.method() before super(), TypeScript cannot
    // So we call static helper, then pass merged result to super()
    const merged = CropPage.validateAndMerge(options);
    super(merged, relativeDir, saveImageOps, defaultProcessingImageShape);

    // Extract validated options
    this.useColoredCanny = this.options.useColoredCanny;

    // Create morphological kernel
    const morphKernel = this.options.morphKernel;
    if (morphKernel) {
      this.morphKernel = cv.getStructuringElement(
        cv.MORPH_RECT,
        new cv.Size(morphKernel[0], morphKernel[1])
      );
    }
  }

  /**
   * Validate and merge options (static helper for constructor).
   * TypeScript doesn't allow calling instance methods before super(),
   * so we use a static helper that combines validate + merge.
   */
  private static validateAndMerge(options: any): Record<string, any> {
    const tuningOptions = options.tuningOptions || {};

    const parsed: any = {
      morphKernel: options.morphKernel || [10, 10],
      useColoredCanny: options.useColoredCanny || false,
      maxPointsPerEdge: options.maxPointsPerEdge || null,
      enableCropping: true,
      tuningOptions: {
        warpMethod: tuningOptions.warpMethod || WarpMethod.PERSPECTIVE_TRANSFORM,
        normalizeConfig: [],
        cannyConfig: [],
        ...tuningOptions,
      },
    };

    // Merge tuning options
    return {
      ...parsed,
      tuningOptions: {
        ...(parsed.tuningOptions || {}),
        ...(options.tuningOptions || {}),
      },
    };
  }

  /**
   * Validate and remap options schema (for polymorphism).
   */
  protected static override validateAndRemapOptionsSchema(options: any): Record<string, any> {
    const tuningOptions = options.tuningOptions || {};

    return {
      morphKernel: options.morphKernel || [10, 10],
      useColoredCanny: options.useColoredCanny || false,
      maxPointsPerEdge: options.maxPointsPerEdge || null,
      enableCropping: true,
      tuningOptions: {
        warpMethod: tuningOptions.warpMethod || WarpMethod.PERSPECTIVE_TRANSFORM,
        normalizeConfig: [],
        cannyConfig: [],
      },
    };
  }

  getClassName(): string {
    return 'CropPage';
  }

  /**
   * Prepare the image before extracting control points.
   * Normalizes the image for better page detection.
   */
  protected prepareImageBeforeExtraction(image: cv.Mat): cv.Mat {
    // Normalize image before page detection
    const normalized = ImageUtils.normalizeSingle(image);
    return normalized || image;
  }

  /**
   * Extract control and destination points from the image.
   *
   * Detects the page boundary and returns corner points for warping.
   *
   * Note: Currently supports PERSPECTIVE_TRANSFORM (4 corners) only.
   * Full edge point support for HOMOGRAPHY/REMAP_GRIDDATA is TODO.
   */
  protected extractControlDestinationPoints(
    image: cv.Mat,
    coloredImage: cv.Mat,
    filePath: string
  ): [PointArray, PointArray, Record<string, any>] {
    // Check colored Canny configuration
    if (this.useColoredCanny && !this.tuningConfig.outputs?.colored_outputs_enabled) {
      logger.warn(
        'Cannot process colored image for CropPage. ' +
        'useColoredCanny is true but colored_outputs_enabled is false.'
      );
    }

    // Use extracted page detection module
    const [corners] = findPageContourAndCorners(image, {
      coloredImage: this.tuningConfig.outputs?.colored_outputs_enabled ? coloredImage : undefined,
      useColoredCanny: this.useColoredCanny,
      morphKernel: this.morphKernel,
      filePath: filePath,
      debugImage: this.debugImage || undefined,
    });

    if (!corners || corners.length !== 4) {
      throw new Error(`Invalid page corners detected: ${corners?.length || 0} points`);
    }

    logger.debug(`Found page corners: ${JSON.stringify(corners)}`);

    // Calculate destination corners
    const [destinationCorners] = ImageUtils.getCroppedWarpedRectanglePoints(corners);

    // For now, only support PERSPECTIVE_TRANSFORM (4 corners)
    // TODO: Add support for HOMOGRAPHY/REMAP_GRIDDATA with edge points
    // This would require porting:
    // - ImageUtils.splitPatchContourOnCorners()
    // - ImageUtils.getControlDestinationPointsFromContour()
    // - MathUtils.selectEdgeFromRectangle()
    const edgeContoursMap = {}; // Empty for now

    return [corners, destinationCorners, edgeContoursMap];
  }

  cleanup(): void {
    if (this.morphKernel) {
      this.morphKernel.delete();
    }
  }
}
