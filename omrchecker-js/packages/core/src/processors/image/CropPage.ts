/**
 * CropPage processor — TypeScript port of Python: src/processors/image/CropPage.py
 *
 * Preprocessor for detecting and cropping the page boundary.
 * Uses edge detection and contour analysis to find the page rectangle,
 * then crops and warps the image to align the page.
 *
 * Extends WarpOnPointsCommon and implements the 3 abstract methods:
 *   - validateAndRemapOptionsSchema
 *   - prepareImageBeforeExtraction
 *   - extractControlDestinationPoints
 */

import cv from '@techstark/opencv-js';
import { WarpOnPointsCommon } from './WarpOnPointsCommon';
import { findPageContourAndCorners } from './page_detection';
import { ImageUtils } from '../../utils/image';
import { WarpMethod } from '../constants';

export interface CropPageOptions {
  morphKernel?: [number, number];
  useColoredCanny?: boolean;
  tuningOptions?: Record<string, any>;
  [key: string]: any;
}

export class CropPage extends WarpOnPointsCommon {
  private morphKernel: cv.Mat;
  private useColoredCanny: boolean;

  static readonly defaults = {
    morphKernel: [10, 10] as [number, number],
    useColoredCanny: false,
  };

  constructor(options: CropPageOptions = {}) {
    // Parent constructor calls validateAndRemapOptionsSchema via polymorphism —
    // the subclass override is already active at this point (JS prototype chain).
    super(options as Record<string, any>);

    // Re-derive the same parsed values the parent used; no extra work beyond
    // what validateAndRemapOptionsSchema already computed.
    const morphKernel: [number, number] =
      (options as any)['morph_kernel'] ??
      options['morphKernel'] ??
      CropPage.defaults.morphKernel;

    this.useColoredCanny =
      (options as any)['use_colored_canny'] ??
      options['useColoredCanny'] ??
      CropPage.defaults.useColoredCanny;

    this.morphKernel = cv.getStructuringElement(
      cv.MORPH_RECT,
      new cv.Size(morphKernel[0], morphKernel[1])
    );
  }

  getClassName(): string {
    return 'CropPage';
  }

  getName(): string {
    return 'CropPage';
  }

  validateAndRemapOptionsSchema(options: Record<string, any>): Record<string, any> {
    const tuningOptions: Record<string, any> = options['tuning_options'] ?? options['tuningOptions'] ?? {};

    const morphKernel: [number, number] =
      options['morph_kernel'] ??
      options['morphKernel'] ??
      CropPage.defaults.morphKernel;

    const useColoredCanny: boolean =
      options['use_colored_canny'] ??
      options['useColoredCanny'] ??
      CropPage.defaults.useColoredCanny;

    return {
      morph_kernel: morphKernel,
      use_colored_canny: useColoredCanny,
      max_points_per_edge: options['max_points_per_edge'] ?? options['maxPointsPerEdge'] ?? null,
      enable_cropping: true,
      tuning_options: {
        warp_method: tuningOptions['warp_method'] ?? WarpMethod.PERSPECTIVE_TRANSFORM,
        normalize_config: [],
        canny_config: [],
      },
    };
  }

  prepareImageBeforeExtraction(image: cv.Mat): cv.Mat {
    return ImageUtils.normalizeSingle(image);
  }

  extractControlDestinationPoints(
    image: cv.Mat,
    _coloredImage: cv.Mat | null,
    filePath: string
  ): [number[][], number[][], any] {
    // Find page corners via edge detection + contour analysis
    const [corners, pageContour] = findPageContourAndCorners(image, {
      morphKernel: this.morphKernel,
      useColoredCanny: this.useColoredCanny,
      filePath,
    });

    // Release the pageContour Mat (we only need corners for perspective transform)
    pageContour.delete();

    // Compute destination rectangle points from ordered corners
    // getCroppedWarpedRectanglePoints expects [tl, tr, br, bl]
    const [warpedPoints] = ImageUtils.getCroppedWarpedRectanglePoints(corners);

    // Return: [controlPoints, destinationPoints, edgeContoursMap]
    // For PERSPECTIVE_TRANSFORM, 4 corners are used directly
    return [corners, warpedPoints, null];
  }

  /**
   * Release the morphKernel Mat to avoid memory leaks.
   */
  dispose(): void {
    if (this.morphKernel && !this.morphKernel.isDeleted()) {
      this.morphKernel.delete();
    }
  }
}
