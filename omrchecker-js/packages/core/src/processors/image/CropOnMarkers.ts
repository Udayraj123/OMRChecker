/**
 * CropOnMarkers - Factory wrapper for marker-based cropping preprocessors.
 *
 * TypeScript port of src/processors/image/CropOnMarkers.py
 * Simplified browser-compatible version.
 *
 * This preprocessor detects markers (either custom template markers or dots/lines)
 * on the page and uses them for precise alignment and cropping.
 *
 * Supports multiple marker types:
 * - FOUR_MARKERS: Custom template-based markers in four corners
 * - ONE_LINE_TWO_DOTS, TWO_DOTS_ONE_LINE: Hybrid configurations
 * - FOUR_DOTS: Four corner dots
 * - TWO_LINES, TWO_LINES_HORIZONTAL: Edge line detection
 *
 * Note: This is a simplified implementation. Full Python version includes:
 * - Template matching with multi-scale search
 * - Marker extraction and preprocessing
 * - Complex dot/line detection using morphological operations
 * - Advanced contour analysis and point selection
 * - Multiple warp methods with control/destination point pairs
 */

import type * as cv from '@techstark/opencv-js';
import { ImageTemplatePreprocessor } from './base';
import { Logger } from '../../utils/logger';

const logger = new Logger('CropOnMarkers');

export interface CropOnMarkersOptions {
  type: 'FOUR_MARKERS' | 'FOUR_DOTS' | 'TWO_LINES' | 'TWO_LINES_HORIZONTAL' | 'ONE_LINE_TWO_DOTS' | 'TWO_DOTS_ONE_LINE';
  referenceImage?: string;
  markerDimensions?: [number, number];
  defaultSelector?: string;
  tuningOptions?: {
    warpMethod?: string;
    [key: string]: any;
  };
  [key: string]: any;
}

/**
 * Factory preprocessor that delegates to marker-specific implementations.
 *
 * Creates either a CropOnCustomMarkers or CropOnDotLines instance based
 * on the marker type configuration.
 */
export class CropOnMarkers extends ImageTemplatePreprocessor {
  private markerType: string;
  // @ts-ignore - Will be used when full implementation is added
  private instance: ImageTemplatePreprocessor | null = null;

  constructor(
    options: CropOnMarkersOptions,
    relativeDir: string,
    saveImageOps: any,
    defaultProcessingImageShape: [number, number]
  ) {
    super(options, relativeDir, saveImageOps, defaultProcessingImageShape);
    this.markerType = options.type;

    // TODO: Instantiate appropriate implementation based on type
    // if (options.type === 'FOUR_MARKERS') {
    //   this.instance = new CropOnCustomMarkers(...);
    // } else {
    //   this.instance = new CropOnDotLines(...);
    // }
  }

  getClassName(): string {
    return 'CropOnMarkers';
  }

  applyFilter(
    image: cv.Mat,
    coloredImage: cv.Mat,
    template: any,
    _filePath: string
  ): [cv.Mat, cv.Mat, any] {
    logger.debug(`Applying ${this.getName()} filter with type: ${this.markerType}`);

    // TODO: Delegate to appropriate instance
    // if (this.instance) {
    //   return this.instance.applyFilter(image, coloredImage, template, filePath);
    // }

    // TODO: Implement marker detection and warping:
    //
    // For FOUR_MARKERS:
    // 1. Load reference marker images
    // 2. Extract and preprocess markers (blur, normalize, erode-subtract)
    // 3. For each quadrant, perform multi-scale template matching
    // 4. Find best match position for each marker
    // 5. Compute warp points from marker positions
    // 6. Apply perspective transform or homography
    //
    // For FOUR_DOTS / TWO_LINES:
    // 1. Define scan zones for each dot/line
    // 2. Apply morphological operations (open/close)
    // 3. Use Canny edge detection
    // 4. Find contours in each zone
    // 5. Extract corners or line edges
    // 6. Select control and destination points
    // 7. Apply warping transformation

    logger.warn(`CropOnMarkers[${this.markerType}] is a placeholder - returning image unchanged`);
    logger.info('Full implementation requires: Template matching, contour detection, perspective warping');

    return [image, coloredImage, template];
  }

  excludeFiles(): string[] {
    // TODO: Return list of reference image files to exclude from processing
    return [];
  }
}

