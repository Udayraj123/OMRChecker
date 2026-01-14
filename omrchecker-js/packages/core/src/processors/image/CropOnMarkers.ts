/**
 * CropOnMarkers - Factory wrapper for marker-based cropping preprocessors.
 *
 * TypeScript port of src/processors/image/CropOnMarkers.py
 *
 * This preprocessor detects markers (either custom template markers or dots/lines)
 * on the page and uses them for precise alignment and cropping.
 *
 * Supports multiple marker types:
 * - FOUR_MARKERS: Custom template-based markers in four corners
 * - ONE_LINE_TWO_DOTS, TWO_DOTS_ONE_LINE: Hybrid configurations
 * - FOUR_DOTS: Four corner dots
 * - TWO_LINES, TWO_LINES_HORIZONTAL: Edge line detection
 */

import type * as cv from '@techstark/opencv-js';
import { ImageTemplatePreprocessor } from './base';
import { CropOnCustomMarkers } from './CropOnCustomMarkers';
import { CropOnDotLines } from './CropOnDotLines';

export interface CropOnMarkersOptions {
  type:
    | 'FOUR_MARKERS'
    | 'FOUR_DOTS'
    | 'TWO_LINES'
    | 'TWO_LINES_HORIZONTAL'
    | 'ONE_LINE_TWO_DOTS'
    | 'TWO_DOTS_ONE_LINE';
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
  protected static readonly __isInternalPreprocessor = false;
  private instance: ImageTemplatePreprocessor;

  constructor(
    options: CropOnMarkersOptions,
    relativeDir: string,
    saveImageOps: any,
    defaultProcessingImageShape: [number, number]
  ) {
    super(options, relativeDir, saveImageOps, defaultProcessingImageShape);

    // Delegate to appropriate implementation based on type
    if (options.type === 'FOUR_MARKERS') {
      this.instance = new CropOnCustomMarkers(
        options,
        relativeDir,
        saveImageOps,
        defaultProcessingImageShape
      );
    } else {
      // All other types use CropOnDotLines
      // TODO: Consider convex hull method for sparse blobs
      this.instance = new CropOnDotLines(
        options,
        relativeDir,
        saveImageOps,
        defaultProcessingImageShape
      );
    }
  }

  getClassName(): string {
    return 'CropOnMarkers';
  }

  applyFilter(
    image: cv.Mat,
    coloredImage: cv.Mat,
    template: any,
    filePath: string
  ): [cv.Mat, cv.Mat, any] {
    return this.instance.applyFilter(image, coloredImage, template, filePath);
  }

  excludeFiles(): string[] {
    return this.instance.excludeFiles();
  }

  toString(): string {
    return this.instance.toString();
  }
}

