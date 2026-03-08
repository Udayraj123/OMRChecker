/**
 * CropOnDotLines — patch-based crop using dot and line scan zones.
 *
 * Port of Python: src/processors/image/crop_on_patches/dot_lines.py
 * Task: omr-sun (refactor: use _buildBaseParsedOptions from CropOnPatchesCommon)
 *
 * Handles layouts: ONE_LINE_TWO_DOTS, TWO_DOTS_ONE_LINE, TWO_LINES,
 * TWO_LINES_HORIZONTAL, CUSTOM.
 */

import cv from '@techstark/opencv-js';
import { CropOnPatchesCommon } from './CropOnPatchesCommon';
import { WarpMethod } from '../constants';

export class CropOnDotLines extends CropOnPatchesCommon {
  // ── Abstract method implementations ─────────────────────────────────────────

  validateAndRemapOptionsSchema(options: Record<string, any>): Record<string, any> {
    const layoutType: string = options['type'];
    // Uses helper from CropOnPatchesCommon (omr-sun)
    const parsedOptions = this._buildBaseParsedOptions(options, layoutType, {
      enableCropping: options['enable_cropping'] ?? true,
      defaultWarpMethod: WarpMethod.PERSPECTIVE_TRANSFORM,
    });
    // TODO: build scan_zones from layout_type and zone descriptors
    return parsedOptions;
  }

  prepareImageBeforeExtraction(image: cv.Mat): cv.Mat {
    return image;
  }

  extractControlDestinationPoints(
    _image: cv.Mat,
    _coloredImage: cv.Mat | null,
    _filePath: string
  ): [number[][], number[][], any] {
    // TODO: implement dot/line detection and point extraction
    throw new Error('CropOnDotLines.extractControlDestinationPoints: not yet implemented');
  }

  findAndSelectPointsFromLine(
    _image: cv.Mat,
    _zonePreset: string,
    _zoneDescription: Record<string, any>,
    _filePath: string
  ): [number[][], number[][], any] {
    // TODO: implement line detection
    throw new Error('CropOnDotLines.findAndSelectPointsFromLine: not yet implemented');
  }

  findDotCornersFromOptions(
    _image: cv.Mat,
    _zoneDescription: Record<string, any>,
    _filePath: string
  ): number[][] {
    // TODO: implement dot corner detection
    throw new Error('CropOnDotLines.findDotCornersFromOptions: not yet implemented');
  }
}
