/**
 * Abstract base class for crop-on-patches image preprocessors.
 *
 * Port of Python: src/processors/image/crop_on_patches/common.py
 * Task: omr-l7m
 *
 * Extends WarpOnPointsCommon and provides shared infrastructure for
 * scan-zone management, point extraction, and scanner-type dispatch.
 *
 * Subclasses must implement:
 *  - findAndSelectPointsFromLine
 *  - findDotCornersFromOptions
 *  - validateAndRemapOptionsSchema
 *  - prepareImageBeforeExtraction
 *  - extractControlDestinationPoints
 */

import cv from '@techstark/opencv-js';
import { WarpOnPointsCommon } from './WarpOnPointsCommon';
import { ScannerType, ScannerTypeValue } from '../constants';

/**
 * Set of scanner types that behave like dots (single-point detection).
 * Subclasses may override this static property to extend with additional types
 * without touching the base class.
 *
 * Port of Python: CropOnPatchesCommon.dot_like_scanner_types
 */
export abstract class CropOnPatchesCommon extends WarpOnPointsCommon {
  /**
   * Scanner types that use single-point (dot-like) detection.
   * Equivalent to Python's `dot_like_scanner_types: ClassVar[frozenset]`.
   * Subclasses can override to add custom scanner types.
   */
  static dotLikeScannerTypes: ReadonlySet<ScannerTypeValue> = new Set([
    ScannerType.PATCH_DOT,
    ScannerType.TEMPLATE_MATCH,
  ]);

  // ── Abstract methods ────────────────────────────────────────────────────────

  abstract findAndSelectPointsFromLine(
    image: cv.Mat,
    zonePreset: string,
    zoneDescription: Record<string, any>,
    filePath: string
  ): [number[][], number[][], any];

  abstract findDotCornersFromOptions(
    image: cv.Mat,
    zoneDescription: Record<string, any>,
    filePath: string
  ): number[][];

  _buildBaseParsedOptions(): any { // TODO: Add parameters and return type
    // TODO: Implement
  }
}
