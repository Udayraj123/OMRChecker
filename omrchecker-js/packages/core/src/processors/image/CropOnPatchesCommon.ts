/**
 * CropOnPatchesCommon - Base class for patch-based cropping
 *
 * TypeScript port of src/processors/image/CropOnPatchesCommon.py
 *
 * This abstract base class provides common functionality for processors that
 * detect control points from scan zones (patches) in the image and use them
 * for warping/cropping.
 *
 * Subclasses must implement:
 * - findDotCornersFromOptions()
 * - findAndSelectPointsFromLine()
 *
 * Common functionality provided:
 * - Scan zone management and validation
 * - Point extraction and selection
 * - Debug visualization
 * - Integration with WarpOnPointsCommon
 */

import cv from '@techstark/opencv-js';
import { WarpOnPointsCommon } from './WarpOnPointsCommon';
import { PointArray, orderFourPoints } from './pointUtils';
import {
  selectPointFromRectangle,
  computeScanZone,
  getEdgeContoursMapFromZonePoints,
  drawScanZone,
  type ZoneDescription as PatchUtilsZoneDescription,
} from './patchUtils';
import { logger } from '../../utils/logger';
import { ImageProcessingError, TemplateValidationError } from '../../core/exceptions';
import { ShapeUtils } from '../../utils/shapes';
import {
  ScannerType,
  SelectorType,
  type SelectorTypeValue,
  type ZonePresetValue,
  type EdgeTypeValue,
} from '../constants';

// Type definitions for scan zones
export type ZoneDescription = PatchUtilsZoneDescription;

export interface ScanZone {
  zonePreset: ZonePresetValue | string;
  zoneDescription: ZoneDescription;
  customOptions?: Record<string, any>;
  runtimeZoneDescription?: ZoneDescription;
}

export interface CropOnPatchesOptions {
  pointsLayout: string;
  defaultSelector: string;
  scanZones: ScanZone[];
  enableCropping?: boolean;
  tuningOptions?: Record<string, any>;
}

/**
 * Base class for processors that crop/warp based on patch detection.
 *
 * This class orchestrates:
 * 1. Scan zone setup and validation
 * 2. Point extraction from dots/lines/markers
 * 3. Debug visualization
 * 4. Integration with warping pipeline
 */
export abstract class CropOnPatchesCommon extends WarpOnPointsCommon {
  protected static readonly __isInternalPreprocessor = true;

  // Class-level configuration (overridden by subclasses)
  protected static readonly defaultScanZoneDescriptions: Record<string, Partial<ZoneDescription>> =
    {};
  protected static readonly defaultPointsSelectorMap: Record<
    string,
    Record<string, SelectorTypeValue>
  > = {};
  protected static readonly scanZonePresetsForLayout: Record<string, ZonePresetValue[]> = {};

  // Instance properties
  protected scanZones: ScanZone[] = [];
  protected defaultPointsSelector: Record<string, SelectorTypeValue> = {};
  declare protected options: CropOnPatchesOptions;

  // Abstract methods for subclasses
  protected abstract findDotCornersFromOptions(
    image: cv.Mat,
    zoneDescription: ZoneDescription,
    filePath: string
  ): PointArray;

  protected abstract findAndSelectPointsFromLine(
    image: cv.Mat,
    zonePreset: ZonePresetValue,
    zoneDescription: ZoneDescription,
    filePath: string
  ): [PointArray, PointArray, PointArray];

  constructor(
    options: any,
    relativeDir: string,
    saveImageOps: any,
    defaultProcessingImageShape: [number, number]
  ) {
    super(options, relativeDir, saveImageOps, defaultProcessingImageShape);

    // Parse and validate scan zones
    this.parseAndApplyScanZonePresetsAndDefaults();
    this.validateScanZones();
    this.validatePointsLayouts();

    // Set default points selector
    const constructor = this.constructor as typeof CropOnPatchesCommon;
    const selectorMap = constructor.defaultPointsSelectorMap;
    this.defaultPointsSelector = selectorMap[this.options.defaultSelector] || {};
  }

  // =========================================================================
  // Scan zone setup and validation
  // =========================================================================

  protected parseAndApplyScanZonePresetsAndDefaults(): void {
    const constructor = this.constructor as typeof CropOnPatchesCommon;
    const defaultDescriptions = constructor.defaultScanZoneDescriptions;
    const scanZones = this.options.scanZones || [];

    this.scanZones = scanZones.map((scanZone) => {
      const zonePreset = scanZone.zonePreset;
      const zoneDescription = scanZone.zoneDescription || {};
      const customOptions = scanZone.customOptions || {};

      // Set label default
      zoneDescription.label = zoneDescription.label || zonePreset;

      // Merge with defaults
      const defaultDescription = defaultDescriptions[zonePreset] || {};
      const mergedDescription = {
        ...defaultDescription,
        ...zoneDescription,
      };

      return {
        zonePreset,
        zoneDescription: mergedDescription,
        customOptions,
      };
    });
  }

  protected validateScanZones(): void {
    const seenLabels = new Set<string>();
    const repeatLabels = new Set<string>();

    for (const scanZone of this.scanZones) {
      const label = scanZone.zoneDescription.label;
      if (seenLabels.has(label)) {
        repeatLabels.add(label);
      }
      seenLabels.add(label);
    }

    if (repeatLabels.size > 0) {
      throw new TemplateValidationError(
        'template',
        `Found repeated labels in scanZones: ${Array.from(repeatLabels).join(', ')}`
      );
    }
  }

  protected validatePointsLayouts(): void {
    const constructor = this.constructor as typeof CropOnPatchesCommon;
    const layoutPresets = constructor.scanZonePresetsForLayout;
    const pointsLayout = this.options.pointsLayout;

    if (pointsLayout !== 'CUSTOM' && !(pointsLayout in layoutPresets)) {
      throw new TemplateValidationError(
        'template',
        `Invalid pointsLayout provided: ${pointsLayout}`
      );
    }

    if (pointsLayout === 'CUSTOM') {
      return;
    }

    // Check that all required zone presets are provided
    const expectedPresets = new Set(layoutPresets[pointsLayout]);
    const providedPresets = new Set(this.scanZones.map((z) => z.zonePreset));

    const missingPresets = Array.from(expectedPresets).filter((p) => !providedPresets.has(p));

    if (missingPresets.length > 0) {
      logger.error(`Missing zone presets: ${missingPresets.join(', ')}`);
      throw new TemplateValidationError(
        'template',
        `Missing zone presets for pointsLayout ${pointsLayout}: ${missingPresets.join(', ')}`
      );
    }
  }

  // =========================================================================
  // Main point extraction (called by WarpOnPointsCommon)
  // =========================================================================

  protected extractControlDestinationPoints(
    image: cv.Mat,
    _coloredImage: cv.Mat | null,
    filePath: string
  ): [PointArray, PointArray, Record<EdgeTypeValue, PointArray>] {
    const controlPoints: PointArray = [];
    const destinationPoints: PointArray = [];
    const zonePresetPoints: Record<string, PointArray> = {};
    const pageCorners: PointArray = [];
    const destinationPageCorners: PointArray = [];

    // Compute runtime zone descriptions
    for (const scanZone of this.scanZones) {
      scanZone.runtimeZoneDescription = this.getRuntimeZoneDescriptionWithDefaults(
        image,
        scanZone
      );
      this.drawScanZone(scanZone.runtimeZoneDescription);
    }

    // TODO: Show debug image if showImageLevel >= 4

    // Extract points from each zone
    for (const scanZone of this.scanZones) {
      const zonePreset = scanZone.zonePreset;
      const zoneDescription = scanZone.runtimeZoneDescription!;
      const scannerType = zoneDescription.scannerType;

      if (scannerType === ScannerType.PATCH_DOT || scannerType === ScannerType.TEMPLATE_MATCH) {
        const [dotPoint, destinationPoint] = this.findAndSelectPointFromDot(
          image,
          zoneDescription,
          filePath
        );

        controlPoints.push(dotPoint);
        destinationPoints.push(destinationPoint);
        zonePresetPoints[zonePreset] = [dotPoint];

        pageCorners.push(dotPoint);
        destinationPageCorners.push(destinationPoint);
      } else if (scannerType === ScannerType.PATCH_LINE) {
        const [zoneControlPoints, zoneDestinationPoints, selectedContour] =
          this.findAndSelectPointsFromLine(
            image,
            zonePreset as ZonePresetValue,
            zoneDescription,
            filePath
          );

        controlPoints.push(...zoneControlPoints);
        destinationPoints.push(...zoneDestinationPoints);
        zonePresetPoints[zonePreset] = selectedContour;

        // Add endpoints to page corners
        pageCorners.push(zoneControlPoints[0], zoneControlPoints[zoneControlPoints.length - 1]);
        destinationPageCorners.push(
          zoneDestinationPoints[0],
          zoneDestinationPoints[zoneDestinationPoints.length - 1]
        );
      }

      // TODO: Draw zone contours if showImageLevel >= 4
    }

    // Build edge contours map
    const edgeContoursMap = getEdgeContoursMapFromZonePoints(zonePresetPoints);

    // For perspective transform, use ordered 4 corners
    if (this.warpMethod === 'PERSPECTIVE_TRANSFORM' && pageCorners.length === 4) {
      // Order points consistently (TL, TR, BR, BL)
      const orderedCorners = orderFourPoints(pageCorners);
      return [orderedCorners, destinationPageCorners, edgeContoursMap];
    }

    return [controlPoints, destinationPoints, edgeContoursMap];
  }

  /**
   * Get runtime zone description with defaults.
   * Can be overridden by subclasses for dynamic zone placement.
   */
  protected getRuntimeZoneDescriptionWithDefaults(
    _image: cv.Mat,
    scanZone: ScanZone
  ): ZoneDescription {
    return scanZone.zoneDescription;
  }

  // =========================================================================
  // Point extraction utilities
  // =========================================================================

  protected findAndSelectPointFromDot(
    image: cv.Mat,
    zoneDescription: ZoneDescription,
    filePath: string
  ): [PointArray[0], PointArray[0]] {
    const zoneLabel = zoneDescription.label;
    const pointsSelector =
      zoneDescription.selector || this.defaultPointsSelector[zoneLabel] || SelectorType.SELECT_CENTER;

    // Find dot corners (subclass-specific)
    const dotRect = this.findDotCornersFromOptions(image, zoneDescription, filePath);

    // Select point from rectangle using extracted utility
    const dotPoint = selectPointFromRectangle(dotRect, pointsSelector);
    if (!dotPoint) {
      throw new ImageProcessingError(`No dot found for zone ${zoneLabel}`);
    }

    // Compute destination rectangle
    const destinationRect = this.computeScanZoneRectangle(zoneDescription, false);
    const destinationPoint = selectPointFromRectangle(destinationRect, pointsSelector);
    if (!destinationPoint) {
      throw new ImageProcessingError(`Could not compute destination point for zone ${zoneLabel}`);
    }

    return [dotPoint, destinationPoint];
  }

  // =========================================================================
  // Zone utilities
  // =========================================================================

  /**
   * Extract image zone from zone description.
   * Delegates to patchUtils.computeScanZone for core logic.
   * Kept for backward compatibility with existing code.
   */
  protected computeScanZoneUtil(
    image: cv.Mat,
    zoneDescription: ZoneDescription
  ): [cv.Mat, [number, number], [number, number]] {
    return computeScanZone(image, zoneDescription);
  }

  /**
   * Compute scan zone rectangle (4 corners).
   */
  protected computeScanZoneRectangle(
    zoneDescription: ZoneDescription,
    includeMargins: boolean
  ): PointArray {
    return ShapeUtils.computeScanZoneRectangle(zoneDescription, includeMargins);
  }

  /**
   * Draw scan zone on debug image.
   * Delegates to patchUtils.drawScanZone for core logic.
   */
  protected drawScanZone(zoneDescription: ZoneDescription): void {
    if (this.tuningConfig.outputs.showImageLevel >= 1 && this.debugImage) {
      drawScanZone(this.debugImage, zoneDescription);
    }
  }

  // =========================================================================
  // Template method overrides
  // =========================================================================

  getClassName(): string {
    return 'CropOnPatchesCommon';
  }

  protected validateAndRemapOptionsSchema(options: any): Record<string, any> {
    // Base implementation - subclasses should override
    return options;
  }

  protected prepareImageBeforeExtraction(image: cv.Mat): cv.Mat {
    // Base implementation - no preprocessing
    return image;
  }

  excludeFiles(): string[] {
    return [];
  }

  toString(): string {
    return `CropOnPatches["${this.options.pointsLayout}"]`;
  }
}

