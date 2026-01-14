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
import { PointArray } from './pointUtils';
import { logger } from '../../utils/logger';
import { ImageProcessingError, TemplateValidationError } from '../../exceptions';
import {
  ScannerType,
  SelectorType,
  type ScannerTypeValue,
  type SelectorTypeValue,
  type ZonePresetValue,
  EDGE_TYPES_IN_ORDER,
  type EdgeTypeValue,
} from '../constants';

// Type definitions for scan zones
export interface ZoneDescription {
  label: string;
  origin?: [number, number];
  dimensions?: [number, number];
  margins?: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
  selector?: SelectorTypeValue;
  scannerType: ScannerTypeValue;
  maxPoints?: number;
}

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
  protected options!: CropOnPatchesOptions;

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
      this.drawScanZoneUtil(scanZone.runtimeZoneDescription);
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
    const edgeContoursMap = this.getEdgeContoursMapFromZonePoints(zonePresetPoints);

    // For perspective transform, use ordered 4 corners
    if (this.warpMethod === 'PERSPECTIVE_TRANSFORM' && pageCorners.length === 4) {
      // TODO: Order points using MathUtils.orderFourPoints
      return [pageCorners, destinationPageCorners, edgeContoursMap];
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

  /**
   * Build edge contours map from detected zone points.
   */
  protected getEdgeContoursMapFromZonePoints(
    zonePresetPoints: Record<string, PointArray>
  ): Record<EdgeTypeValue, PointArray> {
    const edgeContoursMap: Record<EdgeTypeValue, PointArray> = {
      TOP: [],
      RIGHT: [],
      BOTTOM: [],
      LEFT: [],
    };

    // TODO: Implement TARGET_ENDPOINTS_FOR_EDGES mapping
    // For now, return empty map
    logger.debug('Edge contours map building not fully implemented');

    return edgeContoursMap;
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

    // Select point from rectangle
    const dotPoint = this.selectPointFromRectangle(dotRect, pointsSelector);
    if (!dotPoint) {
      throw new ImageProcessingError(`No dot found for zone ${zoneLabel}`);
    }

    // Compute destination rectangle
    const destinationRect = this.computeScanZoneRectangle(zoneDescription, false);
    const destinationPoint = this.selectPointFromRectangle(destinationRect, pointsSelector);
    if (!destinationPoint) {
      throw new ImageProcessingError(`Could not compute destination point for zone ${zoneLabel}`);
    }

    return [dotPoint, destinationPoint];
  }

  protected selectPointFromRectangle(
    rectangle: PointArray,
    selector: SelectorTypeValue
  ): PointArray[0] | null {
    if (rectangle.length !== 4) {
      return null;
    }

    const [tl, tr, br, bl] = rectangle;

    switch (selector) {
      case SelectorType.SELECT_TOP_LEFT:
        return tl;
      case SelectorType.SELECT_TOP_RIGHT:
        return tr;
      case SelectorType.SELECT_BOTTOM_RIGHT:
        return br;
      case SelectorType.SELECT_BOTTOM_LEFT:
        return bl;
      case SelectorType.SELECT_CENTER:
        return [Math.round((tl[0] + br[0]) / 2), Math.round((tl[1] + br[1]) / 2)];
      default:
        return null;
    }
  }

  // =========================================================================
  // Zone utilities
  // =========================================================================

  /**
   * Extract image zone from zone description.
   */
  protected computeScanZoneUtil(
    image: cv.Mat,
    zoneDescription: ZoneDescription
  ): [cv.Mat, [number, number], [number, number]] {
    // TODO: Use ShapeUtils.extractImageFromZoneDescription
    const origin = zoneDescription.origin || [0, 0];
    const dimensions = zoneDescription.dimensions || [image.cols, image.rows];

    const rect = new cv.Rect(origin[0], origin[1], dimensions[0], dimensions[1]);
    const zone = image.roi(rect);

    const zoneStart: [number, number] = [origin[0], origin[1]];
    const zoneEnd: [number, number] = [origin[0] + dimensions[0], origin[1] + dimensions[1]];

    return [zone, zoneStart, zoneEnd];
  }

  /**
   * Compute scan zone rectangle (4 corners).
   */
  protected computeScanZoneRectangle(
    zoneDescription: ZoneDescription,
    includeMargins: boolean
  ): PointArray {
    const origin = zoneDescription.origin || [0, 0];
    const dimensions = zoneDescription.dimensions || [0, 0];
    const margins = zoneDescription.margins || { top: 0, right: 0, bottom: 0, left: 0 };

    let [x, y] = origin;
    let [w, h] = dimensions;

    if (includeMargins) {
      x -= margins.left;
      y -= margins.top;
      w += margins.left + margins.right;
      h += margins.top + margins.bottom;
    }

    // Return corners: TL, TR, BR, BL
    return [
      [x, y],
      [x + w, y],
      [x + w, y + h],
      [x, y + h],
    ];
  }

  /**
   * Draw scan zone on debug image.
   */
  protected drawScanZoneUtil(zoneDescription: ZoneDescription): void {
    // TODO: Implement DrawingUtils.drawBoxDiagonal
    logger.debug(`Drawing scan zone: ${zoneDescription.label}`);
  }

  // =========================================================================
  // Template method overrides
  // =========================================================================

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

