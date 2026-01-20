/**
 * CropOnDotLines - Dot and line-based cropping preprocessor
 *
 * TypeScript port of src/processors/image/CropOnDotLines.py
 *
 * This preprocessor detects dots and lines in predefined scan zones and uses
 * them as control points for perspective correction and cropping.
 *
 * Features:
 * - Dot detection using morphological operations
 * - Line detection with edge extraction
 * - Multiple layout types (ONE_LINE_TWO_DOTS, TWO_LINES, FOUR_DOTS, etc.)
 * - Flexible point selectors (centers, inner/outer edges, etc.)
 */

import cv from '@techstark/opencv-js';
import { CropOnPatchesCommon, type ZoneDescription, type ScanZone } from './CropOnPatchesCommon';
import { PointArray } from './pointUtils';
import { ImageProcessingError, TemplateValidationError } from '../../core/exceptions';
import { ImageUtils } from '../../utils/ImageUtils';
import { MathUtils, EdgeType as MathEdgeType } from '../../utils/math';
import { computeScanZone } from './patchUtils';
import {
  WarpMethod,
  ScannerType,
  SelectorType,
  ZonePreset,
  EdgeType,
  TARGET_EDGE_FOR_LINE,
  type ZonePresetValue,
  type EdgeTypeValue,
  type SelectorTypeValue,
  DOT_ZONE_TYPES_IN_ORDER,
} from '../constants';
import {
  detectDotCorners,
  detectLineCornersAndEdges,
  createStructuringElement,
} from './dotLineDetection';

/**
 * Preprocessor for dot and line-based cropping.
 *
 * Detects alignment dots and lines in the document and uses them as anchor points
 * for perspective correction and cropping.
 */
export class CropOnDotLines extends CropOnPatchesCommon {
  protected static override readonly __isInternalPreprocessor = true;

  // Layout configurations
  protected static override readonly scanZonePresetsForLayout: Record<
    string,
    ZonePresetValue[]
  > = {
    ONE_LINE_TWO_DOTS: [ZonePreset.topRightDot, ZonePreset.bottomRightDot, ZonePreset.leftLine],
    TWO_DOTS_ONE_LINE: [ZonePreset.rightLine, ZonePreset.topLeftDot, ZonePreset.bottomLeftDot],
    TWO_LINES: [ZonePreset.leftLine, ZonePreset.rightLine],
    TWO_LINES_HORIZONTAL: [ZonePreset.topLine, ZonePreset.bottomLine],
    FOUR_DOTS: DOT_ZONE_TYPES_IN_ORDER,
  };

  // Default zone descriptions for each preset
  protected static override readonly defaultScanZoneDescriptions: Record<
    string,
    Partial<ZoneDescription>
  > = {
    // Dot zones
    [ZonePreset.topLeftDot]: {
      scannerType: ScannerType.PATCH_DOT,
      selector: SelectorType.SELECT_CENTER,
      maxPoints: 2,
    },
    [ZonePreset.topRightDot]: {
      scannerType: ScannerType.PATCH_DOT,
      selector: SelectorType.SELECT_CENTER,
      maxPoints: 2,
    },
    [ZonePreset.bottomRightDot]: {
      scannerType: ScannerType.PATCH_DOT,
      selector: SelectorType.SELECT_CENTER,
      maxPoints: 2,
    },
    [ZonePreset.bottomLeftDot]: {
      scannerType: ScannerType.PATCH_DOT,
      selector: SelectorType.SELECT_CENTER,
      maxPoints: 2,
    },
    // Line zones
    [ZonePreset.topLine]: {
      scannerType: ScannerType.PATCH_LINE,
      selector: SelectorType.LINE_OUTER_EDGE,
    },
    [ZonePreset.leftLine]: {
      scannerType: ScannerType.PATCH_LINE,
      selector: SelectorType.LINE_OUTER_EDGE,
    },
    [ZonePreset.bottomLine]: {
      scannerType: ScannerType.PATCH_LINE,
      selector: SelectorType.LINE_OUTER_EDGE,
    },
    [ZonePreset.rightLine]: {
      scannerType: ScannerType.PATCH_LINE,
      selector: SelectorType.LINE_OUTER_EDGE,
    },
    CUSTOM: {},
  };

  // Point selector presets
  protected static override readonly defaultPointsSelectorMap: Record<
    string,
    Record<string, SelectorTypeValue>
  > = {
    CENTERS: {
      [ZonePreset.topLeftDot]: SelectorType.SELECT_CENTER as SelectorTypeValue,
      [ZonePreset.topRightDot]: SelectorType.SELECT_CENTER as SelectorTypeValue,
      [ZonePreset.bottomRightDot]: SelectorType.SELECT_CENTER as SelectorTypeValue,
      [ZonePreset.bottomLeftDot]: SelectorType.SELECT_CENTER as SelectorTypeValue,
      [ZonePreset.leftLine]: SelectorType.LINE_OUTER_EDGE as SelectorTypeValue,
      [ZonePreset.rightLine]: SelectorType.LINE_OUTER_EDGE as SelectorTypeValue,
    },
    INNER_WIDTHS: {
      [ZonePreset.topLeftDot]: SelectorType.SELECT_TOP_RIGHT as SelectorTypeValue,
      [ZonePreset.topRightDot]: SelectorType.SELECT_TOP_LEFT as SelectorTypeValue,
      [ZonePreset.bottomRightDot]: SelectorType.SELECT_BOTTOM_LEFT as SelectorTypeValue,
      [ZonePreset.bottomLeftDot]: SelectorType.SELECT_BOTTOM_RIGHT as SelectorTypeValue,
      [ZonePreset.leftLine]: SelectorType.LINE_INNER_EDGE as SelectorTypeValue,
      [ZonePreset.rightLine]: SelectorType.LINE_INNER_EDGE as SelectorTypeValue,
    },
    INNER_HEIGHTS: {
      [ZonePreset.topLeftDot]: SelectorType.SELECT_BOTTOM_LEFT as SelectorTypeValue,
      [ZonePreset.topRightDot]: SelectorType.SELECT_BOTTOM_RIGHT as SelectorTypeValue,
      [ZonePreset.bottomRightDot]: SelectorType.SELECT_TOP_RIGHT as SelectorTypeValue,
      [ZonePreset.bottomLeftDot]: SelectorType.SELECT_TOP_LEFT as SelectorTypeValue,
      [ZonePreset.leftLine]: SelectorType.LINE_OUTER_EDGE as SelectorTypeValue,
      [ZonePreset.rightLine]: SelectorType.LINE_OUTER_EDGE as SelectorTypeValue,
    },
    INNER_CORNERS: {
      [ZonePreset.topLeftDot]: SelectorType.SELECT_BOTTOM_RIGHT as SelectorTypeValue,
      [ZonePreset.topRightDot]: SelectorType.SELECT_BOTTOM_LEFT as SelectorTypeValue,
      [ZonePreset.bottomRightDot]: SelectorType.SELECT_TOP_LEFT as SelectorTypeValue,
      [ZonePreset.bottomLeftDot]: SelectorType.SELECT_TOP_RIGHT as SelectorTypeValue,
      [ZonePreset.leftLine]: SelectorType.LINE_INNER_EDGE as SelectorTypeValue,
      [ZonePreset.rightLine]: SelectorType.LINE_INNER_EDGE as SelectorTypeValue,
    },
    OUTER_CORNERS: {
      [ZonePreset.topLeftDot]: SelectorType.SELECT_TOP_LEFT as SelectorTypeValue,
      [ZonePreset.topRightDot]: SelectorType.SELECT_TOP_RIGHT as SelectorTypeValue,
      [ZonePreset.bottomRightDot]: SelectorType.SELECT_BOTTOM_RIGHT as SelectorTypeValue,
      [ZonePreset.bottomLeftDot]: SelectorType.SELECT_BOTTOM_LEFT as SelectorTypeValue,
      [ZonePreset.leftLine]: SelectorType.LINE_OUTER_EDGE as SelectorTypeValue,
      [ZonePreset.rightLine]: SelectorType.LINE_OUTER_EDGE as SelectorTypeValue,
    },
  };

  // Edge selector mapping for lines
  private static readonly edgeSelectorMap: Partial<
    Record<ZonePresetValue, Partial<Record<SelectorTypeValue, EdgeTypeValue>>>
  > = {
    [ZonePreset.topLine]: {
      [SelectorType.LINE_INNER_EDGE]: EdgeType.BOTTOM,
      [SelectorType.LINE_OUTER_EDGE]: EdgeType.TOP,
    },
    [ZonePreset.leftLine]: {
      [SelectorType.LINE_INNER_EDGE]: EdgeType.RIGHT,
      [SelectorType.LINE_OUTER_EDGE]: EdgeType.LEFT,
    },
    [ZonePreset.bottomLine]: {
      [SelectorType.LINE_INNER_EDGE]: EdgeType.TOP,
      [SelectorType.LINE_OUTER_EDGE]: EdgeType.BOTTOM,
    },
    [ZonePreset.rightLine]: {
      [SelectorType.LINE_INNER_EDGE]: EdgeType.LEFT,
      [SelectorType.LINE_OUTER_EDGE]: EdgeType.RIGHT,
    },
  };

  // Instance properties
  private lineKernelMorph: cv.Mat;
  private dotKernelMorph: cv.Mat;

  constructor(
    options: any,
    relativeDir: string,
    saveImageOps: any,
    defaultProcessingImageShape: [number, number]
  ) {
    // Python can call self.method() before super(), TypeScript cannot
    // So we call static helper, then pass merged result to super()
    const merged = CropOnDotLines.validateAndMerge(options);
    super(merged, relativeDir, saveImageOps, defaultProcessingImageShape);

    // Extract tuning options for kernels
    const tuningOptions = this.options.tuningOptions || {};
    const lineKernel = tuningOptions.lineKernel || [2, 10];
    const dotKernel = tuningOptions.dotKernel || [5, 5];

    this.lineKernelMorph = createStructuringElement('rect', lineKernel as [number, number]);
    this.dotKernelMorph = createStructuringElement('rect', dotKernel as [number, number]);
  }

  /**
   * Validate and merge options (static helper for constructor).
   * TypeScript doesn't allow calling instance methods before super(),
   * so we use a static helper that combines validate + merge.
   */
  private static validateAndMerge(options: any): Record<string, any> {
    // Validate
    const layoutType = options.type;
    const tuningOptions = options.tuningOptions || {};

    const parsed: any = {
      defaultSelector: options.defaultSelector || 'CENTERS',
      pointsLayout: layoutType,
      enableCropping: options.enableCropping !== false,
      tuningOptions: {
        warpMethod: tuningOptions.warpMethod || WarpMethod.PERSPECTIVE_TRANSFORM,
        ...tuningOptions,
      },
    };

    const parsedScanZones: ScanZone[] = [...(options.scanZones || [])];
    const zonePresets = CropOnDotLines.scanZonePresetsForLayout[layoutType] || [];
    for (const zonePreset of zonePresets) {
      if (zonePreset in options) {
        parsedScanZones.push({
          zonePreset,
          zoneDescription: options[zonePreset] as ZoneDescription,
          customOptions: {},
        });
      }
    }
    parsed.scanZones = parsedScanZones;

    // Merge tuning options
    return {
      ...parsed,
      tuningOptions: {
        ...(parsed.tuningOptions || {}),
        ...(options.tuningOptions || {}),
      },
    };
  }

  protected static override validateAndRemapOptionsSchema(options: any): Record<string, any> {
    const layoutType = options.type;
    const tuningOptions = options.tuningOptions || {};

    const parsedOptions: any = {
      defaultSelector: options.defaultSelector || 'CENTERS',
      pointsLayout: layoutType,
      enableCropping: options.enableCropping !== false,
      tuningOptions: {
        warpMethod: tuningOptions.warpMethod || WarpMethod.PERSPECTIVE_TRANSFORM,
        ...tuningOptions,
      },
    };

    // Build scan zones
    const parsedScanZones: ScanZone[] = [
      ...(options.scanZones || []),
    ];

    const constructor = this.constructor as typeof CropOnDotLines;
    const zonePresets = constructor.scanZonePresetsForLayout[layoutType] || [];

    for (const zonePreset of zonePresets) {
      if (zonePreset in options) {
        parsedScanZones.push({
          zonePreset,
          zoneDescription: options[zonePreset] as ZoneDescription,
          customOptions: {},
        });
      }
    }

    parsedOptions.scanZones = parsedScanZones;
    return parsedOptions;
  }

  /**
   * Find dot corners in a scan zone.
   */
  protected override findDotCornersFromOptions(
    image: cv.Mat,
    zoneDescription: ZoneDescription,
    _filePath: string
  ): PointArray {
    const tuningOptions = this.options.tuningOptions || {};
    const zoneLabel = zoneDescription.label;

    // Extract patch zone
    const [zone, zoneStart] = computeScanZone(image, zoneDescription);

    // Validate blur kernel if provided
    const dotBlurKernel = tuningOptions.dotBlurKernel;
    if (dotBlurKernel) {
      const [zoneH, zoneW] = [zone.rows, zone.cols];
      const [blurH, blurW] = dotBlurKernel;

      if (!(zoneH > blurH && zoneW > blurW)) {
        throw new TemplateValidationError(
          'template',
          `The zone '${zoneLabel}' is smaller than provided dotBlurKernel: [${zoneH}, ${zoneW}] < [${blurH}, ${blurW}]`
        );
      }
    }

    // Detect dot using extracted module
    const dotThreshold = tuningOptions.dotThreshold || 150;
    const corners = detectDotCorners(
      zone,
      zoneStart,
      this.dotKernelMorph,
      dotThreshold,
      dotBlurKernel
    );

    // Clean up zone if it's a roi
    if (zone.ptr !== image.ptr) {
      zone.delete();
    }

    if (!corners) {
      throw new ImageProcessingError(
        `No patch/dot found at origin: ${zoneDescription.origin} with dimensions: ${zoneDescription.dimensions}`
      );
    }

    return corners;
  }

  /**
   * Find line corners and select points from a line zone.
   */
  protected override findAndSelectPointsFromLine(
    image: cv.Mat,
    zonePreset: ZonePresetValue,
    zoneDescription: ZoneDescription,
    _filePath: string
  ): [PointArray, PointArray, PointArray] {
    const zoneLabel = zoneDescription.label;
    const pointsSelector =
      zoneDescription.selector || this.defaultPointsSelector[zoneLabel] || SelectorType.LINE_OUTER_EDGE;

    // Find line corners and edge contours
    const lineEdgeContoursMap = this.findLineCornersAndContours(image, zoneDescription);

    // Select the appropriate edge
    const constructor = this.constructor as typeof CropOnDotLines;
    const edgeSelectorForPreset = constructor.edgeSelectorMap[zonePreset as ZonePresetValue];
    if (!edgeSelectorForPreset) {
      throw new ImageProcessingError(
        `No edge selector mapping found for zone preset: ${zonePreset}`
      );
    }
    const selectedEdgeType = edgeSelectorForPreset[pointsSelector];
    if (!selectedEdgeType) {
      throw new ImageProcessingError(
        `No edge type found for selector ${pointsSelector} in zone preset ${zonePreset}`
      );
    }
    const targetEdgeType = TARGET_EDGE_FOR_LINE[zonePreset];

    let selectedContour = lineEdgeContoursMap[selectedEdgeType] || [];
    let destinationLine = this.selectEdgeFromScanZone(zoneDescription, selectedEdgeType);

    // Ensure clockwise order after extraction
    if (selectedEdgeType !== targetEdgeType) {
      selectedContour = [...selectedContour].reverse();
      destinationLine = [...destinationLine].reverse();
    }

    const maxPoints = zoneDescription.maxPoints || null;

    // Extrapolate destination line to get approximate destination points
    const [controlPoints, destinationPoints] = this.getControlDestinationPointsFromContour(
      selectedContour,
      destinationLine,
      maxPoints
    );

    return [controlPoints, destinationPoints, selectedContour];
  }

  /**
   * Find line corners and edge contours.
   */
  private findLineCornersAndContours(
    image: cv.Mat,
    zoneDescription: ZoneDescription
  ): Record<EdgeTypeValue, PointArray> {
    const tuningOptions = this.options.tuningOptions || {};
    const zoneLabel = zoneDescription.label;

    // Extract patch zone
    const [zone, zoneStart] = computeScanZone(image, zoneDescription);

    // Validate blur kernel if provided
    const lineBlurKernel = tuningOptions.lineBlurKernel;
    if (lineBlurKernel) {
      const [zoneH, zoneW] = [zone.rows, zone.cols];
      const [blurH, blurW] = lineBlurKernel;

      if (!(zoneH > blurH && zoneW > blurW)) {
        throw new TemplateValidationError(
          'template',
          `The zone '${zoneLabel}' is smaller than provided lineBlurKernel: [${zoneH}, ${zoneW}] < [${blurH}, ${blurW}]`
        );
      }
    }

    // Get gamma low from tuning config (TODO: should come from config)
    const gammaLow = 0.7; // Default value
    const lineThreshold = tuningOptions.lineThreshold || 180;

    // Detect line using extracted module
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [_corners, edgeContoursMap] = detectLineCornersAndEdges(
      zone,
      zoneStart,
      this.lineKernelMorph,
      gammaLow,
      lineThreshold,
      lineBlurKernel
    );

    // Clean up zone if it's a roi
    if (zone.ptr !== image.ptr) {
      zone.delete();
    }

    if (!edgeContoursMap) {
      throw new ImageProcessingError(
        `No line match found at origin: ${zoneDescription.origin} with dimensions: ${zoneDescription.dimensions}`
      );
    }

    return edgeContoursMap;
  }

  /**
   * Select edge points from scan zone rectangle.
   */
  private selectEdgeFromScanZone(
    zoneDescription: ZoneDescription,
    edgeType: EdgeTypeValue
  ): PointArray {
    const origin = zoneDescription.origin || [0, 0];
    const dimensions = zoneDescription.dimensions || [0, 0];

    // Create rectangle using MathUtils
    const rectangle = MathUtils.getRectanglePointsFromBox(origin, dimensions);

    // Select edge from rectangle using MathUtils
    // EdgeTypeValue from constants.ts is compatible with EdgeType enum from math.ts
    // Both use the same string values ('TOP', 'RIGHT', 'BOTTOM', 'LEFT')
    // Convert EdgeTypeValue to MathEdgeType enum
    const mathEdgeType = edgeType as unknown as MathEdgeType;
    return MathUtils.selectEdgeFromRectangle(rectangle, mathEdgeType);
  }

  /**
   * Get control and destination points from contour.
   */
  private getControlDestinationPointsFromContour(
    contour: PointArray,
    destinationLine: PointArray,
    maxPoints: number | null
  ): [PointArray, PointArray] {
    if (contour.length === 0 || destinationLine.length < 2) {
      return [[], []];
    }

    // Use ImageUtils for proper implementation
    return ImageUtils.getControlDestinationPointsFromContour(
      contour,
      [destinationLine[0], destinationLine[1]],
      maxPoints
    );
  }

  /**
   * Clean up OpenCV Mat objects.
   */
  override cleanup(): void {
    super.cleanup();

    if (this.lineKernelMorph) {
      this.lineKernelMorph.delete();
    }
    if (this.dotKernelMorph) {
      this.dotKernelMorph.delete();
    }
  }

  getClassName(): string {
    return 'CropOnDotLines';
  }
}

