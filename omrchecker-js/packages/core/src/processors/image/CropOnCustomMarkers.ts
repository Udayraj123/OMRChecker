/**
 * CropOnCustomMarkers - Custom marker-based cropping preprocessor
 *
 * TypeScript port of src/processors/image/CropOnCustomMarkers.py
 *
 * This preprocessor detects custom marker images (QR codes, logos, etc.) in
 * predefined scan zones and uses them as control points for perspective correction.
 *
 * Features:
 * - Multi-scale template matching for robust detection
 * - Automatic quadrant placement when zones not specified
 * - Reference marker image preprocessing
 * - Multiple selectors (centers, inner/outer corners, etc.)
 */

import cv from '@techstark/opencv-js';
import { CropOnPatchesCommon, type ZoneDescription, type ScanZone } from './CropOnPatchesCommon';
import { PointArray } from './pointUtils';
import { logger } from '../../utils/logger';
import { ImageProcessingError, TemplateValidationError } from '../../core/exceptions';
import { ImageUtils } from '../../utils/ImageUtils';
import {
  WarpMethod,
  ScannerType,
  SelectorType,
  ZonePreset,
  type ZonePresetValue,
  type SelectorTypeValue,
  MARKER_ZONE_TYPES_IN_ORDER,
} from '../constants';
import {
  prepareMarkerTemplate,
  detectMarkerInPatch,
  type ReferenceZone,
} from './markerDetection';

/**
 * Preprocessor for custom marker-based cropping.
 *
 * Detects custom marker images in the document and uses them as anchor points
 * for perspective correction and cropping.
 */
export class CropOnCustomMarkers extends CropOnPatchesCommon {
  protected static override readonly __isInternalPreprocessor = true;

  // Layout configurations
  protected static override readonly scanZonePresetsForLayout: Record<
    string,
    ZonePresetValue[]
  > = {
    FOUR_MARKERS: MARKER_ZONE_TYPES_IN_ORDER,
  };

  // Default zone descriptions for each marker preset
  protected static override readonly defaultScanZoneDescriptions: Record<
    string,
    Partial<ZoneDescription>
  > = {
    [ZonePreset.topLeftMarker]: {
      scannerType: ScannerType.TEMPLATE_MATCH,
      maxPoints: 2,
    },
    [ZonePreset.topRightMarker]: {
      scannerType: ScannerType.TEMPLATE_MATCH,
      maxPoints: 2,
    },
    [ZonePreset.bottomRightMarker]: {
      scannerType: ScannerType.TEMPLATE_MATCH,
      maxPoints: 2,
    },
    [ZonePreset.bottomLeftMarker]: {
      scannerType: ScannerType.TEMPLATE_MATCH,
      maxPoints: 2,
    },
    CUSTOM: {},
  };

  // Point selector presets
  protected static override readonly defaultPointsSelectorMap: Record<
    string,
    Record<string, SelectorTypeValue>
  > = {
    CENTERS: {
      [ZonePreset.topLeftMarker]: SelectorType.SELECT_CENTER as SelectorTypeValue,
      [ZonePreset.topRightMarker]: SelectorType.SELECT_CENTER as SelectorTypeValue,
      [ZonePreset.bottomRightMarker]: SelectorType.SELECT_CENTER as SelectorTypeValue,
      [ZonePreset.bottomLeftMarker]: SelectorType.SELECT_CENTER as SelectorTypeValue,
    },
    INNER_WIDTHS: {
      [ZonePreset.topLeftMarker]: SelectorType.SELECT_TOP_RIGHT as SelectorTypeValue,
      [ZonePreset.topRightMarker]: SelectorType.SELECT_TOP_LEFT as SelectorTypeValue,
      [ZonePreset.bottomRightMarker]: SelectorType.SELECT_BOTTOM_LEFT as SelectorTypeValue,
      [ZonePreset.bottomLeftMarker]: SelectorType.SELECT_BOTTOM_RIGHT as SelectorTypeValue,
    },
    INNER_HEIGHTS: {
      [ZonePreset.topLeftMarker]: SelectorType.SELECT_BOTTOM_LEFT as SelectorTypeValue,
      [ZonePreset.topRightMarker]: SelectorType.SELECT_BOTTOM_RIGHT as SelectorTypeValue,
      [ZonePreset.bottomRightMarker]: SelectorType.SELECT_TOP_RIGHT as SelectorTypeValue,
      [ZonePreset.bottomLeftMarker]: SelectorType.SELECT_TOP_LEFT as SelectorTypeValue,
    },
    INNER_CORNERS: {
      [ZonePreset.topLeftMarker]: SelectorType.SELECT_BOTTOM_RIGHT as SelectorTypeValue,
      [ZonePreset.topRightMarker]: SelectorType.SELECT_BOTTOM_LEFT as SelectorTypeValue,
      [ZonePreset.bottomRightMarker]: SelectorType.SELECT_TOP_LEFT as SelectorTypeValue,
      [ZonePreset.bottomLeftMarker]: SelectorType.SELECT_TOP_RIGHT as SelectorTypeValue,
    },
    OUTER_CORNERS: {
      [ZonePreset.topLeftMarker]: SelectorType.SELECT_TOP_LEFT as SelectorTypeValue,
      [ZonePreset.topRightMarker]: SelectorType.SELECT_TOP_RIGHT as SelectorTypeValue,
      [ZonePreset.bottomRightMarker]: SelectorType.SELECT_BOTTOM_RIGHT as SelectorTypeValue,
      [ZonePreset.bottomLeftMarker]: SelectorType.SELECT_BOTTOM_LEFT as SelectorTypeValue,
    },
  };

  // Instance properties
  private loadedReferenceImages: Map<string, cv.Mat> = new Map();
  private markerForZoneLabel: Map<string, cv.Mat> = new Map();
  private minMatchingThreshold: number;
  private markerRescaleRange: [number, number];
  private markerRescaleSteps: number;
  private applyErodeSubtract: boolean;

  constructor(
    options: any,
    relativeDir: string,
    saveImageOps: any,
    defaultProcessingImageShape: [number, number]
  ) {
    super(options, relativeDir, saveImageOps, defaultProcessingImageShape);

    // Extract tuning options
    const tuningOptions = this.options.tuningOptions || {};
    this.minMatchingThreshold = tuningOptions.minMatchingThreshold || 0.3;
    this.markerRescaleRange = tuningOptions.markerRescaleRange || [85, 115];
    this.markerRescaleSteps = tuningOptions.markerRescaleSteps || 5;
    this.applyErodeSubtract = tuningOptions.applyErodeSubtract !== false;

    // Initialize marker templates
    this.initResizedMarkers();
  }

  protected override validateAndRemapOptionsSchema(options: any): Record<string, any> {
    const referenceImagePath = options.referenceImage;
    const layoutType = options.type;
    const tuningOptions = options.tuningOptions || {};

    const parsedOptions: any = {
      defaultSelector: options.defaultSelector || 'CENTERS',
      pointsLayout: layoutType,
      enableCropping: true,
      tuningOptions: {
        warpMethod: tuningOptions.warpMethod || WarpMethod.PERSPECTIVE_TRANSFORM,
        ...tuningOptions,
      },
    };

    // Build scan zones from marker zone presets
    const defaultDimensions = options.markerDimensions || null;
    const parsedScanZones: ScanZone[] = [];

    const constructor = this.constructor as typeof CropOnCustomMarkers;
    const zonePresets = constructor.scanZonePresetsForLayout[layoutType] || [];

    for (const zonePreset of zonePresets) {
      const localDescription = options[zonePreset] || {};
      const localCustomOptions = { ...localDescription.customOptions };
      delete localDescription.customOptions;

      parsedScanZones.push({
        zonePreset,
        zoneDescription: {
          dimensions: defaultDimensions,
          ...localDescription,
        } as ZoneDescription,
        customOptions: {
          referenceImage: referenceImagePath,
          markerDimensions: defaultDimensions,
          ...localCustomOptions,
        },
      });
    }

    parsedOptions.scanZones = parsedScanZones;
    return parsedOptions;
  }

  protected override validateScanZones(): void {
    super.validateScanZones();

    // Additional marker-specific validations
    for (const scanZone of this.scanZones) {
      const zonePreset = scanZone.zonePreset;
      const zoneDescription = scanZone.zoneDescription;
      const customOptions = scanZone.customOptions || {};
      const zoneLabel = zoneDescription.label;

      if (MARKER_ZONE_TYPES_IN_ORDER.includes(zonePreset as ZonePresetValue)) {
        if (!customOptions.referenceImage) {
          throw new TemplateValidationError(
            'template',
            `referenceImage not provided for custom marker zone ${zoneLabel}`
          );
        }

        // TODO: Check if reference image file exists
        // const referenceImagePath = this.getRelativePath(customOptions.referenceImage);
        // if (!fs.existsSync(referenceImagePath)) {
        //   throw new ImageReadError(`Marker reference image not found: ${referenceImagePath}`);
        // }
      }
    }
  }

  /**
   * Initialize resized markers from reference images.
   */
  private initResizedMarkers(): void {
    for (const scanZone of this.scanZones) {
      const zoneDescription = scanZone.zoneDescription;
      const customOptions = scanZone.customOptions || {};
      const zoneLabel = zoneDescription.label;
      const scannerType = zoneDescription.scannerType;

      if (scannerType !== ScannerType.TEMPLATE_MATCH) {
        continue;
      }

      const referenceImagePath = customOptions.referenceImage;
      if (!referenceImagePath) {
        continue;
      }

      // Load reference image (or reuse if already loaded)
      // Note: In browser environment, this would need to be done asynchronously
      // For now, we'll log a message and skip. Real implementation would:
      // 1. Accept File objects or data URLs
      // 2. Load asynchronously using ImageUtils.loadImage()
      // 3. Cache in loadedReferenceImages

      logger.info(
        `To use marker ${zoneLabel}, provide reference image as File or data URL. ` +
        `Path-based loading (${referenceImagePath}) is not supported in browser environment.`
      );

      // If you have a File object or data URL, you can load it like this:
      // const referenceImage = await ImageUtils.loadImage(fileOrDataURL, 0); // 0 = grayscale
      // this.loadedReferenceImages.set(referenceImagePath, referenceImage);

      // For now, skip this marker
      continue;

      // Commented out code below shows what would happen if image was loaded:
      /*
      // Get reference zone (defaults to entire image)
      const referenceZone: ReferenceZone =
        customOptions.referenceZone || this.getDefaultScanZoneForImage(referenceImage);

      // Extract and prepare marker
      const extractedMarker = this.extractMarkerFromReference(
        referenceImage,
        referenceZone,
        customOptions
      );

      this.markerForZoneLabel.set(zoneLabel, extractedMarker);
      */
    }
  }

  /**
   * Load reference image from File or data URL (browser-compatible).
   *
   * This method should be called manually after construction if you want to use
   * custom markers in a browser environment.
   *
   * @param zoneLabel - Label of the zone to load marker for
   * @param imageSource - File object, Blob, or data URL string
   * @returns Promise that resolves when marker is loaded
   */
  async loadReferenceImageForZone(
    zoneLabel: string,
    imageSource: File | Blob | string
  ): Promise<void> {
    try {
      // Find the zone
      const scanZone = this.scanZones.find((z) => z.zoneDescription.label === zoneLabel);
      if (!scanZone) {
        throw new Error(`Zone ${zoneLabel} not found`);
      }

      const customOptions = scanZone.customOptions || {};
      const referenceImagePath = customOptions.referenceImage || zoneLabel;

      // Load image
      const referenceImage = await ImageUtils.loadImage(imageSource, 0); // 0 = grayscale
      this.loadedReferenceImages.set(referenceImagePath, referenceImage);

      // Get reference zone (defaults to entire image)
      const referenceZone: ReferenceZone =
        customOptions.referenceZone || this.getDefaultScanZoneForImage(referenceImage);

      // Extract and prepare marker
      const extractedMarker = this.extractMarkerFromReference(
        referenceImage,
        referenceZone,
        customOptions
      );

      this.markerForZoneLabel.set(zoneLabel, extractedMarker);

      logger.info(`Loaded marker for zone: ${zoneLabel}`);
    } catch (error) {
      logger.error(`Failed to load marker for zone ${zoneLabel}: ${error}`);
      throw new Error(`Failed to load marker for zone ${zoneLabel}: ${error}`);
    }
  }

  /**
   * Extract marker template from reference image.
   */
  private extractMarkerFromReference(
    referenceImage: cv.Mat,
    referenceZone: ReferenceZone,
    customOptions: Record<string, any>
  ): cv.Mat {
    const markerDimensions = customOptions.markerDimensions || (this.options as any).markerDimensions;
    const blurKernel = customOptions.markerBlurKernel || [5, 5];

    return prepareMarkerTemplate(
      referenceImage,
      referenceZone,
      markerDimensions,
      blurKernel as [number, number],
      this.applyErodeSubtract
    );
  }

  /**
   * Get default scan zone covering entire image.
   */
  private getDefaultScanZoneForImage(image: cv.Mat): ReferenceZone {
    return {
      origin: [1, 1],
      dimensions: [image.cols - 1, image.rows - 1],
    };
  }

  protected override getRuntimeZoneDescriptionWithDefaults(
    image: cv.Mat,
    scanZone: ScanZone
  ): ZoneDescription {
    const zonePreset = scanZone.zonePreset;
    const zoneDescription = scanZone.zoneDescription;

    // Non-marker zones use description as-is
    if (!MARKER_ZONE_TYPES_IN_ORDER.includes(zonePreset as ZonePresetValue)) {
      return zoneDescription;
    }

    const origin = zoneDescription.origin;
    const dimensions = zoneDescription.dimensions;

    // If origin/dimensions not specified, compute from quadrant
    if (!origin || !dimensions) {
      const zoneLabel = zoneDescription.label;
      const markerShape = this.markerForZoneLabel.get(zoneLabel);
      if (!markerShape) {
        logger.warn(`No marker found for zone ${zoneLabel}, using original description`);
        return zoneDescription;
      }

      const imageShape: [number, number] = [image.rows, image.cols];
      const markerDims: [number, number] = [markerShape.rows, markerShape.cols];

      const quadrantDescription = this.getQuadrantZoneDescription(
        zonePreset as ZonePresetValue,
        imageShape,
        markerDims
      );

      return {
        ...quadrantDescription,
        ...zoneDescription,
        origin: zoneDescription.origin || quadrantDescription.origin,
        dimensions: zoneDescription.dimensions || quadrantDescription.dimensions,
        margins: zoneDescription.margins || quadrantDescription.margins,
      };
    }

    return zoneDescription;
  }

  /**
   * Compute zone description for a quadrant.
   */
  private getQuadrantZoneDescription(
    patchType: ZonePresetValue,
    imageShape: [number, number],
    markerShape: [number, number]
  ): ZoneDescription {
    const [h, w] = imageShape;
    const halfHeight = Math.floor(h / 2);
    const halfWidth = Math.floor(w / 2);
    const [markerH, markerW] = markerShape;

    let zoneStart: [number, number];
    let zoneEnd: [number, number];

    switch (patchType) {
      case ZonePreset.topLeftMarker:
        zoneStart = [1, 1];
        zoneEnd = [halfWidth, halfHeight];
        break;
      case ZonePreset.topRightMarker:
        zoneStart = [halfWidth, 1];
        zoneEnd = [w, halfHeight];
        break;
      case ZonePreset.bottomRightMarker:
        zoneStart = [halfWidth, halfHeight];
        zoneEnd = [w, h];
        break;
      case ZonePreset.bottomLeftMarker:
        zoneStart = [1, halfHeight];
        zoneEnd = [halfWidth, h];
        break;
      default:
        throw new TemplateValidationError('template', `Unexpected quadrant patch_type ${patchType}`);
    }

    // Center the marker in the quadrant
    const origin: [number, number] = [
      Math.floor((zoneStart[0] + zoneEnd[0] - markerW) / 2),
      Math.floor((zoneStart[1] + zoneEnd[1] - markerH) / 2),
    ];

    const marginHorizontal = (zoneEnd[0] - zoneStart[0] - markerW) / 2 - 1;
    const marginVertical = (zoneEnd[1] - zoneStart[1] - markerH) / 2 - 1;

    return {
      label: patchType,
      origin,
      dimensions: [markerW, markerH],
      margins: {
        top: marginVertical,
        right: marginHorizontal,
        bottom: marginVertical,
        left: marginHorizontal,
      },
      scannerType: ScannerType.TEMPLATE_MATCH,
    };
  }

  /**
   * Find marker corners in a patch zone.
   */
  protected override findDotCornersFromOptions(
    image: cv.Mat,
    zoneDescription: ZoneDescription,
    _filePath: string
  ): PointArray {
    const zoneLabel = zoneDescription.label;

    // Extract patch zone
    const [patchZone, zoneStart, _zoneEnd] = this.computeScanZoneUtil(image, zoneDescription);

    // Get marker template
    const marker = this.markerForZoneLabel.get(zoneLabel);
    if (!marker) {
      throw new ImageProcessingError(`No marker template found for zone ${zoneLabel}`);
    }

    // Detect marker using multi-scale template matching
    const corners = detectMarkerInPatch(
      patchZone,
      marker,
      zoneStart,
      this.markerRescaleRange,
      this.markerRescaleSteps,
      this.minMatchingThreshold
    );

    // Clean up patch zone if it's a roi
    if (patchZone.ptr !== image.ptr) {
      patchZone.delete();
    }

    if (!corners) {
      throw new ImageProcessingError(`No marker found in patch ${zoneLabel}`);
    }

    return corners as PointArray;
  }

  /**
   * Not used by marker detection (only for dot/line processors).
   */
  protected override findAndSelectPointsFromLine(
    _image: cv.Mat,
    _zonePreset: ZonePresetValue,
    _zoneDescription: ZoneDescription,
    _filePath: string
  ): [PointArray, PointArray, PointArray] {
    throw new Error('CropOnCustomMarkers does not support line detection');
  }

  override excludeFiles(): string[] {
    return Array.from(this.loadedReferenceImages.keys());
  }

  protected override prepareImageBeforeExtraction(image: cv.Mat): cv.Mat {
    // Apply erode-subtract preprocessing if enabled
    if (!this.applyErodeSubtract) {
      return image;
    }

    // Erode-subtract enhances edges:
    // normalized_image = normalize(image - erode(image))
    try {
      const kernel = cv.Mat.ones(5, 5, cv.CV_8U);
      const eroded = new cv.Mat();
      cv.erode(image, eroded, kernel, new cv.Point(-1, -1), 5);
      kernel.delete();

      const subtracted = new cv.Mat();
      cv.subtract(image, eroded, subtracted);
      eroded.delete();

      const normalized = new cv.Mat();
      cv.normalize(subtracted, normalized, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
      subtracted.delete();

      return normalized;
    } catch (error) {
      logger.error(`Error in erode-subtract preprocessing: ${error}`);
      return image;
    }
  }

  /**
   * Clean up OpenCV Mat objects.
   */
  override cleanup(): void {
    super.cleanup();

    // Clean up loaded images
    for (const mat of this.loadedReferenceImages.values()) {
      mat.delete();
    }
    this.loadedReferenceImages.clear();

    // Clean up marker templates
    for (const mat of this.markerForZoneLabel.values()) {
      mat.delete();
    }
    this.markerForZoneLabel.clear();
  }

  getClassName(): string {
    return 'CropOnCustomMarkers';
  }
}

