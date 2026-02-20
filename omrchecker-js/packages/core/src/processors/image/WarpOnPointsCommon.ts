/**
 * WarpOnPointsCommon - Refactored Version
 *
 * This is the refactored base class for processors that warp images based on control points.
 * The original 404-line monolithic class has been broken down into:
 *
 * 1. warpStrategies.ts - Different warping transformation methods (Strategy pattern)
 * 2. pointUtils.ts - Point parsing, validation, and manipulation utilities
 *
 * This class now focuses on:
 * - Orchestration of the warping pipeline
 * - Configuration management
 * - Debug visualization and image saving
 * - Template-specific abstract methods for subclasses
 */

import cv from '../../utils/opencv';
import { ImageTemplatePreprocessor, SaveImageOps } from './base';
import { WarpStrategyFactory, WarpStrategy } from './warpStrategies';
import { PointArray, orderFourPoints } from './pointUtils';
import { logger } from '../../utils/logger';
import { ImageUtils } from '../../utils/ImageUtils';
import { InteractionUtils } from '../../utils/InteractionUtils';
import { DrawingUtils } from '../../utils/drawing';
import { appendSaveImage } from '../../utils/ImageSaver';
import { WarpMethod, WarpMethodFlags, type WarpMethodValue } from '../constants';
import type { TuningConfig } from '../../template/types';
import { deepMerge } from '../../utils/object';
import { MathUtils } from '../../utils/math';

/**
 * Base class for image processors that apply warping transformations.
 *
 * This class provides a template method pattern for warping images:
 * 1. Prepare image for extraction (subclass-specific)
 * 2. Extract control and destination points (subclass-specific)
 * 3. Parse and validate points
 * 4. Apply warping strategy
 * 5. Save debug images and visualizations
 *
 * Subclasses must implement:
 * - validateAndRemapOptionsSchema()
 * - prepareImageBeforeExtraction()
 * - extractControlDestinationPoints()
 */
export abstract class WarpOnPointsCommon extends ImageTemplatePreprocessor {
  protected static readonly __isInternalPreprocessor = true;

  // Map configuration flags to OpenCV constants
  private static readonly warpMethodFlagsMap: Record<string, number> = {
    [WarpMethodFlags.INTER_LINEAR]: cv.INTER_LINEAR,
    [WarpMethodFlags.INTER_CUBIC]: cv.INTER_CUBIC,
    [WarpMethodFlags.INTER_NEAREST]: cv.INTER_NEAREST,
  };

  protected tuningConfig: TuningConfig;
  protected enableCropping: boolean;
  protected warpMethod: WarpMethodValue;
  protected warpMethodFlag: number;
  protected warpStrategy: WarpStrategy;
  protected debugImage: cv.Mat | null = null;
  protected debugHstack: cv.Mat[] = [];
  protected debugVstack: cv.Mat[][] = [];

  constructor(
    options: Record<string, unknown>,
    relativeDir: string,
    saveImageOps: SaveImageOps,
    defaultProcessingImageShape: [number, number]
  ) {
    // TypeScript doesn't allow calling this.method() before super()
    // So subclasses must validate in their constructors and pass merged options here

    // Initialize parent with (already validated and merged) options
    super(options, relativeDir, saveImageOps, defaultProcessingImageShape);

    // Store tuning config after parent initialization
    this.tuningConfig = saveImageOps.tuningConfig;

    // Extract configuration
    const opts = this.options;
    const tuningOptions = opts.tuningOptions || {};

    // Cropping configuration
    this.enableCropping = opts.enableCropping ?? false;

    // Determine warp method (default depends on cropping)
    this.warpMethod =
      tuningOptions.warp_method ??
      (this.enableCropping ? WarpMethod.PERSPECTIVE_TRANSFORM : WarpMethod.HOMOGRAPHY);

    // Get interpolation flag
    const flagName = (tuningOptions.warp_method_flag || 'INTER_LINEAR') as string;
    this.warpMethodFlag =
      WarpOnPointsCommon.warpMethodFlagsMap[flagName] ?? cv.INTER_LINEAR;

    // Create the appropriate warp strategy
    this.warpStrategy = this.createWarpStrategy();
  }

  /**
   * Create the appropriate warp strategy based on configuration.
   */
  private createWarpStrategy(): WarpStrategy {
    // Method-specific configurations
    const methodConfigs: Record<string, Record<string, unknown>> = {
      [WarpMethod.PERSPECTIVE_TRANSFORM]: {
        interpolationFlag: this.warpMethodFlag,
      },
      [WarpMethod.HOMOGRAPHY]: {
        interpolationFlag: this.warpMethodFlag,
        useRansac: false,
      },
      [WarpMethod.REMAP_GRIDDATA]: {
        interpolationMethod: 'cubic',
      },
      [WarpMethod.DOC_REFINE]: {},
    };

    // Get config for this method, default to interpolation_flag only
    const strategyConfig = methodConfigs[this.warpMethod] ?? {
      interpolationFlag: this.warpMethodFlag,
    };

    return WarpStrategyFactory.create(this.warpMethod, strategyConfig);
  }

  // =========================================================================
  // Abstract methods for subclasses
  // =========================================================================

  /**
   * Validate and transform processor-specific options.
   *
   * Subclasses must override this to define their schema.
   *
   * @param _options - Raw options to validate and transform
   * @returns Validated and transformed options
   */
  protected static validateAndRemapOptionsSchema(_options: Record<string, unknown>): Record<string, unknown> {
    throw new Error('Subclass must implement validateAndRemapOptionsSchema');
  }

  /**
   * Merge tuning options from original options into parsed options.
   *
   * This ensures tuningOptions from the original config aren't lost during validation.
   * Can be overridden by subclasses if custom merge logic is needed.
   *
   * @param parsedOptions - Options returned from validateAndRemapOptionsSchema
   * @param originalOptions - Original raw options
   * @returns Merged options
   */
  protected static mergeTuningOptions(
    parsedOptions: Record<string, unknown>,
    originalOptions: Record<string, unknown>
  ): Record<string, unknown> {
    return deepMerge(
      {
        tuningOptions: originalOptions.tuningOptions || {},
      },
      parsedOptions
    );
  }

  /**
   * Prepare the image before extracting control points.
   *
   * Subclasses can apply preprocessing (blur, threshold, etc).
   */
  protected abstract prepareImageBeforeExtraction(image: cv.Mat): cv.Mat;

  /**
   * Extract control and destination points from the image.
   *
   * @returns Tuple of [controlPoints, destinationPoints, edgeContoursMap]
   *   - controlPoints: Points in the source image
   *   - destinationPoints: Corresponding points in the target space
   *   - edgeContoursMap: Optional edge map for doc-refine method
   */
  protected abstract extractControlDestinationPoints(
    image: cv.Mat,
    coloredImage: cv.Mat | null,
    filePath: string
  ): [PointArray, PointArray, Record<string, unknown>];

  /**
   * Return list of files to exclude from processing
   */
  excludeFiles(): string[] {
    return [];
  }

  // =========================================================================
  // Main processing pipeline
  // =========================================================================

  /**
   * Apply the warping transformation to the image.
   *
   * This is the main entry point called by the processing pipeline.
   *
   * @param image - Grayscale input image
   * @param coloredImage - Colored version of input
   * @param template - Template configuration (unused in base class)
   * @param filePath - Path to the image file (for logging/debugging)
   * @returns Tuple of [warpedImage, warpedColoredImage, template]
   */
  applyFilter(
    image: cv.Mat,
    coloredImage: cv.Mat,
    template: Record<string, any>,
    filePath: string
  ): [cv.Mat, cv.Mat, Record<string, any>] {
    const config = this.tuningConfig;

    this.debugHstack = [];
    this.debugVstack = [];

    // Step 1: Prepare image (subclass-specific)
    const preparedImage = this.prepareImageBeforeExtraction(image);

    // Initialize debug state with prepared image so dimensions stay in sync after warp
    this.debugImage = preparedImage.clone();

    // Step 2: Extract control/destination points (subclass-specific)
    const coloredOrNull = coloredImage || null;
    const [controlPoints, destinationPoints, edgeContoursMap] =
      this.extractControlDestinationPoints(preparedImage, coloredOrNull, filePath);

    // Step 3: Parse and validate points
    const [parsedControlPoints, parsedDestinationPoints, warpedDimensions] =
      this.parseAndPreparePoints(image, controlPoints, destinationPoints);

    logger.debug(
      `Cropping Enabled: ${this.enableCropping}\n` +
        `Control points: ${parsedControlPoints.length}\n` +
        `Warped dimensions: ${warpedDimensions}`
    );

    // Step 4: Apply warping using strategy
    const [warpedImage, warpedColoredImage] = this.applyWarpStrategy(
      image,
      coloredOrNull,
      parsedControlPoints,
      parsedDestinationPoints,
      warpedDimensions,
      edgeContoursMap
    );

    // Step 5: Debug visualization and image saving
    this.saveDebugVisualizations(
      config,
      filePath,
      image,
      warpedImage,
      warpedColoredImage,
      parsedControlPoints,
      parsedDestinationPoints
    );

    // Clean up prepared image if it's different from input
    if (preparedImage.ptr !== image.ptr) {
      preparedImage.delete();
    }

    return [warpedImage, warpedColoredImage || coloredImage, template];
  }

  // =========================================================================
  // Point parsing and preparation
  // =========================================================================

  /**
   * Parse, deduplicate, and prepare control/destination points.
   *
   * @param image - Input image (for dimension reference)
   * @param controlPoints - Raw control points from extraction
   * @param destinationPoints - Raw destination points from extraction
   * @returns Tuple of [parsedControlPoints, parsedDestinationPoints, warpedDimensions]
   */
  private parseAndPreparePoints(
    image: cv.Mat,
    controlPoints: PointArray,
    destinationPoints: PointArray
  ): [PointArray, PointArray, [number, number]] {
    // Deduplicate points using Map to preserve order
    const uniquePairs = new Map<string, [number, number]>();
    for (let i = 0; i < controlPoints.length; i++) {
      const ctrl = controlPoints[i];
      const ctrlKey = `${ctrl[0]},${ctrl[1]}`;
      if (!uniquePairs.has(ctrlKey)) {
        uniquePairs.set(ctrlKey, destinationPoints[i]);
      }
    }

    const parsedControlPoints: PointArray = [];
    const parsedDestinationPoints: PointArray = [];

    uniquePairs.forEach((dest, ctrlKey) => {
      const [x, y] = ctrlKey.split(',').map(Number);
      parsedControlPoints.push([x, y]);
      parsedDestinationPoints.push(dest);
    });

    // Calculate warped dimensions
    const [w, h] = [image.cols, image.rows];
    const warpedDimensions = this.calculateWarpedDimensions(
      [w, h],
      parsedDestinationPoints
    );

    return [parsedControlPoints, parsedDestinationPoints, warpedDimensions];
  }

  /**
   * Calculate warped dimensions based on cropping settings.
   */
  private calculateWarpedDimensions(
    defaultDims: [number, number],
    destinationPoints: PointArray
  ): [number, number] {
    if (!this.enableCropping) {
      return defaultDims;
    }

    const [destinationBox, rectangleDimensions] = MathUtils.getBoundingBoxOfPoints(
      destinationPoints
    );

    // Shift points to origin for cropping (modifies in-place)
    const shiftedPoints = MathUtils.shiftPointsToOrigin(destinationBox[0], destinationPoints);
    for (let i = 0; i < destinationPoints.length; i++) {
      destinationPoints[i][0] = shiftedPoints[i][0];
      destinationPoints[i][1] = shiftedPoints[i][1];
    }

    return rectangleDimensions;
  }

  // =========================================================================
  // Warping strategy application
  // =========================================================================

  /**
   * Apply the configured warp strategy.
   *
   * @param image - Grayscale input
   * @param coloredImage - Optional colored input
   * @param controlPoints - Validated control points
   * @param destinationPoints - Validated destination points
   * @param warpedDimensions - Output dimensions
   * @param edgeContoursMap - Optional edge map for doc-refine
   * @returns Tuple of [warpedGray, warpedColored]
   */
  private applyWarpStrategy(
    image: cv.Mat,
    coloredImage: cv.Mat | null,
    controlPoints: PointArray,
    destinationPoints: PointArray,
    warpedDimensions: [number, number],
    edgeContoursMap: Record<string, any>
  ): [cv.Mat, cv.Mat | null] {
    // Prepare points for perspective transform
    let preparedControl = controlPoints;
    let preparedDest = destinationPoints;
    let preparedDims = warpedDimensions;

    [preparedControl, preparedDest, preparedDims] = this.preparePointsForStrategy(
      controlPoints,
      destinationPoints,
      warpedDimensions
    );

    // Build strategy kwargs
    const strategyKwargs = this.buildStrategyKwargs(edgeContoursMap);

    // Select colored input based on config
    const coloredInput = this.getColoredInput(coloredImage);

    // Apply the warp (strategy warps image, colored input, and debug image in one go)
    const [warpedImage, warpedColoredImage, warpedDebugImage] =
      this.warpStrategy.warpImage(
        image,
        coloredInput,
        preparedControl,
        preparedDest,
        preparedDims,
        this.debugImage,
        strategyKwargs
      );

    if (warpedDebugImage != null) {
      this.debugImage = warpedDebugImage;
    }

    return [warpedImage, warpedColoredImage];
  }

  /**
   * Prepare points specifically for perspective transform if needed.
   */
  private preparePointsForStrategy(
    controlPoints: PointArray,
    destinationPoints: PointArray,
    warpedDimensions: [number, number]
  ): [PointArray, PointArray, [number, number]] {
    if (this.warpMethod !== WarpMethod.PERSPECTIVE_TRANSFORM) {
      return [controlPoints, destinationPoints, warpedDimensions];
    }

    if (controlPoints.length !== 4) {
      throw new Error(
        `Expected 4 control points for perspective transform, found ${controlPoints.length}. ` +
          `Use tuningOptions['warpMethod'] for different methods.`
      );
    }

    // Order the 4 points consistently (TL, TR, BR, BL)
    const orderedControl = orderFourPoints(controlPoints);

    // Recalculate destination points from ordered control points
    const [newDestinationPoints, newDimensions] =
      ImageUtils.getCroppedWarpedRectanglePoints(orderedControl);

    return [orderedControl, newDestinationPoints, newDimensions];
  }

  /**
   * Build kwargs dict for strategy based on warp method.
   */
  private buildStrategyKwargs(edgeContoursMap: Record<string, any>): Record<string, unknown> {
    if (this.warpMethod !== WarpMethod.DOC_REFINE) {
      return {};
    }

    if (!edgeContoursMap) {
      throw new Error('DOC_REFINE method requires edge_contours_map');
    }

    return { edgeContoursMap };
  }

  /**
   * Return colored image only if colored outputs are enabled.
   */
  private getColoredInput(coloredImage: cv.Mat | null): cv.Mat | null {
    const coloredOutputsEnabled = this.tuningConfig?.outputs?.colored_outputs_enabled ?? false;
    return coloredOutputsEnabled ? coloredImage : null;
  }

  // =========================================================================
  // Debug visualization and image saving
  // =========================================================================

  /**
   * Save debug images and show interactive visualizations.
   *
   * @param config - Tuning configuration
   * @param filePath - Path for display titles
   * @param _originalImage - Original input image
   * @param _warpedImage - Warped output
   * @param warpedColoredImage - Optional warped colored output
   * @param _controlPoints - Control points used
   * @param _destinationPoints - Destination points used
   */
  private saveDebugVisualizations(
    config: Record<string, any>,
    filePath: string,
    _originalImage: cv.Mat,
    _warpedImage: cv.Mat,
    warpedColoredImage: cv.Mat | null,
    _controlPoints: PointArray,
    _destinationPoints: PointArray
  ): void {
    const showImageLevel = config?.outputs?.show_image_level ?? 0;

    // Show high-detail visualizations if configured
    if (showImageLevel >= 4) {
      this.showHighDetailVisualizations(
        config,
        filePath,
        _originalImage,
        _warpedImage,
        _controlPoints,
        _destinationPoints
      );
    }

    // Save images at appropriate levels
    this.saveDebugImages(_warpedImage, warpedColoredImage);

    // Show final preview if configured
    if (showImageLevel >= 5) {
      const title = `${this.enableCropping ? 'Cropped' : 'Warped'} Image Preview`;
      InteractionUtils.show(title, warpedColoredImage || _warpedImage);
      logger.info(`${title}: ${filePath}`);
    }
  }

  /**
   * Show detailed debug visualizations.
   */
  private showHighDetailVisualizations(
    config: Record<string, any>,
    filePath: string,
    _originalImage: cv.Mat,
    _warpedImage: cv.Mat,
    _controlPoints: PointArray,
    destinationPoints: PointArray
  ): void {
    const titlePrefix = this.enableCropping ? 'Cropped Image' : 'Warped Image';

    // Draw convex hull if cropping (debug_image is warped, so use destination-space points)
    if (this.enableCropping && this.debugImage) {
      const destInt = destinationPoints.map((p) => [
        Math.round(p[0]),
        Math.round(p[1]),
      ]);
      DrawingUtils.drawConvexHull(this.debugImage, destInt);
    }

    if (config.outputs.show_image_level >= 5 && this.debugImage) {
      InteractionUtils.show('Anchor Points', this.debugImage);
      logger.info('Anchor Points visualization');
    }

    // Match lines visualization - would require match points data structure
    // For now, log that this feature is available
    logger.debug(`${titlePrefix} with Match Lines: ${filePath}`);
  }

  /**
   * Save warped and debug images.
   */
  private async saveDebugImages(_warpedImage: cv.Mat, _warpedColoredImage: cv.Mat | null): Promise<void> {
    // Save warped image
    if (_warpedImage && !_warpedImage.empty()) {
      await appendSaveImage('warped', _warpedImage, { format: 'png' });
      logger.debug('Saved warped image');
    }

    // Save colored warped image if available
    if (_warpedColoredImage && !_warpedColoredImage.empty()) {
      await appendSaveImage('warped_colored', _warpedColoredImage, { format: 'png' });
      logger.debug('Saved colored warped image');
    }

    // Save debug image with anchor points if available
    if (this.debugImage && !this.debugImage.empty()) {
      await appendSaveImage('anchor_points', this.debugImage, { format: 'png' });
      logger.debug('Saved anchor points debug image');
    }
  }

  /**
   * Clean up OpenCV Mat objects to prevent memory leaks.
   */
  cleanup(): void {
    if (this.debugImage) {
      this.debugImage.delete();
      this.debugImage = null;
    }

    for (const mat of this.debugHstack) {
      mat.delete();
    }
    this.debugHstack = [];

    for (const row of this.debugVstack) {
      for (const mat of row) {
        mat.delete();
      }
    }
    this.debugVstack = [];
  }
}

