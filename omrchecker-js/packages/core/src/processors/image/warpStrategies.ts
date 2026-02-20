/**
 * Image Warping Strategies
 *
 * Extracted from WarpOnPointsCommon to provide focused, testable
 * implementations of different warping/transformation methods.
 *
 * Each strategy encapsulates a specific approach to transforming images
 * based on control and destination points.
 */

import cv from '../../utils/opencv';
import { logger } from '../../utils/logger';
import { WarpMethod, type WarpMethodValue } from '../constants';

/**
 * Abstract base class for image warping strategies.
 *
 * Each strategy implements a specific method for transforming an image
 * from control points to destination points.
 */
export abstract class WarpStrategy {
  /**
   * Apply warping transformation to images.
   *
   * @param image - Grayscale input image
   * @param coloredImage - Optional colored version
   * @param controlPoints - Source points in the original image
   * @param destinationPoints - Target points in the warped image
   * @param warpedDimensions - [width, height] of output image
   * @param debugImage - Optional debug/overlay image to warp with same transform
   * @param kwargs - Strategy-specific parameters
   * @returns Tuple of [warpedGrayImage, warpedColoredImage, warpedDebugImage]
   */
  abstract warpImage(
    image: cv.Mat,
    coloredImage: cv.Mat | null,
    controlPoints: number[][],
    destinationPoints: number[][],
    warpedDimensions: [number, number],
    debugImage?: cv.Mat | null,
    kwargs?: Record<string, any>
  ): [cv.Mat, cv.Mat | null, cv.Mat | null];

  /**
   * Return the name of this warping strategy
   */
  abstract getName(): string;
}

/**
 * Perspective transformation using 4-point homography.
 *
 * This is the most common method for document rectification.
 * Requires exactly 4 control points forming a quadrilateral.
 */
export class PerspectiveTransformStrategy extends WarpStrategy {
  private interpolationFlag: number;

  /**
   * Initialize perspective transform strategy.
   *
   * @param interpolationFlag - OpenCV interpolation method
   *   - cv.INTER_LINEAR: Bilinear (default, good balance)
   *   - cv.INTER_CUBIC: Bicubic (slower, higher quality)
   *   - cv.INTER_NEAREST: Nearest neighbor (fastest, lower quality)
   */
  constructor(interpolationFlag: number = cv.INTER_LINEAR) {
    super();
    this.interpolationFlag = interpolationFlag;
  }

  getName(): string {
    return 'PerspectiveTransform';
  }

  warpImage(
    image: cv.Mat,
    coloredImage: cv.Mat | null,
    controlPoints: number[][],
    destinationPoints: number[][],
    warpedDimensions: [number, number],
    debugImage?: cv.Mat | null,
    _kwargs?: Record<string, any>
  ): [cv.Mat, cv.Mat | null, cv.Mat | null] {
    if (controlPoints.length !== 4) {
      throw new Error(
        `PerspectiveTransform requires exactly 4 control points, got ${controlPoints.length}`
      );
    }

    // Convert to cv.Mat for OpenCV
    const controlPts = cv.matFromArray(4, 1, cv.CV_32FC2, controlPoints.flat());
    const destPts = cv.matFromArray(4, 1, cv.CV_32FC2, destinationPoints.flat());

    // Compute perspective transformation matrix (once)
    const transformMatrix = cv.getPerspectiveTransform(controlPts, destPts);

    // Apply to grayscale image
    const [width, height] = warpedDimensions;
    const warpedImage = new cv.Mat();
    cv.warpPerspective(
      image,
      warpedImage,
      transformMatrix,
      new cv.Size(width, height),
      this.interpolationFlag
    );

    // Apply to colored image if provided
    let warpedColoredImage: cv.Mat | null = null;
    if (coloredImage !== null) {
      warpedColoredImage = new cv.Mat();
      cv.warpPerspective(
        coloredImage,
        warpedColoredImage,
        transformMatrix,
        new cv.Size(width, height),
        this.interpolationFlag
      );
    }

    // Apply same transform to debug image if provided
    let warpedDebugImage: cv.Mat | null = null;
    if (debugImage != null && !debugImage.empty()) {
      warpedDebugImage = new cv.Mat();
      cv.warpPerspective(
        debugImage,
        warpedDebugImage,
        transformMatrix,
        new cv.Size(width, height),
        this.interpolationFlag
      );
    }

    // Clean up temporary matrices
    controlPts.delete();
    destPts.delete();
    transformMatrix.delete();

    logger.debug(
      `Applied perspective transform: [${image.rows}, ${image.cols}] -> [${width}, ${height}]`
    );

    return [warpedImage, warpedColoredImage, warpedDebugImage];
  }
}

/**
 * Homography-based transformation using N points.
 *
 * More flexible than perspective transform, can use more than 4 points.
 * Uses cv.findHomography with least-squares fitting.
 */
export class HomographyStrategy extends WarpStrategy {
  private interpolationFlag: number;
  private useRansac: boolean;
  private ransacThreshold: number;

  /**
   * Initialize homography strategy.
   *
   * @param interpolationFlag - OpenCV interpolation method
   * @param useRansac - Use RANSAC for robust estimation
   * @param ransacThreshold - RANSAC reprojection threshold (pixels)
   */
  constructor(
    interpolationFlag: number = cv.INTER_LINEAR,
    useRansac: boolean = false,
    ransacThreshold: number = 3.0
  ) {
    super();
    this.interpolationFlag = interpolationFlag;
    this.useRansac = useRansac;
    this.ransacThreshold = ransacThreshold;
  }

  getName(): string {
    return 'Homography';
  }

  warpImage(
    image: cv.Mat,
    coloredImage: cv.Mat | null,
    controlPoints: number[][],
    destinationPoints: number[][],
    warpedDimensions: [number, number],
    debugImage?: cv.Mat | null,
    _kwargs?: Record<string, any>
  ): [cv.Mat, cv.Mat | null, cv.Mat | null] {
    if (controlPoints.length < 4) {
      throw new Error(
        `Homography requires at least 4 control points, got ${controlPoints.length}`
      );
    }

    // Convert to cv.Mat
    const controlPts = cv.matFromArray(
      controlPoints.length,
      1,
      cv.CV_32FC2,
      controlPoints.flat()
    );
    const destPts = cv.matFromArray(
      destinationPoints.length,
      1,
      cv.CV_32FC2,
      destinationPoints.flat()
    );

    // Compute homography (once)
    const method = this.useRansac ? cv.RANSAC : 0;
    const mask = new cv.Mat();
    const homography = cv.findHomography(
      controlPts,
      destPts,
      method,
      this.ransacThreshold
    );

    if (homography.empty()) {
      controlPts.delete();
      destPts.delete();
      mask.delete();
      homography.delete();
      throw new Error('Failed to compute homography matrix');
    }

    // Apply warping
    const [width, height] = warpedDimensions;
    const warpedImage = new cv.Mat();
    cv.warpPerspective(
      image,
      warpedImage,
      homography,
      new cv.Size(width, height),
      this.interpolationFlag
    );

    let warpedColoredImage: cv.Mat | null = null;
    if (coloredImage !== null) {
      warpedColoredImage = new cv.Mat();
      cv.warpPerspective(
        coloredImage,
        warpedColoredImage,
        homography,
        new cv.Size(width, height),
        this.interpolationFlag
      );
    }

    let warpedDebugImage: cv.Mat | null = null;
    if (debugImage != null && !debugImage.empty()) {
      warpedDebugImage = new cv.Mat();
      cv.warpPerspective(
        debugImage,
        warpedDebugImage,
        homography,
        new cv.Size(width, height),
        this.interpolationFlag
      );
    }

    // Count inliers if using RANSAC
    let inliers = controlPoints.length;
    if (this.useRansac && !mask.empty()) {
      inliers = cv.countNonZero(mask);
    }

    // Clean up
    controlPts.delete();
    destPts.delete();
    mask.delete();
    homography.delete();

    logger.debug(`Applied homography with ${inliers}/${controlPoints.length} inliers`);

    return [warpedImage, warpedColoredImage, warpedDebugImage];
  }
}

/**
 * Grid-based interpolation using custom interpolation.
 *
 * Creates a dense warp field by interpolating between sparse control points.
 * Useful for non-linear transformations and when you have many control points.
 *
 * Note: This is a simplified version compared to Python's scipy.interpolate.griddata.
 * For production use, consider implementing proper interpolation or using a library.
 */
export class GridDataRemapStrategy extends WarpStrategy {
  // interpolationMethod is for future use when full griddata is implemented
  // private interpolationMethod: 'linear' | 'nearest' | 'cubic';

  /**
   * Initialize griddata remap strategy.
   *
   * @param interpolationMethod - 'linear', 'nearest', or 'cubic' (not yet used)
   */
  constructor(_interpolationMethod: 'linear' | 'nearest' | 'cubic' = 'cubic') {
    super();
    // this.interpolationMethod = interpolationMethod;
  }

  getName(): string {
    return 'GridDataRemap';
  }

  warpImage(
    image: cv.Mat,
    coloredImage: cv.Mat | null,
    controlPoints: number[][],
    destinationPoints: number[][],
    warpedDimensions: [number, number],
    debugImage?: cv.Mat | null,
    kwargs?: Record<string, any>
  ): [cv.Mat, cv.Mat | null, cv.Mat | null] {
    // TODO: Implement proper grid interpolation
    // This requires a JavaScript interpolation library or custom implementation
    // For now, fall back to perspective transform if we have 4 points

    logger.warn(
      'GridDataRemap not fully implemented in TypeScript, falling back to perspective transform'
    );

    if (controlPoints.length === 4) {
      const perspectiveStrategy = new PerspectiveTransformStrategy();
      return perspectiveStrategy.warpImage(
        image,
        coloredImage,
        controlPoints,
        destinationPoints,
        warpedDimensions,
        debugImage,
        kwargs
      );
    }

    throw new Error(
      'GridDataRemap requires JavaScript interpolation library (not yet implemented)'
    );
  }
}

/**
 * Document rectification using custom scanline-based approach.
 *
 * Uses edge contours to create a detailed warp field that preserves
 * document structure better than simple perspective transform.
 *
 * Note: Requires rectify helper which may not be ported yet.
 */
export class DocRefineRectifyStrategy extends WarpStrategy {
  getName(): string {
    return 'DocRefineRectify';
  }

  warpImage(
    _image: cv.Mat,
    _coloredImage: cv.Mat | null,
    _controlPoints: number[][],
    _destinationPoints: number[][],
    _warpedDimensions: [number, number],
    _debugImage?: cv.Mat | null,
    kwargs?: Record<string, any>
  ): [cv.Mat, cv.Mat | null, cv.Mat | null] {
    const edgeContoursMap = kwargs?.edgeContoursMap;
    if (!edgeContoursMap) {
      throw new Error("DocRefineRectify requires 'edgeContoursMap' in kwargs");
    }

    // TODO: Port rectify helper from Python
    logger.warn('DocRefineRectify not yet implemented in TypeScript');

    throw new Error('DocRefineRectify requires rectify helper (not yet ported)');
  }
}

/**
 * Factory for creating warp strategy instances.
 *
 * Centralizes strategy creation and configuration.
 */
export class WarpStrategyFactory {
  private static strategies: Record<
    WarpMethodValue,
    new (...args: any[]) => WarpStrategy
  > = {
    [WarpMethod.PERSPECTIVE_TRANSFORM]: PerspectiveTransformStrategy,
    [WarpMethod.HOMOGRAPHY]: HomographyStrategy,
    [WarpMethod.REMAP_GRIDDATA]: GridDataRemapStrategy,
    [WarpMethod.DOC_REFINE]: DocRefineRectifyStrategy,
    [WarpMethod.WARP_AFFINE]: PerspectiveTransformStrategy, // Fallback to perspective
  };

  /**
   * Create a warp strategy by name.
   *
   * @param methodName - Strategy name (e.g., 'PERSPECTIVE_TRANSFORM')
   * @param config - Strategy-specific configuration
   * @returns Configured WarpStrategy instance
   * @throws Error if method_name is unknown
   */
  static create(methodName: WarpMethodValue, config: Record<string, any> = {}): WarpStrategy {
    const StrategyClass = this.strategies[methodName];

    if (!StrategyClass) {
      const available = Object.keys(this.strategies).join(', ');
      throw new Error(
        `Unknown warp method '${methodName}'. Available: ${available}`
      );
    }

    // Pass config as constructor arguments (need to match constructor signatures)
    if (methodName === WarpMethod.PERSPECTIVE_TRANSFORM || methodName === WarpMethod.WARP_AFFINE) {
      return new StrategyClass(config.interpolationFlag);
    } else if (methodName === WarpMethod.HOMOGRAPHY) {
      return new StrategyClass(
        config.interpolationFlag,
        config.useRansac,
        config.ransacThreshold
      );
    } else if (methodName === WarpMethod.REMAP_GRIDDATA) {
      return new StrategyClass(config.interpolationMethod);
    } else {
      return new StrategyClass();
    }
  }

  /**
   * Return list of available warp method names
   */
  static getAvailableMethods(): WarpMethodValue[] {
    return Object.keys(this.strategies) as WarpMethodValue[];
  }
}

