/**
 * Warp strategies for image rectification and perspective correction.
 *
 * Provides an abstract WarpStrategy interface and concrete implementations:
 * - PerspectiveTransformStrategy: 4-point perspective warp using getPerspectiveTransform
 * - HomographyStrategy: Multi-point homography warp using findHomography
 * - GridDataRemapStrategy: Grid-data remap (browser fallback: perspective approximation)
 * - DocRefineRectifyStrategy: Document refinement (requires external rectify, not browser-available)
 *
 * Port of Python src/processors/image/warp_strategies.py
 */

import cv from '@techstark/opencv-js';

/**
 * Result returned by all warp strategy implementations.
 */
export interface WarpResult {
  warpedGray: cv.Mat;
  warpedColored: cv.Mat | null;
  warpedDebug: cv.Mat | null;
}

/**
 * Abstract base class for all warp strategies.
 */
export abstract class WarpStrategy {
  abstract warpImage(
    image: cv.Mat,
    coloredImage: cv.Mat | null,
    controlPoints: number[][],
    destinationPoints: number[][],
    warpedDimensions: [number, number],
    debugImage?: cv.Mat | null
  ): WarpResult;

  abstract getName(): string;
}

/**
 * Perspective transform strategy using getPerspectiveTransform.
 * Requires exactly 4 control points.
 */
export class PerspectiveTransformStrategy extends WarpStrategy {
  interpolationFlag: number;

  constructor(interpolationFlag?: number) {
    super();
    this.interpolationFlag = interpolationFlag ?? cv.INTER_LINEAR;
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
    debugImage?: cv.Mat | null
  ): WarpResult {
    if (controlPoints.length !== 4) {
      throw new Error(
        `PerspectiveTransform requires exactly 4 control points, got ${controlPoints.length}`
      );
    }

    const [w, h] = warpedDimensions;
    const controlMat = cv.matFromArray(4, 1, cv.CV_32FC2, controlPoints.flat());
    const destMat = cv.matFromArray(4, 1, cv.CV_32FC2, destinationPoints.flat());
    const M = cv.getPerspectiveTransform(controlMat, destMat);
    const dsize = new cv.Size(w, h);

    const warpedGray = new cv.Mat();
    cv.warpPerspective(image, warpedGray, M, dsize, this.interpolationFlag);

    let warpedColored: cv.Mat | null = null;
    if (coloredImage) {
      warpedColored = new cv.Mat();
      cv.warpPerspective(coloredImage, warpedColored, M, dsize, this.interpolationFlag);
    }

    let warpedDebug: cv.Mat | null = null;
    if (debugImage) {
      warpedDebug = new cv.Mat();
      cv.warpPerspective(debugImage, warpedDebug, M, dsize, this.interpolationFlag);
    }

    controlMat.delete();
    destMat.delete();
    M.delete();

    return { warpedGray, warpedColored, warpedDebug };
  }
}

/**
 * Options for HomographyStrategy constructor.
 */
export interface HomographyStrategyOptions {
  interpolationFlag?: number;
  useRansac?: boolean;
  ransacThreshold?: number;
}

/**
 * Homography strategy using findHomography.
 * Requires at least 4 control points. Supports RANSAC outlier rejection.
 */
export class HomographyStrategy extends WarpStrategy {
  interpolationFlag: number;
  useRansac: boolean;
  ransacThreshold: number;

  constructor(options?: HomographyStrategyOptions) {
    super();
    this.interpolationFlag = options?.interpolationFlag ?? cv.INTER_LINEAR;
    this.useRansac = options?.useRansac ?? false;
    this.ransacThreshold = options?.ransacThreshold ?? 3.0;
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
    debugImage?: cv.Mat | null
  ): WarpResult {
    if (controlPoints.length < 4) {
      throw new Error(
        `Homography requires at least 4 control points, got ${controlPoints.length}`
      );
    }

    const [w, h] = warpedDimensions;
    const n = controlPoints.length;
    const controlMat = cv.matFromArray(n, 1, cv.CV_32FC2, controlPoints.flat());
    const destMat = cv.matFromArray(n, 1, cv.CV_32FC2, destinationPoints.flat());

    // OpenCV.js findHomography returns the matrix directly
    const method = this.useRansac ? cv.RANSAC : 0;
    const M: cv.Mat = cv.findHomography(controlMat, destMat, method, this.ransacThreshold);

    if (!M || M.rows === 0) {
      controlMat.delete();
      destMat.delete();
      if (M) M.delete();
      throw new Error('Failed to compute homography matrix');
    }

    const dsize = new cv.Size(w, h);

    const warpedGray = new cv.Mat();
    cv.warpPerspective(image, warpedGray, M, dsize, this.interpolationFlag);

    let warpedColored: cv.Mat | null = null;
    if (coloredImage) {
      warpedColored = new cv.Mat();
      cv.warpPerspective(coloredImage, warpedColored, M, dsize, this.interpolationFlag);
    }

    let warpedDebug: cv.Mat | null = null;
    if (debugImage) {
      warpedDebug = new cv.Mat();
      cv.warpPerspective(debugImage, warpedDebug, M, dsize, this.interpolationFlag);
    }

    controlMat.delete();
    destMat.delete();
    M.delete();

    return { warpedGray, warpedColored, warpedDebug };
  }
}

/**
 * Grid data remap strategy.
 *
 * In Python this uses scipy.interpolate.griddata which is not available in browser.
 * Browser fallback: uses a perspective transform approximation with the first 4 points.
 * The output shape and dtype match the Python implementation.
 */
export class GridDataRemapStrategy extends WarpStrategy {
  interpolationMethod: string;

  constructor(interpolationMethod: string = 'cubic') {
    super();
    this.interpolationMethod = interpolationMethod;
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
    debugImage?: cv.Mat | null
  ): WarpResult {
    // scipy.interpolate.griddata is not available in browser.
    // Use first 4 points for perspective approximation as browser fallback.
    const ctrl4 = controlPoints.slice(0, 4);
    const dest4 = destinationPoints.slice(0, 4);
    const perspStrategy = new PerspectiveTransformStrategy(cv.INTER_LINEAR);
    return perspStrategy.warpImage(image, coloredImage, ctrl4, dest4, warpedDimensions, debugImage);
  }
}

/**
 * Document refinement rectification strategy.
 *
 * Requires an external `rectify` helper (src/processors/helpers/rectify) which
 * is not available in the browser environment. Throws in browser context.
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
    _debugImage?: cv.Mat | null
  ): WarpResult {
    throw new Error('DocRefineRectify is not available in browser environment');
  }
}

/**
 * Factory for creating WarpStrategy instances by method name.
 */
export class WarpStrategyFactory {
  private static readonly strategies: Record<
    string,
    new (...args: any[]) => WarpStrategy
  > = {
    PERSPECTIVE_TRANSFORM: PerspectiveTransformStrategy,
    HOMOGRAPHY: HomographyStrategy,
    REMAP_GRIDDATA: GridDataRemapStrategy,
    DOC_REFINE: DocRefineRectifyStrategy,
  };

  /**
   * Create a WarpStrategy by method name with optional configuration.
   *
   * @param methodName - One of: PERSPECTIVE_TRANSFORM, HOMOGRAPHY, REMAP_GRIDDATA, DOC_REFINE
   * @param config - Optional configuration object passed to the strategy constructor
   * @throws Error if the method name is unknown
   */
  static create(methodName: string, config?: Record<string, unknown>): WarpStrategy {
    const StrategyClass = this.strategies[methodName];
    if (!StrategyClass) {
      throw new Error(`Unknown warp method '${methodName}'.`);
    }
    if (config !== undefined) {
      return new StrategyClass(config);
    }
    return new StrategyClass();
  }

  /**
   * Returns the list of available warp method names.
   */
  static getAvailableMethods(): string[] {
    return Object.keys(this.strategies);
  }
}
