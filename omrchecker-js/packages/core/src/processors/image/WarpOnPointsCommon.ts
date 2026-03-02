/**
 * Abstract base class for warp-on-points image preprocessors.
 *
 * Port of Python: src/processors/image/WarpOnPointsCommon.py
 *
 * Subclasses must implement:
 *  - validateAndRemapOptionsSchema
 *  - prepareImageBeforeExtraction
 *  - extractControlDestinationPoints
 */

import cv from '@techstark/opencv-js';
import { WarpStrategy, WarpStrategyFactory } from './warp_strategies';
import { WarpMethod, WARP_METHOD_FLAG_VALUES } from '../constants';
import { MathUtils } from '../../utils/math';
import { ImageUtils } from '../../utils/image';
import { TemplateError } from '../../utils/exceptions';

export abstract class WarpOnPointsCommon {
  enableCropping: boolean;
  warpMethod: string;
  warpMethodFlag: number;
  warpStrategy: WarpStrategy;
  coloredOutputsEnabled: boolean;

  constructor(options: Record<string, any> = {}) {
    // Call abstract validate method — polymorphic dispatch reaches subclass implementation
    const parsed = this.validateAndRemapOptionsSchema(options);

    const tuningOptions: Record<string, any> =
      options.tuning_options ?? parsed.tuning_options ?? {};

    this.enableCropping = parsed.enable_cropping ?? options.enable_cropping ?? false;
    this.warpMethod =
      tuningOptions.warp_method ??
      (this.enableCropping ? WarpMethod.PERSPECTIVE_TRANSFORM : WarpMethod.HOMOGRAPHY);
    this.warpMethodFlag =
      WARP_METHOD_FLAG_VALUES[tuningOptions.warp_method_flag ?? 'INTER_LINEAR'] ?? 1;
    this.coloredOutputsEnabled = options.colored_outputs_enabled ?? false;
    this.warpStrategy = WarpStrategyFactory.create(this.warpMethod);
  }

  // ── Abstract methods ──────────────────────────────────────────────────────────

  /**
   * Validate and remap the options schema. Must be implemented by subclasses.
   */
  abstract validateAndRemapOptionsSchema(options: Record<string, any>): Record<string, any>;

  /**
   * Prepare (pre-process) the image before control-point extraction.
   */
  abstract prepareImageBeforeExtraction(image: cv.Mat): cv.Mat;

  /**
   * Extract control and destination point arrays from the prepared image.
   * Returns [controlPoints, destinationPoints, edgeContoursMap].
   */
  abstract extractControlDestinationPoints(
    image: cv.Mat,
    coloredImage: cv.Mat | null,
    filePath: string
  ): [number[][], number[][], any];

  // ── No-op hook (overridden in tests / subclasses) ─────────────────────────────

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  appendSaveImage(..._args: any[]): void {}

  // ── Public pipeline ───────────────────────────────────────────────────────────

  applyFilter(
    image: cv.Mat,
    coloredImage: cv.Mat | null,
    template: any,
    filePath: string
  ): [cv.Mat, cv.Mat | null, any] {
    const prepared = this.prepareImageBeforeExtraction(image);
    const [controlPts, destPts, edgeMap] = this.extractControlDestinationPoints(
      prepared,
      coloredImage,
      filePath
    );
    const [parsedCtrl, parsedDest, dims] = this._parseAndPreparePoints(
      prepared,
      controlPts,
      destPts
    );
    const [warpedImage, warpedColored] = this._applyWarpStrategy(
      image,
      coloredImage,
      parsedCtrl,
      parsedDest,
      dims,
      edgeMap
    );
    this.appendSaveImage('Warped Image', [4, 5, 6], warpedImage, warpedColored);
    return [warpedImage, warpedColored, template];
  }

  // ── Internal helpers ──────────────────────────────────────────────────────────

  _parseAndPreparePoints(
    image: cv.Mat,
    controlPoints: number[][],
    destinationPoints: number[][]
  ): [number[][], number[][], [number, number]] {
    // Deduplicate by control point (preserve first-seen order)
    const seen = new Map<string, number[]>();
    const uniqueCtrl: number[][] = [];
    const uniqueDest: number[][] = [];

    for (let i = 0; i < controlPoints.length; i++) {
      const key = JSON.stringify(controlPoints[i]);
      if (!seen.has(key)) {
        seen.set(key, destinationPoints[i]);
        uniqueCtrl.push(controlPoints[i]);
        uniqueDest.push(destinationPoints[i]);
      }
    }

    const dims = this._calculateWarpedDimensions(
      [image.cols, image.rows] as [number, number],
      uniqueDest
    );
    return [uniqueCtrl, uniqueDest, dims];
  }

  _calculateWarpedDimensions(
    defaultDims: [number, number],
    destinationPoints: number[][]
  ): [number, number] {
    if (!this.enableCropping) return defaultDims;

    const pts = destinationPoints as [number, number][];
    const { boundingBox, boxDimensions } = MathUtils.getBoundingBoxOfPoints(pts);

    // Shift all destination points so the bounding-box top-left becomes the origin
    const fromOrigin: [number, number] = [-boundingBox[0][0], -boundingBox[0][1]];
    const shifted = MathUtils.shiftPointsFromOrigin(fromOrigin, pts);

    for (let i = 0; i < destinationPoints.length; i++) {
      destinationPoints[i] = shifted[i];
    }

    return boxDimensions;
  }

  _applyWarpStrategy(
    image: cv.Mat,
    coloredImage: cv.Mat | null,
    controlPoints: number[][],
    destinationPoints: number[][],
    warpedDimensions: [number, number],
    _edgeContoursMap: any
  ): [cv.Mat, cv.Mat | null] {
    const [ctrl, dest, dims] = this._preparePointsForStrategy(
      controlPoints,
      destinationPoints,
      warpedDimensions
    );
    const coloredInput = this.coloredOutputsEnabled ? coloredImage : null;
    const result = this.warpStrategy.warpImage(image, coloredInput, ctrl, dest, dims);
    return [result.warpedGray, result.warpedColored];
  }

  _preparePointsForStrategy(
    controlPoints: number[][],
    destinationPoints: number[][],
    warpedDimensions: [number, number]
  ): [number[][], number[][], [number, number]] {
    if (this.warpMethod !== WarpMethod.PERSPECTIVE_TRANSFORM) {
      return [controlPoints, destinationPoints, warpedDimensions];
    }
    if (controlPoints.length !== 4) {
      throw new TemplateError(
        `Expected 4 control points for perspective transform, found ${controlPoints.length}.`
      );
    }
    const { rect: orderedCtrl } = MathUtils.orderFourPoints(
      controlPoints as [number, number][]
    );
    const [newDest, newDims] = ImageUtils.getCroppedWarpedRectanglePoints(
      orderedCtrl as number[][]
    );
    return [orderedCtrl as number[][], newDest, newDims];
  }
}
