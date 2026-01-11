/**
 * Image manipulation utilities for OMRChecker
 *
 * TypeScript port of src/utils/image.py
 * Uses DRY patterns with core methods delegating to multiple public APIs
 * All OpenCV conversions use matUtils for consistency
 */

import cv from '@techstark/opencv-js';
import { MatUtils } from './opencv/matUtils';
import { CLR_WHITE, type ColorTuple } from './constants';

/**
 * Image utility class with DRY patterns
 * Core methods are private and reused by multiple public methods
 */
export class ImageUtils {
  // ==================== RESIZE OPERATIONS (DRY Pattern) ====================

  /**
   * DRY: Core resize logic (used by 4 public methods)
   * @private
   */
  private static resizeCore(
    mat: cv.Mat,
    width: number,
    height: number
  ): cv.Mat {
    if (MatUtils.dimensionsMatch(mat, width, height)) {
      return mat.clone();
    }

    const dst = new cv.Mat();
    cv.resize(mat, dst, MatUtils.toSize([width, height]));
    return dst;
  }

  /**
   * Resize a single image (calculates missing dimension)
   */
  static resizeSingle(
    mat: cv.Mat,
    width?: number,
    height?: number
  ): cv.Mat {
    const [h, w] = [mat.rows, mat.cols];

    if (!width && !height) {
      return mat.clone();
    }

    let finalWidth = width;
    let finalHeight = height;

    if (!finalHeight && finalWidth) {
      finalHeight = Math.floor((h * finalWidth) / w);
    } else if (!finalWidth && finalHeight) {
      finalWidth = Math.floor((w * finalHeight) / h);
    }

    return this.resizeCore(mat, finalWidth!, finalHeight!);
  }

  /**
   * Resize multiple images
   */
  static resizeMultiple(
    mats: cv.Mat[],
    width?: number,
    height?: number
  ): cv.Mat[] {
    return mats.map((mat) => this.resizeSingle(mat, width, height));
  }

  /**
   * Resize to match shape [height, width]
   */
  static resizeToShape(shape: [number, number], ...mats: cv.Mat[]): cv.Mat[] {
    const [h, w] = shape;
    return this.resizeMultiple(mats, w, h);
  }

  /**
   * Resize to match dimensions [width, height]
   */
  static resizeToDimensions(
    dims: [number, number],
    ...mats: cv.Mat[]
  ): cv.Mat[] {
    const [w, h] = dims;
    return this.resizeMultiple(mats, w, h);
  }

  // ==================== NORMALIZATION OPERATIONS (DRY Pattern) ====================

  /**
   * DRY: Core normalize logic (used by 2 methods)
   * @private
   */
  private static normalizeCore(
    mat: cv.Mat,
    alpha: number,
    beta: number,
    normType: number
  ): cv.Mat {
    const dst = new cv.Mat();
    cv.normalize(mat, dst, alpha, beta, normType);
    return dst;
  }

  /**
   * Normalize a single image
   */
  static normalizeSingle(
    mat: cv.Mat,
    alpha = 0,
    beta = 255,
    normType = cv.NORM_MINMAX
  ): cv.Mat {
    return this.normalizeCore(mat, alpha, beta, normType);
  }

  /**
   * Normalize multiple images
   */
  static normalize(
    mats: cv.Mat[],
    alpha = 0,
    beta = 255,
    normType = cv.NORM_MINMAX
  ): cv.Mat[] {
    return mats.map((mat) => this.normalizeCore(mat, alpha, beta, normType));
  }

  // ==================== PADDING OPERATIONS (DRY Pattern) ====================

  /**
   * DRY: Core padding logic (used by 5 methods)
   * @private
   */
  private static padCore(
    mat: cv.Mat,
    top: number,
    bottom: number,
    left: number,
    right: number,
    value: ColorTuple
  ): cv.Mat {
    const dst = new cv.Mat();
    cv.copyMakeBorder(
      mat,
      dst,
      top,
      bottom,
      left,
      right,
      cv.BORDER_CONSTANT,
      MatUtils.toScalar(value)
    );
    return dst;
  }

  /**
   * Pad image to a specific height
   */
  static padImageToHeight(
    mat: cv.Mat,
    maxHeight: number,
    value: ColorTuple = CLR_WHITE
  ): cv.Mat {
    const padding = Math.max(0, maxHeight - mat.rows);
    return this.padCore(mat, 0, padding, 0, 0, value);
  }

  /**
   * Pad image to a specific width
   */
  static padImageToWidth(
    mat: cv.Mat,
    maxWidth: number,
    value: ColorTuple = CLR_WHITE
  ): cv.Mat {
    const padding = Math.max(0, maxWidth - mat.cols);
    return this.padCore(mat, 0, 0, 0, padding, value);
  }

  /**
   * Pad image from center (equal padding on opposite sides)
   */
  static padImageFromCenter(
    mat: cv.Mat,
    paddingWidth: number,
    paddingHeight = 0,
    value: ColorTuple = CLR_WHITE
  ): cv.Mat {
    const top = Math.floor(paddingHeight / 2);
    const bottom = paddingHeight - top;
    const left = Math.floor(paddingWidth / 2);
    const right = paddingWidth - left;
    return this.padCore(mat, top, bottom, left, right, value);
  }

  /**
   * Pad images to match the largest dimensions
   */
  static padImagesToMax(
    images: cv.Mat[],
    value: ColorTuple = CLR_WHITE
  ): cv.Mat[] {
    if (images.length === 0) return [];

    const maxHeight = Math.max(...images.map((img) => img.rows));
    const maxWidth = Math.max(...images.map((img) => img.cols));

    return images.map((img) => {
      const padHeight = maxHeight - img.rows;
      const padWidth = maxWidth - img.cols;
      return this.padCore(img, 0, padHeight, 0, padWidth, value);
    });
  }

  /**
   * Pad image uniformly on all sides
   */
  static padImageUniform(
    mat: cv.Mat,
    padding: number,
    value: ColorTuple = CLR_WHITE
  ): cv.Mat {
    return this.padCore(mat, padding, padding, padding, padding, value);
  }

  // ==================== STACKING OPERATIONS ====================

  /**
   * Stack images horizontally
   */
  static stackImagesHorizontal(images: cv.Mat[]): cv.Mat {
    if (images.length === 0) {
      throw new Error('Cannot stack empty array of images');
    }
    if (images.length === 1) {
      return images[0].clone();
    }

    // Pad to same height
    const paddedImages = this.padImagesToMax(images);

    // Create MatVector for hconcat
    const matVector = new cv.MatVector();
    paddedImages.forEach((img) => matVector.push_back(img));

    // Horizontal concatenation
    const dst = new cv.Mat();
    cv.hconcat(matVector, dst);

    // Clean up
    matVector.delete();
    paddedImages.forEach((img) => img.delete());

    return dst;
  }

  /**
   * Stack images vertically
   */
  static stackImagesVertical(images: cv.Mat[]): cv.Mat {
    if (images.length === 0) {
      throw new Error('Cannot stack empty array of images');
    }
    if (images.length === 1) {
      return images[0].clone();
    }

    // Pad to same width
    const maxWidth = Math.max(...images.map((img) => img.cols));
    const paddedImages = images.map((img) => this.padImageToWidth(img, maxWidth));

    // Create MatVector for vconcat
    const matVector = new cv.MatVector();
    paddedImages.forEach((img) => matVector.push_back(img));

    // Vertical concatenation
    const dst = new cv.Mat();
    cv.vconcat(matVector, dst);

    // Clean up
    matVector.delete();
    paddedImages.forEach((img) => img.delete());

    return dst;
  }

  // ==================== COLOR OPERATIONS ====================

  /**
   * Convert grayscale to BGR
   */
  static grayToBGR(mat: cv.Mat): cv.Mat {
    if (mat.channels() === 3) {
      return mat.clone();
    }

    const dst = new cv.Mat();
    cv.cvtColor(mat, dst, cv.COLOR_GRAY2BGR);
    return dst;
  }

  /**
   * Convert BGR to grayscale
   */
  static bgrToGray(mat: cv.Mat): cv.Mat {
    if (mat.channels() === 1) {
      return mat.clone();
    }

    const dst = new cv.Mat();
    cv.cvtColor(mat, dst, cv.COLOR_BGR2GRAY);
    return dst;
  }

  // ==================== UTILITY OPERATIONS ====================

  /**
   * Get image dimensions as [width, height]
   */
  static getDimensions(mat: cv.Mat): [number, number] {
    return [mat.cols, mat.rows];
  }

  /**
   * Get image shape as [height, width, channels]
   */
  static getShape(mat: cv.Mat): [number, number, number] {
    return [mat.rows, mat.cols, mat.channels()];
  }

  /**
   * Check if image is grayscale
   */
  static isGrayscale(mat: cv.Mat): boolean {
    return mat.channels() === 1;
  }

  /**
   * Clone an image
   */
  static clone(mat: cv.Mat): cv.Mat {
    return mat.clone();
  }
}

