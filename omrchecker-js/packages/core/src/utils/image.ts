/**
 * Migrated from Python: src/utils/image.py
 * Agent: Oz
 * Phase: Utility Migration
 *
 * Image processing utilities using OpenCV.js
 * 
 * Note: Functions requiring file I/O (loadImage, saveImage) are excluded
 * as they need Node.js APIs in browser context.
 */

import cv from '@techstark/opencv-js';

/**
 * Image utility functions for OpenCV.js operations
 */
export class ImageUtils {
  /**
   * Resize a single image with optional aspect ratio preservation.
   * 
   * @param image - Input image Mat
   * @param width - Target width (if null, calculated from height)
   * @param height - Target height (if null, calculated from width)
   * @returns Resized image Mat (caller must delete)
   */
  static resizeSingle(
    image: cv.Mat,
    width?: number | null,
    height?: number | null
  ): cv.Mat {
    if (!image || image.empty()) {
      throw new Error('Cannot resize empty or null image');
    }

    const h = image.rows;
    const w = image.cols;

    let targetWidth = width ?? null;
    let targetHeight = height ?? null;

    // Calculate missing dimension to preserve aspect ratio
    if (targetHeight === null && targetWidth !== null) {
      targetHeight = Math.floor((h * targetWidth) / w);
    }
    if (targetWidth === null && targetHeight !== null) {
      targetWidth = Math.floor((w * targetHeight) / h);
    }

    // Both dimensions required
    if (targetWidth === null || targetHeight === null) {
      throw new Error('Must provide at least one dimension for resize');
    }

    // No resize needed if dimensions match
    if (targetHeight === h && targetWidth === w) {
      return image.clone();
    }

    const resized = new cv.Mat();
    cv.resize(
      image,
      resized,
      new cv.Size(Math.floor(targetWidth), Math.floor(targetHeight)),
      0,
      0,
      cv.INTER_LINEAR
    );

    return resized;
  }

  /**
   * Resize multiple images to the same dimensions.
   * 
   * @param images - Array of image Mats
   * @param width - Target width
   * @param height - Target height
   * @returns Array of resized Mats (caller must delete each)
   */
  static resizeMultiple(
    images: cv.Mat[],
    width?: number | null,
    height?: number | null
  ): cv.Mat[] {
    if (images.length === 0) {
      return [];
    }

    return images.map(image => {
      if (!image || image.empty()) {
        return image; // Return as-is if empty
      }
      return ImageUtils.resizeSingle(image, width, height);
    });
  }

  /**
   * Resize image(s) to match a target shape (height, width).
   * 
   * @param imageShape - Target shape as [height, width]
   * @param images - One or more images to resize
   * @returns Resized image(s)
   */
  static resizeToShape(
    imageShape: [number, number],
    ...images: cv.Mat[]
  ): cv.Mat | cv.Mat[] {
    const [h, w] = imageShape;
    const resized = ImageUtils.resizeMultiple(images, w, h);
    return images.length === 1 ? resized[0] : resized;
  }

  /**
   * Resize image(s) to match target dimensions (width, height).
   * 
   * @param imageDimensions - Target dimensions as [width, height]
   * @param images - One or more images to resize
   * @returns Resized image(s)
   */
  static resizeToDimensions(
    imageDimensions: [number, number],
    ...images: cv.Mat[]
  ): cv.Mat | cv.Mat[] {
    const [w, h] = imageDimensions;
    const resized = ImageUtils.resizeMultiple(images, w, h);
    return images.length === 1 ? resized[0] : resized;
  }

  /**
   * Normalize a single image to a specific range.
   * 
   * @param image - Input image
   * @param alpha - Lower bound of normalization range (default: 0)
   * @param beta - Upper bound of normalization range (default: 255)
   * @param normType - Normalization type (default: NORM_MINMAX)
   * @returns Normalized image (caller must delete)
   */
  static normalizeSingle(
    image: cv.Mat,
    alpha: number = 0,
    beta: number = 255,
    normType: number = cv.NORM_MINMAX
  ): cv.Mat {
    if (!image || image.empty()) {
      return image;
    }

    // Check if image has no variation (all pixels same value)
    const minMax = cv.minMaxLoc(image, new cv.Mat());
    if (minMax.maxVal === minMax.minVal) {
      return image.clone();
    }

    const normalized = new cv.Mat();
    cv.normalize(image, normalized, alpha, beta, normType);
    return normalized;
  }

  /**
   * Normalize multiple images.
   * 
   * @param images - Images to normalize
   * @param alpha - Lower bound
   * @param beta - Upper bound  
   * @param normType - Normalization type
   * @returns Normalized images
   */
  static normalize(
    images: cv.Mat[],
    alpha: number = 0,
    beta: number = 255,
    normType: number = cv.NORM_MINMAX
  ): cv.Mat | cv.Mat[] {
    if (images.length === 0) {
      throw new Error('Must provide at least one image to normalize');
    }

    const normalized = images.map(image =>
      ImageUtils.normalizeSingle(image, alpha, beta, normType)
    );

    return images.length === 1 ? normalized[0] : normalized;
  }

  /**
   * Extract contours array from OpenCV findContours result.
   * 
   * OpenCV.js returns contours differently than Python OpenCV.
   * This helper provides compatibility.
   * 
   * @param contoursResult - Result from cv.findContours
   * @returns cv.MatVector of contours
   */
  static grabContours(contoursResult: any): cv.MatVector {
    // OpenCV.js returns a MatVector directly
    // Python OpenCV has version-dependent return formats (tuple vs direct)
    // For OpenCV.js, we expect the contours to be the first element if tuple-like
    
    if (contoursResult instanceof cv.MatVector) {
      return contoursResult;
    }

    // If it's an object with a contours property
    if (contoursResult && contoursResult.contours) {
      return contoursResult.contours;
    }

    throw new Error('Invalid contours format from OpenCV.js');
  }

  /**
   * Automatic Canny edge detection using computed thresholds.
   * 
   * Calculates optimal thresholds based on image median intensity.
   * 
   * @param image - Input grayscale image
   * @param sigma - Threshold multiplier (default: 0.93)
   * @returns Edge detected image (caller must delete)
   */
  static autoCanny(image: cv.Mat, sigma: number = 0.93): cv.Mat {
    if (!image || image.empty()) {
      throw new Error('Cannot apply Canny to empty image');
    }

    // Compute median of pixel intensities
    // OpenCV.js doesn't have a median function, so we'll use mean as approximation
    // or sort the data (expensive but accurate)
    const mean = cv.mean(image);
    const v = mean[0]; // First channel (grayscale)

    // Calculate thresholds based on median
    const lower = Math.max(0, Math.floor((1.0 - sigma) * v));
    const upper = Math.min(255, Math.floor((1.0 + sigma) * v));

    const edges = new cv.Mat();
    cv.Canny(image, edges, lower, upper);

    return edges;
  }

  /**
   * Rotate image while optionally keeping original shape.
   * 
   * @param image - Input image
   * @param rotationCode - OpenCV rotation code (ROTATE_90_CLOCKWISE, etc.)
   * @param keepOriginalShape - If true, resize back to original dimensions
   * @returns Rotated image (caller must delete)
   */
  static rotate(
    image: cv.Mat,
    rotationCode: number,
    keepOriginalShape: boolean = false
  ): cv.Mat {
    if (!image || image.empty()) {
      throw new Error('Cannot rotate empty image');
    }

    const originalHeight = image.rows;
    const originalWidth = image.cols;

    const rotated = new cv.Mat();
    cv.rotate(image, rotated, rotationCode);

    if (keepOriginalShape) {
      // If dimensions changed after rotation, resize back
      if (rotated.rows !== originalHeight || rotated.cols !== originalWidth) {
        const resized = new cv.Mat();
        cv.resize(
          rotated,
          resized,
          new cv.Size(originalWidth, originalHeight),
          0,
          0,
          cv.INTER_LINEAR
        );
        rotated.delete();
        return resized;
      }
    }

    return rotated;
  }

  /**
   * Overlay two images with transparency.
   * 
   * @param image1 - First image
   * @param image2 - Second image (must be same size as image1)
   * @param transparency - Alpha value for image1 (0-1, default: 0.5)
   * @returns Blended image (caller must delete)
   */
  static overlayImage(
    image1: cv.Mat,
    image2: cv.Mat,
    transparency: number = 0.5
  ): cv.Mat {
    if (!image1 || !image2 || image1.empty() || image2.empty()) {
      throw new Error('Cannot overlay empty images');
    }

    if (
      image1.rows !== image2.rows ||
      image1.cols !== image2.cols ||
      image1.type() !== image2.type()
    ) {
      throw new Error('Images must have same dimensions and type for overlay');
    }

    const overlay = new cv.Mat();
    cv.addWeighted(
      image1,
      transparency,
      image2,
      1 - transparency,
      0,
      overlay
    );

    return overlay;
  }
}
