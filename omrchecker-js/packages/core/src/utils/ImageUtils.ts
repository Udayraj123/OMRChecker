/**
 * Image processing utilities and wrappers over OpenCV functions.
 *
 * TypeScript port of src/utils/image.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import cv from './opencv';
import { Logger } from './logger';
import { MathUtils } from './math';

const logger = new Logger('ImageUtils');

/**
 * Static utility class for common image processing operations.
 */
export class ImageUtils {
  /**
   * Load an image from a File object or data URL.
   *
   * In browser environment, we work with File objects or base64 data URLs
   * instead of file paths.
   *
   * @param imageSource - File object, Blob, or data URL string
   * @param mode - Image mode: 0=grayscale, 1=color, -1=unchanged
   * @returns Promise resolving to loaded image as cv.Mat
   */
  static async loadImage(
    imageSource: File | Blob | string,
    mode: number = 0 // 0 = grayscale
  ): Promise<cv.Mat> {
    try {
      let imageData: string;

      if (typeof imageSource === 'string') {
        // Already a data URL
        imageData = imageSource;
      } else {
        // Convert File/Blob to data URL
        imageData = await this.fileToDataURL(imageSource);
      }

      // Load image using OpenCV
      return await this.loadImageFromDataURL(imageData, mode);
    } catch (error) {
      logger.error(`Failed to load image: ${error}`);
      throw new Error(`Failed to load image: ${error}`);
    }
  }

  /**
   * Convert File or Blob to data URL.
   */
  private static fileToDataURL(file: File | Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  /**
   * Load image from data URL using OpenCV.
   */
  private static loadImageFromDataURL(dataURL: string, mode: number): Promise<cv.Mat> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        try {
          const mat = cv.imread(img);
          if (mat.empty()) {
            reject(new Error('Loaded image is empty'));
            return;
          }

          // Convert to requested mode
          let result: cv.Mat;
          if (mode === 0) { // Grayscale
            result = new cv.Mat();
            cv.cvtColor(mat, result, cv.COLOR_RGBA2GRAY);
            mat.delete();
          } else if (mode === 1) { // Color
            result = new cv.Mat();
            cv.cvtColor(mat, result, cv.COLOR_RGBA2BGR);
            mat.delete();
          } else { // Unchanged
            result = mat;
          }

          resolve(result);
        } catch (error) {
          reject(error);
        }
      };
      img.onerror = () => reject(new Error('Failed to load image from data URL'));
      img.src = dataURL;
    });
  }

  /**
   * Read image utility for OMR processing.
   * Returns both grayscale and colored versions based on config.
   *
   * @param imageSource - Image source (File, Blob, or data URL)
   * @param coloredOutputsEnabled - Whether colored outputs are enabled
   * @returns Promise resolving to [grayImage, coloredImage]
   */
  static async readImageUtil(
    imageSource: File | Blob | string,
    coloredOutputsEnabled: boolean = false
  ): Promise<[cv.Mat, cv.Mat | null]> {
    if (coloredOutputsEnabled) {
      const coloredImage = await this.loadImage(imageSource, 1); // Color mode
      const grayImage = new cv.Mat();
      cv.cvtColor(coloredImage, grayImage, cv.COLOR_BGR2GRAY);
      return [grayImage, coloredImage];
    } else {
      const grayImage = await this.loadImage(imageSource, 0); // Grayscale mode
      return [grayImage, null];
    }
  }

  /**
   * Save/export image as data URL or blob.
   *
   * @param image - OpenCV Mat to save
   * @param format - Image format ('png', 'jpg')
   * @returns Data URL of the image
   */
  static saveImage(image: cv.Mat, format: string = 'png'): string {
    const canvas = document.createElement('canvas');
    cv.imshow(canvas, image);
    return canvas.toDataURL(`image/${format}`);
  }

  /**
   * Resize single image to specified dimensions.
   *
   * @param image - Image to resize
   * @param width - Target width (optional)
   * @param height - Target height (optional)
   * @returns Resized image
   */
  static resizeSingle(
    image: cv.Mat | null,
    width?: number,
    height?: number
  ): cv.Mat | null {
    if (!image || image.empty()) {
      return null;
    }

    const h = image.rows;
    const w = image.cols;

    // Calculate missing dimension if only one is provided
    let targetWidth = width;
    let targetHeight = height;

    if (targetHeight === undefined && targetWidth !== undefined) {
      targetHeight = Math.floor((h * targetWidth) / w);
    }
    if (targetWidth === undefined && targetHeight !== undefined) {
      targetWidth = Math.floor((w * targetHeight) / h);
    }

    // No resize needed if dimensions match
    if (targetHeight === h && targetWidth === w) {
      return image;
    }

    const resized = new cv.Mat();
    cv.resize(
      image,
      resized,
      new cv.Size(targetWidth!, targetHeight!),
      0,
      0,
      cv.INTER_AREA
    );

    return resized;
  }

  /**
   * Resize multiple images to the same dimensions.
   *
   * @param images - Array of images to resize
   * @param width - Target width
   * @param height - Target height
   * @returns Array of resized images or single image if only one input
   */
  static resizeMultiple(
    images: cv.Mat[],
    width?: number,
    height?: number
  ): cv.Mat[] | cv.Mat {
    if (images.length === 1) {
      return this.resizeSingle(images[0], width, height)!;
    }
    return images.map((img) => this.resizeSingle(img, width, height)!);
  }

  /**
   * Resize image(s) to match a specific shape [height, width].
   *
   * @param imageShape - Target shape as [height, width]
   * @param images - Images to resize
   * @returns Resized image(s)
   */
  static resizeToShape(imageShape: [number, number], ...images: cv.Mat[]): cv.Mat[] | cv.Mat {
    const [h, w] = imageShape;
    return this.resizeMultiple(images, w, h);
  }

  /**
   * Resize image(s) to specific dimensions [width, height].
   *
   * @param imageDimensions - Target dimensions as [width, height]
   * @param images - Images to resize
   * @returns Resized image(s)
   */
  static resizeToDimensions(
    imageDimensions: [number, number],
    ...images: cv.Mat[]
  ): cv.Mat[] | cv.Mat {
    const [w, h] = imageDimensions;
    return this.resizeMultiple(images, w, h);
  }

  /**
   * Rotate image by specified OpenCV rotation flag.
   *
   * @param image - Image to rotate
   * @param rotation - OpenCV rotation flag (ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE)
   * @param keepOriginalShape - If true, resize back to original dimensions
   * @returns Rotated image
   */
  static rotate(image: cv.Mat, rotation: cv.RotateFlags, keepOriginalShape: boolean = false): cv.Mat {
    const rotated = new cv.Mat();
    cv.rotate(image, rotated, rotation);

    if (keepOriginalShape) {
      const imageShape: [number, number] = [image.rows, image.cols];
      const resized = this.resizeToShape(imageShape, rotated) as cv.Mat;
      rotated.delete();
      return resized;
    }

    return rotated;
  }

  /**
   * Normalize image to specified range.
   *
   * @param image - Image to normalize
   * @param alpha - Minimum value (default: 0)
   * @param beta - Maximum value (default: 255)
   * @param normType - Normalization type (default: NORM_MINMAX)
   * @returns Normalized image
   */
  static normalizeSingle(
    image: cv.Mat | null,
    alpha: number = 0,
    beta: number = 255,
    normType: number = cv.NORM_MINMAX
  ): cv.Mat | null {
    if (!image || image.empty()) {
      return image;
    }

    // Check if image has constant values
    const minMaxResult = cv.minMaxLoc(image, new cv.Mat());
    if (minMaxResult.maxVal === minMaxResult.minVal) {
      return image;
    }

    const normalized = new cv.Mat();
    cv.normalize(image, normalized, alpha, beta, normType);
    return normalized;
  }

  /**
   * Normalize multiple images.
   *
   * @param images - Images to normalize
   * @param alpha - Minimum value
   * @param beta - Maximum value
   * @param normType - Normalization type
   * @returns Normalized images
   */
  static normalize(
    images: cv.Mat[],
    alpha: number = 0,
    beta: number = 255,
    normType: number = cv.NORM_MINMAX
  ): cv.Mat[] | cv.Mat {
    if (images.length === 1) {
      return this.normalizeSingle(images[0], alpha, beta, normType)!;
    }
    return images.map((img) => this.normalizeSingle(img, alpha, beta, normType)!);
  }

  /**
   * Apply automatic Canny edge detection with adaptive thresholds.
   *
   * @param image - Input image
   * @param sigma - Threshold adjustment factor (default: 0.93)
   * @returns Edge-detected image
   */
  static autoCanny(image: cv.Mat, sigma: number = 0.93): cv.Mat {
    // Calculate median of pixel intensities
    const pixels = [];
    for (let i = 0; i < image.rows; i++) {
      for (let j = 0; j < image.cols; j++) {
        pixels.push(image.ucharPtr(i, j)[0]);
      }
    }
    pixels.sort((a, b) => a - b);
    const median = pixels[Math.floor(pixels.length / 2)];

    // Calculate adaptive thresholds
    const lower = Math.max(0, Math.floor((1.0 - sigma) * median));
    const upper = Math.min(255, Math.floor((1.0 + sigma) * median));

    const edges = new cv.Mat();
    cv.Canny(image, edges, lower, upper);
    return edges;
  }

  /**
   * Adjust image gamma (brightness/contrast).
   *
   * @param image - Input image
   * @param gamma - Gamma value (< 1 = lighter, > 1 = darker)
   * @returns Gamma-adjusted image
   */
  static adjustGamma(image: cv.Mat, gamma: number = 1.0): cv.Mat {
    // Build lookup table for gamma correction
    // Uses inverse gamma (1/gamma) for correction formula
    // gamma > 1: lighter midtones (correction for dark display)
    // gamma < 1: darker midtones (correction for bright display)
    const invGamma = 1.0 / gamma;
    const table = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
      table[i] = Math.floor(Math.pow(i / 255.0, invGamma) * 255);
    }

    const lookupTable = new cv.Mat(1, 256, cv.CV_8UC1);
    lookupTable.data.set(table);

    const result = new cv.Mat();
    cv.LUT(image, lookupTable, result);

    lookupTable.delete();
    return result;
  }

  /**
   * Grab contours from findContours result (handles OpenCV version differences).
   *
   * @param contours - Result from cv.findContours
   * @returns Contours array
   */
  static grabContours(contours: any): cv.MatVector {
    // OpenCV.js typically returns the contours directly
    // This is mainly for compatibility with Python version
    return contours;
  }

  /**
   * Get padded horizontal stack of images.
   *
   * @param images - Array of images to stack horizontally
   * @returns Horizontally stacked image
   */
  static getPaddedHstack(images: cv.Mat[]): cv.Mat {
    const maxHeight = Math.max(...images.map((img) => img.rows));
    const paddedImages = images.map((img) => this.padImageToHeight(img, maxHeight));

    // Create horizontally stacked image
    const totalWidth = paddedImages.reduce((sum, img) => sum + img.cols, 0);
    const result = new cv.Mat(maxHeight, totalWidth, images[0].type());

    let xOffset = 0;
    for (const img of paddedImages) {
      const roi = result.roi(new cv.Rect(xOffset, 0, img.cols, img.rows));
      img.copyTo(roi);
      roi.delete();
      xOffset += img.cols;
    }

    // Cleanup padded images
    paddedImages.forEach((img, idx) => {
      if (img !== images[idx]) img.delete();
    });

    return result;
  }

  /**
   * Get padded vertical stack of images.
   *
   * @param images - Array of images to stack vertically
   * @returns Vertically stacked image
   */
  static getPaddedVstack(images: cv.Mat[]): cv.Mat {
    const maxWidth = Math.max(...images.map((img) => img.cols));
    const paddedImages = images.map((img) => this.padImageToWidth(img, maxWidth));

    // Create vertically stacked image
    const totalHeight = paddedImages.reduce((sum, img) => sum + img.rows, 0);
    const result = new cv.Mat(totalHeight, maxWidth, images[0].type());

    let yOffset = 0;
    for (const img of paddedImages) {
      const roi = result.roi(new cv.Rect(0, yOffset, img.cols, img.rows));
      img.copyTo(roi);
      roi.delete();
      yOffset += img.rows;
    }

    // Cleanup padded images
    paddedImages.forEach((img, idx) => {
      if (img !== images[idx]) img.delete();
    });

    return result;
  }

  /**
   * Pad image to specified height (adds padding at bottom).
   *
   * @param image - Image to pad
   * @param maxHeight - Target height
   * @param value - Padding color (default: white [255, 255, 255])
   * @returns Padded image
   */
  static padImageToHeight(image: cv.Mat, maxHeight: number, value: number[] = [255, 255, 255]): cv.Mat {
    if (image.rows >= maxHeight) {
      return image;
    }

    const padded = new cv.Mat();
    const borderValue = new cv.Scalar(...value);
    cv.copyMakeBorder(
      image,
      padded,
      0,
      maxHeight - image.rows,
      0,
      0,
      cv.BORDER_CONSTANT,
      borderValue
    );
    return padded;
  }

  /**
   * Pad image to specified width (adds padding on right side).
   *
   * @param image - Image to pad
   * @param maxWidth - Target width
   * @param value - Padding color (default: white [255, 255, 255])
   * @returns Padded image
   */
  static padImageToWidth(image: cv.Mat, maxWidth: number, value: number[] = [255, 255, 255]): cv.Mat {
    if (image.cols >= maxWidth) {
      return image;
    }

    const padded = new cv.Mat();
    const borderValue = new cv.Scalar(...value);
    cv.copyMakeBorder(
      image,
      padded,
      0,
      0,
      0,
      maxWidth - image.cols,
      cv.BORDER_CONSTANT,
      borderValue
    );
    return padded;
  }

  /**
   * Pad image from center with specified padding.
   *
   * @param image - Image to pad
   * @param paddingWidth - Horizontal padding
   * @param paddingHeight - Vertical padding (default: 0)
   * @param value - Padding color (default: 255)
   * @returns [Padded image, padding range]
   */
  static padImageFromCenter(
    image: cv.Mat,
    paddingWidth: number,
    paddingHeight: number = 0,
    value: number = 255
  ): [cv.Mat, number[]] {
    const inputHeight = image.rows;
    const inputWidth = image.cols;

    const padRange = [
      paddingHeight,
      paddingHeight + inputHeight,
      paddingWidth,
      paddingWidth + inputWidth,
    ];

    // Create white image
    const whiteImage = new cv.Mat(
      paddingHeight * 2 + inputHeight,
      paddingWidth * 2 + inputWidth,
      image.type(),
      new cv.Scalar(value)
    );

    // Copy original image to center
    const roi = whiteImage.roi(
      new cv.Rect(paddingWidth, paddingHeight, inputWidth, inputHeight)
    );
    image.copyTo(roi);
    roi.delete();

    return [whiteImage, padRange];
  }

  /**
   * Get control and destination points from a contour and destination line.
   *
   * Interpolates points along the contour and maps them to points along the destination line.
   * Used for line-based warping where we have an edge contour and want to align it with a straight edge.
   *
   * @param contour - Array of points along the detected edge
   * @param destinationLine - [start, end] points of the target line
   * @param maxPoints - Maximum number of points to use (null = use all)
   * @returns Tuple of [controlPoints, destinationPoints]
   */
  static getControlDestinationPointsFromContour(
    contour: [number, number][],
    destinationLine: [[number, number], [number, number]],
    maxPoints: number | null = null
  ): [[number, number][], [number, number][]] {
    if (contour.length === 0 || destinationLine.length < 2) {
      return [[], []];
    }

    // Sample points from contour if needed
    let sampledContour = contour;
    if (maxPoints !== null && contour.length > maxPoints) {
      sampledContour = this.samplePointsFromArray(contour, maxPoints);
    }

    // Interpolate destination points along the destination line
    const numPoints = sampledContour.length;
    const destinationPoints = this.interpolatePointsAlongLine(
      destinationLine[0],
      destinationLine[1],
      numPoints
    );

    return [sampledContour, destinationPoints];
  }

  /**
   * Sample N evenly-spaced points from an array.
   *
   * @param points - Array of points
   * @param count - Number of points to sample
   * @returns Sampled points
   */
  private static samplePointsFromArray(
    points: [number, number][],
    count: number
  ): [number, number][] {
    if (points.length <= count) {
      return points;
    }

    const step = (points.length - 1) / (count - 1);
    const sampled: [number, number][] = [];

    for (let i = 0; i < count; i++) {
      const index = Math.round(i * step);
      sampled.push(points[index]);
    }

    return sampled;
  }

  /**
   * Interpolate N points evenly along a line.
   *
   * @param start - Start point [x, y]
   * @param end - End point [x, y]
   * @param count - Number of points to generate
   * @returns Array of interpolated points
   */
  private static interpolatePointsAlongLine(
    start: [number, number],
    end: [number, number],
    count: number
  ): [number, number][] {
    const points: [number, number][] = [];

    for (let i = 0; i < count; i++) {
      const t = count > 1 ? i / (count - 1) : 0;
      points.push([
        Math.round(start[0] + t * (end[0] - start[0])),
        Math.round(start[1] + t * (end[1] - start[1])),
      ]);
    }

    return points;
  }

  /**
   * Clip rectangle zone to image bounds.
   *
   * @param rectangle - Rectangle as [[x1, y1], [x2, y2]]
   * @param image - Image to clip to
   * @returns Clipped rectangle
   */
  static clipZoneToImageBounds(
    rectangle: [[number, number], [number, number]],
    image: cv.Mat
  ): [[number, number], [number, number]] {
    const h = image.rows;
    const w = image.cols;
    let [zoneStart, zoneEnd] = rectangle;

    // Clip to image top-left
    zoneStart = [Math.max(0, zoneStart[0]), Math.max(0, zoneStart[1])];
    zoneEnd = [Math.max(0, zoneEnd[0]), Math.max(0, zoneEnd[1])];

    // Clip to image bottom-right
    zoneStart = [Math.min(w, zoneStart[0]), Math.min(h, zoneStart[1])];
    zoneEnd = [Math.min(w, zoneEnd[0]), Math.min(h, zoneEnd[1])];

    return [zoneStart as [number, number], zoneEnd as [number, number]];
  }

  /**
   * Overlay two images with transparency.
   *
   * @param image1 - First image
   * @param image2 - Second image
   * @param transparency - Transparency of first image (0-1, default: 0.5)
   * @returns Overlaid image
   */
  static overlayImage(image1: cv.Mat, image2: cv.Mat, transparency: number = 0.5): cv.Mat {
    const overlay = new cv.Mat();
    cv.addWeighted(image1, transparency, image2, 1 - transparency, 0, overlay);
    return overlay;
  }

  /**
   * Get cropped and warped rectangle points for perspective transform.
   *
   * @param orderedPageCorners - Four corners in order [tl, tr, br, bl]
   * @returns [warped points, dimensions]
   */
  static getCroppedWarpedRectanglePoints(
    orderedPageCorners: [number, number][]
  ): [[number, number][], [number, number]] {
    // Note: This utility would just find a good size ratio for the cropped image to look more realistic
    // but since we're anyway resizing the image, it doesn't make much sense to use these calculations
    const [tl, tr, br, bl] = orderedPageCorners;

    const lengthT = MathUtils.distance(tr, tl);
    const lengthB = MathUtils.distance(br, bl);
    const lengthR = MathUtils.distance(tr, br);
    const lengthL = MathUtils.distance(tl, bl);

    // Compute the width of the new image
    const maxWidth = Math.max(Math.floor(lengthT), Math.floor(lengthB));

    // Compute the height of the new image
    const maxHeight = Math.max(Math.floor(lengthR), Math.floor(lengthL));

    // Now that we have the dimensions of the new image, construct
    // the set of destination points to obtain a "birds eye view",
    // (i.e. top-down view) of the image
    const warpedPoints: [number, number][] = [
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1],
    ];

    const warpedBoxDimensions: [number, number] = [maxWidth, maxHeight];

    return [warpedPoints, warpedBoxDimensions];
  }
}

