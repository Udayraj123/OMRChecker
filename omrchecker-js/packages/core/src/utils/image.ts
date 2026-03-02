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
import { MathUtils } from './math';

/**
 * Image utility functions for OpenCV.js operations
 */
export class ImageUtils {
  /**
   * Compute warped rectangle points for a perspective-cropped region.
   *
   * Given four ordered corner points [tl, tr, br, bl], computes the destination
   * rectangle dimensions that preserve the maximum width and height, and returns
   * the axis-aligned destination points starting from the origin.
   *
   * Port of Python ImageUtils.get_cropped_warped_rectangle_points.
   *
   * @param orderedCorners - Four points [tl, tr, br, bl] as [x, y] arrays
   * @returns Tuple of [warpedPoints, [maxWidth, maxHeight]]
   */
  static getCroppedWarpedRectanglePoints(
    orderedCorners: number[][]
  ): [number[][], [number, number]] {
    const [tl, tr, br, bl] = orderedCorners;
    const maxWidth = Math.max(
      Math.floor(MathUtils.distance(tr as [number, number], tl as [number, number])),
      Math.floor(MathUtils.distance(br as [number, number], bl as [number, number]))
    );
    const maxHeight = Math.max(
      Math.floor(MathUtils.distance(tr as [number, number], br as [number, number])),
      Math.floor(MathUtils.distance(tl as [number, number], bl as [number, number]))
    );
    const warpedPoints: number[][] = [
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1],
    ];
    return [warpedPoints, [maxWidth, maxHeight]];
  }

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
   * Pad an image symmetrically from the center with a constant border value.
   *
   * Port of Python ImageUtils.pad_image_from_center.
   *
   * @param image - Input image Mat
   * @param paddingWidth - Number of pixels to pad on each side horizontally
   * @param paddingHeight - Number of pixels to pad on each side vertically (default: 0)
   * @param value - Border fill value (default: 255 = white)
   * @returns Object containing paddedImage Mat and padRange [top, bottom, left, right]
   *          padRange indices into the padded image where the original image was placed.
   *          The caller is responsible for deleting the returned paddedImage.
   */
  static padImageFromCenter(
    image: cv.Mat,
    paddingWidth: number,
    paddingHeight: number = 0,
    value: number = 255
  ): { paddedImage: cv.Mat; padRange: [number, number, number, number] } {
    const padRange: [number, number, number, number] = [
      paddingHeight,
      paddingHeight + image.rows,
      paddingWidth,
      paddingWidth + image.cols,
    ];
    const paddedImage = new cv.Mat();
    cv.copyMakeBorder(
      image,
      paddedImage,
      paddingHeight,
      paddingHeight,
      paddingWidth,
      paddingWidth,
      cv.BORDER_CONSTANT,
      new cv.Scalar(value, value, value, value)
    );
    return { paddedImage, padRange };
  }

  /**
   * Apply gamma correction to an image.
   *
   * Builds a lookup table mapping each pixel intensity i → ((i/255)^(1/gamma))*255,
   * then applies it via cv.LUT.
   *
   * Port of Python ImageUtils.adjust_gamma.
   *
   * @param image - Input grayscale image Mat (CV_8UC1)
   * @param gamma - Gamma value (default: 1.0, >1 brightens, <1 darkens)
   * @returns Gamma-corrected image Mat (caller must delete)
   */
  static adjustGamma(image: cv.Mat, gamma: number = 1.0): cv.Mat {
    const invGamma = 1.0 / gamma;
    const tableData = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
      tableData[i] = Math.round(Math.pow(i / 255.0, invGamma) * 255);
    }
    const lutMat = cv.matFromArray(1, 256, cv.CV_8UC1, Array.from(tableData));
    const result = new cv.Mat();
    cv.LUT(image, lutMat, result);
    lutMat.delete();
    return result;
  }

  /**
   * Split a contour's boundary points into four edge groups based on proximity
   * to each side of a quadrilateral defined by four corner points.
   *
   * The patch corners are first ordered [tl, tr, br, bl] via MathUtils.orderFourPoints.
   * Each source contour point is assigned to the nearest edge (TOP, RIGHT, BOTTOM, LEFT)
   * using point-to-segment distance. Corner points are inserted at the start and end of
   * each edge's list, and the list is reversed if needed to maintain clockwise order.
   *
   * Port of Python ImageUtils.split_patch_contour_on_corners.
   *
   * @param patchCorners - Four [x, y] corner points of the bounding patch
   * @param sourceContour - Array of [x, y] points from the detected contour boundary
   * @returns Object with orderedCorners [tl, tr, br, bl] and edgeContoursMap
   *          keyed by 'TOP' | 'RIGHT' | 'BOTTOM' | 'LEFT'
   */
  static splitPatchContourOnCorners(
    patchCorners: [number, number][],
    sourceContour: [number, number][]
  ): {
    orderedCorners: [number, number][];
    edgeContoursMap: Record<string, [number, number][]>;
  } {
    const { rect } = MathUtils.orderFourPoints(patchCorners);
    const orderedCorners = Array.from(rect) as [number, number][];
    // orderedCorners = [tl, tr, br, bl]

    const edgeTypes = ['TOP', 'RIGHT', 'BOTTOM', 'LEFT'];
    const edgeContoursMap: Record<string, [number, number][]> = {
      TOP: [],
      RIGHT: [],
      BOTTOM: [],
      LEFT: [],
    };

    // Point-to-line-segment distance helper
    function distToSegment(
      px: number,
      py: number,
      ax: number,
      ay: number,
      bx: number,
      by: number
    ): number {
      const dx = bx - ax;
      const dy = by - ay;
      const lenSq = dx * dx + dy * dy;
      if (lenSq === 0) return Math.hypot(px - ax, py - ay);
      let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
      t = Math.max(0, Math.min(1, t));
      return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
    }

    // Assign each source contour point to the nearest edge
    for (const pt of sourceContour) {
      let minDist = Infinity;
      let nearestEdge = 'TOP';
      for (let i = 0; i < 4; i++) {
        const [ax, ay] = orderedCorners[i];
        const [bx, by] = orderedCorners[(i + 1) % 4];
        const d = distToSegment(pt[0], pt[1], ax, ay, bx, by);
        if (d < minDist) {
          minDist = d;
          nearestEdge = edgeTypes[i];
        }
      }
      edgeContoursMap[nearestEdge].push(pt);
    }

    // Ensure clockwise order and add corner points at start/end of each edge
    for (let i = 0; i < 4; i++) {
      const edgeType = edgeTypes[i];
      const startPt = orderedCorners[i];
      const endPt = orderedCorners[(i + 1) % 4];
      const edgeContour = edgeContoursMap[edgeType];

      if (edgeContour.length > 0) {
        const distToFirst = MathUtils.distance(startPt, edgeContour[0]);
        const distToLast = MathUtils.distance(startPt, edgeContour[edgeContour.length - 1]);
        if (distToLast < distToFirst) {
          edgeContour.reverse();
        }
      }
      edgeContoursMap[edgeType].unshift(startPt);
      edgeContoursMap[edgeType].push(endPt);
    }

    return { orderedCorners, edgeContoursMap };
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
