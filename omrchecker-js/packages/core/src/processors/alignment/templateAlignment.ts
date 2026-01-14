/**
 * Template alignment utilities.
 *
 * TypeScript port of src/processors/alignment/template_alignment.py
 * Implements browser-compatible alignment using ORB/AKAZE features.
 *
 * Alignment Methods:
 * 1. Phase Correlation - Fast global shift detection
 * 2. ORB/AKAZE Feature Matching - Robust feature-based alignment
 * 3. K-Nearest Interpolation - Per-bubble coordinate adjustment
 */

import cv from '@techstark/opencv-js';
import { ImageUtils } from '../../utils/ImageUtils';
import { MathUtils } from '../../utils/math';
import { Logger } from '../../utils/logger';

const logger = new Logger('TemplateAlignment');

// Alignment configuration constants
const MIN_MATCH_COUNT = 10;
const LOWE_RATIO_THRESHOLD = 0.75;
const MAX_RANSAC_REPROJ_THRESHOLD = 5.0;

/**
 * Result of template alignment operation
 */
export interface AlignmentResult {
  grayImage: cv.Mat;
  coloredImage: cv.Mat;
  template: any;
}

/**
 * Apply template alignment to images.
 *
 * This function:
 * 1. Resizes images to template dimensions
 * 2. Iterates through field blocks and computes alignment shifts
 * 3. Returns aligned images and updated template
 *
 * Note: This is a simplified version. Full implementation would include:
 * - k-nearest interpolation for warp coordinate computation
 * - SIFT/ORB feature matching
 * - Phase correlation for shift detection
 *
 * @param grayImage - Grayscale input image
 * @param coloredImage - Colored input image
 * @param template - Template configuration with alignment data
 * @param config - Tuning configuration
 * @returns Aligned images and updated template
 */
export function applyTemplateAlignment(
  grayImage: cv.Mat,
  coloredImage: cv.Mat,
  template: any,
  _config: any
): AlignmentResult {
  logger.debug('Starting template alignment');

  // Get alignment configuration
  const alignment = template.alignment;
  const templateMargins = alignment?.margins || alignment?.margins || {
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  };
  const templateMaxDisplacement = alignment?.maxDisplacement ||
                                  alignment?.max_displacement ||
                                  0;

  // Get pre-processed alignment images
  const grayAlignmentImage = alignment?.grayAlignmentImage ||
                            alignment?.gray_alignment_image;
  const coloredAlignmentImage = alignment?.coloredAlignmentImage ||
                               alignment?.colored_alignment_image;

  if (!grayAlignmentImage) {
    logger.debug('No alignment image found, returning original images');
    return { grayImage, coloredImage, template };
  }

  // Get template dimensions
  const templateDimensions = template.templateDimensions ||
                            template.template_dimensions;

  if (!templateDimensions) {
    logger.warn('No template dimensions found, skipping resize');
    return { grayImage, coloredImage, template };
  }

  // Resize all images to template dimensions
  // Note: resize also creates a copy
  const resizedImages = ImageUtils.resizeToDimensions(
    templateDimensions,
    grayImage,
    coloredImage,
    grayAlignmentImage,
    coloredAlignmentImage
  );

  let alignedGrayImage: cv.Mat;
  let alignedColoredImage: cv.Mat;
  let _alignedGrayAlignment: cv.Mat;
  let _alignedColoredAlignment: cv.Mat;

  if (Array.isArray(resizedImages)) {
    [alignedGrayImage, alignedColoredImage, _alignedGrayAlignment, _alignedColoredAlignment] =
      resizedImages as cv.Mat[];
  } else {
    alignedGrayImage = resizedImages as cv.Mat;
    alignedColoredImage = coloredImage;
    _alignedGrayAlignment = grayAlignmentImage;
    _alignedColoredAlignment = coloredAlignmentImage;
  }

  // Suppress unused variable warnings - these will be used for advanced alignment algorithms
  void _alignedGrayAlignment;
  void _alignedColoredAlignment;

  // Get field blocks
  const fieldBlocks = template.fieldBlocks || template.field_blocks || [];

  // Iterate through field blocks and compute alignment
  for (const fieldBlock of fieldBlocks) {
    const fieldBlockName = fieldBlock.name;

    // Initialize shifts to zero
    fieldBlock.shifts = [0, 0];

    const boundingBoxOrigin = fieldBlock.boundingBoxOrigin ||
                             fieldBlock.bounding_box_origin;
    const boundingBoxDimensions = fieldBlock.boundingBoxDimensions ||
                                 fieldBlock.bounding_box_dimensions;
    const fieldBlockAlignment = fieldBlock.alignment || {};

    // Get margins and max displacement for this field block
    const margins = fieldBlockAlignment.margins || templateMargins;
    const maxDisplacement = fieldBlockAlignment.maxDisplacement ||
                           fieldBlockAlignment.max_displacement ||
                           templateMaxDisplacement;

    if (maxDisplacement === 0) {
      // Skip alignment computation for this field block if allowed displacement is zero
      logger.debug(`Skipping alignment for ${fieldBlockName} (maxDisplacement = 0)`);
      continue;
    }

    if (!boundingBoxOrigin || !boundingBoxDimensions) {
      logger.warn(`Missing bounding box data for field block: ${fieldBlockName}`);
      continue;
    }

    // Compute zone boundaries (with margins)
    const zoneStart = [
      Math.floor(boundingBoxOrigin[0] - margins.left),
      Math.floor(boundingBoxOrigin[1] - margins.top),
    ];
    const zoneEnd = [
      Math.floor(boundingBoxOrigin[0] + margins.right + boundingBoxDimensions[0]),
      Math.floor(boundingBoxOrigin[1] + margins.bottom + boundingBoxDimensions[1]),
    ];

    logger.debug(
      `Processing field block: ${fieldBlockName}`,
      `Zone: [${zoneStart}] to [${zoneEnd}]`,
      `MaxDisplacement: ${maxDisplacement}`
    );

    // Extract zone from both images for alignment
    try {
      const blockGrayImage = alignedGrayImage.roi(
        new cv.Rect(zoneStart[0], zoneStart[1], zoneEnd[0] - zoneStart[0], zoneEnd[1] - zoneStart[1])
      );
      const blockGrayAlignment = _alignedGrayAlignment.roi(
        new cv.Rect(zoneStart[0], zoneStart[1], zoneEnd[0] - zoneStart[0], zoneEnd[1] - zoneStart[1])
      );

      // Method 1: Phase Correlation (fast, simple shifts)
      const phaseShifts = getPhaseCorrelationShifts(blockGrayAlignment, blockGrayImage);

      if (phaseShifts && maxDisplacement > 0) {
        const [xShift, yShift] = phaseShifts;

        // Check if shifts are within max displacement
        const shiftMagnitude = Math.sqrt(xShift * xShift + yShift * yShift);

        if (shiftMagnitude <= maxDisplacement) {
          fieldBlock.shifts = phaseShifts;
          logger.debug(`Applied phase correlation shifts for ${fieldBlockName}: [${xShift}, ${yShift}]`);
        } else {
          logger.warn(
            `Phase correlation shifts too large for ${fieldBlockName}: ` +
            `${shiftMagnitude.toFixed(1)} > ${maxDisplacement}`
          );

          // Method 2: Feature-based alignment (more robust)
          const featureMatches = getFeatureMatches(
            blockGrayAlignment,
            blockGrayImage,
            maxDisplacement,
            false // Use ORB (faster than AKAZE)
          );

          if (featureMatches && featureMatches.displacementPairs.length > 0) {
            // Compute average displacement from feature matches
            let avgX = 0;
            let avgY = 0;

            for (const pair of featureMatches.displacementPairs) {
              const [dest, src] = pair;
              avgX += src[0] - dest[0];
              avgY += src[1] - dest[1];
            }

            avgX /= featureMatches.displacementPairs.length;
            avgY /= featureMatches.displacementPairs.length;

            fieldBlock.shifts = [Math.round(avgX), Math.round(avgY)];

            logger.debug(
              `Applied feature-based shifts for ${fieldBlockName}: ` +
              `[${avgX.toFixed(1)}, ${avgY.toFixed(1)}] from ${featureMatches.displacementPairs.length} matches`
            );

            // Cleanup
            featureMatches.keypoints1.delete();
            featureMatches.keypoints2.delete();
            featureMatches.goodMatches.delete();
          } else {
            logger.warn(`No reliable alignment found for ${fieldBlockName}, keeping shifts at [0, 0]`);
          }
        }
      } else {
        logger.debug(`Phase correlation failed for ${fieldBlockName}, keeping shifts at [0, 0]`);
      }

      // Cleanup ROIs
      blockGrayImage.delete();
      blockGrayAlignment.delete();
    } catch (error) {
      logger.error(`Alignment failed for field block ${fieldBlockName}:`, error);
      // Keep shifts at [0, 0]
    }
  }

  logger.debug('Completed template alignment');

  return {
    grayImage: alignedGrayImage,
    coloredImage: alignedColoredImage,
    template,
  };
}

/**
 * Get phase correlation shifts between two images.
 *
 * Phase correlation is a fast method to detect translational shifts
 * between two similar images using FFT.
 *
 * @param alignmentImage - Reference image
 * @param grayImage - Input image to align
 * @returns [x_shift, y_shift] or null if detection fails
 */
export function getPhaseCorrelationShifts(
  alignmentImage: cv.Mat,
  grayImage: cv.Mat
): [number, number] | null {
  try {
    // Use OpenCV's phaseCorrelate function
    const hann = new cv.Mat();
    const shift = cv.phaseCorrelate(alignmentImage, grayImage, hann);

    hann.delete();

    const xShift = Math.round(shift.x);
    const yShift = Math.round(shift.y);

    logger.debug(`Phase correlation shifts: [${xShift}, ${yShift}]`);

    return [xShift, yShift];
  } catch (error) {
    logger.warn('Phase correlation failed:', error);
    return null;
  }
}

/**
 * Get feature-based alignment using ORB or AKAZE.
 *
 * SIFT is not available in OpenCV.js, so we use ORB (faster) or AKAZE (more accurate).
 * This function:
 * 1. Detects keypoints and computes descriptors
 * 2. Matches features between images
 * 3. Filters good matches using Lowe's ratio test
 * 4. Computes homography with RANSAC
 *
 * @param alignmentImage - Reference image
 * @param grayImage - Input image to align
 * @param maxDisplacement - Maximum allowed displacement for match filtering
 * @param useAKAZE - Use AKAZE instead of ORB (slower but more accurate)
 * @returns Object with matches, good matches, and displacement pairs
 */
export function getFeatureMatches(
  alignmentImage: cv.Mat,
  grayImage: cv.Mat,
  maxDisplacement: number,
  useAKAZE: boolean = false
): {
  keypoints1: cv.KeyPointVector;
  keypoints2: cv.KeyPointVector;
  goodMatches: cv.DMatchVector;
  displacementPairs: number[][][];
} | null {
  let detector: cv.ORB | cv.AKAZE | null = null;
  let descriptors1: cv.Mat | null = null;
  let descriptors2: cv.Mat | null = null;
  const keypoints1 = new cv.KeyPointVector();
  const keypoints2 = new cv.KeyPointVector();

  try {
    // Create feature detector
    if (useAKAZE) {
      detector = new cv.AKAZE();
      logger.debug('Using AKAZE feature detector');
    } else {
      detector = new cv.ORB();
      logger.debug('Using ORB feature detector');
    }

    // Detect keypoints and compute descriptors
    descriptors1 = new cv.Mat();
    descriptors2 = new cv.Mat();

    detector.detectAndCompute(alignmentImage, new cv.Mat(), keypoints1, descriptors1);
    detector.detectAndCompute(grayImage, new cv.Mat(), keypoints2, descriptors2);

    logger.debug(`Keypoints: ${keypoints1.size()} in alignment, ${keypoints2.size()} in gray`);

    if (keypoints1.size() < MIN_MATCH_COUNT || keypoints2.size() < MIN_MATCH_COUNT) {
      logger.warn('Not enough keypoints detected');
      return null;
    }

    // Match descriptors using BFMatcher
    const matcher = new cv.BFMatcher(
      useAKAZE ? cv.NORM_HAMMING : cv.NORM_HAMMING,
      false
    );
    const matches = new cv.DMatchVectorVector();
    matcher.knnMatch(descriptors1, descriptors2, matches, 2);

    // Apply Lowe's ratio test and displacement filter
    const goodMatches = new cv.DMatchVector();
    const displacementPairs: number[][][] = [];

    for (let i = 0; i < matches.size(); i++) {
      const match = matches.get(i);

      if (match.size() >= 2) {
        const m = match.get(0);
        const n = match.get(1);

        // Lowe's ratio test
        if (m.distance < LOWE_RATIO_THRESHOLD * n.distance) {
          const kp1 = keypoints1.get(m.queryIdx);
          const kp2 = keypoints2.get(m.trainIdx);

          const sourcePoint = [kp1.pt.x, kp1.pt.y];
          const destPoint = [kp2.pt.x, kp2.pt.y];

          // Check displacement constraint
          const displacement = MathUtils.distance(sourcePoint, destPoint);

          if (maxDisplacement === 0 || displacement <= maxDisplacement) {
            goodMatches.push_back(m);
            // Note: reversed for warping (dest -> source)
            displacementPairs.push([destPoint, sourcePoint]);
          }
        }
      }
    }

    logger.debug(`Good matches: ${goodMatches.size()} / ${matches.size()}`);

    // Cleanup
    matches.delete();
    matcher.delete();
    descriptors1?.delete();
    descriptors2?.delete();
    detector?.delete();

    if (goodMatches.size() < MIN_MATCH_COUNT) {
      logger.warn(`Not enough good matches: ${goodMatches.size()} < ${MIN_MATCH_COUNT}`);
      goodMatches.delete();
      keypoints1.delete();
      keypoints2.delete();
      return null;
    }

    return {
      keypoints1,
      keypoints2,
      goodMatches,
      displacementPairs,
    };
  } catch (error) {
    logger.error('Feature matching failed:', error);

    // Cleanup on error
    descriptors1?.delete();
    descriptors2?.delete();
    keypoints1.delete();
    keypoints2.delete();
    detector?.delete();

    return null;
  }
}

/**
 * Compute homography transform from matched features.
 *
 * @param keypoints1 - Keypoints from first image
 * @param keypoints2 - Keypoints from second image
 * @param goodMatches - Good matches between keypoints
 * @returns Homography matrix or null if computation fails
 */
export function computeHomography(
  keypoints1: cv.KeyPointVector,
  keypoints2: cv.KeyPointVector,
  goodMatches: cv.DMatchVector
): cv.Mat | null {
  try {
    // Extract matched point coordinates
    const srcPoints: number[] = [];
    const dstPoints: number[] = [];

    for (let i = 0; i < goodMatches.size(); i++) {
      const match = goodMatches.get(i);
      const kp1 = keypoints1.get(match.queryIdx);
      const kp2 = keypoints2.get(match.trainIdx);

      srcPoints.push(kp1.pt.x, kp1.pt.y);
      dstPoints.push(kp2.pt.x, kp2.pt.y);
    }

    // Create matrices
    const srcMat = cv.matFromArray(
      goodMatches.size(),
      1,
      cv.CV_32FC2,
      srcPoints
    );
    const dstMat = cv.matFromArray(
      goodMatches.size(),
      1,
      cv.CV_32FC2,
      dstPoints
    );

    // Compute homography with RANSAC
    const homography = cv.findHomography(
      srcMat,
      dstMat,
      cv.RANSAC,
      MAX_RANSAC_REPROJ_THRESHOLD
    );

    srcMat.delete();
    dstMat.delete();

    if (homography.empty()) {
      logger.warn('Failed to compute homography');
      homography.delete();
      return null;
    }

    logger.debug('Homography computed successfully');
    return homography;
  } catch (error) {
    logger.error('Homography computation failed:', error);
    return null;
  }
}

