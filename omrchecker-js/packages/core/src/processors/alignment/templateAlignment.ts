/**
 * Template alignment utilities.
 *
 * TypeScript port of src/processors/alignment/template_alignment.py
 * Simplified version for browser use.
 *
 * Note: Full Python implementation includes k-nearest interpolation and
 * complex field block alignment. This is a simplified version that provides
 * basic alignment structure. Advanced alignment features can be added incrementally.
 */

import type * as cv from '@techstark/opencv-js';
import { ImageUtils } from '../../utils/ImageUtils';
import { Logger } from '../../utils/logger';

const logger = new Logger('TemplateAlignment');

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
  config: any
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
  let alignedGrayAlignment: cv.Mat;
  let alignedColoredAlignment: cv.Mat;

  if (Array.isArray(resizedImages)) {
    [alignedGrayImage, alignedColoredImage, alignedGrayAlignment, alignedColoredAlignment] =
      resizedImages as cv.Mat[];
  } else {
    alignedGrayImage = resizedImages as cv.Mat;
    alignedColoredImage = coloredImage;
    alignedGrayAlignment = grayAlignmentImage;
    alignedColoredAlignment = coloredAlignmentImage;
  }

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
      `Zone: [${zoneStart}] to [${zoneEnd}]`
    );

    // TODO: Implement advanced alignment algorithms:
    // - Method 1: Phase correlation shifts
    // - Method 2: SIFT/ORB feature matching
    // - Method 3: k-nearest interpolation (current Python implementation)
    // - Method 4: Per-field warping
    //
    // For now, shifts remain [0, 0] as initialized above.
    // This provides the structure for incremental implementation.
  }

  logger.debug('Completed template alignment');

  return {
    grayImage: alignedGrayImage,
    coloredImage: alignedColoredImage,
    template,
  };
}

/**
 * Get global alignment transform between two images.
 *
 * This is a placeholder for future implementation of global image alignment
 * using feature matching (SIFT, ORB, etc.) or phase correlation.
 *
 * @param sourceImage - Source image to align
 * @param targetImage - Target reference image
 * @returns Transform matrix or null if alignment fails
 */
export function getGlobalAlignmentTransform(
  sourceImage: cv.Mat,
  targetImage: cv.Mat
): cv.Mat | null {
  logger.debug('Global alignment transform - not yet implemented');
  // TODO: Implement using OpenCV.js feature detection and matching
  // - detectAndCompute with ORB or AKAZE (SIFT not available in OpenCV.js)
  // - Match descriptors
  // - Find homography with RANSAC
  // - Return transformation matrix
  return null;
}

