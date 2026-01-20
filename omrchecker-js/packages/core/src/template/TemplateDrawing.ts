/**
 * Template drawing classes.
 *
 * TypeScript port of src/processors/layout/template_drawing.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import * as cv from '@techstark/opencv-js';
import { MARKED_TEMPLATE_TRANSPARENCY } from '../utils/constants';
import { DrawingUtils } from '../utils/drawing';
import { ImageUtils } from '../utils/image';
import { InteractionUtils } from '../utils/InteractionUtils';
import type { TemplateLayoutData } from './TemplateLoader';
import type { FieldInterpretation } from '../processors/detection/base/interpretation';

/**
 * Template drawing class.
 *
 * Wrapper for template drawing operations.
 */
export class TemplateDrawing {
  public template: TemplateLayoutData;

  constructor(template: TemplateLayoutData) {
    this.template = template;
  }

  /**
   * Draw template layout.
   *
   * @param grayImage - Grayscale image
   * @param coloredImage - Colored image
   * @param config - Configuration object
   * @param args - Additional arguments
   * @param kwargs - Additional keyword arguments
   * @returns Tuple of [final_marked, colored_final_marked]
   */
  drawTemplateLayout(
    grayImage: cv.Mat,
    coloredImage: cv.Mat,
    config: any,
    ...args: any[]
  ): [cv.Mat, cv.Mat] {
    const template = this.template;
    // TODO: extract field_id_to_interpretation from template itself
    return TemplateDrawingUtils.drawTemplateLayout(
      grayImage,
      coloredImage,
      template,
      config,
      ...args
    );
  }

  /**
   * Draw only field blocks.
   *
   * @param image - Image to draw on
   * @param shifted - Whether to use shifted positions
   * @param shouldCopy - Whether to copy the image before drawing
   * @param thickness - Line thickness
   * @param border - Border thickness
   * @returns Marked image
   */
  drawOnlyFieldBlocks(
    image: cv.Mat,
    shifted: boolean = true,
    shouldCopy: boolean = true,
    thickness: number = 3,
    border: number = 3
  ): cv.Mat {
    const template = this.template;
    return TemplateDrawing.drawFieldBlocksLayoutUtil(
      template,
      image,
      shifted,
      shouldCopy,
      thickness,
      border
    );
  }

  /**
   * Static utility to draw field blocks layout.
   *
   * @param template - Template layout
   * @param image - Image to draw on
   * @param shifted - Whether to use shifted positions
   * @param shouldCopy - Whether to copy the image before drawing
   * @param thickness - Line thickness
   * @param border - Border thickness
   * @returns Marked image
   */
  static drawFieldBlocksLayoutUtil(
    template: TemplateLayoutData,
    image: cv.Mat,
    shifted: boolean = true,
    shouldCopy: boolean = true,
    thickness: number = 3,
    border: number = 3
  ): cv.Mat {
    const markedImage = shouldCopy ? image.clone() : image;
    for (const fieldBlock of template.fieldBlocks) {
      fieldBlock.drawing.drawFieldBlock(markedImage, shifted, thickness, border);
    }
    return markedImage;
  }
}

/**
 * Template drawing utilities.
 *
 * Static utility class for template drawing operations.
 */
export class TemplateDrawingUtils {
  /**
   * Draw template layout.
   *
   * @param grayImage - Grayscale image
   * @param coloredImage - Colored image
   * @param template - Template layout
   * @param config - Configuration object
   * @param args - Additional arguments
   * @param kwargs - Additional keyword arguments
   * @returns Tuple of [final_marked, colored_final_marked]
   */
  static drawTemplateLayout(
    grayImage: cv.Mat,
    coloredImage: cv.Mat,
    template: TemplateLayoutData,
    config: any,
    ...args: any[]
  ): [cv.Mat, cv.Mat] {
    const finalMarked = TemplateDrawingUtils.drawTemplateLayoutUtil(
      grayImage,
      'GRAYSCALE',
      template,
      config,
      ...args
    );

    let coloredFinalMarked = coloredImage;
    if (config.outputs?.colored_outputs_enabled) {
      coloredFinalMarked = TemplateDrawingUtils.drawTemplateLayoutUtil(
        coloredFinalMarked,
        'COLORED',
        template,
        config,
        ...args
      );
    }

    if (config.outputs?.show_image_level >= 1) {
      // TODO: Implement pause and resize_to_height when InteractionUtils.show supports them
      InteractionUtils.show('Final Marked Template', finalMarked, {
        title: 'Final Marked Template',
        resizeToFit: true,
      });
      if (config.outputs?.colored_outputs_enabled) {
        InteractionUtils.show('Final Marked Template (Colored)', coloredFinalMarked, {
          title: 'Final Marked Template (Colored)',
          resizeToFit: true,
        });
      }
    }

    // Save images if saveImageOps is provided
    if (template.saveImageOps) {
      const keys = Array.from({ length: 7 }, (_, i) => i + 1); // range(1, 7)
      template.saveImageOps.appendSaveImage(
        'Marked Template',
        keys,
        finalMarked,
        coloredFinalMarked
      );
    }

    return [finalMarked, coloredFinalMarked];
  }

  /**
   * Draw template layout utility.
   *
   * @param image - Image to draw on
   * @param imageType - Image type ('GRAYSCALE' or 'COLORED')
   * @param template - Template layout
   * @param config - Configuration object
   * @param fieldIdToInterpretation - Map of field ID to interpretation
   * @param evaluationMeta - Evaluation metadata
   * @param evaluationConfigForResponse - Evaluation configuration
   * @param shifted - Whether to use shifted positions
   * @param border - Border thickness
   * @returns Marked image
   */
  static drawTemplateLayoutUtil(
    image: cv.Mat,
    imageType: 'GRAYSCALE' | 'COLORED',
    template: TemplateLayoutData,
    config: any,
    fieldIdToInterpretation?: Record<string, FieldInterpretation>,
    evaluationMeta?: any,
    evaluationConfigForResponse?: any,
    shifted: boolean = false,
    border: number = -1
  ): cv.Mat {
    const [resized] = ImageUtils.resizeToDimensions(
      template.templateDimensions,
      image
    );
    let markedImage = resized;

    const transparentLayer = markedImage.clone();

    if (!fieldIdToInterpretation) {
      // Draw only field blocks
      const templateDrawing = new TemplateDrawing(template);
      return templateDrawing.drawOnlyFieldBlocks(
        markedImage,
        shifted,
        false,
        3,
        border
      );
    }

    if (config.outputs?.saveImageLevel >= 1) {
      // Create a copy of the marked image for saving
      const markedImageCopy = markedImage.clone();

      // Draw marked bubbles without evaluation meta
      TemplateDrawingUtils.drawAllFields(
        markedImageCopy,
        imageType,
        template,
        fieldIdToInterpretation,
        undefined, // evaluationMeta
        undefined // evaluationConfigForResponse
      );

      // Save images if saveImageOps is provided
      if (template.saveImageOps) {
        const keys = Array.from({ length: 6 }, (_, i) => i + 2); // range(2, 7)
        if (imageType === 'GRAYSCALE') {
          template.saveImageOps.appendSaveImage(
            'Marked Image',
            keys,
            markedImageCopy,
            undefined
          );
        } else {
          template.saveImageOps.appendSaveImage(
            'Marked Image',
            keys,
            undefined,
            markedImageCopy
          );
        }
      }
    }

    markedImage = TemplateDrawingUtils.drawAllFields(
      markedImage,
      imageType,
      template,
      fieldIdToInterpretation,
      evaluationMeta,
      evaluationConfigForResponse
    );

    // Draw evaluation summary
    if (evaluationMeta) {
      markedImage = TemplateDrawingUtils.drawEvaluationSummary(
        markedImage,
        evaluationMeta,
        evaluationConfigForResponse
      );
    }

    // Translucent overlay
    cv.addWeighted(
      markedImage,
      MARKED_TEMPLATE_TRANSPARENCY,
      transparentLayer,
      1 - MARKED_TEMPLATE_TRANSPARENCY,
      0,
      markedImage
    );

    return markedImage;
  }

  /**
   * Draw all fields.
   *
   * @param markedImage - Image to draw on
   * @param imageType - Image type ('GRAYSCALE' or 'COLORED')
   * @param template - Template layout
   * @param fieldIdToInterpretation - Map of field ID to interpretation
   * @param evaluationMeta - Evaluation metadata
   * @param evaluationConfigForResponse - Evaluation configuration
   * @returns Marked image
   */
  static drawAllFields(
    markedImage: cv.Mat,
    imageType: 'GRAYSCALE' | 'COLORED',
    template: TemplateLayoutData,
    fieldIdToInterpretation: Record<string, FieldInterpretation>,
    evaluationMeta?: any,
    evaluationConfigForResponse?: any
  ): cv.Mat {
    for (const field of template.allFields) {
      const fieldInterpretation = fieldIdToInterpretation[field.id];
      if (fieldInterpretation?.drawing) {
        fieldInterpretation.drawing.drawFieldInterpretation(
          markedImage,
          imageType,
          evaluationMeta,
          evaluationConfigForResponse
        );
      }
    }
    return markedImage;
  }

  /**
   * Draw evaluation summary.
   *
   * @param markedImage - Image to draw on
   * @param evaluationMeta - Evaluation metadata
   * @param evaluationConfigForResponse - Evaluation configuration
   * @returns Marked image
   */
  static drawEvaluationSummary(
    markedImage: cv.Mat,
    evaluationMeta: any,
    evaluationConfigForResponse?: any
  ): cv.Mat {
    if (!evaluationConfigForResponse) {
      return markedImage;
    }

    // Draw answers summary
    if (evaluationConfigForResponse.drawAnswersSummary?.enabled) {
      const formattedAnswersSummary =
        evaluationConfigForResponse.getFormattedAnswersSummary();
      const [text, position, size, thickness] = formattedAnswersSummary;
      DrawingUtils.drawText(
        markedImage,
        text,
        position,
        size,
        thickness,
        false, // centered
        undefined // color - use default
      );
    }

    // Draw score
    if (evaluationConfigForResponse.drawScore?.enabled) {
      const formattedScore = evaluationConfigForResponse.getFormattedScore(
        evaluationMeta?.score || 0
      );
      const [text, position, size, thickness] = formattedScore;
      DrawingUtils.drawText(
        markedImage,
        text,
        position,
        size,
        thickness,
        false, // centered
        undefined // color - use default
      );
    }

    return markedImage;
  }
}

