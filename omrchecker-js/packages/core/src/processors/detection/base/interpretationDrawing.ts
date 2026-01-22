/**
 * Base interpretation drawing class.
 *
 * TypeScript port of src/processors/detection/base/interpretation_drawing.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import cv from '../../../utils/opencv';
import type { FieldInterpretation } from './interpretation';

/**
 * Abstract base class for field interpretation drawing.
 *
 * Provides interface for drawing field interpretations on marked images.
 */
export abstract class FieldInterpretationDrawing {
  public fieldInterpretation: FieldInterpretation;
  public tuningConfig: Record<string, unknown>;
  public field: FieldInterpretation['field'];

  constructor(fieldInterpretation: FieldInterpretation) {
    this.fieldInterpretation = fieldInterpretation;
    this.tuningConfig = fieldInterpretation.tuningConfig;
    this.field = fieldInterpretation.field;
  }

  /**
   * Abstract method to draw field interpretation.
   *
   * Must be implemented by subclasses.
   *
   * @param markedImage - Image to draw on
   * @param imageType - Image type ('GRAYSCALE' or 'COLORED')
   * @param evaluationMeta - Evaluation metadata
   * @param evaluationConfigForResponse - Evaluation configuration
   */
  abstract drawFieldInterpretation(
    markedImage: cv.Mat,
    imageType: 'GRAYSCALE' | 'COLORED',
    evaluationMeta?: any,
    evaluationConfigForResponse?: any
  ): void;
}

