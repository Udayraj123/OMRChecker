/**
 * Bubbles field interpretation drawing.
 *
 * TypeScript port of src/processors/detection/bubbles_threshold/interpretation_drawing.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import cv from '../../../utils/opencv';
import { AnswerMatcher, AnswerType } from '../../evaluation/AnswerMatcher';
import type { EvaluationConfigForSet } from '../../evaluation/EvaluationConfigForSet';
import {
  CLR_BLACK,
  CLR_GRAY,
  CLR_NEAR_BLACK,
  CLR_WHITE,
  TEXT_SIZE,
} from '../../../utils/constants';
import { DrawingUtils } from '../../../utils/drawing';
import { FieldInterpretationDrawing } from '../base/interpretationDrawing';
import type { InterpretationDrawing } from '../base/interpretation';
import type { BubblesFieldInterpretation } from './interpretation';
import type { BubbleInterpretation } from './interpretation';

/**
 * Drawing class for bubble field interpretations.
 *
 * Handles drawing of bubble interpretations with and without verdicts.
 */
export class BubblesFieldInterpretationDrawing
  extends FieldInterpretationDrawing
  implements InterpretationDrawing
{
  constructor(fieldInterpretation: BubblesFieldInterpretation) {
    super(fieldInterpretation);
  }

  /**
   * Draw field interpretation.
   *
   * @param markedImage - Image to draw on
   * @param imageType - Image type ('GRAYSCALE' or 'COLORED')
   * @param evaluationMeta - Evaluation metadata
   * @param evaluationConfigForResponse - Evaluation configuration
   */
  drawFieldInterpretation(
    markedImage: cv.Mat,
    imageType: 'GRAYSCALE' | 'COLORED',
    evaluationMeta?: any,
    evaluationConfigForResponse?: EvaluationConfigForSet
  ): void {
    const fieldLabel = this.field.fieldLabel;
    const bubbleInterpretations = (this.fieldInterpretation as BubblesFieldInterpretation)
      .bubbleInterpretations;
    const shouldDrawQuestionVerdicts =
      evaluationMeta !== undefined && evaluationConfigForResponse !== undefined;
    const questionHasVerdict =
      evaluationMeta !== undefined &&
      fieldLabel in evaluationMeta.questions_meta;

    if (
      shouldDrawQuestionVerdicts &&
      questionHasVerdict &&
      evaluationConfigForResponse?.drawQuestionVerdicts?.enabled
    ) {
      const questionMeta = evaluationMeta.questions_meta[fieldLabel];
      // Draw answer key items
      BubblesFieldInterpretationDrawing.drawBubblesAndDetectionsWithVerdicts(
        markedImage,
        imageType,
        bubbleInterpretations,
        questionMeta,
        evaluationConfigForResponse
      );
    } else {
      BubblesFieldInterpretationDrawing.drawBubblesAndDetectionsWithoutVerdicts(
        markedImage,
        bubbleInterpretations,
        evaluationConfigForResponse
      );
    }
  }

  /**
   * Draw bubbles and detections with verdicts.
   *
   * @param markedImage - Image to draw on
   * @param imageType - Image type
   * @param bubbleInterpretations - Bubble interpretations
   * @param questionMeta - Question metadata
   * @param evaluationConfigForResponse - Evaluation configuration
   */
  static drawBubblesAndDetectionsWithVerdicts(
    markedImage: cv.Mat,
    imageType: 'GRAYSCALE' | 'COLORED',
    bubbleInterpretations: BubbleInterpretation[],
    questionMeta: any,
    evaluationConfigForResponse: EvaluationConfigForSet
  ): void {
    for (const bubbleInterpretation of bubbleInterpretations) {
      BubblesFieldInterpretationDrawing.drawUnitBubbleInterpretationWithVerdicts(
        bubbleInterpretation,
        markedImage,
        evaluationConfigForResponse,
        questionMeta,
        imageType,
        1 / 12
      );
    }

    if (evaluationConfigForResponse.drawAnswerGroups?.enabled) {
      BubblesFieldInterpretationDrawing.drawAnswerGroupsForBubbles(
        markedImage,
        imageType,
        questionMeta,
        bubbleInterpretations,
        evaluationConfigForResponse
      );
    }
  }

  /**
   * Draw bubbles and detections without verdicts.
   *
   * @param markedImage - Image to draw on
   * @param bubbleInterpretations - Bubble interpretations
   * @param evaluationConfigForResponse - Evaluation configuration
   */
  static drawBubblesAndDetectionsWithoutVerdicts(
    markedImage: cv.Mat,
    bubbleInterpretations: BubbleInterpretation[],
    evaluationConfigForResponse?: EvaluationConfigForSet
  ): void {
    for (const bubbleInterpretation of bubbleInterpretations) {
      const bubble = bubbleInterpretation.itemReference;
      const bubbleDimensions = bubble.dimensions;
      const shiftedPosition = tuple(
        bubble.getShiftedPosition()
      ) as [number, number];
      const bubbleValue = String(bubble.bubbleValue);

      if (bubbleInterpretation.isAttempted) {
        DrawingUtils.drawBox(
          markedImage,
          shiftedPosition,
          bubbleDimensions,
          CLR_GRAY,
          'BOX_FILLED',
          1 / 12
        );
        if (
          // Note: this mimics the default true behaviour for draw_detected_bubble_texts
          evaluationConfigForResponse === undefined ||
          evaluationConfigForResponse.drawDetectedBubbleTexts?.enabled
        ) {
          DrawingUtils.drawText(
            markedImage,
            bubbleValue,
            shiftedPosition,
            TEXT_SIZE,
            int(1 + 3.5 * TEXT_SIZE),
            false,
            CLR_NEAR_BLACK
          );
        }
      } else {
        DrawingUtils.drawBox(
          markedImage,
          shiftedPosition,
          bubbleDimensions,
          undefined,
          'BOX_HOLLOW',
          1 / 10
        );
      }
    }
  }

  /**
   * Draw unit bubble interpretation with verdicts.
   *
   * @param bubbleInterpretation - Bubble interpretation
   * @param markedImage - Image to draw on
   * @param evaluationConfigForResponse - Evaluation configuration
   * @param questionMeta - Question metadata
   * @param imageType - Image type
   * @param thicknessFactor - Thickness factor
   */
  static drawUnitBubbleInterpretationWithVerdicts(
    bubbleInterpretation: BubbleInterpretation,
    markedImage: cv.Mat,
    evaluationConfigForResponse: EvaluationConfigForSet,
    questionMeta: any,
    imageType: 'GRAYSCALE' | 'COLORED',
    _thicknessFactor: number = 1 / 12
  ): void {
    const bonusType = questionMeta.bonus_type;

    const bubble = bubbleInterpretation.itemReference;
    const bubbleDimensions = bubble.dimensions;
    const shiftedPosition = tuple(
      bubble.getShiftedPosition()
    ) as [number, number];
    const bubbleValue = String(bubble.bubbleValue);

    // TODO: support for customLabels may change this logic

    // Enhanced bounding box for expected answer:
    if (AnswerMatcher.isPartOfSomeAnswer(questionMeta, bubbleValue)) {
      DrawingUtils.drawBox(
        markedImage,
        shiftedPosition,
        bubbleDimensions,
        CLR_BLACK,
        'BOX_HOLLOW',
        0
      );
    }

    // Filled box in case of marked bubble or bonus case
    if (bubbleInterpretation.isAttempted || bonusType !== undefined) {
      const [verdictSymbol, verdictColor, verdictSymbolColor, thicknessFactorResult] =
        evaluationConfigForResponse.getEvaluationMetaForQuestion(
          questionMeta,
          bubbleInterpretation.isAttempted,
          imageType
        );

      // Bounding box for marked bubble or bonus bubble
      if (verdictColor !== '' && typeof verdictColor !== 'string') {
        const [position, positionDiagonal] = DrawingUtils.drawBox(
          markedImage,
          shiftedPosition,
          bubbleDimensions,
          verdictColor,
          'BOX_FILLED',
          thicknessFactorResult
        );

        // Symbol for the marked bubble or bonus bubble
        if (verdictSymbol !== '') {
          const symbolColorTuple =
            typeof verdictSymbolColor === 'string' ? CLR_BLACK : verdictSymbolColor;
          DrawingUtils.drawSymbol(
            markedImage,
            verdictSymbol,
            position,
            positionDiagonal,
            symbolColorTuple
          );
        }
      }

      // Symbol of the field value for marked bubble
      if (
        bubbleInterpretation.isAttempted &&
        evaluationConfigForResponse.drawDetectedBubbleTexts?.enabled
      ) {
        DrawingUtils.drawText(
          markedImage,
          bubbleValue,
          shiftedPosition,
          TEXT_SIZE,
          int(1 + 3.5 * TEXT_SIZE),
          false,
          CLR_NEAR_BLACK
        );
      }
    } else {
      DrawingUtils.drawBox(
        markedImage,
        shiftedPosition,
        bubbleDimensions,
        undefined,
        'BOX_HOLLOW',
        1 / 10
      );
    }
  }

  /**
   * Draw answer groups for bubbles.
   *
   * @param markedImage - Image to draw on
   * @param imageType - Image type
   * @param questionMeta - Question metadata
   * @param bubbleInterpretations - Bubble interpretations
   * @param evaluationConfigForResponse - Evaluation configuration
   */
  static drawAnswerGroupsForBubbles(
    markedImage: cv.Mat,
    imageType: 'GRAYSCALE' | 'COLORED',
    questionMeta: any,
    bubbleInterpretations: BubbleInterpretation[],
    evaluationConfigForResponse: EvaluationConfigForSet
  ): void {
    // Note: currently draw_answer_groups is limited for questions with upto 4 values
    const answerType = questionMeta.answer_type;
    if (answerType === AnswerType.STANDARD) {
      return;
    }
    const boxEdges = ['TOP', 'RIGHT', 'BOTTOM', 'LEFT'];
    let colorSequence = evaluationConfigForResponse.drawAnswerGroups?.color_sequence || [];
    if (imageType === 'GRAYSCALE') {
      colorSequence = new Array(colorSequence.length).fill(CLR_WHITE);
    }

    for (const bubbleInterpretation of bubbleInterpretations) {
      const bubble = bubbleInterpretation.itemReference;
      const bubbleDimensions = bubble.dimensions;
      const shiftedPosition = tuple(
        bubble.getShiftedPosition()
      ) as [number, number];
      const bubbleValue = String(bubble.bubbleValue);
      const matchedGroups = AnswerMatcher.getMatchedAnswerGroups(
        questionMeta,
        bubbleValue
      );
      for (const answerIndex of matchedGroups) {
        const boxEdge = boxEdges[answerIndex % 4] as 'TOP' | 'RIGHT' | 'BOTTOM' | 'LEFT';
        const colorValue = colorSequence[answerIndex % 4];
        const color =
          typeof colorValue === 'string'
            ? ([0, 0, 0] as [number, number, number])
            : (colorValue as [number, number, number]);
        DrawingUtils.drawGroup(
          markedImage,
          shiftedPosition,
          bubbleDimensions,
          boxEdge,
          color
        );
      }
    }
  }
}

// Helper functions to match Python behavior
function tuple(arr: number[] | [number, number]): [number, number] {
  if (Array.isArray(arr) && arr.length >= 2) {
    return [arr[0], arr[1]];
  }
  return [0, 0];
}

function int(value: number): number {
  return Math.floor(value);
}

