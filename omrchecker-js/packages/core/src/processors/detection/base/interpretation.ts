/**
 * Interpretation base classes.
 *
 * TypeScript port of src/processors/detection/base/interpretation.py
 * Defines base classes for field interpretation results.
 */

import type { Field } from '../../layout/field/base';
import { TextDetection } from './detection';

/**
 * Base class for single interpretation result.
 *
 * Wraps a TextDetection result with interpretation metadata.
 */
export class BaseInterpretation {
  public textDetection: TextDetection | null;
  public isAttempted: boolean;
  public detectedText: string;

  constructor(textDetection: TextDetection | null) {
    this.textDetection = textDetection;
    this.isAttempted = textDetection !== null;
    this.detectedText = this.isAttempted && textDetection ? textDetection.detectedText || '' : '';
  }

  /**
   * Get the interpreted value.
   *
   * @returns Detected text value
   */
  getValue(): string {
    return this.detectedText;
  }
}

/**
 * Drawing instance interface (forward declaration).
 */
export interface InterpretationDrawing {
  // Stub for now, can be extended when needed
}

/**
 * Abstract base class for field interpretation.
 *
 * Manages interpretation results for a field, including confidence metrics
 * and multiple interpretation results (for multi-marking scenarios).
 */
export abstract class FieldInterpretation {
  public tuningConfig: Record<string, unknown>;
  public field: Field;
  public isAttempted: boolean | null = null;
  public emptyValue: string;
  public fieldLevelConfidenceMetrics: Record<string, unknown> = {};
  public interpretations: BaseInterpretation[] = [];
  public drawing: InterpretationDrawing;

  constructor(
    tuningConfig: Record<string, unknown>,
    field: Field,
    fileLevelDetectionAggregates: unknown,
    fileLevelInterpretationAggregates: unknown
  ) {
    this.tuningConfig = tuningConfig;
    this.field = field;
    this.emptyValue = field.emptyValue;

    // TODO: make get_drawing_instance fetch singleton classes?
    this.drawing = this.getDrawingInstance();

    this.runInterpretation(
      field,
      fileLevelDetectionAggregates,
      fileLevelInterpretationAggregates
    );
  }

  /**
   * Abstract method to get drawing instance.
   * Must be implemented by subclasses.
   */
  abstract getDrawingInstance(): InterpretationDrawing;

  /**
   * Abstract method to run interpretation.
   * Must be implemented by subclasses.
   *
   * @param field - Field to interpret
   * @param fileLevelDetectionAggregates - File-level detection aggregates
   * @param fileLevelInterpretationAggregates - File-level interpretation aggregates
   */
  abstract runInterpretation(
    field: Field,
    fileLevelDetectionAggregates: unknown,
    fileLevelInterpretationAggregates: unknown
  ): void;

  /**
   * Abstract method to get field interpretation string.
   * Must be implemented by subclasses.
   *
   * @returns Final interpretation string
   */
  abstract getFieldInterpretationString(): string;

  /**
   * Get field-level confidence metrics.
   *
   * @returns Confidence metrics
   */
  getFieldLevelConfidenceMetrics(): Record<string, unknown> {
    return this.fieldLevelConfidenceMetrics;
  }

  /**
   * Insert/merge additional field-level confidence metrics.
   *
   * @param nextFieldLevelConfidenceMetrics - Additional metrics to merge
   */
  insertFieldLevelConfidenceMetrics(
    nextFieldLevelConfidenceMetrics: Record<string, unknown>
  ): void {
    this.fieldLevelConfidenceMetrics = {
      ...this.fieldLevelConfidenceMetrics,
      ...nextFieldLevelConfidenceMetrics,
    };
  }
}

