/**
 * Typed models for evaluation configuration.
 *
 * TypeScript port of src/schemas/models/evaluation.py
 * Maintains 1:1 correspondence with Python implementation.
 */

/**
 * Configuration for drawing score on output images.
 */
export interface DrawScoreConfig {
  enabled?: boolean;
  position?: [number, number];
  score_format_string?: string;
  size?: number;
}

/**
 * Configuration for drawing answers summary on output images.
 */
export interface DrawAnswersSummaryConfig {
  enabled?: boolean;
  position?: [number, number];
  answers_summary_format_string?: string;
  size?: number;
}

/**
 * Configuration for drawing answer groups.
 */
export interface DrawAnswerGroupsConfig {
  enabled?: boolean;
  color_sequence?: string[];
}

/**
 * Configuration for drawing question verdicts on output images.
 */
export interface DrawQuestionVerdictsConfig {
  enabled?: boolean;
  verdict_colors?: {
    correct?: string | null;
    neutral?: string | null;
    incorrect?: string | null;
    bonus?: string | null;
  };
  verdict_symbol_colors?: {
    positive?: string;
    neutral?: string;
    negative?: string;
    bonus?: string;
  };
  draw_answer_groups?: DrawAnswerGroupsConfig;
}

/**
 * Configuration for drawing detected bubble texts.
 */
export interface DrawDetectedBubbleTextsConfig {
  enabled?: boolean;
}

/**
 * Configuration for evaluation outputs and visualization.
 */
export interface OutputsConfiguration {
  should_explain_scoring?: boolean;
  should_export_explanation_csv?: boolean;
  draw_score?: DrawScoreConfig;
  draw_answers_summary?: DrawAnswersSummaryConfig;
  draw_question_verdicts?: DrawQuestionVerdictsConfig;
  draw_detected_bubble_texts?: DrawDetectedBubbleTextsConfig;
}

/**
 * Main evaluation configuration object.
 *
 * This represents the structure of evaluation.json files used for answer key
 * matching and scoring configuration.
 */
export class EvaluationConfig {
  options: Record<string, any> = {};
  marking_schemes: Record<string, any> = {};
  conditional_sets: any[] = [];
  outputs_configuration: OutputsConfiguration = {};

  constructor(
    options?: Record<string, any>,
    marking_schemes?: Record<string, any>,
    conditional_sets?: any[],
    outputs_configuration?: OutputsConfiguration
  ) {
    this.options = options || {};
    this.marking_schemes = marking_schemes || {};
    this.conditional_sets = conditional_sets || [];
    this.outputs_configuration = outputs_configuration || {};
  }

  /**
   * Create EvaluationConfig from dictionary (typically from JSON).
   *
   * @param data - Dictionary containing evaluation configuration data
   * @returns EvaluationConfig instance with nested configs
   */
  static fromDict(data: Record<string, any>): EvaluationConfig {
    // Parse outputs_configuration if present
    const outputsConfigData = data.outputs_configuration || {};
    const outputsConfig: OutputsConfiguration = {
      should_explain_scoring: outputsConfigData.should_explain_scoring || false,
      should_export_explanation_csv:
        outputsConfigData.should_export_explanation_csv || false,
      draw_score: {
        enabled: false,
        position: [200, 200],
        score_format_string: 'Score: {score}',
        size: 1.5,
        ...outputsConfigData.draw_score,
      },
      draw_answers_summary: {
        enabled: false,
        position: [200, 600],
        answers_summary_format_string:
          'Correct: {correct} Incorrect: {incorrect} Unmarked: {unmarked}',
        size: 1.0,
        ...outputsConfigData.draw_answers_summary,
      },
      draw_question_verdicts: {
        enabled: true,
        verdict_colors: {
          correct: '#00FF00',
          neutral: null,
          incorrect: '#FF0000',
          bonus: '#00DDDD',
          ...outputsConfigData.draw_question_verdicts?.verdict_colors,
        },
        verdict_symbol_colors: {
          positive: '#000000',
          neutral: '#000000',
          negative: '#000000',
          bonus: '#000000',
          ...outputsConfigData.draw_question_verdicts?.verdict_symbol_colors,
        },
        draw_answer_groups: {
          enabled: true,
          color_sequence: ['#8DFBC4', '#F7FB8D', '#8D9EFB', '#EA666F'],
          ...outputsConfigData.draw_question_verdicts?.draw_answer_groups,
        },
      },
      draw_detected_bubble_texts: {
        enabled: true,
        ...outputsConfigData.draw_detected_bubble_texts,
      },
    };

    return new EvaluationConfig(
      data.options,
      data.marking_schemes,
      data.conditional_sets,
      outputsConfig
    );
  }

  /**
   * Convert EvaluationConfig to dictionary for JSON serialization.
   *
   * @returns Dictionary representation of the evaluation config
   */
  toDict(): Record<string, any> {
    return {
      options: this.options,
      marking_schemes: this.marking_schemes,
      conditional_sets: this.conditional_sets,
      outputs_configuration: this.outputs_configuration,
    };
  }
}

