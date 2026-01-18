/**
 * Evaluation configuration defaults.
 *
 * TypeScript port of src/schemas/defaults/evaluation.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import {
  DEFAULT_ANSWERS_SUMMARY_FORMAT_STRING,
  DEFAULT_SCORE_FORMAT_STRING,
} from '../constants';
import { EvaluationConfig } from '../models/evaluation';

/**
 * Create default evaluation config instance.
 *
 * Port of EVALUATION_CONFIG_DEFAULTS from Python.
 */
export const EVALUATION_CONFIG_DEFAULTS = new EvaluationConfig(
  {},
  {},
  [],
  {
    should_explain_scoring: false,
    should_export_explanation_csv: false,
    draw_score: {
      enabled: false,
      position: [200, 200],
      score_format_string: DEFAULT_SCORE_FORMAT_STRING,
      size: 1.5,
    },
    draw_answers_summary: {
      enabled: false,
      position: [200, 600],
      answers_summary_format_string: DEFAULT_ANSWERS_SUMMARY_FORMAT_STRING,
      size: 1.0,
    },
    draw_question_verdicts: {
      enabled: true,
      verdict_colors: {
        correct: '#00FF00',
        neutral: null,
        incorrect: '#FF0000',
        bonus: '#00DDDD',
      },
      verdict_symbol_colors: {
        positive: '#000000',
        neutral: '#000000',
        negative: '#000000',
        bonus: '#000000',
      },
      draw_answer_groups: {
        enabled: true,
        color_sequence: ['#8DFBC4', '#F7FB8D', '#8D9EFB', '#EA666F'],
      },
    },
    draw_detected_bubble_texts: {
      enabled: true,
    },
  }
);

