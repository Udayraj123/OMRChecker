/**
 * Comprehensive tests for EvaluationConfigForSet class.
 *
 * Tests all critical methods including validation, parsing, and answer matching.
 * Ported from Python test_evaluation_config_for_set.py
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { EvaluationConfigForSet } from '../EvaluationConfigForSet';
import { SectionMarkingScheme, DEFAULT_SECTION_KEY } from '../SectionMarkingScheme';
import { FieldDefinitionError, ConfigError, OMRCheckerError } from '../../../core/exceptions';
import type { EvaluationOptions } from '../EvaluationConfigForSet';

/**
 * Create minimal valid evaluation options for testing.
 */
function createMinimalEvaluationOptions(): EvaluationOptions {
  return {
    questions_in_order: ['q1', 'q2', 'q3'],
    answers_in_order: ['A', 'B', 'C'],
  };
}

/**
 * Create minimal marking schemes for testing.
 */
function createMinimalMarkingSchemes(): Record<string, any> {
  return {
    [DEFAULT_SECTION_KEY]: {
      correct: 1,
      incorrect: 0,
      unmarked: 0,
    },
  };
}

/**
 * Create mock template for testing.
 */
function createMockTemplate() {
  return {
    global_empty_val: '',
  };
}

/**
 * Create mock tuning config for testing.
 */
function createMockTuningConfig() {
  return {
    outputs: {
      filter_out_multimarked_files: false,
    },
  };
}

describe('EvaluationConfigForSet', () => {
  describe('Initialization', () => {
    it('should initialize with local answers', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      expect(config.setName).toBe('DEFAULT_SET');
      expect(config.questionsInOrder.length).toBe(3);
      expect(config.answersInOrder.length).toBe(3);
      expect(config.questionsInOrder).toEqual(['q1', 'q2', 'q3']);
      expect(config.answersInOrder).toEqual(['A', 'B', 'C']);
    });

    it('should initialize with parent config', () => {
      const parentOptions = createMinimalEvaluationOptions();
      const parentMarkingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const parentConfig = new EvaluationConfigForSet(
        'PARENT_SET',
        parentOptions,
        parentMarkingSchemes,
        template.global_empty_val
      );

      const childOptions: EvaluationOptions = {
        questions_in_order: ['q4'],
        answers_in_order: ['D'],
      };
      const childMarkingSchemes = createMinimalMarkingSchemes();

      const childConfig = new EvaluationConfigForSet(
        'CHILD_SET',
        childOptions,
        childMarkingSchemes,
        template.global_empty_val,
        parentConfig
      );

      expect(childConfig.hasConditionalSets).toBe(true);
      expect(childConfig.questionsInOrder.length).toBe(4); // q1, q2, q3, q4
      expect(childConfig.questionsInOrder).toContain('q4');
    });

    it('should initialize with custom marking scheme', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = {
        ...createMinimalMarkingSchemes(),
        SECTION_1: {
          questions: ['q1', 'q2'],
          marking: {
            correct: 2,
            incorrect: -1,
            unmarked: 0,
          },
        },
      };
      const template = createMockTemplate();

      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      expect(config.hasCustomMarking).toBe(true);
      expect('SECTION_1' in config.sectionMarkingSchemes).toBe(true);
    });
  });

  describe('validateQuestions', () => {
    it('should validate questions with equal lengths', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();

      // Should not throw
      expect(() => {
        new EvaluationConfigForSet(
          'DEFAULT_SET',
          options,
          markingSchemes,
          template.global_empty_val
        );
      }).not.toThrow();
    });

    it('should throw error for unequal lengths', () => {
      const options: EvaluationOptions = {
        questions_in_order: ['q1', 'q2', 'q3'],
        answers_in_order: ['A', 'B'], // Only 2 instead of 3
      };
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();

      expect(() => {
        new EvaluationConfigForSet(
          'DEFAULT_SET',
          options,
          markingSchemes,
          template.global_empty_val
        );
      }).toThrow();
    });
  });

  describe('validateMarkingSchemes', () => {
    it('should validate marking schemes with no overlap', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = {
        ...createMinimalMarkingSchemes(),
        SECTION_1: {
          questions: ['q1'],
          marking: { correct: 1, incorrect: 0, unmarked: 0 },
        },
        SECTION_2: {
          questions: ['q2'],
          marking: { correct: 1, incorrect: 0, unmarked: 0 },
        },
      };
      const template = createMockTemplate();

      // Should not throw
      expect(() => {
        new EvaluationConfigForSet(
          'DEFAULT_SET',
          options,
          markingSchemes,
          template.global_empty_val
        );
      }).not.toThrow();
    });

    it('should throw error for overlapping marking schemes', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = {
        ...createMinimalMarkingSchemes(),
        SECTION_1: {
          questions: ['q1', 'q2'],
          marking: { correct: 1, incorrect: 0, unmarked: 0 },
        },
        SECTION_2: {
          questions: ['q2', 'q3'], // Overlaps with SECTION_1
          marking: { correct: 1, incorrect: 0, unmarked: 0 },
        },
      };
      const template = createMockTemplate();

      expect(() => {
        new EvaluationConfigForSet(
          'DEFAULT_SET',
          options,
          markingSchemes,
          template.global_empty_val
        );
      }).toThrow(FieldDefinitionError);
    });

    it('should throw error for missing questions in marking schemes', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = {
        ...createMinimalMarkingSchemes(),
        SECTION_1: {
          questions: ['q99'], // Not in questions_in_order
          marking: { correct: 1, incorrect: 0, unmarked: 0 },
        },
      };
      const template = createMockTemplate();

      expect(() => {
        new EvaluationConfigForSet(
          'DEFAULT_SET',
          options,
          markingSchemes,
          template.global_empty_val
        );
      }).toThrow();
    });
  });

  describe('validateAnswers', () => {
    it('should validate answers with no multi-marked', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      const tuningConfig = createMockTuningConfig();
      tuningConfig.outputs.filter_out_multimarked_files = true;

      // Should not throw
      expect(() => {
        config.validateAnswers(tuningConfig);
      }).not.toThrow();
    });

    it('should throw error for multi-marked standard answers', () => {
      const options: EvaluationOptions = {
        questions_in_order: ['q1', 'q2', 'q3'],
        answers_in_order: ['AB', 'B', 'C'], // "AB" is multi-marked
      };
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      const tuningConfig = createMockTuningConfig();
      tuningConfig.outputs.filter_out_multimarked_files = true;

      expect(() => {
        config.validateAnswers(tuningConfig);
      }).toThrow(ConfigError);
    });

    it('should throw error for multi-marked multiple correct answers', () => {
      const options: EvaluationOptions = {
        questions_in_order: ['q1', 'q2', 'q3'],
        answers_in_order: [['AB', 'CD'], 'B', 'C'], // Multi-marked
      };
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      const tuningConfig = createMockTuningConfig();
      tuningConfig.outputs.filter_out_multimarked_files = true;

      expect(() => {
        config.validateAnswers(tuningConfig);
      }).toThrow(ConfigError);
    });
  });

  describe('validateFormatStrings', () => {
    it('should validate valid format strings', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      // Set valid format strings
      config.drawAnswersSummary = {
        enabled: true,
        answers_summary_format_string: '{correct}/{incorrect}',
        position: [200, 600],
        size: 1.0,
      };
      config.drawScore = {
        enabled: true,
        score_format_string: 'Score: {score}',
        position: [200, 200],
        size: 1.5,
      };

      // Should not throw
      expect(() => {
        config.validateFormatStrings();
      }).not.toThrow();
    });

    it('should throw error for invalid format strings in answers summary', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      config.drawAnswersSummary = {
        enabled: true,
        answers_summary_format_string: '{invalid_variable}',
        position: [200, 600],
        size: 1.0,
      };

      expect(() => {
        config.validateFormatStrings();
      }).toThrow(ConfigError);
    });

    it('should throw error for invalid format strings in score', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      config.drawScore = {
        enabled: true,
        score_format_string: '{invalid_variable}',
        position: [200, 200],
        size: 1.5,
      };

      expect(() => {
        config.validateFormatStrings();
      }).toThrow(ConfigError);
    });
  });

  describe('prepareAndValidateOmrResponse', () => {
    it('should validate valid OMR response', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      const concatenatedResponse = { q1: 'A', q2: 'B', q3: 'C' };

      // Should not throw
      expect(() => {
        config.prepareAndValidateOmrResponse(concatenatedResponse);
      }).not.toThrow();
    });

    it('should throw error for missing question keys', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      const concatenatedResponse = { q1: 'A', q2: 'B' }; // Missing q3

      expect(() => {
        config.prepareAndValidateOmrResponse(concatenatedResponse);
      }).toThrow();
    });

    it('should handle allow_streak flag', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      const concatenatedResponse = { q1: 'A', q2: 'B', q3: 'C' };

      config.prepareAndValidateOmrResponse(concatenatedResponse, true);

      expect(config.allowStreak).toBe(true);
    });
  });

  describe('matchAnswerForQuestion', () => {
    it('should match correct answer', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      const concatenatedResponse = { q1: 'A', q2: 'B', q3: 'C' };
      config.prepareAndValidateOmrResponse(concatenatedResponse);

      // matchAnswerForQuestion returns {delta, verdict, answerMatcher, schemaVerdict}
      const result = config.matchAnswerForQuestion(0.0, 'q1', 'A');

      expect(result).toBeDefined();
      expect(result.delta).toBeGreaterThanOrEqual(0); // Should be non-negative for correct
      expect(['correct', 'answer-match']).toContain(result.verdict);
      expect(result.schemaVerdict).toBe('correct');
    });

    it('should match incorrect answer', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      const concatenatedResponse = { q1: 'A', q2: 'B', q3: 'C' };
      config.prepareAndValidateOmrResponse(concatenatedResponse);

      const result = config.matchAnswerForQuestion(0.0, 'q1', 'B'); // Wrong answer

      expect(result.delta).toBeLessThanOrEqual(0); // Should be zero or negative for incorrect
      expect(['incorrect', 'neutral']).toContain(result.schemaVerdict);
    });

    it('should match unmarked answer', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      const concatenatedResponse = { q1: '', q2: 'B', q3: 'C' };
      config.prepareAndValidateOmrResponse(concatenatedResponse);

      const result = config.matchAnswerForQuestion(0.0, 'q1', '');

      expect(result.delta).toBe(0); // Should be zero for unmarked
      expect(result.schemaVerdict).toBe('unmarked');
    });
  });

  describe('getEvaluationMetaForQuestion', () => {
    it('should get evaluation metadata for a question', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      // Enable draw_question_verdicts so verdict_colors is initialized
      config.drawQuestionVerdicts = {
        enabled: true,
        verdict_colors: {
          correct: '#00FF00',
          incorrect: '#FF0000',
          neutral: undefined,
          bonus: '#00DDDD',
        },
        verdict_symbol_colors: {
          positive: '#000000',
          negative: '#000000',
          neutral: '#000000',
          bonus: '#000000',
        },
        draw_answer_groups: {
          enabled: true,
          color_sequence: ['#8DFBC4', '#F7FB8D', '#8D9EFB', '#EA666F'],
        },
      };

      // Initialize verdict colors (set directly since parseDrawQuestionVerdicts may not exist)
      config.verdictColors = {
        correct: [0, 255, 0],
        incorrect: [0, 0, 255],
        neutral: [0, 0, 255],
        bonus: [255, 221, 0],
      };
      config.verdictSymbolColors = {
        positive: [0, 0, 0],
        negative: [0, 0, 0],
        neutral: [0, 0, 0],
        bonus: [0, 0, 0],
      };

      const questionMeta = {
        verdict: 'correct',
        bonus_type: null,
        delta: 1.0,
        question_verdict: 'correct',
        question_schema_verdict: 'correct',
      };

      const isFieldMarked = true;

      const meta = config.getEvaluationMetaForQuestion(
        questionMeta,
        isFieldMarked,
        'GRAYSCALE'
      );

      expect(Array.isArray(meta)).toBe(true);
      expect(meta.length).toBe(4); // verdict_symbol, verdict_color, verdict_symbol_color, thickness_factor
      expect(meta[0]).toBeDefined(); // verdict_symbol
      expect(meta[1]).toBeDefined(); // verdict_color
      expect(meta[2]).toBeDefined(); // verdict_symbol_color
      expect(meta[3]).toBeDefined(); // thickness_factor
    });
  });

  describe('getFormattedAnswersSummary', () => {
    it('should get formatted answers summary', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      // Set some verdict counts
      config.schemaVerdictCounts = {
        correct: 2,
        incorrect: 1,
        unmarked: 0,
      };

      config.drawAnswersSummary = {
        enabled: true,
        answers_summary_format_string: '{correct}/{incorrect}',
        position: [200, 600],
        size: 1.0,
      };

      // Returns [answers_format, position, size, thickness]
      const summary = config.getFormattedAnswersSummary();

      expect(Array.isArray(summary)).toBe(true);
      expect(summary.length).toBe(4);
      expect(typeof summary[0]).toBe('string'); // answers_format
      expect(summary[0].length).toBeGreaterThan(0);
    });
  });

  describe('getFormattedScore', () => {
    it('should get formatted score', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      config.drawScore = {
        enabled: true,
        score_format_string: 'Score: {score}',
        position: [200, 200],
        size: 1.5,
      };

      const score = 85.5;

      // Returns [score_format, position, size, thickness]
      const formatted = config.getFormattedScore(score);

      expect(Array.isArray(formatted)).toBe(true);
      expect(formatted.length).toBe(4);
      expect(typeof formatted[0]).toBe('string'); // score_format
      expect(formatted[0]).toContain('85');
    });
  });

  describe('resetEvaluation', () => {
    it('should reset evaluation state', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      // Set some counts
      config.schemaVerdictCounts = {
        correct: 5,
        incorrect: 2,
        unmarked: 1,
      };

      config.resetEvaluation();

      expect(config.schemaVerdictCounts.correct).toBe(0);
      expect(config.schemaVerdictCounts.incorrect).toBe(0);
      expect(config.schemaVerdictCounts.unmarked).toBe(0);
    });
  });

  describe('getMarkingSchemeForQuestion', () => {
    it('should get default marking scheme', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      const scheme = config.getMarkingSchemeForQuestion('q1');

      expect(scheme).toBeInstanceOf(SectionMarkingScheme);
      expect(scheme.sectionKey).toBe(DEFAULT_SECTION_KEY);
    });

    it('should get custom marking scheme', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = {
        ...createMinimalMarkingSchemes(),
        SECTION_1: {
          questions: ['q1'],
          marking: { correct: 2, incorrect: -1, unmarked: 0 },
        },
      };
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      const scheme = config.getMarkingSchemeForQuestion('q1');

      expect(scheme.sectionKey).toBe('SECTION_1');
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty OMR response', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      const concatenatedResponse: Record<string, string> = {};

      expect(() => {
        config.prepareAndValidateOmrResponse(concatenatedResponse);
      }).toThrow();
    });

    it('should handle extra questions in OMR response', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      const concatenatedResponse = { q1: 'A', q2: 'B', q3: 'C', q4: 'D' }; // Extra q4

      // Should not throw (extra questions are allowed)
      expect(() => {
        config.prepareAndValidateOmrResponse(concatenatedResponse);
      }).not.toThrow();
    });

    it('should handle reset evaluation multiple times', () => {
      const options = createMinimalEvaluationOptions();
      const markingSchemes = createMinimalMarkingSchemes();
      const template = createMockTemplate();
      const config = new EvaluationConfigForSet(
        'DEFAULT_SET',
        options,
        markingSchemes,
        template.global_empty_val
      );

      config.schemaVerdictCounts = { correct: 5, incorrect: 2, unmarked: 1 };
      config.resetEvaluation();

      config.schemaVerdictCounts = { correct: 3, incorrect: 1, unmarked: 0 };
      config.resetEvaluation();

      expect(config.schemaVerdictCounts.correct).toBe(0);
      expect(config.schemaVerdictCounts.incorrect).toBe(0);
      expect(config.schemaVerdictCounts.unmarked).toBe(0);
    });
  });
});

