/**
 * Tests for EvaluationConfig
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { EvaluationConfig, DEFAULT_SET_NAME } from '../EvaluationConfig';
import { EvaluationConfigForSet } from '../EvaluationConfigForSet';
import { DEFAULT_SECTION_KEY } from '../SectionMarkingScheme';

describe('EvaluationConfig', () => {
  const mockTemplate = {
    global_empty_val: '',
  };

  describe('Basic Configuration', () => {
    it('should create default evaluation config', () => {
      const config = new EvaluationConfig(
        '/test/dir',
        'evaluation.json',
        {
          options: {
            questions_in_order: ['q1', 'q2', 'q3'],
            answers_in_order: ['A', 'B', 'C'],
          },
          marking_schemes: {
            [DEFAULT_SECTION_KEY]: {
              correct: 1,
              incorrect: 0,
              unmarked: 0,
            },
          },
        },
        mockTemplate,
        {}
      );

      expect(config.path).toBe('evaluation.json');
      expect(config.defaultEvaluationConfig).toBeInstanceOf(EvaluationConfigForSet);
      expect(config.defaultEvaluationConfig.setName).toBe(DEFAULT_SET_NAME);
      expect(config.defaultEvaluationConfig.questionsInOrder).toEqual(['q1', 'q2', 'q3']);
    });

    it('should use default marking scheme if not provided', () => {
      const config = new EvaluationConfig(
        '/test/dir',
        'evaluation.json',
        {
          options: {
            questions_in_order: ['q1'],
            answers_in_order: ['A'],
          },
        },
        mockTemplate,
        {}
      );

      expect(config.defaultEvaluationConfig).toBeInstanceOf(EvaluationConfigForSet);
      expect(config.defaultEvaluationConfig.defaultMarkingScheme).toBeDefined();
    });
  });

  describe('Conditional Sets', () => {
    it('should process conditional sets', () => {
      const config = new EvaluationConfig(
        '/test/dir',
        'evaluation.json',
        {
          options: {
            questions_in_order: ['q1', 'q2'],
            answers_in_order: ['A', 'B'],
          },
          marking_schemes: {
            [DEFAULT_SECTION_KEY]: {
              correct: 1,
              incorrect: 0,
              unmarked: 0,
            },
          },
          conditional_sets: [
            {
              name: 'Set A',
              matcher: {
                formatString: '{set_type}',
                matchRegex: '^A$',
              },
              evaluation: {
                options: {
                  questions_in_order: ['q1', 'q2'],
                  answers_in_order: ['B', 'A'], // Different answers
                },
              },
            },
          ],
        },
        mockTemplate,
        {}
      );

      expect(config.conditionalSets).toHaveLength(1);
      expect(config.conditionalSets[0][0]).toBe('Set A');
      expect(config.setMapping['Set A']).toBeInstanceOf(EvaluationConfigForSet);
      expect(config.setMapping['Set A'].setName).toBe('Set A');
    });

    it('should reject duplicate set names', () => {
      expect(() => {
        new EvaluationConfig(
          '/test/dir',
          'evaluation.json',
          {
            options: {
              questions_in_order: ['q1'],
              answers_in_order: ['A'],
            },
            conditional_sets: [
              {
                name: 'SetA',
                matcher: { formatString: '{x}', matchRegex: '.*' },
                evaluation: { options: {} },
              },
              {
                name: 'SetA', // Duplicate
                matcher: { formatString: '{y}', matchRegex: '.*' },
                evaluation: { options: {} },
              },
            ],
          },
          mockTemplate,
          {}
        );
      }).toThrow(/Repeated set name/);
    });

    it('should reject conditional set with answers_in_order but no questions_in_order', () => {
      expect(() => {
        new EvaluationConfig(
          '/test/dir',
          'evaluation.json',
          {
            options: {
              questions_in_order: ['q1', 'q2'],
              answers_in_order: ['A', 'B'],
            },
            marking_schemes: {
              [DEFAULT_SECTION_KEY]: {
                correct: 1,
                incorrect: 0,
                unmarked: 0,
              },
            },
            conditional_sets: [
              {
                name: 'Set A',
                matcher: {
                  formatString: '{set_type}',
                  matchRegex: '^A$',
                },
                evaluation: {
                  options: {
                    answers_in_order: ['B', 'C'], // Missing questions_in_order
                  },
                },
              },
            ],
          },
          mockTemplate,
          {}
        );
      }).toThrow(/provides 'answers_in_order' but missing 'questions_in_order'/);
    });

    it('should reject conditional set with questions_in_order but no answers_in_order', () => {
      expect(() => {
        new EvaluationConfig(
          '/test/dir',
          'evaluation.json',
          {
            options: {
              questions_in_order: ['q1', 'q2'],
              answers_in_order: ['A', 'B'],
            },
            marking_schemes: {
              [DEFAULT_SECTION_KEY]: {
                correct: 1,
                incorrect: 0,
                unmarked: 0,
              },
            },
            conditional_sets: [
              {
                name: 'Set A',
                matcher: {
                  formatString: '{set_type}',
                  matchRegex: '^A$',
                },
                evaluation: {
                  options: {
                    questions_in_order: ['q1', 'q2'], // Missing answers_in_order
                  },
                },
              },
            ],
          },
          mockTemplate,
          {}
        );
      }).toThrow(/provides 'questions_in_order' but missing 'answers_in_order'/);
    });
  });

  describe('Set Matching', () => {
    let config: EvaluationConfig;

    beforeEach(() => {
      config = new EvaluationConfig(
        '/test/dir',
        'evaluation.json',
        {
          options: {
            questions_in_order: ['q1', 'q2'],
            answers_in_order: ['A', 'B'],
          },
          marking_schemes: {
            [DEFAULT_SECTION_KEY]: {
              correct: 1,
              incorrect: 0,
              unmarked: 0,
            },
          },
          conditional_sets: [
            {
              name: 'Set A',
              matcher: {
                formatString: '{set_number}',
                matchRegex: '^A$',
              },
              evaluation: {
                options: {
                  questions_in_order: ['q1', 'q2'],
                  answers_in_order: ['B', 'C'],
                },
              },
            },
            {
              name: 'Set B',
              matcher: {
                formatString: '{set_number}',
                matchRegex: '^B$',
              },
              evaluation: {
                options: {
                  questions_in_order: ['q1', 'q2'],
                  answers_in_order: ['C', 'D'],
                },
              },
            },
          ],
        },
        mockTemplate,
        {}
      );
    });

    it('should match first conditional set', () => {
      const matchedSet = config.getMatchingSet({ set_number: 'A' }, 'test.jpg');
      expect(matchedSet).toBe('Set A');
    });

    it('should match second conditional set', () => {
      const matchedSet = config.getMatchingSet({ set_number: 'B' }, 'test.jpg');
      expect(matchedSet).toBe('Set B');
    });

    it('should return null if no match', () => {
      const matchedSet = config.getMatchingSet({ set_number: 'C' }, 'test.jpg');
      expect(matchedSet).toBeNull();
    });

    it('should match using file_name', () => {
      const configWithFileName = new EvaluationConfig(
        '/test/dir',
        'evaluation.json',
        {
          options: {
            questions_in_order: ['q1'],
            answers_in_order: ['A'],
          },
          conditional_sets: [
            {
              name: 'Test Files',
              matcher: {
                formatString: '{file_name}',
                matchRegex: 'test.*\\.jpg',
              },
              evaluation: {
                options: {},
              },
            },
          ],
        },
        mockTemplate,
        {}
      );

      const matchedSet = configWithFileName.getMatchingSet({}, '/path/to/test_001.jpg');
      expect(matchedSet).toBe('Test Files');
    });

    it('should match using file_path', () => {
      const configWithFilePath = new EvaluationConfig(
        '/test/dir',
        'evaluation.json',
        {
          options: {
            questions_in_order: ['q1'],
            answers_in_order: ['A'],
          },
          conditional_sets: [
            {
              name: 'Batch 1',
              matcher: {
                formatString: '{file_path}',
                matchRegex: '.*/batch1/.*',
              },
              evaluation: {
                options: {},
              },
            },
          ],
        },
        mockTemplate,
        {}
      );

      const matchedSet = configWithFilePath.getMatchingSet({}, '/data/batch1/test.jpg');
      expect(matchedSet).toBe('Batch 1');
    });
  });

  describe('Get Evaluation Config For Response', () => {
    let config: EvaluationConfig;

    beforeEach(() => {
      config = new EvaluationConfig(
        '/test/dir',
        'evaluation.json',
        {
          options: {
            questions_in_order: ['q1', 'q2'],
            answers_in_order: ['A', 'B'],
          },
          marking_schemes: {
            [DEFAULT_SECTION_KEY]: {
              correct: 1,
              incorrect: 0,
              unmarked: 0,
            },
          },
          conditional_sets: [
            {
              name: 'Set A',
              matcher: {
                formatString: '{set_type}',
                matchRegex: '^A$',
              },
              evaluation: {
                options: {
                  questions_in_order: ['q1', 'q2'],
                  answers_in_order: ['B', 'C'],
                },
              },
            },
          ],
        },
        mockTemplate,
        {}
      );
    });

    it('should return default config if no match', () => {
      const evalConfig = config.getEvaluationConfigForResponse(
        { q1: 'A', q2: 'B', set_type: 'C' },
        'test.jpg'
      );

      expect(evalConfig).toBe(config.defaultEvaluationConfig);
      expect(evalConfig.setName).toBe(DEFAULT_SET_NAME);
    });

    it('should return matched set config', () => {
      const evalConfig = config.getEvaluationConfigForResponse(
        { q1: 'A', q2: 'B', set_type: 'A' },
        'test.jpg'
      );

      expect(evalConfig).toBe(config.setMapping['Set A']);
      expect(evalConfig.setName).toBe('Set A');
      expect(evalConfig.answersInOrder).toEqual(['B', 'C']);
    });

    it('should return EvaluationConfigForSet instance with methods', () => {
      const evalConfig = config.getEvaluationConfigForResponse(
        { q1: 'A', q2: 'B' },
        'test.jpg'
      );

      expect(evalConfig).toBeInstanceOf(EvaluationConfigForSet);
      expect(typeof evalConfig.prepareAndValidateOmrResponse).toBe('function');
      expect(typeof evalConfig.matchAnswerForQuestion).toBe('function');
      expect(typeof evalConfig.getMarkingSchemeForQuestion).toBe('function');
    });
  });

  describe('Exclude Files', () => {
    it('should collect exclude files from all sets', () => {
      const config = new EvaluationConfig(
        '/test/dir',
        'evaluation.json',
        {
          options: {
            questions_in_order: ['q1'],
            answers_in_order: ['A'],
          },
        },
        mockTemplate,
        {}
      );

      expect(config.getExcludeFiles()).toEqual([]);
      expect(config.excludeFiles).toEqual([]);
    });
  });

  describe('Parent Config Merging', () => {
    it('should pass default config as parent to conditional sets', () => {
      const config = new EvaluationConfig(
        '/test/dir',
        'evaluation.json',
        {
          options: {
            questions_in_order: ['q1', 'q2', 'q3'],
            answers_in_order: ['A', 'B', 'C'],
          },
          marking_schemes: {
            [DEFAULT_SECTION_KEY]: {
              correct: 1,
              incorrect: 0,
              unmarked: 0,
            },
          },
          conditional_sets: [
            {
              name: 'Set A',
              matcher: {
                formatString: '{set}',
                matchRegex: '^A$',
              },
              evaluation: {
                options: {
                  // Override only q2
                  questions_in_order: ['q2'],
                  answers_in_order: ['D'],
                },
              },
            },
          ],
        },
        mockTemplate,
        {}
      );

      const setA = config.setMapping['Set A'];

      // Should inherit all questions from parent
      expect(setA.questionsInOrder).toContain('q1');
      expect(setA.questionsInOrder).toContain('q2');
      expect(setA.questionsInOrder).toContain('q3');

      // But q2's answer should be overridden
      expect(setA.answersInOrder[setA.questionsInOrder.indexOf('q2')]).toBe('D');
    });
  });

  describe('Empty Value Handling', () => {
    it('should use template empty value', () => {
      const templateWithEmpty = {
        global_empty_val: '-',
      };

      const config = new EvaluationConfig(
        '/test/dir',
        'evaluation.json',
        {
          options: {
            questions_in_order: ['q1'],
            answers_in_order: ['A'],
          },
        },
        templateWithEmpty,
        {}
      );

      const matcher = config.defaultEvaluationConfig.questionToAnswerMatcher['q1'];
      expect(matcher.emptyValue).toBe('-');
    });

    it('should default to empty string if no template value', () => {
      const templateWithoutEmpty = {};

      const config = new EvaluationConfig(
        '/test/dir',
        'evaluation.json',
        {
          options: {
            questions_in_order: ['q1'],
            answers_in_order: ['A'],
          },
        },
        templateWithoutEmpty,
        {}
      );

      const matcher = config.defaultEvaluationConfig.questionToAnswerMatcher['q1'];
      expect(matcher.emptyValue).toBe('');
    });
  });
});

