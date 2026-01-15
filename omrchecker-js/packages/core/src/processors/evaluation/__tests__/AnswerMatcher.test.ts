/**
 * Tests for AnswerMatcher
 */

import { describe, it, expect } from 'vitest';
import { AnswerMatcher, AnswerType, Verdict, SchemaVerdict } from '../AnswerMatcher';
import { SectionMarkingScheme, DEFAULT_SECTION_KEY } from '../SectionMarkingScheme';

describe('AnswerMatcher', () => {
  // Create a basic default marking scheme for tests
  const createDefaultScheme = () => {
    return new SectionMarkingScheme(
      DEFAULT_SECTION_KEY,
      {
        correct: 1,
        incorrect: 0,
        unmarked: 0,
      },
      'DEFAULT',
      ''
    );
  };

  describe('Answer Type Detection', () => {
    it('should detect STANDARD answer type', () => {
      const scheme = createDefaultScheme();
      const matcher = new AnswerMatcher('A', scheme);
      expect(matcher.answerType).toBe(AnswerType.STANDARD);
      expect(matcher.answerItem).toBe('A');
    });

    it('should detect MULTIPLE_CORRECT answer type', () => {
      const scheme = createDefaultScheme();
      const matcher = new AnswerMatcher(['A', 'B', 'AB'], scheme);
      expect(matcher.answerType).toBe(AnswerType.MULTIPLE_CORRECT);
      expect(matcher.answerItem).toEqual(['A', 'B', 'AB']);
    });

    it('should detect MULTIPLE_CORRECT_WEIGHTED answer type', () => {
      const scheme = createDefaultScheme();
      const matcher = new AnswerMatcher(
        [
          ['A', 1],
          ['B', 2],
          ['AB', 3],
        ],
        scheme
      );
      expect(matcher.answerType).toBe(AnswerType.MULTIPLE_CORRECT_WEIGHTED);
      expect(matcher.answerItem).toEqual([
        ['A', 1],
        ['B', 2],
        ['AB', 3],
      ]);
    });
  });

  describe('Standard Answer Matching', () => {
    it('should match correct answer', () => {
      const scheme = createDefaultScheme();
      const matcher = new AnswerMatcher('A', scheme);
      const result = matcher.getVerdictMarking('A');

      expect(result.verdict).toBe(Verdict.ANSWER_MATCH);
      expect(result.delta).toBe(1);
    });

    it('should detect incorrect answer', () => {
      const scheme = createDefaultScheme();
      const matcher = new AnswerMatcher('A', scheme);
      const result = matcher.getVerdictMarking('B');

      expect(result.verdict).toBe(Verdict.NO_ANSWER_MATCH);
      expect(result.delta).toBe(0);
    });

    it('should detect unmarked answer', () => {
      const scheme = createDefaultScheme();
      const matcher = new AnswerMatcher('A', scheme);
      const result = matcher.getVerdictMarking('');

      expect(result.verdict).toBe(Verdict.UNMARKED);
      expect(result.delta).toBe(0);
    });
  });

  describe('Multiple Correct Answer Matching', () => {
    it('should match any of the correct answers', () => {
      const scheme = createDefaultScheme();
      const matcher = new AnswerMatcher(['A', 'B', 'AB'], scheme);

      const resultA = matcher.getVerdictMarking('A');
      expect(resultA.verdict).toBe(`${Verdict.ANSWER_MATCH}-A`);
      expect(resultA.delta).toBe(1);

      const resultB = matcher.getVerdictMarking('B');
      expect(resultB.verdict).toBe(`${Verdict.ANSWER_MATCH}-B`);
      expect(resultB.delta).toBe(1);

      const resultAB = matcher.getVerdictMarking('AB');
      expect(resultAB.verdict).toBe(`${Verdict.ANSWER_MATCH}-AB`);
      expect(resultAB.delta).toBe(1);
    });

    it('should detect incorrect answer not in list', () => {
      const scheme = createDefaultScheme();
      const matcher = new AnswerMatcher(['A', 'B'], scheme);
      const result = matcher.getVerdictMarking('C');

      expect(result.verdict).toBe(Verdict.NO_ANSWER_MATCH);
      expect(result.delta).toBe(0);
    });
  });

  describe('Weighted Answer Matching', () => {
    it('should use custom weights for each answer', () => {
      const scheme = createDefaultScheme();
      const matcher = new AnswerMatcher(
        [
          ['A', 1],
          ['B', 2],
          ['AB', 3],
        ],
        scheme
      );

      const resultA = matcher.getVerdictMarking('A');
      expect(resultA.delta).toBe(1);

      const resultB = matcher.getVerdictMarking('B');
      expect(resultB.delta).toBe(2);

      const resultAB = matcher.getVerdictMarking('AB');
      expect(resultAB.delta).toBe(3);
    });

    it('should handle fraction scores', () => {
      const scheme = createDefaultScheme();
      const matcher = new AnswerMatcher(
        [
          ['A', '1/2'],
          ['B', '3/4'],
        ],
        scheme
      );

      const resultA = matcher.getVerdictMarking('A');
      expect(resultA.delta).toBe(0.5);

      const resultB = matcher.getVerdictMarking('B');
      expect(resultB.delta).toBe(0.75);
    });
  });

  describe('Schema Verdict Mapping', () => {
    it('should map ANSWER_MATCH to CORRECT', () => {
      const verdict = AnswerMatcher.getSchemaVerdict(
        AnswerType.STANDARD,
        Verdict.ANSWER_MATCH,
        1
      );
      expect(verdict).toBe(SchemaVerdict.CORRECT);
    });

    it('should map NO_ANSWER_MATCH to INCORRECT', () => {
      const verdict = AnswerMatcher.getSchemaVerdict(
        AnswerType.STANDARD,
        Verdict.NO_ANSWER_MATCH,
        0
      );
      expect(verdict).toBe(SchemaVerdict.INCORRECT);
    });

    it('should map UNMARKED to UNMARKED', () => {
      const verdict = AnswerMatcher.getSchemaVerdict(
        AnswerType.STANDARD,
        Verdict.UNMARKED,
        0
      );
      expect(verdict).toBe(SchemaVerdict.UNMARKED);
    });

    it('should map negative weighted answers to INCORRECT', () => {
      const verdict = AnswerMatcher.getSchemaVerdict(
        AnswerType.MULTIPLE_CORRECT_WEIGHTED,
        Verdict.ANSWER_MATCH,
        -1
      );
      expect(verdict).toBe(SchemaVerdict.INCORRECT);
    });
  });

  describe('Section Explanation', () => {
    it('should return section key for standard answers', () => {
      const scheme = new SectionMarkingScheme(
        'Math',
        { correct: 1, incorrect: 0, unmarked: 0, questions: ['q1', 'q2'] },
        'DEFAULT',
        ''
      );
      const matcher = new AnswerMatcher('A', scheme);
      expect(matcher.getSectionExplanation()).toBe('Math');
    });

    it('should return custom weights for weighted answers', () => {
      const scheme = createDefaultScheme();
      const matcher = new AnswerMatcher([['A', 1], ['B', 2]], scheme);
      const explanation = matcher.getSectionExplanation();
      expect(explanation).toContain('Custom Weights');
    });
  });

  describe('Deep Copy Behavior', () => {
    it('should deep copy marking from section scheme', () => {
      // Create a scheme with streak marking (array values)
      const scheme = new SectionMarkingScheme(
        'Streak',
        {
          marking_type: 'verdict_level_streak',
          correct: [1, 2, 3], // Array value
          incorrect: 0,
          unmarked: 0,
        },
        'DEFAULT',
        ''
      );

      const matcher = new AnswerMatcher('A', scheme);

      // Modify matcher's marking
      matcher.marking[Verdict.ANSWER_MATCH] = 999;

      // Scheme's marking should be unchanged (deep copy, not reference)
      expect(scheme.marking[Verdict.ANSWER_MATCH]).not.toBe(999);
    });

    it('should not share references with section marking scheme', () => {
      const scheme = createDefaultScheme();
      const matcher1 = new AnswerMatcher('A', scheme);
      const matcher2 = new AnswerMatcher('B', scheme);

      // Modify first matcher's marking
      matcher1.marking[Verdict.ANSWER_MATCH] = 10;

      // Second matcher should have original value (independent copy)
      expect(matcher2.marking[Verdict.ANSWER_MATCH]).toBe(1);
    });
  });
});

