/**
 * Tests for SectionMarkingScheme
 */

import { describe, it, expect } from 'vitest';
import { SectionMarkingScheme, MarkingSchemeType, DEFAULT_SECTION_KEY } from '../SectionMarkingScheme';
import { AnswerType, Verdict } from '../AnswerMatcher';

describe('SectionMarkingScheme', () => {
  describe('Basic Configuration', () => {
    it('should create default marking scheme', () => {
      const scheme = new SectionMarkingScheme(
        DEFAULT_SECTION_KEY,
        {
          correct: 1,
          incorrect: -0.25,
          unmarked: 0,
        },
        'DEFAULT',
        ''
      );

      expect(scheme.sectionKey).toBe(DEFAULT_SECTION_KEY);
      expect(scheme.markingType).toBe(MarkingSchemeType.DEFAULT);
      expect(scheme.marking[Verdict.ANSWER_MATCH]).toBe(1);
      expect(scheme.marking[Verdict.NO_ANSWER_MATCH]).toBe(-0.25);
      expect(scheme.marking[Verdict.UNMARKED]).toBe(0);
    });

    it('should create custom section scheme', () => {
      const scheme = new SectionMarkingScheme(
        'Physics',
        {
          questions: ['q1', 'q2', 'q3'],
          marking: {
            correct: 2,
            incorrect: -0.5,
            unmarked: 0,
          },
        },
        'DEFAULT',
        ''
      );

      expect(scheme.sectionKey).toBe('Physics');
      expect(scheme.questions).toEqual(['q1', 'q2', 'q3']);
      expect(scheme.marking[Verdict.ANSWER_MATCH]).toBe(2);
    });
  });

  describe('Delta Calculation', () => {
    it('should calculate basic delta for correct answer', () => {
      const scheme = new SectionMarkingScheme(
        DEFAULT_SECTION_KEY,
        { correct: 1, incorrect: -0.25, unmarked: 0 },
        'DEFAULT',
        ''
      );

      const answerMarking = { ...scheme.marking };
      const result = scheme.getDeltaAndUpdateStreak(
        answerMarking,
        AnswerType.STANDARD,
        Verdict.ANSWER_MATCH,
        false
      );

      expect(result.delta).toBe(1);
      expect(result.currentStreak).toBe(0);
      expect(result.updatedStreak).toBe(0);
    });

    it('should calculate delta for incorrect answer', () => {
      const scheme = new SectionMarkingScheme(
        DEFAULT_SECTION_KEY,
        { correct: 1, incorrect: -0.25, unmarked: 0 },
        'DEFAULT',
        ''
      );

      const answerMarking = { ...scheme.marking };
      const result = scheme.getDeltaAndUpdateStreak(
        answerMarking,
        AnswerType.STANDARD,
        Verdict.NO_ANSWER_MATCH,
        false
      );

      expect(result.delta).toBe(-0.25);
    });

    it('should handle fraction scores', () => {
      const scheme = new SectionMarkingScheme(
        DEFAULT_SECTION_KEY,
        { correct: '3/4', incorrect: '-1/4', unmarked: 0 },
        'DEFAULT',
        ''
      );

      expect(scheme.marking[Verdict.ANSWER_MATCH]).toBe(0.75);
      expect(scheme.marking[Verdict.NO_ANSWER_MATCH]).toBe(-0.25);
    });
  });

  describe('Streak Bonuses - Verdict Level', () => {
    it('should track separate streaks for each verdict', () => {
      const scheme = new SectionMarkingScheme(
        'Bonus',
        {
          marking_type: 'verdict_level_streak',
          correct: [1, 2, 3], // Increases with streak
          incorrect: 0,
          unmarked: 0,
        },
        'DEFAULT',
        ''
      );

      const answerMarking = { ...scheme.marking };

      // First correct answer - streak 0
      let result = scheme.getDeltaAndUpdateStreak(
        answerMarking,
        AnswerType.STANDARD,
        Verdict.ANSWER_MATCH,
        true
      );
      expect(result.delta).toBe(1);
      expect(result.currentStreak).toBe(0);
      expect(result.updatedStreak).toBe(1);

      // Second correct answer - streak 1
      result = scheme.getDeltaAndUpdateStreak(
        answerMarking,
        AnswerType.STANDARD,
        Verdict.ANSWER_MATCH,
        true
      );
      expect(result.delta).toBe(2);
      expect(result.currentStreak).toBe(1);
      expect(result.updatedStreak).toBe(2);

      // Third correct answer - streak 2
      result = scheme.getDeltaAndUpdateStreak(
        answerMarking,
        AnswerType.STANDARD,
        Verdict.ANSWER_MATCH,
        true
      );
      expect(result.delta).toBe(3);
      expect(result.currentStreak).toBe(2);
      expect(result.updatedStreak).toBe(3);
    });

    it('should reset streak on incorrect answer', () => {
      const scheme = new SectionMarkingScheme(
        'Bonus',
        {
          marking_type: 'verdict_level_streak',
          correct: [1, 2, 3],
          incorrect: 0,
          unmarked: 0,
        },
        'DEFAULT',
        ''
      );

      const answerMarking = { ...scheme.marking };

      // Build streak
      scheme.getDeltaAndUpdateStreak(
        answerMarking,
        AnswerType.STANDARD,
        Verdict.ANSWER_MATCH,
        true
      );
      scheme.getDeltaAndUpdateStreak(
        answerMarking,
        AnswerType.STANDARD,
        Verdict.ANSWER_MATCH,
        true
      );

      // Incorrect answer resets
      let result = scheme.getDeltaAndUpdateStreak(
        answerMarking,
        AnswerType.STANDARD,
        Verdict.NO_ANSWER_MATCH,
        true
      );
      expect(result.currentStreak).toBe(0);

      // Next correct starts from 0 again
      result = scheme.getDeltaAndUpdateStreak(
        answerMarking,
        AnswerType.STANDARD,
        Verdict.ANSWER_MATCH,
        true
      );
      expect(result.currentStreak).toBe(0);
      expect(result.delta).toBe(1);
    });
  });

  describe('Bonus Types', () => {
    it('should detect BONUS_FOR_ALL', () => {
      const scheme = new SectionMarkingScheme(
        'BONUS_Section',
        {
          correct: 1,
          incorrect: 1, // Positive for incorrect
          unmarked: 1,  // Positive for unmarked
        },
        'DEFAULT',
        ''
      );

      expect(scheme.getBonusType()).toBe('BONUS_FOR_ALL');
    });

    it('should detect BONUS_ON_ATTEMPT', () => {
      const scheme = new SectionMarkingScheme(
        'BONUS_Section',
        {
          correct: 1,
          incorrect: 1,  // Positive for incorrect
          unmarked: 0,   // Zero for unmarked
        },
        'DEFAULT',
        ''
      );

      expect(scheme.getBonusType()).toBe('BONUS_ON_ATTEMPT');
    });

    it('should return null for no bonus', () => {
      const scheme = new SectionMarkingScheme(
        'Regular',
        {
          correct: 1,
          incorrect: 0,  // Zero or negative
          unmarked: 0,
        },
        'DEFAULT',
        ''
      );

      expect(scheme.getBonusType()).toBeNull();
    });
  });

  describe('Streak Reset', () => {
    it('should reset all streaks', () => {
      const scheme = new SectionMarkingScheme(
        'Test',
        {
          marking_type: 'verdict_level_streak',
          correct: [1, 2, 3],
          incorrect: 0,
          unmarked: 0,
        },
        'DEFAULT',
        ''
      );

      // Build some streaks
      const answerMarking = { ...scheme.marking };
      scheme.getDeltaAndUpdateStreak(
        answerMarking,
        AnswerType.STANDARD,
        Verdict.ANSWER_MATCH,
        true
      );

      // Reset
      scheme.resetAllStreaks();

      // Check that streaks are back to 0
      const result = scheme.getDeltaAndUpdateStreak(
        answerMarking,
        AnswerType.STANDARD,
        Verdict.ANSWER_MATCH,
        true
      );
      expect(result.currentStreak).toBe(0);
    });
  });

  describe('Validation', () => {
    it('should validate streak marking length vs question count', () => {
      // Create a scheme with 5 questions but only 3 streak levels
      // This should log a warning but not throw an error
      const scheme = new SectionMarkingScheme(
        'Physics',
        {
          marking_type: 'verdict_level_streak',
          questions: ['q1', 'q2', 'q3', 'q4', 'q5'],
          marking: {
            correct: [1, 2, 3], // Only 3 levels for 5 questions
            incorrect: 0,
            unmarked: 0,
          },
        },
        'DEFAULT',
        ''
      );

      // Should not throw, just warn
      expect(scheme.sectionKey).toBe('Physics');
      expect(scheme.questions).toHaveLength(5);
    });

    it('should not warn if streak levels match or exceed question count', () => {
      // Create a scheme with 3 questions and 5 streak levels
      const scheme = new SectionMarkingScheme(
        'Math',
        {
          marking_type: 'verdict_level_streak',
          questions: ['q1', 'q2', 'q3'],
          marking: {
            correct: [1, 2, 3, 4, 5], // 5 levels for 3 questions - OK
            incorrect: 0,
            unmarked: 0,
          },
        },
        'DEFAULT',
        ''
      );

      expect(scheme.sectionKey).toBe('Math');
      expect(scheme.questions).toHaveLength(3);
    });

    it('should not validate if not using streak marking', () => {
      // Default marking type should not trigger validation
      const scheme = new SectionMarkingScheme(
        'Chemistry',
        {
          questions: ['q1', 'q2', 'q3', 'q4', 'q5'],
          marking: {
            correct: 1,
            incorrect: 0,
            unmarked: 0,
          },
        },
        'DEFAULT',
        ''
      );

      expect(scheme.markingType).toBe(MarkingSchemeType.DEFAULT);
      expect(scheme.questions).toHaveLength(5);
    });
  });

  describe('Deep Copy Behavior', () => {
    it('should create independent copy with deepcopyWithQuestions', () => {
      const original = new SectionMarkingScheme(
        'Original',
        {
          marking_type: 'verdict_level_streak',
          questions: ['q1', 'q2', 'q3'],
          marking: {
            correct: [1, 2, 3],
            incorrect: 0,
            unmarked: 0,
          },
        },
        'DEFAULT',
        ''
      );

      // Create a copy with different questions
      const copy = original.deepcopyWithQuestions(['q4', 'q5']);

      // Verify questions are different
      expect(copy.questions).toEqual(['q4', 'q5']);
      expect(original.questions).toEqual(['q1', 'q2', 'q3']);

      // Verify marking is deep copied (not shared reference)
      // Build a streak in the original
      const originalMarking = { ...original.marking };
      original.getDeltaAndUpdateStreak(
        originalMarking,
        AnswerType.STANDARD,
        Verdict.ANSWER_MATCH,
        true
      );

      // Copy should have independent streaks
      const copyMarking = { ...copy.marking };
      const result = copy.getDeltaAndUpdateStreak(
        copyMarking,
        AnswerType.STANDARD,
        Verdict.ANSWER_MATCH,
        true
      );

      // Copy should start from streak 0, not affected by original
      expect(result.currentStreak).toBe(0);
    });

    it('should not share marking object references', () => {
      const original = new SectionMarkingScheme(
        'Original',
        {
          questions: ['q1', 'q2'],
          marking: {
            correct: 1,
            incorrect: -0.25,
            unmarked: 0,
          },
        },
        'DEFAULT',
        ''
      );

      const copy = original.deepcopyWithQuestions(['q3', 'q4']);

      // Modify copy's marking
      copy.marking[Verdict.ANSWER_MATCH] = 999;

      // Original should be unchanged
      expect(original.marking[Verdict.ANSWER_MATCH]).toBe(1);
      expect(copy.marking[Verdict.ANSWER_MATCH]).toBe(999);
    });
  });
});

