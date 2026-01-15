/**
 * Section marking scheme - Handles scoring rules for question sections
 * Port of Python's src/processors/evaluation/section_marking_scheme.py
 */

import { AnswerMatcher, AnswerType, SchemaVerdict, Verdict } from './AnswerMatcher';
import { Logger } from '../../utils/logger';
import { deepClone } from '../../utils/object';

const logger = new Logger('SectionMarkingScheme');

/**
 * Marking scheme types
 */
export enum MarkingSchemeType {
  DEFAULT = 'default',
  VERDICT_LEVEL_STREAK = 'verdict_level_streak',
  SECTION_LEVEL_STREAK = 'section_level_streak',
}

/**
 * Default section key
 */
export const DEFAULT_SECTION_KEY = 'DEFAULT';

/**
 * Bonus section prefix
 */
export const BONUS_SECTION_PREFIX = 'BONUS_';

/**
 * Section marking scheme configuration
 */
export interface SectionSchemeConfig {
  marking_type?: string;
  questions?: string[];
  marking?: {
    correct: number | number[];
    incorrect: number | number[];
    unmarked: number | number[];
  };
  correct?: number | number[];
  incorrect?: number | number[];
  unmarked?: number | number[];
}

/**
 * Section marking scheme - manages scoring for a section of questions
 */
export class SectionMarkingScheme {
  public sectionKey: string;
  public setName: string;
  public emptyValue: string;
  public markingType: MarkingSchemeType;
  public questions: string[] | null;
  public marking: Record<string, number>;

  // Streak tracking
  private streaks: Record<SchemaVerdict, number>;
  private sectionLevelStreak: number;
  private previousStreakVerdict: SchemaVerdict | null;

  constructor(
    sectionKey: string,
    sectionScheme: SectionSchemeConfig,
    setName: string,
    emptyValue: string
  ) {
    this.sectionKey = sectionKey;
    this.setName = setName;
    this.emptyValue = emptyValue;
    this.markingType = (sectionScheme.marking_type as MarkingSchemeType) || MarkingSchemeType.DEFAULT;
    this.streaks = {} as Record<SchemaVerdict, number>;
    this.sectionLevelStreak = 0;
    this.previousStreakVerdict = null;

    this.resetAllStreaks();

    // DEFAULT marking scheme follows a shorthand
    if (sectionKey === DEFAULT_SECTION_KEY) {
      this.questions = null;
      this.marking = this.parseVerdictMarkingFromScheme(sectionScheme);
    } else {
      this.questions = sectionScheme.questions || [];
      const markingConfig = sectionScheme.marking || sectionScheme;
      this.marking = this.parseVerdictMarkingFromScheme(markingConfig);
    }

    this.validateMarkingScheme();
  }

  /**
   * Reset all streak counters
   */
  resetAllStreaks(): void {
    if (this.markingType === MarkingSchemeType.VERDICT_LEVEL_STREAK) {
      this.streaks = {
        [SchemaVerdict.CORRECT]: 0,
        [SchemaVerdict.INCORRECT]: 0,
        [SchemaVerdict.UNMARKED]: 0,
      };
    } else {
      this.sectionLevelStreak = 0;
      this.previousStreakVerdict = null;
    }
  }

  /**
   * Parse verdict marking from scheme configuration
   */
  private parseVerdictMarkingFromScheme(
    sectionScheme: SectionSchemeConfig | any
  ): Record<string, number> {
    const parsedMarking: Record<string, number> = {};

    for (const verdict of Object.values(Verdict)) {
      const schemaVerdict = AnswerMatcher.getSchemaVerdict(AnswerType.STANDARD, verdict, 0);

      // Try to get from the scheme
      let schemaVerdictMarking: number;
      if (sectionScheme[schemaVerdict] !== undefined) {
        schemaVerdictMarking = this.parseVerdictMarking(sectionScheme[schemaVerdict]);
      } else {
        // Default values
        schemaVerdictMarking = schemaVerdict === SchemaVerdict.CORRECT ? 1 : 0;
      }

      // Warn if positive marks for incorrect in non-bonus sections
      if (
        this.markingType === MarkingSchemeType.DEFAULT &&
        schemaVerdictMarking > 0 &&
        schemaVerdict === SchemaVerdict.INCORRECT &&
        !this.sectionKey.startsWith(BONUS_SECTION_PREFIX)
      ) {
        logger.warn(
          `Found positive marks (${schemaVerdictMarking.toFixed(2)}) for incorrect answer ` +
          `in schema '${this.sectionKey}'. For Bonus sections, prefer adding prefix 'BONUS_' to the scheme name.`
        );
      }

      parsedMarking[verdict] = schemaVerdictMarking;
    }

    return parsedMarking;
  }

  /**
   * Parse verdict marking (handles fractions and arrays for streaks)
   */
  private parseVerdictMarking(marking: any): number {
    if (Array.isArray(marking)) {
      // For streak-based marking, use first value as default
      return this.parseFloatOrFraction(marking[0]);
    }
    return this.parseFloatOrFraction(marking);
  }

  /**
   * Parse float or fraction string to number
   */
  private parseFloatOrFraction(value: string | number): number {
    if (typeof value === 'number') {
      return value;
    }
    if (value.includes('/')) {
      const [num, denom] = value.split('/').map(Number);
      return num / denom;
    }
    return parseFloat(value);
  }

  /**
   * Get delta for a specific verdict and streak
   */
  private getDeltaForVerdict(
    answerMatcherMarking: Record<string, number>,
    questionVerdict: string,
    currentStreak: number
  ): number {
    const marking = answerMatcherMarking[questionVerdict];

    if (Array.isArray(marking)) {
      // Return score based on streak index
      return marking[currentStreak] || marking[marking.length - 1];
    }

    if (currentStreak > 0) {
      logger.warn(
        `Non-zero streak (${currentStreak}) for verdict ${questionVerdict} in scheme ${this}. ` +
        `Using non-streak score for this verdict.`
      );
    }

    return marking;
  }

  /**
   * Get delta and update streak based on verdict
   */
  getDeltaAndUpdateStreak(
    answerMatcherMarking: Record<string, number>,
    answerType: AnswerType,
    questionVerdict: string,
    allowStreak: boolean
  ): {
    delta: number;
    currentStreak: number;
    updatedStreak: number;
  } {
    const schemaVerdict = AnswerMatcher.getSchemaVerdict(answerType, questionVerdict, 0);

    let currentStreak: number;
    let updatedStreak: number;
    let delta: number;

    if (this.markingType === MarkingSchemeType.VERDICT_LEVEL_STREAK) {
      currentStreak = this.streaks[schemaVerdict];

      // Reset all streaks
      this.resetAllStreaks();

      // Increase only current verdict streak
      if (allowStreak && schemaVerdict !== SchemaVerdict.UNMARKED) {
        this.streaks[schemaVerdict] = currentStreak + 1;
      }

      delta = this.getDeltaForVerdict(answerMatcherMarking, questionVerdict, currentStreak);
      updatedStreak = this.streaks[schemaVerdict];

    } else if (this.markingType === MarkingSchemeType.SECTION_LEVEL_STREAK) {
      currentStreak = this.sectionLevelStreak;
      const previousVerdict = this.previousStreakVerdict;

      // Reset
      this.resetAllStreaks();

      // Increase only if same verdict continues
      if (
        allowStreak &&
        (previousVerdict === null || schemaVerdict === previousVerdict)
      ) {
        this.sectionLevelStreak = currentStreak + 1;
      }

      this.previousStreakVerdict = schemaVerdict;
      delta = this.getDeltaForVerdict(answerMatcherMarking, questionVerdict, currentStreak);
      updatedStreak = this.sectionLevelStreak;

    } else {
      // Default - no streaks
      currentStreak = 0;
      updatedStreak = 0;
      delta = this.getDeltaForVerdict(answerMatcherMarking, questionVerdict, currentStreak);
    }

    return { delta, currentStreak, updatedStreak };
  }

  /**
   * Create a deep copy with different questions
   * Port of Python's deepcopy_with_questions method
   */
  deepcopyWithQuestions(questions: string[]): SectionMarkingScheme {
    const clone = deepClone(this);
    clone.updateQuestions(questions);
    return clone;
  }

  /**
   * Update questions for this scheme
   */
  updateQuestions(questions: string[]): void {
    this.questions = questions;
    this.validateMarkingScheme();
  }

  /**
   * Validate marking scheme configuration
   */
  private validateMarkingScheme(): void {
    // Validate streak-based marking schemes
    if (
      this.markingType === MarkingSchemeType.VERDICT_LEVEL_STREAK ||
      this.markingType === MarkingSchemeType.SECTION_LEVEL_STREAK
    ) {
      // Check if any marking values are arrays (for streak bonuses)
      const streakMarkings: Array<{ verdict: string; length: number }> = [];

      for (const [verdict, marking] of Object.entries(this.marking)) {
        if (Array.isArray(marking)) {
          streakMarkings.push({ verdict, length: marking.length });
        }
      }

      // If we have questions and streak markings, validate lengths
      if (this.questions && this.questions.length > 0 && streakMarkings.length > 0) {
        const maxPossibleStreak = this.questions.length;

        for (const { verdict, length } of streakMarkings) {
          if (length < maxPossibleStreak) {
            logger.warn(
              `Marking scheme '${this.sectionKey}': Verdict '${verdict}' has ${length} streak levels ` +
              `but section has ${maxPossibleStreak} questions. Consider adding more streak levels ` +
              `or the last level will be used for all remaining streaks.`
            );
          }
        }
      }
    }
  }

  /**
   * Get bonus type for this section
   */
  getBonusType(): string | null {
    if (
      this.markingType === MarkingSchemeType.VERDICT_LEVEL_STREAK ||
      this.marking[Verdict.NO_ANSWER_MATCH] <= 0
    ) {
      return null;
    }
    if (this.marking[Verdict.UNMARKED] > 0) {
      return 'BONUS_FOR_ALL';
    }
    if (this.marking[Verdict.UNMARKED] === 0) {
      return 'BONUS_ON_ATTEMPT';
    }
    return null;
  }

  toString(): string {
    return this.sectionKey;
  }
}

