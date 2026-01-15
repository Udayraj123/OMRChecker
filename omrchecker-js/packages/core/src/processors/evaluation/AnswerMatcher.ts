/**
 * Answer matcher - Determines how answers are compared and scored
 * Port of Python's src/processors/evaluation/answer_matcher.py
 */

import { Logger } from '../../utils/logger';
import { deepClone } from '../../utils/object';
import type { SectionMarkingScheme } from './SectionMarkingScheme';

const logger = new Logger('AnswerMatcher');

/**
 * Answer types supported by the evaluation system
 */
export enum AnswerType {
  /** Single correct answer: 'A' */
  STANDARD = 'STANDARD',
  /** Multiple acceptable answers: ['A', 'B', 'AB'] */
  MULTIPLE_CORRECT = 'MULTIPLE_CORRECT',
  /** Weighted answers: [['A', 1], ['B', 2], ['AB', 3]] */
  MULTIPLE_CORRECT_WEIGHTED = 'MULTIPLE_CORRECT_WEIGHTED',
}

/**
 * Verdict outcomes for answer matching
 */
export enum Verdict {
  ANSWER_MATCH = 'ANSWER-MATCH',
  NO_ANSWER_MATCH = 'NO-ANSWER-MATCH',
  UNMARKED = 'UNMARKED',
}

/**
 * Schema-level verdict categories
 */
export enum SchemaVerdict {
  CORRECT = 'correct',
  INCORRECT = 'incorrect',
  UNMARKED = 'unmarked',
}

/** Mapping from Verdict to SchemaVerdict */
export const VERDICT_TO_SCHEMA_VERDICT: Record<Verdict, SchemaVerdict> = {
  [Verdict.ANSWER_MATCH]: SchemaVerdict.CORRECT,
  [Verdict.NO_ANSWER_MATCH]: SchemaVerdict.INCORRECT,
  [Verdict.UNMARKED]: SchemaVerdict.UNMARKED,
};

/** Verdicts in evaluation order */
export const VERDICTS_IN_ORDER = [
  Verdict.ANSWER_MATCH,
  Verdict.NO_ANSWER_MATCH,
  Verdict.UNMARKED,
];

/** Schema verdicts in order */
export const SCHEMA_VERDICTS_IN_ORDER = [
  SchemaVerdict.CORRECT,
  SchemaVerdict.INCORRECT,
  SchemaVerdict.UNMARKED,
];

/**
 * Answer matcher class - handles different answer types and scoring
 */
export class AnswerMatcher {
  public answerType: AnswerType;
  public answerItem!: string | string[] | Array<[string, number]>;
  public marking!: Record<string, number>;
  public emptyValue!: string;
  private sectionMarkingScheme: SectionMarkingScheme;

  constructor(answerItem: any, sectionMarkingScheme: SectionMarkingScheme) {
    this.sectionMarkingScheme = sectionMarkingScheme;
    this.answerType = this.getAnswerType(answerItem);
    this.parseAndSetAnswerItem(answerItem);
    this.setLocalMarkingDefaults(sectionMarkingScheme);
  }

  /**
   * Check if answer is a standard string answer
   */
  private static isStandardAnswer(element: any): element is string {
    return typeof element === 'string' && element.length >= 1;
  }

  /**
   * Check if element is a marking score (string or number)
   */
  private static isMarkingScore(element: any): boolean {
    return typeof element === 'string' || typeof element === 'number';
  }

  /**
   * Parse float or fraction string to number
   */
  private static parseFloatOrFraction(value: string | number): number {
    if (typeof value === 'number') {
      return value;
    }
    // Handle fractions like "1/2"
    if (value.includes('/')) {
      const [num, denom] = value.split('/').map(Number);
      return num / denom;
    }
    return parseFloat(value);
  }

  /**
   * Get schema verdict from question verdict and delta
   */
  static getSchemaVerdict(
    answerType: AnswerType,
    questionVerdict: string,
    delta: number | null = null
  ): SchemaVerdict {
    // Negative custom weights should be considered as incorrect
    if (
      delta !== null &&
      delta < 0 &&
      answerType === AnswerType.MULTIPLE_CORRECT_WEIGHTED
    ) {
      return SchemaVerdict.INCORRECT;
    }

    for (const verdict of VERDICTS_IN_ORDER) {
      if (questionVerdict.startsWith(verdict)) {
        return VERDICT_TO_SCHEMA_VERDICT[verdict];
      }
    }

    throw new Error(`Unable to determine schema verdict for: ${questionVerdict}`);
  }

  /**
   * Determine the answer type from answer item structure
   */
  private getAnswerType(answerItem: any): AnswerType {
    // Standard answer: 'A'
    if (AnswerMatcher.isStandardAnswer(answerItem)) {
      return AnswerType.STANDARD;
    }

    if (Array.isArray(answerItem)) {
      // Multiple correct: ['A', 'B', 'AB']
      if (
        answerItem.length >= 2 &&
        answerItem.every((item) => AnswerMatcher.isStandardAnswer(item))
      ) {
        return AnswerType.MULTIPLE_CORRECT;
      }

      // Multiple correct weighted: [['A', 1], ['B', 2]]
      if (
        answerItem.length >= 1 &&
        answerItem.every(
          (item) =>
            Array.isArray(item) &&
            item.length === 2 &&
            AnswerMatcher.isStandardAnswer(item[0]) &&
            AnswerMatcher.isMarkingScore(item[1])
        )
      ) {
        return AnswerType.MULTIPLE_CORRECT_WEIGHTED;
      }
    }

    logger.error(`Unable to determine answer type for: ${JSON.stringify(answerItem)}`);
    throw new Error('Unable to determine answer type');
  }

  /**
   * Parse and set the answer item with proper types
   */
  private parseAndSetAnswerItem(answerItem: any): void {
    if (this.answerType === AnswerType.MULTIPLE_CORRECT_WEIGHTED) {
      // Parse answer scores for weighted answers
      this.answerItem = answerItem.map(([answer, score]: [string, any]) => [
        answer,
        AnswerMatcher.parseFloatOrFraction(score),
      ]);
    } else {
      this.answerItem = answerItem;
    }
  }

  /**
   * Set local marking defaults based on section scheme
   */
  /**
   * Set local marking defaults based on section scheme
   * Port of Python's set_local_marking_defaults method
   */
  private setLocalMarkingDefaults(sectionMarkingScheme: SectionMarkingScheme): void {
    this.emptyValue = sectionMarkingScheme.emptyValue;

    // Deep copy section marking locally (matches Python's deepcopy)
    this.marking = deepClone(sectionMarkingScheme.marking);

    if (this.answerType === AnswerType.STANDARD) {
      // No local overrides needed
    } else if (this.answerType === AnswerType.MULTIPLE_CORRECT) {
      const allowedAnswers = this.answerItem as string[];
      // Override marking scheme scores for each allowed answer
      for (const allowedAnswer of allowedAnswers) {
        this.marking[`${Verdict.ANSWER_MATCH}-${allowedAnswer}`] =
          this.marking[Verdict.ANSWER_MATCH];
      }
    } else if (this.answerType === AnswerType.MULTIPLE_CORRECT_WEIGHTED) {
      const weightedAnswers = this.answerItem as Array<[string, number]>;
      for (const [allowedAnswer, score] of weightedAnswers) {
        this.marking[`${Verdict.ANSWER_MATCH}-${allowedAnswer}`] = score;
      }
    }
  }

  /**
   * Get verdict and marking for a marked answer
   */
  getVerdictMarking(
    markedAnswer: string,
    allowStreak: boolean = false
  ): {
    verdict: string;
    delta: number;
    currentStreak: number;
    updatedStreak: number;
  } {
    let verdict: string;

    if (this.answerType === AnswerType.STANDARD) {
      verdict = this.getStandardVerdict(markedAnswer);
    } else if (this.answerType === AnswerType.MULTIPLE_CORRECT) {
      verdict = this.getMultipleCorrectVerdict(markedAnswer);
    } else {
      verdict = this.getMultipleCorrectWeightedVerdict(markedAnswer);
    }

    const { delta, currentStreak, updatedStreak } =
      this.sectionMarkingScheme.getDeltaAndUpdateStreak(
        this.marking,
        this.answerType,
        verdict,
        allowStreak
      );

    return { verdict, delta, currentStreak, updatedStreak };
  }

  /**
   * Get verdict for standard answer type
   */
  private getStandardVerdict(markedAnswer: string): string {
    const allowedAnswer = this.answerItem as string;
    if (markedAnswer === this.emptyValue) {
      return Verdict.UNMARKED;
    }
    if (markedAnswer === allowedAnswer) {
      return Verdict.ANSWER_MATCH;
    }
    return Verdict.NO_ANSWER_MATCH;
  }

  /**
   * Get verdict for multiple correct answer type
   */
  private getMultipleCorrectVerdict(markedAnswer: string): string {
    const allowedAnswers = this.answerItem as string[];
    if (markedAnswer === this.emptyValue) {
      return Verdict.UNMARKED;
    }
    if (allowedAnswers.includes(markedAnswer)) {
      return `${Verdict.ANSWER_MATCH}-${markedAnswer}`;
    }
    return Verdict.NO_ANSWER_MATCH;
  }

  /**
   * Get verdict for multiple correct weighted answer type
   */
  private getMultipleCorrectWeightedVerdict(markedAnswer: string): string {
    const weightedAnswers = this.answerItem as Array<[string, number]>;
    const allowedAnswers = weightedAnswers.map(([answer]) => answer);

    if (markedAnswer === this.emptyValue) {
      return Verdict.UNMARKED;
    }
    if (allowedAnswers.includes(markedAnswer)) {
      return `${Verdict.ANSWER_MATCH}-${markedAnswer}`;
    }
    return Verdict.NO_ANSWER_MATCH;
  }

  /**
   * Get section explanation string
   */
  getSectionExplanation(): string | null {
    if (
      this.answerType === AnswerType.STANDARD ||
      this.answerType === AnswerType.MULTIPLE_CORRECT
    ) {
      return this.sectionMarkingScheme.sectionKey;
    }
    if (this.answerType === AnswerType.MULTIPLE_CORRECT_WEIGHTED) {
      return `Custom Weights: ${JSON.stringify(this.marking)}`;
    }
    return null;
  }

  /**
   * Get matched set name
   */
  getMatchedSetName(): string {
    return this.sectionMarkingScheme.setName;
  }

  toString(): string {
    return String(this.answerItem);
  }
}

