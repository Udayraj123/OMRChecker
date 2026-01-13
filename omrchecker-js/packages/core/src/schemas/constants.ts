/**
 * Schema constants for OMRChecker.
 *
 * TypeScript port of src/schemas/constants.py
 * Maintains 1:1 correspondence with Python implementation.
 */

// Default keys
export const DEFAULT_SECTION_KEY = 'DEFAULT';
export const DEFAULT_SET_NAME = 'DEFAULT_SET';
export const BONUS_SECTION_PREFIX = 'BONUS';

/**
 * Verdict enum for answer matching
 */
export const Verdict = {
  ANSWER_MATCH: 'answer-match',
  NO_ANSWER_MATCH: 'no-answer-match',
  UNMARKED: 'unmarked',
} as const;

export const VERDICTS_IN_ORDER = [
  Verdict.ANSWER_MATCH,
  Verdict.NO_ANSWER_MATCH,
  Verdict.UNMARKED,
] as const;

/**
 * Schema verdict enum for evaluation
 */
export const SchemaVerdict = {
  CORRECT: 'correct',
  INCORRECT: 'incorrect',
  UNMARKED: 'unmarked',
} as const;

/**
 * Marking scheme type enum
 */
export const MarkingSchemeType = {
  DEFAULT: 'default',
  VERDICT_LEVEL_STREAK: 'verdict_level_streak',
  SECTION_LEVEL_STREAK: 'section_level_streak',
} as const;

export const MARKING_SCHEME_TYPES_IN_ORDER = [
  MarkingSchemeType.DEFAULT,
  MarkingSchemeType.VERDICT_LEVEL_STREAK,
  MarkingSchemeType.SECTION_LEVEL_STREAK,
] as const;

export const SCHEMA_VERDICTS_IN_ORDER = [
  SchemaVerdict.CORRECT,
  SchemaVerdict.INCORRECT,
  SchemaVerdict.UNMARKED,
] as const;

/**
 * Default format strings
 */
export const DEFAULT_SCORE_FORMAT_STRING = 'Score: {score}';
export const DEFAULT_ANSWERS_SUMMARY_FORMAT_STRING = SCHEMA_VERDICTS_IN_ORDER.map(
  (verdict) => `${verdict.charAt(0).toUpperCase() + verdict.slice(1)}: {${verdict}}`
).join(' ');

/**
 * Verdict to schema verdict mapping
 */
export const VERDICT_TO_SCHEMA_VERDICT = {
  [Verdict.ANSWER_MATCH]: SchemaVerdict.CORRECT,
  [Verdict.NO_ANSWER_MATCH]: SchemaVerdict.INCORRECT,
  [Verdict.UNMARKED]: SchemaVerdict.UNMARKED,
} as const;

/**
 * Answer type enum
 */
export const AnswerType = {
  // Standard answer type allows single correct answers. They can have multiple characters(multi-marked) as well.
  // Useful for any standard response e.g. 'A', '01', '99', 'AB', etc
  STANDARD: 'standard',
  // Multiple correct answer type covers multiple correct answers
  // Useful for ambiguous/bonus questions e.g. ['A', 'B'], ['1', '01'], ['A', 'B', 'AB'], etc
  MULTIPLE_CORRECT: 'multiple-correct',
  // Multiple correct weighted answer covers multiple answers with weights
  // Useful for partial marking e.g. [['A', 2], ['B', 0.5], ['AB', 2.5]], [['1', 0.5], ['01', 1]], etc
  MULTIPLE_CORRECT_WEIGHTED: 'multiple-correct-weighted',
} as const;

// Type exports for convenience
export type VerdictType = (typeof Verdict)[keyof typeof Verdict];
export type SchemaVerdictType = (typeof SchemaVerdict)[keyof typeof SchemaVerdict];
export type MarkingSchemeTypeValue = (typeof MarkingSchemeType)[keyof typeof MarkingSchemeType];
export type AnswerTypeValue = (typeof AnswerType)[keyof typeof AnswerType];

