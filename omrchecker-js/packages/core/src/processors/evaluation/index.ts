/**
 * Evaluation module exports
 */

// Main processor
export { EvaluationProcessor } from './EvaluationProcessor';

// Core evaluation classes
export { AnswerMatcher, AnswerType, Verdict, SchemaVerdict } from './AnswerMatcher';
export { SectionMarkingScheme, MarkingSchemeType, DEFAULT_SECTION_KEY, BONUS_SECTION_PREFIX } from './SectionMarkingScheme';
export { EvaluationConfig, DEFAULT_SET_NAME } from './EvaluationConfig';
export { EvaluationConfigForSet } from './EvaluationConfigForSet';

// Types
export type { SectionSchemeConfig } from './SectionMarkingScheme';
export type { EvaluationJSON, ConditionalSet } from './EvaluationConfig';
export type { EvaluationOptions, OutputsConfiguration } from './EvaluationConfigForSet';


