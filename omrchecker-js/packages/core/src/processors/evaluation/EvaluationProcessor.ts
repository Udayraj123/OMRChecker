/**
 * Evaluation Processor for scoring OMR responses.
 *
 * TypeScript port of src/processors/evaluation/processor.py
 * Maintains 1:1 correspondence with Python implementation.
 *
 * This processor:
 * 1. Takes the concatenated OMR response
 * 2. Evaluates it against the answer key from evaluation config
 * 3. Computes score and evaluation metadata
 * 4. Stores results in context
 */

import { Processor, ProcessingContext } from '../base';
import { Logger } from '../../utils/logger';

const logger = new Logger('EvaluationProcessor');

/**
 * Configuration for evaluation, containing answer keys and marking schemes.
 *
 * Note: Full implementations are available in:
 * - EvaluationConfig.ts (ported from evaluation_config.py)
 * - SectionMarkingScheme.ts (ported from section_marking_scheme.py)
 * - AnswerMatcher.ts (ported from answer_matcher.py)
 */
export interface EvaluationConfig {
  /**
   * Get evaluation config for a specific response
   */
  getEvaluationConfigForResponse(
    concatenatedResponse: Record<string, string>,
    filePath: string
  ): EvaluationConfigForResponse | null;
}

/**
 * Evaluation configuration for a specific response.
 */
export interface EvaluationConfigForResponse {
  /**
   * Check if scoring explanation should be shown
   */
  getShouldExplainScoring(): boolean;

  /**
   * Get formatted answers summary
   */
  getFormattedAnswersSummary(formatString: string): [string, ...any[]];

  /**
   * Get the list of questions in order
   */
  questionsInOrder: string[];

  /**
   * Prepare and validate the OMR response
   */
  prepareAndValidateOmrResponse(
    concatenatedResponse: Record<string, string>,
    allowStreak: boolean
  ): void;

  /**
   * Match answer for a specific question
   */
  matchAnswerForQuestion(
    currentScore: number,
    question: string,
    markedAnswer: string
  ): [number, string, AnswerMatcher, string];

  /**
   * Get marking scheme for a question
   */
  getMarkingSchemeForQuestion(question: string): MarkingScheme;

  /**
   * Conditionally print explanation if configured
   */
  conditionallyPrintExplanation(): void;
}

/**
 * Answer matcher result.
 */
export interface AnswerMatcher {
  answerItem: any;
  answerType: string;
}

/**
 * Marking scheme for a question.
 */
export interface MarkingScheme {
  /**
   * Get bonus type for this marking scheme
   */
  getBonusType(): string | null;
}

/**
 * Metadata for a single question's evaluation.
 */
export interface QuestionMeta {
  question: string;
  questionVerdict: string;
  markedAnswer: string;
  delta: number;
  currentScore: number;
  answerItem: any;
  answerType: string;
  bonusType: string | null;
  questionSchemaVerdict: string;
}

/**
 * Evaluation metadata containing score and per-question details.
 */
export interface EvaluationMetaDict {
  score: number;
  questionsMeta: Record<string, QuestionMeta>;
  formattedAnswersSummary: string;
}

/**
 * Default format string for answers summary.
 */
const DEFAULT_ANSWERS_SUMMARY_FORMAT_STRING = 'correct: {correct} incorrect: {incorrect}';

/**
 * Evaluation Processor for scoring OMR responses.
 */
export class EvaluationProcessor extends Processor {
  private evaluationConfig: EvaluationConfig | null;

  /**
   * Initialize the Evaluation processor.
   *
   * @param evaluationConfig - The evaluation configuration containing answer keys
   */
  constructor(evaluationConfig: EvaluationConfig | null) {
    super();
    this.evaluationConfig = evaluationConfig;
  }

  /**
   * Get the name of this processor.
   *
   * @returns Processor name
   */
  getName(): string {
    return 'Evaluation';
  }

  /**
   * Execute evaluation and scoring.
   *
   * @param context - Processing context with OMR response
   * @returns Updated context with score and evaluation metadata
   */
  process(context: ProcessingContext): ProcessingContext {
    logger.debug(`Starting ${this.getName()} processor`);

    if (!this.evaluationConfig) {
      logger.debug('No evaluation config provided, skipping evaluation');
      context.score = 0;
      context.evaluationMeta = null;
      return context;
    }

    const concatenatedOmrResponse = context.omrResponse;
    const filePath = context.filePath;

    // Get the evaluation config for this specific response
    const evaluationConfigForResponse = this.evaluationConfig.getEvaluationConfigForResponse(
      concatenatedOmrResponse,
      filePath
    );

    if (evaluationConfigForResponse === null) {
      logger.debug('No matching evaluation config for this response');
      context.score = 0;
      context.evaluationMeta = null;
      return context;
    }

    // Log the read response if not explaining scoring
    if (!evaluationConfigForResponse.getShouldExplainScoring()) {
      logger.info(`Read Response: \n${JSON.stringify(concatenatedOmrResponse, null, 2)}`);
    }

    // Evaluate the response
    const [score, evaluationMeta] = this.evaluateConcatenatedResponse(
      concatenatedOmrResponse,
      evaluationConfigForResponse
    );

    // Get formatted answers summary
    const [defaultAnswersSummary] = evaluationConfigForResponse.getFormattedAnswersSummary(
      DEFAULT_ANSWERS_SUMMARY_FORMAT_STRING
    );

    // Store results in context
    context.score = score;
    context.evaluationMeta = evaluationMeta;
    context.evaluationConfigForResponse = evaluationConfigForResponse;
    context.defaultAnswersSummary = defaultAnswersSummary;

    logger.debug(`Completed ${this.getName()} processor with score: ${score}`);

    return context;
  }

  /**
   * Evaluate concatenated response against evaluation config.
   *
   * This is a utility function to calculate score and metadata.
   * Port of evaluate_concatenated_response from evaluation_meta.py
   *
   * @param concatenatedResponse - The OMR response dictionary
   * @param evaluationConfigForResponse - Evaluation config for this response
   * @returns Tuple of [score, evaluation metadata]
   */
  private evaluateConcatenatedResponse(
    concatenatedResponse: Record<string, string>,
    evaluationConfigForResponse: EvaluationConfigForResponse
  ): [number, EvaluationMetaDict] {
    // Prepare and validate the response
    evaluationConfigForResponse.prepareAndValidateOmrResponse(concatenatedResponse, true);

    // Initialize evaluation metadata
    const evaluationMeta: {
      score: number;
      questionsMeta: Record<string, QuestionMeta>;
    } = {
      score: 0.0,
      questionsMeta: {},
    };

    // Evaluate each question
    for (const question of evaluationConfigForResponse.questionsInOrder) {
      const markedAnswer = concatenatedResponse[question];

      const [delta, questionVerdict, answerMatcher, questionSchemaVerdict] =
        evaluationConfigForResponse.matchAnswerForQuestion(
          evaluationMeta.score,
          question,
          markedAnswer
        );

      const markingScheme = evaluationConfigForResponse.getMarkingSchemeForQuestion(question);
      const bonusType = markingScheme.getBonusType();

      evaluationMeta.score += delta;

      // Create question metadata
      const questionMeta: QuestionMeta = {
        question,
        questionVerdict,
        markedAnswer,
        delta,
        currentScore: evaluationMeta.score,
        answerItem: answerMatcher.answerItem,
        answerType: answerMatcher.answerType,
        bonusType,
        questionSchemaVerdict,
      };

      evaluationMeta.questionsMeta[question] = questionMeta;
    }

    // Print explanation if configured
    evaluationConfigForResponse.conditionallyPrintExplanation();

    // Get formatted answers summary
    const [formattedAnswersSummary] = evaluationConfigForResponse.getFormattedAnswersSummary(
      DEFAULT_ANSWERS_SUMMARY_FORMAT_STRING
    );

    return [
      evaluationMeta.score,
      {
        score: evaluationMeta.score,
        questionsMeta: evaluationMeta.questionsMeta,
        formattedAnswersSummary,
      },
    ];
  }
}

