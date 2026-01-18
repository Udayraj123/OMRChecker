/**
 * Evaluation configuration for a specific set of responses
 * Simplified port of Python's src/processors/evaluation/evaluation_config_for_set.py
 */

import { AnswerMatcher, AnswerType, SchemaVerdict, Verdict } from './AnswerMatcher';
import { SectionMarkingScheme, DEFAULT_SECTION_KEY } from './SectionMarkingScheme';
import { Logger } from '../../utils/logger';
import { CLR_BLACK, CLR_WHITE } from '../../utils/constants';
import { FieldDefinitionError, ConfigError, OMRCheckerError } from '../../core/exceptions';
import { SCHEMA_VERDICTS_IN_ORDER } from '../../schemas/constants';
import type {
  DrawQuestionVerdictsConfig,
  DrawAnswerGroupsConfig,
  DrawDetectedBubbleTextsConfig,
} from '../../schemas/models/evaluation';

const logger = new Logger('EvaluationConfigForSet');

/**
 * Options for evaluation configuration
 */
export interface EvaluationOptions {
  questions_in_order?: string[];
  answers_in_order?: any[];
}

/**
 * Outputs configuration for drawing and display
 */
export interface OutputsConfiguration {
  draw_answers_summary?: {
    answers_summary_format_string?: string;
    position?: [number, number];
    size?: number;
  };
  draw_score?: {
    score_format_string?: string;
    position?: [number, number];
    size?: number;
  };
  should_explain_scoring?: boolean;
  should_export_explanation_csv?: boolean;
}

/**
 * Evaluation configuration for a specific set
 */
export class EvaluationConfigForSet {
  public setName: string;
  public questionsInOrder: string[];
  public answersInOrder: any[];
  public sectionMarkingSchemes: Record<string, SectionMarkingScheme>;
  public defaultMarkingScheme!: SectionMarkingScheme;
  public questionToScheme: Record<string, SectionMarkingScheme>;
  public questionToAnswerMatcher: Record<string, AnswerMatcher>;
  public schemaVerdictCounts: Record<SchemaVerdict, number>;
  public hasCustomMarking: boolean;
  public hasStreakMarking: boolean;
  public allowStreak: boolean;
  public hasConditionalSets: boolean;
  public shouldExplainScoring: boolean;
  public shouldExportExplanationCsv: boolean;
  public drawQuestionVerdicts?: DrawQuestionVerdictsConfig;
  public drawAnswerGroups?: DrawAnswerGroupsConfig;
  public drawDetectedBubbleTexts?: DrawDetectedBubbleTextsConfig;
  public drawAnswersSummary?: {
    enabled?: boolean;
    answers_summary_format_string?: string;
    position?: [number, number];
    size?: number;
  };
  public drawScore?: {
    enabled?: boolean;
    score_format_string?: string;
    position?: [number, number];
    size?: number;
  };
  public verdictColors?: Record<string, [number, number, number]>;
  public verdictSymbolColors?: Record<string, [number, number, number]>;

  constructor(
    setName: string,
    options: EvaluationOptions,
    markingSchemes: Record<string, any>,
    emptyValue: string,
    parentEvaluationConfig?: EvaluationConfigForSet
  ) {
    this.setName = setName;
    this.hasCustomMarking = false;
    this.hasStreakMarking = false;
    this.allowStreak = false;
    this.hasConditionalSets = parentEvaluationConfig !== undefined;

    // Parse questions and answers
    const { questionsInOrder, answersInOrder } = this.parseLocalQuestionAnswers(options);

    // Merge with parent if exists
    if (parentEvaluationConfig) {
      const merged = this.mergeWithParent(
        parentEvaluationConfig,
        questionsInOrder,
        answersInOrder
      );
      this.questionsInOrder = merged.questions;
      this.answersInOrder = merged.answers;
    } else {
      this.questionsInOrder = questionsInOrder;
      this.answersInOrder = answersInOrder;
    }

    this.validateQuestions();

    // Parse marking schemes
    this.sectionMarkingSchemes = {};
    this.questionToScheme = {};

    for (const [sectionKey, sectionScheme] of Object.entries(markingSchemes)) {
      const sectionMarkingScheme = new SectionMarkingScheme(
        sectionKey,
        sectionScheme,
        this.setName,
        emptyValue
      );

      if (sectionKey === DEFAULT_SECTION_KEY) {
        this.defaultMarkingScheme = sectionMarkingScheme;
      } else {
        this.sectionMarkingSchemes[sectionKey] = sectionMarkingScheme;
        if (sectionMarkingScheme.questions) {
          for (const q of sectionMarkingScheme.questions) {
            this.questionToScheme[q] = sectionMarkingScheme;
          }
        }
        this.hasCustomMarking = true;
      }
    }

    // Validate marking schemes
    this.validateMarkingSchemes();

    // Parse answers and create answer matchers
    this.questionToAnswerMatcher = this.parseAnswersAndMapQuestions();

    // Note: validateAnswers requires tuningConfig, which may not be available in constructor
    // It should be called separately if needed

    this.schemaVerdictCounts = {
      [SchemaVerdict.CORRECT]: 0,
      [SchemaVerdict.INCORRECT]: 0,
      [SchemaVerdict.UNMARKED]: 0,
    };

    this.shouldExplainScoring = false;
    this.shouldExportExplanationCsv = false;

    // Note: validateFormatStrings requires outputs_configuration, which may not be available in constructor
    // It should be called separately if needed

    this.resetEvaluation();
  }

  /**
   * Parse local question/answer configuration
   */
  private parseLocalQuestionAnswers(
    options: EvaluationOptions
  ): { questionsInOrder: string[]; answersInOrder: any[] } {
    const questionsInOrder = options.questions_in_order || [];
    const answersInOrder = options.answers_in_order || [];

    return { questionsInOrder, answersInOrder };
  }

  /**
   * Merge questions and answers with parent config
   */
  private mergeWithParent(
    parent: EvaluationConfigForSet,
    localQuestions: string[],
    localAnswers: any[]
  ): { questions: string[]; answers: any[] } {
    const localMap = new Map<string, any>();
    for (let i = 0; i < localQuestions.length; i++) {
      localMap.set(localQuestions[i], localAnswers[i]);
    }

    const mergedQuestions: string[] = [];
    const mergedAnswers: any[] = [];

    // Add parent questions, override with local if exists
    for (let i = 0; i < parent.questionsInOrder.length; i++) {
      const question = parent.questionsInOrder[i];
      mergedQuestions.push(question);
      if (localMap.has(question)) {
        mergedAnswers.push(localMap.get(question));
      } else {
        mergedAnswers.push(parent.answersInOrder[i]);
      }
    }

    const parentQuestions = new Set(parent.questionsInOrder);

    // Add new local questions not in parent
    for (let i = 0; i < localQuestions.length; i++) {
      const question = localQuestions[i];
      if (!parentQuestions.has(question)) {
        mergedQuestions.push(question);
        mergedAnswers.push(localAnswers[i]);
      }
    }

    return { questions: mergedQuestions, answers: mergedAnswers };
  }

  /**
   * Validate questions configuration
   */
  private validateQuestions(): void {
    if (this.questionsInOrder.length !== this.answersInOrder.length) {
      throw new Error(
        `Unequal lengths for questions_in_order (${this.questionsInOrder.length}) ` +
        `and answers_in_order (${this.answersInOrder.length})`
      );
    }
  }

  /**
   * Parse answers and create answer matcher map
   */
  private parseAnswersAndMapQuestions(): Record<string, AnswerMatcher> {
    const questionToAnswerMatcher: Record<string, AnswerMatcher> = {};

    for (let i = 0; i < this.questionsInOrder.length; i++) {
      const question = this.questionsInOrder[i];
      const answerItem = this.answersInOrder[i];
      const sectionMarkingScheme = this.getMarkingSchemeForQuestion(question);
      const answerMatcher = new AnswerMatcher(answerItem, sectionMarkingScheme);

      questionToAnswerMatcher[question] = answerMatcher;

      if (answerMatcher.answerType === AnswerType.MULTIPLE_CORRECT_WEIGHTED) {
        this.hasCustomMarking = true;
      }
    }

    return questionToAnswerMatcher;
  }

  /**
   * Get marking scheme for a specific question
   */
  getMarkingSchemeForQuestion(question: string): SectionMarkingScheme {
    return this.questionToScheme[question] || this.defaultMarkingScheme;
  }

  /**
   * Validate marking schemes configuration.
   *
   * Port of Python's validate_marking_schemes method.
   * Ensures no overlapping questions between sections and all questions have answer keys.
   */
  private validateMarkingSchemes(): void {
    const sectionQuestions = new Set<string>();

    for (const [sectionKey, sectionScheme] of Object.entries(
      this.sectionMarkingSchemes
    )) {
      if (sectionKey === DEFAULT_SECTION_KEY) {
        continue;
      }

      const currentSet = new Set(sectionScheme.questions || []);

      // Check for overlapping questions
      for (const question of currentSet) {
        if (sectionQuestions.has(question)) {
          throw new FieldDefinitionError(
            `Section '${sectionKey}' has overlapping question(s) with other sections locally`,
            { section_key: sectionKey }
          );
        }
      }

      // Union with existing questions
      for (const question of currentSet) {
        sectionQuestions.add(question);
      }
    }

    // Check that all questions in marking schemes have answer keys
    const allQuestions = new Set(this.questionsInOrder);
    const missingQuestions = Array.from(sectionQuestions).filter(
      (q) => !allQuestions.has(q)
    );

    if (missingQuestions.length > 0) {
      missingQuestions.sort();
      logger.error(`Missing answer key for: ${missingQuestions.join(', ')}`);
      throw new OMRCheckerError(
        'Some questions are missing in the answer key for the given marking scheme(s)',
        { missing_questions: missingQuestions }
      );
    }
  }

  /**
   * Validate answers configuration.
   *
   * Port of Python's validate_answers method.
   * Ensures answer keys don't contain multi-marked answers when filter_out_multimarked_files is enabled.
   */
  validateAnswers(tuningConfig?: any): void {
    if (!tuningConfig?.outputs?.filterOutMultimarkedFiles) {
      return;
    }

    let containsMultiMarkedAnswer = false;

    for (let i = 0; i < this.questionsInOrder.length; i++) {
      const question = this.questionsInOrder[i];
      const answerItem = this.answersInOrder[i];
      const answerType = this.questionToAnswerMatcher[question].answerType;

      switch (answerType) {
        case AnswerType.STANDARD:
          if (typeof answerItem === 'string' && answerItem.length > 1) {
            containsMultiMarkedAnswer = true;
          }
          break;

        case AnswerType.MULTIPLE_CORRECT:
          if (Array.isArray(answerItem)) {
            for (const singleAnswer of answerItem) {
              if (typeof singleAnswer === 'string' && singleAnswer.length > 1) {
                containsMultiMarkedAnswer = true;
                break;
              }
            }
          }
          break;

        case AnswerType.MULTIPLE_CORRECT_WEIGHTED:
          if (Array.isArray(answerItem)) {
            for (const item of answerItem) {
              if (Array.isArray(item) && item.length >= 1) {
                const singleAnswer = item[0];
                if (typeof singleAnswer === 'string' && singleAnswer.length > 1) {
                  containsMultiMarkedAnswer = true;
                  break;
                }
              }
            }
          }
          break;
      }

      if (containsMultiMarkedAnswer) {
        throw new ConfigError(
          'Provided answer key contains multiple correct answer(s), but config.filter_out_multimarked_files is True. Scoring will get skipped.',
          { filter_out_multimarked_files: true }
        );
      }
    }
  }

  /**
   * Validate format strings for drawing.
   *
   * Port of Python's validate_format_strings method.
   * Ensures format strings are valid and contain only allowed variables.
   */
  validateFormatStrings(): void {
    // Validate answers summary format string
    if (this.drawAnswersSummary?.answers_summary_format_string) {
      const answersSummaryFormatString =
        this.drawAnswersSummary.answers_summary_format_string;

      try {
        // Try to format with schema verdict counts
        const formatArgs: Record<string, number> = {};
        for (const verdict of SCHEMA_VERDICTS_IN_ORDER) {
          formatArgs[verdict] = this.schemaVerdictCounts[verdict] || 0;
        }
        answersSummaryFormatString.replace(
          /\{(\w+)\}/g,
          (match, key) => {
            if (!(key in formatArgs)) {
              throw new Error(`Unknown variable: ${key}`);
            }
            return match;
          }
        );
      } catch (error) {
        throw new ConfigError(
          `The format string should contain only allowed variables ${SCHEMA_VERDICTS_IN_ORDER.join(', ')}. answers_summary_format_string=${answersSummaryFormatString}`,
          {
            answers_summary_format_string: answersSummaryFormatString,
            allowed_variables: SCHEMA_VERDICTS_IN_ORDER,
          }
        );
      }
    }

    // Validate score format string
    if (this.drawScore?.score_format_string) {
      const scoreFormatString = this.drawScore.score_format_string;

      try {
        // Try to format with score=0
        scoreFormatString.replace(/\{(\w+)\}/g, (match, key) => {
          if (key !== 'score') {
            throw new Error(`Unknown variable: ${key}`);
          }
          return match;
        });
      } catch (error) {
        throw new ConfigError(
          `The format string should contain only allowed variables ['score']. score_format_string=${scoreFormatString}`,
          {
            score_format_string: scoreFormatString,
            allowed_variables: ['score'],
          }
        );
      }
    }
  }

  /**
   * Prepare and validate OMR response
   */
  prepareAndValidateOmrResponse(
    concatenatedResponse: Record<string, string>,
    allowStreak: boolean = false
  ): void {
    this.allowStreak = allowStreak;
    this.resetEvaluation();

    const omrResponseKeys = new Set(Object.keys(concatenatedResponse));
    const allQuestions = new Set(this.questionsInOrder);

    const missingQuestions = Array.from(allQuestions).filter(
      (q) => !omrResponseKeys.has(q)
    );

    if (missingQuestions.length > 0) {
      logger.error(`Missing OMR response for: ${missingQuestions.join(', ')}`);
      throw new Error(
        `Some question keys are missing in the OMR response: ${missingQuestions.join(', ')}`
      );
    }
  }

  /**
   * Match answer for a specific question and return verdict/delta
   */
  matchAnswerForQuestion(
    currentScore: number,
    question: string,
    markedAnswer: string
  ): {
    delta: number;
    verdict: string;
    answerMatcher: AnswerMatcher;
    schemaVerdict: SchemaVerdict;
  } {
    const answerMatcher = this.questionToAnswerMatcher[question];
    const { verdict, delta } =
      answerMatcher.getVerdictMarking(markedAnswer, this.allowStreak);

    const schemaVerdict = AnswerMatcher.getSchemaVerdict(
      answerMatcher.answerType,
      verdict,
      delta
    );

    this.schemaVerdictCounts[schemaVerdict]++;

    // TODO: Add explanation logging if needed
    if (this.shouldExplainScoring) {
      logger.debug(
        `Question ${question}: ${verdict}, delta=${delta}, score=${currentScore + delta}`
      );
    }

    return { delta, verdict, answerMatcher, schemaVerdict };
  }

  /**
   * Reset evaluation state
   */
  resetEvaluation(): void {
    this.schemaVerdictCounts = {
      [SchemaVerdict.CORRECT]: 0,
      [SchemaVerdict.INCORRECT]: 0,
      [SchemaVerdict.UNMARKED]: 0,
    };

    // Reset streaks in all section schemes
    for (const scheme of Object.values(this.sectionMarkingSchemes)) {
      scheme.resetAllStreaks();
    }
  }

  /**
   * Get exclude files list (empty for simplified version)
   */
  getExcludeFiles(): string[] {
    return [];
  }

  /**
   * Get evaluation meta for question (verdict colors, symbols, etc.).
   *
   * Port of Python's get_evaluation_meta_for_question method.
   *
   * @param questionMeta - Question metadata
   * @param isFieldMarked - Whether the field is marked
   * @param imageType - Image type ('GRAYSCALE' or 'COLORED')
   * @returns Tuple of [symbol, color, symbolColor, thicknessFactor]
   */
  getEvaluationMetaForQuestion(
    questionMeta: any,
    isFieldMarked: boolean,
    imageType: 'GRAYSCALE' | 'COLORED'
  ): [string, string | [number, number, number], string | [number, number, number], number] {
    const symbolPositive = '+';
    const symbolNegative = '-';
    const symbolNeutral = 'o';
    const symbolBonus = '*';
    const symbolUnmarked = '';
    const thicknessFactor = 1 / 12;

    const bonusType = questionMeta.bonus_type;
    const questionVerdict = questionMeta.question_verdict;
    const questionSchemaVerdict = questionMeta.question_schema_verdict;
    const delta = questionMeta.delta || 0;

    const colorCorrect = this.verdictColors?.correct || [0, 255, 0]; // Green
    const colorIncorrect = this.verdictColors?.incorrect || [0, 0, 255]; // Red
    const colorNeutral =
      this.verdictColors?.neutral || this.verdictColors?.incorrect || [0, 0, 255];
    const colorBonus = this.verdictColors?.bonus || [255, 221, 0]; // Cyan-like

    const symbolColorPositive =
      this.verdictSymbolColors?.positive || [0, 0, 0]; // Black
    const symbolColorNegative =
      this.verdictSymbolColors?.negative || [0, 0, 0]; // Black
    const symbolColorNeutral =
      this.verdictSymbolColors?.neutral || [0, 0, 0]; // Black
    const symbolColorBonus =
      this.verdictSymbolColors?.bonus || [0, 0, 0]; // Black

    let symbol: string;
    let color: string | [number, number, number];
    let symbolColor: string | [number, number, number];

    if (isFieldMarked) {
      // Always render symbol as per delta (regardless of bonus) for marked bubbles
      if (delta > 0) {
        symbol = symbolPositive;
      } else if (delta < 0) {
        symbol = symbolNegative;
      } else {
        symbol = symbolNeutral;
      }

      // Update colors for marked bubbles
      if (imageType === 'GRAYSCALE') {
        color = CLR_WHITE;
        symbolColor = CLR_BLACK;
      } else {
        // Apply colors as per delta
        if (delta > 0) {
          color = colorCorrect;
          symbolColor = symbolColorPositive;
        } else if (delta < 0) {
          color = colorIncorrect;
          symbolColor = symbolColorNegative;
        } else {
          color = colorNeutral;
          symbolColor = symbolColorNeutral;
        }

        // Override bonus colors if bubble was marked but verdict was not correct
        if (
          bonusType !== undefined &&
          (questionVerdict === Verdict.UNMARKED ||
            questionVerdict === Verdict.NO_ANSWER_MATCH)
        ) {
          color = colorBonus;
          symbolColor = symbolColorBonus;
        }
      }
    } else {
      symbol = symbolUnmarked;
      // In case of unmarked bubbles, we draw the symbol only if bonus type is BONUS_FOR_ALL
      if (bonusType === 'BONUS_FOR_ALL') {
        symbol = symbolPositive;
      } else if (bonusType === 'BONUS_ON_ATTEMPT') {
        if (questionSchemaVerdict === Verdict.UNMARKED) {
          // Case of bonus on attempt with blank question - show neutral symbol on all bubbles(on attempt)
          symbol = symbolNeutral;
        } else {
          // Case of bonus on attempt with one or more marked bubbles - show bonus symbol on remaining bubbles
          symbol = symbolBonus;
        }
      }

      // Apply bonus colors for all bubbles
      if (bonusType !== undefined) {
        color = colorBonus;
        symbolColor = symbolColorBonus;
      } else {
        color = '';
        symbolColor = '';
      }
    }

    return [symbol, color, symbolColor, thicknessFactor];
  }

  /**
   * Get formatted answers summary for drawing.
   *
   * Port of Python's get_formatted_answers_summary method.
   *
   * @param answersSummaryFormatString - Optional format string override
   * @returns Tuple of [formattedText, position, size, thickness]
   */
  getFormattedAnswersSummary(
    answersSummaryFormatString?: string
  ): [string, [number, number], number, number] {
    const formatString =
      answersSummaryFormatString ||
      this.drawAnswersSummary?.answers_summary_format_string ||
      'Correct: {correct} Incorrect: {incorrect} Unmarked: {unmarked}';

    const answersFormat = formatString
      .replace('{correct}', String(this.schemaVerdictCounts[SchemaVerdict.CORRECT]))
      .replace('{incorrect}', String(this.schemaVerdictCounts[SchemaVerdict.INCORRECT]))
      .replace('{unmarked}', String(this.schemaVerdictCounts[SchemaVerdict.UNMARKED]));

    const position =
      this.drawAnswersSummary?.position || ([200, 600] as [number, number]);
    const size = this.drawAnswersSummary?.size || 1.0;
    const thickness = Math.floor(size * 2);

    return [answersFormat, position, size, thickness];
  }

  /**
   * Get formatted score for drawing.
   *
   * Port of Python's get_formatted_score method.
   *
   * @param score - Score value
   * @returns Tuple of [formattedText, position, size, thickness]
   */
  getFormattedScore(score: number): [string, [number, number], number, number] {
    const scoreFormatString =
      this.drawScore?.score_format_string || 'Score: {score}';
    const scoreFormat = scoreFormatString.replace(
      '{score}',
      String(Math.round(score * 100) / 100)
    );

    const position = this.drawScore?.position || ([200, 200] as [number, number]);
    const size = this.drawScore?.size || 1.5;
    const thickness = Math.floor(size * 2);

    return [scoreFormat, position, size, thickness];
  }
}

