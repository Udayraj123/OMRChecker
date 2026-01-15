/**
 * Evaluation configuration for a specific set of responses
 * Simplified port of Python's src/processors/evaluation/evaluation_config_for_set.py
 */

import { AnswerMatcher, AnswerType, SchemaVerdict } from './AnswerMatcher';
import { SectionMarkingScheme, DEFAULT_SECTION_KEY } from './SectionMarkingScheme';
import { Logger } from '../../utils/logger';

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

    // Parse answers and create answer matchers
    this.questionToAnswerMatcher = this.parseAnswersAndMapQuestions();

    this.schemaVerdictCounts = {
      [SchemaVerdict.CORRECT]: 0,
      [SchemaVerdict.INCORRECT]: 0,
      [SchemaVerdict.UNMARKED]: 0,
    };

    this.shouldExplainScoring = false;
    this.shouldExportExplanationCsv = false;

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
}

