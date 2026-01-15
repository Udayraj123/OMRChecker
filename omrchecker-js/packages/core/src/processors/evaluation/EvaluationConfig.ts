/**
 * Evaluation configuration - Manages conditional sets and default evaluation
 * Port of Python's src/processors/evaluation/evaluation_config.py
 */

import { Logger } from '../../utils/logger';
import { deepMerge } from '../../utils/object';
import { EvaluationConfigForSet } from './EvaluationConfigForSet';
import { DEFAULT_SECTION_KEY } from './SectionMarkingScheme';

const logger = new Logger('EvaluationConfig');

/**
 * Default set name
 */
export const DEFAULT_SET_NAME = 'DEFAULT';

/**
 * Conditional set configuration
 */
export interface ConditionalSet {
  name: string;
  matcher: {
    formatString: string;
    matchRegex: string;
  };
  evaluation: EvaluationJSON;
}

/**
 * Evaluation JSON structure
 */
export interface EvaluationJSON {
  source_type?: 'local' | 'csv' | 'image_and_csv';
  options?: {
    questions_in_order?: string[];
    answers_in_order?: any[];
    answer_key_csv_path?: string;
    answer_key_image_path?: string;
  };
  marking_schemes?: {
    [sectionKey: string]: any;
  };
  outputs_configuration?: {
    draw_answers_summary?: any;
    draw_detected_bubble_texts?: any;
    draw_question_verdicts?: any;
    draw_score?: any;
    should_explain_scoring?: boolean;
    should_export_explanation_csv?: boolean;
  };
  conditional_sets?: ConditionalSet[];
}

/**
 * Evaluation configuration class
 */
export class EvaluationConfig {
  public path: string;
  public conditionalSets: Array<[string, ConditionalSet['matcher']]>;
  public defaultEvaluationConfig: EvaluationConfigForSet;
  public setMapping: Record<string, EvaluationConfigForSet>;
  public excludeFiles: string[];
  private emptyValue: string;

  constructor(
    _currDir: string,
    localEvaluationPath: string,
    evaluationJson: EvaluationJSON,
    template: any,
    _tuningConfig: any
  ) {
    this.path = localEvaluationPath;
    this.emptyValue = template?.global_empty_val || '';

    // Extract conditional sets
    const conditionalSets = evaluationJson.conditional_sets || [];
    const defaultEvaluationJsonCopy = { ...evaluationJson };
    delete defaultEvaluationJsonCopy.conditional_sets;

    this.conditionalSets = conditionalSets.map(set => [set.name, set.matcher]);
    this.validateConditionalSets();

    // Create default evaluation config using EvaluationConfigForSet
    const options = defaultEvaluationJsonCopy.options || {};
    const markingSchemes = defaultEvaluationJsonCopy.marking_schemes || {
      [DEFAULT_SECTION_KEY]: {
        correct: 1,
        incorrect: 0,
        unmarked: 0,
      },
    };

    this.defaultEvaluationConfig = new EvaluationConfigForSet(
      DEFAULT_SET_NAME,
      options,
      markingSchemes,
      this.emptyValue
    );

    const partialDefaultEvaluationJson = {
      outputs_configuration: defaultEvaluationJsonCopy.outputs_configuration,
    };

    this.excludeFiles = this.defaultEvaluationConfig.getExcludeFiles();
    this.setMapping = {};

    // Process conditional sets
    for (const conditionalSet of conditionalSets) {
      const setName = conditionalSet.name;
      const evaluationJsonForSet = conditionalSet.evaluation;

      logger.debug(`Processing conditional set: ${setName}`);

      // Merge configurations
      const mergedEvaluationJson = deepMerge(
        partialDefaultEvaluationJson,
        evaluationJsonForSet
      );

      // Create EvaluationConfigForSet instance for this set
      const setOptions = mergedEvaluationJson.options || {};
      const setMarkingSchemes = mergedEvaluationJson.marking_schemes || markingSchemes;

      const evaluationConfigForSet = new EvaluationConfigForSet(
        setName,
        setOptions,
        setMarkingSchemes,
        this.emptyValue,
        this.defaultEvaluationConfig
      );

      this.setMapping[setName] = evaluationConfigForSet;
      this.excludeFiles.push(...evaluationConfigForSet.getExcludeFiles());
    }
  }

  /**
   * Get excluded files list
   */
  getExcludeFiles(): string[] {
    return this.excludeFiles;
  }

  /**
   * Validate conditional sets (no duplicate names)
   */
  private validateConditionalSets(): void {
    const allNames = new Set<string>();

    for (const [name] of this.conditionalSets) {
      if (allNames.has(name)) {
        throw new Error(
          `Repeated set name '${name}' in conditional_sets in the evaluation.json: ${this.path}`
        );
      }
      allNames.add(name);
    }
  }

  /**
   * Get evaluation config for a specific response
   */
  getEvaluationConfigForResponse(
    concatenatedResponse: Record<string, string>,
    filePath: string
  ): EvaluationConfigForSet {
    const matchedKey = this.getMatchingSet(concatenatedResponse, filePath);

    if (matchedKey === null) {
      return this.defaultEvaluationConfig;
    }

    return this.setMapping[matchedKey];
  }

  /**
   * Get matching set based on response and file path
   */
  getMatchingSet(
    concatenatedResponse: Record<string, string>,
    filePath: string
  ): string | null {
    const fileName = filePath.split('/').pop() || '';

    const formattingFields = {
      ...concatenatedResponse,
      file_path: filePath,
      file_name: fileName,
    };

    // Loop through all sets and return first matched
    for (const [name, matcher] of this.conditionalSets) {
      const { formatString, matchRegex } = matcher;

      try {
        // Simple string replacement for format string
        let formattedString = formatString;
        for (const [key, value] of Object.entries(formattingFields)) {
          formattedString = formattedString.replace(`{${key}}`, String(value));
        }

        // Test regex match
        const regex = new RegExp(matchRegex);
        if (regex.test(formattedString)) {
          return name;
        }
      } catch (error: unknown) {
        logger.error(`Error matching set '${name}':`, error instanceof Error ? error.message : String(error));
        return null;
      }
    }

    return null;
  }

  toString(): string {
    return this.path;
  }
}

