/**
 * Evaluation Schema for OMRChecker.
 *
 * TypeScript port of src/schemas/evaluation_schema.py
 * Maintains 1:1 correspondence with Python implementation.
 *
 * This schema validates evaluation configuration files that define:
 * - Answer sources (CSV, image+CSV, or local)
 * - Marking schemes and scoring rules
 * - Output visualization configurations
 */

import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import { loadCommonDefs } from './common';
import {
  DEFAULT_SECTION_KEY,
  SCHEMA_VERDICTS_IN_ORDER,
  MARKING_SCHEME_TYPES_IN_ORDER,
  MarkingSchemeType,
} from './constants';

const ajv = new Ajv({ allErrors: true, strict: false });
addFormats(ajv);

// Marking score regex pattern
const MARKING_SCORE_REGEX = '-?(\\d+)(/(\\d+))?';

/**
 * Marking score without streak support
 */
const markingScoreWithoutStreak = {
  oneOf: [
    {
      description: 'The marking score as a string. We can pass natural fractions as well',
      type: 'string',
      pattern: MARKING_SCORE_REGEX,
    },
    {
      description: 'The marking score as a number. It can be negative as well',
      type: 'number',
    },
  ],
};

/**
 * Marking score with streak support (includes array for consecutive streak marking)
 */
const markingScoreWithStreak = {
  oneOf: [
    ...markingScoreWithoutStreak.oneOf,
    {
      description:
        'An array will be used for consecutive streak based marking. Note: original order from questions_in_order will be used.',
      type: 'array',
      items: {
        oneOf: [...markingScoreWithoutStreak.oneOf],
      },
    },
  ],
};

/**
 * Section marking object without streak support
 */
const sectionMarkingWithoutStreakObject = {
  description: 'The marking object describes verdict-wise score deltas',
  required: SCHEMA_VERDICTS_IN_ORDER,
  type: 'object',
  additionalProperties: false,
  properties: Object.fromEntries(
    SCHEMA_VERDICTS_IN_ORDER.map((verdict) => [
      verdict,
      { $ref: '#/$defs/marking_score_without_streak' },
    ])
  ),
};

/**
 * Section marking object with streak support
 */
const sectionMarkingWithStreakObject = {
  ...sectionMarkingWithoutStreakObject,
  properties: Object.fromEntries(
    SCHEMA_VERDICTS_IN_ORDER.map((verdict) => [
      verdict,
      { $ref: '#/$defs/marking_score_with_streak' },
    ])
  ),
};

/**
 * Conditional marking type configurations
 */
const customSectionMarkingObjectConditions = [
  {
    if: { properties: { marking_type: { const: MarkingSchemeType.DEFAULT } } },
    then: {
      properties: {
        marking: { $ref: '#/$defs/section_marking_without_streak_object' },
      },
    },
  },
  {
    if: {
      properties: {
        marking_type: { const: MarkingSchemeType.SECTION_LEVEL_STREAK },
      },
    },
    then: {
      properties: {
        marking: { $ref: '#/$defs/section_marking_with_streak_object' },
      },
    },
  },
  {
    if: {
      properties: {
        marking_type: { const: MarkingSchemeType.VERDICT_LEVEL_STREAK },
      },
    },
    then: {
      properties: {
        marking: { $ref: '#/$defs/section_marking_with_streak_object' },
      },
    },
    else: {
      properties: {
        marking: { $ref: '#/$defs/section_marking_without_streak_object' },
      },
    },
  },
];

/**
 * Options for image and CSV source types
 */
const imageAndCsvOptions = {
  description: 'The options needed if source type is image and csv',
  required: ['answer_key_csv_path'],
  dependentRequired: {
    answer_key_image_path: ['answer_key_csv_path', 'questions_in_order'],
  },
  type: 'object',
  additionalProperties: false,
  properties: {
    answer_key_csv_path: {
      description: 'The path to the answer key csv relative to the evaluation.json file',
      type: 'string',
    },
    answer_key_image_path: {
      description: 'The path to the answer key image relative to the evaluation.json file',
      type: 'string',
    },
    questions_in_order: {
      $ref: '#/$defs/array_of_strings',
      description: 'An array of fields to treat as questions when the answer key image is provided',
    },
  },
};

/**
 * Options for local questions and answers
 */
const localQuestionsAndAnswersOptions = {
  description: 'This method allows setting questions and their answers within the evaluation file itself',
  additionalProperties: false,
  required: ['answers_in_order', 'questions_in_order'],
  type: 'object',
  properties: {
    questions_in_order: {
      $ref: '#/$defs/array_of_strings',
      description: 'An array of fields to treat as questions specified in an order to apply evaluation',
    },
    answers_in_order: {
      oneOf: [
        {
          description: 'An array of answers in the same order as provided array of questions',
          type: 'array',
          items: {
            oneOf: [
              {
                type: 'string',
              },
              {
                type: 'array',
                items: { type: 'string' },
                minItems: 2,
              },
              {
                type: 'array',
                items: {
                  type: 'array',
                  minItems: 2,
                  maxItems: 2,
                  prefixItems: [
                    { type: 'string' },
                    { $ref: '#/$defs/marking_score_without_streak' },
                  ],
                },
              },
            ],
          },
        },
      ],
    },
  },
};

/**
 * Common properties for evaluation schema
 */
const commonEvaluationSchemaProperties = {
  source_type: { type: 'string', enum: ['csv', 'image_and_csv', 'local'] },
  options: { type: 'object' },
  marking_schemes: {
    type: 'object',
    required: [DEFAULT_SECTION_KEY],
    patternProperties: {
      [`^${DEFAULT_SECTION_KEY}$`]: {
        $ref: '#/$defs/section_marking_without_streak_object',
      },
      [`^(?!${DEFAULT_SECTION_KEY}$).*`]: {
        description: 'A section that defines custom marking for a subset of the questions',
        additionalProperties: false,
        required: ['marking', 'questions'],
        type: 'object',
        properties: {
          marking_type: {
            type: 'string',
            enum: [...MARKING_SCHEME_TYPES_IN_ORDER],
          },
          questions: {
            oneOf: [
              { $ref: '#/$defs/field_string_type' },
              {
                type: 'array',
                items: { $ref: '#/$defs/field_string_type' },
              },
            ],
          },
          marking: true,
        },
        allOf: [...customSectionMarkingObjectConditions],
      },
    },
  },
  outputs_configuration: {
    description: 'The configuration for outputs produced from the evaluation',
    type: 'object',
    required: [],
    additionalProperties: false,
    properties: {
      should_explain_scoring: {
        description: 'Whether to print the table explaining question-wise verdicts',
        type: 'boolean',
      },
      should_export_explanation_csv: {
        description: 'Whether to export the explanation of evaluation results as a CSV file',
        type: 'boolean',
      },
      draw_score: {
        description: 'The configuration for drawing the final score',
        type: 'object',
        required: ['enabled'],
        additionalProperties: false,
        properties: {
          enabled: {
            description: 'The toggle for enabling the configuration',
            type: 'boolean',
          },
          position: {
            description: 'The position of the score box',
            $ref: '#/$defs/two_positive_integers',
          },
          score_format_string: {
            description: 'The format string to compose the score string. Supported variables - {score}',
            type: 'string',
          },
          size: {
            description: 'The font size for the score box',
            type: 'number',
          },
        },
        allOf: [
          {
            if: { properties: { enabled: { const: true } } },
            then: {
              required: ['position', 'score_format_string'],
            },
          },
        ],
      },
      draw_answers_summary: {
        description: 'The configuration for drawing the answers summary',
        type: 'object',
        required: ['enabled'],
        additionalProperties: false,
        properties: {
          enabled: {
            description: 'The toggle for enabling the configuration',
            type: 'boolean',
          },
          position: {
            description: 'The position of the answers summary box',
            $ref: '#/$defs/two_positive_integers',
          },
          answers_summary_format_string: {
            description: 'The format string to compose the answer summary. Supported variables - {correct}, {incorrect}, {unmarked}',
            type: 'string',
          },
          size: {
            description: 'The font size for the answers summary box',
            type: 'number',
          },
        },
        allOf: [
          {
            if: { properties: { enabled: { const: true } } },
            then: {
              required: ['position', 'answers_summary_format_string'],
            },
          },
        ],
      },
      draw_question_verdicts: {
        type: 'object',
        additionalProperties: false,
        required: ['enabled'],
        properties: {
          enabled: { type: 'boolean' },
          verdict_colors: {
            description: 'The mapping from delta sign notions to the corresponding colors',
            type: 'object',
            additionalProperties: false,
            required: ['correct', 'neutral', 'incorrect', 'bonus'],
            properties: {
              correct: {
                description: 'The color of the field box when delta > 0',
                $ref: '#/$defs/matplotlib_color',
              },
              neutral: {
                description: 'The color of the field box when delta == 0 (defaults to incorrect)',
                oneOf: [
                  { $ref: '#/$defs/matplotlib_color' },
                  { type: 'null' },
                ],
              },
              incorrect: {
                description: 'The color of the field box when delta < 0',
                $ref: '#/$defs/matplotlib_color',
              },
              bonus: {
                description: 'The color of the field box when delta > 0 and question is part of a bonus scheme',
                $ref: '#/$defs/matplotlib_color',
              },
            },
          },
          verdict_symbol_colors: {
            description: 'The mapping from verdict symbols(based on delta sign) to the corresponding colors',
            type: 'object',
            additionalProperties: false,
            required: ['positive', 'neutral', 'negative', 'bonus'],
            properties: {
              positive: {
                description: 'The color of \'+\' symbol when delta > 0',
                $ref: '#/$defs/matplotlib_color',
              },
              neutral: {
                description: 'The color of \'o\' symbol when delta == 0',
                $ref: '#/$defs/matplotlib_color',
              },
              negative: {
                description: 'The color of \'-\' symbol when delta < 0',
                $ref: '#/$defs/matplotlib_color',
              },
              bonus: {
                description: 'The color of \'*\' symbol when delta > 0 and question is part of a bonus scheme',
                $ref: '#/$defs/matplotlib_color',
              },
            },
          },
          draw_answer_groups: {
            type: 'object',
            additionalProperties: false,
            required: [],
            properties: {
              enabled: { type: 'boolean' },
              color_sequence: {
                type: 'array',
                items: {
                  $ref: '#/$defs/matplotlib_color',
                },
                minItems: 4,
                maxItems: 4,
              },
            },
            allOf: [
              {
                if: { properties: { enabled: { const: true } } },
                then: {
                  required: ['color_sequence'],
                },
              },
            ],
          },
        },
        allOf: [
          {
            if: { properties: { enabled: { const: true } } },
            then: {
              required: [
                'verdict_colors',
                'verdict_symbol_colors',
                'draw_answer_groups',
              ],
            },
          },
        ],
      },
      draw_detected_bubble_texts: {
        type: 'object',
        additionalProperties: false,
        required: ['enabled'],
        properties: { enabled: { type: 'boolean' } },
      },
    },
  },
};

/**
 * Conditions based on source type
 */
const commonEvaluationSchemaConditions = [
  {
    if: { properties: { source_type: { const: 'csv' } } },
    then: { properties: { options: { $ref: '#/$defs/image_and_csv_options' } } },
  },
  {
    if: { properties: { source_type: { const: 'image_and_csv' } } },
    then: { properties: { options: { $ref: '#/$defs/image_and_csv_options' } } },
  },
  {
    if: { properties: { source_type: { const: 'local' } } },
    then: {
      properties: {
        options: { $ref: '#/$defs/local_questions_and_answers_options' },
      },
    },
  },
];

/**
 * Main evaluation schema
 */
export const EVALUATION_SCHEMA = {
  // Removed $schema to avoid AJV validation issues
  $id: 'https://github.com/Udayraj123/OMRChecker/tree/master/src/schemas/evaluation-schema.json',
  $defs: {
    ...loadCommonDefs([
      'array_of_strings',
      'two_positive_integers',
      'field_string_type',
      'matplotlib_color',
    ]),
    marking_score_without_streak: markingScoreWithoutStreak,
    marking_score_with_streak: markingScoreWithStreak,
    section_marking_without_streak_object: sectionMarkingWithoutStreakObject,
    section_marking_with_streak_object: sectionMarkingWithStreakObject,
    image_and_csv_options: imageAndCsvOptions,
    local_questions_and_answers_options: localQuestionsAndAnswersOptions,
  },
  title: 'Evaluation Schema',
  description: 'The OMRChecker evaluation schema',
  type: 'object',
  required: ['source_type', 'options', 'marking_schemes'],
  additionalProperties: false,
  properties: {
    ...commonEvaluationSchemaProperties,
    conditional_sets: {
      description:
        'An array of answer sets with their conditions. These will override the default values in case of any conflict',
      type: 'array',
      items: {
        description:
          'Each item represents a conditional evaluation schema to apply for the given matcher',
        type: 'object',
        required: ['name', 'matcher', 'evaluation'],
        additionalProperties: false,
        properties: {
          name: { type: 'string' },
          matcher: {
            description: 'Mapping response fields from default layout to the set name',
            type: 'object',
            required: ['formatString', 'matchRegex'],
            additionalProperties: false,
            properties: {
              formatString: {
                description:
                  "Format string composed of the response variables to apply the regex on e.g. '{Roll}' or '{Roll}-{barcode}'",
                type: 'string',
              },
              matchRegex: {
                description:
                  "The regex to match on the composed field string e.g. to match a suffix value: '.*-SET1'",
                type: 'string',
                format: 'regex',
              },
            },
          },
          evaluation: {
            description:
              'The custom evaluation schema to apply if given matcher is satisfied',
            type: 'object',
            required: ['source_type', 'options', 'marking_schemes'],
            additionalProperties: false,
            properties: {
              ...commonEvaluationSchemaProperties,
            },
            allOf: [...commonEvaluationSchemaConditions],
          },
        },
      },
    },
  },
  allOf: [...commonEvaluationSchemaConditions],
};

/**
 * Compiled validator for evaluation schema
 */
export const validateEvaluationSchema = ajv.compile(EVALUATION_SCHEMA);

/**
 * Validate an evaluation configuration object
 *
 * @param config - The evaluation configuration to validate
 * @returns True if valid, throws error if invalid
 */
export function validateEvaluation(config: any): boolean {
  const valid = validateEvaluationSchema(config);
  if (!valid) {
    const errors = validateEvaluationSchema.errors;
    throw new Error(
      `Evaluation schema validation failed:\n${JSON.stringify(errors, null, 2)}`
    );
  }
  return true;
}

