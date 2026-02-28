import { z } from 'zod';

// Draw Score Configuration
export const DrawScoreConfigSchema = z.object({
  enabled: z.boolean().default(false),
  position: z.array(z.number().int()).length(2).default([200, 200]),
  scoreFormatString: z.string().default('Score: {score}'),
  size: z.number().default(1.5),
});

export type DrawScoreConfig = z.infer<typeof DrawScoreConfigSchema>;

export function createDrawScoreConfig(data: Partial<DrawScoreConfig> = {}): DrawScoreConfig {
  return DrawScoreConfigSchema.parse({
    enabled: data.enabled ?? false,
    position: data.position ?? [200, 200],
    scoreFormatString: data.scoreFormatString ?? 'Score: {score}',
    size: data.size ?? 1.5,
  });
}

// Draw Answers Summary Configuration
export const DrawAnswersSummaryConfigSchema = z.object({
  enabled: z.boolean().default(false),
  position: z.array(z.number().int()).length(2).default([200, 600]),
  answersSummaryFormatString: z
    .string()
    .default('Correct: {correct} Incorrect: {incorrect} Unmarked: {unmarked}'),
  size: z.number().default(1.0),
});

export type DrawAnswersSummaryConfig = z.infer<typeof DrawAnswersSummaryConfigSchema>;

export function createDrawAnswersSummaryConfig(
  data: Partial<DrawAnswersSummaryConfig> = {}
): DrawAnswersSummaryConfig {
  return DrawAnswersSummaryConfigSchema.parse({
    enabled: data.enabled ?? false,
    position: data.position ?? [200, 600],
    answersSummaryFormatString:
      data.answersSummaryFormatString ??
      'Correct: {correct} Incorrect: {incorrect} Unmarked: {unmarked}',
    size: data.size ?? 1.0,
  });
}

// Draw Answer Groups Configuration
export const DrawAnswerGroupsConfigSchema = z.object({
  enabled: z.boolean().default(true),
  colorSequence: z
    .array(z.string())
    .default(['#8DFBC4', '#F7FB8D', '#8D9EFB', '#EA666F']),
});

export type DrawAnswerGroupsConfig = z.infer<typeof DrawAnswerGroupsConfigSchema>;

export function createDrawAnswerGroupsConfig(
  data: Partial<DrawAnswerGroupsConfig> = {}
): DrawAnswerGroupsConfig {
  return DrawAnswerGroupsConfigSchema.parse({
    enabled: data.enabled ?? true,
    colorSequence: data.colorSequence ?? ['#8DFBC4', '#F7FB8D', '#8D9EFB', '#EA666F'],
  });
}

// Draw Question Verdicts Configuration
export const DrawQuestionVerdictsConfigSchema = z.object({
  enabled: z.boolean().default(true),
  verdictColors: z
    .record(z.string(), z.string().nullable())
    .default({
      correct: '#00FF00',
      neutral: null,
      incorrect: '#FF0000',
      bonus: '#00DDDD',
    }),
  verdictSymbolColors: z
    .record(z.string(), z.string())
    .default({
      positive: '#000000',
      neutral: '#000000',
      negative: '#000000',
      bonus: '#000000',
    }),
  drawAnswerGroups: DrawAnswerGroupsConfigSchema.default(createDrawAnswerGroupsConfig()),
});

export type DrawQuestionVerdictsConfig = z.infer<typeof DrawQuestionVerdictsConfigSchema>;

export function createDrawQuestionVerdictsConfig(
  data: Partial<DrawQuestionVerdictsConfig> = {}
): DrawQuestionVerdictsConfig {
  return DrawQuestionVerdictsConfigSchema.parse({
    enabled: data.enabled ?? true,
    verdictColors: data.verdictColors ?? {
      correct: '#00FF00',
      neutral: null,
      incorrect: '#FF0000',
      bonus: '#00DDDD',
    },
    verdictSymbolColors: data.verdictSymbolColors ?? {
      positive: '#000000',
      neutral: '#000000',
      negative: '#000000',
      bonus: '#000000',
    },
    drawAnswerGroups: data.drawAnswerGroups ?? createDrawAnswerGroupsConfig(),
  });
}

// Draw Detected Bubble Texts Configuration
export const DrawDetectedBubbleTextsConfigSchema = z.object({
  enabled: z.boolean().default(true),
});

export type DrawDetectedBubbleTextsConfig = z.infer<typeof DrawDetectedBubbleTextsConfigSchema>;

export function createDrawDetectedBubbleTextsConfig(
  data: Partial<DrawDetectedBubbleTextsConfig> = {}
): DrawDetectedBubbleTextsConfig {
  return DrawDetectedBubbleTextsConfigSchema.parse({
    enabled: data.enabled ?? true,
  });
}

// Outputs Configuration
export const OutputsConfigurationSchema = z.object({
  shouldExplainScoring: z.boolean().default(false),
  shouldExportExplanationCsv: z.boolean().default(false),
  drawScore: DrawScoreConfigSchema.default(createDrawScoreConfig()),
  drawAnswersSummary: DrawAnswersSummaryConfigSchema.default(createDrawAnswersSummaryConfig()),
  drawQuestionVerdicts: DrawQuestionVerdictsConfigSchema.default(
    createDrawQuestionVerdictsConfig()
  ),
  drawDetectedBubbleTexts: DrawDetectedBubbleTextsConfigSchema.default(
    createDrawDetectedBubbleTextsConfig()
  ),
});

export type OutputsConfiguration = z.infer<typeof OutputsConfigurationSchema>;

export function createOutputsConfiguration(
  data: Partial<OutputsConfiguration> = {}
): OutputsConfiguration {
  return OutputsConfigurationSchema.parse({
    shouldExplainScoring: data.shouldExplainScoring ?? false,
    shouldExportExplanationCsv: data.shouldExportExplanationCsv ?? false,
    drawScore: data.drawScore ?? createDrawScoreConfig(),
    drawAnswersSummary: data.drawAnswersSummary ?? createDrawAnswersSummaryConfig(),
    drawQuestionVerdicts: data.drawQuestionVerdicts ?? createDrawQuestionVerdictsConfig(),
    drawDetectedBubbleTexts: data.drawDetectedBubbleTexts ?? createDrawDetectedBubbleTextsConfig(),
  });
}

// Main Evaluation Configuration
export const EvaluationConfigSchema = z.object({
  sourceType: z.string().default('local'),
  options: z.record(z.string(), z.any()).default({}),
  markingSchemes: z.record(z.string(), z.any()).default({}),
  conditionalSets: z.array(z.any()).default([]),
  outputsConfiguration: OutputsConfigurationSchema.default(createOutputsConfiguration()),
});

export type EvaluationConfig = z.infer<typeof EvaluationConfigSchema>;

export function createEvaluationConfig(
  data: Partial<EvaluationConfig> = {}
): EvaluationConfig {
  return EvaluationConfigSchema.parse({
    sourceType: data.sourceType ?? 'local',
    options: data.options ?? {},
    markingSchemes: data.markingSchemes ?? {},
    conditionalSets: data.conditionalSets ?? [],
    outputsConfiguration: data.outputsConfiguration ?? createOutputsConfiguration(),
  });
}

// Parse from JSON (with camelCase keys)
export function parseEvaluationFromJSON(json: Record<string, any>): EvaluationConfig {
  return EvaluationConfigSchema.parse(json);
}

// Convert to JSON (camelCase preserved)
export function evaluationToJSON(config: EvaluationConfig): Record<string, any> {
  return config; // Already in camelCase format
}
