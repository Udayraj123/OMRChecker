import { z } from 'zod';

// Alignment Margins Configuration
export const AlignmentMarginsConfigSchema = z.object({
  top: z.number().int().default(0),
  bottom: z.number().int().default(0),
  left: z.number().int().default(0),
  right: z.number().int().default(0),
});

export type AlignmentMarginsConfig = z.infer<typeof AlignmentMarginsConfigSchema>;

export function createAlignmentMarginsConfig(
  data: Partial<AlignmentMarginsConfig> = {}
): AlignmentMarginsConfig {
  return AlignmentMarginsConfigSchema.parse({
    top: data.top ?? 0,
    bottom: data.bottom ?? 0,
    left: data.left ?? 0,
    right: data.right ?? 0,
  });
}

// Alignment Configuration
export const AlignmentConfigSchema = z.object({
  margins: AlignmentMarginsConfigSchema.default(createAlignmentMarginsConfig()),
  maxDisplacement: z.number().int().default(10),
  referenceImage: z.string().nullable().optional(),
  maxMatchCount: z.number().int().nullable().optional(),
  anchorWindowSize: z.array(z.number().int()).nullable().optional(),
});

export type AlignmentConfig = z.infer<typeof AlignmentConfigSchema>;

export function createAlignmentConfig(data: Partial<AlignmentConfig> = {}): AlignmentConfig {
  return AlignmentConfigSchema.parse({
    margins: data.margins ?? createAlignmentMarginsConfig(),
    maxDisplacement: data.maxDisplacement ?? 10,
    referenceImage: data.referenceImage ?? null,
    maxMatchCount: data.maxMatchCount ?? null,
    anchorWindowSize: data.anchorWindowSize ?? null,
  });
}

// Output Columns Configuration
export const OutputColumnsConfigSchema = z.object({
  customOrder: z.array(z.string()).default([]),
  sortType: z.string().default('ALPHANUMERIC'),
  sortOrder: z.string().default('ASC'),
});

export type OutputColumnsConfig = z.infer<typeof OutputColumnsConfigSchema>;

export function createOutputColumnsConfig(
  data: Partial<OutputColumnsConfig> = {}
): OutputColumnsConfig {
  return OutputColumnsConfigSchema.parse({
    customOrder: data.customOrder ?? [],
    sortType: data.sortType ?? 'ALPHANUMERIC',
    sortOrder: data.sortOrder ?? 'ASC',
  });
}

// Sort Files Configuration
export const SortFilesConfigSchema = z.object({
  enabled: z.boolean().default(false),
});

export type SortFilesConfig = z.infer<typeof SortFilesConfigSchema>;

export function createSortFilesConfig(data: Partial<SortFilesConfig> = {}): SortFilesConfig {
  return SortFilesConfigSchema.parse({
    enabled: data.enabled ?? false,
  });
}

// Main Template Configuration
export const TemplateConfigSchema = z.object({
  // Required template properties
  bubbleDimensions: z.array(z.number().int()).length(2).default([10, 10]),
  templateDimensions: z.array(z.number().int()).length(2).default([1200, 1600]),

  // Configuration properties
  alignment: AlignmentConfigSchema.default(createAlignmentConfig()),
  conditionalSets: z.array(z.any()).default([]),
  customLabels: z.record(z.string(), z.any()).default({}),
  customBubbleFieldTypes: z.record(z.string(), z.any()).default({}),
  emptyValue: z.string().default(''),
  fieldBlocks: z.record(z.string(), z.any()).default({}),
  fieldBlocksOffset: z.array(z.number().int()).length(2).default([0, 0]),
  output: z.boolean().default(false),
  outputColumns: OutputColumnsConfigSchema.default(createOutputColumnsConfig()),
  outputImageShape: z.array(z.number().int()).default([]),
  preProcessors: z.array(z.any()).default([]),
  processingImageShape: z.array(z.number().int()).length(2).default([900, 650]),
  sortFiles: SortFilesConfigSchema.default(createSortFilesConfig()),
});

export type TemplateConfig = z.infer<typeof TemplateConfigSchema>;

export function createTemplateConfig(data: Partial<TemplateConfig> = {}): TemplateConfig {
  return TemplateConfigSchema.parse({
    bubbleDimensions: data.bubbleDimensions ?? [10, 10],
    templateDimensions: data.templateDimensions ?? [1200, 1600],
    alignment: data.alignment ?? createAlignmentConfig(),
    conditionalSets: data.conditionalSets ?? [],
    customLabels: data.customLabels ?? {},
    customBubbleFieldTypes: data.customBubbleFieldTypes ?? {},
    emptyValue: data.emptyValue ?? '',
    fieldBlocks: data.fieldBlocks ?? {},
    fieldBlocksOffset: data.fieldBlocksOffset ?? [0, 0],
    output: data.output ?? false,
    outputColumns: data.outputColumns ?? createOutputColumnsConfig(),
    outputImageShape: data.outputImageShape ?? [],
    preProcessors: data.preProcessors ?? [],
    processingImageShape: data.processingImageShape ?? [900, 650],
    sortFiles: data.sortFiles ?? createSortFilesConfig(),
  });
}

// Parse from JSON (with camelCase keys)
export function parseTemplateFromJSON(json: Record<string, any>): TemplateConfig {
  return TemplateConfigSchema.parse(json);
}

// Convert to JSON (camelCase preserved)
export function templateToJSON(config: TemplateConfig): Record<string, any> {
  return config; // Already in camelCase format
}
