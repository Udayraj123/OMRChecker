import { z } from 'zod';

// Thresholding Configuration
export const ThresholdingConfigSchema = z.object({
  gammaLow: z.number().default(0.7),
  minGapTwoBubbles: z.number().int().default(30),
  minJump: z.number().int().default(25),
  confidentJumpSurplusForDisparity: z.number().int().default(25),
  minJumpSurplusForGlobalFallback: z.number().int().default(5),
  globalThresholdMargin: z.number().int().default(10),
  jumpDelta: z.number().int().default(30),
  globalPageThreshold: z.number().int().default(200),
  globalPageThresholdStd: z.number().int().default(10),
  minJumpStd: z.number().int().default(15),
  jumpDeltaStd: z.number().int().default(5),
});

export type ThresholdingConfig = z.infer<typeof ThresholdingConfigSchema>;

// Subset for interpretation threshold calculation
export interface InterpretationThresholdConfig {
  minJump: number;
  jumpDelta: number;
  minGapTwoBubbles: number;
  minJumpSurplusForGlobalFallback: number;
  confidentJumpSurplusForDisparity: number;
  globalThresholdMargin: number;
  globalPageThreshold: number;
}

// Subset for outlier deviation threshold calculation
export interface OutlierDeviationThresholdConfig {
  minJumpStd: number;
  globalPageThresholdStd: number;
}

// Subset for fallback threshold calculation
export interface FallbackThresholdConfig {
  globalPageThreshold: number;
  minJump: number;
}

export function createThresholdingConfig(
  data: Partial<ThresholdingConfig> = {}
): ThresholdingConfig {
  return ThresholdingConfigSchema.parse(data);
}

// Outputs Configuration (simplified for browser)
export const OutputsConfigSchema = z.object({
  outputMode: z.string().default('default'),
  displayImageDimensions: z.array(z.number().int()).length(2).default([720, 1080]),
  showImageLevel: z.number().int().default(0),
  showPreprocessorsDiff: z.record(z.string(), z.boolean()).default({}),
  saveImageLevel: z.number().int().default(1),
  showLogsByType: z
    .record(z.string(), z.boolean())
    .default({
      critical: true,
      error: true,
      warning: true,
      info: true,
      debug: false,
    }),
  saveDetections: z.boolean().default(true),
  coloredOutputsEnabled: z.boolean().default(false),
  saveImageMetrics: z.boolean().default(false),
  showConfidenceMetrics: z.boolean().default(false),
  filterOutMultimarkedFiles: z.boolean().default(false),
});

export type OutputsConfig = z.infer<typeof OutputsConfigSchema>;

export function createOutputsConfig(data: Partial<OutputsConfig> = {}): OutputsConfig {
  return OutputsConfigSchema.parse(data);
}

// Processing Configuration (simplified - browser runs single-threaded)
export const ProcessingConfigSchema = z.object({
  maxParallelWorkers: z.number().int().default(1),
});

export type ProcessingConfig = z.infer<typeof ProcessingConfigSchema>;

export function createProcessingConfig(
  data: Partial<ProcessingConfig> = {}
): ProcessingConfig {
  return ProcessingConfigSchema.parse(data);
}

// Alignment Configuration
export const AlignmentConfigSchema = z.object({
  enabled: z.boolean().default(true),
});

export type AlignmentConfig = z.infer<typeof AlignmentConfigSchema>;

export function createAlignmentConfig(data: Partial<AlignmentConfig> = {}): AlignmentConfig {
  return AlignmentConfigSchema.parse(data);
}

// Shift Detection Configuration
export const ShiftDetectionConfigSchema = z.object({
  enabled: z.boolean().default(false),
  globalMaxShiftPixels: z.number().int().default(50),
  perBlockMaxShiftPixels: z.record(z.string(), z.number().int()).default({}),
  confidenceReductionMin: z.number().default(0.1),
  confidenceReductionMax: z.number().default(0.5),
  bubbleMismatchThreshold: z.number().int().default(3),
  fieldMismatchThreshold: z.number().int().default(1),
});

export type ShiftDetectionConfig = z.infer<typeof ShiftDetectionConfigSchema>;

export function createShiftDetectionConfig(
  data: Partial<ShiftDetectionConfig> = {}
): ShiftDetectionConfig {
  return ShiftDetectionConfigSchema.parse(data);
}

// ML Configuration (browser-compatible - excludes training features)
export const MLConfigSchema = z.object({
  enabled: z.boolean().default(false),
  modelPath: z.string().nullable().optional(),
  confidenceThreshold: z.number().default(0.7),
  useForLowConfidenceOnly: z.boolean().default(true),
  fieldBlockDetectionEnabled: z.boolean().default(false),
  fieldBlockModelPath: z.string().nullable().optional(),
  fieldBlockConfidenceThreshold: z.number().default(0.75),
  fusionEnabled: z.boolean().default(true),
  fusionStrategy: z.string().default('confidence_weighted'),
  discrepancyThreshold: z.number().default(0.3),
  shiftDetection: ShiftDetectionConfigSchema.default(createShiftDetectionConfig()),
});

export type MLConfig = z.infer<typeof MLConfigSchema>;

export function createMLConfig(data: Partial<MLConfig> = {}): MLConfig {
  return MLConfigSchema.parse(data);
}

// Main Config (browser-compatible - excludes experimental/CLI features)
export const ConfigSchema = z.object({
  thresholding: ThresholdingConfigSchema.default(createThresholdingConfig()),
  outputs: OutputsConfigSchema.default(createOutputsConfig()),
  processing: ProcessingConfigSchema.default(createProcessingConfig()),
  alignment: AlignmentConfigSchema.default(createAlignmentConfig()),
  ml: MLConfigSchema.default(createMLConfig()),
});

export type Config = z.infer<typeof ConfigSchema>;

export function createConfig(data: Partial<Config> = {}): Config {
  return ConfigSchema.parse({
    thresholding: data.thresholding ?? createThresholdingConfig(),
    outputs: data.outputs ?? createOutputsConfig(),
    processing: data.processing ?? createProcessingConfig(),
    alignment: data.alignment ?? createAlignmentConfig(),
    ml: data.ml ?? createMLConfig(),
  });
}

// Parse from JSON (with camelCase keys)
export function parseConfigFromJSON(json: Record<string, any>): Config {
  return ConfigSchema.parse(json);
}

// Convert to JSON (camelCase preserved)
export function configToJSON(config: Config): Record<string, any> {
  return config; // Already in camelCase format
}
