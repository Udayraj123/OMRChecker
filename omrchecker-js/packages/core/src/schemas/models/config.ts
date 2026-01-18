/**
 * Typed models for configuration.
 *
 * TypeScript port of src/schemas/models/config.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import { Logger } from '../../utils/logger';

const logger = new Logger('ConfigModels');

/**
 * Configuration for bubble thresholding algorithm.
 */
export interface ThresholdingConfig {
  GAMMA_LOW: number;
  MIN_GAP_TWO_BUBBLES: number;
  MIN_JUMP: number;
  CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY: number;
  MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK: number;
  GLOBAL_THRESHOLD_MARGIN: number;
  JUMP_DELTA: number;
  GLOBAL_PAGE_THRESHOLD: number;
  GLOBAL_PAGE_THRESHOLD_STD: number;
  MIN_JUMP_STD: number;
  JUMP_DELTA_STD: number;
}

/**
 * A single file grouping rule with dynamic path pattern.
 */
export interface GroupingRule {
  name: string;
  priority: number;
  destination_pattern: string; // Full path pattern: "folder/{booklet}/roll_{roll}.jpg"
  matcher: {
    formatString: string;
    matchRegex: string;
  };
  action?: 'symlink' | 'copy';
  collision_strategy?: 'skip' | 'increment' | 'overwrite';
}

/**
 * Configuration for automatic file organization.
 */
export class FileGroupingConfig {
  enabled: boolean = false;
  rules: GroupingRule[] = [];
  default_pattern: string = 'ungrouped/{original_name}'; // Default for non-matching files

  // Always available fields in patterns (built-in)
  static readonly BUILTIN_FIELDS: Set<string> = new Set([
    'file_path',
    'file_name',
    'file_stem',
    'original_name',
    'is_multi_marked',
  ]);

  // Fields that require evaluation to be enabled
  static readonly EVALUATION_FIELDS: Set<string> = new Set(['score']);

  /**
   * Validate the file grouping configuration.
   *
   * @param template - Optional template object to check available OMR fields
   * @param hasEvaluation - Whether evaluation is enabled
   * @returns List of validation error messages (empty if valid)
   */
  validate(template?: any, hasEvaluation: boolean = false): string[] {
    const errors: string[] = [];

    if (!this.enabled) {
      return errors; // Skip validation if disabled
    }

    // Validate default pattern
    const patternErrors = this._validatePattern(
      this.default_pattern,
      'default_pattern',
      template,
      hasEvaluation
    );
    errors.push(...patternErrors);

    // Validate each rule
    for (let i = 0; i < this.rules.length; i++) {
      const rule = this.rules[i];
      const ruleErrors = this._validateRule(rule, i + 1, template, hasEvaluation);
      errors.push(...ruleErrors);
    }

    // Check for duplicate priorities
    const priorities = this.rules.map((r) => r.priority);
    const uniquePriorities = new Set(priorities);
    if (priorities.length !== uniquePriorities.size) {
      const duplicates = priorities.filter(
        (p, idx) => priorities.indexOf(p) !== idx
      );
      errors.push(
        `Duplicate rule priorities found: ${Array.from(new Set(duplicates)).join(', ')}. ` +
          'Each rule should have a unique priority.'
      );
    }

    return errors;
  }

  /**
   * Validate a single grouping rule.
   */
  private _validateRule(
    rule: GroupingRule,
    ruleNum: number,
    template: any,
    hasEvaluation: boolean
  ): string[] {
    const errors: string[] = [];
    const prefix = `Rule #${ruleNum} ('${rule.name}')`;

    // Validate destination pattern
    const patternErrors = this._validatePattern(
      rule.destination_pattern,
      `${prefix} destination_pattern`,
      template,
      hasEvaluation
    );
    errors.push(...patternErrors);

    // Validate matcher format string
    const matcherFormat = rule.matcher?.formatString || '';
    const matcherErrors = this._validatePattern(
      matcherFormat,
      `${prefix} matcher.formatString`,
      template,
      hasEvaluation,
      false
    );
    errors.push(...matcherErrors);

    // Validate regex pattern
    try {
      new RegExp(rule.matcher?.matchRegex || '');
    } catch (e) {
      errors.push(
        `${prefix}: Invalid regex pattern in matcher.matchRegex: ${e instanceof Error ? e.message : String(e)}`
      );
    }

    // Validate action
    if (rule.action && !['symlink', 'copy'].includes(rule.action)) {
      errors.push(
        `${prefix}: Invalid action '${rule.action}'. Must be 'symlink' or 'copy'.`
      );
    }

    // Validate collision strategy
    if (
      rule.collision_strategy &&
      !['skip', 'increment', 'overwrite'].includes(rule.collision_strategy)
    ) {
      errors.push(
        `${prefix}: Invalid collision_strategy '${rule.collision_strategy}'. ` +
          "Must be 'skip', 'increment', or 'overwrite'."
      );
    }

    return errors;
  }

  /**
   * Validate a pattern string for field availability.
   */
  private _validatePattern(
    pattern: string,
    patternName: string,
    template: any,
    hasEvaluation: boolean,
    allowEmpty: boolean = true
  ): string[] {
    const errors: string[] = [];

    if (!pattern && !allowEmpty) {
      errors.push(`${patternName}: Pattern cannot be empty`);
      return errors;
    }

    if (!pattern) {
      return errors;
    }

    // Extract field names from pattern using regex
    // Pattern: {field_name} or {field_name:format}
    const fieldNameRegex = /\{([^}:]+)(?::[^}]+)?\}/g;
    const fieldNames = new Set<string>();
    let match;
    while ((match = fieldNameRegex.exec(pattern)) !== null) {
      fieldNames.add(match[1]);
    }

    // Check each field
    for (const fieldName of fieldNames) {
      // Check if it's a built-in field
      if (FileGroupingConfig.BUILTIN_FIELDS.has(fieldName)) {
        continue;
      }

      // Check if it requires evaluation
      if (FileGroupingConfig.EVALUATION_FIELDS.has(fieldName)) {
        if (!hasEvaluation) {
          errors.push(
            `${patternName}: Field '{${fieldName}}' requires evaluation.json ` +
              'to be present. Either add evaluation.json or remove this field from the pattern.'
          );
        }
        continue;
      }

      // Check if it's an OMR field from template
      if (template) {
        // Get all fields from template
        const templateFields = new Set<string>();
        if (template.allFields) {
          for (const field of template.allFields) {
            templateFields.add(field.fieldLabel || field.id);
          }
        }

        if (!templateFields.has(fieldName)) {
          // Provide helpful error message
          const available = Array.from(
            new Set([
              ...Array.from(FileGroupingConfig.BUILTIN_FIELDS),
              ...Array.from(FileGroupingConfig.EVALUATION_FIELDS),
              ...Array.from(templateFields),
            ])
          ).sort();
          errors.push(
            `${patternName}: Field '{${fieldName}}' not found in template. ` +
              `Available fields: ${available.map((f) => `{${f}}`).join(', ')}`
          );
        }
      } else {
        // No template available for validation - just warn
        logger.warn(
          `${patternName}: Cannot validate field '{${fieldName}}' without template context`
        );
      }
    }

    return errors;
  }
}

/**
 * Configuration for output behavior and visualization.
 */
export interface OutputsConfig {
  output_mode?: string;
  display_image_dimensions?: [number, number];
  show_image_level?: number;
  show_preprocessors_diff?: Record<string, boolean>;
  save_image_level?: number;
  show_logs_by_type?: {
    critical?: boolean;
    error?: boolean;
    warning?: boolean;
    info?: boolean;
    debug?: boolean;
  };
  save_detections?: boolean;
  colored_outputs_enabled?: boolean;
  coloredOutputsEnabled?: boolean; // Alias for compatibility
  save_image_metrics?: boolean;
  show_confidence_metrics?: boolean;
  filter_out_multimarked_files?: boolean;
  file_grouping?: FileGroupingConfig;
}

/**
 * Configuration for parallel processing.
 */
export interface ProcessingConfig {
  max_parallel_workers?: number;
}

/**
 * Configuration for ML-based shift detection and application.
 */
export interface ShiftDetectionConfig {
  enabled?: boolean;
  global_max_shift_pixels?: number; // Global limit for all blocks
  per_block_max_shift_pixels?: Record<string, number>; // Per-block overrides

  // Confidence adjustment on mismatch
  confidence_reduction_min?: number; // Min reduction (1 bubble diff)
  confidence_reduction_max?: number; // Max reduction (many diffs)

  // Comparison thresholds
  bubble_mismatch_threshold?: number; // Flag if >3 bubbles differ
  field_mismatch_threshold?: number; // Flag if any field response differs
}

/**
 * Configuration for ML-based detection and training.
 */
export interface MLConfig {
  // General ML settings
  enabled?: boolean;

  // Bubble detection (Stage 2) settings
  model_path?: string | null;
  confidence_threshold?: number;
  use_for_low_confidence_only?: boolean;

  // Field block detection (Stage 1) settings
  field_block_detection_enabled?: boolean;
  field_block_model_path?: string | null;
  field_block_confidence_threshold?: number;

  // Detection fusion settings
  fusion_enabled?: boolean;
  fusion_strategy?: string; // Options: confidence_weighted, ml_fallback, traditional_primary
  discrepancy_threshold?: number; // Flag if responses differ with high confidence

  // Shift detection settings
  shift_detection?: ShiftDetectionConfig;

  // Training data collection settings
  collect_training_data?: boolean;
  min_training_confidence?: number;
  training_data_dir?: string;
  model_output_dir?: string;

  // Hierarchical training settings
  collect_field_block_data?: boolean; // Collect both stages
  field_block_dataset_dir?: string;
  bubble_dataset_dir?: string;
}

/**
 * Configuration for workflow visualization and debugging.
 */
export interface VisualizationConfig {
  enabled?: boolean;
  capture_processors?: string[];
  capture_frequency?: string; // Options: "always", "on_change"
  include_colored?: boolean;
  max_image_width?: number;
  embed_images?: boolean;
  export_format?: string; // Options: "html", "json"
  output_dir?: string;
  auto_open_browser?: boolean;
}

/**
 * Main configuration object for OMRChecker.
 *
 * This replaces the DotMap-based tuning_config throughout the codebase.
 */
export class Config {
  path: string;
  thresholding: ThresholdingConfig;
  outputs: OutputsConfig;
  processing: ProcessingConfig;
  ml: MLConfig;
  visualization?: VisualizationConfig;

  constructor(
    path: string,
    thresholding?: Partial<ThresholdingConfig>,
    outputs?: Partial<OutputsConfig>,
    processing?: Partial<ProcessingConfig>,
    ml?: Partial<MLConfig>,
    visualization?: Partial<VisualizationConfig>
  ) {
    this.path = path;
    this.thresholding = {
      GAMMA_LOW: 0.7,
      MIN_GAP_TWO_BUBBLES: 30,
      MIN_JUMP: 25,
      CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY: 25,
      MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK: 5,
      GLOBAL_THRESHOLD_MARGIN: 10,
      JUMP_DELTA: 30,
      GLOBAL_PAGE_THRESHOLD: 200,
      GLOBAL_PAGE_THRESHOLD_STD: 10,
      MIN_JUMP_STD: 15,
      JUMP_DELTA_STD: 5,
      ...thresholding,
    };
    this.outputs = {
      output_mode: 'default',
      display_image_dimensions: [720, 1080],
      show_image_level: 0,
      show_preprocessors_diff: {},
      save_image_level: 1,
      show_logs_by_type: {
        critical: true,
        error: true,
        warning: true,
        info: true,
        debug: false,
      },
      save_detections: true,
      colored_outputs_enabled: false,
      save_image_metrics: false,
      show_confidence_metrics: false,
      filter_out_multimarked_files: false,
      file_grouping: new FileGroupingConfig(),
      ...outputs,
    };
    this.processing = {
      max_parallel_workers: 1,
      ...processing,
    };
    this.ml = {
      enabled: false,
      model_path: null,
      confidence_threshold: 0.7,
      use_for_low_confidence_only: true,
      collect_training_data: false,
      min_training_confidence: 0.85,
      shift_detection: {},
      ...ml,
    };
    this.visualization = visualization;
  }

  /**
   * Create Config from dictionary (typically from JSON).
   *
   * @param data - Dictionary containing configuration data
   * @returns Config instance with nested configs
   */
  static fromDict(data: Record<string, any>): Config {
    const outputsData = data.outputs || {};

    // Parse file_grouping nested structure
    let fileGrouping: FileGroupingConfig | undefined;
    if (outputsData.file_grouping) {
      const groupingData = outputsData.file_grouping;
      const rulesData = groupingData.rules || [];
      const rules: GroupingRule[] = rulesData.map((rule: any) => ({
        name: rule.name,
        priority: rule.priority,
        destination_pattern: rule.destination_pattern,
        matcher: rule.matcher,
        action: rule.action || 'symlink',
        collision_strategy: rule.collision_strategy || 'skip',
      }));

      fileGrouping = new FileGroupingConfig();
      fileGrouping.enabled = groupingData.enabled || false;
      fileGrouping.default_pattern =
        groupingData.default_pattern || 'ungrouped/{original_name}';
      fileGrouping.rules = rules;
    }

    return new Config(
      data.path || 'config.json',
      data.thresholding,
      { ...outputsData, file_grouping: fileGrouping },
      data.processing,
      data.ml,
      data.visualization
    );
  }

  /**
   * Convert Config to dictionary for JSON serialization.
   *
   * @returns Dictionary representation of the config
   */
  toDict(): Record<string, any> {
    return {
      path: this.path,
      thresholding: this.thresholding,
      outputs: {
        ...this.outputs,
        file_grouping: this.outputs.file_grouping
          ? {
              enabled: this.outputs.file_grouping.enabled,
              default_pattern: this.outputs.file_grouping.default_pattern,
              rules: this.outputs.file_grouping.rules,
            }
          : undefined,
      },
      processing: this.processing,
      ml: this.ml,
      visualization: this.visualization,
    };
  }

  /**
   * Get value by key (backwards compatibility).
   *
   * @param key - The key to retrieve
   * @param defaultValue - Default value if key not found
   * @returns The value if found, otherwise default
   */
  get(key: string, defaultValue?: any): any {
    return (this as any)[key] ?? defaultValue;
  }
}

