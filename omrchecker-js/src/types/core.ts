// Auto-generated from Python dataclasses
// Do not edit manually - regenerate with 2-generate-interfaces.js

/**
 * ThresholdingConfig
 * Source: schemas/models/config.py
 */
export interface ThresholdingConfig {
  gammaLow?: number;  // Python: gamma_low
  minGapTwoBubbles?: number;  // Python: min_gap_two_bubbles
  minJump?: number;  // Python: min_jump
  confidentJumpSurplusForDisparity?: number;  // Python: confident_jump_surplus_for_disparity
  minJumpSurplusForGlobalFallback?: number;  // Python: min_jump_surplus_for_global_fallback
  globalThresholdMargin?: number;  // Python: global_threshold_margin
  jumpDelta?: number;  // Python: jump_delta
  globalPageThreshold?: number;  // Python: global_page_threshold
  globalPageThresholdStd?: number;  // Python: global_page_threshold_std
  minJumpStd?: number;  // Python: min_jump_std
  jumpDeltaStd?: number;  // Python: jump_delta_std
}

/**
 * GroupingRule
 * Source: schemas/models/config.py
 */
export interface GroupingRule {
  name: string;
  priority: number;
  destinationPattern: str  # Full path pattern: "folder/{booklet}/roll_{roll}.jpg";  // Python: destination_pattern
  matcher: dict  # { "formatString": "...", "matchRegex": "..." };
  action?: string;
  collisionStrategy?: string;  // Python: collision_strategy
}

/**
 * FileGroupingConfig
 * Source: schemas/models/config.py
 */
export interface FileGroupingConfig {
  enabled?: boolean;
  rules?: GroupingRule[];
  defaultPattern?: string;  // Python: default_pattern
  BUILTIN_FIELDS?: ClassVar[set[str]];
  EVALUATION_FIELDS?: ClassVar[set[str]];
  Args?: template: Optional template object to check available OMR fields;
  hasEvaluation: Whether evaluation is enabled;  // Python: has_evaluation
  rule: GroupingRule,;
  ruleNum: int,;  // Python: rule_num
  hasEvaluation: bool,;  // Python: has_evaluation
  pattern: str,;
  patternName: str,;  // Python: pattern_name
  hasEvaluation: bool,;  // Python: has_evaluation
  allowEmpty?: boolean;  // Python: allow_empty
  try?: formatter;
  else: # No template available for validation - just warn;
}

/**
 * OutputsConfig
 * Source: schemas/models/config.py
 */
export interface OutputsConfig {
  outputMode?: string;  // Python: output_mode
  displayImageDimensions?: number[];  // Python: display_image_dimensions
  showImageLevel?: number;  // Python: show_image_level
  showPreprocessorsDiff?: Record<[str, bool], string>;  // Python: show_preprocessors_diff
  saveImageLevel?: number;  // Python: save_image_level
  showLogsByType?: Record<[str, bool], string>;  // Python: show_logs_by_type
  saveDetections?: boolean;  // Python: save_detections
  coloredOutputsEnabled?: boolean;  // Python: colored_outputs_enabled
  saveImageMetrics?: boolean;  // Python: save_image_metrics
  showConfidenceMetrics?: boolean;  // Python: show_confidence_metrics
  filterOutMultimarkedFiles?: boolean;  // Python: filter_out_multimarked_files
  fileGrouping?: FileGroupingConfig;  // Python: file_grouping
}

/**
 * ProcessingConfig
 * Source: schemas/models/config.py
 */
export interface ProcessingConfig {
  maxParallelWorkers?: number;  // Python: max_parallel_workers
}

/**
 * ShiftDetectionConfig
 * Source: schemas/models/config.py
 */
export interface ShiftDetectionConfig {
  enabled?: boolean;
  globalMaxShiftPixels?: number;  // Python: global_max_shift_pixels
  perBlockMaxShiftPixels?: Record<[str, int], string>;  // Python: per_block_max_shift_pixels
  confidenceReductionMin?: number;  // Python: confidence_reduction_min
  confidenceReductionMax?: number;  // Python: confidence_reduction_max
  bubbleMismatchThreshold?: number;  // Python: bubble_mismatch_threshold
  fieldMismatchThreshold?: number;  // Python: field_mismatch_threshold
}

/**
 * MLConfig
 * Source: schemas/models/config.py
 */
export interface MLConfig {
  enabled?: boolean;
  modelPath?: string | null;  // Python: model_path
  confidenceThreshold?: number;  // Python: confidence_threshold
  useForLowConfidenceOnly?: boolean;  // Python: use_for_low_confidence_only
  fieldBlockDetectionEnabled?: boolean;  // Python: field_block_detection_enabled
  fieldBlockModelPath?: string | null;  // Python: field_block_model_path
  fieldBlockConfidenceThreshold?: number;  // Python: field_block_confidence_threshold
  fusionEnabled?: boolean;  // Python: fusion_enabled
  fusionStrategy?: string;  // Python: fusion_strategy
  discrepancyThreshold?: number;  // Python: discrepancy_threshold
  shiftDetection?: ShiftDetectionConfig;  // Python: shift_detection
  collectTrainingData?: boolean;  // Python: collect_training_data
  minTrainingConfidence?: number;  // Python: min_training_confidence
  trainingDataDir?: string;  // Python: training_data_dir
  modelOutputDir?: string;  // Python: model_output_dir
  collectFieldBlockData?: boolean;  // Python: collect_field_block_data
  fieldBlockDatasetDir?: string;  // Python: field_block_dataset_dir
  bubbleDatasetDir?: string;  // Python: bubble_dataset_dir
}

/**
 * VisualizationConfig
 * Source: schemas/models/config.py
 */
export interface VisualizationConfig {
  enabled?: boolean;
  captureProcessors?: string[];  // Python: capture_processors
  captureFrequency?: string;  // Python: capture_frequency
  includeColored?: boolean;  // Python: include_colored
  maxImageWidth?: number;  // Python: max_image_width
  embedImages?: boolean;  // Python: embed_images
  exportFormat?: string;  // Python: export_format
  outputDir?: string;  // Python: output_dir
  autoOpenBrowser?: boolean;  // Python: auto_open_browser
}

/**
 * DrawScoreConfig
 * Source: schemas/models/evaluation.py
 */
export interface DrawScoreConfig {
  enabled?: boolean;
  position?: number[];
  scoreFormatString?: string;  // Python: score_format_string
  size?: number;
}

/**
 * DrawAnswersSummaryConfig
 * Source: schemas/models/evaluation.py
 */
export interface DrawAnswersSummaryConfig {
  enabled?: boolean;
  position?: number[];
  answersSummaryFormatString?: string;  // Python: answers_summary_format_string
  size?: number;
}

/**
 * DrawAnswerGroupsConfig
 * Source: schemas/models/evaluation.py
 */
export interface DrawAnswerGroupsConfig {
  enabled?: boolean;
  colorSequence?: string[];  // Python: color_sequence
}

/**
 * DrawQuestionVerdictsConfig
 * Source: schemas/models/evaluation.py
 */
export interface DrawQuestionVerdictsConfig {
  enabled?: boolean;
  verdictColors?: Record<[str, str | None], string>;  // Python: verdict_colors
  verdictSymbolColors?: Record<[str, str], string>;  // Python: verdict_symbol_colors
  drawAnswerGroups?: DrawAnswerGroupsConfig;  // Python: draw_answer_groups
}

/**
 * DrawDetectedBubbleTextsConfig
 * Source: schemas/models/evaluation.py
 */
export interface DrawDetectedBubbleTextsConfig {
  enabled?: boolean;
}

/**
 * OutputsConfiguration
 * Source: schemas/models/evaluation.py
 */
export interface OutputsConfiguration {
  shouldExplainScoring?: boolean;  // Python: should_explain_scoring
  shouldExportExplanationCsv?: boolean;  // Python: should_export_explanation_csv
  drawScore?: DrawScoreConfig;  // Python: draw_score
  drawAnswersSummary?: DrawAnswersSummaryConfig;  // Python: draw_answers_summary
  drawQuestionVerdicts?: DrawQuestionVerdictsConfig;  // Python: draw_question_verdicts
  drawDetectedBubbleTexts?: DrawDetectedBubbleTextsConfig;  // Python: draw_detected_bubble_texts
}

/**
 * AlignmentMarginsConfig
 * Source: schemas/models/template.py
 */
export interface AlignmentMarginsConfig {
  top?: number;
  bottom?: number;
  left?: number;
  right?: number;
}

/**
 * AlignmentConfig
 * Source: schemas/models/template.py
 */
export interface AlignmentConfig {
  margins?: AlignmentMarginsConfig;
  maxDisplacement?: number;  // Python: max_displacement
}

/**
 * OutputColumnsConfig
 * Source: schemas/models/template.py
 */
export interface OutputColumnsConfig {
  customOrder?: string[];  // Python: custom_order
  sortType?: string;  // Python: sort_type
  sortOrder?: string;  // Python: sort_order
}

/**
 * SortFilesConfig
 * Source: schemas/models/template.py
 */
export interface SortFilesConfig {
  enabled?: boolean;
}
