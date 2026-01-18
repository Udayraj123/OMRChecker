/**
 * Configuration defaults.
 *
 * TypeScript port of src/schemas/defaults/config.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import { Config } from '../models/config';
import { FileGroupingConfig } from '../models/config';
import { SUPPORTED_PROCESSOR_NAMES } from '../../utils/constants';

/**
 * Create default config instance.
 *
 * Port of CONFIG_DEFAULTS from Python.
 */
export const CONFIG_DEFAULTS = new Config(
  'config.json',
  {
    GAMMA_LOW: 0.7,
    MIN_GAP_TWO_BUBBLES: 30,
    MIN_JUMP: 25,
    CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY: 25,
    MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK: 5,
    GLOBAL_THRESHOLD_MARGIN: 10,
    JUMP_DELTA: 30,
    // Note: tune this value to avoid empty bubble detections
    GLOBAL_PAGE_THRESHOLD: 200,
    GLOBAL_PAGE_THRESHOLD_STD: 10,
    MIN_JUMP_STD: 15,
    JUMP_DELTA_STD: 5,
  },
  {
    output_mode: 'default',
    display_image_dimensions: [720, 1080],
    show_image_level: 0,
    show_preprocessors_diff: Object.fromEntries(
      SUPPORTED_PROCESSOR_NAMES.map((name) => [name, false])
    ),
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
  },
  {
    max_parallel_workers: 1, // Number of worker threads for parallel processing (1 = sequential)
  },
  {
    enabled: false,
    model_path: null,
    confidence_threshold: 0.7,
    use_for_low_confidence_only: true,
    collect_training_data: false,
    min_training_confidence: 0.85,
    shift_detection: {
      enabled: false,
      global_max_shift_pixels: 50,
      per_block_max_shift_pixels: {},
      confidence_reduction_min: 0.1,
      confidence_reduction_max: 0.5,
      bubble_mismatch_threshold: 3,
      field_mismatch_threshold: 1,
    },
  }
);

