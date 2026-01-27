/**
 * Tests for config schema validation
 *
 * TypeScript port of relevant tests from src/tests/test_config_validations.py
 */

import { describe, it, expect } from 'vitest';
import { validateConfig } from '../configSchema';

describe('ConfigSchema', () => {
  it('should validate minimal valid config', () => {
    const config = {
      outputs: {
        save_image_level: 0,
      },
    };

    const result = validateConfig(config);
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('should reject invalid threshold values out of range', () => {
    const config = {
      thresholding: {
        min_gap_two_bubbles: 150, // Max is 100
      },
    };

    const result = validateConfig(config);
    expect(result.valid).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });

  it('should accept valid thresholding config', () => {
    const config = {
      thresholding: {
        min_gap_two_bubbles: 50,
        min_jump: 25,
        global_page_threshold: 200,
      },
    };

    const result = validateConfig(config);
    expect(result.valid).toBe(true);
  });

  it('should validate output mode enum', () => {
    const validConfig = {
      outputs: {
        output_mode: 'default',
      },
    };

    const invalidConfig = {
      outputs: {
        output_mode: 'invalid_mode',
      },
    };

    expect(validateConfig(validConfig).valid).toBe(true);
    expect(validateConfig(invalidConfig).valid).toBe(false);
  });

  it('should validate show_image_level range', () => {
    const validConfig = {
      outputs: {
        show_image_level: 3,
      },
    };

    const invalidConfig = {
      outputs: {
        show_image_level: 10, // Max is 6
      },
    };

    expect(validateConfig(validConfig).valid).toBe(true);
    expect(validateConfig(invalidConfig).valid).toBe(false);
  });

  it('should validate ML confidence thresholds', () => {
    const validConfig = {
      ml: {
        confidence_threshold: 0.7,
        min_training_confidence: 0.85,
      },
    };

    const invalidConfig = {
      ml: {
        confidence_threshold: 1.5, // Max is 1.0
      },
    };

    expect(validateConfig(validConfig).valid).toBe(true);
    expect(validateConfig(invalidConfig).valid).toBe(false);
  });

  it('should validate processing worker count range', () => {
    const validConfig = {
      processing: {
        max_parallel_workers: 4,
      },
      outputs: {}, // Need outputs section due to required structure
    };

    const invalidConfig = {
      processing: {
        max_parallel_workers: 20, // Max is 16
      },
      outputs: {},
    };

    const validResult = validateConfig(validConfig);
    if (!validResult.valid) {
      console.log('Valid config errors:', JSON.stringify(validResult.errors, null, 2));
    }
    expect(validResult.valid).toBe(true);
    expect(validateConfig(invalidConfig).valid).toBe(false);
  });

  it('should validate fusion strategy enum', () => {
    const validConfig = {
      ml: {
        fusion_strategy: 'confidence_weighted',
      },
    };

    const invalidConfig = {
      ml: {
        fusion_strategy: 'invalid_strategy',
      },
    };

    expect(validateConfig(validConfig).valid).toBe(true);
    expect(validateConfig(invalidConfig).valid).toBe(false);
  });

  it('should handle complex nested config', () => {
    const config = {
      thresholding: {
        min_gap_two_bubbles: 30,
        global_page_threshold: 180,
      },
      outputs: {
        output_mode: 'moderation',
        save_image_level: 2,
        colored_outputs_enabled: true,
        show_logs_by_type: {
          error: true,
          warning: true,
          info: false,
        },
      },
      processing: {
        max_parallel_workers: 4,
      },
      ml: {
        enabled: true,
        confidence_threshold: 0.8,
      },
    };

    const result = validateConfig(config);
    expect(result.valid).toBe(true);
  });
});

